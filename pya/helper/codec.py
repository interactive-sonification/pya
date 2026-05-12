"""
Audio codec module supporting high-quality audio file reading
with consistent float32 output.
Supports WAV, AIFF, FLAC (via SoundFile) and MP3 (via FFmpeg)
while maintaining maximum precision.
"""

import os
import queue
import re
import subprocess
import sys
import threading
import time

import numpy as np
import soundfile as sf

__all__ = [
    "DecodeError",
    "NoFileError",
    "NoBackendError",
    "UnsupportedError",
    "BitWidthError",
    "FFmpegError",
    "FFmpegNotInstalledError",
    "FFmpegReadTimeoutError",
    "CommunicationError",
    "BaseAudioFile",
    "SoundFileAudioFile",
    "QueueReaderThread",
    "popen_multiple",
    "ffmpeg_available",
    "FFmpegAudioFile",
    "audio_read",
]

COMMANDS = ("ffmpeg", "avconv")

if sys.platform == "win32":
    PROC_FLAGS = 0x08000000
else:
    PROC_FLAGS = 0


class DecodeError(Exception):
    """Base excoeption class for all decoding errors."""


class NoFileError(DecodeError):
    """File not found."""


class NoBackendError(DecodeError):
    """The file could not be decoded by any backend. Either no backends
    are available or each available backend failed to decode the file.
    """


class UnsupportedError(DecodeError):
    """File is not supported, support WAV, AIFF, FLAC and MP3."""


class BitWidthError(DecodeError):
    """Unsupported bit width."""


class FFmpegError(DecodeError):
    """Base class for FFmpeg errors."""


class FFmpegNotInstalledError(FFmpegError):
    """FFmpeg is not installed."""


class FFmpegReadTimeoutError(FFmpegError):
    """Reading from the ffmpeg command-line tool timed out."""


class CommunicationError(FFmpegError):
    """Raised when the output of FFmpeg is not parseable."""


class BaseAudioFile:
    """Base class defining the interface for audio file objects."""

    def __init__(self):
        self._channels = 0
        self._samplerate = 0
        self._duration = 0
        self._bytes_per_sample = 2
        self._subtype = "PCM_16"

    @property
    def channels(self) -> int:
        """Number of audio channels."""
        return self._channels

    @property
    def samplerate(self) -> int:
        """Sample rate in Hz."""
        return self._samplerate

    @property
    def duration(self) -> float:
        """Length of the audio in seconds (a float)."""
        return self._duration

    def read_data(self, block_samples=1024):
        """Read audio data as float32 numpy arrays.

        Parameters:
        ----------
        block_samples : int
            Number of samples to read per block.

        Returns:
        -------
        np.ndarray
            Generator yielding numpy f32 arrays of shape (samples, channels)
        """
        raise NotImplementedError

    def close(self):
        """Close the audio file and release resources."""
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()
        return False

    def __iter__(self):
        return self.read_data()


class SoundFileAudioFile(BaseAudioFile):
    """Using SoundFile package to read WAV, AIFF, AIF, FLAC"""

    def __init__(self, filename):
        super().__init__()
        try:
            self._file = sf.SoundFile(filename)
            self._check()
            self._channels = self._file.channels
            self._samplerate = self._file.samplerate
            self._duration = float(len(self._file)) / self._samplerate
            self._subtype = self._file.subtype
            self._bytes_per_sample = self._get_bytes_per_sample()

        except Exception as e:
            raise UnsupportedError(f"Failed to open {filename}: {e}")

    def _get_bytes_per_sample(self) -> int:
        """Determine bytes per sample based on subtype."""
        subtype_to_bytes = {
            "PCM_16": 2,  # 16-bit
            "PCM_24": 3,  # 24-bit
            "PCM_32": 4,  # 32-bit
            "FLOAT": 4,  # 32-bit float
            "DOUBLE": 8,  # 64-bit float
        }
        return subtype_to_bytes.get(self._subtype, 2)

    def _check(self):
        """Verify file format is supported."""
        if self._file.format not in ["WAV", "AIFF", "FLAC", "AIF"]:
            self.close()
            raise UnsupportedError()

    def read_data(self, block_samples=1024):
        """Read audio data as f32 arrays"""
        while True:
            data = self._file.read(block_samples)
            if len(data) == 0:
                break

            if self._channels > 1:
                data = data.reshape(-1, self._channels)
            yield data.astype(np.float32)

    def close(self):
        """Close the audio file."""
        if hasattr(self, "_file"):
            self._file.close()


class QueueReaderThread(threading.Thread):
    """A thread that consumes data from a filehandle and sends the data
    over a Queue."""

    def __init__(self, fh, blocksize=1024, discard=False):
        super(QueueReaderThread, self).__init__()
        self.fh = fh
        self.blocksize = blocksize
        self.daemon = True
        self.discard = discard
        self.queue = None if discard else queue.Queue()

    def run(self):
        while True:
            data = self.fh.read(self.blocksize)
            if not self.discard:
                self.queue.put(data)
            if not data:
                break  # Stream closed (EOF).


def popen_multiple(commands, command_args, *args, **kwargs):
    """Like `subprocess.Popen`, but can try multiple commands in case
    some are not available.
    `commands` is an iterable of command names and `command_args` are
    the rest of the arguments that, when appended to the command name,
    make up the full first argument to `subprocess.Popen`. The
    other positional and keyword arguments are passed through.
    """
    for i, command in enumerate(commands):
        cmd = [command] + command_args
        try:
            return subprocess.Popen(cmd, *args, **kwargs)
        except OSError:
            if i == len(commands) - 1:
                # No more commands to try.
                raise


def ffmpeg_available():
    """Detect whether the FFmpeg backend can be used on this system."""
    try:
        proc = popen_multiple(
            COMMANDS,
            ["-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=PROC_FLAGS,
        )
    except OSError:
        return False
    else:
        proc.wait()
        return proc.returncode == 0


# For Windows error switch management, we need a lock to keep the mode
# adjustment atomic.
windows_error_mode_lock = threading.Lock()


class FFmpegAudioFile(BaseAudioFile):
    """An audio file decoded by the ffmpeg command-line utility."""

    def __init__(self, filename, block_size=4096):
        super().__init__()

        # On Windows, we need to disable the subprocess's crash dialog
        # in case it dies. Passing SEM_NOGPFAULTERRORBOX to SetErrorMode
        # disables this behavior.
        windows = sys.platform.startswith("win")
        # This is only for windows.
        if windows:
            windows_error_mode_lock.acquire()
            SEM_NOGPFAULTERRORBOX = 0x0002
            import ctypes

            # We call SetErrorMode in two steps to avoid overriding
            # existing error mode.
            previous_error_mode = ctypes.windll.kernel32.SetErrorMode(
                SEM_NOGPFAULTERRORBOX
            )
            ctypes.windll.kernel32.SetErrorMode(
                previous_error_mode | SEM_NOGPFAULTERRORBOX
            )
        try:
            self.devnull = open(os.devnull)

            self.proc = popen_multiple(
                COMMANDS,
                ["-i", filename, "-f", "f32le", "-acodec", "pcm_f32le", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=self.devnull,
                creationflags=PROC_FLAGS if windows else 0,
            )

        except OSError:
            raise FFmpegNotInstalledError()

        finally:
            # Reset previous error mode on Windows. (We can change this
            # back now because the flag was inherited by the subprocess;
            # we don't need to keep it set in the parent process.)
            if windows:
                try:
                    import ctypes

                    ctypes.windll.kernel32.SetErrorMode(previous_error_mode)
                finally:
                    windows_error_mode_lock.release()

        # Start another thread to consume the standard output of the
        # process, which contains raw audio data.
        self.stdout_reader = QueueReaderThread(self.proc.stdout, block_size)
        self.stdout_reader.start()

        # Read relevant information from stderr.
        self._raw_str_info = ""
        self._get_info()

        # Start a separate thread to read the rest of the data from
        # stderr. This (a) avoids filling up the OS buffer and (b)
        # collects the error output for diagnosis.
        self.stderr_reader = QueueReaderThread(self.proc.stderr)
        self.stderr_reader.start()

    @property
    def raw_str_info(self) -> str:
        """Example info: 'duration: 00:00:00.81, start: 0.025057, bitrate: 84 kb/sstream #0:0: audio: mp3 (mp3float), 44100 hz, mono, fltp, 82 kb/s'"""  # noqa: E501
        return self._raw_str_info

    def _get_info(self):
        """Parsee FFmpeg output for relevant information."""
        out_parts = []
        while True:
            line = self.proc.stderr.readline()
            if not line:
                # EOF and data not found.
                raise CommunicationError("stream info not found")

            if isinstance(line, bytes):
                line = line.decode("utf8", "ignore")

            line = line.strip().lower()

            if "no such file" in line:
                raise IOError("file not found")
            elif "invalid data found" in line:
                raise UnsupportedError()
            elif "duration:" in line:
                out_parts.append(line)
            elif "audio:" in line:
                out_parts.append(line)
                self._raw_str_info = "".join(out_parts)
                self._parse_info(self._raw_str_info)
                break

    def _parse_info(self, str_info):
        """Given relevant data from the ffmpeg output, set audio
        parameter fields on this object.
        Example: 'duration: 00:00:00.81, start: 0.025057, bitrate: 84 kb/sstream #0:0: audio: mp3 (mp3float), 44100 hz, mono, fltp, 82 kb/s'
        """  # noqa: E501
        # Sample rate.
        match = re.search(r"(\d+) hz", str_info)
        if match:
            self._samplerate = int(match.group(1))
        else:
            self._samplerate = 0

        # Channel count.
        match = re.search(r"hz, ([^,]+),", str_info)
        if match:
            mode = match.group(1)
            if mode == "stereo":
                self._channels = 2
            else:
                cmatch = re.match(r"(\d+)\.?(\d)?", mode)
                if cmatch:
                    self._channels = sum(map(int, cmatch.group().split(".")))
                else:
                    self._channels = 1
        else:
            self._channels = 0

        # Duration.
        match = re.search(r"duration: (\d+):(\d+):(\d+).(\d)", str_info)
        if match:
            durparts = list(map(int, match.groups()))
            self._duration = (
                durparts[0] * 60 * 60
                + durparts[1] * 60
                + durparts[2]
                + float(durparts[3]) / 10
            )
        else:
            self._duration = 0

    def read_data(self, timeout=300.0):
        """Read blocks of 32-bit float data as numpy arrays."""
        start_time = time.time()

        while True:
            try:
                data = self.stdout_reader.queue.get(timeout=timeout)
                if not data:
                    break

                # Convert bytes to numpy float32 array
                numpy_data = np.frombuffer(data, dtype=np.float32)

                if self._channels > 1:
                    numpy_data = numpy_data.reshape(-1, self._channels)

                yield numpy_data

            except queue.Empty:
                end_time = time.time()
                if end_time - start_time >= timeout:
                    raise FFmpegReadTimeoutError(
                        "ffmpeg output: {}".format(
                            "".join(self.stderr_reader.queue.queue)
                        )
                    )
                start_time = end_time

    def close(self):
        """Close the ffmpeg process used to perform the decoding."""
        if hasattr(self, "proc"):
            # First check the process's execution status before attempting to
            # kill it. This fixes an issue on Windows Subsystem for Linux where
            # ffmpeg closes normally on its own, but never updates
            # `returncode`.
            self.proc.poll()

            # Kill the process if it is still running.
            if self.proc.returncode is None:
                self.proc.kill()
                self.proc.wait()

            # Wait for the stream-reading threads to exit. (They need to
            # stop reading before we can close the streams.)
            if hasattr(self, "stderr_reader"):
                self.stderr_reader.join()
            if hasattr(self, "stdout_reader"):
                self.stdout_reader.join()

            # Close the stdout and stderr streams that were opened by Popen,
            # which should occur regardless of if the process terminated
            # cleanly.
            self.proc.stdout.close()
            self.proc.stderr.close()
        # Close the handle to os.devnull, which is opened regardless of if
        # a subprocess is successfully created.
        if hasattr(self, "devnull"):
            self.devnull.close()


def audio_read(filepath):
    """Main entry point for audio file decoding.

    Supports:
    - WAV, AIFF, FLAC (via SoundFile):
        - Maintains original bit depth
        - Converts to float32 maintaining full precision
    - MP3 (via FFmpeg):
        - Decodes to 32-bit float

    Returns:
        An audio file object that yields numpy float32 arrays via read_data().
        Multi-channel audio is returned as (samples, channels) arrays.

    Properties:
        channels: number of audio channels
        samplerate: sample rate in Hz
        duration: length in seconds
    """
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()

    if ext in [".wav", ".aiff", ".flac", ".aif"]:
        return SoundFileAudioFile(filepath)

    # Fall back to FFmpeg
    if ffmpeg_available() and ext == ".mp3":
        return FFmpegAudioFile(filepath)

    raise UnsupportedError(f"Unsupported file type: {ext}")
