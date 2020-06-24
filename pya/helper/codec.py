#  This file handles opening audio files. 
import wave
import aifc
import sunau
import audioop
import struct
import sys
import subprocess 
import re
import time
import os
import threading
from warnings import warn
try:
    import queue
except ImportError:
    import Queue as queue

COMMANDS = ('ffmpeg', 'avconv')

if sys.platform == "win32":
    PROC_FLAGS = 0x08000000
else:
    PROC_FLAGS = 0

# Produce two-byte (16-bit) output samples.
TARGET_WIDTH = 2
# Python 3.4 added support for 24-bit (3-byte) samples.
if sys.version_info > (3, 4, 0):
    SUPPORTED_WIDTHS = (1, 2, 3, 4)
else:
    SUPPORTED_WIDTHS = (1, 2, 4)


class DecodeError(Exception):
    """The base exception class for all decoding errors raised by this
    package."""


class NoBackendError(DecodeError):
    """The file could not be decoded by any backend. Either no backends
    are available or each available backend failed to decode the file.
    """


class UnsupportedError(DecodeError):
    """File is not an AIFF, WAV, or Au file."""


class BitWidthError(DecodeError):
    """The file uses an unsupported bit width."""


class FFmpegError(DecodeError):
    pass


class CommunicationError(FFmpegError):
    """Raised when the output of FFmpeg is not parseable."""


class NotInstalledError(FFmpegError):
    """Could not find the ffmpeg binary."""


class ReadTimeoutError(FFmpegError):
    """Reading from the ffmpeg command-line tool timed out."""


def byteswap(s):
    """Swaps the endianness of the bytesting s, which must be an array
    of shorts (16-bit signed integers). This is probably less efficient
    than it should be.
    """
    assert len(s) % 2 == 0
    parts = []
    for i in range(0, len(s), 2):
        chunk = s[i: i + 2]
        newchunk = struct.pack('<h', *struct.unpack('>h', chunk))
        parts.append(newchunk)
    return b''.join(parts)


class RawAudioFile(object):
    """An AIFF, WAV, or Au file that can be read by the Python standard
    library modules ``wave``, ``aifc``, and ``sunau``."""
    def __init__(self, filename):
        self._fh = open(filename, 'rb')  
        try:  # aifc format
            self._file = aifc.open(self._fh)
        except aifc.Error:
            # Return to the beginning of the file to try the next reader.
            self._fh.seek(0)
        else:
            self._needs_byteswap = True
            self._check()
            return

        try:  # .wav format
            self._file = wave.open(self._fh)
        except wave.Error:
            self._fh.seek(0)
            pass
        else:
            self._needs_byteswap = False
            self._check()
            return

        try:  # sunau format. 
            self._file = sunau.open(self._fh)
        except sunau.Error:
            self._fh.seek(0)
            pass
        else:
            self._needs_byteswap = True
            self._check()
            return

        # None of the three libraries could open the file.
        self._fh.close()
        raise UnsupportedError()

    def _check(self):
        """Check that the files' parameters allow us to decode it and
        raise an error otherwise.
        """
        if self._file.getsampwidth() not in SUPPORTED_WIDTHS:
            self.close()
            raise BitWidthError()

    def close(self):
        """Close the underlying file."""
        self._file.close()
        self._fh.close()

    @property
    def channels(self):
        """Number of audio channels."""
        return self._file.getnchannels()

    @property
    def samplerate(self):
        """Sample rate in Hz."""
        return self._file.getframerate()

    @property
    def duration(self):
        """Length of the audio in seconds (a float)."""
        return float(self._file.getnframes()) / self.samplerate

    def read_data(self, block_samples=1024):
        """Generates blocks of PCM data found in the file."""
        old_width = self._file.getsampwidth()

        while True:
            data = self._file.readframes(block_samples)
            if not data:
                break

            # Make sure we have the desired bitdepth and endianness.
            data = audioop.lin2lin(data, old_width, TARGET_WIDTH)
            if self._needs_byteswap and self._file.getcomptype() != 'sowt':
                # Big-endian data. Swap endianness.
                data = byteswap(data)
            yield data

    # Context manager.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # Iteration.
    def __iter__(self):
        return self.read_data()


# This part is for ffmpeg read. 
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
    # """Detect if the FFmpeg backend can be used on this system."""
    # proc = popen_multiple(
    #     COMMANDS,
    #     ['-version'],
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    # )
    # proc.wait()
    # return (proc.returncode == 0)
    """Detect whether the FFmpeg backend can be used on this system.
    """
    try:
        proc = popen_multiple(
            COMMANDS,
            ['-version'],
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


class FFmpegAudioFile(object):
    """An audio file decoded by the ffmpeg command-line utility."""
    def __init__(self, filename, block_size=4096):
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
            previous_error_mode = \
                ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX)
            ctypes.windll.kernel32.SetErrorMode(
                previous_error_mode | SEM_NOGPFAULTERRORBOX
            )
        try:
            self.devnull = open(os.devnull)

            self.proc = popen_multiple(
                COMMANDS,
                ['-i', filename, '-f', 's16le', '-'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=self.devnull,
            )

        except OSError:
            raise NotInstalledError()

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
        self._get_info()

        # Start a separate thread to read the rest of the data from
        # stderr. This (a) avoids filling up the OS buffer and (b)
        # collects the error output for diagnosis.
        self.stderr_reader = QueueReaderThread(self.proc.stderr)
        self.stderr_reader.start()

    def read_data(self, timeout=10.0):
        """Read blocks of raw PCM data from the file."""
        # Read from stdout in a separate thread and consume data from
        # the queue.
        start_time = time.time()
        while True:
            # Wait for data to be available or a timeout.
            data = None
            try:
                data = self.stdout_reader.queue.get(timeout=timeout)
                if data:
                    yield data
                else:
                    # End of file.
                    break
            except queue.Empty:
                # Queue read timed out.
                end_time = time.time()
                if not data:
                    if end_time - start_time >= timeout:
                        # Nothing interesting has happened for a while --
                        # FFmpeg is probably hanging.
                        raise ReadTimeoutError('ffmpeg output: {}'.format(
                            ''.join(self.stderr_reader.queue.queue)
                        ))
                    else:
                        start_time = end_time
                        # Keep waiting.
                        continue

    def _get_info(self):
        """Reads the tool's output from its stderr stream, extracts the
        relevant information, and parses it.
        """
        out_parts = []
        while True:
            line = self.proc.stderr.readline()
            if not line:
                # EOF and data not found.
                raise CommunicationError("stream info not found")

            # In Python 3, result of reading from stderr is bytes.
            if isinstance(line, bytes):
                line = line.decode('utf8', 'ignore')

            line = line.strip().lower()

            if 'no such file' in line:
                raise IOError('file not found')
            elif 'invalid data found' in line:
                raise UnsupportedError()
            elif 'duration:' in line:
                out_parts.append(line)
            elif 'audio:' in line:
                out_parts.append(line)
                self._parse_info(''.join(out_parts))
                break

    def _parse_info(self, s):
        """Given relevant data from the ffmpeg output, set audio
        parameter fields on this object.
        """
        # Sample rate.
        match = re.search(r'(\d+) hz', s)
        if match:
            self.samplerate = int(match.group(1))
        else:
            self.samplerate = 0

        # Channel count.
        match = re.search(r'hz, ([^,]+),', s)
        if match:
            mode = match.group(1)
            if mode == 'stereo':
                self.channels = 2
            else:
                cmatch = re.match(r'(\d+)\.?(\d)?', mode)
                if cmatch:
                    self.channels = sum(map(int, cmatch.group().split('.')))
                else:
                    self.channels = 1
        else:
            self.channels = 0

        # Duration.
        match = re.search(
            r'duration: (\d+):(\d+):(\d+).(\d)', s
        )
        if match:
            durparts = list(map(int, match.groups()))
            duration = (durparts[0] * 60 * 60 + durparts[1] * 60 + durparts[2] + float(durparts[3]) / 10)
            self.duration = duration
        else:
            # No duration found.
            self.duration = 0

    def close(self):
        """Close the ffmpeg process used to perform the decoding."""
        if hasattr(self, 'proc'):
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
            if hasattr(self, 'stderr_reader'):
                self.stderr_reader.join()
            if hasattr(self, 'stdout_reader'):
                self.stdout_reader.join()

            # Close the stdout and stderr streams that were opened by Popen,
            # which should occur regardless of if the process terminated
            # cleanly.
            self.proc.stdout.close()
            self.proc.stderr.close()
        # Close the handle to os.devnull, which is opened regardless of if
        # a subprocess is successfully created.
        self.devnull.close()

    def __del__(self):
        self.close()

    # Iteration.
    def __iter__(self):
        return self.read_data()

    # Context manager.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def available_backends():
    """Returns a list of backends that are available on this system."""
    # Standard-library WAV and AIFF readers.
    ab = [RawAudioFile]
    # Audioread also supports other backends such as coreaudio and gst. But 
    # to simplify, we only use the standard library and ffmpeg. 
    try:
        if ffmpeg_available():  # FFmpeg.
            ab.append(FFmpegAudioFile)
    except:
        warn("Fail to find FFMPEG backend, please refer to project Github page for installation guide. For now Mp3 is not supported.")
    return ab


def audio_read(fp):
    backends = available_backends()
    for BackendClass in backends:
        try:
            return BackendClass(fp)
        except DecodeError:
            pass
    raise NoBackendError("Couldn't find a suitable backend to load the file. Most likely FFMPEG is not installed. Check github repo for installation guide.")  # If all backends fails 