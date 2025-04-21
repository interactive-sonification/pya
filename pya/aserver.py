from .helper.backend import determine_backend
from .helper.ringbuffer import RingBuffer
import copy
import logging
import time
from typing import Optional, Union
from warnings import warn

import numpy as np


_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class AudioEvent:
    """Audio Event with timing. This is used in Aserver for scheduling playback."""

    def __init__(self, signal: np.ndarray, onset: float, out_channel: int = 0):
        self._signal = signal
        self._onset = onset
        self._out_channel = out_channel
        self._pos = 0
        self.completed = False

    @property
    def pos(self) -> int:
        return self._pos

    @property
    def onset(self) -> float:
        return self._onset

    @property
    def out_channel(self) -> int:
        return self._out_channel

    @property
    def signal(self) -> np.ndarray:
        return self._signal

    @property
    def remaining_samples(self) -> int:
        return self._signal.shape[0] - self.pos

    def get_next_chunk(self, max_samples: int) -> np.ndarray:
        """Retrieves the next portion of audio samples from the signal, advancing the playback position.
        The function handles both partial and complete chunks, automatically marking the event as completed
        when all samples have been played.

        Parameters
        ----------
        max_samples : int
            Maximum number of samples to return

        Returns
        -------
        np.ndarray
            Array of audio samples, empty if event is completed
        """
        if self.completed:
            return np.array([])

        samples = min(max_samples, self.remaining_samples)
        chunk = self._signal[self._pos: self._pos + samples]
        self._pos += samples

        if self._pos >= self._signal.shape[0]:
            self.completed = True

        return chunk


class EventScheduler:
    """Manages scheduling of audio events for playback."""

    def __init__(self, sr: int):
        self._sr = sr
        self.pending_events = []
        self.active_events = []

    def schedule(self, event: AudioEvent):
        idx = 0
        while (idx < len(self.pending_events) and self.pending_events[idx].onset < event.onset):
            idx += 1

        self.pending_events.insert(idx, event)

    def process(
        self, current_time: float, output_buffer: np.ndarray, block_size: int
    ) -> np.ndarray:
        end_time = current_time + block_size / self._sr

        idx = 0
        while idx < len(self.pending_events):
            if self.pending_events[idx].onset <= end_time:
                self.active_events.append(self.pending_events.pop(idx))
            else:
                idx += 1

        idx = 0
        while idx < len(self.active_events):
            event = self.active_events[idx]

            if event.onset > current_time:
                offset_samples = int((event.onset - current_time) * self._sr)
            else:
                offset_samples = 0

            audio_chunk = event.get_next_chunk(block_size - offset_samples)

            if len(audio_chunk) > 0:
                channels = min(
                    audio_chunk.shape[1] if len(audio_chunk.shape) > 1 else 1,
                    output_buffer.shape[1] - event.out_channel,
                )

                if len(audio_chunk.shape) == 1:
                    audio_chunk = audio_chunk.reshape(-1, 1)

                end_offset = offset_samples + audio_chunk.shape[0]
                end_channel = event.out_channel + channels

                output_buffer[
                    offset_samples:end_offset, event.out_channel: end_channel
                ] += audio_chunk[:, :channels]

            if event.completed:
                self.active_events.pop(idx)
            else:
                idx += 1

        return output_buffer


class Aserver:
    """Pya audio server
    Based on pyaudio, works as a FIFO style audio stream pipeline,
    allowing Asig.play() to send audio segement into the stream.

    Examples:
    -----------
    >>> from pya import *
    >>> ser = Aserver()
    >>> ser.boot()
    AServer: sr: 44100, blocksize: ...,
             Stream Active: True, Device: ...
    >>> asine = Ugen().sine()
    >>> asine.play(server=ser)
    Asig('sine'): 1 x 44100 @ 44100Hz = 1.000s cn=['0']
    >>> ser.quit()  # Important to call quit() to close the stream when you are done. Or use context manager.
    """

    default = None  # that's the default Aserver if Asigs play via it

    @staticmethod
    def startup_default_server(**kwargs):
        if Aserver.default is None:
            _LOGGER.info("Aserver startup_default_server: create and boot")
            Aserver.default = Aserver(**kwargs)  # using all default settings
            Aserver.default.boot()
            _LOGGER.info("Default server info: %s", Aserver.default)
        else:
            _LOGGER.info("Aserver default_server already set.")
        return Aserver.default

    @staticmethod
    def shutdown_default_server():
        if isinstance(Aserver.default, Aserver):
            Aserver.default.quit()
            del Aserver.default
            Aserver.default = None
        else:
            warn("Aserver:shutdown_default_server: no default_server to shutdown")

    def __init__(
        self,
        sr: int = 44100,
        bs: Optional[int] = None,
        device: Optional[int] = None,
        channels: Optional[int] = None,
        backend=None,
        **kwargs,
    ):
        """Aserver manages an pyaudio stream, using its aserver callback
        to feed dispatched signals to output at the right time.

        Parameters
        ----------
        sr : int
            Sampling rate (Default value = 44100)
        bs : int
            Override block size or buffer size set by chosen backend
        device : int
            The device index based on pya.device_info(), default is None which will set
            the default device from PyAudio
        channels : int
            number of channel, default is the max output channels of the device
        kwargs : backend parameter

        Returns
        -------
        _ : numpy.ndarray
            numpy array of the recorded audio signal.
        """
        # TODO check if channels is overwritten by the device.
        self.sr = sr
        self.stream = None
        self.backend = determine_backend(**kwargs) if backend is None else backend
        self.bs = bs or self.backend.bs
        # Get audio devices to input_device and output_device
        self.input_devices = []
        self.output_devices = []
        for i in range(self.backend.get_device_count()):
            if int(self.backend.get_device_info_by_index(i)["maxInputChannels"]) > 0:
                self.input_devices.append(self.backend.get_device_info_by_index(i))
            if int(self.backend.get_device_info_by_index(i)["maxOutputChannels"]) > 0:
                self.output_devices.append(self.backend.get_device_info_by_index(i))

        self._device = (
            self.backend.get_default_output_device_info()["index"]
            if device is None
            else device
        )
        self._channels = channels or self.max_out_chn

        # Give extra periods for the ring buffer
        self.ring_buffer = RingBuffer(self.bs * 4, self.channels)

        self.event_scheduler = EventScheduler(self.sr)

        self.mix_buffer = np.zeros((self.bs, self.channels), dtype=self.backend.dtype)

        self.stream = None
        self.gain = 1.0
        self.boot_time = 0  # time.time() when stream starts
        self.block_time = 0  # estimated time stamp for current block
        self._stop = True
        self._is_active = False

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, val: int):
        """
        Set the number of channels. Aserver needs reboot.
        """
        if val > self.max_out_chn:
            raise ValueError(f"AServer: channels {val} > max {self.max_out_chn}")
        self._channels = val

    @property
    def device_dict(self):
        return self.backend.get_device_info_by_index(self._device)

    @property
    def max_out_chn(self) -> int:
        return int(self.device_dict["maxOutputChannels"])

    @property
    def max_in_chn(self) -> int:
        return int(self.device_dict["maxInputChannels"])

    @property
    def is_active(self) -> bool:
        return self.stream is not None and self.stream.is_active()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, val):
        self._device = (
            val
            if val is not None
            else self.backend.get_default_output_device_info()["index"]
        )
        if self.max_out_chn < self.channels:
            warn(
                f"Aserver: warning: {self.channels}>{self.max_out_chn} channels requested - truncated."
            )
            self.channels = self.max_out_chn

    def __repr__(self):
        msg = f"""AServer: sr: {self.sr}, blocksize: {self.bs},
         Stream Active: {self.is_active}, Device: {self.device_dict['name']}, Index: {self.device_dict['index']}"""
        return msg

    def get_devices(self, verbose: bool = False):
        """Return (and optionally print) available input and output device"""
        if verbose:
            print("Input Devices: ")
            [
                print(
                    f"Index: {i['index']}, Name: {i['name']},  Channels: {i['maxInputChannels']}"
                )
                for i in self.input_devices
            ]
            print("Output Devices: ")
            [
                print(
                    f"Index: {i['index']}, Name: {i['name']}, Channels: {i['maxOutputChannels']}"
                )
                for i in self.output_devices
            ]
        return self.input_devices, self.output_devices

    def set_device(self, idx: int, reboot: bool = True):
        """Set audio device, an alternative way is to direct set the device property, i.e. Aserver.device = 1,
        but that will not reboot the server.

        Parameters
        ----------
        idx : int
            Index of the device
        reboot : bool
            If true the server will reboot. (Default value = True)
        """
        self._device = idx
        if reboot:
            try:
                self.quit()
            except AttributeError:
                _LOGGER.warning(" Reboot while no active stream")
            try:
                self.boot()
            except OSError:
                raise OSError("Error: Invalid device. Server did not boot.")

    def boot(self):
        """boot Aserver = start stream, setting its callback to this callback."""
        if self.is_active:
            _LOGGER.info("Aserver already running...")
            return -1
        self.boot_time = time.time()
        self.block_time = self.boot_time
        self.ring_buffer.clear()
        # self.block_cnt = 0
        self.stream = self.backend.open(
            channels=self.channels,
            rate=self.sr,
            input_flag=False,
            output_flag=True,
            frames_per_buffer=self.bs,
            output_device_index=self.device,
            stream_callback=self._play_callback,
        )
        self._is_active = self.stream.is_active()
        _LOGGER.info("Server Booted")
        return self

    def quit(self):
        """Aserver quit server: stop stream and terminate pa"""
        if not self.is_active:
            _LOGGER.info("Stream not active")
            return -1
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                _LOGGER.info("Aserver stopped.")
        except AttributeError:
            _LOGGER.info("No stream found...")
        self.stream = None

    def play(self, asig, onset: Union[int, float] = 0, out: int = 0, **kwargs):
        """Dispatch asigs or arrays for given onset.

        asig: pya.Asig
            An Asig object
        onset: int or float
            Time when the sound should play, 0 means asap
        out: int
            Output channel
        """
        if not self.is_active:
            raise RuntimeError("Aserver not active")
        self._stop = False

        if asig.sr != self.sr:
            asig = asig.resample(self.sr)

        if onset < 1e6:
            rt_onset = time.time() + onset
        else:
            rt_onset = onset

        if asig.sig.dtype != self.backend.dtype:
            asig_data = asig.sig.astype(self.backend.dtype)
        else:
            asig_data = asig.sig

        event = AudioEvent(asig_data, rt_onset, out)
        self.event_scheduler.schedule(event)

        if "block" in kwargs and kwargs["block"]:
            if onset > 0:  # here really omset and not rt_onset!
                _LOGGER.warning("blocking inactive with play(onset>0)")
            else:
                time.sleep(asig.get_duration())

        return self

    def _play_callback(self, in_data, frame_count, time_info, flag):
        """callback function, called from pastream thread when data needed."""
        current_time = self.block_time
        self.block_time += self.bs / self.sr

        # This is fast.
        self.mix_buffer.fill(0)

        if not self._stop:
            self.event_scheduler.process(current_time, self.mix_buffer, self.bs)

        output_data = self.mix_buffer * (self.backend.range * self.gain)

        return self.backend.process_buffer(output_data)

    def stop(self):
        self._stop = True
        self.event_scheduler.pending_events = []
        self.event_scheduler.active_events = []
        self.ring_buffer.clear()

    def __enter__(self):
        return self.boot()

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit"""
        _LOGGER.info("Exiting context manager. Cleaning up stream and backend")
        self.quit()
        self.backend.terminate()

    def __del__(self):
        """Backup cleanup, only if context manager wasn't used"""
        if hasattr(self, "stream") and self.stream is not None:
            try:
                self.quit()
                self.backend.terminate()
            except:
                pass  # Ignore cleanup errors during shutdown
