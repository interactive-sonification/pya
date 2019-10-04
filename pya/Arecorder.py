# Arecorder class
import time
import logging
from sys import platform
from warnings import warn
import numpy as np
import pyaudio
from . import Asig
from . import Aserver
from enum import Enum, unique


_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


@unique
class State(Enum):
    ERROR = -1
    DEFAULT = 0
    STOPPED = 1
    RECORDING = 2
    PAUSED = 3


class Arecorder(Aserver):
    """pya audio recorder
    Based on pyaudio, uses callbacks to save audio data
    for pyaudio signals into ASigs

    Examples:
    -----------
    from pya import *

    """

    def __init__(self, sr=44100, bs=256, input_device=None, output_device=None, channels=2, format=pyaudio.paFloat32):
        # TODO I think the channel should not be set.
        super().__init__(sr=sr, bs=bs, device=output_device, channels=channels, format=format)
        self.record_buffer = []
        self.recordings = []  # store recorded Asigs, time stamp in label
        self._input_device = 0
        if input_device is None:
            self.input_device = self.pa.get_default_input_device_info()[
                'index']
        else:
            self.input_device = input_device
        self.state = State.DEFAULT

    @property
    def input_device(self):
        return self._input_device

    @input_device.setter
    def input_device(self, val):
        self._input_device = val
        self.input_device_dict = self.pa.get_device_info_by_index(
            self._input_device)
        self.max_in_chn = self.input_device_dict['maxInputChannels']
        if self.max_in_chn < self.channels:
            warn(
                f"Aserver: warning: {self.channels}>{self.max_in_chn} channels requested - truncated.")
            self.channels = self.max_in_chn

    def boot(self):
        """boot recorder"""
        # when input = True, the channels refers to the input channels.
        self.pastream = self.pa.open(format=self.format, channels=self.channels,
                                     frames_per_buffer=self.bs, rate=self.sr,
                                     input_device_index=self.input_device, output=False,
                                     input=True, stream_callback=self._recorder_callback)
        self.boot_time = time.time()
        self.block_time = self.boot_time
        self.block_cnt = 0
        self.record_buffer = []
        self.pastream.start_stream()
        _LOGGER.info("Server Booted")
        return self

    def _recorder_callback(self, in_data, frame_count, time_info, flag):
        """Callback function during streaming. """
        self.block_cnt += 1
        if self.state == State.RECORDING:
            sigar = np.frombuffer(in_data, dtype=self.dtype)
            # (chunk length, chns)
            data_float = np.reshape(sigar, (len(sigar) // self.channels, self.channels))
            self.record_buffer.append(data_float)
            # E = 10 * np.log10(np.mean(data_float ** 2)) # energy in dB
            # os.write(1, f"\r{E}    | {self.block_cnt}".encode())
        return None, pyaudio.paContinue

    def record(self):
        self.record_buffer = []
        self.state = State.RECORDING

    def pause(self):
        self.state = State.PAUSED

    def resume(self):
        """If state is at paused, continue recording. Otherwise do nothing"""
        if self.state == State.PAUSED:
            self.state = State.RECORDING

    def stop(self):
        self.state = State.STOPPED
        if len(self.record_buffer) > 0:
            sig = np.squeeze(np.vstack(self.record_buffer))
            self.recordings.append(Asig(sig, self.sr, label=""))
        else:
            _LOGGER.info(" Stopped. There is no recording in the record_buffer")

    def reset_recordings(self):
        self.recordings = []
        self.state = State.DEFAULT

    def get_latest_recording(self):
        return self.recordings[-1]

    def __getattr__(self, item):
        if item.startswith('is_'):
            return self.state == getattr(State, item[3:])
        else:
            return super().__getattr__(item)

    def __dir__(self):  # for code completion
        return super().__dir__() + ['is_{}'.format(s.name) for s in State]
