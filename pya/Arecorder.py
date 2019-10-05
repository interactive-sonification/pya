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


class Arecorder(Aserver):
    """pya audio recorder
    Based on pyaudio, uses callbacks to save audio data
    for pyaudio signals into ASigs

    Examples:
    -----------
    from pya import *

    """

    def __init__(self, sr=44100, bs=256, input_device=None, channels=2, format=pyaudio.paFloat32):
        # TODO I think the channel should not be set.
        super().__init__(sr=sr, bs=bs, channels=channels, format=format)
        self.record_buffer = []
        self.recordings = []  # store recorded Asigs, time stamp in label
        self._input_device = 0
        if input_device is None:
            self.input_device = self.pa.get_default_input_device_info()[
                'index']
        else:
            self.input_device = input_device
        self._recording = False

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
            warn(f"Aserver: warning: {self.channels}>{self.max_in_chn} channels requested - truncated.")
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
        self._recording = False
        self.pastream.start_stream()
        _LOGGER.info("Server Booted")
        return self

    def _recorder_callback(self, in_data, frame_count, time_info, flag):
        """Callback function during streaming. """
        self.block_cnt += 1
        if self._recording:
            sigar = np.frombuffer(in_data, dtype=self.dtype)
            # (chunk length, chns)
            data_float = np.reshape(sigar, (len(sigar) // self.channels, self.channels))
            self.record_buffer.append(data_float)
            # E = 10 * np.log10(np.mean(data_float ** 2)) # energy in dB
            # os.write(1, f"\r{E}    | {self.block_cnt}".encode())
        return None, pyaudio.paContinue

    def record(self):
        """Activate recording"""
        self._recording = True

    def pause(self):
        """Pause the recording, but the record_buffer remains"""
        self._recording = False

    def stop(self):
        """Stop recording, then stores the data from record_buffer into recordings"""
        self._recording = False
        if len(self.record_buffer) > 0:
            sig = np.squeeze(np.vstack(self.record_buffer))
            self.recordings.append(Asig(sig, self.sr, label=""))
            self.record_buffer = []
        else:
            _LOGGER.info(" Stopped. There is no recording in the record_buffer")

    def __repr__(self):
        state = False
        if self.pastream:
            state = self.pastream.is_active()
        msg = f"""Arecorder: sr: {self.sr}, blocksize: {self.bs}, Stream Active: {state}
           Input: {self.input_device_dict['name']}, Index: {self.input_device_dict['index']}
           """        
        return msg
