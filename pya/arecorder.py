# Arecorder class
import logging
import numbers
import time
from typing import Optional, Union

import numpy as np
from . import Asig
from . import Aserver
from pyamapping import db_to_amp


_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class Arecorder(Aserver):
    """pya audio recorder Based on pyaudio, uses callbacks to save audio data
    for pyaudio signals into ASigs

    Examples:
    -----------
    >>> from pya import Arecorder
    >>> import time
    >>> ar = Arecorder().boot()
    >>> ar.record()
    >>> time.sleep(1)
    >>> ar.stop()
    >>> print(ar.recordings)  # doctest:+ELLIPSIS
    [Asig(''): ... x ... @ 44100Hz = ...
    """

    def __init__(self, sr: int = 44100, bs: int = 256, device: Optional[int] = None,
                 channels: Optional[int] = None, backend=None, **kwargs):
        super().__init__(sr=sr, bs=bs, device=device, 
                         backend=backend, **kwargs)
        self.record_buffer = []
        self.recordings = []  # store recorded Asigs, time stamp in label
        self._recording = False
        self._record_all = True
        self.tracks = slice(None)
        self._device = self.backend.get_default_input_device_info()['index'] if device is None else device
        self._channels = channels or self.max_in_chn
        self.gains = np.ones(self._channels)

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, val: int):
        """
        Set the number of channels. Aserver needs reboot.
        """
        if val > self.max_in_chn:
            raise ValueError(f"AServer: channels {val} > max {self.max_in_chn}")
        self._channels = val

    def set_tracks(self, tracks: Union[list, np.ndarray], gains: Union[list, np.ndarray]):
        """Define the number of track to be recorded and their gains.

        parameters
        ----------
        tracks : list or numpy.ndarray
            A list of input channel indices. By default None (record all channels)
        gains : list of numpy.ndarray
            A list of gains in decibel. Needs to be same length as tracks.
        """
        if isinstance(tracks, list) and isinstance(gains, list):
            if len(tracks) != len(gains):
                raise AttributeError("tracks and gains should be equal length lists.")
            elif len(tracks) > self.channels or len(gains) > self.channels:
                raise AttributeError("argument cannot be larger than channels.")
            self.tracks = tracks
            self.gains = np.array([db_to_amp(g) for g in gains], dtype="float32")
        elif isinstance(tracks, numbers.Number) and isinstance(gains, numbers.Number):
            self.tracks = [tracks]
            self.gains = db_to_amp(gains)
        else:
            raise TypeError("Arguments need to be both list or both number.")

    def reset(self):
        self.tracks = slice(None)
        self.gains = np.ones(self.channels)

    def boot(self):
        self.boot_time = time.time()
        self.block_time = self.boot_time
        # self.block_cnt = 0
        self.record_buffer = []
        self._recording = False
        self.stream = self.backend.open(rate=self.sr, channels=self.channels, frames_per_buffer=self.bs,
                                        input_device_index=self.device, output_flag=False,
                                        input_flag=True, stream_callback=self._recorder_callback)
        _LOGGER.info("Server Booted")
        return self

    def _recorder_callback(self, in_data, frame_count, time_info, flag):
        """Callback function during streaming. """
        # self.block_cnt += 1
        if self._recording:
            sigar = np.frombuffer(in_data, dtype=self.backend.dtype)
            # (chunk length, chns)
            data_float = np.reshape(sigar, (len(sigar) // self.channels, self.channels))
            data_float = data_float[:, self.tracks] * self.gains  # apply channel selection and gains.
            # if not self._record_all
            self.record_buffer.append(data_float)
            # E = 10 * np.log10(np.mean(data_float ** 2)) # energy in dB
            # os.write(1, f"\r{E}    | {self.block_cnt}".encode())
        return self.backend.process_buffer(None)

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
        if self.stream:
            state = self.stream.is_active()
        msg = f"""Arecorder: sr: {self.sr}, blocksize: {self.bs}, Stream Active: {state}
           Input: {self.device_dict['name']}, Index: {self.device_dict['index']}
           """
        return msg
