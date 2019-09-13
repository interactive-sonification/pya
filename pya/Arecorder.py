# Arecorder class 
import time
import logging
from sys import platform
from warnings import warn
import numpy as np
import pyaudio
from . import Asig

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class Arecorder:
    """pya audio recorder
    Based on pyaudio, uses callbacks to save audio data
    for pyaudio signals into ASigs
    
    Examples:
    -----------
    from pya import *

    """

    def __init__(self, sr=44100, bs=256, device=None, channels=2, format=pyaudio.paFloat32):
        self.sr = sr
        self.bs = bs
        self.pa = pyaudio.PyAudio()
        self.channels = channels
        self._status = pyaudio.paComplete
        self.input_devices = []
        self.recordings = []  # store recorded Asigs, time stamp in label
        for i in range(self.pa.get_device_count()):
            if self.pa.get_device_info_by_index(i)['maxInputChannels'] > 0:
                self.input_devices.append(self.pa.get_device_info_by_index(i))
        if device is None:
            self.device = self.pa.get_default_input_device_info()['index']
        else:
            self.device = device
        self.pastream = None
        self.format = format
        self.dtype = 'float32'  # for pyaudio.paFloat32
        self.range = 1.0
        self.boot_time = None  # time.time() when stream starts
        self.block_cnt = None  # nr. of callback invocations
        self.block_duration = self.bs / self.sr  # nominal time increment per callback
        self.block_time = None  # estimated time stamp for current block
        self._stop = True
        if self.format == pyaudio.paInt16:
            self.dtype = 'int16'
            self.range = 32767
        if self.format not in [pyaudio.paInt16, pyaudio.paFloat32]:
            warn(f"Arecorder: currently unsupported pyaudio format {self.format}")
        self.empty_buffer = np.zeros((self.bs, self.channels), dtype=self.dtype)
        self.record_buffer = [] 
        self.device_dict = self.pa.get_device_info_by_index(self.device)

    def _recorder_callback(self, in_data, frame_count, time_info, flag):
        self.block_cnt += 1
        if self.recording_flag:
            # print(self.block_cnt, len(in_data), )
            sigar = np.frombuffer(in_data, dtype=self.dtype)
            data_float = np.reshape(sigar, (len(sigar)//self.channels, self.channels)) # (chunk length, chns)
            self.record_buffer.append(data_float)
            # E = 10 * np.log10(np.mean(data_float ** 2)) # energy in dB
            # os.write(1, f"\r{E}    | {self.block_cnt}".encode())
        return None, pyaudio.paContinue 

    def boot(self): 
        """boot recorder"""
        self.pastream = self.pa.open(format=self.format, channels=self.channels,
                        frames_per_buffer=self.bs, rate=self.sr,
                        input_device_index=self.device, output=False, input=True, 
                        stream_callback=self._recorder_callback)        
        self.boot_time = time.time()
        self.block_time = self.boot_time
        self.block_cnt = 0
        self.record_buffer = []
        self.recording_flag = False
        self.pastream.start_stream()
        _LOGGER.info("Server Booted")
        return self

    def quit(self):
        """Arecorder quit: stop stream and terminate pa"""
        if not self.pastream.is_active():
            _LOGGER.info("Arecorder:quit: stream not active")
            return -1
        try:
            self.pastream.stop_stream()
            self.pastream.close()
            _LOGGER.info("Arecorder stopped.")
        except AttributeError:
            _LOGGER.info("No stream found...")
        self.pastream = None
    
    def record(self):
        if self.recording_flag:
            _LOGGER.info("Arecorder:record: is already recording")
        else:
            self.record_buffer = []
            self.recording_flag = True

    def pause(self):
        if self.recording_flag:
            self.recording_flag = False
        else:
            _LOGGER.info("Arecorder:pause: can pause only when recording")

    def resume(self):
        if self.recording_flag:
            _LOGGER.info("Arecorder:resume: can only resume when paused or stopped")
        else:
            self.recording_flag = True

    def stop(self):
        if self.recording_flag:
            self.recording_flag = False
            sig = np.squeeze(np.vstack(self.record_buffer))
            self.recordings.append(Asig(sig, self.sr, label=""))
        else:
            _LOGGER.info("Arecorder:stop: can only stop when recording")

    def reset_recordings(self):
        self.recordings = []

    def get_latest_recording(self):
        return self.recordings[-1]

    def __del__(self):
        self.pa.terminate()

    def __repr__(self):
        state = False
        if self.pastream:
            state = self.pastream.is_active()
        msg = f"""Arecorder: sr: {self.sr}, blocksize: {self.bs}, Stream Active: {state}, 
            Device: {self.device_dict["name"]}, Index: {self.device_dict['index']}"""
        return msg

