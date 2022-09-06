from .base import BackendBase

import pyaudio
import time


class PyAudioBackend(BackendBase):

    _boot_delay = 0.5  # a short delay to prevent PyAudio racing conditions
    bs = 512

    def __init__(self, format=pyaudio.paFloat32):
        self.pa = pyaudio.PyAudio()
        self.format = format
        if format == pyaudio.paInt16:
            self.dtype = 'int16'
            self.range = 32767
        elif format == pyaudio.paFloat32:
            self.dtype = 'float32'  # for pyaudio.paFloat32
            self.range = 1.0
        else:
            raise AttributeError(f"Aserver: currently unsupported pyaudio format {self.format}")

    def get_device_count(self):
        return self.pa.get_device_count()

    def get_device_info_by_index(self, idx):
        return self.pa.get_device_info_by_index(idx)

    def get_default_input_device_info(self):
        return self.pa.get_default_input_device_info()

    def get_default_output_device_info(self):
        return self.pa.get_default_output_device_info()

    def open(self, rate, channels, input_flag, output_flag, frames_per_buffer, 
             input_device_index=None, output_device_index=None, start=True, 
             input_host_api_specific_stream_info=None, output_host_api_specific_stream_info=None, 
             stream_callback=None):
        stream = self.pa.open(rate=rate, channels=channels, format=self.format, input=input_flag, output=output_flag,
                              input_device_index=input_device_index, output_device_index=output_device_index,
                              frames_per_buffer=frames_per_buffer, start=start,
                              input_host_api_specific_stream_info=input_host_api_specific_stream_info,
                              output_host_api_specific_stream_info=output_host_api_specific_stream_info,
                              stream_callback=stream_callback)
        time.sleep(self._boot_delay)  # give stream some time to be opened completely
        return stream

    def process_buffer(self, buffer):
        return buffer, pyaudio.paContinue

    def terminate(self):
        if self.pa:
            self.pa.terminate()
            self.pa = None
