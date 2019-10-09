import pyaudio
from .base import BackendBase


class PyAudioBackend(BackendBase):

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

    def open(self, channels, rate, input_flag, output_flag, frames_per_buffer, input_device_index=None,
             output_device_index=None, stream_callback=None):
        return self.pa.open(format=self.format, channels=channels, rate=rate, input=input_flag, output=output_flag,
                            frames_per_buffer=frames_per_buffer, output_device_index=output_device_index,
                            stream_callback=stream_callback)

    def process_buffer(self, buffer):
        return buffer, pyaudio.paContinue

    def terminate(self):
        self.pa.terminate()
