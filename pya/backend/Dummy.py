from .base import BackendBase, StreamBase
import numpy as np
from threading import Thread
import time


class DummyBackend(BackendBase):

    dtype = 'float32'
    range = 1
    bs = 256

    def __init__(self):
        self.dummy_devices = [dict(maxInputChannels=10, maxOutputChannels=10, index=0, name="DummyDevice")]

    def get_device_count(self):
        return len(self.dummy_devices)

    def get_device_info_by_index(self, idx):
        return self.dummy_devices[idx]

    def get_default_input_device_info(self):
        return self.dummy_devices[0]

    def get_default_output_device_info(self):
        return self.dummy_devices[0]

    def open(self, *args, input_flag, output_flag, rate, frames_per_buffer, channels, stream_callback=None, **kwargs):
        checker = 'maxInputChannels' if input_flag else 'maxOutputChannels'
        if channels > self.dummy_devices[0][checker]:
            raise OSError("[Errno -9998] Invalid number of channels")
        stream = DummyStream(input_flag=input_flag, output_flag=output_flag, 
                             rate=rate, frames_per_buffer=frames_per_buffer, channels=channels,
                             stream_callback=stream_callback)
        stream.start_stream()
        return stream

    def process_buffer(self, buffer):
        return buffer

    def terminate(self):
        pass


class DummyStream(StreamBase):

    def __init__(self, input_flag, output_flag, frames_per_buffer, rate, channels, stream_callback):
        self.input_flag = input_flag
        self.output_flag = output_flag
        self.rate = rate
        self.stream_callback = stream_callback
        self.frames_per_buffer = frames_per_buffer
        self.channels = channels
        self._is_active = False
        self.in_thread = None
        self.out_thread = None
        self.samples_out = []

    def stop_stream(self):
        self._is_active = False
        while (self.in_thread and self.in_thread.is_alive()) or \
              (self.out_thread and self.out_thread.is_alive()):
            time.sleep(0.1)

    def close(self):
        self.stop_stream()

    def _generate_data(self):
        while self._is_active:
            sig = np.zeros(self.frames_per_buffer * self.channels, dtype=DummyBackend.dtype)
            self.stream_callback(sig, frame_count=None, time_info=None, flag=None)
            time.sleep(0.05)

    def _process_data(self):
        while self._is_active:
            data = self.stream_callback(None, None, None, None)
            if np.any(data):
                self.samples_out.append(data)
            time.sleep(0.05)

    def start_stream(self):
        self._is_active = True
        if self.input_flag:
            self.in_thread = Thread(target=self._generate_data)
            self.in_thread.daemon = True
            self.in_thread.start()
        if self.output_flag:
            self.out_thread = Thread(target=self._process_data)
            self.out_thread.daemon = True
            self.out_thread.start()

    def is_active(self):
        return self._is_active
