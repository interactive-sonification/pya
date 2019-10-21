from .base import BackendBase, StreamBase
import numpy as np
from threading import Thread
import time


class DummyBackend(BackendBase):

    dtype = 'float32'
    range = 1

    def __init__(self, dummy_devices=None):
        if dummy_devices is None:
            self.dummy_devices = [dict(maxInputChannels=2, maxOutputChannels=2, index=0, name="DummyDevice")]

    def get_device_count(self):
        return len(self.dummy_devices)

    def get_device_info_by_index(self, idx):
        return self.dummy_devices[idx]

    def get_default_input_device_info(self):
        return self.dummy_devices[0]

    def get_default_output_device_info(self):
        return self.dummy_devices[0]

    def open(self, *args, input_flag, output_flag, rate, stream_callback=None, **kwargs):
        stream = DummyStream(input_flag=input_flag, output_flag=output_flag, rate=rate, stream_callback=stream_callback)
        stream.start_stream()
        return stream

    def process_buffer(self, buffer):
        return buffer

    def terminate(self):
        pass


class DummyStream(StreamBase):

    def __init__(self, input_flag, output_flag, rate, stream_callback):
        self.input_flag = input_flag
        self.output_flag = output_flag
        self.rate = rate
        self.stream_callback = stream_callback
        self._is_active = False
        self.cb_thread = None

    def stop_stream(self):
        self._is_active = False

    def close(self):
        pass

    def _generate_data(self):
        while self._is_active:
            sig = np.zeros((self.rate, 1), dtype=DummyBackend.dtype)
            self.stream_callback(sig, frame_count=None, time_info=None, flag=None)
            time.sleep(0.01)

    def start_stream(self):
        self._is_active = True
        if self.input_flag:
            self.cb_thread = Thread(target=self._generate_data)
            self.cb_thread.start()

    def is_active(self):
        return self._is_active
