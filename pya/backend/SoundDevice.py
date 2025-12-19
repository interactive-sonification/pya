import sounddevice as sd
from .base import BackendBase, StreamBase
import numpy as np
import time

class SoundDeviceBackend(BackendBase):
    _boot_delay = 0.5  # a short delay to prevent PyAudio racing conditions
    bs = 512

    def __init__(self, format=np.float32):
        if not sd._initialized:
            sd._initialize()
        self.format = format
        self.dtype = format
        if format in [np.int16, "int16"]:
            self.range = 32767
        elif format in [np.float32, "float32"]:
            self.range = 1.0
        else:
            raise AttributeError(f"Aserver: currently unsupported pyaudio format {self.format}")

    def get_device_count(self):
        return len(sd.query_devices())

    def get_device_info_by_index(self, idx):
        return sd.query_devices(idx)

    def get_default_input_device_info(self):
        in_idx, _ = sd.default.device
        return sd.query_devices(in_idx)

    def get_default_output_device_info(self):
        _, out_idx = sd.default.device
        return sd.query_devices(out_idx)

    def open(self, rate, channels, input_flag, output_flag, frames_per_buffer, 
             input_device_index=None, output_device_index=None, start=True, 
             input_host_api_specific_stream_info=None, output_host_api_specific_stream_info=None, 
             stream_callback=None):
        kwargs = dict(
            samplerate=rate,
            blocksize=frames_per_buffer,
            device=(input_device_index, output_device_index),
            channels=channels,
            dtype=self.dtype,
            extra_settings=(
                input_host_api_specific_stream_info,
                output_host_api_specific_stream_info
            ),
            callback=stream_callback
        )
        if input_flag and output_flag:
            stream = sd.Stream(**kwargs)
        elif input_flag:
            stream = sd.InputStream(**kwargs)
        elif output_flag:
            stream = sd.OutputStream(**kwargs)
        else:
            raise ValueError("Both input flag and output flag were set to 0.")
        if start:
            stream.start()
        time.sleep(self._boot_delay)  # give stream some time to be opened completely
        return SoundDeviceStream(stream)

    def process_buffer(self, buffer):
        return buffer

    def terminate(self):
        sd._terminate()



class SoundDeviceStream(StreamBase):
    def __init__(self, sd_stream: sd.Stream):
        self.sd_stream = sd_stream
    def is_active(self):
        return self.sd_stream.active

    def start_stream(self):
        self.sd_stream.start()

    def stop_stream(self):
        self.sd_stream.stop()

    def close(self):
        self.sd_stream.close()
