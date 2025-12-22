import sounddevice as sd
from .base import BackendBase, StreamBase
import numpy as np
import time

def translate_dict_keys(input_dict: dict, translate_dict: dict):
    # Translates keys in input_dict to be replaced by the values that they are mapped to by translate_dict
    return {
        translate_dict.get(key, key): value
        for key, value in input_dict.items()
    }

class SoundDeviceBackend(BackendBase):
    _boot_delay = 0.5  # a short delay to prevent PyAudio racing conditions
    bs = 512
    
    translate_dict = {
        'hostapi': 'hostApi',
        'max_input_channels': 'maxInputChannels',
        'max_output_channels': 'maxOutputChannels',
        'default_low_input_latency': 'defaultLowInputLatency',
        'default_low_output_latency': 'defaultLowOutputLatency',
        'default_high_input_latency': 'defaultHighInputLatency',
        'default_high_output_latency': 'defaultHighOutputLatency',
        'default_samplerate': 'defaultSampleRate',
    }

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
        return translate_dict_keys(sd.query_devices(idx), self.translate_dict)

    def get_default_input_device_info(self):
        in_idx, _ = sd.default.device
        return self.get_device_info_by_index(in_idx)

    def get_default_output_device_info(self):
        _, out_idx = sd.default.device
        return self.get_device_info_by_index(out_idx)

    def open(self, rate, channels, input_flag, output_flag, frames_per_buffer, 
             input_device_index=None, output_device_index=None, start=True, 
             input_host_api_specific_stream_info=None, output_host_api_specific_stream_info=None, 
             stream_callback=None):
        if not input_flag and not output_flag:
            raise ValueError("Input flag and output flag were both False!")
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
            callback=self.make_callback(stream_callback, input_flag, output_flag)
        )
        if input_flag and output_flag:
            stream = sd.Stream(**kwargs)
        elif input_flag:
            stream = sd.InputStream(**kwargs)
        else:
            stream = sd.OutputStream(**kwargs)
        if start:
            stream.start()
        time.sleep(self._boot_delay)  # give stream some time to be opened completely
        return SoundDeviceStream(stream)

    def process_buffer(self, buffer):
        return buffer

    def terminate(self):
        sd._terminate()
    
    def make_callback(self, server_callback, input_flag, output_flag):
        if input_flag and output_flag:
            def new_callback(indata, outdata, frames, time, status):
                data = server_callback(indata, frames, time, status)
                outdata[:len(data)] = data
            return new_callback
        if input_flag and not output_flag: # output flag false
            return server_callback
        # input flag false, output flag true
        indata = np.array([])
        def new_callback(outdata, frames, time, status):
            data = server_callback(indata, frames, time, status)
            outdata[:len(data)] = data
        return new_callback



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
