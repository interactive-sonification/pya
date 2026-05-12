# Test arecorder class.
import importlib.util
from unittest import TestCase, mock

import numpy as np
import pytest

from pya import Arecorder

has_pyaudio = importlib.util.find_spec("pyaudio") is not None


FAKE_INPUT = {
    "index": 0,
    "structVersion": 2,
    "name": "Mock Input",
    "hostApi": 0,
    "maxInputChannels": 1,
    "maxOutputChannels": 0,
    "defaultLowInputLatency": 0.04852607709750567,
    "defaultLowOutputLatency": 0.01,
    "defaultHighInputLatency": 0.05868480725623583,
    "defaultHighOutputLatency": 0.1,
    "defaultSampleRate": 44100.0,
}

FAKE_OUTPUT = {
    "index": 1,
    "structVersion": 2,
    "name": "Mock Output",
    "hostApi": 0,
    "maxInputChannels": 2,
    "maxOutputChannels": 0,
    "defaultLowInputLatency": 0.01,
    "defaultLowOutputLatency": 0.02,
    "defaultHighInputLatency": 0.03,
    "defaultHighOutputLatency": 0.04,
    "defaultSampleRate": 44100.0,
}

FAKE_AUDIO_INTERFACE = {
    "index": 2,
    "structVersion": 2,
    "name": "Mock Audio Interface",
    "hostApi": 0,
    "maxInputChannels": 14,
    "maxOutputChannels": 14,
    "defaultLowInputLatency": 0.01,
    "defaultLowOutputLatency": 0.02,
    "defaultHighInputLatency": 0.03,
    "defaultHighOutputLatency": 0.04,
    "defaultSampleRate": 48000.0,
}


class MockRecorder(mock.MagicMock):
    def get_device_count(self):
        return 2

    def get_device_info_by_index(self, arg):
        if arg == 0:
            return FAKE_INPUT
        elif arg == 1:
            return FAKE_OUTPUT
        elif arg == 2:
            return FAKE_AUDIO_INTERFACE
        else:
            raise AttributeError("Invalid device index.")

    def get_default_input_device_info(self):
        return FAKE_INPUT

    def get_default_output_device_info(self):
        return FAKE_OUTPUT

    # def open(self, *args, **kwargs):


class MockBackend:
    """Mock audio backend for testing"""

    def __init__(self, **kwargs):
        self.dummy_devices = [
            {
                "index": 0,
                "maxInputChannels": 2,
                "maxOutputChannels": 2,
                "defaultSampleRate": 44100,
            }
        ]
        self.dtype = "float32"
        self.range = 1.0
        self.bs = 256

    def get_device_count(self):
        return len(self.dummy_devices)

    def get_device_info_by_index(self, idx):
        return self.dummy_devices[idx]

    def get_default_input_device_info(self):
        return self.dummy_devices[0]

    def get_default_output_device_info(self):
        return self.dummy_devices[0]

    def open(self, **kwargs):
        return MockStream()

    def terminate(self):
        pass


class MockStream:
    """Mock audio stream for testing"""

    def __init__(self):
        self._active = True

    def is_active(self):
        return self._active

    def stop_stream(self):
        self._active = False

    def close(self):
        self._active = False


class TestArecorderBase(TestCase):
    def setUp(self):
        self.backend = MockBackend()

    def test_boot(self):
        ar = Arecorder(backend=self.backend).boot()
        self.assertTrue(ar.is_active)
        ar.quit()
        self.assertFalse(ar.is_active)

    def test_record(self):
        ar = Arecorder(backend=self.backend).boot()
        ar.record()
        # Simulate some data
        ar.record_buffer = [np.zeros((256, 2))]  # Mock some audio data
        ar.stop()
        self.assertEqual(len(ar.recordings), 1)
        ar.quit()

    def test_channels(self):
        ar = Arecorder(channels=2, backend=self.backend)
        self.assertEqual(ar.channels, 2)


class TestArecorder(TestArecorderBase):
    __test__ = True


class TestMockArecorder(TestCase):
    @pytest.mark.skipif(not has_pyaudio, reason="requires pyaudio to be installed")
    def test_mock_arecorder(self):
        mock_recorder = MockRecorder()
        with mock.patch("pyaudio.PyAudio", return_value=mock_recorder):
            ar = Arecorder()
            self.assertEqual(
                "Mock Input", ar.backend.get_default_input_device_info()["name"]
            )
            ar.boot()
            self.assertTrue(mock_recorder.open.called)
            ar.record()
            # time.sleep(2)
            ar.pause()
            ar.record()
            ar.recordings.clear()
            self.assertEqual(0, len(ar.recordings))
            # ar.stop()  # Dont know how to mock the stop.
            # TODO How to mock a result.

        # Mock multiple input devices.
        ar.set_device(2, reboot=True)  # Set to multiple device
        self.assertEqual(ar.max_in_chn, 14)
