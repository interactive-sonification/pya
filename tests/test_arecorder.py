# Test arecorder class.
import time
from pya import Arecorder, Aserver, find_device
from unittest import TestCase, mock
import pytest
import pyaudio

# check if we have an output device
has_input = False
try:
    pyaudio.PyAudio().get_default_input_device_info()
    has_input = True
except OSError:
    pass


FAKE_INPUT = {'index': 0,
              'structVersion': 2,
              'name': 'Mock Input',
              'hostApi': 0,
              'maxInputChannels': 1,
              'maxOutputChannels': 0,
              'defaultLowInputLatency': 0.04852607709750567,
              'defaultLowOutputLatency': 0.01,
              'defaultHighInputLatency': 0.05868480725623583,
              'defaultHighOutputLatency': 0.1,
              'defaultSampleRate': 44100.0}

FAKE_OUTPUT = {'index': 1,
               'structVersion': 2,
               'name': 'Mock Output',
               'hostApi': 0,
               'maxInputChannels': 2,
               'maxOutputChannels': 0,
               'defaultLowInputLatency': 0.01,
               'defaultLowOutputLatency': 0.02,
               'defaultHighInputLatency': 0.03,
               'defaultHighOutputLatency': 0.04,
               'defaultSampleRate': 44100.0}

FAKE_AUDIO_INTERFACE = {'index': 2,
                        'structVersion': 2,
                        'name': 'Mock Audio Interface',
                        'hostApi': 0,
                        'maxInputChannels': 14,
                        'maxOutputChannels': 14,
                        'defaultLowInputLatency': 0.01,
                        'defaultLowOutputLatency': 0.02,
                        'defaultHighInputLatency': 0.03,
                        'defaultHighOutputLatency': 0.04,
                        'defaultSampleRate': 48000.0}


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


class TestArecorderBase(TestCase):
    __test__ = False
    backend = None
    max_inputs = backend.dummy_devices[0]['maxInputChannels'] if backend else 0

    @pytest.mark.xfail(reason="Test may get affected by PortAudio bug or potential unsuitable audio device.")
    def test_boot(self):
        ar = Arecorder(backend=self.backend).boot()
        self.assertTrue(ar.is_active)
        ar.quit()
        self.assertFalse(ar.is_active)

    @pytest.mark.xfail(reason="Test may get affected by PortAudio bug or potential unsuitable audio device.")
    def test_arecorder(self):
        ar = Arecorder(channels=1, backend=None).boot()
        self.assertEqual(ar.sr, 44100)
        ar.record()
        time.sleep(1.)
        ar.pause()
        time.sleep(0.2)
        ar.record()
        time.sleep(1.)
        ar.stop()
        asig = ar.recordings
        self.assertIsInstance(asig, list)
        self.assertEqual(asig[-1].sr, 44100)
        ar.recordings.clear()
        ar.quit()

    @pytest.mark.xfail(reason="Test may get affected by PortAudio bug or potential unsuitable audio device.")
    def test_combined_inout(self):
        # test if two streams can be opened on the same device
        # can only be tested when a device with in- and output capabilities is available
        devices = find_device(min_input=1, min_output=1)
        if devices:
            # set the buffer size low to provoke racing condition
            # observed in https://github.com/interactive-sonification/pya/issues/23
            # the occurrence of this bug depends on the machine load and will only appear when two streams
            # are initialized back-to-back
            bs = 128
            d = devices[0]  # we only need to test one device, we take the first one
            recorder = Arecorder(device=d['index'], bs=bs)
            player = Aserver(device=d['index'], bs=bs)
            player.boot()
            recorder.boot()  # initialized record and boot sequentially to provoke racing condition
            recorder.record()
            time.sleep(1.)
            recorder.stop()
            player.quit()
            self.assertEqual(len(recorder.recordings), 1)  # we should have one Asig recorded
            self.assertGreater(recorder.recordings[0].sig.shape[0], 10 * bs, 
                               "Recording length is too short, < 10buffers")
            recorder.quit()

    @pytest.mark.xfail(reason="Test may get affected by PortAudio bug or potential unsuitable audio device.")
    def test_custom_channels(self):
        s = Arecorder(channels=self.max_inputs, backend=self.backend)
        s.boot()
        self.assertTrue(s.is_active)
        s.quit()
        self.assertFalse(s.is_active)

    @pytest.mark.xfail(reason="Test may get affected by PortAudio bug or potential unsuitable audio device.")
    def test_invalid_channels(self):
        """Raise an exception if booting with channels greater than max channels of the device. Dummy has 10"""
        if self.backend:
            s = Arecorder(channels=self.max_inputs + 1, backend=self.backend)
            with self.assertRaises(OSError):
                s.boot()
        else:
            s = Arecorder(channels=-1, backend=self.backend)
            with self.assertRaises(ValueError):
                s.boot()

    @pytest.mark.xfail(reason="Some devices may not have inputs")
    def test_default_channels(self):
        if self.backend:
            s = Arecorder(backend=self.backend)
            self.assertEqual(s.channels, self.backend.dummy_devices[0]['maxInputChannels'])
        else:
            s = Arecorder()
            self.assertGreater(s.channels, 0, "No input channel found")


class TestArecorder(TestArecorderBase):
    __test__ = True


class TestMockArecorder(TestCase):

    def test_mock_arecorder(self):
        mock_recorder = MockRecorder()
        with mock.patch('pyaudio.PyAudio', return_value=mock_recorder):
            ar = Arecorder()
            self.assertEqual(
                "Mock Input",
                ar.backend.get_default_input_device_info()['name'])
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
