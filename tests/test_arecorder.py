# Test arecorder class.
import time
from pya import Arecorder
from unittest import TestCase, skipUnless, mock
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


class MockRecorder(mock.MagicMock):
    def get_device_count(self):
        return 2

    def get_device_info_by_index(self, arg):
        return FAKE_INPUT if arg == 0 else FAKE_OUTPUT

    def get_default_input_device_info(self):
        return FAKE_INPUT

    def get_default_output_device_info(self):
        return FAKE_OUTPUT


class TestArecorder(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @skipUnless(has_input, "PyAudio found no input device.")
    def test_arecorder(self):
        ar = Arecorder(channels=1).boot()
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

    def test_mock_arecorder(self):
        mock_recorder = MockRecorder()
        with mock.patch('pyaudio.PyAudio', return_value=mock_recorder):
            ar = Arecorder()
            self.assertEqual(
                "Mock Input", 
                ar.pa.get_default_input_device_info()['name'])
            ar.boot()
            self.assertTrue(mock_recorder.open.called)
            ar.record()
            # time.sleep(0.5)
            ar.pause()
            ar.record()
            ar.recordings.clear()
            self.assertEqual(0, len(ar.recordings))
            # ar.stop()  # Dont know how to mock the stop. 
            # TODO How to mock a result. 