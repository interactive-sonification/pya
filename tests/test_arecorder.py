# Test arecorder class. 
import time
from pya import Arecorder
from unittest import TestCase, skipUnless
import pyaudio

# check if we have an output device
has_input = False
try:
    pyaudio.PyAudio().get_default_input_device_info()['index']
    has_input = True
except OSError:
    pass

class TestArecorder(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @skipUnless(has_input, "PyAudio found no output device.")
    def test_arecorder(self):
        ar = Arecorder(channels=1).boot()
        self.assertEqual(ar.sr, 44100)
        ar.record()
        time.sleep(1.)
        ar.pause()
        time.sleep(0.2)
        ar.resume()
        time.sleep(1.)
        ar.stop()
        asig = ar.recordings
        self.assertIsInstance(asig, list)
        a1_last = ar.get_latest_recording()
        self.assertEqual(a1_last.sr, 44100)
        ar.reset_recordings()
        ar.quit()