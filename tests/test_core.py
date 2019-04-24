from unittest import TestCase
from pya import *
import numpy as np 


# Basic testing.

class TestPya(TestCase):

    def setUp(self):
        self.sig = np.sin(2*np.pi* 100 * np.linspace(0,1,44100))
        self.asine = Asig(self.sig, sr=44100,label="test_sine")
        self.asineWithName = Asig(self.sig, sr=44100,label="test_sine", cn = 'sine')
        self.sig2ch = np.repeat(self.sig, 2).reshape(((44100, 2)))
        self.astereo = Asig(self.sig2ch, sr=44100, label="sterep", cn=['l', 'r'])
        # self.astereo = Asig("/examples/samples/stereoTest.wav", label='stereo', cn=['l','r'])
        self.sig16ch = np.repeat(self.sig, 16).reshape(((44100, 16)))
        self.asine16ch = Asig(self.sig16ch, sr=44100, label="test_sine_16ch")

    def tearDown(self):
        pass

    def test_duration(self):
        result = self.asine.get_duration()
        expected = 1.
        self.assertEqual(result, expected)

    def test_fader(self):
        result = self.asine.fade_in(dur=0.2)
        self.assertIsInstance(result, Asig)

        result = self.asine.fade_out(dur=0.2)
        self.assertIsInstance(result, Asig)