from unittest import TestCase
from pya import *
import numpy as np 


class TestPya(TestCase):
    """Test the following:
        duration, fader, samples, channels, channel names,
        TODO, add the rest.
    """

    def setUp(self):
        self.sig = np.sin(2 * np.pi * 100 * np.linspace(0, 1, 44100))
        self.asine = Asig(self.sig, sr=44100, label="test_sine")
        self.asineWithName = Asig(self.sig, sr=44100, label="test_sine", cn=['sine'])
        self.sig2ch = np.repeat(self.sig, 2).reshape((44100, 2))
        self.astereo = Asig(self.sig2ch, sr=44100, label="sterep", cn=['l', 'r'])
        self.sig16ch = np.repeat(self.sig, 16).reshape((44100, 16))
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

    def test_samples(self):
        as1 = Asig(np.ones((100, 4)), sr=100)

        self.assertEqual(100, as1.samples)

    def test_channels(self):
        as1 = Asig(np.ones((100, 4)), sr=100)
        self.assertEqual(4, as1.channels)

    def test_cn(self):

        self.assertEqual(self.astereo.cn, ['l', 'r'])
        self.astereo.cn = ['left', 'right']  # Test changing the cn
        self.assertEqual(self.astereo.cn, ['left', 'right'])
        with self.assertRaises(ValueError):
            self.astereo.cn = ['left', 'right', 'middle']

        with self.assertRaises(TypeError):  # If list is not string only, TypeError
            self.astereo.cn = ["b", 10]

        self.assertEqual(self.astereo.cn, ['left', 'right'])
