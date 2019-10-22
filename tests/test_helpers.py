from unittest import TestCase
from pya import midicps, cpsmidi, Asig, spectrum
import numpy as np 
import pyaudio


has_input = False
try:
    pyaudio.PyAudio().get_default_input_device_info()
    has_input = True
except OSError:
    pass


class TestHelpers(TestCase):
    """Test helper functions
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

    def test_midi_conversion(self):
        m = 69
        f = 440
        self.assertEqual(f, midicps(m))
        self.assertEqual(m, cpsmidi(f))

    def test_spectrum(self):
        # Not tested expected outcome yet. 
        frq, Y = spectrum(self.asine.sig, self.asine.samples, self.asine.channels, self.asine.sr)
        frqs, Ys = spectrum(self.astereo.sig, self.astereo.samples, self.astereo.channels, self.astereo.sr)