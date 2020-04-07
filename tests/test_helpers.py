from unittest import TestCase
from pya import midicps, cpsmidi, Asig, spectrum, padding, shift_bit_length
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

    def test_padding(self):
        """Pad silence to signal. Support 1-3D tensors."""
        tensor1 = np.arange(5)
        padded = padding(tensor1, 2, tail=True)
        self.assertTrue(np.array_equal(padded, np.array([0, 1, 2, 3, 4, 0, 0])))
        padded = padding(tensor1, 2, tail=False)
        self.assertTrue(np.array_equal(padded, np.array([0, 0, 0, 1, 2, 3, 4])))

        tensor2 = np.ones((3, 3))
        padded = padding(tensor2, 2, tail=True)
        self.assertTrue(np.array_equal(padded, np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], 
                                                         [0., 0., 0.], [0., 0., 0.]])))
        padded = padding(tensor2, 2, tail=False, constant_values=5)
        self.assertTrue(np.array_equal(padded, np.array([[5., 5., 5.], [5., 5., 5.], 
                                                         [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])))

        tensor3 = np.ones((2, 2, 2))
        padded = padding(tensor3, 2)
        self.assertTrue(np.array_equal(padded, np.array([[[1., 1.], 
                                                          [1., 1.], 
                                                          [0., 0.], 
                                                          [0., 0.]], 
                                                        [[1., 1.], 
                                                         [1., 1.], 
                                                         [0., 0.], 
                                                         [0., 0.]]])))
        padded = padding(tensor3, 2, tail=False)
        self.assertTrue(np.array_equal(padded, np.array([[[0., 0.], 
                                                          [0., 0.],
                                                          [1., 1.], 
                                                          [1., 1.]
                                                          ], 
                                                        [[0., 0.], 
                                                         [0., 0.],
                                                         [1., 1.], 
                                                         [1., 1.]]])))

    def test_shift_bit_length(self):
        next = shift_bit_length(255)
        self.assertEqual(next, 256)

        next = shift_bit_length(0)
        self.assertEqual(next, 2)

        next = shift_bit_length(256)
        self.assertEqual(next, 256)

        with self.assertRaises(AttributeError):
            next = shift_bit_length(-2)

