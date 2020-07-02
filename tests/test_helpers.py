from unittest import TestCase
from pya import Asig, Ugen
from pya.helper import spectrum, padding, next_pow2, is_pow2, midicps, cpsmidi
from pya.helper import signal_to_frame, magspec, powspec, linlin
from pya.helper import hz2mel, mel2hz
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
        self.asineWithName = Asig(self.sig, sr=44100,
                                  label="test_sine", cn=['sine'])
        self.sig2ch = np.repeat(self.sig, 2).reshape((44100, 2))
        self.astereo = Asig(self.sig2ch, sr=44100, label="sterep",
                            cn=['l', 'r'])
        self.sig16ch = np.repeat(self.sig, 16).reshape((44100, 16))
        self.asine16ch = Asig(self.sig16ch, sr=44100,
                              label="test_sine_16ch")

    def tearDown(self):
        pass

    def test_linlin(self):
        self.assertTrue(np.array_equal(linlin(np.arange(0, 10),
                                              smi=0, sma=10, dmi=0., dma=1.0),
                                       np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5,
                                                 0.6, 0.7, 0.8, 0.9])))

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

    def test_next_pow2(self):
        next = next_pow2(255)
        self.assertEqual(next, 256)

        next = next_pow2(0)
        self.assertEqual(next, 2)

        next = next_pow2(256)
        self.assertEqual(next, 256)

        with self.assertRaises(AttributeError):
            _ = next_pow2(-2)

    def test_is_pow2(self):
        self.assertTrue(is_pow2(2))
        self.assertTrue(is_pow2(256))
        self.assertTrue(is_pow2(1))
        self.assertFalse(is_pow2(145))
        self.assertFalse(is_pow2(-128))
        self.assertFalse(is_pow2(0))

    def test_signal_to_frame(self):
        sq = Ugen().square(freq=20, sr=8000, channels=1)
        frames = signal_to_frame(sq.sig, 400, 400)
        self.assertEqual(frames.shape, (20, 400))
        frames = signal_to_frame(sq.sig, 400, 200)
        self.assertEqual(frames.shape, (39, 400))

    def test_magspec_pspec(self):
        # Magnitude spectrum
        sq = Ugen().square(freq=20, sr=8000, channels=1)
        frames = signal_to_frame(sq.sig, 400, 400)
        mag = magspec(frames, 512)
        self.assertEqual(mag.shape, (20, 257))
        self.assertTrue((mag >= 0.).all())  # All elements should be non-negative
        ps = powspec(frames, 512)
        self.assertEqual(ps.shape, (20, 257))
        self.assertTrue((ps >= 0.).all())  # All elements should be non-negative

    def test_melhzconversion(self):
        self.assertAlmostEqual(hz2mel(440), 549.64, 2)
        self.assertAlmostEqual(mel2hz(549.64), 440, 2)
