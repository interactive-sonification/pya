from unittest import TestCase, mock
from unittest.mock import patch
from pya import Asig
import numpy as np
from math import inf
import os


class TestAsig(TestCase):
    """Unit Tests for Asig"""
    def setUp(self):
        self.sig = np.sin(2 * np.pi * 100 * np.linspace(0, 1, 44100))
        self.asine = Asig(self.sig, sr=44100, label="test_sine")
        self.asineWithName = Asig(self.sig, sr=44100, label="test_sine", cn=['sine'])
        self.sig2ch = np.repeat(self.sig, 2).reshape((44100, 2))
        self.astereo = Asig(self.sig2ch, sr=44100, label="sterep", cn=['l', 'r'])
        self.sig16ch = np.repeat(self.sig, 16).reshape((44100, 16))
        self.asine16ch = Asig(self.sig16ch, sr=44100, label="test_sine_16ch")
        self.asigconst = Asig(1.0, sr=100, label="constant signal", cn=['0']) + 0.5

    def tearDown(self):
        pass

    def test_asig_constructor(self):
        # Integer constructor
        asig = Asig(1000)
        asig = Asig(1000, channels=3)
        self.assertEqual(asig.samples, 1000)
        self.assertEqual(asig.channels, 3)

    def test_asig_plot(self):
        self.asine.plot()
        self.astereo.plot(offset=1., scale=0.5)

    def test_duration(self):
        self.assertEqual(self.asine.get_duration(), 1.)
        get_time = self.asine.get_times()
        self.assertTrue(np.array_equal(np.linspace(0,
                                                   (self.asine.samples - 1) / self.asine.sr,
                                                   self.asine.samples), self.asine.get_times()))

    def test_dur_property(self):
        self.assertEqual(self.asine.dur, 1.)

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

        with self.assertRaises(TypeError):  # If list is not string only, TypeError
            asig = Asig(1000, channels=3, cn=3)

        self.assertEqual(self.astereo.cn, ['left', 'right'])

    def test_remove_DC(self):
        result = self.asigconst.remove_DC()
        self.assertEqual(0, np.max(result.sig))
        result = Asig(100, channels=2) + 0.25
        result[:, 1] = 0.5
        self.assertEqual([0.25, 0.5], list(np.max(result.sig, 0)))
        self.assertEqual([0., 0.], list(result.remove_DC().sig.max(axis=0)))

    def test_norm(self):
        result = self.astereo.norm()
        result = self.astereo.norm(norm=1., dcflag=True)
        self.assertEqual(1, np.max(result.sig))
        result = self.astereo.norm(norm=2, dcflag=True)
        self.assertEqual(2, np.max(result.sig))
        result = self.astereo.norm(norm=-3, dcflag=True)
        self.assertEqual(3, np.max(result.sig))
        result = self.astereo.norm(norm=-4, in_db=True, dcflag=True)
        self.assertAlmostEqual(0.6309, np.max(result.sig), places=3)

    def test_gain(self):
        result = self.astereo.gain(amp=0.)
        self.assertEqual(0, np.max(result.sig))
        result = self.astereo.gain(amp=2.)
        self.assertEqual(2, np.max(result.sig))
        result = self.astereo.gain(db=3.)
        with self.assertRaises(AttributeError):
            _ = self.astereo.gain(amp=1, db=3.)
        result = self.astereo.gain()  # by default amp=1. nothing change.

    def test_rms(self):
        result = self.asine16ch.rms()

    def test_plot(self):
        self.asine.plot(xlim=(0, 1), ylim=(-1, 1))
        self.asine.plot(fn='db')
        self.astereo.plot(offset=1)
        self.asine16ch.plot(offset=1, scale=0.5)

    def test_add(self):
        # # test + - * / ==
        a = Asig(np.arange(4), sr=2, label="", channels=1)
        b0 = np.arange(4) + 10
        b1 = Asig(b0, sr=2, label="", channels=1)
        adding = a + b1  # asig + asig
        self.assertIsInstance(adding, Asig)
        self.assertTrue(np.array_equal([10, 12, 14, 16], adding.sig))

        # asig + ndarray  actually we don't encourage that. Because of sampling rate may differ
        # also because ndarray + asig works. so it is strongly against adding asig with ndarray.
        # just maker another asig and add both together.
        adding = a + b0
        self.assertIsInstance(adding, Asig)
        self.assertTrue(np.array_equal([10, 12, 14, 16], adding.sig))

        # adding with different size will extend to the new size.
        b2 = Asig(np.arange(6), sr=2)
        with self.assertRaises(ValueError):
            adding = a + b2
        adding = a.x + b2
        self.assertEqual(b2.samples, 6)

        # TODO  to decide what to do with different channels. Currently not allow.
        # so that it wont get too messy.
        b3 = Asig(np.ones((4, 2)), sr=2)
        # adding different channels asigs are not allowed.
        with self.assertRaises(ValueError):
            adding = a + b3

        # Both multichannels are ok.
        adding = b3 + b3
        self.assertTrue(np.array_equal(np.ones((4, 2)) + np.ones((4, 2)), adding.sig))

        # Test bound mode. In the audio is not extended. but bound to the lower one.
        adding = a.bound + b2
        self.assertEqual(adding.samples, 4)

        adding = b2.bound + a
        self.assertEqual(adding.samples, 4)

    def test_mul(self):
        # Testing multiplication beween asig and asig, or asig with a scalar.
        a = Asig(np.arange(4), sr=2)
        a2 = Asig(np.arange(8), sr=2)
        a4ch = Asig(np.ones((4, 4)), sr=2)
        a4ch2 = Asig(np.ones((8, 4)), sr=2)

        self.assertTrue(np.array_equal([0, 4, 8, 12], (a * 4).sig))
        self.assertTrue(np.array_equal([0, 4, 8, 12], (4 * a).sig))
        self.assertTrue(np.array_equal([0, 1, 4, 9], (a * a).sig))
        self.assertTrue(np.array_equal([0, 1, 4, 9], (a.bound * a2).sig))
        self.assertTrue(np.array_equal([0., 1., 4., 9., 4., 5., 6., 7.], (a.x * a2).sig))

    def test_subtract(self):
        a = Asig(np.arange(4), sr=2)
        b = Asig(np.ones(4), sr=2)
        self.assertTrue(np.array_equal([-1, 0, 1, 2], (a - 1).sig))
        self.assertTrue(np.array_equal([1, 0, -1, -2], (1 - a).sig))
        self.assertTrue(np.array_equal([-1, 0, 1, 2], (a - b).sig))
        self.assertTrue(np.array_equal([1, 0, -1, -2], (b - a).sig))
        a = Asig(np.arange(4), sr=2)
        b = Asig(np.ones(6), sr=2)
        self.assertTrue(np.array_equal([-1, 0, 1, 2], (a.bound - b).sig))
        with self.assertRaises(ValueError):
            adding = a - b
        self.assertTrue(np.array_equal([-1, 0, 1, 2, -1, -1], (a.x - b).sig))

    def test_division(self):
        # Testing multiplication beween asig and asig, or asig with a scalar.
        #
        a = Asig(np.arange(4), sr=2)
        a2 = Asig(np.arange(8), sr=2)
        a4ch = Asig(np.ones((4, 4)), sr=2)
        a4ch2 = Asig(np.ones((8, 4)), sr=2)

        self.assertTrue(np.array_equal([0, 0.25, 0.5, 0.75], (a / 4).sig))
        self.assertTrue(np.allclose([inf, 4, 2, 1.33333333], (4 / a).sig))
        self.assertTrue(np.array_equal(np.ones((4, 4)) / 2, (a4ch / 2).sig))

    def test_windowing(self):
        asig = Asig(np.ones(10), sr=2)
        asig_windowed = asig.window_op(nperseg=2, stride=1,
                                       win='hann', fn='rms', pad='mirror')
        self.assertTrue(np.allclose([1., 0.70710677, 0.70710677, 0.70710677,
                                    0.70710677, 0.70710677, 0.70710677,
                                    0.70710677, 0.70710677, 1.],
                                    asig_windowed.sig))

        asig2ch = Asig(np.ones((10, 2)), sr=2)
        asig2ch.window_op(nperseg=2, stride=1, win='hann', fn='rms', pad='mirror')
        a = [1., 0.70710677, 0.70710677, 0.70710677,
             0.70710677, 0.70710677, 0.70710677,
             0.70710677, 0.70710677, 1.]
        res = np.array([a, a]).T
        self.assertTrue(np.allclose(a, asig_windowed.sig))

    def test_convolve(self):
        # Do self autocorrelatin, the middle point should always have a corr val near 1.0
        test = Asig(np.sin(np.arange(0, 21)), sr=21)
        result = test.convolve(test.sig[::-1], mode='same')
        # The middle point should have high corr
        self.assertTrue(result.sig[10] > 0.99, msg="middle point of a self correlation should always has high corr val.")
        # Test different modes
        self.assertEqual(result.samples, test.samples, msg="'same' mode should result in the same size")
        result = test.convolve(test.sig[::-1], mode='full')

        self.assertEqual(result.samples, test.samples * 2 - 1, msg="full mode should have 2x - 1 samples.")

        # Test input type
        ir = Asig(test.sig[::-1], sr=21)
        result = test.convolve(ir, mode='same')
        self.assertTrue(result.sig[10] > 0.99, msg="middle point of a self correlation should always has high corr val.")

        with self.assertRaises(TypeError, msg="ins can only be array or Asig"):
            result = test.convolve("string input")

        # test signal is multichannel
        # TODO define what to check.
        test = Asig(np.ones((20, 10)), sr=20)
        result = test.convolve(test)

    # # At the top I import Asig by: from pya import Asig
    @mock.patch("pya.asig.wavfile")
    def test_save_wavefile(self, mock_wavfile):

        test = Asig(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]), sr=6)
        test.save_wavfile(fname="mock save")
        mock_wavfile.write.assert_called_once()

        # int 16
        test = Asig(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]), sr=6)
        test.save_wavfile(dtype="int16")

        # unit8
        test = Asig(np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]), sr=6)
        test.save_wavfile(dtype="uint8")
