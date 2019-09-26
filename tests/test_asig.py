from unittest import TestCase
from pya import *
import numpy as np 


class TestAsig(TestCase):
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

    def test_asig_constructor(self):
        # Integer constructor
        asig = Asig(1000)
        asig = Asig(1000, channels=3)
        self.assertEqual(asig.samples, 1000)
        self.assertEqual(asig.channels, 3)
        print(self.asine)

    def test_asig_plot(self):
        self.asine.plot()
        self.astereo.plot(offset=1., scale=0.5)

    def test_duration(self):
        result = self.asine.get_duration()
        self.assertEqual(result, 1.)

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

    def test_norm(self):
        result = self.astereo.norm()
        result = self.astereo.norm(norm=1., dcflag=True)
        self.assertEqual(1, np.max(result.sig))
        result = self.astereo.norm(norm=2, dcflag=True)
        self.assertEqual(2, np.max(result.sig))
        result = self.astereo.norm(norm=-3, dcflag=True)
        self.assertEqual(3, np.max(result.sig))
        result = self.astereo.norm(norm=-4, in_db=True, dcflag=True)

    def test_gain(self):
        result = self.astereo.gain(amp=0.)
        self.assertEqual(0, np.max(result.sig))
        result = self.astereo.gain(amp=2.)
        self.assertEqual(2, np.max(result.sig))
        result = self.astereo.gain(db=3.)
        with self.assertRaises(AttributeError):
            _ = self.astereo.gain(amp=1, db=3.)

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
        b2 = np.arange(4) + 10

        adding = a + b1  # asig + asig
        self.assertIsInstance(adding, Asig)
        self.assertTrue(np.array_equal([10, 12, 14, 16], adding.sig))

        # asig + ndarray  actually we don't encourage that. Because of sampling rate may differ
        adding = a + b0
        self.assertIsInstance(adding, Asig)
        self.assertTrue(np.array_equal([10, 12, 14, 16], adding.sig))

        # adding with different size will extend to the new size. 
        b3 = Asig(np.arange(6), sr=2)
        adding = a + b3
        self.assertEqual(b3.samples, 6)

        # TODO  to decide what to do with different channels. Currently not allow. 
        # so that it wont get too messy. 
        b4 = Asig(np.ones((4, 2)), sr=2)
        with self.assertRaises(ValueError):  # If list is not string only, TypeError
            adding = a + b4

        adding = b4 + b4
        self.assertTrue(np.array_equal(np.ones((4, 2)) + np.ones((4, 2)), adding.sig))