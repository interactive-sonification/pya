from unittest import TestCase
from pya import *
import numpy as np


class TestUgen(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sine(self):
        sine = Ugen().sine(
            freq=44100 / 16, amp=0.5, dur=0.001, sr=44100 // 2, channels=2
        )
        self.assertEqual(44100 // 2, sine.sr)
        self.assertEqual(0.5, np.max(sine.sig))
        self.assertEqual((22, 2), sine.sig.shape)
        sine = Ugen().sine(
            freq=44100 / 16, amp=0.5, n_rows=400, sr=44100 // 2, channels=2
        )
        self.assertEqual(44100 // 2, sine.sr)
        self.assertEqual(0.5, np.max(sine.sig))
        self.assertEqual((400, 2), sine.sig.shape)

    def test_cos(self):
        cos = Ugen().cos(freq=44100 / 16, amp=0.5, dur=0.001, sr=44100 // 2, channels=2)
        self.assertEqual(44100 // 2, cos.sr)
        self.assertEqual(0.5, np.max(cos.sig))
        self.assertEqual((22, 2), cos.sig.shape)
        cos = Ugen().cos(freq=44100 / 16, amp=0.5, n_rows=44, sr=44100 // 2, channels=2)
        self.assertEqual(44100 // 2, cos.sr)
        self.assertEqual(0.5, np.max(cos.sig))
        self.assertEqual((44, 2), cos.sig.shape)

    def test_square(self):
        square = Ugen().square(freq=200, amp=0.5, dur=1.0, sr=44100 // 2, channels=2)
        self.assertEqual(44100 // 2, square.sr)
        self.assertAlmostEqual(0.5, np.max(square.sig), places=5)
        self.assertEqual((44100 // 2, 2), square.sig.shape)

    def test_sawooth(self):
        saw = Ugen().sawtooth(freq=200, amp=0.5, dur=1, sr=44100 // 2, channels=2)
        self.assertEqual(44100 // 2, saw.sr)
        self.assertAlmostEqual(0.5, np.max(saw.sig), places=5)
        self.assertEqual((44100 // 2, 2), saw.sig.shape)

    def test_noise(self):
        white = Ugen().noise(
            type="white", amp=0.2, dur=1.0, sr=1000, cn=["white"], label="white_noise"
        )
        pink = Ugen().noise(type="pink")
        self.assertEqual(white.sr, 1000)
        self.assertEqual(white.cn, ["white"])
        self.assertEqual(white.label, "white_noise")
        white_2ch = Ugen().noise(type="pink", channels=2)
        self.assertEqual(white_2ch.channels, 2)

    def test_dur_n_rows_exception(self):
        # An exception should be raised if both dur and n_rows are define.
        with self.assertRaises(AttributeError):
            asig = Ugen().sine(dur=1.0, n_rows=400)
