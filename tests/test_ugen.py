from unittest import TestCase
from pya import *
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)


class TestUgen(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sine(self):
        sine = Ugen().sine(freq=200, amp=0.5, dur=1.0, sr=44100 // 2, channels=2)
        self.assertEqual(44100 // 2, sine.sr)
        self.assertAlmostEqual(0.5, np.max(sine.sig), places=6)
        self.assertEqual((44100 // 2, 2), sine.sig.shape)

    def test_cos(self):
        cos = Ugen().cos(freq=200, amp=0.5, dur=1.0, sr=44100 // 2, channels=2)
        self.assertEqual(44100 // 2, cos.sr)
        self.assertAlmostEqual(0.5, np.max(cos.sig), places=6)
        self.assertEqual((44100 // 2, 2), cos.sig.shape)

    def test_square(self):
        square = Ugen().square(freq=200, amp=0.5, dur=1.0, sr=44100 // 2, channels=2)
        self.assertEqual(44100 // 2, square.sr)
        self.assertAlmostEqual(0.5, np.max(square.sig), places=6)
        self.assertEqual((44100 // 2, 2), square.sig.shape)

    def test_sawooth(self):
        saw = Ugen().sawtooth(freq=200, amp=0.5, dur=1.0, sr=44100 // 2, channels=2)
        self.assertEqual(44100 // 2, saw.sr)
        self.assertAlmostEqual(0.5, np.max(saw.sig), places=6)
        self.assertEqual((44100 // 2, 2), saw.sig.shape)

    def test_noise(self):
        white = Ugen().noise(type="white")
        pink = Ugen().noise(type="pink")
