from unittest import TestCase
from pya import *
import numpy as np
import logging
logging.basicConfig(level = logging.DEBUG)


class TestSlicing(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sine(self):
        sine = Ugen().sine(freq=200, amp=0.5, dur=1.0, sr=44100//2, channels=2)
        self.assertEqual(44100//2, sine.sr)
        self.assertAlmostEqual(0.5, np.max(sine.sig), places=6)
        self.assertEqual((44100//2, 2), sine.sig.shape)

