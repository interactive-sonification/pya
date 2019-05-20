from unittest import TestCase
from pya import Ugen
import numpy as np
import logging
logging.basicConfig(level = logging.DEBUG)


class TestSlicing(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sine(self):
        pas
        # sine = ugen.sine(freq=440, amp=1.0, dur=1.0, sr=44100, channels=1, cn=["sine"], label="sine")
        # self.assertEqual(44100, sine.samples)
        #
        # square = ugen.square(freq=440, amp=1.0, dur=1.0, duty=0.4, sr=44100, channels=1, cn=None, label="square")
        # self.assertEqual(44100, square.samples)
        #
        # saw = ugen.sawtooth(freq=440, amp=1.0, dur=1.0, width=1., sr=100, channels=1, cn=None, label="sawtooth")
        # self.assertEqual(100, saw.samples)