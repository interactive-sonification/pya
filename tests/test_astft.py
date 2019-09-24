from unittest import TestCase
from pya import Ugen, Astft
import numpy as np
# import logging
# logging.basicConfig(level=logging.DEBUG)


class TestAstft(TestCase):

    def setUp(self):
        self.asig = Ugen().sine()
        self.asig2 = Ugen().sine(channels=2)

    def tearDown(self):
        pass

    def test_constructor(self):
        astft = self.asig.to_stft()
        self.assertEqual(astft.sr, 44100)
        astft = self.asig.to_stft(sr=2000)
        self.assertEqual(astft.sr, 2000)
        signal = self.asig2.sig        
        # astft = Astft(signal, sr=44100, window='hamming')
        # self.assertEqual(astft.sr, 44100)
        # self.assertEqual(astft.window, 'hamming')