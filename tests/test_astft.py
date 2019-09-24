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

    def test_input_as_asig(self):
        astft = self.asig.to_stft()
        self.assertEqual(astft.sr, 44100)
        astft = self.asig.to_stft(sr=2000)
        self.assertEqual(astft.sr, 2000)
        signal = self.asig2.sig        

    def test_multichannel_asig(self):
        # Test conversion of a multi channel asig's astft. 
        asine = Ugen().sawtooth(channels=3)
        astft = asine.to_stft()
        
    # def test_input_as_stft(self):
    #     x = np.random.rand(100, 3) 