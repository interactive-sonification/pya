from unittest import TestCase
from pya import *
import numpy as np
import logging
logging.basicConfig(level = logging.DEBUG)


class TestSlicing(TestCase):

    def setUp(self):
        self.sig = np.sin(2*np.pi* 100 * np.linspace(0,1,44100))
        self.asine = Asig(self.sig, sr=44100,label="test_sine")
        self.asineWithName = Asig(self.sig, sr=44100,label="test_sine", cn = 'sine')
        self.astereo = Asig("../examples/samples/stereoTest.wav", label='stereo', cn = ['l','r'])
        self.asentence = Asig("../examples/samples/sentence.wav", label='sentence', cn = 'sen')

    def tearDown(self):
        pass

    def test_route(self):
        # Shift a mono signal to chan 4 should result in a 4 channels signals
        result = self.asine.route(3)
        self.assertEqual(4, result.channels)

        result = self.asineWithName.route(3)
        self.assertEqual(4, result.channels)

        