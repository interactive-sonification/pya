from unittest import TestCase
from pya import *
import numpy as np
import logging
logging.basicConfig(level = logging.DEBUG)


class TestPlay(TestCase):

    def setUp(self):
        self.sig = np.sin(2*np.pi* 100 * np.linspace(0,1,44100))
        self.asine = Asig(self.sig, sr=44100,label="test_sine")
        self.asineWithName = Asig(self.sig, sr=44100,label="test_sine", cn = ['sine'])
        # self.astereo = Asig("../examples/samples/stereoTest.wav", label='stereo', cn = ['l','r'])
        # self.asentence = Asig("../examples/samples/sentence.wav", label='sentence', cn = 'sen')
        self.sig2ch = np.repeat(self.sig, 2).reshape(((44100, 2)))
        self.astereo = Asig(self.sig2ch, sr=44100, label="stereo", cn=['l', 'r'])

    def tearDown(self):
        pass

    def test_play(self):
        # Shift a mono signal to chan 4 should result in a 4 channels signals
        s = Aserver()
        s.boot()
        self.asine.play(server = s)

    def test_gain(self):
        result = (self.asine * 0.2).sig
        expected = self.asine.sig * 0.2
        self.assertTrue(np.array_equal(result, expected))

        expected = self.sig * self.sig
        result = (self.asine * self.asine).sig
        self.assertTrue(np.array_equal(result, expected))

    def test_resample(self):
        # This test currently only check if there is error running the code, but not whether resampling is correct
        result = self.asine.resample(target_sr=44100//2, rate=1, kind='linear')
        self.assertIsInstance(result, Asig)
