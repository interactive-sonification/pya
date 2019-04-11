from unittest import TestCase
from pya import *
import numpy as np 


class TestPya(TestCase):

    def setUp(self):
        self.sig = np.sin(2*np.pi*100*np.linspace(0,1,44100//2))
        self.asine = Asig(self.sig, sr=44100//2,label="test_sine")
        self.astereo = Asig("../samples/stereoTest.wav", label='stereo', cn = ['l','r'])
    


