from pya import Asig, Ugen, Amfcc
from unittest import TestCase
import numpy as np

class TestAmfcc(TestCase):
    def setUp(self):
        self.test_asig = Ugen().square(freq=20, sr=8000)
        self.test_sig = self.test_asig.sig

    def tearDown(self):
        pass

    def test_construct(self):
        # If x is asig, it will ignore sr but use x.sr instead.
        amfcc = Amfcc(self.test_asig, sr=45687)
        self.assertEqual(amfcc.sr, 8000)

        # if x is ndarray and sr is not given
        with self.assertRaises(AttributeError):
            _ = Amfcc(self.test_sig)