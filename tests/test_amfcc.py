# from pya import Amfcc
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

    def test_preemphasis(self):
        self.assertTrue(np.array_equal(np.array([0., 1., 1.5, 2., 2.5]),
                        Amfcc.preemphasis(np.arange(5), coeff=0.5)))

    def test_melfb(self):
        fb = Amfcc.mel_filterbanks(8000)  # Using default
        self.assertEqual(fb.shape, (26, 257))  # nfilters, NFFT // 2 + 1

    def test_lifter(self):
        fake_cepstra = np.ones((20, 13))
        lifted_ceps = Amfcc.lifter(fake_cepstra, L=22)
        self.assertEqual(fake_cepstra.shape, lifted_ceps.shape)
        # if L negative, no lifting applied
        lifted_ceps = Amfcc.lifter(fake_cepstra, L=-3)  
        self.assertTrue(np.array_equal(lifted_ceps, fake_cepstra))
