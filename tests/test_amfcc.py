# from pya import Amfcc
from pya import Asig, Ugen, Amfcc
from unittest import TestCase
import numpy as np
import warnings


class TestAmfcc(TestCase):
    def setUp(self):
        self.test_asig = Ugen().square(freq=20, sr=8000)
        self.test_sig = self.test_asig.sig

    def tearDown(self):
        pass

    def test_construct(self):
        # If x is asig, it will ignore sr but use x.sr instead.
        amfcc = Amfcc(self.test_asig, sr=45687)
        self.assertEqual(amfcc.sr, 8000, msg='sr does not match.')

        # if x is ndarray and sr is not given
        with self.assertRaises(AttributeError):
            _ = Amfcc(self.test_sig)

        # x is ndarray and sr is given.
        amfcc = Amfcc(self.test_sig, sr=1000)
        self.assertEqual(amfcc.sr, 1000, msg="sr does not match.")

        # Unsupported input type
        with self.assertRaises(TypeError):
            _ = Amfcc(x="String")

    def test_get_attributes(self):
        amfcc = Amfcc(self.test_asig)
        self.assertTrue(amfcc.timestamp is not None)
        self.assertTrue(amfcc.features is not None)

    def test_hopsize_greater_than_npframe(self):
        with warnings.catch_warnings(record=True):
            amfcc = Amfcc(self.test_asig, hopsize=100, n_per_frame=50)

    def test_nfft_not_pow2(self):
        with warnings.catch_warnings(record=True):
            amfcc = Amfcc(self.test_asig, nfft=23)

    def test_nowindowing(self):
        amfcc = Amfcc(self.test_asig, window=False)
        result = np.ones((amfcc.n_per_frame,))
        self.assertTrue(np.array_equal(result, amfcc.window))

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

    def test_plot_with_multichannel(self):
        asig = Ugen().sine(channels=2)
        amfcc = asig.to_mfcc()
        with warnings.catch_warnings(record=True):
            amfcc.plot()