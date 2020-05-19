from unittest import TestCase
from pya import Ugen, Aspec, Asig
import warnings
import numpy as np


class TestAspec(TestCase):

    def setUp(self):
        self.asig = Ugen().sine()
        self.asig2 = Ugen().sine(channels=2, cn=['a', 'b'])
        self.asig_no_name = Ugen().sine(channels=3)

    def tearDown(self):
        pass

    def test_constructor(self):
        aspec = self.asig.to_spec()
        self.assertEqual(aspec.sr, 44100)
        aspec = self.asig2.to_spec()
        self.assertEqual(aspec.cn, self.asig2.cn)
        # The input can also be just an numpy array
        sig = Ugen().square().sig
        aspec = Aspec(sig, sr=400, label='square', cn=['a'])
        self.assertEqual(aspec.sr, 400)
        self.assertEqual(aspec.label, 'square')
        self.assertEqual(aspec.cn, ['a'])
        with self.assertRaises(TypeError):
            _ = Aspec(x=3)
        print(aspec)

    def test_plot(self):
        self.asig.to_spec().plot()
        self.asig.to_spec().plot(xlim=(0, 0.5), ylim=(0., 1.0))

    def test_cn_conflict(self):
        with warnings.catch_warnings(record=True):
            _ = Aspec(self.asig, cn=['jf', 'dj'])
