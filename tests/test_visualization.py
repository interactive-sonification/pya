from unittest import TestCase
from pya import *


class TestVisualization(TestCase):

    def setUp(self):
        self.asig = Ugen().sine()
        self.aspec = self.asig.to_spec()
        self.astft = self.asig.to_stft()
        self.amfcc = self.asig.to_mfcc()
        self.alst = [self.asig, self.aspec, self.astft, self.amfcc]

    def tearDown(self):
        pass

    def test_asig_plot_default(self):
        self.asig.plot()

    def test_asig_plot_args(self):
        self.asig.plot(xlim=(0, 100), ylim=(0, 100))

    def test_asig_fn_db(self):
        self.asig.plot(fn='db')

    def test_asig_fn_nocallable(self):
        with self.assertRaises(AttributeError):
            self.asig.plot(fn='something')

    def test_asig_multichannels(self):
        sig2d = Ugen().sine(channels=4, cn=['a', 'b', 'c', 'd']) 
        sig2d.plot()

    def test_aspec_plot(self):
        self.aspec.plot()

    def tesst_aspec_plot_lim(self):
        self.aspect.plot(xlim=(0, 1.), ylim=(0, 100))

    def test_gridplot(self):
        _ = gridplot(self.alst)

    def test_gridplot_valid_colwrap(self):
        _ = gridplot(self.alst, colwrap=3)
        _ = gridplot(self.alst, colwrap=2)

    def test_gridplot_colwrap_too_big(self):
        # colwrap more than list len
        _ = gridplot(self.alst, colwrap=5)

    def test_gridplot_neg_colwrap(self):
        with self.assertRaises(ValueError):
            _ = gridplot(self.alst, colwrap=-1)
