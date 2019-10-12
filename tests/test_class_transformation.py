# Test change between asig, astft and aspec. 
from pya import Asig, Aspec, Astft, Ugen
from unittest import TestCase


class TestClassTransform(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_asig_aspec(self):
        # Create a signale with 3 sine waves and gaps inbetween, 
        # So that it will finds 3 events
        a = Ugen().sine()
        a_spec = a.to_spec()
        a_sig_from_aspec = a_spec.to_sig()
        self.assertIsInstance(a, Asig)
        self.assertIsInstance(a_spec, Aspec)
        self.assertIsInstance(a_sig_from_aspec, Asig)

    def test_asig_astf(self):
        a = Ugen().square()
        a_stft = a.to_stft()
        a_sig_from_stft = a_stft.to_sig()
        self.assertIsInstance(a, Asig)
        self.assertIsInstance(a_stft, Astft)
        self.assertIsInstance(a_sig_from_stft, Asig)