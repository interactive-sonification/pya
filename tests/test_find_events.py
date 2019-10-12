from pya import Asig, Ugen
from unittest import TestCase


class TestFindEvents(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_events(self):
        # Create a signale with 3 sine waves and 
        # gaps inbetween, So that it will finds 3 events"""
        a = Ugen().sine()
        a.x[a.samples:] = Asig(0.2)
        a.x[a.samples:] = Ugen().sine(freq=200)
        a.x[a.samples:] = Asig(0.2)
        a.x[a.samples:] = Ugen().sine(freq=20)
        a.x[a.samples:] = Asig(0.2)
        a.find_events(sil_thr=-30, evt_min_dur=0.2, sil_min_dur=0.04)
        self.assertEqual(3, a._['events'].shape[0])
