import unittest

from pya.agen import config
from pya.agen.lib import *

config.UPSAMPLE_WARNING = False


class SampleRatesTest(unittest.TestCase):
    def test_sine_osc_sr(self):
        gen = SinOsc(sr=None, freq=Line(start=1.0, end=2.0, dur=1, sr=44100))
        self.assertEqual(gen.sr, 44100)

        gen_2 = SinOsc(
            sr=None,
            freq=Line(start=1.0, end=2.0, dur=1, sr=5000),
            phase=Line(start=0.0, end=0.5, dur=1, sr=10_000),
        )
        self.assertEqual(gen_2.sr, 10_000)

    def test_operations_sr(self):
        gen = SinOsc(sr=1000, freq=10)
        gen_2 = SinOsc(sr=10_000, freq=20)

        self.assertEqual((gen + gen_2).sr, 10_000)
        self.assertEqual((gen * gen_2).sr, 10_000)
        self.assertEqual((gen / gen_2).sr, 10_000)
        self.assertEqual((gen**gen_2).sr, 10_000)
        self.assertEqual((gen & gen_2).sr, 10_000)
        self.assertEqual((-gen).sr, 1000)
        self.assertEqual(gen[[0, 0]].sr, 1000)

    def test_helpers_sr(self):
        base = SinOsc(sr=100, freq=10)
        self.assertEqual(base.stereo().sr, 100)
        self.assertEqual(base.cos().sr, 100)
        self.assertEqual(base.skip(seconds=1).sr, 100)
        self.assertEqual(base.delay(seconds=1).sr, 100)
        self.assertEqual(base.limit(seconds=1).sr, 100)
        self.assertEqual(base.fade_in(seconds=1).sr, 100)
        self.assertEqual(base.fade_out(seconds=1).sr, 100)
        self.assertEqual(base.fade(both=1).sr, 100)
        self.assertEqual(base.apply(np.clip, -1, 1).sr, 100)


if __name__ == "__main__":
    unittest.main()
