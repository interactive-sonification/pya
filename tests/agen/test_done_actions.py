import unittest

from pya.agen.lib import *


class DoneActionTest(unittest.TestCase):
    def test_stop_len(self):
        gen = Line(0, 1, 3, done="stop")
        self.assertAlmostEqual(gen.gen_asig().get_duration(), 3.0)

        gen2 = SinOsc(freq=Line(0, 1, 3, done="stop"))
        self.assertAlmostEqual(gen2.gen_asig().get_duration(), 3.0)

        gen3 = SinOsc(freq=2) * Line(0, 1, 4, done="stop")
        self.assertAlmostEqual(gen3.gen_asig().get_duration(), 4.0)

    def test_loop(self):
        gen = Line(0, 1, 2, done="loop")
        asig = gen.gen_asig(seconds=100)
        self.assertAlmostEqual(asig.get_duration(), 100.0)
        self.assertAlmostEqual(asig.sig[int(3.0 * gen.sr)], 0.5)

        gen2 = Line(0, 1, 2, done="loop") + Line(0, 0, 10, done="stop")
        asig2 = gen2.gen_asig()
        self.assertAlmostEqual(asig2.sig[int(3.0 * gen.sr)], 0.5)

    def test_last(self):
        gen = Line(0, 1, 1, done="last", sr=100)
        asig = gen.gen_asig(seconds=100)
        self.assertAlmostEqual(asig.get_duration(), 100.0)
        self.assertAlmostEqual(asig.sig[-1], 0.99)
        self.assertAlmostEqual(asig.sig[-20], 0.99)

    def test_last_complex(self):
        # This also fails
        gen_long = Line(0, 1, 10, done="stop", sr=1000)
        gen = Line(0, 1, 1, done="last", sr=100) + gen_long
        asig = gen.gen_asig()
        gen_long.reset()
        gen_long_asig = gen_long.gen_asig()
        self.assertEqual(asig.get_duration(), gen_long_asig.get_duration())

    def test_zero(self):
        gen = Line(0, 1, 1, done="zero", sr=100)
        asig = gen.gen_asig(seconds=100)
        self.assertAlmostEqual(asig.get_duration(), 100.0)
        self.assertAlmostEqual(asig.sig[-1], 0)
        self.assertAlmostEqual(asig.sig[-20], 0)

    def test_with_done(self):
        gen = Line(0, 1, 1, sr=100, done="stop").with_done("last")
        asig = gen.gen_asig(seconds=100)
        self.assertAlmostEqual(asig.get_duration(), 100.0)
        self.assertAlmostEqual(asig.sig[-1], 0.99)
        self.assertAlmostEqual(asig.sig[-20], 0.99)



if __name__ == "__main__":
    unittest.main()
