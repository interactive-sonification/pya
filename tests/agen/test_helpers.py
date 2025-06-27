import unittest

import numpy as np

from pya.agen.lib import Line, SinOsc


class HelperMethodsTest(unittest.TestCase):
    def test_zero_padding(self):
        gen = SinOsc(freq=Line(20, 200, 1))
        delayed = gen.delay(samples=1000, padding="zero")
        a = gen.gen_asig()
        b = delayed.gen_asig()
        self.assertTrue(np.all(b.sig[:1000] == np.zeros(1000)))
        self.assertTrue(np.all(b.sig[1000:] == a.sig))

    def test_first_padding(self):
        gen = SinOsc(freq=Line(20, 200, 1), phase=0.5 * np.pi)
        delayed = gen.delay(samples=1000, padding="first")
        a = gen.gen_asig()
        b = delayed.gen_asig()
        self.assertTrue(np.all(b.sig[:1000] == a.sig[0]))
        self.assertTrue(np.all(b.sig[1000:] == a.sig))

    def test_skip(self):
        gen = SinOsc(freq=Line(20, 200, 1), phase=0.5 * np.pi)
        skipped = gen.skip(samples=1000)
        a = gen.gen_asig()
        b = skipped.gen_asig()
        self.assertTrue(np.all(b.sig == a.sig[1000:]))

    def test_limit(self):
        gen = SinOsc(freq=10)
        self.assertEqual(gen.limit(seconds=1).gen_asig().dur, 1.0)
        self.assertEqual(gen.limit(samples=1000).gen_asig().samples, 1000)

        # Limit with done action
        gen = SinOsc(freq=10).limit(seconds=1, done="last")
        asig = gen.gen_asig(seconds=10)
        self.assertEqual(asig.dur, 10.0)
        self.assertTrue(np.all(asig.sig[-1000:] == asig.sig[-1]))

    def test_fade(self):
        gen = SinOsc(freq=Line(10, 100, 1), phase=0.5 * np.pi)
        faded = gen.fade_in(seconds=1).gen_asig()
        self.assertEqual(faded.sig[0], 0)
        self.assertTrue((faded.sig[gen.sr :] == gen.gen_asig().sig[gen.sr :]).all())

        faded_out = gen.fade_out(seconds=1).gen_asig()
        self.assertTrue(
            (faded_out.sig[: -gen.sr] == gen.gen_asig().sig[: -gen.sr]).all()
        )
        self.assertAlmostEqual(faded_out.sig[-1].item(), 0, places=3)

    def test_with_label(self):
        gen = SinOsc(freq=10)
        self.assertEqual(gen.with_label("test").label, "test")


if __name__ == "__main__":
    unittest.main()
