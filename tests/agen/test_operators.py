import unittest

import numpy as np

from pya.agen.lib import Line, OneZero, SinOsc


class TestOperators(unittest.TestCase):
    def test_math_operators(self):
        gen1 = SinOsc(freq=Line(20, 200, 1))
        gen2 = Line(0.01, 1, 1)
        sig1 = gen1.gen_asig().sig
        sig2 = gen2.gen_asig().sig

        def isclose(a, b, atol=1e-5):
            return np.isclose(a, b, atol=atol, equal_nan=True).all()

        self.assertTrue(isclose((gen1 + gen2).gen_asig().sig, sig1 + sig2))
        self.assertTrue(isclose((gen1 - gen2).gen_asig().sig, sig1 - sig2))
        self.assertTrue(isclose((gen1 * gen2).gen_asig().sig, sig1 * sig2))
        self.assertTrue(isclose((gen1 / gen2).gen_asig().sig, sig1 / sig2))
        self.assertTrue(
            isclose(((gen1 + 1) ** gen2).gen_asig().sig, (sig1 + 1) ** sig2, atol=1e-2)
        )
        self.assertTrue(isclose((-gen1).gen_asig().sig, -sig1))

    def test_concat(self):
        gen1 = SinOsc(freq=Line(20, 200, 1))
        gen2 = Line(0.01, 1, 1)
        sig1 = gen1.gen_asig().sig
        sig2 = gen2.gen_asig().sig

        self.assertTrue(
            np.isclose((gen1 & gen2).gen_asig().sig, np.concatenate([sig1, sig2])).all()
        )

    def test_currying(self):
        gen = SinOsc(freq=Line(20, 200, 1))
        self.assertTrue(
            (
                (gen | OneZero.p(0.5)).gen_asig().sig
                == OneZero(gen, 0.5).gen_asig().sig
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
