import unittest

import numpy as np

from pya.agen.core import multi_channel
from pya.agen.lib import SinOsc, expand_channels

from .helpers import IdentityGen


class ChannelTest(unittest.TestCase):
    def test_channel_count(self):
        gen = SinOsc(freq=multi_channel(1, 2, 3))
        self.assertEqual(gen.channels, 3)

    def test_channel_slicing(self):
        gen = multi_channel(
            1, 2, 3, 4, 5, 
            cn=["1", "2", "3", "4", "5"], 
        ).limit(seconds=1)
        for i in range(5):
            self.assertTrue(np.all(gen[i].gen_asig().sig == i + 1))
            self.assertTrue(np.all(gen[str(i+1)].gen_asig().sig == i + 1))

        self.assertEqual(gen[[0, 3]].channels, 2)
        self.assertTrue(np.all(gen[[0, 3]].gen_asig().sig == [1, 4]))

        self.assertEqual(gen[1:4].channels, 3)
        self.assertTrue(np.all(gen[1:4].gen_asig().sig == [2, 3, 4]))

        self.assertEqual(gen[[True, False, True, False, True]].channels, 3)
        self.assertTrue(
            np.all(gen[[True, False, True, False, True]].gen_asig().sig == [1, 3, 5])
        )

        self.assertEqual(gen[["1", "3", "5"]].channels, 3)
        self.assertTrue(np.all(gen[["1", "3", "5"]].gen_asig().sig == [1, 3, 5]))

    def test_expand_channels(self):
        gen = multi_channel(1, 2, 3).limit(seconds=1)
        expanded = expand_channels(gen, 5, strategy="cycle")
        self.assertEqual(expanded.channels, 5)
        self.assertTrue(np.all(expanded.gen_asig().sig == [1, 2, 3, 1, 2]))

        expanded = expand_channels(gen, 5, strategy="last")
        self.assertEqual(expanded.channels, 5)
        self.assertTrue(np.all(expanded.gen_asig().sig == [1, 2, 3, 3, 3]))

        expanded = expand_channels(gen, 5, strategy="zero")
        self.assertEqual(expanded.channels, 5)
        self.assertTrue(np.all(expanded.gen_asig().sig == [1, 2, 3, 0, 0]))

    def test_multi_channel(self):
        gen = multi_channel(1, 2, 3, multi_channel(4, 5, 6)).limit(seconds=1)
        self.assertEqual(gen.channels, 6)
        self.assertTrue(np.all(gen.gen_asig().sig == [1, 2, 3, 4, 5, 6]))

        gen = multi_channel(
            multi_channel(4, 5, 6), SinOsc(freq=multi_channel(10, 20))
        ).limit(seconds=1)
        self.assertEqual(gen.channels, 5)
        self.assertTrue(np.all(gen.gen_asig().sig[:, :3] == [4, 5, 6]))
        self.assertTrue(
            (
                gen.gen_asig()[:, 3].sig
                == SinOsc(freq=10).limit(seconds=1).gen_asig().sig.flatten()
            ).all()
        )
        self.assertTrue(
            (
                gen.gen_asig()[:, 4].sig
                == SinOsc(freq=20).limit(seconds=1).gen_asig().sig.flatten()
            ).all()
        )

    def test_mix(self):
        gen = multi_channel(1, 2, 3, 4, 5).limit(seconds=1)
        mixed = gen.mix()
        self.assertEqual(mixed.channels, 1)
        self.assertTrue(np.all(mixed.gen_asig().sig == 15))

        # Test with channels with different lengths
        gen = multi_channel(
            SinOsc(freq=1).limit(seconds=1),
            SinOsc(freq=2).limit(seconds=2),
        )
        mixed_sig = gen.mix().gen_asig()
        self.assertEqual(mixed_sig.dur, 1)

    def test_combine_agens(self):
        gen_1 = IdentityGen(1).limit(seconds=1)
        gen_2 = multi_channel(1, 2).limit(seconds=1)
        gen_3 = multi_channel(1, 2, 3).limit(seconds=1)

        self.assertTrue(np.all((gen_1 + gen_2).gen_asig().sig == [2, 3]))

        with self.assertRaises(ValueError):
            (gen_2 + gen_3).gen_asig()


if __name__ == "__main__":
    unittest.main()
