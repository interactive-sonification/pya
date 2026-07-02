import traceback
import unittest

import numpy as np

from pya.agen import config
from pya.agen.core import AGen, AGenState, CacheFlags
from pya.agen.lib import SinOsc


class StateTest(unittest.TestCase):
    def test_prealloc_memory(self):
        state = AGenState(CacheFlags(prealloc_memory=True, max_samples=2000), config.AUDIO_RATE)
        # We need to use _AGentState__samples instead of __samples due to Pythons name mangling for 
        # private attributes/functions
        self.assertEqual(state._AGenState__samples.shape[0], 2000)  # type: ignore

    def test_ring_buffer(self):
        state = AGenState(CacheFlags(max_samples=2000), config.AUDIO_RATE)
        state.add_to_cache(np.ones(1500))
        state.add_to_cache(np.ones(1500))
        with self.assertRaises(ValueError):
            state.get_from_cache(500, 0)

    def test_cache_length(self):
        state = AGenState(CacheFlags(), config.AUDIO_RATE)
        state.add_to_cache(np.arange(100))
        self.assertEqual(state.length, 100)
        state.add_to_cache(np.arange(50))
        self.assertEqual(state.length, 150)

    def test_get_from_cache(self):
        state = AGenState(CacheFlags(), config.AUDIO_RATE)
        state.add_to_cache(np.arange(100))
        self.assertTrue(np.all(state.get_from_cache(100, 0) == np.arange(100)))
        self.assertTrue(np.all(state.get_from_cache(50, 50) == np.arange(50) + 50))

        with self.assertRaises(ValueError):
            state.get_from_cache(50, 100)

        with self.assertRaises(ValueError):
            state.get_from_cache(1, 100)

        state.mark_finished()

        self.assertEqual(state.get_from_cache(50, 100).shape, (0,))
        self.assertEqual(state.get_from_cache(1, 1000).shape, (0,))
        self.assertTrue(np.all(state.get_from_cache(100, 50) == np.arange(50) + 50))

    def test_immutability(self):
        state = AGenState(CacheFlags(), config.AUDIO_RATE)
        state.add_to_cache(np.arange(100))
        view = state.get_from_cache(100, 0)
        with self.assertRaises(ValueError):
            view[0] = 100
        self.assertFalse(view.flags.writeable)

    def test_consecutive_blocks(self):
        test_self = self

        class TestGen(AGen):
            def _generate_new(self, sample_count, start, channel):
                last_sample = self.state.data.get("last_sample", -1)
                test_self.assertEqual(start, last_sample + 1)
                self.state.data["last_sample"] = start + sample_count - 1
                return np.zeros(sample_count)

        gen = TestGen().limit(seconds=5)
        gen2 = gen + gen.skip(seconds=1) + gen.delay(seconds=1)
        gen2.gen_asig()


if __name__ == "__main__":
    unittest.main()
