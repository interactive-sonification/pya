from unittest import TestCase
from pya import *
import numpy as np
# import logging
# logging.basicConfig(level=logging.DEBUG)


class TestSlicing(TestCase):

    def setUp(self):
        self.sig = np.sin(2 * np.pi * 100 * np.linspace(0, 1, 44100))
        self.asine = Asig(self.sig, sr=44100, label="test_sine")
        self.sig4 = np.sin(2 * np.pi * 100 * np.linspace(0, 4, 44100 * 4))  # 4second sine
        self.asine4 = Asig(self.sig4, sr=44100, label="test_sine")
        self.sig2ch = np.repeat(self.sig, 2).reshape((44100, 2))
        self.astereo = Asig(self.sig2ch, sr=44100, label="stereo", cn=['l', 'r'])

    def tearDown(self):
        pass

    def test_int(self):
        # Integer getitem
        self.assertAlmostEqual(self.asine4[4].sig, self.sig4[4])

    def test_intlist(self):
        # Integer list test. """
        self.assertTrue(np.allclose(self.asine[[2, 4, 5]].sig, self.sig[[2, 4, 5]]))

    def test_namelist(self):
        # Check whether I can pass a list of column names and get the same result"""
        result = self.astereo[:, ["l", "r"]]
        expect = self.astereo[:, [0, 1]]
        self.assertEqual(result, expect)

    def test_bytes(self):
        result = self.astereo[:, bytes([0, 1])]
        expect = self.astereo[:, [0, 1]]
        self.assertEqual(result, expect)

    def test_numpy_array(self):
        result = self.astereo[:, np.array([1, 0])]
        expect = self.astereo[:, [1, 0]]
        self.assertEqual(result, expect)

    def test_invalid_index(self):
        # pass false class or module instance should return initial signal
        self.assertEqual(self.astereo[:, self.astereo], self.astereo)
        self.assertEqual(self.astereo[:, np], self.astereo)

    def test_timeSlicing(self):
        # Check whether time slicing equals sample slicing."""
        result = self.asine[{0: 1.0}]
        expect = self.asine[:44100]
        self.assertEqual(expect, result)

        # Check negative time work"""
        result2 = self.asine4[{1: -1}]  # Play from 1s. to the last 1.s
        expect2 = self.asine4[44100: -44100]
        self.assertEqual(expect2, result2)

    def test_tuple(self):
        # single channel, jump sample
        result = self.astereo[0:44100:2, 0]
        expected_sig = self.astereo.sig[0:44100:2, 0]
        self.assertTrue(np.array_equal(result.sig, expected_sig))

        result = self.astereo[0:10:2, ['l']]
        expected_sig = self.astereo.sig[0:10:2, 0]
        self.assertTrue(np.array_equal(result.sig, expected_sig))  # Check if signal equal
        self.assertEqual(result.cn, ['l'])  # Check whether the new column name is correct

        # channel name slice as list.
        # ("both channels using col_name")
        result = self.astereo[0:44100:2, ['l', 'r']]
        expected_sig = self.astereo.sig[0:44100:2, :]
        self.assertTrue(np.array_equal(result.sig, expected_sig))

        # Bool slice
        # ("bool list channel selection")
        # This is a special case for scling as numpy return (n, 1) rather than (n,) if we use
        # bool list to single out a channel.
        result = self.astereo[360:368, [False, True]]
        expected_sig = self.astereo.sig[360:368:1, [False, True]]
        self.assertTrue(np.array_equal(result.sig, expected_sig))
        # time slicing
        result = self.astereo[{1: -1}, 0]  # Play from 1s. to the last 1.s
        expect = self.astereo[44100: -44100, 0]
        self.assertEqual(expect, result)

        # time slicing
        # ("time slicing.")
        time_range = {1: -1}   # first to last second.
        result = self.astereo[time_range, :]   # Play from 1s. to the last 1.s
        expect = self.astereo[44100: -44100, :]
        self.assertEqual(expect, result)
