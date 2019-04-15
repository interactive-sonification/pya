from unittest import TestCase
from pya import *

import numpy as np

class TestSlicing(TestCase):

    def setUp(self):
        self.sig = np.sin(2*np.pi* 100 * np.linspace(0,1,44100))
        self.asine = Asig(self.sig, sr=44100,label="test_sine")
        self.astereo = Asig("../examples/samples/stereoTest.wav", label='stereo', cn = ['l','r'])
        self.sig4 = np.sin(2*np.pi* 100 * np.linspace(0,4,44100 * 4))  # 4second sine
        self.asine4 = Asig(self.sig, sr=44100,label="test_sine")

    def tearDown(self):
        pass

    def test_int(self):
        self.assertAlmostEqual(self.asine[4].sig, self.sig[4])

    def test_intlist(self):
        self.assertTrue(np.array_equal(self.asine[[2, 4, 5]].sig, self.sig[[2, 4, 5]]))

    def test_namelist(self):
        """Check whether I can pass a list of column names and get the same result"""
        result = self.astereo[["l", "r"]]
        expect = self.astereo[:,[0, 1]]
        self.assertEqual(result, expect)

    def test_timeSlicing(self):
        """Check whether time slicing equals sample slicing."""
        result = self.asine[{0 : 1.0}]
        expect = self.asine[:44100]
        self.assertEqual(expect, result)

        """Check negative time work"""
        result2 = self.asine4[{1: -1}]  # Play from 1s. to the last 1.s
        expect2 = self.asine4[44100: -44100]
        self.assertEqual(expect2, result2)

    def test_tuple(self):
        result = self.astereo[0:44100:2, 0]
        expected_sig = self.astereo.sig[0:44100:2, 0]
        self.assertTrue(np.array_equal(result.sig, expected_sig))

        result = self.astereo[0:44100:2, 'l']
        expected_sig = self.astereo.sig[0:44100:2, 0]
        self.assertTrue(np.array_equal(result.sig, expected_sig))  # Check if signal equal
        self.assertEqual(result.cn, 'l')  # Check whether the new column name is correct

        result = self.astereo[0:44100:2, ['l', 'r']]
        expected_sig = self.astereo.sig[0:44100:2, :]
        self.assertTrue(np.array_equal(result.sig, expected_sig))


        result = self.astereo[0:368, [False, True]]
        expected_sig = self.astereo.sig[0:368:1, [False, True]]
        self.assertTrue(np.array_equal(result.sig, expected_sig))