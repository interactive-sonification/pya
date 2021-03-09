from unittest import TestCase
from pya import *
import numpy as np
# import logging
# logging.basicConfig(level=logging.DEBUG)


class TestSetitem(TestCase):

    def setUp(self):
        self.dur = 3.5
        self.sr = 1000
        self.ts = np.linspace(0, self.dur, int(self.dur * self.sr))
        self.sig = np.sin(2 * np.pi * 50 * self.ts ** 1.9)
        self.a1 = Asig(self.sig, sr=self.sr, channels=1, cn=['a'], label='1ch-sig')
        self.ak = Asig(np.tile(self.sig.reshape(3500, 1), (1, 4)), sr=self.sr,
                       label='4ch-sig', cn=['a', 'b', 'c', 'd'])
        self.one = np.ones(self.sr)
        self.aones = Asig(self.one, sr=self.sr, cn=['o'], label='ones')
        self.zero = np.zeros(self.sr)
        self.azeros = Asig(self.zero, sr=self.sr, cn=['z'], label='zeros')
        self.noise = np.random.random(self.sr)
        self.anoise = Asig(self.noise, sr=self.sr, cn=['n'], label='noise')
        self.aramp = Asig(np.arange(1000), sr=self.sr, label='ramp')

    def tearDown(self):
        pass

    def test_default(self):
        # Testing of default mode, which should behave as Numpy should."""
        self.azeros[10] = self.aones[10].sig   # value as asig
        self.assertEqual(self.aones[10], self.azeros[10])
        self.azeros[2] = self.aones[4].sig   # value as ndarray
        self.assertEqual(self.aones[4], self.azeros[2])
        self.azeros[3:6] = [1, 2, 3]  # value as list
        self.assertTrue(np.array_equal(self.azeros[3:6].sig, [1, 2, 3]))
        r = self.azeros
        r[3:6] = np.array([3, 4, 5])
        self.assertTrue(np.array_equal(r[3:6].sig, np.array([3, 4, 5])))
        r = self.azeros[:10]  # value as asig
        self.assertTrue(r, self.azeros[:10])
        self.azeros[{0.2: 0.4}] = self.anoise[{0.5: 0.7}]
        self.ak[{1: 2}, ['d']] = self.ak[{0: 1}, ['a']]
        self.assertTrue(np.array_equal(self.ak[{1: 2}, ['d']], self.ak[{0: 1}, ['a']]))

    def test_bound(self):
        # Testing of bound mode. Redundant array will not be assigned"""
        subject = self.aramp
        subject.b[:10] += self.aones[:10]  # This case should be the same as default
        result = np.arange(10) + np.ones(10)
        self.assertTrue(np.array_equal(subject[:10].sig, result))

        subject = self.aramp  # set 10 samples with 20 samples in bound mode
        subject.b[-10:] += np.arange(20)
        result = np.arange(1000)[-10:] + np.arange(10)
        self.assertTrue(np.array_equal(subject[-10:].sig, result))

        subject = Asig(np.arange(1000), sr=self.sr, label='ramp')
        subject.b[-10:] *= 2        # Test __mul__ also.
        result = np.arange(1000)[-10:] * 2
        self.assertTrue(np.array_equal(subject[-10:].sig, result))

        # # Multi channel case
        self.ak.b[{2: None}, ['a', 'b']] = np.zeros(shape=(3000, 2))
        result = np.sum(self.ak[{2: None}, ['a', 'b']].sig)
        self.assertEqual(result, 0.0)

    def test_extend(self):
        # Testing of extend mode, longer array will force the taker to extend its shape."""
        a = Asig(0.8, sr=1000, channels=4, cn=['a', 'b', 'c', 'd'])
        b = np.sin(2 * np.pi * 100 * np.linspace(0, 0.6, int(1000 * 0.6)))
        b = Asig(b)
        # test with extend set mono signal to a, initially only 0.8secs long...
        a.x[:, 0] = 0.2 * b  # this fits in without need to extend
        self.assertEqual(a.samples, 800)
        a.x[300:, 1] = 0.5 * b
        self.assertEqual(a.samples, 900)
        a.x[1300:, 'c'] = 0.2 * b[::2]  # compressed sig in ch 'c'
        self.assertEqual(a.samples, 1600)
        a.x[1900:, 3] = 0.2 * b[300:]  # only end of tone in ch 'd'
        self.assertEqual(a.samples, 2200)

        a = Asig(0.8, sr=1000, channels=1, cn=['a'])  # Test with mono signal
        b = np.sin(2 * np.pi * 100 * np.linspace(0, 0.6, int(1000 * 0.6)))
        b = Asig(b)
        a.x[:, 0] = 0.2 * b  # this fits in without need to extend
        self.assertEqual(a.samples, 800)

    def test_replace(self):
        b = np.ones(290)
        a = np.sin(2 * np.pi * 40 * np.linspace(0, 1, 100))
        a = Asig(a)
        a.overwrite[40:50] = b
        self.assertEqual(a.samples, 100 - 10 + 290)  # First make sure size is correct
        c = np.sum(a[50:60].sig)   # Then make sure replace value is correct
        self.assertEqual(c, 10)
        with self.assertRaises(ValueError):
            # Passing 2 chan to 4 chan asig should raise ValueError
            self.ak.overwrite[{1.: 1.5}] = np.zeros((int(44100 * 0.6), 2))

    def test_numpy_index(self):
        self.azeros[np.arange(0, 10)] = np.ones(10)
        self.assertTrue(np.array_equal(self.azeros[np.arange(0, 10)].sig, self.aones[np.arange(0, 10)].sig))

    def test_byte_index(self):
        self.azeros[bytes([0, 1, 2])] = np.ones(3)
        self.assertTrue(np.array_equal(self.azeros[[0, 1, 2]].sig, self.aones[[0, 1, 2]].sig))

    def test_asig_index(self):
        self.azeros[self.aones.sig.astype(bool)] = self.aones.sig
        self.assertTrue(np.array_equal(np.ones(self.sr), self.azeros.sig))

    def test_invalid_slicing_type(self):
        self.azeros[self.aones] = self.aones.sig
        self.assertTrue(np.array_equal(np.zeros(self.sr), self.azeros.sig))
