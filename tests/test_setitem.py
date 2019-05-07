from unittest import TestCase
from pya import *
import numpy as np
import logging
logging.basicConfig(level = logging.DEBUG)


class TestSetitem(TestCase):

    def setUp(self):
        self.dur = 3.5
        self.sr = 1000
        self.ts = np.linspace(0, self.dur, int(self.dur*self.sr))
        self.sig = np.sin(2*np.pi*50*self.ts**1.9)
        self.a1 = Asig(self.sig, sr=self.sr, channels=1, cn=['a'], label='1ch-sig')
        self.ak = Asig(np.tile(self.sig.reshape(3500,1), (1,4)), sr=self.sr, 
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

        self.azeros[10] = self.aones[10].sig   # value as asig
        self.assertEqual(self.aones[10], self.azeros[10])

        self.azeros[2] = self.aones[4].sig   # value as ndarray
        self.assertEqual(self.aones[4], self.azeros[2])

        self.azeros[3:6] = [1,2,3]  # value as list
        self.assertTrue(np.array_equal(self.azeros[3:6].sig, [1,2,3]))

        r = self.azeros
        r[3:6] = np.array([3,4,5])
        self.assertTrue(np.array_equal(r[3:6].sig, np.array([3,4,5])))

        r = self.azeros[:10]  # value as asig
        self.assertTrue(r, self.azeros[:10])

        self.azeros[{0.2:0.4}] = self.anoise[{0.5:0.7}]

        self.ak[{1:2}, ['d']] = self.ak[{0:1}, ['a']]
        self.assertTrue(np.array_equal(self.ak[{1:2}, ['d']], self.ak[{0:1}, ['a']]))
        

    def test_bound(self):
        subject = self.aramp
        subject.b[:10] += self.aones[:10]  # This case should be the same as default
        result = np.arange(10) + np.ones(10)
        self.assertTrue(np.array_equal(subject[:10].sig, result))

        subject = self.aramp
        subject.b[-10:] += np.arange(20)
        result = np.arange(1000)[-10:]  + np.arange(10)
        self.assertTrue(np.array_equal(subject[-10:].sig, result))

        subject = Asig(np.arange(1000), sr=self.sr, label='ramp')
        subject.b[-10:] *= 2
        result =  np.arange(1000)[-10:] * 2
        self.assertTrue(np.array_equal(subject[-10:].sig, result))

        # # Multi channel case?
        self.ak.b[{2:None}, ['a', 'b']] = np.zeros(shape=(3000, 2))

        result = np.sum(self.ak[{2:None}, ['a', 'b']].sig)

        self.assertEqual(result, 0.0)



    def test_extend(self):
        pass

    def test_replace(self):
        pass
    
