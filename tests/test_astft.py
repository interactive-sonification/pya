from unittest import TestCase, mock
from pya import Ugen, Astft
import numpy as np


class MockPlot(mock.MagicMock):
    pass


class TestAstft(TestCase):

    def setUp(self):
        self.asig = Ugen().sine()
        self.asig2 = Ugen().sine(channels=2, cn=['a', 'b'])
        self.asig_no_name = Ugen().sine(channels=3)

    def tearDown(self):
        pass

    def test_input_as_asig(self):
        astft = self.asig.to_stft()
        self.assertEqual(astft.sr, 44100)
        astft = self.asig.to_stft(sr=2000)
        self.assertEqual(astft.sr, 2000)
        signal = self.asig2.sig

    def test_wrong_input_type(self):
        with self.assertRaises(TypeError):
            asig = Astft(x=3, sr=500)

    def test_multichannel_asig(self):
        # Test conversion of a multi channel asig's astft. 
        asine = Ugen().sawtooth(channels=3, cn=['a', 'b', 'c'])
        astft = asine.to_stft()
        self.assertEqual(astft.channels, 3)
        self.assertEqual(len(astft.cn), 3)

    def test_input_as_stft(self):
        sr = 10e3
        N = 1e5
        amp = 2 * np.sqrt(2)
        noise_power = 0.01 * sr / 2
        time = np.arange(N) / float(sr)
        mod = 500 * np.cos(2 * np.pi * 0.25 * time)
        carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
        noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
        noise *= np.exp(-time / 5)
        x = carrier + noise
        astft = Astft(x, sr, label="test")

    def test_plot(self):
        self.asig.to_stft().plot()