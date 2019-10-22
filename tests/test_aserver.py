from unittest import TestCase, skipUnless, mock
from pya import *
import numpy as np
import time


class TestAserver(TestCase):

    def setUp(self) -> None:
        self.backend = DummyBackend()
        self.sig = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        self.asine = Asig(self.sig, sr=44100, label="test_sine")

    def test_default_server(self):
        Aserver.startup_default_server(backend=self.backend, bs=512, channels=4)
        s = Aserver.default
        self.assertEqual(s, Aserver.default)
        self.asine.play()
        time.sleep(0.5)
        s.stop()
        self.assertGreater(len(s.stream.samples_out), 0)
        sample = s.stream.samples_out[0]
        self.assertEqual(sample.shape[0], 512)
        self.assertEqual(sample.shape[1], 4)
        self.assertAlmostEqual(np.max(sample), 1, places=2)
        Aserver.shutdown_default_server()
        self.assertIsNone(s.stream)

    def test_play_float(self):
        s = Aserver(backend=self.backend)
        s.boot()
        self.asine.play(server=s)
        time.sleep(0.5)
        s.stop()
        self.assertGreater(len(s.stream.samples_out), 0)
        sample = s.stream.samples_out[0]
        self.assertEqual(sample.shape[0], s.bs)
        self.assertEqual(sample.shape[1], s.channels)
        self.assertAlmostEqual(np.max(sample), 1, places=2)
        s.quit()

    def test_repr(self):
        s = Aserver(backend=self.backend)
        s.boot()
        print(s)
        s.quit()

    def test_get_devices(self):
        s = Aserver(backend=self.backend)
        d_in, d_out = s.get_devices(verbose=True)
        self.assertListEqual(d_in, d_out)
        self.assertListEqual(d_in, self.backend.dummy_devices)

    def test_boot_twice(self):
        s = Aserver(backend=self.backend)
        s.boot()
        self.assertEqual(s.boot(), -1)
        s.quit()

    def test_quit_not_booted(self):
        s = Aserver(backend=self.backend)
        self.assertEqual(s.quit(), -1)

    def test_incompatible_backend(self):
        s = Aserver(backend=self.backend)
        sig = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100) * np.iinfo(np.int16).max).astype(np.int16)
        asine = Asig(sig, sr=44100)
        s.boot()
        asine.play(server=s)
        s.quit()

