from unittest import TestCase
from pya import *
import numpy as np
import time


class TestAserver(TestCase):

    def setUp(self) -> None:
        self.backend = DummyBackend()
        self.sig = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        self.asine = Asig(self.sig, sr=44100, label="test_sine")
        self.max_channels = self.backend.dummy_devices[0]['maxOutputChannels']

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

    def test_custom_channels(self):
        """Server should boot with a valid channel"""
        for ch in [1, 5, 7, 10]:
            s = Aserver(device=0, channels=ch, backend=self.backend)
            s.boot()
            self.assertTrue(s.is_active)
            s.stop()
            s.quit()
            self.assertFalse(s.is_active)

    def test_invalid_channels(self):
        """Raise an exception if booting with channels greater than max channels of the device. Dummy has 10"""
        ch = 100
        s = Aserver(device=0, channels=ch, backend=self.backend)
        with self.assertRaises(OSError):
            s.boot()

    def test_default_channels(self):
        s = Aserver(device=0, backend=self.backend)
        self.assertEqual(s.channels, self.max_channels)

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

    def test_play_in_context_manager(self):
        with Aserver(backend=self.backend) as s:
            self.asine.play(server=s)
            time.sleep(0.5)
            sample = s.stream.samples_out[0]
            self.assertEqual(sample.shape[0], s.bs)
            self.assertEqual(sample.shape[1], s.channels)
            self.assertAlmostEqual(np.max(sample), 1, places=2)

    def test_none_blocking_play(self):
        t0 = time.time()
        with Aserver(backend=self.backend) as s:
            self.asine.play(server=s)
            self.asine.play(server=s)
            self.asine.play(server=s)
        dur = time.time() - t0
        self.assertLess(dur, self.asine.dur)

    def test_blocking_play(self):
        t0 = time.time()
        with Aserver(backend=self.backend) as s:
            self.asine.play(server=s, block=True)
            self.asine.play(server=s, block=True)
            self.asine.play(server=s, block=True)
        delta = time.time() - t0 - self.asine.dur * 3
        # plus a few hundred ms of aserver boot and stop time
        is_dur_reasonable = delta > 0 and delta < 1.0
        self.assertTrue(is_dur_reasonable)

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

