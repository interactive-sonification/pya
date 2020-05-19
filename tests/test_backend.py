from .test_play import TestPlayBase
from .test_arecorder import TestArecorderBase
from pya.backend import DummyBackend
from pya.backend import JupyterBackend
from unittest import TestCase
import time


# check if we have an output device
class TestDummyBackendPlay(TestPlayBase):

    __test__ = True
    backend = DummyBackend()


class TestDummyBackendRecord(TestArecorderBase):

    __test__ = True
    backend = DummyBackend()


class TestJupyterBackendPlay(TestCase):
    def test_boot(self):
        b = JupyterBackend()
        s = b.open(channels=2, rate=44100)
        time.sleep(0.2)
        self.assertTrue(s.loop.is_running())
        s.close()
        self.assertFalse(s.thread.is_alive())