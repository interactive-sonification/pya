from .test_play import TestPlayBase
from .test_arecorder import TestArecorderBase
from pya.backend import DummyBackend, JupyterBackend

from unittest import TestCase, mock
import asyncio


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
        with mock.patch('asyncio.create_task') as m:
            b.open(channels=2, rate=44100)
            self.assertTrue(m.called)
