from pya.backend import DummyBackend

from .test_arecorder import TestArecorderBase
from .test_play import TestPlayBase

try:
    has_j_backend = True
except Exception:
    has_j_backend = False


# check if we have an output device
class TestDummyBackendPlay(TestPlayBase):
    __test__ = True
    backend = DummyBackend()


class TestDummyBackendRecord(TestArecorderBase):
    __test__ = True
    backend = DummyBackend()


# class TestJupyterBackendPlay(TestCase):
#     @skipUnless(has_j_backend, "pya has no Jupyter Backend installed.")
#     def test_boot(self):
#         b = JupyterBackend()
#         s = b.open(channels=2, rate=44100)
#         is_running = wait(s.loop.is_running, seconds=10)
#         self.assertTrue(is_running)

#         # Just use the built-in close method
#         s.close()
#         self.assertFalse(s.thread.is_alive())
