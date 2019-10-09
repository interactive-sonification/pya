from .test_play import TestPlayBase
from .test_arecorder import TestArecorderBase
from pya.backend.Dummy import DummyBackend

from pya import *
import numpy as np


# check if we have an output device
class TestDummyBackendPlay(TestPlayBase):

    __test__ = True
    backend = DummyBackend()


class TestDummyBackendRecord(TestArecorderBase):

    __test__ = True
    backend = DummyBackend()

