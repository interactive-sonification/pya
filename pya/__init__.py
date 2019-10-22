"""Collection of classes and functions for processing audio signals
 in python and jupyter notebooks, for synthesis, effects, analysis and plotting.
"""
from .Asig import Asig
from .Astft import Astft
from .Aspec import Aspec
from .Aserver import Aserver
from .Arecorder import Arecorder
from .Ugen import Ugen
from .version import __version__
from .helper import *
from .backend import *


def startup(**kwargs):
    return Aserver.startup_default_server(**kwargs)


def shutdown(**kwargs):
    Aserver.shutdown_default_server(**kwargs)
