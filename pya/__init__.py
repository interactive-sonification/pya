"""Collection of classes and functions for processing audio signals
 in python and jupyter notebooks, for synthesis, effects, analysis and plotting.
"""
from .pya import Asig, Aspec, Astft
from .Aserver import Aserver
from .Ugen import Ugen
from .helpers import ampdb, dbamp, cpsmidi, midicps, linlin, clip, record, timeit, audioread_load, buf_to_float
from .version import __version__

def startup(**kwargs):
    return Aserver.startup_default_server(**kwargs)


def shutdown(**kwargs):
    Aserver.shutdown_default_server(**kwargs)
