"""Collection of classes and functions for processing audio signals
 in python and jupyter notebooks, for synthesis, effects, analysis and plotting.
"""
from .pya import Asig, Aspec, Astft
from .Aserver import Aserver

# from .pya import *
from .helpers import ampdb, dbamp, cpsmidi, midicps, linlin, clip, record, timeit


def startup(**kwargs):
    return Aserver.startup_default_server(**kwargs)


def shutdown(**kwargs):
    Aserver.shutdown_default_server(**kwargs)
