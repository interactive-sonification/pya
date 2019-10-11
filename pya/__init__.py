"""Collection of classes and functions for processing audio signals
 in python and jupyter notebooks, for synthesis, effects, analysis and plotting.
"""
from .Asig import Asig, Metadata
from .Astft import Astft
from .Aspec import Aspec
from .Aserver import Aserver
from .Arecorder import Arecorder
from .Ugen import Ugen
from .version import __version__
from .helper import ampdb, dbamp, cpsmidi, midicps, linlin, clip
from .helper import audio_from_file, buf_to_float, spectrum, audio_from_file
from .helper import normalize, record, device_info


def startup(**kwargs):
    return Aserver.startup_default_server(**kwargs)


def shutdown(**kwargs):
    Aserver.shutdown_default_server(**kwargs)


__all__ = ['Asig', 'Aspec', 'Astft', 'Aserver', 'Arecorder', 'Ugen',
           'startup', 'shutdown', 'Metadata']

__all__ += ['ampdb', 'dbamp', 'cpsmidi', 'midicps', 'linlin', 'clip', 
            'audio_from_file', 'buf_to_float', 'spectrum', 
            'audio_from_file', 'normalize', 'record', 'device_info']