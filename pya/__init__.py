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
from .helper import ampdb, dbamp, cpsmidi, midicps, linlin, clip, get_length,\
                    timeit, audio_from_file, buf_to_float, spectrum, audio_from_file, \
                    normalize, record
                    

def startup(**kwargs):
    return Aserver.startup_default_server(**kwargs)


def shutdown(**kwargs):
    Aserver.shutdown_default_server(**kwargs)
    
__all__ = ['Asig', 'Aspec', 'Astft', 'Aserver', 'Arecorder', 'Ugen',
           'startup', 'shutdown']

__all__ += ['ampdb', 'dbamp', 'cpsmidi', 'midicps', 'linlin', 'clip', 'get_length',\
                    'audio_from_file', 'buf_to_float', 'spectrum', 'audio_from_file', \
                        'normalize', 'record']
