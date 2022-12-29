from .amfcc import Amfcc
from .arecorder import Arecorder
from .aserver import Aserver
from .asig import Asig
from .aspec import Aspec
from .astft import Astft
from .backend import *
from .helper import *
from .ugen import Ugen
from .version import __version__


def startup(**kwargs):
    return Aserver.startup_default_server(**kwargs)


def shutdown(**kwargs):
    Aserver.shutdown_default_server(**kwargs)
