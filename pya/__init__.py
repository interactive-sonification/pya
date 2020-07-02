from .aserver import Aserver
from .asig import Asig
from .astft import Astft
from .aspec import Aspec
from .amfcc import Amfcc
from .arecorder import Arecorder
from .ugen import Ugen
from .version import __version__
from .helper import *
# from .helper.visualization import basicplots
from .backend import *


def startup(**kwargs):
    return Aserver.startup_default_server(**kwargs)


def shutdown(**kwargs):
    Aserver.shutdown_default_server(**kwargs)
