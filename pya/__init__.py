from pya.amfcc import Amfcc
from pya.arecorder import Arecorder
from pya.aserver import Aserver
from pya.asig import Asig
from pya.aspec import Aspec
from pya.astft import Astft

# from .helper.visualization import basicplots
from pya.backend import *  # noqa: F403
from pya.helper import *  # noqa: F403
from pya.ugen import Ugen

__all__ = ["Ugen", "Asig", "Aspec", "Astft", "Arecorder", "Amfcc"]


def startup(**kwargs):
    return Aserver.startup_default_server(**kwargs)


def shutdown(**kwargs):
    Aserver.shutdown_default_server(**kwargs)
