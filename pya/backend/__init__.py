import logging

from pya.backend.Dummy import DummyBackend
from pya.helper.backend import determine_backend

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())

__all__ = ["DummyBackend", "determine_backend"]

try:
    from pya.backend.Jupyter import JupyterBackend as JupyterBackend

    __all__.append("JupyterBackend")
except ImportError:  # pragma: no cover
    _LOGGER.warning("Jupyter backend not found.")
    pass

try:
    from pya.backend.PyAudio import PyAudioBackend as PyAudioBackend

    __all__.append("PyAudioBackend")
except ImportError:  # pragma: no cover
    _LOGGER.warning("PyAudio backend not found.")
    pass
