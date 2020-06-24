from .Dummy import DummyBackend
from .PyAudio import PyAudioBackend
import logging

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


try:
    from .Jupyter import JupyterBackend
except ImportError:  # pragma: no cover
    _LOGGER.warning("Jupyter backend not found.")
    pass
from ..helper.backend import determine_backend
