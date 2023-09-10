import contextlib
import time
from typing import Callable


def wait(condition: Callable[[], bool], seconds: float = 10, interval: float = 0.1):
    start = time.time()
    while time.time() - start <= seconds:
        if condition() is True:
            return True
        time.sleep(interval)
    return False


def check_for_input() -> bool:
    with contextlib.suppress(ImportError, OSError):
        import pyaudio
        pyaudio.PyAudio().get_default_input_device_info()
        return True
    return False


def check_for_output() -> bool:
    with contextlib.suppress(ImportError, OSError):
        import pyaudio
        pyaudio.PyAudio().get_default_output_device_info()
        return True
    return False
