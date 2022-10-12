import time
from typing import Callable


def wait(condition: Callable[[], bool], seconds: float = 10, interval: float = 0.1):
    start = time.time()
    while time.time() - start <= seconds:
        if condition() is True:
            return True
        time.sleep(interval)
    return False
