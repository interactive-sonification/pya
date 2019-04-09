
import numpy as np 
from scipy import signal

def _get_length(dur, sr):
    if isinstance(dur, float):
        length = int(dur * sr)
    elif isinstance(dur, int):
        length = dur
    else:
        raise TypeError("Unrecognise type for dur, int (samples) or float (seconds) only")
    return length

def sine(freq=440, amp=1.0, dur=1.0, sr=44100, cn=None):
    from .pya import Asig
    length = _get_length(dur, sr)
    sig = amp * np.sin(2*np.pi*freq*np.linspace(0, dur, length))
    return Asig(sig, sr=sr, label="sine", channels=1, cn=cn)

def square(freq=440, amp=1.0, duty=0.4, dur=1.0, sr=44100, cn=None):
    from .pya import Asig
    length = _get_length(dur, sr)
    sig = amp * signal.square(2 * np.pi * freq \
        * np.linspace(0, dur, length, endpoint=False), duty=duty)
    return Asig(sig, sr=sr, label="square", channels=1, cn=cn)