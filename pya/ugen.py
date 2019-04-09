
import numpy as np 

def sine(freq=440, amp=1.0, dur=1.0, sr=44100, cn=None):
    from .pya import Asig
    if isinstance(dur, float):
        length = int(dur * sr)
    elif isinstance(dur, int):
        length = dur
    else:
        raise TypeError("Unrecognise type for dur, int (samples) or float (seconds) only")
    sig = np.sin(2*np.pi*freq*np.linspace(0, dur, length))
    return Asig(sig, sr=sr, label="sine", channels=1, cn=cn)