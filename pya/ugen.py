
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

def _normalize(d):
    # d is a (n x dimension) np array
    d -= np.min(d, axis=0)
    d /= np.ptp(d, axis=0)
    return d

def sine(freq=440, amp=1.0, dur=1.0, sr=44100, channels=1, cn=None):
    from .pya import Asig
    length = _get_length(dur, sr)
    sig = amp * np.sin(2*np.pi*freq*np.linspace(0, dur, length))
    return Asig(sig, sr=sr, label="sine", channels=channels, cn=cn)

def square(freq=440, amp=1.0, dur=1.0, duty=0.4, sr=44100, channels=1, cn=None):
    from .pya import Asig
    length = _get_length(dur, sr)
    sig = amp * signal.square(2 * np.pi * freq \
        * np.linspace(0, dur, length, endpoint=False), duty=duty)
    return Asig(sig, sr=sr, label="square", channels=channels, cn=cn)


def sawtooth(freq=440, amp=1.0, dur=1.0, width=1., sr=44100, channels=1, cn=None):
    from .pya import Asig
    length = _get_length(dur, sr)
    sig = amp * signal.sawtooth(2 * np.pi * freq \
        * np.linspace(0, dur, length, endpoint=False), width=width)
    return Asig(sig, sr=sr, label="sawtooth", channels=channels, cn=cn)

def noise(typ="white", amp=1.0, dur=1.0, sr=44100, channels=1, cn=None):
    from .pay import Asig
    length = _get_length(dur, sr)
    # Question is that will be that be too slow. 
    if typ == "white" or "white_noise":
        sig = np.random.rand(length) * amp  # oR may switch to normal

    elif typ == "pink" or "pink_noise":
        b0,b1,b2,b3,b4,b5,b6 = 0,0,0,0,0,0,0
        sig = []
        for i in range(length):
            white = np.random.random() * 1.98 - 0.99
            b0 = 0.99886 * b0 + white * 0.0555179
            b1 = 0.99332 * b1 + white * 0.0750759 
            b2 = 0.96900 * b2 + white * 0.1538520 
            b3 = 0.86650 * b3 + white * 0.3104856 
            b4 = 0.55000 * b4 + white * 0.5329522 
            b5 = -0.7616 * b5 - white * 0.0168980 
            sig.append(b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362)
            b6 = white * 0.115926
        sig = _normalize(sig) * amp

    # elif typ == "brown" or "brown_noise"





    return Asig(sig, sr=sr, label=typ, channels=channels, cn=cn)