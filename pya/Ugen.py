from .pya import Asig
import numpy as np
from scipy import signal
from .helpers import get_length, normalize


class Ugen(Asig):
    def __init__(self):
        pass  

    def sine(self, freq=440, amp=1.0, dur=1.0, sr=44100, channels=1, cn=None, label="sine"):
        length = get_length(dur, sr)
        sig = amp * np.sin(2 * np.pi * freq * np.linspace(0, dur, length))
        if channels > 1:
            sig = np.repeat(sig, channels)
            sig = sig.reshape((length, channels))

        return Asig(sig, sr=sr, label=label, channels=channels, cn=cn)

    def square(self, freq=440, amp=1.0, dur=1.0, duty=0.4, sr=44100, channels=1, cn=None, label="square"):
        length = get_length(dur, sr)
        sig = amp * signal.square(2 * np.pi * freq * np.linspace(0, dur, length, endpoint=False),
                                  duty=duty)
        if channels > 1:
            sig = np.repeat(sig, channels)
            sig = sig.reshape((length, channels))
        return Asig(sig, sr=sr, label=label, channels=channels, cn=cn)

    def sawtooth(self, freq=440, amp=1.0, dur=1.0, width=1., sr=44100, channels=1, cn=None, label="sawtooth"):
        length = get_length(dur, sr)
        sig = amp * signal.sawtooth(2 * np.pi * freq * np.linspace(0, dur, length, endpoint=False),
                                    width=width)
        if channels > 1:
            sig = np.repeat(sig, channels)
            sig = sig.reshape((length, channels))
        return Asig(sig, sr=sr, label=label, channels=channels, cn=cn)

    def noise(self, type="white", amp=1.0, dur=1.0, sr=44100, channels=1, cn=None, label="noise"):
        length = get_length(dur, sr)
        # Question is that will be that be too slow.
        if type == "white" or "white_noise":
            sig = np.random.rand(length) * amp  # oR may switch to normal

        elif type == "pink" or "pink_noise":
            # Based on Paul Kellet's method
            b0, b1, b2, b3, b4, b5, b6 = 0, 0, 0, 0, 0, 0, 0
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
            sig = normalize(sig) * amp
        if channels > 1:
            sig = np.repeat(sig, channels)
            sig = sig.reshape((length, channels))
        return Asig(sig, sr=sr, channels=channels, cn=cn, label=label)