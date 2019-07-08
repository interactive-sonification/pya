from .pya import Asig
import numpy as np
from scipy import signal
from .helpers import get_length, normalize


class Ugen(Asig):
    """Unit Generator for to create Asig with predefined signal"""
    def __init__(self):
        pass  

    def sine(self, freq=440, amp=1.0, dur=1.0, sr=44100, channels=1, cn=None, label="sine"):
        """Generate Sine signal Asig object. 

        Parameters
        ----------
        freq : int, float
            signal frequency (Default value = 440)
        amp : int, float
            signal amplitude (Default value = 1.0)
        dur : int, float
            duration in second (Default value = 1.0)
        sr : int
            sampling rate (Default value = 44100)
        channels : int
            number of channels (Default value = 1)
        cn : list of string
            channel names as a list. The size needs to match the number of channels (Default value = None)
        label : string
            identifier of the object (Default value = "sine")

        Returns
        -------
        Asig object
        """
        length = get_length(dur, sr)
        sig = amp * np.sin(2 * np.pi * freq * np.linspace(0, dur, length))
        if channels > 1:
            sig = np.repeat(sig, channels)
            sig = sig.reshape((length, channels))
        return Asig(sig, sr=sr, label=label, channels=channels, cn=cn)

    def cos(self, freq=440, amp=1.0, dur=1.0, sr=44100, channels=1, cn=None, label="cosine"):
        """Generate Cosine signal Asig object. 

        Parameters
        ----------
        freq : int, float
            signal frequency (Default value = 440)
        amp : int, float
            signal amplitude (Default value = 1.0)
        dur : int, float
            duration in second (Default value = 1.0)
        sr : int
            sampling rate (Default value = 44100)
        channels : int
            number of channels (Default value = 1)
        cn : list of string
            channel names as a list. The size needs to match the number of channels (Default value = None)
        label : string
            identifier of the object (Default value = "cosine")

        Returns
        -------
        Asig object
        """
        length = get_length(dur, sr)
        sig = amp * np.cos(2 * np.pi * freq * np.linspace(0, dur, length))
        if channels > 1:
            sig = np.repeat(sig, channels)
            sig = sig.reshape((length, channels))
        return Asig(sig, sr=sr, label=label, channels=channels, cn=cn)

    def square(self, freq=440, amp=1.0, dur=1.0, duty=0.4, sr=44100, channels=1, cn=None, label="square"):
        """Generate square wave signal Asig object. 

        Parameters
        ----------
        freq : int, float
            signal frequency (Default value = 440)
        amp : int, float
            signal amplitude (Default value = 1.0)
        dur : int, float
            duration in second (Default value = 1.0)
        duty : float
            duty cycle (Default value = 0.4)
        sr : int
            sampling rate (Default value = 44100)
        channels : int
            number of channels (Default value = 1)
        cn : list of string
            channel names as a list. The size needs to match the number of channels (Default value = None)
        label : string
            identifier of the object (Default value = "square")

        Returns
        -------
        Asig object
        """
        length = get_length(dur, sr)
        sig = amp * signal.square(2 * np.pi * freq * np.linspace(0, dur, length, endpoint=False),
                                  duty=duty)
        if channels > 1:
            sig = np.repeat(sig, channels)
            sig = sig.reshape((length, channels))
        return Asig(sig, sr=sr, label=label, channels=channels, cn=cn)

    def sawtooth(self, freq=440, amp=1.0, dur=1.0, width=1., sr=44100, channels=1, cn=None, label="sawtooth"):
        """Generate sawtooth wave signal Asig object. 

        Parameters
        ----------
        freq : int, float
            signal frequency (Default value = 440)
        amp : int, float
            signal amplitude (Default value = 1.0)
        dur : int, float
            duration in second (Default value = 1.0)
        width : float
            tooth width (Default value = 1.0)
        sr : int
            sampling rate (Default value = 44100)
        channels : int
            number of channels (Default value = 1)
        cn : list of string
            channel names as a list. The size needs to match the number of channels (Default value = None)
        label : string
            identifier of the object (Default value = "sawtooth")

        Returns
        -------
        Asig object
        """
        length = get_length(dur, sr)
        sig = amp * signal.sawtooth(2 * np.pi * freq * np.linspace(0, dur, length, endpoint=False),
                                    width=width)
        if channels > 1:
            sig = np.repeat(sig, channels)
            sig = sig.reshape((length, channels))
        return Asig(sig, sr=sr, label=label, channels=channels, cn=cn)

    def noise(self, type="white", amp=1.0, dur=1.0, sr=44100, channels=1, cn=None, label="noise"):
        """Generate noise signal Asig object. 

        Parameters
        ----------
        type : string
            type of noise, currently available: 'white' and 'pink' (Default value = 'white')
        amp : int, float
            signal amplitude (Default value = 1.0)
        dur : int, float
            duration in second (Default value = 1.0)
        sr : int
            sampling rate (Default value = 44100)
        channels : int
            number of channels (Default value = 1)
        cn : list of string
            channel names as a list. The size needs to match the number of channels (Default value = None)
        label : string
            identifier of the object (Default value = "square")

        Returns
        -------
        Asig object
        """
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
