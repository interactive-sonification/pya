from . import Asig
import numpy as np
from scipy import signal
from .helper import normalize


def get_num_of_rows(dur, n_rows, sr):
    """Return total number of samples. If dur is set, return dur*sr, if num_samples is set, return num_samples,
    if both set, raise an AttributeError. Only use one of the two.
    """
    if dur and n_rows is None:
        return int(dur * sr)
    elif dur is None and n_rows is None:
        return int(sr)
    elif n_rows and dur is None:
        return int(n_rows)
    else:
        raise AttributeError("Only use either dur or n_rows to specify the number of rows of the signal.")


class Ugen(Asig):
    """Unit Generator for to create Asig with predefined signal

    Currently avaiable:
        sine, cos, square, sawtooth, noise

    Examples
    --------
    Create common waveform in Asig.

    >>> from pya import Ugen
    >>> # Create a sine wave of 440Hz at 44100Hz sr for 2 seconds. Same for cos()
    >>> sine = Ugen().sine(freq=440, amp=0.8, dur=2.,label="sine")
    >>> # Create a square wave of 25Hz, 2000 samples at 100 sr, stereo.
    >>> sq = Ugen().square(freq=25, n_rows=2000, sr=100, channels=2, cn=['l', 'r'])
    >>> # Make a white noise, another option is 'pink', at 44100Hz for 1second.
    >>> noi = Ugen().noise(type='white')
    """
    def __init__(self):
        pass

    def sine(self, freq=440, amp=1.0, dur=None, n_rows=None,
             sr=44100, channels=1, cn=None, label="sine"):
        """Generate Sine signal Asig object.

        Parameters
        ----------
        freq : int, float
            signal frequency (Default value = 440)
        amp : int, float
            signal amplitude (Default value = 1.0)
        dur : float
            duration in second. dur and num_rows only use one of the two. (Default value = 1.0)
        num_rows : int
            number of rows (samples). dur and num_rows only use one of the two(Default value = None)
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
        Asig
        """
        length = get_num_of_rows(dur, n_rows, sr)
        sig = amp * np.sin(2 * np.pi * freq * np.linspace(0, length / sr, length, endpoint=False))
        if channels > 1:
            sig = np.repeat(sig, channels)
            sig = sig.reshape((length, channels))
        return Asig(sig, sr=sr, label=label, channels=channels, cn=cn)

    def cos(self, freq=440, amp=1.0, dur=None, n_rows=None,
            sr=44100, channels=1, cn=None, label="cosine"):
        """Generate Cosine signal Asig object.

        Parameters
        ----------
        freq : int, float
            signal frequency (Default value = 440)
        amp : int, float
            signal amplitude (Default value = 1.0)
        dur : int, float
            duration in second. dur and num_rows only use one of the two. (Default value = 1.0)
        num_rows : int
            number of rows (samples). dur and num_rows only use one of the two(Default value = None)
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
        Asig
        """
        length = get_num_of_rows(dur, n_rows, sr)
        sig = amp * np.cos(2 * np.pi * freq * np.linspace(0, length / sr, length, endpoint=False))
        if channels > 1:
            sig = np.repeat(sig, channels)
            sig = sig.reshape((length, channels))
        return Asig(sig, sr=sr, label=label, channels=channels, cn=cn)

    def square(self, freq=440, amp=1.0, dur=None, n_rows=None,
               duty=0.5, sr=44100, sample_shift=0.5,
               channels=1, cn=None, label="square"):
        """Generate square wave signal Asig object.

        Parameters
        ----------
        freq : int, float
            signal frequency (Default value = 440)
        amp : int, float
            signal amplitude (Default value = 1.0)
        dur : int, float
            duration in second. dur and num_rows only use one of the two. (Default value = 1.0)
        num_rows : int
            number of row (samples). dur and num_rows only use one of the two(Default value = None)
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
        Asig
        """
        length = get_num_of_rows(dur, n_rows, sr)
        sig = amp * signal.square(
            2 * np.pi * freq * ((sample_shift / length) + np.linspace(0, length / sr, length, endpoint=False)),
            duty=duty)
        if channels > 1:
            sig = np.repeat(sig, channels)
            sig = sig.reshape((length, channels))
        return Asig(sig, sr=sr, label=label, channels=channels, cn=cn)

    def sawtooth(self, freq=440, amp=1.0, dur=None, n_rows=None,
                 width=1., sr=44100, channels=1, cn=None, label="sawtooth"):
        """Generate sawtooth wave signal Asig object.

        Parameters
        ----------
        freq : int, float
            signal frequency (Default value = 440)
        amp : int, float
            signal amplitude (Default value = 1.0)
        dur : int, float
            duration in second. dur and num_rows only use one of the two. (Default value = 1.0)
        num_rows : int
            number of rows (samples). dur and num_rows only use one of the two(Default value = None)
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
        Asig
        """
        length = get_num_of_rows(dur, n_rows, sr)
        sig = amp * signal.sawtooth(2 * np.pi * freq * np.linspace(0, length / sr, length, endpoint=False),
                                    width=width)
        if channels > 1:
            sig = np.repeat(sig, channels)
            sig = sig.reshape((length, channels))
        return Asig(sig, sr=sr, label=label, channels=channels, cn=cn)

    def noise(self, type="white", amp=1.0, dur=None, n_rows=None,
              sr=44100, channels=1, cn=None, label="noise"):
        """Generate noise signal Asig object.

        Parameters
        ----------
        type : string
            type of noise, currently available: 'white' and 'pink' (Default value = 'white')
        amp : int, float
            signal amplitude (Default value = 1.0)
        dur : int, float
            duration in second. dur and num_rows only use one of the two. (Default value = 1.0)
        num_rows : int
            number of rows (samples). dur and num_rows only use one of the two(Default value = None)
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
        Asig
        """
        length = get_num_of_rows(dur, n_rows, sr)
        # Question is that will be that be too slow.]
        if type == "white" or type == "white_noise":
            sig = (np.random.rand(length) - 0.5) * 2. * amp

        elif type == "pink" or type == "pink_noise":
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
