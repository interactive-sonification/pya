# helpers.py
#Collection of small helper functions
import numpy as np
import pyaudio
import time
from scipy.fftpack import fft


class _error(Exception):    
    pass


def record(dur=2, channels=1, rate=44100, chunk=256):
    """Record audio

    Parameters
    ----------
    dur : int
        Duration (Default value = 2)
    channels : int
        Number of channels (Default value = 1)
    rate : int
        Audio sample rate (Default value = 44100)
    chunk :
         (Default value = 256)

    Returns
    -------
    _ : numpy.ndarray
        numpy array of the recorded audio signal.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True,
                    output=True, frames_per_buffer=chunk)
    buflist = []
    for _ in range(0, int(rate / chunk * dur)):
        data = stream.read(chunk)
        buflist.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    return np.frombuffer(b''.join(buflist), dtype=np.int16)

# def play(sig, num_channels=1, sr=44100, block=False):
#     """Plays audio signal via simpleaudio

#     Arguments:
#         sig {iterable} -- Signal to be played
#     Keyword Arguments:
#         num_channels {int} -- Number of channels (default: {1})
#         sr {int} -- Audio sample rate (default: {44100})
#         block {bool} -- if True, block until playback is finished
#                         (default: {False})
#     Returns:
#         simpleaudio play_obj -- see description in simpleaudio
#     """
#     play_obj = sa.play_buffer((32767*sig).astype(np.int16), num_channels=num_channels,
#     bytes_per_sample=2, sample_rate=sr)
#     if block:
#         play_obj.wait_done() # wait for playback to finish before returning
#     return play_obj


def linlin(x, smi, sma, dmi, dma):
    """Linear mapping

    Parameters
    ----------
    x : float
        input value
    smi : float
        input range's minimum
    sma : float
        input range's maximum
    dmi : float
        input range's minimum
    dma :
        
    Returns
    -------
    _ : float
        mapped output
    """
    return (x - smi) / (sma - smi) * (dma - dmi) + dmi


def midicps(m):
    """Convert midi number into cycle per second"""
    return 440.0 * 2 ** ((m - 69) / 12.0)


def cpsmidi(c):
    """Convert cycle per second into midi number"""
    return 69 + 12 * np.log2(c / 440.0)


def clip(value, minimum=-float("inf"), maximum=float("inf")):
    """Signal hard clipping"""
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def dbamp(db):
    """Convert db to amplitude"""
    return 10 ** (db / 20.0)


def ampdb(amp):
    """Convert amplitude to db"""
    return 20 * np.log10(amp)


def timeit(method):
    """Decorator to time methods, print out the time for executing the method"""
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' %
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


def spectrum(sig, samples, channels, sr):
    """Return spectrum of a given signal. This method return spectrum matrix if input signal is multi-channels.

    Parameters
    ----------
    sig : numpy.ndarray
        signal array
    samples : int
        total amount of samples
    channels : int
        signal channels
    sr : int
        sampling rate

    Returns
    ---------
    frq : numpy.ndarray
        frequencies
    Y : numpy.ndarray
        FFT of the signal.
    """
    nrfreqs = samples // 2 + 1
    frq = np.linspace(0, 0.5 * sr, nrfreqs)  # one sides frequency range
    if channels == 1:
        Y = fft(sig)[:nrfreqs]  # / self.samples
    else:
        Y = np.array(np.zeros((nrfreqs, channels)), dtype=complex)
        for i in range(channels):
            Y[:, i] = fft(sig[:, i])[:nrfreqs]
    return frq, Y


def get_length(dur, sr):
    """Return total number of samples based on duration in second (dur) and sampling rate (sr)"""
    if isinstance(dur, float):
        length = int(dur * sr)
    elif isinstance(dur, int):
        length = dur
    else:
        raise TypeError("Unrecognise type for dur, int (samples) or float (seconds) only")
    return length


def normalize(d):
    """Return the normalized input array"""
    # d is a (n x dimension) np array
    d -= np.min(d, axis=0)
    d /= np.ptp(d, axis=0)
    return d
