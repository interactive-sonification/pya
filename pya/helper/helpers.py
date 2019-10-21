# Collection of small helper functions
import numpy as np
import pyaudio
from scipy.fftpack import fft
from .codec import audio_read


class _error(Exception):    
    pass


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


# def timeit(method):
#     """Decorator to time methods, print out the time for executing the method"""
#     def timed(*args, **kw):
#         ts = time.time()
#         result = method(*args, **kw)
#         te = time.time()
#         if 'log_time' in kw:
#             name = kw.get('log_name', method.__name__.upper())
#             kw['log_time'][name] = int((te - ts) * 1000)
#         else:
#             print('%r  %2.2f ms' %
#                   (method.__name__, (te - ts) * 1000))
#         return result
#     return timed


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


def normalize(d):
    """Return the normalized input array"""
    # d is a (n x dimension) np array
    d -= np.min(d, axis=0)
    d /= np.ptp(d, axis=0)
    return d


def audio_from_file(path, dtype=np.float32):
    '''Load an audio buffer using audioread.
    This loads one block at a time, and then concatenates the results.
    '''
    y = []  # audio array
    with audio_read(path) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels
        s_start = 0
        s_end = np.inf
        n = 0
        for frame in input_file:
            frame = buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)
            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]
            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels))
    else:
        y = np.empty(0, dtype=dtype)
        sr_native = 0
    return y, sr_native


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.
    See Also
    --------
    buf_to_float
    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer
    n_bytes : int [1, 2, 4]
        The number of bytes per sample in `x`
    dtype : numeric type
        The target output type (default: 32-bit float)
    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """
    # Invert the scale of the data
    scale = 1. / float(1 << ((8 * n_bytes) - 1))
    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)
    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


def device_info():
    """Return a formatted string about available audio devices and their info"""
    pa = pyaudio.PyAudio()
    line1 = (f"idx {'Device Name':25}{'INP':4}{'OUT':4}   SR   INP-(Lo|Hi)  OUT-(Lo/Hi) (Latency in ms)")
    devs = [pa.get_device_info_by_index(i) for i in range(pa.get_device_count())]
    lines = [line1]
    for i, d in enumerate(devs):
        p1 = f"{i:<4g}{d['name'].strip():24}{d['maxInputChannels']:4}{d['maxOutputChannels']:4}"
        p2 = f" {int(d['defaultSampleRate'])} "
        p3 = f"{d['defaultLowInputLatency']*1000:6.2g} {d['defaultHighInputLatency']*1000:6.0f}"
        p4 = f"{d['defaultLowOutputLatency']*1000:6.2g} {d['defaultHighOutputLatency']*1000:6.0f}"
        lines.append(p1 + p2 + p3 + p4)
    print(*lines, sep='\n')


def find_device(min_input=0, min_output=0):
    pa = pyaudio.PyAudio()
    res = []
    for idx in range(pa.get_device_count()):
        dev = pa.get_device_info_by_index(idx)
        if dev['maxInputChannels'] >= min_input and dev['maxOutputChannels'] >= min_output:
            res.append(dev)
    return res
