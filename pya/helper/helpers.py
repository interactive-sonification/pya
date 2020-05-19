# Collection of small helper functions
import numpy as np
import pyaudio
from scipy.fftpack import fft
from .codec import audio_read
import logging
import decimal
import math


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


def dbamp(db):
    """Convert db to amplitude"""
    return 10 ** (db / 20.0)


def ampdb(amp):
    """Convert amplitude to db"""
    return 20 * np.log10(amp)


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
    return devs


def find_device(min_input=0, min_output=0):
    pa = pyaudio.PyAudio()
    res = []
    for idx in range(pa.get_device_count()):
        dev = pa.get_device_info_by_index(idx)
        if dev['maxInputChannels'] >= min_input and dev['maxOutputChannels'] >= min_output:
            res.append(dev)
    return res


def padding(x, width, tail=True, constant_values=0):
    """Pad signal with certain width, support 1-3D tensors. 
    Use it to add silence to a signal
    TODO: CHECK pad array


    Parameters
    ----------
    x : np.ndarray
        A numpy array
    width : int
        The amount of padding. 
    tail : bool
        If true pad to the tail, else pad to the start.
    constant_values : int or float or None
        The value to be padded, add None will pad nan to the array

    Returns
    -------
    _ : np.ndarray
        Padded array
    """
    pad = (0, width) if tail else (width, 0)
    if x.ndim == 1:
        return np.pad(x, (pad), mode='constant', constant_values=constant_values)
    elif x.ndim == 2:
        return np.pad(x, (pad, (0, 0)), mode='constant', constant_values=constant_values)
    elif x.ndim == 3:
        return np.pad(x, ((0, 0), pad, (0, 0)), mode='constant', constant_values=constant_values)
    else:
        raise AttributeError("only support ndim 1 or 2, 3. For higher please just use np.pad ")


def is_pow2(val):
    """Check if input is a power of 2 return a bool result."""
    return False if val <= 0 else math.log(val, 2).is_integer()


def next_pow2(x):
    """Find the closest pow of 2 that is great or equal or x, 
    based on shift_bit_length

    Parameters
    ----------
    x : int
        A positive number

    Returns
    -------
    _ : int
        The cloest  integer that is greater or equal to input x.
    """
    if x < 0:
        raise AttributeError("x needs to be positive integer.")
    return 1 << (x - 1).bit_length()


def round_half_up(number):
    """Round up if >= .5"""
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def rolling_window(a, window, step=1):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def signal_to_frame(sig, n_per_frame, frame_step, window=None, stride_trick=True):
    """Frame a signal into overlapping frames.

    Parameters
    ----------
    sig : numpy.ndarray
        The audio signal
    n_per_frame : int
        Number of samples each frame
    frame_step : int
        Number of samples after the start of the previous frame that the next frame should begin.
    window : numpy.ndarray or None
        A window array, e.g,
    stride_trick : bool
        Use stride trick to compute the rolling window and window multiplication faster

    Returns
    -------
    _ : numpy.ndarray
        an array of frames.
    """
    slen = len(sig)
    n_per_frame = int(round_half_up(n_per_frame))
    frame_step = int(round_half_up(frame_step))
    if slen <= n_per_frame:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - n_per_frame) / frame_step))
    padlen = int((numframes - 1) * frame_step + n_per_frame)
    padsignal = np.concatenate((sig, np.zeros((padlen - slen,))))  # Pad zeros to signal

    if stride_trick:
        if window is not None:
            win = window
        else:
            win = np.ones(n_per_frame)
        frames = rolling_window(padsignal, window=n_per_frame, step=frame_step)
    else:
        indices = np.tile(np.arange(0, n_per_frame), (numframes, 1)) + np.tile(
            np.arange(0, numframes * frame_step, frame_step), (n_per_frame, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = padsignal[indices]
        if window is not None:
            win = window
        else:
            win = np.ones(n_per_frame)
        win = np.tile(win, (numframes, 1))
    return frames * win


def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames.
    If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    Parameters
    ----------
    frames : numpy.ndarray
        The framed array, each row is a frame, can be just a single frame.
    NFFT : int
        FFT length. If NFFT > frame_len, the frames are zero_padded.

    Returns
    -------
    _ : numpy.ndarray
        If frames is an NxD matrix, output will be Nx(NFFT/2+1).
        Each row will be the magnitude spectrum of the corresponding frame.
    """
    if np.shape(frames)[1] > NFFT:
        logging.warning(f'frame length {np.shape(frames)[1]} is greater than FFT size {NFFT}, '
                        f'frame will be truncated. Increase NFFT to avoid.')
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.abs(complex_spec)


def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames, 
    first comeputer the magnitude spectrum

    Parameters
    ----------
    frames : numpy.ndarray
        Framed signal, can be just a single frame.
    NFFT : int
        The FFT length to use. If NFFT > frame_len, the frames are zero-padded.

    Returns
    -------
    _ : numpy array
        Power spectrum of the framed signal. 
        Each row has the size of NFFT / 2 + 1 due to rfft.
    """
    return 1.0 / NFFT * np.square(magspec(frames, NFFT))


def hz2mel(hz):
    """Convert a value in Hertz to Mels

    Parameters
    ----------
    hz : number of array
        value in Hz, can be an array

    Returns:
    --------
    _ : number of array
        value in Mels, same type as the input.
    """
    return 2595 * np.log10(1 + hz / 700.)


def mel2hz(mel):
    """Convert a value in Hertz to Mels

    Parameters
    ----------
    hz : number of array
        value in Hz, can be an array

    Returns:
    --------
    _ : number of array
        value in Mels, same type as the input.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)
