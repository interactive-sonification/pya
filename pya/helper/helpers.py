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
    """Pad signal with certain width, support 1-3D tensors. Use it to add silence to a signal

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


def shift_bit_length(x):
    if x < 0:
        raise AttributeError("x needs to be positive integer.")
    return 1<<(x-1).bit_length()


# def _spectrogram(x=None, S=None, n_fft=2048, hop_length=512, power=1,
#                  win_length=None, window='hann', center=True, pad_mode='reflect'):
#     '''Helper function to retrieve a magnitude spectrogram.
#     This is primarily used in feature extraction functions that can operate on
#     either audio time-series or spectrogram input.
#     Parameters
#     ----------
#     y : None or np.ndarray [ndim=1]
#         If provided, an audio time series
#
#     n_fft : int > 0
#         STFT window size
#     hop_length : int > 0
#         STFT hop length
#     power : float > 0
#         Exponent for the magnitude spectrogram,
#         e.g., 1 for energy, 2 for power, etc.
#     win_length : int <= n_fft [scalar]
#         Each frame of audio is windowed by `window()`.
#         The window will be of length `win_length` and then padded
#         with zeros to match `n_fft`.
#         If unspecified, defaults to ``win_length = n_fft``.
#     window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
#         - a window specification (string, tuple, or number);
#           see `scipy.signal.get_window`
#         - a window function, such as `scipy.signal.hanning`
#         - a vector or array of length `n_fft`
#         .. see also:: `filters.get_window`
#     center : boolean
#         - If `True`, the signal `y` is padded so that frame
#           `t` is centered at `y[t * hop_length]`.
#         - If `False`, then frame `t` begins at `y[t * hop_length]`
#     pad_mode : string
#         If `center=True`, the padding mode to use at the edges of the signal.
#         By default, STFT uses reflection padding.
#
#     Returns
#     -------
#     spectro : numpy.ndarray
#         - If `S` is provided as input, then `S_out == S`
#         - Else, `S_out = |stft(y, ...)|**power`
#     n_fft : int > 0
#         - If `S` is provided, then `n_fft` is inferred from `S`
#         - Else, copied from input
#     '''
#
#
#     # Otherwise, compute a magnitude spectrogram from input
#     S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length,
#                     win_length=win_length, center=center,
#                     window=window, pad_mode=pad_mode))**power
#
#     return spectro, n_fft


def preemphasis(x, coeff=0.95):
    """Pre-emphasis filter to whiten the spectrum.
    Pre-emphasis is a way of compensating for the rapid decaying spectrum of speech.
    Can often skip this step in the cases of music for example
    """
    return np.append(x[0], x[1:] - coeff * x[:-1])


def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if np.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            np.shape(frames)[1], NFFT)
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)


def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0 / NFFT * np.square(magspec(frames, NFFT))


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def rolling_window(a, window, step=1):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def signal_to_frame(sig, nframe, frame_step, window=None, stride_trick=True):
    """Frame a signal into overlapping frames.

    Parameters
    ----------
    sig : numpy.ndarray
        the audio signal
    frame_len : int
        number of samples each frame
    frame_step : int
        number of samples after the start of the previous frame that the next frame should begin.
    window : numpy.ndarray or None
        a window array, e.g,
    stride_trick : bool
        use stride trick to compute the rolling window and window multiplication faster

    Returns
    -------
    _ : numpy.ndarray
        an array of frames.
    """
    slen = len(sig)
    nframe = int(round_half_up(nframe))
    frame_step = int(round_half_up(frame_step))
    if slen <= nframe:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - nframe) / frame_step))

    padlen = int((numframes - 1) * frame_step + nframe)
    padsignal = np.concatenate((sig, np.zeros((padlen - slen,))))  # Pad zeros to signal

    if stride_trick:
        win = window if window else np.ones(nframe)
        frames = rolling_window(padsignal, window=nframe, step=frame_step)
    else:
        indices = np.tile(np.arange(0, nframe), (numframes, 1)) + np.tile(
            np.arange(0, numframes * frame_step, frame_step), (nframe, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = padsignal[indices]
        win = window if window else np.ones(nframe)
        win = np.tile(win, (numframes, 1))
    return frames * win


def magspec(frames, nfft):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).

    Parameters
    ----------
    frames : numpy.ndarray
        Framed signal array, each row is a frame.
    nfft : int
        fft size.

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if np.shape(frames)[1] > nfft:
        logging.warning('frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
                        np.shape(frames)[1], nfft)
    complex_spec = np.fft.rfft(frames, nfft)

    return np.absolute(complex_spec)

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

def get_filterbanks(self, sr, nfilt=20, nfft=512,  lowfreq=0, highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    Parameters
    ----------
    nfilt : the number of filters in the filterbank, default 20.
    nfft :

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the sample rate of the signal we are working with, in Hz. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or sr // 2

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft + 1) * mel2hz(melpoints) / sr)

    filter_banks = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bin[j]), int(bin[j + 1])):
            filter_banks[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            filter_banks[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
    return filter_banks


def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.
    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes, ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L / 2.) * np.sin(np.pi * n / L)
        return lift * cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra