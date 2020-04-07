from . import Astft
from . import Asig
import numpy as np
from .helper import shift_bit_length, preemphasis, signal_to_frame, round_half_up, magspec
from .helper import mel2hz, hz2mel
import logging
from scipy.signal import get_window

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class Amfcc:
    """Mel filtered Fourier spectrum (MFCC) class
    
    Steps of mfcc:
        Frame the signal into short frames.
        For each frame calculate the periodogram estimate of the power spectrum.
        Apply the mel filterbank to the power spectra, sum the energy in each filter.
        Take the logarithm of all filterbank energies.
        Take the DCT of the log filterbank energies.
        Keep DCT coefficients 2-13, discard the rest.
        
    Attributes
    ----------
    x : Asig or numpy.ndarray
        x can be two forms, the most commonly used is an Asig object. 
        Such as directly acquired from an Asig object via Asig.to_stft().
    sr : int
        sampling rate, this is only necessary if x is not Asig. (Default value = None)
    label : str
        name of the Asig. (Default value = None)
    window : str
        type of the window function (Default value = 'hann')
    nperseg : int
        number of samples per stft segment (Default value = '256')
    noverlap : int
        number of samples to overlap between segments (Default value = None)
    detrend : str or function or bool
        Specifies how to detrend each segment. If detrend is a string, 
        it is passed as the type argument to the detrend function. If it is a function, 
        it takes a segment and returns a detrended segment. If detrend is False, 
        no detrending is done. (Default value = False).
    return_onesided : bool
        If True, return a one-sided spectrum for real data. If False return a two-sided spectrum. 
        Defaults to True, but for complex data, a two-sided spectrum is always returned. (Default value = True)
    boundary : str or None
        Specifies whether the input signal is extended at both ends, and how to generate the new values, 
        in order to center the first windowed segment on the first input point. 
        This has the benefit of enabling reconstruction of the first input point 
        when the employed window function starts at zero. 
        Valid options are ['even', 'odd', 'constant', 'zeros', None]. Defaults to ‘zeros’, 
        for zero padding extension. I.e. [1, 2, 3, 4] is extended to [0, 1, 2, 3, 4, 0] for nperseg=3. (Default value = 'zeros')
    padded : bool
        Specifies whether the input signal is zero-padded at the end to make the signal fit exactly into 
        an integer number of window segments, so that all of the signal is included in the output. 
        Defaults to True. Padding occurs after boundary extension, if boundary is not None, and padded is True, 
        as is the default. (Default value = True)
    axis : int
        Axis along which the STFT is computed; the default is over the last axis. (Default value = -1)
    cn : list or None
        Channel names of the Asig, this will be used for the Astft for consistency. (Default value = None)
    """

    def __init__(self, x, sr=None, label=None, nmfcc=20, window='hann', nperseg=256,
                 noverlap=None, nfft=512, detrend=False, return_onesided=True,
                 boundary='zeros', padded=True, axis=-1, cn=None):
        """Parameter needed:

        x : signal
        sr : sampling rate
        nperseg : window length per frame.
        noverlap : number of overlap perframe
        nfft : number of fft.


        """
        # First prepare for parameters
        if isinstance(x, Asig.Asig):
            self.sr = x.sr
            self.x = x.sig
        elif isinstance(x, np.ndarray):
            self.x = x
            if sr:
                self.sr = sr
            else:
                raise AttributeError("If x is an array, sr (sampling rate) needs to be defined.")
        else:
            raise TypeError("x can only be either a numpy.ndarray or pya.Asig object.")

        if not nperseg:  # More likely to set it as default.
            self.nperseg = round_half_up(sr * 0.025)  # 25ms length window,
        else:
            self.nperseg = nperseg

        if not noverlap:  # More likely to set it as default
            self.noverlap = round_half_up(sr * 0.01)  # 10ms overlap
        else:
            self.noverlap

        if self.noverlap > self.nperseg:
            raise _LOGGER.warning("noverlap great than nperseg, this leaves gaps between frames.")

        self.nfft = nfft  # default to be 512

        self.window = get_window(window, self.nperseg)
        self.frames = signal_to_frame(self.x, self.nperseg, self.nperseg - self.noverlap, self.window)
        self.mspec = magspec(self.frames, self.nfft)  # Magnitude of spectrum
        self.pspec = 1.0 / self.nfft * np.square(self.mspec(self.frames, self.nfft))  # Power spectrum

        self.update_filterbanks()  # Use the default filter banks.

    def update_filterbanks(self, nfilt=20, lowfreq=0, highfreq=None):
        """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
        to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
        :param nfilt: the number of filters in the filterbank, default 20.
        :param nfft: the FFT size. Default is 512.
        :param samplerate: the sample rate of the signal we are working with, in Hz. Affects mel spacing.
        :param lowfreq: lowest band edge of mel filters, default 0 Hz
        :param highfreq: highest band edge of mel filters, default samplerate/2
        :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
        """
        highfreq = highfreq or self.sr // 2
        if highfreq > self.sr:
            raise AttributeError("Upper frequency band edge should not exceed the nyquist frequency")            assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

        # compute points evenly spaced in mels
        lowmel = hz2mel(lowfreq)
        highmel = hz2mel(highfreq)
        melpoints = np.linspace(lowmel, highmel, nfilt + 2)
        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = np.floor((nfft + 1) * mel2hz(melpoints) / samplerate)

        self.filter_banks = np.zeros([nfilt, nfft // 2 + 1])
        for j in range(0, nfilt):
            for i in range(int(bin[j]), int(bin[j + 1])):
                self.filter_banks[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
            for i in range(int(bin[j + 1]), int(bin[j + 2])):
                self.filter_banks[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
        return self
