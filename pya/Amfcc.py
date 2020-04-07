from . import Astft
from . import Asig
import numpy as np
from .helper import shift_bit_length, preemphasis, signal_to_frame, round_half_up, magspec
from .helper import mel2hz, hz2mel, get_filterbanks, lifter
import logging
from scipy.signal import get_window
from scipy.fftpack import dct

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
                 noverlap=None, nfft=512, ncep=13, ceplifter=22, append_energy=True, detrend=False, return_onesided=True,
                 boundary='zeros', padded=True, axis=-1, cn=None):
        """Parameter needed:

        x : signal
        sr : sampling rate
        nperseg : window length per frame.
        noverlap : number of overlap perframe
        nfft : number of fft.

        features : numpy.ndarray
            MFCC feature array

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

        self.nfft = nfft  # default to be 512 Todo change the default to the next pow 2 of nperseg.

        self.window = get_window(window, self.nperseg)
        self.ncep = ncep  # Number of cepstrum
        self.ceplifter  # Lifter's cepstral coefficient

        # Framing signal.
        self.frames = signal_to_frame(self.x, self.nperseg, self.nperseg - self.noverlap, self.window)

        # Computer power spectrum
        self.mspec = magspec(self.frames, self.nfft)  # Magnitude of spectrum, rfft then np.abs()
        self.pspec = 1.0 / self.nfft * np.square(self.mspec(self.frames, self.nfft))  # Power spectrum

        # Total energy of each frame based on the power spectrum
        self.frame_energy = np.sum(self.pspec, 1)
        # Replace 0 with the smallest float positive number
        self.frame_energy = np.where(self.frame_energy==0, np.finfo(float).eps, self.frame_energy)
        self.filter_banks = get_filterbanks(self.sr, nfilt=20, nfft=self.nfft)  # Use the default filter banks.

        # filter bank energies are the features.
        self.fb_energy = np.dot(self.pspec, self.filter_banks.T)
        self.fb_energy = np.where(self.fb_energy==0, np.finfo(float).eps, self.fb_energy)
        self.features = dct(self.fb_energy, type=2, axis=1, norm='ortho')[:, :self.ncep]  # Discrete cosine transform

        # Liftering operation is similar to filtering operation in the frequency domain
        # where a desired quefrency region for analysis is selected by multiplying the whole cepstrum
        # by a rectangular window at the desired position.
        # There are two types of liftering performed, low-time liftering and high-time liftering.
        # Low-time liftering operation is performed to extract the vocal tract characteristics in the quefrency domain
        # and high-time liftering is performed to get the excitation characteristics of the analysis speech frame.

        self.features = lifter(self.features, self.ceplifter)

        # Replace first cepstral coefficient with log of frame energy
        if append_energy:
            self.features[:, 0] = np.log(self.frame_energy)



