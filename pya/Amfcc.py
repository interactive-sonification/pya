import numpy as np
from .helper import next_pow2, preemphasis, signal_to_frame, round_half_up, magspec
from .helper import mel2hz, hz2mel, get_filterbanks, lifter
import logging
from scipy.signal import get_window
from scipy.fftpack import dct
from . import Asig

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class Amfcc:
    """Mel filtered Fourier spectrum (MFCC) class

    Steps of mfcc:
        Frame the signal into short frames.
        For each frame calculate the periodogram estimate of the power spectrum.
        Apply the mel filterbank to the power spectra, sum the energy in each filter.
        Take the DCT of the log filterbank energies.
        Keep DCT coefficients 2-13, discard the rest.
        Take the logarithm of all filterbank energies.

    Attributes
    ----------
    x : Asig or numpy.ndarray
        x can be two forms, the most commonly used is an Asig object. 
        Such as directly acquired from an Asig object via Asig.to_stft().
    sr : int
        sampling rate, this is only necessary if x is not Asig. (Default value = None)
    label : str
        name of the Asig. (Default value = None)
    n_per_frame : int
        number of samples per frame (Default value = '256')
    noverlap : int
        number of samples to overlap between frames (Default value = None)
    window : str
        type of the window function (Default value='hann'), use scipy.signal.get_window to return a numpy array.
        If None, np.ones() with the according nperseg size will return  will return.

    cn : list or None
        Channel names of the Asig, this will be used for the Astft for consistency. (Default value = None)
    """

    def __init__(self, x, sr=None, label='mfcc', nmfcc=20, window='hann', n_per_frame=256,
                 noverlap=None, nfft=512, ncep=13, ceplifter=22, append_energy=True, cn=None):
        """Parameter needed:

        x : signal
        sr : sampling rate
        nperseg : window length per frame.
        noverlap : number of overlap perframe
        nfft : number of fft.

        features : numpy.ndarray
            MFCC feature array

        """
        # ----------Prepare attributes -------------------------
        # First prepare for parameters
        if type(x) == Asig:
            self.sr = x.sr
            self.x = x.sig
            self.label = x.label + '_' + label
        elif isinstance(x, np.ndarray):
            self.x = x
            if sr:
                self.sr = sr
            else:
                raise AttributeError("If x is an array, sr (sampling rate) needs to be defined.")
            self.label = label
        else:
            raise TypeError("x can only be either a numpy.ndarray or pya.Asig object.")

        if not n_per_frame:  # More likely to set it as default.
            self.n_per_frame = round_half_up(self.sr * 0.025)  # 25ms length window,
        else:
            self.n_per_frame = n_per_frame

        if not noverlap:  # More likely to set it as default
            self.noverlap = round_half_up(self.sr * 0.01)  # 10ms overlap
        else:
            self.noverlap

        if self.noverlap > self.n_per_frame:
            raise _LOGGER.warning("noverlap great than nperseg, this leaves gaps between frames.")

        self.nfft = nfft  # default to be 512 Todo change the default to the next pow 2 of nperseg.

        self.window = get_window(window, self.n_per_frame)
        self.ncep = ncep  # Number of cepstrum
        self.ceplifter = ceplifter  # Lifter's cepstral coefficient
        # -------------------------------------------------

        # Framing signal.
        self.frames = signal_to_frame(self.x, self.n_per_frame, self.n_per_frame - self.noverlap, self.window)

        # Computer power spectrum
        self.mspec = magspec(self.frames, self.nfft)  # Magnitude of spectrum, rfft then np.abs()
        self.pspec = 1.0 / self.nfft * np.square(self.mspec)  # Power spectrum

        # Total energy of each frame based on the power spectrum
        self.frame_energy = np.sum(self.pspec, 1)
        # Replace 0 with the smallest float positive number
        self.frame_energy = np.where(self.frame_energy == 0, np.finfo(float).eps, self.frame_energy)
        self.filter_banks = get_filterbanks(self.sr, nfilt=20, nfft=self.nfft)  # Use the default filter banks.

        # filter bank energies are the features.
        self.fb_energy = np.dot(self.pspec, self.filter_banks.T)
        self.fb_energy = np.where(self.fb_energy == 0, np.finfo(float).eps, self.fb_energy)

        #  Keep DCT coefficients 2-13, discard the rest.
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

    def __repr__(self):
        # ToDO add more info to msg
        return f"Amfcc label {self.label}, sr {self.sr}"

    # def plot(self):
    #     # Need to know what is to plot. How to arrange plot.