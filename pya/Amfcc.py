import numpy as np
from .helper import next_pow2, preemphasis, signal_to_frame, round_half_up, magspec
from .helper import mel2hz, hz2mel, mel_filterbanks, lifter
import logging
from scipy.signal import get_window
from scipy.fftpack import dct
from . import Asig
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        number of samples per frame, default is the sample equivalent for 25ms
    hopsize : int
        number of samples of each successive frame, the overlap amount is n_per_frame - hopsize. Default is
        the sample equivalent of 10ms.
    nfft : int
        FFT size, default to be next power of 2 integer of n_per_frame
    window : str
        type of the window function (Default value='hann'), use scipy.signal.get_window to return a numpy array.
        If None, np.ones() with the according nperseg size will return  will return.
    ncep : int
        Number of cepstrum, default 13
    cepliter : int
        Lifter's cepstral coefficient, default 22
    frames : numpy.ndarray
        The original signal being reshape into frame based on n_per_frame and hopsize.
    """

    def __init__(self, x, sr=None, label='mfcc', n_per_frame=None,
                 hopsize=None, nfft=None, window='hann', nfilters=26, ncep=13, ceplifter=22,
                 preemph=0.95, append_energy=True, cn=None):
        """Parameter needed:

        x : signal
        sr : sampling rate
        nperseg : window length per frame.
        hopsize : number of overlap perframe
        nfft : number of fft.

        features : numpy.ndarray
            MFCC feature array

        """
        # ----------Prepare attributes ------------`-------------
        # First prepare for parameters
        if type(x) == Asig:
            self.sr = x.sr
            self.x = x.sig
            self.label = x.label + '_' + label
            self.duration = x.get_duration()

        elif isinstance(x, np.ndarray):
            self.x = x
            if sr:
                self.sr = sr
            else:
                raise AttributeError("If x is an array, sr (sampling rate) needs to be defined.")
            self.duration = np.shape(x)[0] / self.sr
            self.label = label
        else:
            raise TypeError("x can only be either a numpy.ndarray or pya.Asig object.")

        self.n_per_frame = n_per_frame or int(round_half_up(self.sr * 0.025))  # 25ms length window,


        self.hopsize = hopsize or int(round_half_up(self.sr * 0.01))  # default 10ms overlap
        if self.hopsize > self.n_per_frame:
            raise _LOGGER.warning("noverlap great than nperseg, this leaves gaps between frames.")
        self.nfft = nfft or next_pow2(self.n_per_frame)

        self.window = get_window(window, self.n_per_frame) if window else np.ones((self.n_per_frame,))
        self.nfilters = nfilters
        self.ncep = ncep  # Number of cepstrum
        self.ceplifter = ceplifter  # Lifter's cepstral coefficient
        # -------------------------------------------------

        # Framing signal.
        pre_emp_sig = preemphasis(self.x, coeff=preemph)
        self.frames = signal_to_frame(pre_emp_sig, self.n_per_frame, self.hopsize, self.window)

        # Computer power spectrum
        mspec = magspec(self.frames, self.nfft)  # Magnitude of spectrum, rfft then np.abs()
        pspec = 1.0 / self.nfft * np.square(mspec)  # Power spectrum

        # Total energy of each frame based on the power spectrum
        self.frame_energy = np.sum(pspec, 1)
        # Replace 0 with the smallest float positive number
        self.frame_energy = np.where(self.frame_energy == 0, np.finfo(float).eps, self.frame_energy)

        # Prepare Mel filter
        self.filter_banks = mel_filterbanks(self.sr, nfilters=self.nfilters, nfft=self.nfft)  # Use the default filter banks.

        # filter bank energies are the features.
        self.features = np.dot(pspec, self.filter_banks.T)
        self.features = np.where(self.features == 0, np.finfo(float).eps, self.features)
        self.features = np.log(self.features)

        #  Keep DCT coefficients 2-13, discard the rest.
        self.features = dct(self.features, type=2, axis=1, norm='ortho')[:, :self.ncep]  # Discrete cosine transform

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

    @property
    def nframes(self):
        return self.frames.shape[0]

    @property
    def timestamp(self):
        return np.linspace(0, self.duration, self.nframes)

    def __repr__(self):
        # ToDO add more info to msg
        return f"Amfcc object, label {self.label}, sr {self.sr}"

    def plot(self, cmap='inferno', corlorbar=True, colorbar_alignment='right',
             x_as_time='True', nxlabel=8, **kwargs):
        """Plot Amfcc.features via matshow, x is frames/time, y is the MFCCs

        Parameters
        ----------

        """
        plt.figure()
        ax = plt.gca()
        im = ax.matshow(self.features.T, cmap=plt.get_cmap(cmap), origin='lower', **kwargs)
        xticks = np.linspace(0, self.nframes, nxlabel, dtype=int)
        ax.set_xticks(xticks)
        if x_as_time:
            xlabels = np.round(np.linspace(0, self.duration, nxlabel), decimals=1)
            # Replace x ticks with timestamps
            ax.set_xticklabels(xlabels)
            ax.xaxis.tick_bottom()
        if corlorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(colorbar_alignment, size="2%", pad=0.03)
            _ = plt.colorbar(im, cax=cax)   # Add
        return self  # ToDo maybe return the axis instead.