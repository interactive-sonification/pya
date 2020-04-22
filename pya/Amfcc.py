import numpy as np
from .helper import next_pow2, preemphasis, signal_to_frame, round_half_up, magspec
from .helper import mel2hz, hz2mel, mel_filterbanks, lifter, is_pow2
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
        - Frame the signal into short frames.
        - For each frame calculate the periodogram estimate of the power spectrum.
        - Apply the mel filterbank to the power spectra, sum the energy in each filter.
        - Take the DCT of the log filterbank energies.
        - Keep DCT coefficients 2-13, discard the rest.
        - Take the logarithm of all filterbank energies.

    Attributes
    ----------
    x : Asig or numpy.ndarray
        x can be two forms, the most commonly used is an Asig object. 
        Such as directly acquired from an Asig object via Asig.to_stft().
    sr : int
        sampling rate, this is only necessary if x is not Asig.
    duration : float
        Duration of the signal in second,
    label : str
        A string label as an identifier.
    n_per_frame : int
        Number of samples per frame
    hopsize : int
        Number of samples of each successive frame.
    nfft : int
        FFT size, default to be next power of 2 integer of n_per_frame
    window : str
        Type of the window function (Default value='hann'), use scipy.signal.get_window to
        return a numpy array. If None, no windowing will be applied.
    nfilters : int
        The number of mel filters. Default is 26
    ncep : int
        Number of cepstrum. Default is 13
    cepliter : int
        Lifter's cepstral coefficient. Default is 22
    frames : numpy.ndarray
        The original signal being reshape into frame based on n_per_frame and hopsize.
    frame_energy : numpy.ndarray
        Total power spectrum energy of each frame.
    filter_banks : numpy.ndarray
        An array of mel filters
    cepstra : numpy.ndarray
        An array of the MFCC coeffcient, size: nframes x ncep
    """

    def __init__(self, x, sr=None, label='', n_per_frame=None,
                 hopsize=None, nfft=None, window='hann', nfilters=26, ncep=13, ceplifter=22,
                 preemph=0.95, append_energy=True, cn=None):
        """Initialize Amfcc object

        Parameters
        ----------
        x : Asig or numpy.ndarray
            x can be two forms, the most commonly used is an Asig object.
            Such as directly acquired from an Asig object via Asig.to_stft().
        sr : int, optional
            Sampling rate, this is not necessary if x is an Asig object as it has sr already.
        label : str, optional
            Identifier for the object
        n_per_frame : int, optional
            Number of samples per frame. Default is the equivalent of 25ms based on the sr.
        hopsize : int, optional
            Number of samples of each successive frame, Default is the sample equivalent of 10ms.
        nfft : int, optional
            FFT size, default to be next power of 2 integer of n_per_frame
        window : str, optional
            Type of the window function (Default value='hann'), use scipy.signal.get_window to
            return a numpy array. If None, no windowing will be applied.
        nfilters : int
            The number of mel filters. Default is 26
        ncep : int
            Number of cepstrum. Default is 13
        ceplifter : int
            Lifter's cepstral coefficient. Default is 22
        preemph : float
            Preemphasis coefficient. Default is 0.95
        append_energy : bool
            If true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
        cn : list
            A list of channel name based on the Asig.
        """
        # ----------Prepare attributes ------------`-------------
        # First prepare for parameters
        if type(x) == Asig.Asig:
            self.sr = x.sr
            self.x = x.sig
            self.label = x.label + "_mfccs"
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
        if not is_pow2(self.nfft):
            raise _LOGGER.warning("nfft is not a power of 2, this may effects computation time.")

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
        self.cepstra = np.dot(pspec, self.filter_banks.T)
        self.cepstra = np.where(self.cepstra == 0, np.finfo(float).eps, self.cepstra)
        self.cepstra = np.log(self.cepstra)

        self.cepstra = dct(self.cepstra, type=2, axis=1, norm='ortho')[:, :self.ncep]  # Discrete cosine transform

        self.cepstra = lifter(self.cepstra, self.ceplifter)

        # Replace first cepstral coefficient with log of frame energy
        if append_energy:
            self.cepstra[:, 0] = np.log(self.frame_energy)

    @property
    def nframes(self):
        return self.frames.shape[0]

    @property
    def timestamp(self):
        return np.linspace(0, self.duration, self.nframes)

    @property
    def features(self):
        """The features refer to the cepstra"""
        return self.cepstra

    def __repr__(self):
        # ToDO add more info to msg
        return f"Amfcc({self.label}): sr {self.sr}, length: {self.duration} s"

    def plot(self, cmap='inferno', show_bar=True,
             x_as_time=True, nxlabel=8, axis=None, **kwargs):
        """Plot Amfcc.features via matshow, x is frames/time, y is the MFCCs

        Parameters
        ----------
        figsize : (float, float), optional, default: None
             Figure size, width, height in inches, Default = [6.4, 4.8]
        show_bar
        x_as_time
        nxlabel
        kwargs
        cmap : string

        """
        ax = plt.gca() or axis
        # self.im = ax.matshow(self.cepstra.T, cmap=plt.get_cmap(cmap), origin='lower', aspect='auto', **kwargs)
        self.im = ax.pcolormesh(self.cepstra.T, cmap=plt.get_cmap(cmap))
        xticks = np.linspace(0, self.nframes, nxlabel, dtype=int)
        ax.set_xticks(xticks)
        # ax.set_ytitle("MFCC Coefficient")
        if x_as_time:
            xlabels = np.round(np.linspace(0, self.duration, nxlabel), decimals=2)
            # Replace x ticks with timestamps
            ax.set_xticklabels(xlabels)
            ax.xaxis.tick_bottom()
            # ax.set_xtitle("Time (s)")
        if show_bar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size="2%", pad=0.03)
            _ = plt.colorbar(self.im, cax=cax)   # Add
        return self  # ToDo maybe return the axis instead.