import numpy as np
from warnings import warn
from .helper import next_pow2, signal_to_frame, round_half_up, magspec
from .helper import mel2hz, hz2mel, is_pow2
from .helper import basicplot
from scipy.signal import get_window
from scipy.fftpack import dct
import pya.asig
import logging

# _LOGGER = logging.getLogger(__name__)
# _LOGGER.addHandler(logging.NullHandler())


class Amfcc:
    """Mel filtered Fourier spectrum (MFCC) class,
    this class is inspired by jameslyons/python_speech_features,
    https://github.com/jameslyons/python_speech_features
    Steps of mfcc:
        * Frame the signal into short frames.
        * For each frame calculate the periodogram estimate of the
        power spectrum.
        * Apply the mel filterbank to the power spectra, sum the energy
        in each filter.
        * Take the DCT of the log filterbank energies.
        * Keep DCT coefficients 2-13, discard the rest.
        * Take the logarithm of all filterbank energies.

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
        Type of the window function (Default value='hann'),
        use scipy.signal.get_window to return a numpy array.
        If None, no windowing will be applied.
    nfilters : int
        The number of mel filters. Default is 26
    ncep : int
        Number of cepstrum. Default is 13
    cepliter : int
        Lifter's cepstral coefficient. Default is 22
    frames : numpy.ndarray
        The original signal being reshape into frame based on
        n_per_frame and hopsize.
    frame_energy : numpy.ndarray
        Total power spectrum energy of each frame.
    filter_banks : numpy.ndarray
        An array of mel filters
    cepstra : numpy.ndarray
        An array of the MFCC coeffcient, size: nframes x ncep
    """

    def __init__(self, x, sr=None, label='', n_per_frame=None,
                 hopsize=None, nfft=None, window='hann', nfilters=26,
                 ncep=13, ceplifter=22, preemph=0.95,
                 append_energy=True, cn=None):
        """Initialize Amfcc object

        Parameters
        ----------
        x : Asig or numpy.ndarray
            x can be two forms, the most commonly used is an Asig object.
            Such as directly acquired from an Asig object via Asig.to_stft().
        sr : int, optional
            Sampling rate, this is not necessary if x is an Asig object as
            it has sr already.
        label : str, optional
            Identifier for the object
        n_per_frame : int, optional
            Number of samples per frame. Default is the equivalent of
            25ms based on the sr.
        hopsize : int, optional
            Number of samples of each successive frame, Default is the
            sample equivalent of 10ms.
        nfft : int, optional
            FFT size, default to be next power of 2 integer of n_per_frame
        window : str, optional
            Type of the window function (Default value='hann'),
            use scipy.signal.get_window to
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
            If true, the zeroth cepstral coefficient is replaced with the log
            of the total frame energy.
        cn : list
            A list of channel name based on the Asig.
        """
        # ----------Prepare attributes ------------`-------------
        # First prepare for parameters
        # x represent the audio signal, which can be Asig object or np.array.
        self.im = None
        if type(x) == pya.asig.Asig:
            self.sr = x.sr
            self.x = x.sig
            self.label = ''.join([x.label, "_mfccs"])
            self.duration = x.get_duration()
            self.channels = x.channels
            self.cn = x.cn

        elif isinstance(x, np.ndarray):
            self.x = x
            if sr:
                self.sr = sr
            else:
                msg = "If x is an array," \
                    " sra(sampling rate) needs to be defined."
                raise AttributeError(msg)
            self.duration = np.shape(x)[0] / self.sr
            self.label = label
            self.channels = 1 if self.x.ndim == 1 else self.x.shape[1]
            self.cn = None
        else:
            msg = "x can only be either a numpy.ndarray or pya.Asig object."
            raise TypeError(msg)

        # default 25ms length window.
        self.n_per_frame = n_per_frame or int(round_half_up(self.sr * 0.025))
        # default 10ms overlap
        self.hopsize = hopsize or int(round_half_up(self.sr * 0.01))
        if self.hopsize > self.n_per_frame:
            msg = "noverlap > nperseg, this leaves gaps between frames."
            warn(msg)
        self.nfft = nfft or next_pow2(self.n_per_frame)
        if not is_pow2(self.nfft):
            msg = "nfft is not power of 2, this may effects computation time."
            warn(msg)
        if window:
            self.window = get_window(window, self.n_per_frame)
        else:
            self.window = np.ones((self.n_per_frame,))
        self.nfilters = nfilters
        self.ncep = ncep  # Number of cepstrum
        self.ceplifter = ceplifter  # Lifter's cepstral coefficient
        # -------------------------------------------------

        # Framing signal.
        pre_emp_sig = self.preemphasis(self.x, coeff=preemph)
        self.frames = signal_to_frame(pre_emp_sig,
                                      self.n_per_frame, self.hopsize,
                                      self.window)

        # Computer power spectrum
        # Magnitude of spectrum, rfft then np.abs()
        mspec = magspec(self.frames, self.nfft)
        pspec = 1.0 / self.nfft * np.square(mspec)  # Power spectrum

        # Total energy of each frame based on the power spectrum
        self.frame_energy = np.sum(pspec, 1)
        # Replace 0 with the smallest float positive number
        self.frame_energy = np.where(self.frame_energy == 0,
                                     np.finfo(float).eps,
                                     self.frame_energy)

        # Prepare Mel filter
        # Use the default filter banks.
        self.filter_banks = Amfcc.mel_filterbanks(self.sr,
                                                  nfilters=self.nfilters,
                                                  nfft=self.nfft)

        # filter bank energies are the features.
        self.cepstra = np.dot(pspec, self.filter_banks.T)
        self.cepstra = np.where(self.cepstra == 0,
                                np.finfo(float).eps, self.cepstra)
        self.cepstra = np.log(self.cepstra)

        # Discrete cosine transform
        self.cepstra = dct(self.cepstra, type=2,
                           axis=1, norm='ortho')[:, :self.ncep]

        self.cepstra = Amfcc.lifter(self.cepstra, self.ceplifter)

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
        return f"Amfcc({self.label}): sr {self.sr}, length: {self.duration} s"

    @staticmethod
    def preemphasis(x, coeff=0.97):
        """Pre-emphasis filter to whiten the spectrum.
        Pre-emphasis is a way of compensating for the
        rapid decaying spectrum of speech.
        Can often skip this step in the cases of music for example

        Parameters
        ----------
        x : numpy.ndarray
            Signal array
        coeff : float
            Preemphasis coefficient. The larger the stronger smoothing
            and the slower response to change.

        Returns
        -------
        _ : numpy.ndarray
            The whitened signal.
        """
        return np.append(x[0], x[1:] - coeff * x[:-1])

    @staticmethod
    def mel_filterbanks(sr, nfilters=26, nfft=512, lowfreq=0, highfreq=None):
        """Compute a Mel-filterbank. The filters are stored in the rows,
        the columns correspond to fft bins. The filters are returned as
        an array of size nfilt * (nfft/2 + 1)

        Parameters
        ----------
        sr : int
            Sampling rate
        nfilters : int
            The number of filters, default 20
        nfft : int
            The size of FFT, default 512
        lowfreq : int or float
            The lowest band edge of the mel filters, default 0 Hz
        highfreq : int or float
            The highest band edge of the mel filters, default sr // 2

        Returns
        -------
        _ : numpy.ndarray
            A numpy array of size nfilt * (nfft/2 + 1)
            containing filterbank. Each row holds 1 filter.
        """
        highfreq = highfreq or sr // 2

        # compute points evenly spaced in mels
        lowmel = hz2mel(lowfreq)
        highmel = hz2mel(highfreq)
        melpoints = np.linspace(lowmel, highmel, nfilters + 2)
        # our points are in Hz, but we use fft bins, so we have to convert
        #  from Hz to fft bin number
        bin = np.floor((nfft + 1) * mel2hz(melpoints) / sr)

        filter_banks = np.zeros([nfilters, nfft // 2 + 1])
        for j in range(0, nfilters):
            for i in range(int(bin[j]), int(bin[j + 1])):
                filter_banks[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
            for i in range(int(bin[j + 1]), int(bin[j + 2])):
                filter_banks[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
        return filter_banks

    @staticmethod
    def lifter(cepstra, L=22):
        """Apply a cepstral lifter the the matrix of cepstra.
        This has the effect of increasing the magnitude of
        the high frequency DCT coeffs.

        Liftering operation is similar to filtering operation in the
        frequency domain
        where a desired quefrency region for analysis is selected
        by multiplying the whole cepstrum
        by a rectangular window at the desired position.
        There are two types of liftering performed,
        low-time liftering and high-time liftering.
        Low-time liftering operation is performed to extract
        the vocal tract characteristics in the quefrency domain
        and high-time liftering is performed to get the excitation
        characteristics of the analysis speech frame.


        Parameters
        ----------
        cepstra : numpy.ndarray
            The matrix of mel-cepstra
        L : int
            The liftering coefficient to use. Default is 22,
            since cepstra usually has 13 elements, 22
            L will result almost half pi of sine lift.
            It essential try to emphasis to lower ceptral coefficient
            while deemphasize higher ceptral coefficient as they are
            less discriminative for speech contents.
        """
        if L > 0:
            nframes, ncoeff = np.shape(cepstra)
            n = np.arange(ncoeff)
            lift = 1 + (L / 2.) * np.sin(np.pi * n / L)
            return lift * cepstra
        else:
            # values of L <= 0, do nothing
            return cepstra

    def plot(self, cmap='inferno', show_bar=True,
             offset=0, scale=1., xlim=None, ylim=None,
             x_as_time=True, nxlabel=8, ax=None, **kwargs):
        """Plot Amfcc.features via matshow, x is frames/time, y is the MFCCs

        Parameters
        ----------
        figsize : (float, float), optional, default: None
            Figure size, width, height in inches, Default = [6.4, 4.8]
        cmap : str
            colormap for matplotlib. Default is 'inferno'.
        show_bar : bool, optional
            Default is True, show colorbar.
        x_as_time : bool, optional
            Default is True, show x axis as time or sample index.
        nxlabel : int, optional
            The amountt of labels on the x axis. Default is 8 .
        """
        if self.channels > 1:
            warn("Multichannel mfcc is not yet implemented. Please use "
                 "mono signal for now, no plot is made")
            return self
        im, ax = basicplot(self.cepstra.T, None,
                           channels=self.channels,
                           cn=self.cn, offset=offset, scale=scale,
                           ax=ax, typ='mfcc', show_bar=show_bar,
                           xlabel='time', xlim=xlim, ylim=ylim, **kwargs)
        self.im = im
        xticks = np.linspace(0, self.nframes, nxlabel, dtype=int)
        ax.set_xticks(xticks)
        # ax.set_("MFCC Coefficient")
        if x_as_time:
            xlabels = np.round(np.linspace(0, self.duration, nxlabel),
                               decimals=2)
            # Replace x ticks with timestamps
            ax.set_xticklabels(xlabels)
            ax.xaxis.tick_bottom()
            # ax.set_xtitle("Time (s)")
        return self
