import numbers
from warnings import warn
import logging
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.fftpack import fft, fftfreq, ifft
from scipy.io import wavfile
from . import Asig
from .helper import ampdb, dbamp, linlin, timeit, spectrum, audio_from_file

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


# TODO, check with multichannel
class Astft:
    """Audio spectrogram (STFT) class, attributes refers to scipy.signal.stft. With an addition
        attribute cn being the list of channel names, and label being the name of the Asig
    """

    def __init__(self, x, sr=None, label=None, window='hann', nperseg=256,
                 noverlap=None, nfft=None, detrend=False, return_onesided=True,
                 boundary='zeros', padded=True, axis=-1, cn=None):
        """__init__() method

        Parameters
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
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.detrend = detrend
        self.return_onesided = return_onesided
        self.boundary = boundary
        self.padded = padded
        self.axis = axis
        self.cn = cn

        if type(x) == Asig.Asig:
            # TODO multichannel.
            self.sr = x.sr
            if sr:
                self.sr = sr  # explicitly given sr overwrites Asig
            self.freqs, self.times, self.stft = scipy.signal.stft(
                x.sig, fs=self.sr, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                detrend=detrend, return_onesided=return_onesided, boundary=boundary, padded=padded, axis=axis)
            self.label = x.label + "_stft"
            self.samples = x.samples
            self.channels = x.channels
        elif isinstance(x, np.ndarray) and x.ndim >= 2:
            self.stft = x
            self.sr = 44100
            if sr:
                self.sr = sr
            self.samples = (len(x) - 1) * 2
            self.channels = 1
            if len(np.shape(x)) > 2:
                self.channels = np.shape(x)[2]
            # TODO: set other values, particularly check if self.times and self.freqs are correct
            self.ntimes, self.nfreqs, = np.shape(self.stft)
            self.times = np.linspace(0, (self.nperseg - self.noverlap) * self.ntimes / self.sr, self.ntimes)
            self.freqs = np.linspace(0, self.sr // 2, self.nfreqs)
        else:
            raise TypeError("Unknown initializer or wrong stft shape ")
        if label:
            self.label = label

    def to_sig(self, **kwargs):
        """Create signal from stft, i.e. perform istft, kwargs overwrite Astft values for istft

        Parameters
        ----------
        **kwargs : str
            optional keyboard arguments used in istft: 
                'sr', 'window', 'nperseg', 'noverlap', 'nfft', 'input_onesided', 'boundary'.
            also convert 'sr' to 'fs' since scipy uses 'fs' as sampling frequency.

        Returns
        -------
        _ : Asig
            Asig
        """
        for k in ['sr', 'window', 'nperseg', 'noverlap', 'nfft', 'input_onesided', 'boundary']:
            if k in kwargs.keys():
                kwargs[k] = self.__getattribute__(k)
        if 'sr' in kwargs.keys():
            kwargs['fs'] = kwargs['sr']
            del kwargs['sr']
        _, sig = scipy.signal.istft(self.stft, **kwargs)  # _ since 1st return value 'times' unused
        return Asig.Asig(sig, sr=self.sr, label=self.label + '_2sig', cn=self.cn)

    def plot(self, fn=lambda x: x, ax=None, xlim=None, ylim=None, **kwargs):
        """Plot spectrogram

        Parameters
        ----------
        fn : func
            a function, by default is bypass
        ax : matplotlib.axes
            you can assign your plot to specific axes (Default value = None)
        xlim : tuple or list
            x_axis range (Default value = None)
        ylim : tuple or list
            y_axis range (Default value = None)
        **kwargs :
            keyward arguments of matplotlib's pcolormesh

        Returns
        -------
        _ : Asig
            self
        """
        if ax is None:
            plt.pcolormesh(self.times, self.freqs, fn(np.abs(self.stft)), **kwargs)
            plt.colorbar()
            if ylim is not None:
                plt.ylim([ylim[0], ylim[1]])
        else:
            ax.pcolormesh(self.times, self.freqs, fn(np.abs(self.stft)), **kwargs)
            if ylim is not None:
                ax.set_ylim(ylim[0], ylim[1])
        return self

    def __repr__(self):
        return "Astft('{}'): {} x {} @ {} Hz = {:.3f} s".format(
            self.label, self.channels, self.samples, self.sr, self.samples / self.sr, cn=self.cn)

