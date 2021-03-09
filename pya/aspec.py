import logging
import numpy as np
import scipy.interpolate
import pya.asig
from .helper import basicplot


_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class Aspec:
    """Audio spectrum class using rfft"""
    def __init__(self, x, sr=44100, label=None, cn=None):
        """__init__() method
        Parameters
        ----------
        x : Asig or numpy.ndarray
            audio signal
        sr : int
            sampling rate (Default value = 44100)
        label : str or None
            Asig label (Default value = None)
        cn : list or Nonpya.asige
            Channel names (Default value = None)
        """
        if type(x) == pya.asig.Asig:
            self.sr = x.sr
            self.rfftspec = np.fft.rfft(x.sig, axis=0)
            self.label = x.label + "_spec"
            self.samples = x.samples
            self.channels = x.channels
            self.cn = cn or x.cn
        elif type(x) == list or type(x) == np.ndarray:
            # TODO. This is in the assumption x is spec. which is wrong. We define x to be the audio signals instead.
            self.rfftspec = np.array(x)
            self.sr = sr
            self.samples = (len(x) - 1) * 2
            self.channels = x.ndim
        else:
            s = "argument x must be an Asig or an array"
            raise TypeError(s)
        if label:
            self.label = label
        if cn:
            self.cn = cn
        self.nr_freqs = self.samples // 2 + 1
        self.freqs = np.linspace(0, self.sr / 2, self.nr_freqs)

    def get_duration(self):
        """Return the duration in second."""
        return self.samples / self.sr

    def to_sig(self):
        """Convert Aspec into Asig"""
        return pya.asig.Asig(np.fft.irfft(self.rfftspec),
                             sr=self.sr, label=self.label + '_2sig', cn=self.cn)

    def weight(self, weights, freqs=None, curve=1, kind='linear'):
        """TODO

        Parameters
        ----------
        weights :

        freqs :
             (Default value = None)
        curve :
             (Default value = 1)
        kind :
             (Default value = 'linear')

        Returns
        -------

        """
        nfreqs = len(weights)
        if not freqs:
            given_freqs = np.linspace(0, self.freqs[-1], nfreqs)
        else:
            if nfreqs != len(freqs):
                _LOGGER.error("len(weights)!=len(freqs)")
                return self
            if all(freqs[i] < freqs[i + 1] for i in range(len(freqs) - 1)):
                # check if list is monotonous
                if freqs[0] > 0:
                    freqs = np.insert(np.array(freqs), 0, 0)
                    weights = np.insert(np.array(weights), 0, weights[0])
                if freqs[-1] < self.sr / 2:
                    freqs = np.insert(np.array(freqs), -1, self.sr / 2)
                    weights = np.insert(np.array(weights), -1, weights[-1])
            else:
                _LOGGER.error("Aspec.weight error: freqs not sorted")
                return self
            given_freqs = freqs
        if nfreqs != self.nr_freqs:
            interp_fn = scipy.interpolate.interp1d(given_freqs,
                                                   weights, kind=kind)
            # ToDo: curve segmentwise!!!
            rfft_new = self.rfftspec * interp_fn(self.freqs) ** curve
        else:
            rfft_new = self.rfftspec * weights ** curve
        return Aspec(rfft_new, self.sr,
                     label=self.label + "_weighted", cn=self.cn)

    def plot(self, fn=np.abs, ax=None,
             offset=0, scale=1,
             xlim=None, ylim=None, **kwargs):
        """Plot spectrum

        Parameters
        ----------
        fn : func
            function for processing the rfft spectrum. (Default value = np.abs)
        x_as_time : bool, optional
            By default x axis display the time, if faulse display samples
        xlim : tuple or list or None
            Set x axis range (Default value = None)
        ylim : tuple or list or None
            Set y axis range (Default value = None)
        offset : int or float
            This is the absolute value each plot is shift vertically
            to each other.
        scale : float
            Scaling factor of the plot, use in multichannel plotting.
        **kwargs :
            Keyword arguments for matplotlib.pyplot.plot()
        Returns
        -------
        _ : Asig
            self
        """
        _, ax = basicplot(fn(self.rfftspec), self.freqs, channels=self.channels,
                          cn=self.cn, offset=offset, scale=scale,
                          ax=ax, typ='plot',
                          xlabel='freq (Hz)', ylabel=f'{fn.__name__}(freq)',
                          xlim=xlim, ylim=ylim, **kwargs)
        return self

    def __repr__(self):
        return "Aspec('{}'): {} x {} @ {} Hz = {:.3f} s".format(
            self.label, self.channels, self.samples,
            self.sr, self.samples / self.sr)
