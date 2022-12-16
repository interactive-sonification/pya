import numbers
from warnings import warn
import logging
from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.signal
from scipy.io import wavfile
from . import Aserver
import pya.aspec
import pya.astft
import pya.amfcc
from .helper import ampdb, dbamp, linlin
from .helper import spectrum, audio_from_file, padding
from .helper import basicplot


_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class Asig:
    """Audio signal class.
    Asig enables manipulation of audio signals in the style of numpy and more.
    Asig offer functions for plotting (via matplotlib) and playing audio
    (using the pya.Aserver class)

    Attributes
    ----------
    sig : numpy.array
        Array for the audio signal. Can be mono or multichannel.
    sr : int
        Sampling rate
    label : str
        A string label to give the object a unique identifier.
    channels : int
        Number of channels
    cn : list of str, None
        cn short for channel names is a list of string of size channels,
        to give each channel a unique name.
        channel names can be used to subset signal channels in
        a more readible way,
        e.g. asig[:, ['left', 'front']] subsets the left and
        front channels of the signal.
    mix_mode : str or None
        used to extend numpy __setitem__() operation to
        frequent audio manipulations such as
        mixing, extending, boundary, replacing.
        Current Asig supports the mix_modes:
        bound, extend, overwrite.  mix_mode should not be
        set directly but is set temporarilty when using
        the .bound, .extend and .overwrite properties.
    """

    def __init__(self, sig, sr=44100, label="", channels=1, cn=None):
        """__init__ method

        Parameters
        ----------
            sig: numpy.array or int or float or str
                numpy.array for audio signal,
                str for filepath. Currently support two types
                of audio loader: 1) Standard library for .wav, .aiff,
                and ffmpeg for other such as .mp3.
                int create x samples of silence,
                float creates x seconds of seconds.
            sr : int
                Sampling rate
            label : str
                Label for the object
            channels : int
                Number of channels,
                no need to set it if you already have a
                signal for the sig argument.
            cn : list or None
                A list of channel names, size should match the channels.
        """
        if isinstance(sr, int):
            self.sr = sr
        else:
            raise AttributeError("sr needs to be int.")
        self.mix_mode = None
        self._ = {}  # dictionary for further return values
        self.label = label
        self.dtype = "float32"
        if isinstance(sig, str):
            self.sig, self.sr = audio_from_file(sig)
            if self.label == "":
                self.label = sig
        elif isinstance(sig, int):  # sample length
            if channels == 1:
                self.sig = np.zeros(sig).astype(self.dtype)
            else:
                self.sig = np.zeros((sig, channels)).astype(self.dtype)
        elif isinstance(sig, float):  # if float interpret as duration
            if channels == 1:
                self.sig = np.zeros(int(sig * sr)).astype(self.dtype)
            else:
                self.sig = np.zeros(
                    (int(sig * sr), channels)).astype(self.dtype)
        else:
            self.sig = np.array(sig).astype(self.dtype)
        self.cn = cn
        self._set_col_names()

    @property
    def channels(self):
        """Return the number of channels"""
        try:
            return self.sig.shape[1]
        except IndexError:
            return 1

    @property
    def cn(self):
        """Channel names getter"""
        return self._cn

    @cn.setter
    def cn(self, val):
        """Channel names setter"""
        if val is None:
            self._cn = None
        else:
            if len(val) == self.channels:
                # check if all elements are str
                if all(isinstance(x, str) for x in val):
                    self._cn = val
                else:
                    raise TypeError(
                        "channel names cn need to be a list of string(s).")
            else:
                raise ValueError(
                    "list size doesn't match channel numbers {}".format(
                        self.channels)
                )

    @property
    def samples(self) -> int:
        """
        Return the length of signal in samples
        """
        return np.shape(self.sig)[0]

    @property
    def dur(self) -> float:
        """
        Return the duration in seconds
        """
        return self.samples / self.sr

    def save_wavfile(self, fname="asig.wav", dtype="float32"):
        """Save signal as .wav file, return self.

        Parameters
        ----------
        fname : str
            name of the file with .wav (Default value = "asig.wav")
        dtype : str
            datatype (Default value = 'float32')
        """
        if dtype == "int16":
            data = (self.sig * 32767).astype("int16")
        elif dtype == "int32":
            data = (self.sig * 2147483647).astype("int32")
        elif dtype == "uint8":
            data = (self.sig * 127 + 128).astype("uint8")
        elif dtype == "float32":
            data = self.sig.astype("float32")
        wavfile.write(fname, self.sr, data)
        return self

    def _set_col_names(self):
        if self.cn is None:
            self.cn = [str(i) for i in range(self.channels)]
        else:
            if isinstance(self.cn[0], str):
                self.col_name = {self.cn[i]: i for i in range(len(self.cn))}
            else:
                raise TypeError("column names need to be a list of strings")

    def __getitem__(self, index):
        """ Accessing array elements through slicing.
            * int, get signal row asig[4];
            * slice, range and step slicing asig[4:40:2]
                # from 4 to 40 every 2 samples;
            * list, subset rows, asig[[2, 4, 6]]
                # pick out index 2, 4, 6 as a new asig
            * tuple, row and column specific slicing, asig[4:40, 3:5]
                # from 4 to 40, channel 3 and 4
            * Time slicing (unit in seconds) using dict asig[{1:2.5}, :]
                creates indexing of 1s to 2.5s.
            * Channel name slicing: asig['l'] returns channel 'l' as
                a new mono asig. asig[['front', 'rear']], etc...
            * bool, subset channels: asig[:, [True, False]]


        Parameters
        ----------
            index : Number or slice or list or tuple or dict
                Slicing argument.

        Returns
        -------
            a : Asig
                __getitem__ returns a subset of the self based on the slicing.
        """
        if isinstance(index, tuple):
            _LOGGER.debug(" getitem: index is tuple")
            rindex = index[0]
            cindex = index[1] if len(index) > 1 else None
        elif isinstance(index, str):
            _LOGGER.debug(" getitem: index is string")
            # ToDo: decide whether to solve differently,
            # e.g. only via ._[str] or via a .attribute(str) fn
            return self._[index]
        else:
            # if only slice, list, dict, int or float given for row selection
            rindex = index
            cindex = None

        # parse row index rindex into ridx
        # e.g. a[[4,5,7,8,9]], or a[[True, False, True...]]
        if isinstance(rindex, list):
            _LOGGER.debug("list slicing of index.")
            ridx = rindex
            sr = self.sr
        elif isinstance(rindex, int):  # picking a single row
            ridx = rindex
            _LOGGER.debug("integer slicing of index: %d", ridx)
            sr = self.sr
        elif isinstance(rindex, slice):
            _LOGGER.debug(" getitem: row index is slice.")
            _, _, step = rindex.indices(len(self.sig))
            sr = int(self.sr / abs(step))
            ridx = rindex
        elif isinstance(rindex, dict):  # time slicing
            _LOGGER.debug(" getitem: row index is dict. Time slicing.")
            for key, val in rindex.items():
                try:
                    start = int(key * self.sr)
                except TypeError:  # if it is None
                    start = None
                try:
                    stop = int(val * self.sr)
                except TypeError:
                    stop = None
            ridx = slice(start, stop, 1)
            sr = self.sr
            _LOGGER.debug("Time slicing, start: %s, stop: %s",
                          str(start), str(stop))
        else:  # Dont think there is a usecase.
            ridx = rindex
            sr = self.sr

        # now parse cindex
        if hasattr(cindex, "__iter__"):
            _LOGGER.debug(" getitem: column index is iterable.")
            if isinstance(cindex[0], str):
                cidx = [self.col_name.get(s) for s in cindex]
                if cidx is None:
                    _LOGGER.error("Input column names does not exist")
                cn_new = [self.cn[i]
                          for i in cidx] if self.cn is not None else None
            elif isinstance(cindex[0], bool):
                cidx = cindex
                cn_new = list(compress(self.cn, cindex))
            else:
                try:
                    cidx = list(cindex)
                    cn_new = (
                        [self.cn[i]
                            for i in cindex] if self.cn is not None else None
                    )
                except (TypeError, ValueError):
                    cidx = slice(0, 0, 0)
                    cn_new = self.cn
        elif isinstance(cindex, int):
            _LOGGER.debug(" getitem: column index is int.")
            cidx = cindex
            cn_new = [self.cn[cindex]] if self.cn is not None else None
        elif isinstance(cindex, slice):
            _LOGGER.debug(" getitem: column index is slice.")
            cidx = cindex
            cn_new = self.cn[cindex] if self.cn is not None else None
        # if only a single channel name is given.
        elif isinstance(cindex, str):
            _LOGGER.debug(" getitem: column index is string.")
            cidx = self.col_name.get(cindex)
            cn_new = [cindex]
        else:  # if nothing is given, e.g. index = (ridx,) on calling a[:]
            cidx = slice(None, None, None)
            cn_new = self.cn
        # apply ridx and cidx and return result
        sig = self.sig[ridx, cidx] if self.channels > 1 else self.sig[ridx]

        # Squeezing shouldn't be performed here.
        # this is because: a[:10, 0] and a[:10,[True, False]] return
        # (10,) and (10, 1) respectively.
        # Which should be dealt with individually.
        if sig.ndim == 2 and sig.shape[1] == 1:
            # Hot fix this to be consistent with bool slciing
            if not isinstance(cindex[0], bool):
                _LOGGER.debug(
                    "ndim is 2 and channel num is 1, performa np.squeeze")
                sig = np.squeeze(sig)
        if isinstance(sig, numbers.Number):
            _LOGGER.debug("signal is scalar, convert to array")
            sig = np.array(sig)

        a = Asig(sig, sr=sr, label=self.label + "_arrayindexed", cn=cn_new)
        a.mix_mode = self.mix_mode
        return a

    @property
    def x(self):
        """Extend mode: this mode allows destination
        sig size in assignment to be extended through setitem"""
        # Set setitem mode to extend
        self.mix_mode = "extend"
        return self

    extend = x

    @property
    def b(self):
        """Bound mode: this mode allows to truncate a source signal
        in assignment to a limited destination in setitem."""
        # Set setitem mode to bound
        self.mix_mode = "bound"
        return self

    bound = b

    @property
    def o(self):
        """Overwrite mode: this mode cuts and replaces target
        selection by source signal on assignment via setitem"""
        self.mix_mode = "overwrite"
        return self

    overwrite = o

    def __setitem__(self, index, value):
        """setitem: asig[index] = value. This allows all the methods from getitem:
            * numpy style slicing
            * string/string_list for subsetting based on channel name
            * time slicing (unit seconds) via dict.
            * bool slicing to filter out specific channels.
        In addition, there are 4 possible modes:
        (referring to asig as 'dest', and value as 'src'
            1. standard pythonic way that the src and
            dest dimensions need to match
                asig[...] = value
            2. bound mode where src is copied up to the bounds of dest
                asig.b[...] = value
            3. extend mode where dest is dynamically
            extended to make space for src
                asig.x[...] = value
            4. overwrite mode where selected dest subset is
            replaced by specified src regardless the length.
                asig.o[...] = value

        row index:
            * list: e.g. [1,2,3,4,5,6] or [True, ..., False]
                (modes b and x possible)
            * int:  e.g. 0  (i.e. a single sample, so no need for extra modes)
            * slice: e.g. 100:5000:2  (can be used with all modes)
            * dict: e.g. {0.5: 2.5}
                (modes o, b possible, x only if step==1,
                or if step==None and stop=None)

        Parameters
        ----------
            index : Number or slice or list or tuple or dict
                Slicing argument.
            value : Asig or numpy.ndarray or list
                value to set

        Returns
        -------
        _: Asig
            Updated asig
        """
        # TODOs:
        # check if mix_mode copy required on each fn output: if yes implement
        # check all sig = [[no numpy array]] cases
        # a.x[300:,1:2] = 0.5*b with 1-ch b to 4-ch a: shape problem (600, ) to (600, 1)
        mode = self.mix_mode
        self.mix_mode = None  # reset when done

        if isinstance(index, tuple):
            rindex = index[0]
            cindex = index[1] if len(index) > 1 else None
        else:
            rindex = index
            cindex = None

        if isinstance(rindex, (slice, int)):
            ridx = rindex
        elif isinstance(rindex, dict):
            # time slicing
            for key, val in rindex.items():
                try:
                    start = int(key * self.sr)
                except TypeError:
                    start = None
                try:
                    stop = int(val * self.sr)
                except TypeError:
                    stop = None
            ridx = slice(start, stop, 1)
        elif hasattr(rindex, "__iter__"):
            ridx = list(rindex)
        else:
            return  # we cannot determine a row index; return without changes

        # now parse cindex
        if hasattr(cindex, "__iter__"):
            if isinstance(cindex[0], str):
                cidx = [self.col_name.get(s) for s in cindex]
                cidx = cidx[0] if len(cidx) == 1 else cidx  # hotfix for now.
            else:
                try:
                    cidx = list(cindex)
                except TypeError:
                    cidx = slice(None)
        # int, slice are the same.
        elif isinstance(cindex, (int, slice)):
            cidx = cindex
        # if only a single channel name is given.
        elif isinstance(cindex, str):
            cidx = self.col_name.get(cindex)
        else:
            cidx = slice(None)

        _LOGGER.debug("self.sig.ndim == %d", self.sig.ndim)
        if self.sig.ndim == 1:
            final_index = ridx
        else:
            final_index = (ridx, cidx)
        # apply setitem: set dest[ridx,cidx] = src return self

        if isinstance(value, Asig):
            _LOGGER.debug("value is asig")
            src = value.sig

        elif isinstance(value, np.ndarray):
            # numpy array if not Asig, default: sr fits
            _LOGGER.debug("value is ndarray")
            src = value

        elif isinstance(value, list):  # if list
            _LOGGER.debug("value is list")
            src = value
            # for list (assume values for channels), mode makes no sense...
            mode = None

        else:
            _LOGGER.debug("value not asig, ndarray, list")
            src = value
            mode = None  # for scalar types, mode makes no sense...

        if mode is None:
            _LOGGER.debug("Default setitem mode")
            if isinstance(src, numbers.Number):
                self.sig[final_index] = src
            elif isinstance(
                src, list
            ):  # for multichannel signals that is value for each column
                self.sig[final_index] = src
            else:  # numpy.ndarray
                try:
                    self.sig[final_index] = np.broadcast_to(
                        src, self.sig[final_index].shape
                    )
                except ValueError:
                    self.sig[final_index] = src

        elif mode == "bound":
            _LOGGER.debug("setitem bound mode")
            dshape = self.sig[final_index].shape
            dn = dshape[0]  # ToDo: howto get that faster from ridx alone?
            sn = src.shape[0]
            if sn > dn:
                self.sig[final_index] = src[:dn] if len(
                    dshape) == 1 else src[:dn, :]
            else:
                self.sig[final_index][:sn] = src if len(
                    dshape) == 1 else src[:, :]

        elif mode == "extend":
            _LOGGER.debug("setitem extend mode")
            if isinstance(ridx, list):
                _LOGGER.error(
                    "Extend mode not available for row index list"
                )
                return self
            if isinstance(ridx, slice):
                if ridx.step not in [1, None]:
                    raise AttributeError(
                        "Extend mode only available for step-1 slices"
                    )
                if ridx.stop is not None and ridx.stop < self.samples:
                    raise AttributeError(
                        "The current slice does not stop at the end of array."
                    )
            dshape = self.sig[final_index].shape  # d for destination
            # ToDo: howto compute dn faster from ridx shape(self.sig) alone?
            dn = dshape[0]
            sn = src.shape[0]
            if sn <= dn:  # same as bound, since src fits in
                self.sig[final_index][:sn] = np.broadcast_to(
                    src, (sn,) + dshape[1:])
            elif sn > dn:
                self.sig[final_index] = src[:dn]
                # now extend by nn = sn-dn additional rows
                if dn > 0:
                    nn = sn - dn  # nr of needed additional rows
                    # import pdb; pdb.set_trace()
                    self.sig = np.r_[
                        self.sig, np.zeros((nn,) + self.sig.shape[1:])
                    ].astype(self.dtype)
                    if self.sig.ndim == 1:
                        self.sig[-nn:] = src[dn:]
                    else:
                        self.sig[-nn:, cidx] = src[dn:]
                else:  # this is when start is beyond length of dest...
                    nn = ridx.start + sn
                    self.sig = np.r_[
                        self.sig,
                        np.zeros(
                            (nn - self.sig.shape[0],) + self.sig.shape[1:]),
                    ].astype(self.dtype)
                    if self.sig.ndim == 1:
                        self.sig[-sn:] = src
                    else:
                        self.sig[-sn:, cidx] = src

        elif mode == "overwrite":
            # This mode is to replace a subset with an any given shape.
            # Where the end point of the newly insert signal should be.
            _LOGGER.info("setitem overwrite mode")
            start_idx = (
                ridx.start if isinstance(ridx, slice) else 0
            )  # Start index of the ridx,
            stop_idx = (
                ridx.stop if isinstance(ridx, slice) else 0
            )  # Stop index of the rdix
            end = start_idx + src.shape[0]
            # Create a new signal
            # New row is: original samples + (new_signal_sample - the range to replace)
            sig = np.ndarray(
                shape=(
                    self.sig.shape[0] + src.shape[0] - (stop_idx - start_idx),
                    self.channels,
                )
            )
            if sig.ndim == 2 and sig.shape[1] == 1:
                sig = np.squeeze(sig)
            if isinstance(sig, numbers.Number):
                sig = np.array(sig)
            sig[:start_idx] = self.sig[:start_idx]  # Copy the first part over
            # The second part is the new signal
            sig[start_idx:end] = src
            # The final part is the remaining of self.sig
            sig[end:] = self.sig[stop_idx:]
            self.sig = sig  # Update self.sig
        return self

    def resample(self, target_sr=44100, rate=1, kind="linear"):
        """Resample signal based on interpolation, can process multichannel signals.

        Parameters
        ----------
        target_sr : int
            Target sampling rate (Default value = 44100)
        rate : float
            Rate to speed up or slow down the audio (Default value = 1)
        kind : str
            Type of interpolation (Default value = 'linear')

        Returns
        -------
        _ : Asig
            Asig with resampled signal.
        """
        times = np.arange(self.samples) / self.sr
        tsel = (np.arange(np.floor(self.samples / self.sr *
                                   target_sr / rate)) * rate / target_sr)
        if self.channels == 1:
            interp_fn = scipy.interpolate.interp1d(
                times,
                self.sig,
                kind=kind,
                assume_sorted=True,
                bounds_error=False,
                fill_value=self.sig[-1],
            )
            return Asig(
                interp_fn(tsel), target_sr,
                label=self.label + "_resampled", cn=self.cn
            )
        else:
            new_sig = np.ndarray(
                shape=(int(self.samples / self.sr * target_sr / rate), self.channels))
            for i in range(self.channels):
                interp_fn = scipy.interpolate.interp1d(
                    times,
                    self.sig[:, i],
                    kind=kind,
                    assume_sorted=True,
                    bounds_error=False,
                    fill_value=self.sig[-1, i],
                )
                new_sig[:, i] = interp_fn(tsel)
            return Asig(new_sig, target_sr, label=self.label + "_resampled", cn=self.cn)

    def play(self, rate=1, server=None, onset=0, channel=0, block=False):
        """Play Asig audio via Aserver, using Aserver.default (if existing)
        kwargs are propagated to Aserver:play(onset=0, out=0)

        Parameters
        ----------
        rate : float
            Playback rate (Default value = 1)
        **kwargs : str
            'server' : Aserver
                Set which server to play. e.g. s = Aserver(); s.boot(); asig.play(server=s)

        Returns
        -------
        _ : Asig
            return self
        """
        s = server or Aserver.default
        if not isinstance(s, Aserver):
            warn("Asig.play: no default server running, nor server arg specified.")
            return self
        if rate == 1 and self.sr == s.sr:
            asig = self
        else:
            asig = self.resample(s.sr, rate)
        s.play(asig, server=s, onset=onset, out=channel, block=block)
        return self

    def shift_channel(self, shift=0):
        """Shift signal to other channels. This is particular useful for assigning a mono signal to a specific channel.
            * shift = 0: does nothing as the same signal is being routed to the same position
            * shift > 0: shift channels of self.sig 'right', i.e. from [0,..channels-1] to channels [shift,shift+1,...]
            * shift < 0: shift channels of self.sig 'left', i.e. the first shift channels will be discarded.

        Parameters
        ----------
        shift : int
              shift channel amount (Default value = 0)

        Returns
        -------
        _ : Asig
            Rerouted asig
        """
        if isinstance(shift, int):
            # not optimized method here
            new_sig = np.zeros((self.samples, shift + self.channels))
            _LOGGER.debug(
                "Shift by %d, new signal has %d channels", shift, new_sig.shape[1]
            )
            if self.channels == 1:
                new_sig[:, shift] = self.sig
            elif shift > 0:
                new_sig[:, shift: (shift + self.channels)] = self.sig
            elif shift < 0:
                new_sig[:] = self.sig[:, -shift:]
            if self.cn is None:
                new_cn = self.cn
            else:
                if shift > 0:
                    uname_list = ["unnamed" for i in range(shift)]
                    if isinstance(self.cn, list):
                        new_cn = uname_list + self.cn
                    else:
                        new_cn = uname_list.append(self.cn)
                elif shift < 0:
                    new_cn = self.cn[-shift:]
            return Asig(new_sig, self.sr, label=self.label + "_routed", cn=new_cn)
        else:
            raise AttributeError("Argument needs to be int")

    def mono(self, blend=None):
        """Mix channels to mono signal. Perform sig = np.sum(self.sig_copy * blend, axis=1)

        Parameters
        ----------
        blend : list
            list of gain for each channel as a multiplier.
            Do nothing if signal is already mono, raise warning (Default value = None)

        Returns
        -------
        _ : Asig
            A mono Asig object
        """
        if self.channels == 1:
            warn("Signal is already mono")
            return self
        if blend is None:
            blend = np.ones(self.channels) / self.channels
        if len(blend) != self.channels:
            raise AttributeError("len(blend) != self.channels")
        else:
            sig = np.sum(self.sig * blend, axis=1)
            col_names = [
                self.cn[np.argmax(blend)]] if self.cn is not None else None
            return Asig(sig, self.sr, label=self.label + "_blended", cn=col_names)

    def stereo(self, blend=None):
        """Blend all channels of the signal to stereo. Applicable for any single-/ or multi-channel Asig.

        Parameters
        ----------
        blend : list or None
            Usage: For mono, blend=(g1, g2), the  channel will be broadcated to left, right with g1, g2 gains.
            For stereo signal, blend=(g1, g2), each channel is gain adjusted by g1, g2.
            For multichannel: blend = [[list of gains for left channel], [list of gains for right channel]]
            Default value = None, resulting in equal distribution to left and right channel

        Example
        -------
        asig[:,['c1','c2','c3']].stereo[[1, 0.707, 0], [0, 0.707, 1]]
            mixes channel 'c1' to left, 'c2' to center and 'c3' to right channel
            of a new stereo asig. Note that for equal loudness left**2+right**2=1 should be used

        Returns
        -------
        _ : Asig
            A stereo Asig object
        """
        if blend is None:
            left = 1
            right = 1
        else:
            left = blend[0]
            right = blend[1]

        if self.channels == 1:
            left_sig = self.sig * left
            right_sig = self.sig * right
        elif self.channels == 2:
            left_sig = self.sig[:, 0] * left
            right_sig = self.sig[:, 1] * right
        else:
            if len(left) == self.channels and len(right) == self.channels:
                left_sig = np.sum(self.sig * left, axis=1)
                right_sig = np.sum(self.sig * right, axis=1)
            else:
                msg = """For signal channels > 2, argument blend should be a tuple of two lists,
                        each list contains the gain for each channel to be mixed.
                        """
                raise AttributeError(msg)

        sig = np.stack((left_sig, right_sig), axis=1)
        return Asig(sig, self.sr, label=self.label + "_to_stereo", cn=["l", "r"])

    def rewire(self, dic):
        """Rewire channels to flexibly allow weighted channel permutations.

        Parameters
        ----------
        dic : dict
            key = tuple of (source channel, destination channel)
            value = amplitude gain

        Example
        -------
        {(0, 1): 0.2, (5, 0): 0.4}: rewire channel 0 to 1 with gain 0.2, and 5 to 1 with gain 2
        leaving other channels unmodified

        Returns
        -------
        _ : Asig
            Asig with rewired channels..
        """
        max_ch = max(dic, key=lambda x: x[1])[1] + 1
        if max_ch > self.channels:
            new_sig = np.zeros((self.samples, max_ch))
            new_sig[:, : self.channels] = np.copy(self.sig)
        else:
            new_sig = np.copy(self.sig)
        for key, val in dic.items():
            new_sig[:, key[1]] = self.sig[:, key[0]] * val
        return Asig(new_sig, self.sr, label=self.label + "_rewire", cn=self.cn)

    def pan2(self, pan=0.0):
        """Stereo panning of asig to a stereo output.
        Panning is based on constant power panning, see pan below
        Behavior depends on nr of channels self.channels
        * multi-channel signals (self.channels>2) are cut back to stereo and treated as
        * stereo signals (self.channels==2) are channelwise attenuated using cos(angle), sin(angle)
        * mono signals (self.channels==1) result in stereo output asigs.

        Parameters
        ----------
        pan : float
            panning between -1. (left) to 1. (right)  (Default value = 0.)

        Returns
        -------
        _ : Asig
            Asig
        """
        if isinstance(pan, float) or isinstance(pan, int):
            # Stereo panning.
            if pan <= 1.0 and pan >= -1.0:
                angle = linlin(pan, -1, 1, 0, np.pi / 2.0)
                gain = [np.cos(angle), np.sin(angle)]
                if self.channels == 1:
                    # This is actually quite slow
                    newsig = np.repeat(self.sig, 2)
                    newsig_shape = newsig.reshape(-1, 2) * gain
                    new_cn = [str(self.cn), str(self.cn)]
                    return Asig(
                        newsig_shape,
                        self.sr,
                        label=self.label + "_pan2ed",
                        channels=2,
                        cn=new_cn,
                    )
                else:
                    return Asig(
                        self.sig[:, :2] * gain,
                        self.sr,
                        label=self.label + "_pan2ed",
                        cn=self.cn,
                    )
            else:
                raise ValueError("Panning need to be in the range -1. to 1.")
        else:
            raise TypeError("pan needs to be a float number between -1. to 1.")

    def remove_DC(self):
        """remove DC offset

        Parameters
        ----------
        none

        Returns
        -------
        _ : Asig
            channelwise DC-free Asig.
        """
        sig = self.sig - np.mean(self.sig, 0)
        return Asig(sig, sr=self.sr, label=self.label + "_DCfree", cn=self.cn)

    def norm(self, norm=1, in_db=False, dcflag=False):
        # ToDO add channel_wise argument . default True, currently it is the false.
        """Normalize signal

        Parameters
        ----------
        norm : float
            normalize threshold (Default value = 1)
        in_db : bool
            Normally, norm takes amplitude, if in_db, norm's unit is in dB.
        dcflag : bool
            If true, remove DC offset (Default value = False)

        Returns
        -------
        _ : Asig
            normalized Asig.

        """
        if in_db:
            norm = dbamp(norm)
        if dcflag:
            sig = self.sig - np.mean(self.sig, 0)
        else:
            sig = self.sig
        sig = norm * sig / np.max(np.abs(sig), 0)
        return Asig(sig, self.sr, label=self.label + "_normalised", cn=self.cn)

    def gain(self, amp=None, db=None):
        """Apply gain in amplitude or dB, only use one or the other arguments. Argument can be either a scalar
        or a list (to apply individual gain to each channel). The method returns a new asig with gain applied.

        Parameters
        ----------
        amp : float or None
            Amplitude (Default value = None)
        db : float or int or None
            Decibel (Default value = None)

        Returns
        -------
        _ : Asig
            Gain adjusted Asig.
        """
        if db is not None and amp is not None:
            raise AttributeError("Both amp and db are set, use one only.")
        elif db is not None:  # overwrites amp
            amp = dbamp(db)
        elif amp is None:  # default 1 if neither is given
            amp = 1
        return Asig(self.sig * amp, self.sr, label=self.label + "_scaled", cn=self.cn)

    def rms(self, axis=0):
        """Return signal's RMS

        Parameters
        ----------
        axis : int
            Axis to perform np.mean() on (Default value = 0)

        Returns
        -------
        _ : float
            RMS value
        """
        return np.sqrt(np.mean(np.square(self.sig), axis))

    def plot(
        self,
        fn=None,
        offset=0,
        scale=1,
        x_as_time=True,
        ax=None,
        xlim=None,
        ylim=None,
        **kwargs,
    ):
        """Display signal graph

        Parameters
        ----------
        fn : func or None
            Keyword or function (Default value = None)
        offset : int or float
            Offset each channel to create a stacked view (Default value = 0)
        scale : float
            Scale the y value (Default value = 1)
        xlim : tuple or list
            x axis range (Default value = None)
        ylim : tuple or list
            y axis range (Default value = None)
        **kwargs :
            keyword arguments for matplotlib.pyplot.plot()

        Returns
        -------
        _ : Asig
            self, you can use plt.show() to display the plot.
        """
        if fn:
            if fn == "db":

                def fn(x):
                    return np.sign(x) * ampdb((abs(x) * 2 ** 16 + 1))

            elif not callable(fn):
                msg = "Asig.plot: fn is neither keyword nor function"
                raise AttributeError(msg)
            plot_sig = fn(self.sig)
        else:
            plot_sig = self.sig
        xticks = (
            np.arange(0, self.samples) / self.sr
            if x_as_time
            else np.arange(0, self.samples)
        )
        # From here onward we can abstract it.
        self._["plot"], ax = basicplot(
            plot_sig,
            xticks,
            channels=self.channels,
            cn=self.cn,
            offset=offset,
            scale=scale,
            ax=ax,
            typ="plot",
            xlim=xlim,
            ylim=ylim,
            **kwargs,
        )
        return self

    def get_duration(self):
        """Return the duration in second."""
        return self.samples / self.sr

    def get_times(self):
        """Get time stamps for left-edge of sample-and-hold-signal"""
        return np.linspace(0, (self.samples - 1) / self.sr, self.samples)

    def __eq__(self, other):
        """Check if two asig objects have the same signal. But does not care about sr and others"""
        sig_eq = np.array_equal(self.sig, other.sig)
        sr_eq = self.sr == other.sr
        return sig_eq and sr_eq

    def __repr__(self):
        """Report key attributes"""
        return "Asig('{}'): {} x {} @ {}Hz = {:.3f}s cn={}".format(
            self.label,
            self.channels,
            self.samples,
            self.sr,
            self.samples / self.sr,
            self.cn,
        )

    def __mul__(self, other):
        """Magic method for multiplying. You can either multiply a scalar or an Asig object. If muliplying an Asig,
            you don't always need to have same size arrays as audio signals may different in length. If mix_mode
            is set to 'bound' the size is fixed to respect self. If not, the result will respect to whichever the
            bigger array is."""
        selfsig = self.sig
        othersig = other.sig if isinstance(other, Asig) else other
        if isinstance(othersig, numbers.Number):
            return Asig(
                selfsig * othersig,
                self.sr,
                label=self.label + "_multiplied",
                cn=self.cn,
            )
        else:
            if self.mix_mode == "bound":
                if selfsig.shape[0] > othersig.shape[0]:
                    selfsig = selfsig[: othersig.shape[0]]
                elif selfsig.shape[0] < othersig.shape[0]:
                    othersig = othersig[: selfsig.shape[0]]
                result = selfsig * othersig
                self.mix_mode = None
            elif self.mix_mode == "extend":
                if selfsig.shape[0] > othersig.shape[0]:
                    result = selfsig.copy()
                    result[: othersig.shape[0]] *= othersig

                elif selfsig.shape[0] < othersig.shape[0]:
                    result = othersig.copy()  # might not be deep enough.
                    result[: selfsig.shape[0]] *= selfsig
                else:
                    result = selfsig * othersig
            else:
                result = selfsig * othersig
            return Asig(result, self.sr, label=self.label + "_multiplied", cn=self.cn)

    def __rmul__(self, other):
        return Asig(
            self.sig * other, self.sr, label=self.label + "_multiplied", cn=self.cn
        )

    def __truediv__(self, other):
        """Magic method for division. You can either divide a scalar or an Asig object.
        Use division with caution, audio signal is common to reach 0 or near,
        avoid zero division or extremely large result.

        If dividing an Asig, you don't always need to have same size arrays as audio signals
        may different in length. If mix_mode is set to 'bound' the size is fixed to respect self.
        If not, the result will respect to whichever the bigger array is."""
        selfsig = self.sig
        othersig = other.sig if isinstance(other, Asig) else other
        if isinstance(othersig, numbers.Number):
            return Asig(
                selfsig / othersig,
                self.sr,
                label=self.label + "_multiplied",
                cn=self.cn,
            )
        else:
            if self.mix_mode == "bound":
                if selfsig.shape[0] > othersig.shape[0]:
                    selfsig = selfsig[: othersig.shape[0]]
                elif selfsig.shape[0] < othersig.shape[0]:
                    othersig = othersig[: selfsig.shape[0]]
                result = selfsig / othersig
                self.mix_mode = None
            elif self.mix_mode == "extend":
                if selfsig.shape[0] > othersig.shape[0]:
                    result = selfsig.copy()
                    result[: othersig.shape[0]] /= othersig

                elif selfsig.shape[0] < othersig.shape[0]:
                    # a / b = 1 / (b/a)
                    result = othersig.copy()  # might not be deep enough.
                    result[: selfsig.shape[0]] /= selfsig
                    result = 1.0 / result
                else:
                    result = selfsig / othersig
            else:
                result = selfsig / othersig
            return Asig(result, self.sr, label=self.label + "_divided", cn=self.cn)

    def __rtruediv__(self, other):
        return Asig(
            other / self.sig, self.sr, label=self.label + "_divided", cn=self.cn
        )

    def __add__(self, other):
        """Magic method for adding. You can either add a scalar or an Asig object. If adding an Asig,
        you don't always need to have same size arrays as audio signals may different in length. If mix_mode
        is set to 'bound' the size is fixed to respect self. If not, the result will respect to whichever the
        bigger array is."""
        selfsig = self.sig
        othersig = other.sig if isinstance(other, Asig) else other
        if isinstance(othersig, numbers.Number):  # When other is just a scalar
            return Asig(
                selfsig + othersig, self.sr, label=self.label + "_added", cn=self.cn
            )
        else:
            if self.mix_mode == "bound":
                try:
                    if selfsig.shape[0] > othersig.shape[0]:
                        selfsig = selfsig[: othersig.shape[0]]
                    elif selfsig.shape[0] < othersig.shape[0]:
                        othersig = othersig[: selfsig.shape[0]]
                except AttributeError:
                    pass  # When othersig is just a scalar
                result = selfsig + othersig
                self.mix_mode = None
            elif self.mix_mode == "extend":
                # Make the bigger one
                if selfsig.shape[0] > othersig.shape[0]:
                    result = selfsig.copy()
                    result[: othersig.shape[0]] += othersig

                elif selfsig.shape[0] < othersig.shape[0]:
                    result = othersig.copy()
                    result[: selfsig.shape[0]] += selfsig
                else:
                    result = selfsig + othersig
            else:
                result = selfsig + othersig
            return Asig(result, self.sr, label=self.label + "_added", cn=self.cn)

    def __radd__(self, other):
        return Asig(other + self.sig, self.sr, label=self.label + "_added", cn=self.cn)

    def __sub__(self, other):
        """Magic method for subtraction. You can either minus a scalar or an Asig object. If subtracting an Asig,
        you don't always need to have same size arrays as audio signals may different in length. If mix_mode
        is set to 'bound' the size is fixed to respect self. If not, the result will respect to whichever the
        bigger array is."""
        selfsig = self.sig
        othersig = other.sig if isinstance(other, Asig) else other
        if isinstance(othersig, numbers.Number):  # When other is just a scalar
            return Asig(
                selfsig - othersig,
                self.sr,
                label=self.label + "_subtracted",
                cn=self.cn,
            )
        else:
            if self.mix_mode == "bound":
                try:
                    if selfsig.shape[0] > othersig.shape[0]:
                        selfsig = selfsig[: othersig.shape[0]]
                    elif selfsig.shape[0] < othersig.shape[0]:
                        othersig = othersig[: selfsig.shape[0]]
                except AttributeError:
                    pass  # When othersig is just a scalar
                result = selfsig - othersig
                self.mix_mode = None
            elif self.mix_mode == "extend":
                # Make the bigger one
                if selfsig.shape[0] > othersig.shape[0]:
                    result = selfsig.copy()
                    result[: othersig.shape[0]] -= othersig

                elif selfsig.shape[0] < othersig.shape[0]:
                    # a - b = - (b - a)
                    result = othersig.copy()
                    result[: selfsig.shape[0]] -= selfsig
                    result *= -1
                else:
                    result = selfsig - othersig
            else:
                result = selfsig - othersig
            return Asig(result, self.sr, label=self.label + "_subtracted", cn=self.cn)

    def __rsub__(self, other):
        return Asig(
            other - self.sig, self.sr, label=self.label + "_subtracted", cn=self.cn
        )

    def find_events(
        self,
        step_dur=0.001,
        sil_thr=-20,
        evt_min_dur=0,
        sil_min_dur=0.1,
        sil_pad=[0.001, 0.1],
    ):
        """Locate meaningful 'events' in the signal and create event list. Onset detection.

        Parameters
        ----------
        step_dur : float
            duration in seconds of each search step (Default value = 0.001)
        sil_thr : int
            silent threshold in dB (Default value = -20)
        evt_min_dur : float
            minimum duration to be counted as an event (Default value = 0)
        sil_min_dur : float
            minimum duration to be counted as silent (Default value = 0.1)
        sil_pad : list
            this allows you to add a small duration before and after the actual
            found event locations to the event ranges. If it is a list, you can set the padding (Default value = [0.001)
        0.1] :

        Returns
        -------
        _ : Asig
            This method returns self. But the list of events can be accessed through self._['events']
        """
        if self.channels > 1:
            msg = """warning: works only with single channel.
            Tip:  (1) convert to mono first with asig.mono();
            (2) select individual channel: asig[:,0].find_events"""
            warn(msg)
            return -1
        step_samples = int(step_dur * self.sr)
        sil_thr_amp = dbamp(sil_thr)
        sil_flag = True
        sil_count = 0
        sil_min_steps = int(sil_min_dur / step_dur)
        evt_min_steps = int(evt_min_dur * self.sr)
        if type(sil_pad) is list:
            sil_pad_samples = [int(v * self.sr) for v in sil_pad]
        else:
            sil_pad_samples = (int(sil_pad * self.sr),) * 2
        event_list = []
        for i in range(0, self.samples, step_samples):
            rms = self[i: i + step_samples].rms()
            if sil_flag:
                if rms > sil_thr_amp:  # event found
                    sil_flag = False
                    event_begin = i
                    sil_count = 0
                    continue
            else:
                event_end = i
                if rms < sil_thr_amp:
                    sil_count += 1
                else:
                    sil_count = 0  # reset if there is outlier non-silence
                if sil_count > sil_min_steps:  # event ended
                    # The below line is new.
                    if event_end - event_begin >= evt_min_steps:
                        event_list.append([event_begin - sil_pad_samples[0],
                                           event_end - step_samples * sil_min_steps + sil_pad_samples[1], ]
                                          )
                        sil_flag = True
        self._["events"] = np.array(event_list)
        return self

    def select_event(self, index=None, onset=None):
        """This method can be called after find_event (aka onset detection).

        Parameters
        ----------
        index : int or None
            Index of the event (Default value = None)
        onset : int or None
            Onset of the event (Default value = None)

        Returns
        -------
        _ : Asig
            self
        """
        if "events" not in self._:
            warn("select_event: no events, return all")
            return self
        events = self._["events"]
        if onset:
            index = np.argmin(np.abs(events[:, 0] - onset * self.sr))
        if index is not None:
            beg, end = events[index]
            return Asig(
                self.sig[beg:end],
                self.sr,
                label=self.label + f"event_{index}",
                cn=self.cn,
            )
        warn("select_event: neither index nor onset given: return self")
        return self

    def plot_events(self):
        try:
            plt.plot(self.sig)
            for event in self._["events"]:
                plt.axvline(x=event[0])
                plt.axvline(x=event[1], color="r")
        except KeyError:
            raise ValueError(
                "No events found, use find_events() before plotting.")

    def fade_in(self, dur=0.1, curve=1):
        """Fade in the signal at the beginning

        Parameters
        ----------
        dur : float
            Duration in seconds to fade in (Default value = 0.1)
        curve : float
            Curvature of the fader, power of the linspace function. (Default value = 1)

        Returns
        -------
        _ : Asig
            Asig, new asig with the fade in signal
        """
        nsamp = int(dur * self.sr)
        if nsamp > self.samples:
            nsamp = self.samples
            warn("warning: Asig too short for fade_in - adapting fade_in time")
        # TODO simplify this if we decide to make mono signal with dimension 1 instead of None.
        if self.channels == 1:
            ramp = np.linspace(0, 1, nsamp, dtype="float32") ** curve
            return Asig(
                np.hstack((self.sig[:nsamp] * ramp, self.sig[nsamp:])),
                self.sr,
                label=self.label + "_fadein",
                cn=self.cn,
            )
        else:
            ramp = np.meshgrid(
                np.linspace(0, 1, nsamp, dtype="float32"), np.zeros(
                    self.channels)
            )[0].T
            return Asig(
                np.vstack((self.sig[:nsamp] * ramp, self.sig[nsamp:])),
                self.sr,
                label=self.label + "_fadein",
                cn=self.cn,
            )

    def fade_out(self, dur=0.1, curve=1):
        """Fade out the signal at the end

        Parameters
        ----------
        dur : float
            duration in seconds to fade out (Default value = 0.1)
        curve : float
            Curvature of the fader, power of the linspace function. (Default value = 1)

        Returns
        -------
        _ : Asig
            Asig, new asig with the fade out signal
        """
        nsamp = int(dur * self.sr)
        if nsamp > self.samples:
            nsamp = self.samples
            warn("Asig too short for fade_out - adapting fade_out time")
        # TODO simplify this if we decide to make mono signal with dimension 1 instead of None.
        if self.channels == 1:
            ramp = np.linspace(1, 0, nsamp, dtype="float32") ** curve
            return Asig(
                np.hstack((self.sig[:-nsamp], self.sig[-nsamp:] * ramp)),
                self.sr,
                label=self.label + "_fadeout",
                cn=self.cn,
            )
        else:
            ramp = np.meshgrid(
                np.linspace(1, 0, nsamp, dtype="float32"), np.zeros(
                    self.channels)
            )[0].T
            return Asig(
                np.vstack((self.sig[:-nsamp], self.sig[-nsamp:] * ramp)),
                self.sr,
                label=self.label + "_fadeout",
                cn=self.cn,
            )

    def iirfilter(
        self,
        cutoff_freqs,
        btype="bandpass",
        ftype="butter",
        order=4,
        filter="lfilter",
        rp=None,
        rs=None,
    ):
        """iirfilter based on scipy.signal.iirfilter

        Parameters
        ----------
        cutoff_freqs : float or [float, float]
            Cutoff frequency or frequencies.
        btype : str
            Filter type (Default value = 'bandpass')
        ftype : str
            Tthe type of IIR filter. e.g. 'butter', 'cheby1', 'cheby2', 'elip', 'bessel' (Default value = 'butter')
        order : int
            Filter order (Default value = 4)
        filter : str
            The scipy.signal method to call when applying the filter coeffs to the signal.
            by default it is set to scipy.signal.lfilter (one-dimensional).
        rp : float
            For Chebyshev and elliptic filters, provides the maximum ripple in the passband. (dB) (Default value = None)
        rs : float
            For Chebyshev and elliptic filters, provides the minimum attenuation in the stop band. (dB) (Default value = None)

        Returns
        -------
        _ : Asig
            new Asig with the filter applied. also you can access b, a coefficients by doing self._['b']
            and self._['a']

        """
        # TODO scipy.signal.__getattribute__ error
        Wn = np.array(cutoff_freqs) * 2 / self.sr
        b, a = scipy.signal.iirfilter(
            order, Wn, rp=rp, rs=rs, btype=btype, ftype=ftype)
        y = scipy.signal.__getattribute__(filter)(b, a, self.sig, axis=0)
        aout = Asig(y, self.sr, label=self.label + "_iir")
        aout._["b"] = b
        aout._["a"] = a
        _LOGGER.debug("Filter applied.")
        return aout

    def plot_freqz(self, worN, **kwargs):
        """Plot the frequency response of a digital filter. Perform scipy.signal.freqz then plot the response.

        TODO
        Parameters
        ----------
        worN :

        **kwargs :


        Returns
        -------

        """
        w, h = scipy.signal.freqz(self._["b"], self._["a"], worN)
        plt.plot(w * self.sr / 2 / np.pi, ampdb(abs(h)), **kwargs)

    def envelope(self, amps, ts=None, curve=1, kind="linear"):
        """Create an envelop and multiply by the signal.

        Parameters
        ----------
        amps : array
            Amplitude of each breaking point
        ts : array
            Indices of each breaking point (Default value = None)
        curve : int
            Affecting the curvature of the ramp. (Default value = 1)
        kind : str
            The type of interpolation (Default value = 'linear')

        Returns
        -------
        _ : Asig
            Returns a new asig with the enveloped applied to its signal array

        """
        # TODO Check multi-channels.
        nsteps = len(amps)
        duration = self.samples / self.sr
        if nsteps == self.samples:
            sig_new = self.sig * amps ** curve
        else:
            if not ts:
                given_ts = np.linspace(0, duration, nsteps)
            else:
                if nsteps != len(ts):
                    _LOGGER.error("Asig.envelope error: len(amps)!=len(ts)")
                    return self
                if all(
                    ts[i] <= ts[i + 1] for i in range(len(ts) - 1)
                ):  # if list is monotonous
                    if (
                        ts[0] > 0
                    ):  # if first t > 0 extend amps/ts arrays prepending item
                        ts = np.insert(np.array(ts), 0, 0)
                        amps = np.insert(np.array(amps), 0, amps[0])
                    if ts[-1] < duration:  # if last t < duration append amps/ts value
                        ts = np.insert(np.array(ts), -1, duration)
                        amps = np.insert(np.array(amps), -1, amps[-1])
                else:
                    raise AttributeError("Asig.envelope error: ts not sorted")
                given_ts = ts
            if nsteps != self.samples:
                interp_fn = scipy.interpolate.interp1d(
                    given_ts, amps, kind=kind)
                sig_new = (self.sig * interp_fn(np.linspace(0, duration, self.samples)) ** curve)
        return Asig(sig_new, self.sr, label=self.label + "_enveloped", cn=self.cn)

    def adsr(self, att=0, dec=0.1, sus=0.7, rel=0.1, curve=1, kind="linear"):
        """Create and applied a ADSR evelope to signal.

        Parameters
        ----------
        att : float
            attack (Default value = 0)
        dec : float
            decay (Default value = 0.1)
        sus : float
            sustain (Default value = 0.7)
        rel : float
            release. (Default value = 0.1)
        curve : int
            affecting the curvature of the ramp. (Default value = 1)
        kind : str
            The type of interpolation (Default value = 'linear')

        Returns
        -------
        _ : Asig
            returns a new asig with the enveloped applied to its signal array

        """
        dur = self.get_duration()
        return self.envelope(
            [0, 1, sus, sus, 0],
            [0, att, att + dec, dur - rel, dur],
            curve=curve,
            kind=kind,
        )

    def window(self, win="triang", **kwargs):
        """Apply windowing to self.sig

        Parameters
        ----------
        win : str
            Type of window check scipy.signal.get_window for avaiable types. (Default value = 'triang')
        **kwargs :
            keyword arguments for scipy.signal.get_window()

        Returns
        -------
        _ : Asig
            new asig with window applied.

        """
        if not win:
            return self
        winstr = win
        if type(winstr) == tuple:
            winstr = win[0]
        if self.channels == 1:
            return Asig(self.sig * scipy.signal.get_window(win, self.samples, **kwargs),
                        self.sr, label=self.label + "_" + winstr, cn=self.cn)
        else:
            for i in range(self.channels):
                self.sig[:,
                         i] *= scipy.signal.get_window(win, self.samples, **kwargs)
            return Asig(self.sig, self.sr, label=self.label + "_" + winstr, cn=self.cn)

    def window_op(self, nperseg=64, stride=32, win=None, fn="rms", pad="mirror"):
        """TODO add docstring

        Parameters
        ----------
        nperseg :
             (Default value = 64)
        stride :
             (Default value = 32)
        win :
             (Default value = None)
        fn :
             (Default value = 'rms')
        pad :
             (Default value = 'mirror')

        Returns
        -------

        """
        centerpos = np.arange(0, self.samples, stride)
        nsegs = len(centerpos)

        # Probably can simplify this.
        res = (
            np.zeros((nsegs,))
            if self.channels == 1
            else np.zeros((nsegs, self.channels))
        )
        for i, cp in enumerate(centerpos):
            i0 = cp - nperseg // 2
            i1 = cp + nperseg // 2
            if i0 < 0:
                i0 = 0  # ToDo: correct padding!!!
            if i1 >= self.samples:
                i1 = self.samples - 1  # ToDo: correct padding!!!
            if self.channels == 1:
                if isinstance(fn, str):
                    res[i] = self[i0:i1].window(win).__getattribute__(fn)()
                else:  # assume fn to be a function on Asig
                    res[i] = fn(self[i0:i1])
            else:
                if isinstance(fn, str):
                    res[i, :] = self[i0:i1].window(win).__getattribute__(fn)()
                else:  # assume fn to be a function on Asig
                    res[i, :] = fn(self[i0:i1])
        return Asig(
            np.array(res), sr=self.sr // stride, label="window_oped", cn=self.cn
        )

    def overlap_add(
        self,
        nperseg=64,
        stride_in=32,
        stride_out=32,
        jitter_in=None,
        jitter_out=None,
        win=None,
        pad="mirror",
    ):
        """TODO

        Parameters
        ----------
        nperseg :
             (Default value = 64)
        stride_in :
             (Default value = 32)
        stride_out :
             (Default value = 32)
        jitter_in :
             (Default value = None)
        jitter_out :
             (Default value = None)
        win :
             (Default value = None)
        pad :
             (Default value = 'mirror')

        Returns
        -------

        """
        # TODO: check with multichannel ASigs
        # TODO: allow stride_in and stride_out to be arrays of indices
        # TODO: add jitter_in, jitter_out parameters to reduce spectral ringing effects
        res = Asig(
            np.zeros((self.samples // stride_in * stride_out,)),
            sr=self.sr,
            label=self.label + "_ola",
            cn=self.cn,
        )
        ii = 0
        io = 0
        for _ in range(self.samples // stride_in):
            i0 = ii - nperseg // 2
            if jitter_in:
                i0 += np.random.randint(jitter_in)
            i1 = i0 + nperseg
            if i0 < 0:
                i0 = 0  # TODO: correct left zero padding!!!
            if i1 >= self.samples:
                i1 = self.samples - 1  # ToDo: correct right zero padding!!!
            pos = io
            if jitter_out:
                pos += np.random.randint(jitter_out)
            res.add(self[i0:i1].window(win).sig, pos=pos)
            io += stride_out
            ii += stride_in
        return res

    def to_spec(self):
        """Return Aspec object which is the rfft of the signal."""
        return pya.aspec.Aspec(self)

    def to_stft(self, **kwargs):
        """Return Astft object which is the stft of the signal. Keyword arguments are the arguments for
        scipy.signal.stft(). """
        return pya.astft.Astft(self, **kwargs)

    def to_mfcc(
        self,
        n_per_frame=None,
        hopsize=None,
        nfft=None,
        window="hann",
        nfilters=26,
        ncep=13,
        ceplifter=22,
        preemph=0.95,
        append_energy=True,
    ):
        """Return Amfcc object. """
        return pya.amfcc.Amfcc(
            self,
            label=self.label,
            n_per_frame=n_per_frame,
            hopsize=hopsize,
            nfft=nfft,
            window=window,
            nfilters=nfilters,
            ncep=ncep,
            ceplifter=ceplifter,
            preemph=preemph,
            append_energy=append_energy,
            cn=self.cn,
        )

    def plot_spectrum(self, offset=0, scale=1.0, xlim=None, **kwargs):
        """Plot spectrum of the signal

        Parameters
        ----------
        offset : float
            If self.sig is multichannels, this will offset each
            channels to create a stacked view for better viewing (Default value = 0.)
        scale : float
            scale the y_axis (Default value = 1.)
        xlim : tuple
            range of x_axis (Default value = None)
        **kwargs :
            keywords arguments for matplotlib.pyplot.plot()

        Returns
        -------
        _ : Asig
            self
        """
        frq, Y = spectrum(self.sig, self.samples, self.channels, self.sr)
        if self.channels == 1 or (offset == 0 and scale == 1):
            plt.subplot(211)
            plt.plot(frq, np.abs(Y), **kwargs)
            if xlim is not None:
                plt.xlim(xlim[0], xlim[1])
            plt.subplot(212)
            self._["lines"] = plt.plot(frq, np.angle(Y), "b.", markersize=0.2)
            if xlim is not None:
                plt.xlim(xlim[0], xlim[1])

        else:
            # For multichannel signals, a stacked view is plotted
            plt.subplot(211)
            max_y = np.max(np.abs(Y))
            p = []
            for i in range(Y.shape[1]):
                # + i * offset * max_y
                p.append(
                    plt.plot(
                        frq, np.abs(Y[:, i]) * scale + i * offset * max_y, **kwargs
                    )
                )
                plt.text(0, (i + 0.1) * offset * max_y, self.cn[i])
            plt.xlabel("freq (Hz)")
            plt.ylabel("|F(freq)|")
            if xlim is not None:
                plt.xlim(xlim[0], xlim[1])
            plt.subplot(212)
            p = []
            for i in range(Y.shape[1]):
                p.append(
                    plt.plot(
                        frq,
                        np.angle(Y[:, i]) * scale + i * offset * np.pi * 2,
                        "b.",
                        markersize=0.2,
                    )
                )
                plt.text(0, (i + 0.1) * offset * np.pi * 2, self.cn[i])
            plt.ylabel("Angle")
            if xlim is not None:
                plt.xlim(xlim[0], xlim[1])
        return self

    def spectrogram(self, *argv, **kvarg):
        """Perform sicpy.signal.spectrogram and returns: frequencies, array of times, spectrogram
        """
        freqs, times, Sxx = scipy.signal.spectrogram(
            self.sig, self.sr, *argv, **kvarg)
        return freqs, times, Sxx

    def get_size(self):
        """Return signal array shape and duration in seconds."""
        return self.sig.shape, self.sig.shape[0] / self.sr

    # TODO this method is currently commented.
    # def vstack(self, chan):
    #     """Create multichannel signal from mono"""
    #     self.sig = np.vstack([self.sig] * chan)
    #     self.sig = self.sig.transpose()
    #     return self.overwrite(self.sig, self.sr)  # Overwrite the signal
    #     # TODO: replace this (old) overwrite by a hidden private _transplant_sig(ndarr, sr)
    #     # since overwrite is now a property for setitem...

    def append(self, asig, amp=1):
        """Apppend an asig with another. Conditions: the appended asig should have the same channels. If
        appended asig has a different sampling rate, resample it to match the orginal.

        Parameters
        ----------
        asig : Asig
            object to append
        amp : float or int
            aplitude (Default value = 1)

        Returns
        -------
        _ : Asig
            Appended Asig object
        """
        if self.channels != asig.channels:
            raise AttributeError("Asig.append: channels don't match")
        if self.sr != asig.sr:
            _LOGGER.info("resampling appended signal")
            atmp = asig.resample(self.sr)
        else:
            atmp = asig
        return Asig(
            np.hstack((self.sig, atmp.sig)),
            self.sr,
            label=self.label + "+" + asig.label,
            cn=self.cn,
        )

    def add(self, sig, pos=None, amp=1, onset=None):
        """Add a signal

        Parameters
        ----------
        sig : asig
            Signal to add
        pos : int, None
            Postion to add (Default value = None)
        amp : float
            Aplitude (Default value = 1)
        onset : float or None
            Similar to pos but in time rather sample,
            given a value to this will overwrite pos (Default value = None)

        Returns
        -------
        _ : Asig
            Asig with the added signal.

        """
        if type(sig) == Asig:
            n = sig.samples
            sr = sig.sr
            sigar = sig.sig
            if sig.channels != self.channels:
                raise AttributeError("Argument asig has different channels.!")
            if sr != self.sr:
                raise AttributeError(
                    "Dangerous operation, samping rate (sr) not matched. Use resample() first."
                )
        else:
            n = np.shape(sig)[0]
            sr = self.sr  # assume same sr as self
            sigar = sig
        if onset:  # onset overwrites pos, time has priority
            pos = int(onset * self.sr)
        if not pos:
            pos = 0  # add to begin if neither pos nor onset have been specified
        last = pos + n
        if last > self.samples:
            last = self.samples
            sigar = sigar[: last - pos]
        self.sig[pos:last] += amp * sigar
        return self

    def flatten(self):
        """Flatten a multidimentional array into a vector using np.ravel()"""
        return Asig(
            self.sig.ravel(),
            sr=self.sr,
            label=self.label + "_flattened",
            channels=1,
            cn=None,
        )

    def pad(self, width=0, tail=True, constant_values=0, dur=None):
        """Pads the signal

        Parameters
        ----------
        width : int
            The number of samples to append to the tail or head of the array.
        tail : bool
            By default it is True, if trail pad to the end, else pad to the start.
        constant_values: float32
            value to be padded, defaults to 0
        dur : float
            duration to be padded;  if specified, it overrides the parameter width

        Returns
        -------
        _ : Asig
            Asig of the pad signal.
        """
        if dur:
            width = self.sr * dur
        return Asig(
            padding(self.sig, width, tail=tail),
            sr=self.sr,
            label=self.label + "_padded",
            channels=self.channels,
            cn=self.cn,
        )

    def custom(self, func, **kwargs):
        """custom function method. TODO add example"""
        func(self, **kwargs)
        return self

    def convolve(self, sig, mode="full", method="fft", norm="amp"):
        """Convolution based on scipy.signal.convolve.

        Parameters
        ----------
            ins : Asig or array_like
                Input signal to convolve with, e.g. impulse response.
            mode : str {'full', 'valid', 'same'}, optional
                A string indicating the size of the output:
                full
                    The output is the full discrete linear convolution of the inputs. (Default)
                valid
                    The output consists only of those elements that do not rely on the zero-padding.
                    self.sr or ins must be at least as large as the other in every dimension.
                same
                    The output is the same size as self.sr, centered with respect to the full output.
            method : str {'auto', 'direct', 'fft'}, optional
                A string indicating which method to use to calculate the convolution
                direct
                    Compute directly from sums, the definition of convolution
                fft (default)
                    The Fourier Transform is used to perform the convolution by calling fftconvolve
                auto
                    Automatically chooses direct or Fourier method based on an estimate of which is faster.
            norm : str, optional
                If "amp" (default value), the result signal will have the same peak as the original signal.
                If "none" or other unrecognized name, no normalization is applied.

        Returns
        -------
        _ : Asig
            A new asig with convlved signal. The size will depends on mode.
        """
        if isinstance(sig, Asig):
            if sig.sr != self.sr:
                _LOGGER.warning(
                    "sampling rate not matched, perform resampling...")
                sig = sig.resample(target_sr=self.sr)
            sig_array = sig.sig
            sig_size = sig.samples
            sig_channels = sig.channels
        elif isinstance(sig, np.ndarray):
            sig_array = sig
            sig_size = len(sig)
            sig_channels = 1 if sig_array.ndim == 1 else sig_array.shape[1]
        else:
            raise TypeError(
                "Illegal type. ir must be an Asig object or an array.")
        # Compare size of A B, and pad zeros if needed.
        if self.samples > sig_size:
            # pad ir
            sig_array = padding(sig_array, width=self.samples - sig_size)
        else:
            # pad source
            asig = padding(self.sig, width=sig_size - self.samples)

        # Now perform convolution
        if self.channels == 1:
            # If sig is a mono signal:
            result = np.array(
                scipy.signal.convolve(
                    self.sig, sig_array, mode=mode, method=method)
            )
        else:
            if sig_channels > 1 and self.channels != sig_channels:
                raise ValueError(
                    "input signal needs to have the same amount of channels as self.sig."
                )
            # Perform conv on each channel.
            for i in range(self.channels):
                a = self.sig[:, i]
                b = sig_array if sig_channels == 1 else sig_array[:, i]
                r_1ch = np.array(scipy.signal.convolve(
                    a, b, mode=mode, method=method))
                if i == 0:
                    result = np.zeros((len(r_1ch), self.channels))
                result[:, i] = r_1ch
        # Not sure if this is the best way to regulate output volume
        if norm.lower() == "amp":
            result = result / np.max(np.abs(result)) * np.max(np.abs(self.sig))
        # elif norm.lower() == "energy":
        #     result = result / np.var(result, axis=0) * np.var(self.sig, axis=0)
        return Asig(
            result, sr=self.sr, label=self.label, channels=self.channels, cn=self.cn
        )

    def apply(self, fn):
        """apply a fn samplewise for distortion or waveshaping a signal.

        Parameters
        ----------
            fn : scalar function
                function to be used for wave shaping.

        Returns
        -------
        _ : Asig
            A new asig with samplewise applied fn.
        """
        vecfn = np.vectorize(fn)
        sig_out = vecfn(self.sig)
        return Asig(sig_out, sr=self.sr, label=self.label, channels=self.channels, cn=self.cn)
