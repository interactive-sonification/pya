from .Aserver import Aserver
import numbers
from warnings import warn
import logging
from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.signal
from scipy.fftpack import fft, fftfreq, ifft
from scipy.io import wavfile
from .helpers import ampdb, dbamp, linlin, timeit, spectrum, audioread_load
from copy import copy, deepcopy

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class Asig:
    """Audio signal class. Asig enables manipulation of audio signals in the style of numpy and more. 
    Asig offer functions for plotting (via matplotlib) and playing audio (using the pya.Aserver class) 

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
        channel names can be used to subset signal channels in a more readible way, 
        e.g. asig[:, ['left', 'front']] subsets the left and front channels of the signal. 
    mix_mode : str or None
        used to extend numpy __setitem__() operation to frequent audio manipulations such as
        mixing, extending, clipping, replacing. Current Asig supports the mix_modes: 
        bound, extend, overwrite.  mix_mode should not be set directly but is set temporarilty when using 
        the .bound, .extend and .overwrite properties.
    """

    def __init__(self, sig, sr=44100, label="", channels=1, cn=None):
        """__init__ method

        Parameters
        ----------
            sig: numpy.array or int or float or str
                numpy.array for audio signal, str for filepath, int create x samples of silence, 
                float creates x seconds of seconds.
            sr : int 
                Sampling rate
            label : str 
                Label for the object
            channels : int
                Number of channels, no need to set it if you already have a signal for the sig argument.
            cn : list or None
                A list of channel names, size should match the channels.
        """
        self.sr = sr
        self.mix_mode = None
        self._ = {}  # dictionary for further return values
        if isinstance(sig, str):
            self.load_audio_file(sig)
        elif isinstance(sig, int):  # sample length
            if channels == 1:
                self.sig = np.zeros(sig).astype("float32")
            else:
                self.sig = np.zeros((sig, channels)).astype("float32")
        elif isinstance(sig, float):  # if float interpret as duration
            if channels == 1:
                self.sig = np.zeros(int(sig * sr)).astype("float32")
            else:
                self.sig = np.zeros((int(sig * sr), channels)).astype("float32")
        else:
            self.sig = np.array(sig).astype("float32")
        self.label = label
        self.cn = cn
        self._set_col_names()

    @property
    def channels(self):
        """Channel property"""
        try:
            return self.sig.shape[1]
        except IndexError:
            return 1

    @property
    def samples(self):
        """Return the length of signal in samples"""
        return np.shape(self.sig)[0]  # Update it.

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
                if all(isinstance(x, str) for x in val):  # check if all elements are str
                    self._cn = val
                else:
                    raise TypeError("channel names cn need to be a list of string(s).")
            else:
                raise ValueError("list size doesn't match channel numbers {}".format(self.channels))

    def load_audio_file(self, fname):
        """Load WAV or MP3 file, and set self.sig to the signal and self.sr to the sampling rate. 

        Parameters
        ----------
        fname : str
            Path to file. 
            '.wav' files are loaded using scipy.signal.io.wavfile.read().
            '.mp3' files are loaded using the (optional) audioread package, if available.
            Other formats will be added later as needed.
        """
        if fname.endswith('.wav'):
            self.sr, self.sig = wavfile.read(fname)  # load the sample data
            if self.sig.dtype == np.dtype('int16'):
                self.sig = (self.sig / 32768.).astype('float32')

            elif self.sig.dtype != np.dtype('float32'):
                self.sig = self.sig.astype('float32')
            else:
                warn("Unsupported data type.")
        elif fname.endswith('.mp3'):
            try:
                self.sig, self.sr = audioread_load(fname, dtype=np.float32)
            except FileNotFoundError:
                raise FileNotFoundError("File or directory not found")
            except Exception:   # The other error is likely to be backend error from ffmpeg not installed or added to path
                raise ValueError("Can't find a supported backend to encode mp3. This is likely to happen on Windows and Linux os" 
                " if FFmpeg is not installed. Please refers to pyA github or README.md for installation guide.")
        else:
            raise AttributeError("Unsupported file format, use WAV or MP3.")

    def save_wavfile(self, fname="asig.wav", dtype='float32'):
        """Save signal as .wav file

        Parameters
        ----------
        fname : str
            name of the file with .wav (Default value = "asig.wav")
        dtype : str
            datatype (Default value = 'float32')

        Returns
        -------
        : Asig
        """
        if dtype == 'int16':
            data = (self.sig * 32767).astype('int16')
        elif dtype == 'int32':
            data = (self.sig * 2147483647).astype('int32')
        elif dtype == 'uint8':
            data = (self.sig * 127 + 128).astype('uint8')
        elif dtype == 'float32':
            data = self.sig.astype('float32')
        scipy.io.wavfile.write(fname, self.sr, data)
        return self

    def _set_col_names(self):
        # Currently, every newly returned asig has a cn argument that is the same as self.
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
            * slice, range and step slicing asig[4:40:2]  # from 4 to 40 every 2 samples;
            * list, subset rows, asig[[2, 4, 6]]  # pick out index 2, 4, 6 as a new asig
            * tuple, row and column specific slicing, asig[4:40, 3:5]  # from 4 to 40, channel 3 and 4
            * Time slicing (unit in seconds) using dict asig[{1:2.5}, :] creates indexing of 1s to 2.5s.
            * Channel name slicing: asig['l'] returns channel 'l' as a new mono asig. asig[['front', 'rear']], etc...
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
            return self._[index]  # ToDo: decide whether to solve differently, e.g. only via ._[str] or via a .attribute(str) fn 
        else:  # if only slice, list, dict, int or float given for row selection
            rindex = index
            cindex = None

        # parse row index rindex into ridx
        if isinstance(rindex, list):  # e.g. a[[4,5,7,8,9]], or a[[True, False, True...]]
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
            _LOGGER.debug("Time slicing, start: %s, stop: %s", str(start), str(stop))
        else:  # Dont think there is a usecase.
            ridx = rindex
            sr = self.sr

        # now parse cindex
        if isinstance(cindex, list):
            _LOGGER.debug(" getitem: column index is list.")
            if isinstance(cindex[0], str):
                cidx = [self.col_name.get(s) for s in cindex]
                if cidx is None:
                    _LOGGER.error("Input column names does not exist")
                cn_new = [self.cn[i] for i in cidx] if self.cn is not None else None
            elif isinstance(cindex[0], bool):
                cidx = cindex
                cn_new = list(compress(self.cn, cindex))
            elif isinstance(cindex[0], int):
                cidx = cindex
                cn_new = [self.cn[i] for i in cindex] if self.cn is not None else None
        elif isinstance(cindex, int):
            _LOGGER.debug(" getitem: column index is int.")
            cidx = cindex
            cn_new = [self.cn[cindex]] if self.cn is not None else None
        elif isinstance(cindex, slice):
            _LOGGER.debug(" getitem: column index is slice.")
            cidx = cindex
            cn_new = self.cn[cindex] if self.cn is not None else None
        elif isinstance(cindex, str):  # if only a single channel name is given.
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
        # (10,) and (10, 1) respectively. Which should be dealt with individually.
        if sig.ndim == 2 and sig.shape[1] == 1:
            if not isinstance(cindex[0], bool):  # Hot fix this to be consistent with bool slciing
                _LOGGER.debug("ndim is 2 and channel num is 1, performa np.squeeze")
                sig = np.squeeze(sig)
        if isinstance(sig, numbers.Number):
            _LOGGER.debug("signal is scalar, convert to array")
            sig = np.array(sig)

        a = Asig(sig, sr=sr, label=self.label + '_arrayindexed', cn=cn_new)
        a.mix_mode = self.mix_mode
        return a

    # new setitem implementation (TH): in analogy to new __getitem__ and with mix modes
    # work in progress

    @property
    def x(self):
        """Extend mode: this mode allows destination sig size in assignment to be extended through setitem"""
        # Set setitem mode to extend
        self.mix_mode = 'extend'
        return self
    extend = x  # better readable synonym

    @property
    def b(self):
        """Bound mode: this mode allows to truncate a source signal in assignment to a limited destination in setitem."""
        # Set setitem mode to bound
        self.mix_mode = 'bound'
        return self
    bound = b  # better readable synonym

    @property
    def o(self):
        """Overwrite mode: this mode cuts and replaces target selection by source signal on assignment via setitem"""
        self.mix_mode = 'overwrite'
        return self
    overwrite = o

    def __setitem__(self, index, value):
        """setitem: asig[index] = value. This allows all the methods from getitem:
            * numpy style slicing
            * string/string_list slicing for subsetting channels based on channel name self.cn
            * time slicing (unit seconds) via dict.
            * bool slicing to filter out specific channels.
        In addition, there are 4 possible modes: (referring to asig as 'dest', and value as 'src'
            1. standard pythonic way that the src und dest dimensions need to match
                asig[...] = value
            2. bound mode where src is copied up to the bounds of dest
                asig.b[...] = value
            3. extend mode where dest is dynamically extended to make space for src
                asig.x[...] = value
            4. overwrite mode where selected dest subset is replaced by specified src regardless the length.
                asig.o[...] = value

        row index:
            * list: e.g. [1,2,3,4,5,6,7,8] or [True, ..., False] (modes b and x possible)
            * int:  e.g. 0  (i.e. a single sample, so no need for extra modes)
            * slice: e.g. 100:5000:2  (can be used with all modes)
            * dict: e.g. {0.5: 2.5}   (modes o, b possible, x only if step==1, or if step==None and stop=None)

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
        else:  # if only slice, list, dict, int or float given for row selection
            rindex = index
            cindex = None

        # parse row index rindex into ridx
        # sr = self.sr  # unused default case for conversion if not changed by special case
        if isinstance(rindex, list):  # e.g. a[[4,5,7,8,9]], or a[[True, False, True...]]
            ridx = rindex
        elif isinstance(rindex, int):  # picking a single row
            ridx = rindex
        elif isinstance(rindex, slice):
            # _, _, step = rindex.indices(len(self.sig))
            # sr = int(self.sr / abs(step))  # This is unused.
            ridx = rindex
        elif isinstance(rindex, dict):  # time slicing
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
        else:  # Dont think there is a usecase.
            ridx = rindex

        # now parse cindex
        if isinstance(cindex, list):
            if isinstance(cindex[0], str):
                cidx = [self.col_name.get(s) for s in cindex]
                cidx = cidx[0] if len(cidx) == 1 else cidx  # hotfix for now.
            elif isinstance(cindex[0], bool):
                cidx = cindex
            elif isinstance(cindex[0], int):
                cidx = cindex
        elif isinstance(cindex, int) or isinstance(cindex, slice):  # int, slice are the same.
            cidx = cindex
        elif isinstance(cindex, str):  # if only a single channel name is given.
            cidx = self.col_name.get(cindex)
        else:
            cidx = slice(None)
            # cidx = None

        _LOGGER.debug("self.sig.ndim == %d", self.sig.ndim)
        if self.sig.ndim == 1:
            final_index = (ridx)
        else:
            final_index = (ridx, cidx)
        # apply setitem: set dest[ridx,cidx] = src return self

        if isinstance(value, Asig):
            _LOGGER.debug("value is asig")
            src = value.sig

        elif isinstance(value, np.ndarray):  # numpy array if not Asig, default: sr fits
            _LOGGER.debug("value is ndarray")
            src = value

        elif isinstance(value, list):  # if list
            _LOGGER.debug("value is list")
            src = value     
            mode = None   # for list (assume values for channels), mode makes no sense...
            # TODO: check if useful behavior also for numpy arrays
        else:
            _LOGGER.debug("value not asig, ndarray, list")
            src = value
            mode = None  # for scalar types, mode makes no sense...

        if mode is None:
            _LOGGER.debug("Default setitem mode")
            if isinstance(src, numbers.Number):
                self.sig[final_index] = src
            elif isinstance(src, list):  # for multichannel signals that is value for each column
                self.sig[final_index] = src
            else:  # numpy.ndarray
                try:
                    self.sig[final_index] = np.broadcast_to(src, self.sig[final_index].shape)
                except ValueError:
                    self.sig[final_index] = src

        elif mode == 'bound':
            _LOGGER.debug("setitem bound mode")
            dshape = self.sig[final_index].shape
            dn = dshape[0]  # ToDo: howto get that faster from ridx alone?
            sn = src.shape[0]
            if sn > dn:
                self.sig[final_index] = src[:dn] if len(dshape) == 1 else src[:dn, :]
            else:
                self.sig[final_index][:sn] = src if len(dshape) == 1 else src[:, :]

        elif mode == 'extend':
            _LOGGER.debug("setitem extend mode")
            if isinstance(ridx, list):
                _LOGGER.error("Asig.setitem Error: extend mode not available for row index list")
                return self
            if isinstance(ridx, slice):
                if ridx.step not in [1, None]:
                    _LOGGER.error("Asig.setitem Error: extend mode only available for step-1 slices")
                    return self
                if ridx.stop is not None:
                    _LOGGER.error("Asig.setitem Error: extend mode only available if stop is None")
                    return self
            dshape = self.sig[final_index].shape
            dn = dshape[0]  # ToDo: howto compute dn faster from ridx shape(self.sig) alone?
            sn = src.shape[0]
            if sn <= dn:  # same as bound, since src fits in
                self.sig[final_index][:sn] = np.broadcast_to(src, (sn,) + dshape[1:])
            elif sn > dn:
                self.sig[final_index] = src[:dn]
                # now extend by nn = sn-dn additional rows
                if dn > 0:
                    nn = sn - dn  # nr of needed additional rows
                    self.sig = np.r_[self.sig, np.zeros((nn,) + self.sig.shape[1:])]
                    if self.sig.ndim == 1:
                        self.sig[-nn:] = src[dn:]
                    else:
                        self.sig[-nn:, cidx] = src[dn:]
                else:  # this is when start is beyond length of dest...
                    nn = ridx.start + sn
                    self.sig = np.r_[
                        self.sig, np.zeros((nn - self.sig.shape[0],) + self.sig.shape[1:])]
                    if self.sig.ndim == 1:
                        self.sig[-sn:] = src
                    else:
                        self.sig[-sn:, cidx] = src

        elif mode == 'overwrite':
            # This mode is to replace a subset with an any given shape.
            # Where the end point of the newly insert signal should be.
            _LOGGER.info("setitem overwrite mode")
            start_idx = ridx.start if isinstance(ridx, slice) else 0  # Start index of the ridx,
            stop_idx = ridx.stop if isinstance(ridx, slice) else 0  # Stop index of the rdix
            end = start_idx + src.shape[0] 
            # Create a new signal
            # New row is: original samples + (new_signal_sample - the range to be replace)
            sig = np.ndarray(shape=(self.sig.shape[0] + src.shape[0] - (stop_idx - start_idx), self.channels))
            if sig.ndim == 2 and sig.shape[1] == 1:
                sig = np.squeeze(sig)
            if isinstance(sig, numbers.Number):
                sig = np.array(sig)
            sig[:start_idx] = self.sig[:start_idx]  # Copy the first part over
            sig[start_idx:end] = src                       # The second part is the new signal
            sig[end:] = self.sig[stop_idx:]       # The final part is the remaining of self.sig
            self.sig = sig                                 # Update self.sig
        return self

    def resample(self, target_sr=44100, rate=1, kind='linear'):
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
        tsel = np.arange(np.floor(self.samples / self.sr * target_sr / rate)) * rate / target_sr
        if self.channels == 1:
            interp_fn = scipy.interpolate.interp1d(times, self.sig, kind=kind, assume_sorted=True,
                                                   bounds_error=False, fill_value=self.sig[-1])
            return Asig(interp_fn(tsel), target_sr,
                        label=self.label + "_resampled", cn=self.cn)
        else:
            new_sig = np.ndarray(
                shape=(int(self.samples / self.sr * target_sr / rate), self.channels))
            for i in range(self.channels):
                interp_fn = scipy.interpolate.interp1d(
                    times, self.sig[:, i], kind=kind, assume_sorted=True, 
                    bounds_error=False, fill_value=self.sig[-1, i])
                new_sig[:, i] = interp_fn(tsel)
            return Asig(new_sig, target_sr, label=self.label + "_resampled", cn=self.cn)

    def play(self, rate=1, **kwargs):
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
        if 'server' in kwargs.keys():
            s = kwargs['server']
        else:
            s = Aserver.default
        if not isinstance(s, Aserver):
            warn("Asig.play: no default server running, nor server arg specified.")
            return
        if rate == 1 and self.sr == s.sr:
            asig = self
        else:
            asig = self.resample(s.sr, rate)
        s.play(asig, **kwargs)
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
            _LOGGER.debug("Shift by %d, new signal has %d channels", shift, new_sig.shape[1])
            if self.channels == 1:
                new_sig[:, shift] = self.sig
            elif shift > 0:
                new_sig[:, shift:(shift + self.channels)] = self.sig
            elif shift < 0:
                new_sig[:] = self.sig[:, -shift:]
            if self.cn is None:
                new_cn = self.cn
            else:
                if shift > 0:
                    uname_list = ['unnamed' for i in range(shift)]
                    if isinstance(self.cn, list):
                        new_cn = uname_list + self.cn
                    else:
                        new_cn = uname_list.append(self.cn)
                elif shift < 0:
                    new_cn = self.cn[-shift:]
            return Asig(new_sig, self.sr, label=self.label + '_routed', cn=new_cn)
        else:
            warn("Argument needs to be int")
            return self

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
            warn("Asig.to_mono(): len(blend)=%d != %d=Asig.channels -> no action",
                 len(blend), self.channels)
            return self
        else:
            sig = np.sum(self.sig * blend, axis=1)
            col_names = [self.cn[np.argmax(blend)]] if self.cn is not None else None
            return Asig(sig, self.sr, label=self.label + '_blended', cn=col_names)

    def stereo(self, blend=None):
        """Blend all channels of the signal to stereo. Applicable for any single-/ or multi-channel Asig.

        Parameters
        ----------
        blend : list or None
            Usage: blend = [[list of gains for left channel], [list of gains for right channel]]
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
            left, right = (1, 1)
        else:
            left = blend[0]
            right = blend[1]
        if self.channels == 1:
            left_sig = self.sig * left
            right_sig = self.sig * right
            sig = np.stack((left_sig, right_sig), axis=1)
            return Asig(sig, self.sr, label=self.label + '_to_stereo', cn=['l', 'r'])
        elif len(left) == self.channels and len(right) == self.channels:
            left_sig = np.sum(self.sig * left, axis=1)
            right_sig = np.sum(self.sig * right, axis=1)
            sig = np.stack((left_sig, right_sig), axis=1)
            return Asig(sig, self.sr, label=self.label + '_to_stereo', cn=['l', 'r'])
        else:
            warn("Arg needs to be a list of 2 lists for left right. e.g. a[[0.2, 0.3, 0.2],[0.5]:" 
                 "Blend ch[0,1,2] to left and ch0 to right")
            return self

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
        # Find what the largest channel in the newly rewired is .
        max_ch = max(dic, key=lambda x: x[1])[1] + 1 
        if max_ch > self.channels:
            new_sig = np.zeros((self.samples, max_ch))
            new_sig[:, :self.channels] = np.copy(self.sig)
        else:
            new_sig = np.copy(self.sig)
        for key, val in dic.items():
            new_sig[:, key[1]] = self.sig[:, key[0]] * val
        return Asig(new_sig, self.sr, label=self.label + '_rewire', cn=self.cn)

    def pan2(self, pan=0.):
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
        if isinstance(pan, float):
            # Stereo panning.
            if pan <= 1. and pan >= -1.:
                angle = linlin(pan, -1, 1, 0, np.pi / 2.)
                gain = [np.cos(angle), np.sin(angle)]
                if self.channels == 1:
                    newsig = np.repeat(self.sig, 2)  # This is actually quite slow
                    newsig_shape = newsig.reshape(-1, 2) * gain
                    new_cn = [str(self.cn), str(self.cn)]
                    return Asig(newsig_shape, self.sr,
                                label=self.label + "_pan2ed", channels=2, cn=new_cn)
                else:
                    return Asig(self.sig[:, :2] * gain, self.sr, label=self.label + "_pan2ed", cn=self.cn)
            else:
                warn("Panning need to be in the range -1. to 1. Nothing changed.")
                return self
        else:
            warn("pan needs to be a float number between -1. to 1. Nothing changed.")
            return self

    def norm(self, norm=1, dcflag=False):
        """Normalize signal

        Parameters
        ----------
        norm : float
            normalize threshold (Default value = 1)
        dcflag : bool
            If true, remove dc offset (Default value = False)

        Returns
        -------
        _ : Asig
            normalized Asig.

        """
        if dcflag:
            sig = self.sig - np.mean(self.sig, 0)
        else:
            sig = self.sig
        if norm <= 0:  # take negative values as level in dB
            norm = dbamp(norm)
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
        if db:  # overwrites amp
            amp = dbamp(db)
        elif not amp:  # default 1 if neither is given
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

    def plot(self, fn=None, offset=0, scale=1, xlim=None, ylim=None, **kwargs):
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
            if fn == 'db':
                fn = lambda x: np.sign(x) * ampdb((abs(x) * 2 ** 16 + 1))
            elif not callable(fn):
                warn("Asig.plot: fn is neither keyword nor function")
                return self
            plot_sig = fn(self.sig)
        else:
            plot_sig = self.sig
        if self.channels == 1 or (offset == 0 and scale == 1):
            self._['plot'] = plt.plot(np.arange(0, self.samples) / self.sr, plot_sig, **kwargs)
        else:
            p = []
            ts = np.linspace(0, self.samples / self.sr, self.samples)
            for i, c in enumerate(self.sig.T):
                p.append(plt.plot(ts, i * offset + c * scale, **kwargs))
                plt.xlabel("time [s]")
                if self.cn:
                    plt.text(0, (i + 0.1) * offset, self.cn[i])
        if xlim is not None:
            plt.xlim([xlim[0], xlim[1]])
        if ylim is not None:
            plt.ylim([ylim[0], ylim[1]])
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
            self.label, self.channels, self.samples, self.sr, self.samples / self.sr,
            self.cn)

    def __mul__(self, other):
        """Magic method for multiplying. You can either multiply a numpy array or an Asig object. If adding an Asig,
            you don't always need to have same size arrays as audio signals may different in length. If mix_mode
            is set to 'bound' the size is fixed to respect self. If not, the result will respect to whichever the
            bigger array is."""
        selfsig = self.sig
        othersig = other.sig if isinstance(other, Asig) else other
        if isinstance(othersig, numbers.Number):
            return Asig(selfsig * othersig, self.sr, label=self.label + "_multiplied", cn=self.cn)
        else:
            if self.mix_mode is 'bound':
                if selfsig.shape[0] > othersig.shape[0]:
                    selfsig = selfsig[:othersig.shape[0]]
                elif selfsig.shape[0] < othersig.shape[0]:
                    othersig = othersig[:selfsig.shape[0]]
                result = selfsig * othersig
            else:
                if selfsig.shape[0] > othersig.shape[0]:
                    result = selfsig.copy()
                    result[:othersig.shape[0]] *= othersig

                elif selfsig.shape[0] < othersig.shape[0]:
                    result = othersig.copy()
                    result[:selfsig.shape[0]] *= selfsig
                else:
                    result = selfsig * othersig
            return Asig(result, self.sr, label=self.label + "_multiplied", cn=self.cn)

    def __rmul__(self, other):
        if isinstance(other, Asig):
            return Asig(self.sig * other.sig, self.sr, label=self.label + "_multiplied", cn=self.cn)
        else:
            return Asig(self.sig * other, self.sr, label=self.label + "_multiplied", cn=self.cn)

    def __add__(self, other):
        """Magic method for adding. You can either add a numpy array or an Asig object. If adding an Asig,
        you don't always need to have same size arrays as audio signals may different in length. If mix_mode
        is set to 'bound' the size is fixed to respect self. If not, the result will respect to whichever the
        bigger array is."""
        selfsig = self.sig
        othersig = other.sig if isinstance(other, Asig) else other
        if isinstance(othersig, numbers.Number):  # When other is just a scalar
            return Asig(selfsig + othersig, self.sr, label=self.label + "_added", cn=self.cn)
        else:
            if self.mix_mode is 'bound':
                try:
                    if selfsig.shape[0] > othersig.shape[0]:
                        selfsig = selfsig[:othersig.shape[0]]
                    elif selfsig.shape[0] < othersig.shape[0]:
                        othersig = othersig[:selfsig.shape[0]]
                except AttributeError:
                    pass  # When othersig is just a scalar
                result = selfsig + othersig
            else:
                # Make the bigger one
                if selfsig.shape[0] > othersig.shape[0]:
                    result = selfsig.copy()
                    result[:othersig.shape[0]] += othersig

                elif selfsig.shape[0] < othersig.shape[0]:
                    result = othersig.copy()
                    result[:selfsig.shape[0]] += selfsig
                else:
                    result = selfsig + othersig
            return Asig(result, self.sr, label=self.label + "_added", cn=self.cn)

    def __radd__(self, other):
        if isinstance(other, Asig):
            return Asig(self.sig + other.sig, self.sr, label=self.label + "_added", cn=self.cn)
        else:
            return Asig(self.sig + other, self.sr, label=self.label + "_added", cn=self.cn)

    def find_events(self, step_dur=0.001, sil_thr=-20, evt_min_dur=0, sil_min_dur=0.1, sil_pad=[0.001, 0.1]):
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
            sil_pad_samples = (int(sil_pad * self.sr), ) * 2
        event_list = []
        for i in range(0, self.samples, step_samples):
            rms = self[i:i + step_samples].rms()
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
                        event_list.append([
                            event_begin - sil_pad_samples[0],
                            event_end - step_samples * sil_min_steps + sil_pad_samples[1]])
                        sil_flag = True
        self._['events'] = np.array(event_list)
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
        if 'events' not in self._:
            print('select_event: no events, return all')
            return self
        events = self._['events']
        if onset:
            index = np.argmin(np.abs(events[:, 0] - onset * self.sr))
        if index is not None:
            beg, end = events[index]
            # print(beg, end)
            return Asig(self.sig[beg:end], self.sr, label=self.label + f"event_{index}", cn=self.cn)
        _LOGGER.warning('select_event: neither index nor onset given: return self')
        return self

    def fade_in(self, dur=0.1, curve=1):
        """Fade in the signal at the beginning

        Parameters
        ----------
        dur : float
            Duration in seconds to fade in (Default value = 0.1)
        curve : float
            Curvature of the fader. (Default value = 1)

        Returns
        -------
        _ : Asig
            Asig, new asig with the fade in signal
        """
        nsamp = int(dur * self.sr)
        if nsamp > self.samples:
            nsamp = self.samples
            warn("warning: Asig too short for fade_in - adapting fade_in time")
        return Asig(np.hstack((self.sig[:nsamp] * np.linspace(0, 1, nsamp) ** curve, self.sig[nsamp:])),
                    self.sr, label=self.label + "_fadein", cn=self.cn)

    def fade_out(self, dur=0.1, curve=1):
        """Fade out the signal at the end

        Parameters
        ----------
        dur : float
            duration in seconds to fade out (Default value = 0.1)
        curve : float
            Curvature of the fader. (Default value = 1)

        Returns
        -------
        _ : Asig
            Asig, new asig with the fade out signal
        """
        nsamp = int(dur * self.sr)
        if nsamp > self.samples:
            nsamp = self.samples
            warn("Asig too short for fade_out - adapting fade_out time")
        return Asig(np.hstack((self.sig[:-nsamp],
                               self.sig[-nsamp:] * np.linspace(1, 0, nsamp)**curve)),
                    self.sr, label=self.label + "_fadeout", cn=self.cn)

    def iirfilter(self, cutoff_freqs, btype='bandpass', ftype='butter', order=4,
                  filter='lfilter', rp=None, rs=None):
        """iirfilter based on scipy.signal.iirfilter

        Parameters
        ----------
        cutoff_freqs : int
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
        Wn = np.array(cutoff_freqs) * 2 / self.sr
        b, a = scipy.signal.iirfilter(order, Wn, rp=rp, rs=rs, btype=btype, ftype=ftype)
        y = scipy.signal.__getattribute__(filter)(b, a, self.sig, axis=0)
        aout = Asig(y, self.sr, label=self.label + "_iir")
        aout._['b'] = b
        aout._['a'] = a
        _LOGGER.debug('Filter applied.')
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
        w, h = scipy.signal.freqz(self._['b'], self._['a'], worN)
        plt.plot(w * self.sr / 2 / np.pi, ampdb(abs(h)), **kwargs)

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
                warn("channel mismatch!")
                return -1
            if sr != self.sr:
                warn("sr mismatch: use resample")
                return -1
        else:
            n = np.shape(sig)[0]
            sr = self.sr  # assume same sr as self
            sigar = sig
        if onset:   # onset overwrites pos, time has priority
            pos = int(onset * self.sr)
        if not pos:
            pos = 0  # add to begin if neither pos nor onset have been specified
        last = pos + n
        if last > self.samples:
            last = self.samples
            sigar = sigar[:last - pos]
        self.sig[pos:last] += amp * sigar
        return self

    def envelope(self, amps, ts=None, curve=1, kind='linear'):
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
                if all(ts[i] < ts[i + 1] for i in range(len(ts) - 1)):  # if list is monotonous
                    if ts[0] > 0:  # if first t > 0 extend amps/ts arrays prepending item
                        ts = np.insert(np.array(ts), 0, 0)
                        amps = np.insert(np.array(amps), 0, amps[0])
                    if ts[-1] < duration:  # if last t < duration append amps/ts value
                        ts = np.insert(np.array(ts), -1, duration)
                        amps = np.insert(np.array(amps), -1, amps[-1])
                else:
                    _LOGGER.error("Asig.envelope error: ts not sorted")
                    return self
                given_ts = ts
            if nsteps != self.samples:
                interp_fn = scipy.interpolate.interp1d(given_ts, amps, kind=kind)
                sig_new = self.sig * interp_fn(np.linspace(0, duration, self.samples)) ** curve  
        return Asig(sig_new, self.sr, label=self.label + "_enveloped", cn=self.cn)

    def adsr(self, att=0, dec=0.1, sus=0.7, rel=0.1, curve=1, kind='linear'):
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
        return self.envelope([0, 1, sus, sus, 0], [0, att, att + dec, dur - rel, dur],
                             curve=curve, kind=kind)

    def window(self, win='triang', **kwargs):
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
        return Asig(self.sig * scipy.signal.get_window(
            win, self.samples, **kwargs), self.sr, label=self.label + "_" + winstr, cn=self.cn)

    def window_op(self, nperseg=64, stride=32, win=None, fn='rms', pad='mirror'):
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
        res = np.zeros((nsegs, ))
        for i, cp in enumerate(centerpos):
            i0 = cp - nperseg // 2
            i1 = cp + nperseg // 2
            if i0 < 0:
                i0 = 0   # ToDo: correct padding!!!
            if i1 >= self.samples:
                i1 = self.samples - 1  # ToDo: correct padding!!!
            if isinstance(fn, str):
                res[i] = self[i0:i1].window(win).__getattribute__(fn)()
            else:  # assume fn to be a function on Asig
                res[i] = fn(self[i0:i1])
        return Asig(np.array(res), sr=self.sr // stride, label='window_oped', cn=self.cn)

    def overlap_add(self, nperseg=64, stride_in=32, stride_out=32, jitter_in=None, jitter_out=None,
                    win=None, pad='mirror'):
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
        res = Asig(np.zeros((self.samples // stride_in * stride_out, )), sr=self.sr,
                   label=self.label + '_ola', cn=self.cn)
        ii = 0
        io = 0
        for _ in range(self.samples // stride_in):
            i0 = ii - nperseg // 2
            if jitter_in:
                i0 += np.random.randint(jitter_in)
            i1 = i0 + nperseg
            if i0 < 0:
                i0 = 0   # TODO: correct left zero padding!!!
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
        return Aspec(self)

    def to_stft(self, **kwargs):
        """Return Astft object which is the stft of the signal. Keyword arguments are the arguments for
        scipy.signal.stft(). """
        return Astft(self, **kwargs)

    def plot_spectrum(self, offset=0, scale=1., xlim=None, **kwargs):
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
            self._['lines'] = plt.plot(frq, np.angle(Y), 'b.', markersize=0.2)
            if xlim is not None:
                plt.xlim(xlim[0], xlim[1])

        else:
            # For multichannel signals, a stacked view is plotted
            plt.subplot(211)
            max_y = np.max(np.abs(Y))
            p = []
            for i in range(Y.shape[1]):
                p.append(plt.plot(frq, np.abs(Y[:, i]) * scale + i * offset * max_y, **kwargs))  # + i * offset * max_y
                plt.text(0, (i + 0.1) * offset * max_y, self.cn[i])
            plt.xlabel('freq (Hz)')
            plt.ylabel('|F(freq)|')
            if xlim is not None:
                plt.xlim(xlim[0], xlim[1])
            plt.subplot(212)
            p = []
            for i in range(Y.shape[1]):
                p.append(plt.plot(frq, np.angle(Y[:, i]) * scale + i * offset * np.pi * 2, 'b.', markersize=0.2))
                plt.text(0, (i + 0.1) * offset * np.pi * 2, self.cn[i])
            plt.ylabel('Angle')
            if xlim is not None:
                plt.xlim(xlim[0], xlim[1])
        return self

    def spectrogram(self, *argv, **kvarg):
        """Perform sicpy.signal.spectrogram and returns: frequencies, array of times, spectrogram
        """
        freqs, times, Sxx = scipy.signal.spectrogram(self.sig, self.sr, *argv, **kvarg)
        return freqs, times, Sxx

    def size(self):
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
            warn("Asig.append: channels don't match")
            return self
        if self.sr != asig.sr:
            _LOGGER.info("resampling appended signal")
            atmp = asig.resample(self.sr)
        else:
            atmp = asig
        return Asig(np.hstack((self.sig, atmp.sig)), self.sr, label=self.label + "+" + asig.label, cn=self.cn)

    def custom(self, func, **kwargs):
        """custom function method. TODO add example"""
        func(self, **kwargs)
        return self


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
        cn : list or None
            Channel names (Default value = None)
        """
        self.cn = cn
        if type(x) == Asig:
            self.sr = x.sr
            self.rfftspec = np.fft.rfft(x.sig)
            self.label = x.label + "_spec"
            self.samples = x.samples
            self.channels = x.channels
            self.cn = x.cn
            if cn is not None and self.cn != cn:
                warn("Aspec:init: given cn  different from Asig cn: using Asig.cn")
        elif type(x) == list or type(x) == np.ndarray:
            self.rfftspec = np.array(x)
            self.sr = sr
            self.samples = (len(x) - 1) * 2
            self.channels = 1
            if len(np.shape(x)) > 1:
                self.channels = np.shape(x)[1]
        else:
            raise AttributeError("unknown initializer")
        if label:
            self.label = label
        self.nr_freqs = self.samples // 2 + 1
        self.freqs = np.linspace(0, self.sr / 2, self.nr_freqs)

    def to_sig(self):
        """Convert Aspec into Asig"""
        return Asig(np.fft.irfft(self.rfftspec), sr=self.sr, label=self.label + '_2sig', cn=self.cn)

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
            if all(freqs[i] < freqs[i + 1] for i in range(len(freqs) - 1)):  # check if list is monotonous
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
            interp_fn = scipy.interpolate.interp1d(given_freqs, weights, kind=kind)
            rfft_new = self.rfftspec * interp_fn(self.freqs) ** curve  # ToDo: curve segmentwise!!!
        else:
            rfft_new = self.rfftspec * weights ** curve
        return Aspec(rfft_new, self.sr, label=self.label + "_weighted")

    def plot(self, fn=np.abs, xlim=None, ylim=None, **kwargs):  # TODO add ax option
        """Plot spectrum

        Parameters
        ----------
        fn : func
            function for processing the rfft spectrum. (Default value = np.abs)
        xlim : tuple or list or None
            Set x axis range (Default value = None)
        ylim : tuple or list or None
            Set y axis range (Default value = None)
        **kwargs :
            Keyword arguments for matplotlib.pyplot.plot()

        Returns
        -------
        _ : Asig
            self
        """
        plt.plot(self.freqs, fn(self.rfftspec), **kwargs)
        if xlim is not None:
            plt.xlim([xlim[0], xlim[1]])

        if ylim is not None:
            plt.ylim([ylim[0], ylim[1]])

        plt.xlabel('freq (Hz)')
        plt.ylabel(f'{fn.__name__}(freq)')
        return self

    def __repr__(self):
        return "Aspec('{}'): {} x {} @ {} Hz = {:.3f} s".format(
            self.label, self.channels, self.samples, self.sr, self.samples / self.sr)


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
            signal to be converted to stft domain. This can be either a numpy array or an Asig object. 
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
        if type(x) == Asig:
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
        elif type(x) == np.ndarray and np.shape(x) >= 2:
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
            print("error: unknown initializer or wrong stft shape ")
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
        return Asig(sig, sr=self.sr, label=self.label + '_2sig', cn=self.cn)

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
