"""Contains the classes Asig Aspec and later Astft,
to enable sample-precise audio coding with numpy/scipy/python
for multi-channel audio processing & sonification
"""

import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.signal
from scipy.fftpack import fft, fftfreq, ifft
from scipy.io import wavfile
import pyaudio
from .pyaudiostream import PyaudioStream
from .helpers import ampdb, linlin, dbamp, timeit
from .ugen import *  # newly added ugen.
import logging

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class Asig:
    'audio signal class'
    def __init__(self, sig, sr=44100, label="", channels=1, cn=None):
        self.sr = sr
        self._ = {}  # dictionary for further return values
        self.channels = channels
        if isinstance(sig, str):
            self.load_wavfile(sig)
        elif isinstance(sig, int):  # sample length
            if self.channels == 1:
                self.sig = np.zeros(sig)
            else:
                self.sig = np.zeros((sig, self.channels))
        elif isinstance(sig, float):  # if float interpret as duration
            if self.channels == 1:
                self.sig = np.zeros(int(sig * sr))
            else:
                self.sig = np.zeros((int(sig * sr), self.channels))
        else:
            self.sig = np.array(sig)
            try:
                self.channels = self.sig.shape[1]
            except IndexError:
                self.channels = 1
        self.samples = np.shape(self.sig)[0]
        self.label = label
        self.device = 1
        # make a copy for any processing events e.g. (panning, filtering)
        # that needs to process the signal without permanent change.
        self.sig_copy = self.sig.copy()  # It takes around 100ms to copy a 17min audio at 44.1khz
        self.cn = cn
        self._set_col_names()

    def load_wavfile(self, fname):
        # Discuss to change to float32 .
        self.sr, self.sig = wavfile.read(fname)  # load the sample data
        if self.sig.dtype == np.dtype('int16'):
            self.sig = self.sig.astype('float32') / 32768
            try:
                self.channels = self.sig.shape[1]
            except IndexError:
                self.channels = 1
        elif self.sig.dtype != np.dtype('float32'):
            self.sig = self.sig.astype('float32')
            try:
                self.channels = self.sig.shape[1]
            except IndexError:
                self.channels = 1
        else:
            print("load_wavfile: TODO: add format")

        # ToDo: set channels here
    def save_wavfile(self, fname="asig.wav", dtype='float32'):
        if dtype == 'int16':
            data = (self.sig * 32767).astype('int16')
        elif dtype == 'int32':
            data = (self.sig * 2147483647).astype('int32')
        elif dtype == 'uint8':
            data = (self.sig * 127 + 128).astype('uint8')
        elif dtype == 'float32':
            data = self.sig.astype('float32')
        wavfile.write(fname, self.sr, data)
        return self

    def _set_col_names(self):
        # Problem is by doing that generating a new instance will no longer preserve cn.
        if self.cn is None:
            pass
        else:
            if type(self.cn[0]) is str:
                self.col_name = {self.cn[i]: i for i in range(len(self.cn))}
            else:
                raise TypeError("column names need to be a list of strings")

    def __getitem__(self, index):
        """
        Accept numpy style slicing:

        1. Index is integer: a[5], as numpy . get row 5
        2. Index is list: a[[3,4,5]] returns row 3,4,5
        3. Index is slice: a[3:50:1], row slicing
        4. Index is set: a[{1,3}], time slicing.
        5. Index is tuple: tuple always has two elements.
            * Index[0] is int, list, slice or tuple.
            * Index[0] is a set. time slicing.
            * Index[1] is int, list, slice, list of str, list of boolean.
        """
        if isinstance(index, int):
            # Case a[4], 4th row
            return Asig(self.sig[index], self.sr, label=self.label + '_arrayindexed', cn=self.cn)

        elif isinstance(index, list):
            # Case a[[1,2,4]] for row selection or case[['l']] name column selection
            if isinstance(index[0], str):
                col_idx = [self.col_name.get(s) for s in index]
                return Asig(self.sig[:, col_idx], self.sr,
                            label=self.label + '_arrayindexed', cn=index)
            else:
                return Asig(self.sig[index], self.sr,
                            label=self.label + '_arrayindexed', cn=self.cn)

        elif isinstance(index, slice):
            # Case a[start:stop:step],
            start, stop, step = index.indices(len(self.sig))    # index is a slice
            return Asig(self.sig[index], sr=int(self.sr / abs(step)),
                        label=self.label + "_sliced", cn=self.cn)

        elif isinstance(index, dict):
            for key, value in index.items():
                try:
                    start = int(key * self.sr)
                except TypeError:
                    start = None
                try:
                    stop = int(value * self.sr)
                except TypeError:
                    stop = None
            rslice = slice(start, stop, 1)
            sr = self.sr
            return Asig(self.sig[rslice], sr=sr, label=self.label + '_arrayindexed', cn=self.cn)

        elif isinstance(index, tuple):
            # Tuple is when there are dedicated slicing for rows and columns.

            # First check index[0]
            if isinstance(index[0], int) or isinstance(index[0], list):
                # Case a[4, :],
                rslice = index[0]
                sr = self.sr

            elif isinstance(index[0], slice):
                start, stop, step = index[0].indices(len(self.sig))
                sr = int(self.sr / abs(step))
                rslice = index[0]  # row slice

            elif isinstance(index[0], dict):  # Time slicing
                for key, value in index[0].items():
                    try:
                        start = int(key * self.sr)
                    except TypeError:  # if it is None
                        start = None
                    try:
                        stop = int(value * self.sr)
                    except TypeError:
                        stop = None
                rslice = slice(start, stop, 1)
                sr = self.sr
            else:
                rslice = index[0]
                sr = self.sr

            # Now check index[1]:
            # First check if index[1] is channel name slicing
            if type(index[1]) is list and type(index[1][0]) is str:
                col_idx = [self.col_name.get(s) for s in index[1]]
                cn_new = [self.cn[i] for i in col_idx]
                return Asig(self.sig[rslice, col_idx], sr=sr, label=self.label + '_arrayindexed', cn=cn_new)

            # int, list, slice are the same.
            elif isinstance(index[1], int) or isinstance(index[1], list) or isinstance(index[1], slice):
                if isinstance(index[1], list):
                    cn_new = [self.cn[i] for i in index[1]]
                else:
                    cn_new = self.cn[index[1]]
                return Asig(self.sig[rslice, index[1]], sr=sr, label=self.label + '_arrayindexed', cn=cn_new)

            # if only a single channel name is given.
            elif isinstance(index[1], str):
                # The column name should be incorrect afterward.
                return Asig(self.sig[rslice, self.col_name.get(index[1])], sr=sr,
                            label=self.label + '_arrayindexed', cn=index[1])

        else:
            raise TypeError("index must be int, array, or slice")

    def __eq__(self, other):
        sig_eq = np.array_equal(self.sig, other.sig)
        sr_eq = self.sr == other.sr
        return sig_eq and sr_eq

    # TODO: this may not be necessary any more.
    def tslice(self, *tidx):
        if len(tidx) == 1:  # stop
            sl = slice(0, tidx[0] * self.sr)
        elif len(tidx) == 2:  # start and stop:
            sl = slice(int(tidx[0] * self.sr), int(tidx[1] * self.sr))
        else:
            sl = slice(int(tidx[0] * self.sr), int(tidx[1] * self.sr), tidx[2])
        return Asig(self.sig[sl], self.sr, self.label + "_tsliced", cn=self.cn)

    def resample(self, target_sr=44100, rate=1, kind='linear'):
        """Resample signal based on interpolation, can process multichannel"""
        times = np.arange(self.samples) / self.sr
        tsel = np.arange(np.floor(self.samples / self.sr * target_sr / rate)) * rate / target_sr
        if self.channels == 1:
            interp_fn = scipy.interpolate.interp1d(times, self.sig, kind=kind, assume_sorted=True,
                                                   bounds_error=False, fill_value=self.sig[-1])
            return Asig(interp_fn(tsel), target_sr,
                        label=self.label + "_resampled", cn=self.cn)
        else:
            new_sig = np.ndarray(shape=(int(self.samples / self.sr * target_sr / rate), self.channels))
            for i in range(self.channels):
                interp_fn = scipy.interpolate.interp1d(
                    times, self.sig[:, i], kind=kind, assume_sorted=True, bounds_error=False, fill_value=self.sig[-1, i])
                new_sig[:, i] = interp_fn(tsel)
            return Asig(new_sig, target_sr, label=self.label + "_resampled", cn=self.cn)

    def play(self, rate=1, **kwargs):
        """Play Asig audio via Aserver, using Aserver.default (if existing)
        kwargs are propagated to Aserver:play (onset=0, out=0)
        IDEA/ToDo: allow to set server='stream' to create
          which terminates when finished using pyaudiostream
        """
        if 'server' in kwargs.keys():
            s = kwargs['server']
        else:
            s = Aserver.default
        if not isinstance(s, Aserver):
            _LOGGER.error("Asig.play: no default server running, nor server arg specified.")
            return
        if rate == 1 and self.sr == s.sr:
            asig = self
            print(asig)
        else:
            asig = self.resample(s.sr, rate)
            print(asig)
        s.play(asig, **kwargs)
        return self

    def route(self, out=0):
        """Route the signal to n channel. This method shift the signal to out channel:

            * out = 0: does nothing as the same signal is being routed to the same position
            * out > 0: move the first channel of self.sig to out channel, other channels follow
        """
        if isinstance(out, int):
            if out == 0:  # If 0 do nothing.
                return self
            elif out > 0:
                # not optimized method here
                new_sig = np.zeros((self.samples, out + self.channels))

                _LOGGER.debug("Shift to channel %d, new signal has %d channels", out, new_sig.shape[1])
                if self.channels == 1:
                    new_sig[:, out] = self.sig_copy
                else:
                    new_sig[:, out:(out + self.channels)] = self.sig_copy
                _LOGGER.debug("Successfully assign signal to new_sig")
                if self.cn is None:
                    new_cn = self.cn
                else:
                    uname_list = ['unnamed' for i in range(out)]
                    if isinstance(self.cn, list):
                        new_cn = uname_list + self.cn
                    else:
                        new_cn = uname_list.append(self.cn)
                return Asig(new_sig, self.sr, label=self.label + '_routed', cn=new_cn)

            # The left shift function is removed.
            # elif out < 0 and -out < self.channels :
            #     new_sig = self.sig_copy[:, -out:]
            #     return Asig(new_sig, self.sr, label=self.label+'_routed', cn=self.cn)
            else:
                print("left shift over the total channel, nothing happened")
                return self

        elif type(out) is list:
            """
                Several possibilities here:
                    1. sig is mono:
                        convert to multi channels and apply gain.
                    2. sig's channels equals pan size
                    3. sig's channels > pan size and
                    4. sig's channels < pan size
                Dont permanently change self.sig
            """
            if np.max(out) > 1 or np.min(out) < 0.:
                _LOGGER.warning("Warning: list value should be between 0 ~ 1.")
            if self.channels == 1:  # if mono sig.
                new_sig = self.mono2nchanel(self.sig_copy, len(out))
                new_sig *= out  # apply panning.
            elif self.channels == len(out):
                new_sig = self.sig_copy * out
            else:
                raise ValueError("pan size and signal channels don't match")
            return Asig(new_sig, self.sr, label=self.label + '_routed', cn=self.cn)
        else:
            raise TypeError("Argument needs to be a list of 0 ~ 1.")

    @timeit
    def to_mono(self, blend):
        """
            Blend multichannel: if mono signal do nothing.
            [0.33, 0.33, 0,33] blend a 3-chan sigal to a mono signal with 0.33x each
            np.sum is the main computation here. Not much can be done to make it faster.
        """

        if self.channels == 1:
            _LOGGER.warning("Signal is already mono")
            return self

        if len(blend) != self.channels:
            _LOGGER.warning("Arg len %d, signal channel %d. Method did nothing", len(blend), self.channels)
            return self
        else:
            sig = np.sum(self.sig_copy * blend, axis=1)
            return Asig(sig, self.sr, label=self.label + '_blended', cn=self.cn)

    def to_stereo(self, blend):
        """Blend any channel of signal to stereo.

        Usage: blend = [[list], [list]], e.g:

        Mix ch0,1,2 to the left channel with gain of 0.3 each, and ch0,1,2,3 to right with 0.25 gain

        asig[[0.3, 0.3, 0.3], [0.25, 0.25, 0.25 0.25]]

        """
        left = blend[0]
        right = blend[1]
        # [[0.1,0.2,03], [0.4,0.5,0.6]]
        if self.channels == 1:
            left_sig = self.sig_copy * left
            right_sig = self.sig_copy * right
            sig = np.stack((left_sig, right_sig), axis=1)
            return Asig(sig, self.sr, label=self.label + '_to_stereo', cn=self.cn)

        if len(left) == self.channels and len(right) == self.channels:
            left_sig = np.sum(self.sig_copy * left, axis=1)
            right_sig = np.sum(self.sig_copy * right, axis=1)
            sig = np.stack((left_sig, right_sig), axis=1)
            return Asig(sig, self.sr, label=self.label + '_to_stereo', cn=self.cn)
        else:
            _LOGGER.warning("Arg needs to be a list of 2 lists for left right. e.g. a[[0.2, 0.3, 0.2],[0.5]:"
                            "Blend ch[0,1,2] to left and ch0 to right")
            return self

    def rewire(self, dic):
        """rewire channels:
            {(0, 1): 0.5}: move channel 0 to 1 then reduce gain to 0.5
        """
        max_ch = max(dic, key=lambda x: x[1])[1]  # Find what the largest channel in the newly rewired is .
        if max_ch > self.channels:
            new_sig = np.zeros((self.samples, max_ch))
            new_sig[:, :self.channels] = self.sig_copy
        else:
            new_sig = self.sig
        for i, k in enumerate(dic):
            new_sig[k[1]] = self.sig_copy[k[0]] * i
        return Asig(new_sig, self.sr, label=self.label + '_rewire', cn=self.cn)

    def pan2(self, pan=0.):
        """
            pan2 only creates output in stereo, mono will be copy to stereo, stereo works as it should,
            larger channel signal will only has 0 and 1 being changed.
            panning is based on constant power panning.

            # gain multiplication is the main computation cost.
        """
        pan = float(pan)
        if type(pan) is float:
            # Stereo panning.
            if pan <= 1. and pan >= -1.:
                angle = linlin(pan, -1, 1, 0, np.pi / 2.)
                gain = [np.cos(angle), np.sin(angle)]
                if self.channels == 1:
                    newsig = np.repeat(self.sig_copy, 2)  # This is actually quite slow
                    newsig_shape = newsig.reshape(-1, 2) * gain
                    new_cn = [self.cn, self.cn]
                    return Asig(newsig_shape, self.sr,
                                label=self.label + "_pan2ed", channels=2, cn=new_cn)
                else:
                    self.sig_copy[:, :2] *= gain
                    return Asig(self.sig_copy, self.sr, label=self.label + "_pan2ed", cn=self.cn)
            else:
                print("Warning: Scalar panning need to be in the range -1. to 1. nothing changed.")
                return self

    def overwrite(self, sig, sr=None):
        """
        Overwrite the sig with new signal, then readjust the shape.
        """
        self.sig = sig
        try:
            self.channels = self.sig.shape[1]
        except IndexError:
            self.channels = 1
        self.samples = len(self.sig)
        return self

    # This is the original method via simpleaudio
    # def play(self, rate=1, block=False):
    #     if not self.sr in [8000, 11025, 22050, 44100, 48000]:
    #         print("resample as sr is exotic")
    #         self._['play'] = self.resample(44100, rate).play(block=block)['play']
    #     else:
    #         if rate is not 1:
    #             print("resample as rate!=1")
    #             self._['play'] = self.resample(44100, rate).play(block=block)['play']
    #         else:
    #             self._['play'] = play(self.sig, self.channels, self.sr, block=block)
    #     return self

    def norm(self, norm=1, dcflag=False):
        if dcflag:
            self.sig = self.sig - np.mean(self.sig, 0)
        if norm <= 0:  # take negative values as level in dB
            norm = dbamp(norm)
        self.sig = norm * self.sig / np.max(np.abs(self.sig), 0)
        return self

    def gain(self, amp=None, db=None):
        if db:  # overwrites amp
            amp = dbamp(db)
        elif not amp:  # default 1 if neither is given
            amp = 1
        return Asig(self.sig * amp, self.sr, label=self.label + "_scaled", cn=self.cn)

    def rms(self, axis=0):
        return np.sqrt(np.mean(np.square(self.sig), axis))

    def plot(self, fn=None, offset=0, scale=1, **kwargs):
        if fn:
            if fn == 'db':
                fn = lambda x: np.sign(x) * ampdb((abs(x) * 2 ** 16 + 1))
            elif not callable(fn):
                print("Asig.plot: fn is neither keyword nor function")
                return self
            plot_sig = fn(self.sig)
        else:
            plot_sig = self.sig
        if offset == 0 and scale == 1:
            self._['plot'] = plt.plot(np.arange(0, self.samples) / self.sr, plot_sig, **kwargs)
        else:
            p = []
            ts = np.linspace(0, self.samples / self.sr, self.samples)
            for i, c in enumerate(self.sig.T):
                p.append(plt.plot(ts, i * offset + c * scale, **kwargs))
                plt.xlabel("time [s]")
                if self.cn:
                    plt.text(0, (i + 0.1) * offset, self.cn[i])
        return self

    def get_duration(self):
        return self.samples / self.sr

    def get_times(self):
        return np.linspace(0, (self.samples - 1) / self.sr, self.samples)  # timestamps for left-edge of sample-and-hold-signal

    def __repr__(self):
        return "Asig('{}'): {} x {} @ {} Hz = {:.3f} s".format(
            self.label, self.channels, self.samples, self.sr, self.samples / self.sr)

    def __mul__(self, other):
        if isinstance(other, Asig):
            return Asig(self.sig * other.sig, self.sr, label=self.label + "_multiplied", cn=self.cn)
        else:
            return Asig(self.sig * other, self.sr, label=self.label + "_multiplied", cn=self.cn)

    def __rmul__(self, other):
        if isinstance(other, Asig):
            return Asig(self.sig * other.sig, self.sr, label=self.label + "_multiplied", cn=self.cn)
        else:
            return Asig(self.sig * other, self.sr, label=self.label + "_multiplied", cn=self.cn)

    def __add__(self, other):
        if isinstance(other, Asig):
            return Asig(self.sig + other.sig, self.sr, label=self.label + "_added", cn=self.cn)
        else:
            return Asig(self.sig + other, self.sr, label=self.label + "_added", cn=self.cn)

    def __radd__(self, other):
        if isinstance(other, Asig):
            return Asig(self.sig + other.sig, self.sr, label=self.label + "_added", cn=self.cn)
        else:
            return Asig(self.sig + other, self.sr, label=self.label + "_added", cn=self.cn)

    # TODO not checked.
    def find_events(self, step_dur=0.001, sil_thr=-20, sil_min_dur=0.1, sil_pad=[0.001, 0.1]):
        if self.channels > 1:
            print("warning: works only with 1-channel signals")
            return -1
        step_samples = int(step_dur * self.sr)
        sil_thr_amp = dbamp(sil_thr)
        sil_flag = True
        sil_min_steps = int(sil_min_dur / step_dur)
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
                    event_list.append([
                        event_begin - sil_pad_samples[0],
                        event_end - step_samples * sil_min_steps + sil_pad_samples[1]])
                    sil_flag = True
        self._['events'] = np.array(event_list)
        return self

    # TODO not checked.
    def select_event(self, index=None, onset=None):
        if 'events' not in self._:
            print('select_event: no events, return all')
            return self
        events = self._['events']
        if onset:
            index = np.argmin(np.abs(events[:, 0] - onset * self.sr))
        if index is not None:
            beg, end = events[index]
            print(beg, end)
            return Asig(self.sig[beg:end], self.sr, label=self.label + f"event_{index}", cn=self.cn)
        print('select_event: neither index nor onset given: return self')
        return self

    # spectral segment into pieces - incomplete and unused
    # def find_events_spectral(self, nperseg=64, on_threshold=3, off_threshold=2, medfilt_order=15):
    #     tiny = np.finfo(np.dtype('float64')).eps
    #     f, t, Sxx = scipy.signal.spectrogram(self.sig, self.sr, nperseg=nperseg)
    #     env = np.mean(np.log(Sxx + tiny), 0)
    #     sp = np.mean(np.log(Sxx + tiny), 1)
    #     # ts = np.arange(self.samples)/self.sr
    #     envsig = np.log(self.sig**2 + 0.001)
    #     envsig = envsig - np.median(envsig)
    #     menv = scipy.signal.medfilt(env, medfilt_order) - np.median(env)
    #     ibeg = np.where( np.logical_and( menv[1:] > on_threshold, menv[:-1] < on_threshold) )[0]
    #     iend = np.where( np.logical_and( menv[1:] < off_threshold, menv[:-1] > off_threshold) )[0]
    #     return (np.vstack((t[ibeg], t[iend])).T*self.sr).astype('int32')

    def fade_in(self, dur=0.1, curve=1):
        nsamp = int(dur * self.sr)
        if nsamp > self.samples:
            nsamp = self.samples
            print("warning: Asig too short for fade_in - adapting fade_in time")
        return Asig(np.hstack((self.sig[:nsamp] * np.linspace(0, 1, nsamp) ** curve, self.sig[nsamp:])),
                    self.sr, label=self.label + "_fadein", cn=self.cn)

    def fade_out(self, dur=0.1, curve=1):
        nsamp = int(dur * self.sr)
        if nsamp > self.samples:
            nsamp = self.samples
            print("warning: Asig too short for fade_out - adapting fade_out time")
        return Asig(np.hstack((self.sig[:-nsamp],
                               self.sig[-nsamp:] * np.linspace(1, 0, nsamp)**curve)),
                    self.sr, label=self.label + "_fadeout", cn=self.cn)

    def iirfilter(self, cutoff_freqs, btype='bandpass', ftype='butter', order=4,
                  filter='lfilter', rp=None, rs=None):
        Wn = np.array(cutoff_freqs) * 2 / self.sr
        b, a = scipy.signal.iirfilter(order, Wn, rp=rp, rs=rs, btype=btype, ftype=ftype)
        y = scipy.signal.__getattribute__(filter)(b, a, self.sig)
        aout = Asig(y, self.sr, label=self.label + "_iir")
        aout._['b'] = b
        aout._['a'] = a
        return aout

    def plot_freqz(self, worN, **kwargs):
        w, h = scipy.signal.freqz(self._['b'], self._['a'], worN)
        plt.plot(w * self.sr / 2 / np.pi, ampdb(abs(h)), **kwargs)

    def add(self, sig, pos=None, amp=1, onset=None):
        if type(sig) == Asig:
            n = sig.samples
            sr = sig.sr
            sigar = sig.sig
            if sig.channels != self.channels:
                print("channel mismatch!")
                return -1
            if sr != self.sr:
                print("sr mismatch: use resample")
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
        nsteps = len(amps)
        duration = self.samples / self.sr
        if nsteps == self.samples:
            sig_new = self.sig * amps ** curve
        else:
            if not ts:
                given_ts = np.linspace(0, duration, nsteps)
            else:
                if nsteps != len(ts):
                    print("Asig.envelope error: len(amps)!=len(ts)")
                    return self
                if all(ts[i] < ts[i + 1] for i in range(len(ts) - 1)):  # if list is monotonous
                    if ts[0] > 0:  # if first t > 0 extend amps/ts arrays prepending item
                        ts = np.insert(np.array(ts), 0, 0)
                        amps = np.insert(np.array(amps), 0, amps[0])
                    if ts[-1] < duration:  # if last t < duration append amps/ts value
                        ts = np.insert(np.array(ts), -1, duration)
                        amps = np.insert(np.array(amps), -1, amps[-1])
                else:
                    print("Asig.envelope error: ts not sorted")
                    return self
                given_ts = ts
            if nsteps != self.samples:
                interp_fn = scipy.interpolate.interp1d(given_ts, amps, kind=kind)
                sig_new = self.sig * interp_fn(np.linspace(0, duration, self.samples)) ** curve  # ToDo: curve segmentwise!!!
        return Asig(sig_new, self.sr, label=self.label + "_enveloped", cn=self.cn)

    def adsr(self, att=0, dec=0.1, sus=0.7, rel=0.1, curve=1, kind='linear'):
        dur = self.get_duration()
        return self.envelope([0, 1, sus, sus, 0], [0, att, att + dec, dur - rel, dur],
                             curve=curve, kind=kind)

    def window(self, win='triang', **kwargs):
        if not win:
            return self
        winstr = win
        if type(winstr) == tuple:
            winstr = win[0]
        return Asig(self.sig * scipy.signal.get_window(
            win, self.samples, **kwargs), self.sr, label=self.label + "_" + winstr, cn=self.cn)

    def window_op(self, nperseg=64, stride=32, win=None, fn='rms', pad='mirror'):
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
        return Aspec(self)

    def spectrum(self):
        nrfreqs = self.samples // 2 + 1
        frq = np.linspace(0, 0.5 * self.sr, nrfreqs)  # one sides frequency range
        Y = fft(self.sig)[:nrfreqs]  # / self.samples
        return frq, Y

    def plot_spectrum(self, **kwargs):
        frq, Y = self.spectrum()
        plt.subplot(211)
        plt.plot(frq, np.abs(Y), **kwargs)
        plt.xlabel('freq (Hz)')
        plt.ylabel('|F(freq)|')
        plt.subplot(212)
        self._['lines'] = plt.plot(frq, np.angle(Y), 'b.', markersize=0.2)
        return self

    def spectrogram(self, *argv, **kvarg):
        freqs, times, Sxx = scipy.signal.spectrogram(self.sig, self.sr, *argv, **kvarg)
        return freqs, times, Sxx

    def size(self):
        # return samples and length in time:
        return self.sig.shape, self.sig.shape[0] / self.sr

    def mono2nchanel(self, x, chan):
        # Create multichannel signal from mono
        c = np.vstack([x] * chan)
        return c.transpose()

    def custom(self, func, **kwargs):
        """
            A custom function method.
        """

        func(self, **kwargs)
        return self


class Aspec:
    'audio spectrum class using rfft'

    def __init__(self, x, sr=44100, label=None, cn=None):
        self.cn = cn
        if type(x) == Asig:
            self.sr = x.sr
            self.rfftspec = np.fft.rfft(x.sig)
            self.label = x.label + "_spec"
            self.samples = x.samples
            self.channels = x.channels
            self.cn = x.cn
            if self.cn != cn:
                print("Aspec:init: given cn different from Asig cn: using Asig.cn")
        elif type(x) == list or type(x) == np.ndarray:
            self.rfftspec = np.array(x)
            self.sr = sr
            self.samples = (len(x) - 1) * 2
            self.channels = 1
            if len(np.shape(x)) > 1:
                self.channels = np.shape(x)[1]
        else:
            print("error: unknown initializer")
        if label:
            self.label = label
        self.nr_freqs = self.samples // 2 + 1
        self.freqs = np.linspace(0, self.sr / 2, self.nr_freqs)

    def to_sig(self):
        return Asig(np.fft.irfft(self.rfftspec), sr=self.sr, label=self.label + '_2sig', cn=self.cn)

    def weight(self, weights, freqs=None, curve=1, kind='linear'):
        nfreqs = len(weights)
        if not freqs:
            given_freqs = np.linspace(0, self.freqs[-1], nfreqs)
        else:
            if nfreqs != len(freqs):
                print("Aspec.weight error: len(weights)!=len(freqs)")
                return self
            if all(freqs[i] < freqs[i + 1] for i in range(len(freqs) - 1)):  # check if list is monotonous
                if freqs[0] > 0:
                    freqs = np.insert(np.array(freqs), 0, 0)
                    weights = np.insert(np.array(weights), 0, weights[0])
                if freqs[-1] < self.sr / 2:
                    freqs = np.insert(np.array(freqs), -1, self.sr / 2)
                    weights = np.insert(np.array(weights), -1, weights[-1])
            else:
                print("Aspec.weight error: freqs not sorted")
                return self
            given_freqs = freqs
        if nfreqs != self.nr_freqs:
            interp_fn = scipy.interpolate.interp1d(given_freqs, weights, kind=kind)
            rfft_new = self.rfftspec * interp_fn(self.freqs) ** curve  # ToDo: curve segmentwise!!!
        else:
            rfft_new = self.rfftspec * weights ** curve
        return Aspec(rfft_new, self.sr, label=self.label + "_weighted")

    def plot(self, fn=np.abs, **kwargs):
        plt.plot(self.freqs, fn(self.rfftspec), **kwargs)
        plt.xlabel('freq (Hz)')
        plt.ylabel(f'{fn.__name__}(freq)')

    def __repr__(self):
        return "Aspec('{}'): {} x {} @ {} Hz = {:.3f} s".format(
            self.label, self.channels, self.samples, self.sr, self.samples / self.sr)


# TODO, check with multichannel
class Astft:
    'audio spectrogram (STFT) class'

    def __init__(self, x, sr=None, label=None, window='hann', nperseg=256,
                 noverlap=None, nfft=None, detrend=False, return_onesided=True,
                 boundary='zeros', padded=True, axis=-1, cn=None):
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
        """ create signal from stft, i.e. perform istft, kwargs overwrite Astft values for istft
        """
        for k in ['sr', 'window', 'nperseg', 'noverlap', 'nfft', 'input_onesided', 'boundary']:
            if k in kwargs.keys():
                kwargs[k] = self.__getattribute__(k)

        if 'sr' in kwargs.keys():
            kwargs['fs'] = kwargs['sr']
            del kwargs['sr']

        _, sig = scipy.signal.istft(self.stft, **kwargs)  # _ since 1st return value 'times' unused
        return Asig(sig, sr=self.sr, label=self.label + '_2sig', cn=self.cn)

    def plot(self, fn=lambda x: x):
        plt.pcolormesh(self.times, self.freqs, fn(np.abs(self.stft)))
        plt.colorbar()
        return self

    def __repr__(self):
        return "Astft('{}'): {} x {} @ {} Hz = {:.3f} s".format(
            self.label, self.channels, self.samples, self.sr, self.samples / self.sr, cn=self.cn)


# global pya.startup() and shutdown() fns
def startup(**kwargs):
    return Aserver.startup_default_server(**kwargs)


def shutdown(**kwargs):
    Aserver.shutdown_default_server(**kwargs)


class Aserver:

    default = None  # that's the default Aserver if Asigs play via it

    @staticmethod
    def startup_default_server(**kwargs):
        if Aserver.default is None:
            print("Aserver startup_default_server: create and boot")
            Aserver.default = Aserver(**kwargs)  # using all default settings
            Aserver.default.boot()
            print(Aserver.default)
        else:
            print("Aserver default_server already set.")
        return Aserver.default

    @staticmethod
    def shutdown_default_server():
        if isinstance(Aserver.default, Aserver):
            Aserver.default.quit()
            del(Aserver.default)
            Aserver.default = None
        else:
            print("Aserver:shutdown_default_server: no default_server to shutdown")

    """
    Aserver manages an pyaudio stream, using its aserver callback
    to feed dispatched signals to output at the right time
    """
    def __init__(self, sr=44100, bs=256, device=1, channels=2, format=pyaudio.paFloat32):
        self.sr = sr
        self.bs = bs
        self.device = device
        self.pa = pyaudio.PyAudio()
        self.channels = channels
        self.device_dict = self.pa.get_device_info_by_index(self.device)
        """
            self.max_out_chn is not that useful: there can be multiple devices having the same mu
        """
        self.max_out_chn = self.device_dict['maxOutputChannels']
        if self.max_out_chn < self.channels:
            print(f"Aserver: warning: {channels}>{self.max_out_chn} channels requested - truncated.")
            self.channels = self.max_out_chn
        self.format = format
        self.gain = 1.0
        self.srv_onsets = []
        self.srv_asigs = []
        self.srv_curpos = []  # start of next frame to deliver
        self.srv_outs = []  # output channel offset for that asig
        self.pastream = None
        self.dtype = 'float32'  # for pyaudio.paFloat32
        self.range = 1.0
        self.boot_time = None  # time.time() when stream starts
        self.block_cnt = None  # nr. of callback invocations
        self.block_duration = self.bs / self.sr  # nominal time increment per callback
        self.block_time = None  # estimated time stamp for current block
        if self.format == pyaudio.paInt16:
            self.dtype = 'int16'
            self.range = 32767
        if self.format not in [pyaudio.paInt16, pyaudio.paFloat32]:
            print(f"Aserver: currently unsupported pyaudio format {self.format}")
        self.empty_buffer = np.zeros((self.bs, self.channels), dtype=self.dtype)
        self.input_devices = []
        self.output_devices = []
        for i in range(self.pa.get_device_count()):
            if self.pa.get_device_info_by_index(i)['maxInputChannels'] > 0:
                self.input_devices.append(self.pa.get_device_info_by_index(i))
            if self.pa.get_device_info_by_index(i)['maxOutputChannels'] > 0:
                self.output_devices.append(self.pa.get_device_info_by_index(i))

    def __repr__(self):
        state = False
        if self.pastream:
            state = self.pastream.is_active()
        msg = f"""AServer: sr: {self.sr}, blocksize: {self.bs}, Stream Active: {state}
Device: {self.device_dict['name']}, Index: {self.device_dict['index']}"""
        return msg

    def get_devices(self):
        print("Input Devices: ")
        [print(f"Index: {i['index']}, Name: {i['name']},  Channels: {i['maxInputChannels']}")
         for i in self.input_devices]
        print("Output Devices: ")
        [print(f"Index: {i['index']}, Name: {i['name']}, Channels: {i['maxOutputChannels']}")
         for i in self.output_devices]
        return self.input_devices, self.output_devices

    def print_device_info(self):
        print("Input Devices: ")
        [print(i) for i in self.input_devices]
        print("\n")
        print("Output Devices: ")
        [print(o) for o in self.output_devices]

    def set_device(self, idx, reboot=True):
        self.device = idx
        self.device_dict = self.pa.get_device_info_by_index(self.device)
        if reboot:
            self.quit()
            try:
                self.boot()
            except OSError:
                print("Error: Invalid device. Server did not boot.")

    def boot(self):
        """ boot Aserver = start stream, setting its callback to this callback"""
        if self.pastream is not None and self.pastream.is_active():
            print("Aserver:boot: already running...")
            return -1
        self.pastream = self.pa.open(format=self.format, channels=self.channels, rate=self.sr,
                                     input=False, output=True, frames_per_buffer=self.bs,
                                     output_device_index=self.device, stream_callback=self._play_callback)

        self.boot_time = time.time()
        self.block_time = self.boot_time
        self.block_cnt = 0
        self.pastream.start_stream()
        print("Server Booted")
        return self

    def quit(self):
        """Aserver quit server: stop stream and terminate pa"""
        if not self.pastream.is_active():
            print("Aserver:quit: stream not active")
            return -1
        try:
            self.pastream.stop_stream()
            self.pastream.close()
        except AttributeError:
            print("No stream...")
        print("Aserver stopped.")
        self.pastream = None

    def __del__(self):
        self.pa.terminate()

    def play(self, asig, onset=0, out=0, **kwargs):
        """dispatch asigs or arrays for given onset"""
        if out < 0:
            print("Aserver:play: illegal out<0")
            return
        sigid = id(asig)  # for copy check
        if asig.sr != self.sr:
            asig = asig.resample(self.sr)
        if onset < 1e6:
            onset = time.time() + onset
        idx = np.searchsorted(self.srv_onsets, onset)
        self.srv_onsets.insert(idx, onset)
        if asig.sig.dtype != self.dtype:
            if id(asig) == sigid:
                asig = copy.copy(asig)
            asig.sig = asig.sig.astype(self.dtype)
        # copy only relevant channels...
        nchn = min(asig.channels, self.channels - out)  # max number of copyable channels
        # in: [:nchn] out: [out:out+nchn]
        if id(asig) == sigid:
            asig = copy.copy(asig)
        if len(asig.sig.shape) == 1:
            asig.sig = asig.sig.reshape(asig.samples, 1)
        asig.sig = asig.sig[:, :nchn].reshape(asig.samples, nchn)
        asig.channels = nchn
        # so now in callback safely copy to out:out+asig.sig.shape[1]
        self.srv_asigs.insert(idx, asig)
        self.srv_curpos.insert(idx, 0)
        self.srv_outs.insert(idx, out)
        return self

    def _play_callback(self, in_data, frame_count, time_info, flag):
        """callback function, called from pastream thread when data needed"""
        tnow = self.block_time
        self.block_time += self.block_duration
        self.block_cnt += 1
        self.timejitter = time.time() - self.block_time  # just curious - not needed but for time stability check

        if len(self.srv_asigs) == 0 or self.srv_onsets[0] > tnow:  # to shortcut computing
            return (self.empty_buffer, pyaudio.paContinue)

        data = np.zeros((self.bs, self.channels), dtype=self.dtype)
        # iterate through all registered asigs, adding samples to play
        dellist = []  # memorize completed items for deletion
        t_next_block = tnow + self.bs / self.sr
        for i, t in enumerate(self.srv_onsets):
            if t > t_next_block:  # doesn't begin before next block
                break  # since list is always onset-sorted
            a = self.srv_asigs[i]
            c = self.srv_curpos[i]
            if t > tnow:  # first block: apply precise zero padding
                io0 = int((t - tnow) * self.sr)
            else:
                io0 = 0
            tmpsig = a.sig[c:c + self.bs - io0]
            n, nch = tmpsig.shape
            out = self.srv_outs[i]
            data[io0:io0 + n, out:out + nch] += tmpsig  # .reshape(n, nch) not needed as moved to play
            self.srv_curpos[i] += n
            if self.srv_curpos[i] >= a.samples:
                dellist.append(i)  # store for deletion
        # clean up lists
        for i in dellist[::-1]:  # traverse backwards!
            del(self.srv_asigs[i])
            del(self.srv_onsets[i])
            del(self.srv_curpos[i])
            del(self.srv_outs[i])
        return (data * (self.range * self.gain), pyaudio.paContinue)
