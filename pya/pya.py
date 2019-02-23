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

from .helpers import ampdb, linlin, dbamp, play

# from IPython import get_ipython
# from IPython.core.magic import Magics, cell_magic, line_magic, magics_class


class Asig:
    'audio signal class'
    
    def __init__(self, sig, sr=44100, label="", channels=1):
        self.sr = sr
        self._ = {}  # dictionary for further return values
        self.channels = channels
        if isinstance(sig, str):
            self.load_wavfile(sig) 
        elif isinstance(sig, int):  # sample length
            if self.channels==1:
                self.sig = np.zeros(sig)
            else:
                self.sig = np.zeros((sig, self.channels))
        elif isinstance(sig, float): # if float interpret as duration
            if self.channels==1:
                self.sig = np.zeros(int(sig*sr))
            else:
                self.sig = np.zeros((int(sig*sr), self.channels))
        else:
            self.sig = np.array(sig)
        self.label = label
        sigshape = np.shape(self.sig)
        self.samples = sigshape[0]
        if len(sigshape) > 1:
            self.channels = sigshape[1]

    def load_wavfile(self, fname):
        self.sr, self.sig = wavfile.read(fname) # load the sample data
        if self.sig.dtype == np.dtype('int16'):
            self.sig = self.sig.astype('float64')/32768
        elif self.sig.dtype != np.dtype('float64'):
            self.sig = self.sig.astype('float64')
        else:
            print("load_wavfile: TODO: add format")
    
    def save_wavfile(self, fname="asig.wav", dtype='float32'):
        if dtype == 'int16':
            data = (self.sig*32767).astype('int16')
        elif dtype == 'int32':
            data = (self.sig*2147483647).astype('int32')
        elif dtype == 'uint8':
            data = (self.sig*127 + 128).astype('uint8')
        elif dtype == 'float32':
            data = self.sig.astype('float32')
        scipy.io.wavfile.write(fname, self.sr, data)
        return self
            
    def __getitem__(self, index):
        if isinstance(index, int):
            start, stop, step = 0, index, 1
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self.sig))    # index is a slice
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return Asig(self.sig[index], self.sr, self.label+"_arrayindexed")
        else:
            raise TypeError("index must be int, array, or slice")        
        return Asig(self.sig[start:stop:step], int(self.sr/abs(step)), self.label+"_sliced")

    def tslice(self, *tidx):
        if len(tidx) == 1: # stop
            sl = slice(0, tidx[0]*self.sr)
        elif len(tidx) == 2: # start and stop:
            sl = slice(int(tidx[0]*self.sr), int(tidx[1]*self.sr))
        else:
            sl = slice(int(tidx[0]*self.sr), int(tidx[1]*self.sr), tidx[2])
        return Asig(self.sig[sl], self.sr, self.label+"_tsliced")

    def resample(self, target_sr=44100, rate=1, kind='linear'):
        times = np.arange(self.samples)/self.sr
        interp_fn = scipy.interpolate.interp1d(times, self.sig, kind=kind, 
                    assume_sorted=True, bounds_error=False, fill_value=self.sig[-1])
        tsel = np.arange(self.samples/self.sr * target_sr/rate)*rate/target_sr
        return Asig(interp_fn(tsel), target_sr, label=self.label+"_resampled")

    def play(self, rate=1, block=False):
        if not self.sr in [8000, 11025, 22050, 44100, 48000]:
            print("resample as sr is exotic")
            self._['play'] = self.resample(44100, rate).play(block=block)
        else:
            if rate is not 1:
                print("resample as rate!=1")
                self._['play'] = self.resample(44100, rate).play(block=block)
            else:
                self._['play'] = play(self.sig, self.channels, self.sr, block=block)
        return self
    
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
        elif not amp: # default 1 if neither is given
            amp = 1
        return Asig(self.sig*amp, self.sr, label=self.label+"_scaled")
    
    def rms(self, axis=0):
        return np.sqrt(np.mean(np.square(self.sig), axis))
        
    def plot(self, fn=None, **kwargs):
        if fn:
            if fn=='db':
                fn=lambda x: np.sign(x) * ampdb((abs(x)*2**16 + 1))
            elif not callable(fn):
                print("Asig.plot: fn is neither keyword nor function")
                return self
            plot_sig = fn(self.sig) 
        else:
            plot_sig = self.sig
        self._['plot'] = plt.plot(np.arange(0, self.samples)/self.sr, plot_sig, **kwargs)
        return self

    def get_duration(self):
        return self.samples/self.sr

    def get_times(self):
        return np.linspace(0, (self.samples-1)/self.sr, self.samples) # timestamps for left-edge of sample-and-hold-signal

    def __repr__(self):
        return "Asig('{}'): {} x {} @ {} Hz = {:.3f} s".format(
            self.label, self.channels, self.samples, self.sr, self.samples/self.sr)
    
    def find_events(self, step_dur=0.001, sil_thr=-20, sil_min_dur=0.1, sil_pad=[0.001,0.1]):
        if self.channels>1:
            print("warning: works only with 1-channel signals")
            return -1
        step_samples = int(step_dur * self.sr)
        sil_thr_amp = dbamp(sil_thr)
        sil_flag = True
        sil_min_steps = int(sil_min_dur / step_dur)
        if type(sil_pad) is list:
            sil_pad_samples = [int(v*self.sr) for v in sil_pad]
        else:
            sil_pad_samples = (int(sil_pad * self.sr), )*2 
        
        event_list = []
        for i in range(0, self.samples, step_samples):
            rms = self[i:i+step_samples].rms()
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
        self._['events']= np.array(event_list)
        return self

    def select_event(self, index=None, onset=None):
        if not 'events' in self._:
            print('select_event: no events, return all')
            return self
        events = self._['events']
        if onset:
            index = np.argmin(np.abs(events[:,0]-onset*self.sr))
        if not index is None:
            beg, end = events[index]
            print(beg, end)
            return Asig(self.sig[beg:end], self.sr, label=self.label + f"event_{index}")
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
        nsamp = int(dur*self.sr)
        if nsamp>self.samples:
            nsamp = self.samples
            print("warning: Asig too short for fade_in - adapting fade_in time")
        return Asig(np.hstack((self.sig[:nsamp] * np.linspace(0, 1, nsamp)**curve, self.sig[nsamp:])),
                    self.sr, label=self.label+"_fadein")
    
    def fade_out(self, dur=0.1, curve=1):
        nsamp = int(dur*self.sr)
        if nsamp > self.samples:
            nsamp = self.samples
            print("warning: Asig too short for fade_out - adapting fade_out time")
        return Asig(np.hstack((self.sig[:-nsamp], 
                               self.sig[-nsamp:] * np.linspace(1, 0, nsamp)**curve
                              )),
                    self.sr, label=self.label+"_fadeout")

    def iirfilter(self, cutoff_freqs, btype='bandpass', ftype='butter', order=4, 
                    filter='lfilter', rp=None, rs=None):
        Wn = np.array(cutoff_freqs)*2/self.sr
        b, a = scipy.signal.iirfilter(order, Wn, rp=rp, rs=rs, btype=btype, ftype=ftype)
        y = scipy.signal.__getattribute__(filter)(b, a, self.sig)
        aout = Asig(y, self.sr, label=self.label+"_iir")
        aout._['b']= b
        aout._['a']= a
        return aout

    def plot_freqz(self, worN, **kwargs):
        w, h = scipy.signal.freqz(self._['b'], self._['a'], worN)
        plt.plot(w*self.sr/2/np.pi, ampdb(abs(h)), **kwargs)

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
            pos = int(onset*self.sr)
        if not pos:
            pos = 0  # add to begin if neither pos nor onset have been specified
        last = pos + n
        if last > self.samples:
            last = self.samples
            sigar = sigar[:last-pos]
        self.sig[pos:last] += amp*sigar
        return self
    
    def envelope(self, amps, ts=None, curve=1, kind='linear'):
        nsteps = len(amps)
        duration = self.samples/self.sr
        if nsteps == self.samples:
            sig_new = self.sig * amps**curve
        else:
            if not ts:
                given_ts = np.linspace(0, duration, nsteps)
            else:
                if nsteps != len(ts):
                    print("Asig.envelope error: len(amps)!=len(ts)")
                    return self
                if all(ts[i] < ts[i+1] for i in range(len(ts)-1)): # if list is monotonous
                    if ts[0] > 0: # if first t > 0 extend amps/ts arrays prepending item
                        ts = np.insert(np.array(ts), 0, 0)
                        amps = np.insert(np.array(amps), 0, amps[0])
                    if ts[-1] < duration: # if last t < duration append amps/ts value
                        ts = np.insert(np.array(ts), -1, duration)
                        amps = np.insert(np.array(amps), -1, amps[-1])
                else:
                    print("Asig.envelope error: ts not sorted")
                    return self
                given_ts = ts  
            if nsteps != self.samples:
                interp_fn = scipy.interpolate.interp1d(given_ts, amps, kind=kind)
                sig_new = self.sig * interp_fn(np.linspace(0, duration, self.samples))**curve  # ToDo: curve segmentwise!!!
        return Asig(sig_new, self.sr, label=self.label+"_enveloped")
    
    def adsr(self, att=0, dec=0.1, sus=0.7, rel=0.1, curve=1, kind='linear'):
        dur = self.get_duration()
        return self.envelope( [0, 1, sus, sus, 0], [0, att, att+dec, dur-rel, dur], 
                                curve=curve, kind=kind)

    def window(self, win='triang', **kwargs):
        if not win:
            return self
        winstr = win
        if type(winstr)==tuple: 
            winstr = win[0]
        return Asig(self.sig*scipy.signal.get_window(win, self.samples, **kwargs), self.sr, label=self.label+"_"+winstr)
    
    def window_op(self, nperseg=64, stride=32, win=None, fn='rms', pad='mirror'):
        centerpos = np.arange(0, self.samples, stride)
        nsegs = len(centerpos)
        res = np.zeros((nsegs, ))
        for i, cp in enumerate(centerpos):
            i0 = cp - nperseg//2
            i1 = cp + nperseg//2
            if i0 < 0: i0=0   # ToDo: correct padding!!!
            if i1 >= self.samples: i1=self.samples-1  # ToDo: correct padding!!!
            if isinstance(fn, str):
                res[i] = self[i0:i1].window(win).__getattribute__(fn)()
            else: # assume fn to be a function on Asig
                res[i] = fn(self[i0:i1])
        return Asig(np.array(res), sr=self.sr//stride, label='window_oped')
    
    def overlap_add(self, nperseg=64, stride_in=32, stride_out=32, jitter_in=None, jitter_out=None, 
                    win=None, pad='mirror'):
        # TODO: check with multichannel ASigs
        # TODO: allow stride_in and stride_out to be arrays of indices
        # TODO: add jitter_in, jitter_out parameters to reduce spectral ringing effects
        res = Asig( np.zeros((self.samples//stride_in*stride_out, )), sr=self.sr, label=self.label+'_ola')
        ii = 0
        io = 0
        for _ in range(self.samples//stride_in):
            i0 = ii - nperseg//2
            if jitter_in: i0 += np.random.randint(jitter_in)
            i1 = i0 + nperseg
            if i0 < 0: i0 = 0  # TODO: correct left zero padding!!!
            if i1 >= self.samples: 
                i1 = self.samples-1  # ToDo: correct right zero padding!!!
            pos = io
            if jitter_out: pos += np.random.randint(jitter_out)
            res.add(self[i0:i1].window(win).sig, pos=pos) 
            io += stride_out
            ii += stride_in
        return res
    
    def to_spec(self):
        return Aspec(self)

    def spectrum(self):
        nrfreqs = self.samples//2 + 1
        frq = np.linspace(0, 0.5*self.sr, nrfreqs) # one sides frequency range
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


class Aspec:
    'audio spectrum class using rfft'
    
    def __init__(self, x, sr=44100, label=None):
        if type(x) == Asig:
            self.sr = x.sr
            self.rfftspec = np.fft.rfft(x.sig)
            self.label = x.label+"_spec"
            self.samples = x.samples
            self.channels = x.channels
        elif type(x) == list or type(x) == np.ndarray:
            self.rfftspec = np.array(x)
            self.sr = sr 
            self.samples = (len(x)-1)*2
            self.channels = 1
            if len(np.shape(x))>1:
                self.channels = np.shape(x)[1]
        else:
            print("error: unknown initializer")
        if label:
            self.label = label
        self.nr_freqs = self.samples//2+1 
        self.freqs = np.linspace(0, self.sr/2, self.nr_freqs)
        
    def to_sig(self):
        return Asig(np.fft.irfft(self.rfftspec), sr=self.sr, label=self.label+'_2sig')

    def weight(self, weights, freqs=None, curve=1, kind='linear'):
        nfreqs = len(weights)
        if not freqs:
            given_freqs = np.linspace(0, self.freqs[-1], nfreqs)
        else:
            if nfreqs != len(freqs):
                print("Aspec.weight error: len(weights)!=len(freqs)")
                return self
            if all(freqs[i] < freqs[i+1] for i in range(len(freqs)-1)): # check if list is monotonous
                if freqs[0] > 0: 
                    freqs = np.insert(np.array(freqs), 0, 0)
                    weights = np.insert(np.array(weights), 0, weights[0])
                if freqs[-1] < self.sr/2:
                    freqs = np.insert(np.array(freqs), -1, self.sr/2)
                    weights = np.insert(np.array(weights), -1, weights[-1])
            else:
                print("Aspec.weight error: freqs not sorted")
                return self
            given_freqs = freqs  
        if nfreqs != self.nr_freqs:
            interp_fn = scipy.interpolate.interp1d(given_freqs, weights, kind=kind)
            rfft_new = self.rfftspec * interp_fn(self.freqs)**curve  # ToDo: curve segmentwise!!!
        else:
            rfft_new = self.rfftspec * weights**curve
        return Aspec(rfft_new, self.sr, label=self.label+"_weighted")
            
    def plot(self, fn=np.abs, **kwargs):
        plt.plot(self.freqs, fn(self.rfftspec), **kwargs)
        plt.xlabel('freq (Hz)') 
        plt.ylabel(f'{fn.__name__}(freq)')
        
    def __repr__(self):
        return "Aspec('{}'): {} x {} @ {} Hz = {:.3f} s".format(self.label, 
            self.channels, self.samples, self.sr, self.samples/self.sr)


class Astft:
    'audio spectrogram (STFT) class'

    def __init__(self, x, sr=None, label=None, window='hann', nperseg=256, 
                noverlap=None, nfft=None, detrend=False, return_onesided=True, 
                boundary='zeros', padded=True, axis=-1):
        self.window = window
        self.nperseg = nperseg
        self.noverlap =  noverlap
        self.nfft = nfft
        self.detrend = detrend
        self.return_onesided = return_onesided
        self.boundary = boundary
        self.padded = padded
        self.axis = axis
        if type(x) == Asig:
            self.sr = x.sr
            if sr: self.sr = sr  # explicitly given sr overwrites Asig
            self.freqs, self.times, self.stft = scipy.signal.stft(x.sig, fs=self.sr, window=window,
                nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, 
                return_onesided=return_onesided, boundary=boundary, padded=padded, axis=axis)
            self.label = x.label+"_stft"
            self.samples = x.samples
            self.channels = x.channels
        elif type(x) == np.ndarray and np.shape(x)>=2:
            self.stft = x
            self.sr = 44100
            if sr: self.sr = sr 
            self.samples = (len(x)-1)*2
            self.channels = 1
            if len(np.shape(x))>2:
                self.channels = np.shape(x)[2]
            # TODO: set other values, particularly check if self.times and self.freqs are correct
            self.ntimes, self.nfreqs, = np.shape(self.stft)
            self.times = np.linspace(0, (self.nperseg-self.noverlap)*self.ntimes/self.sr, self.ntimes)
            self.freqs = np.linspace(0, self.sr//2, self.nfreqs) 
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
        return Asig(sig, sr=self.sr, label=self.label+'_2sig')
            
    def plot(self, fn = lambda x: x):
        plt.pcolormesh(self.times, self.freqs, fn(np.abs(self.stft)))
        plt.colorbar()
        return self
        
    def __repr__(self):
        return "Astft('{}'): {} x {} @ {} Hz = {:.3f} s".format(self.label, 
            self.channels, self.samples, self.sr, self.samples/self.sr)
