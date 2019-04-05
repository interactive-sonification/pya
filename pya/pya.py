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

from .helpers import ampdb, linlin, dbamp
# from IPython import get_ipython
# from IPython.core.magic import Magics, cell_magic, line_magic, magics_class

default_aserver = None

class Asig:
    'audio signal class'
    def __init__(self, sig, sr=44100, bs = 1024, label="", channels=1):
        self.sr = sr
        self.bs = bs #buffer size for pyaudio
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
            try:
                self.channels = self.sig.shape[1]
            except IndexError:
                self.channels = 1
        self.samples = np.shape(self.sig)[0]
        self.label = label
        self.device_index = 1

    def load_wavfile(self, fname):
        # Discuss to change to float32 . 

        self.sr, self.sig = wavfile.read(fname) # load the sample data
        if self.sig.dtype == np.dtype('int16'):
            self.sig = self.sig.astype('float32')/32768
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

        # Set channel here. 
    
    def save_wavfile(self, fname="asig.wav", dtype='float32'):
        if dtype == 'int16':
            data = (self.sig*32767).astype('int16')
        elif dtype == 'int32':
            data = (self.sig*2147483647).astype('int32')
        elif dtype == 'uint8':
            data = (self.sig*127 + 128).astype('uint8')
        elif dtype == 'float32':
            data = self.sig.astype('float32')
        wavfile.write(fname, self.sr, data)
        return self
            
    def __getitem__(self, index):
        if isinstance(index, int):
            start, stop, step = 0, index, 1
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self.sig))    # index is a slice
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            return Asig(self.sig[index], self.sr, bs = self.bs, label = self.label+"_arrayindexed")
        elif isinstance(index, str):
            return self._[index]
        else:
            raise TypeError("index must be int, array, or slice")        
        return Asig(self.sig[start:stop:step], int(self.sr/abs(step)), bs = self.bs, label= self.label+"_sliced")

    #TODO: this method is not checked with multichannels. 
    def tslice(self, *tidx):
        if len(tidx) == 1: # stop
            sl = slice(0, tidx[0]*self.sr)
        elif len(tidx) == 2: # start and stop:
            sl = slice(int(tidx[0]*self.sr), int(tidx[1]*self.sr))
        else:
            sl = slice(int(tidx[0]*self.sr), int(tidx[1]*self.sr), tidx[2])
        return Asig(self.sig[sl], self.sr, self.label+"_tsliced")

    """
        Bug report: resample cant deal with multichannel 
    """
    
    # def resample(self, target_sr=44100, rate=1, kind='linear'):
    #     # This only work for single channel. 
    #     times = np.arange(self.samples )/self.sr

    #     interp_fn = scipy.interpolate.interp1d(times, self.sig, kind=kind, 
    #                 assume_sorted=True, bounds_error=False, fill_value=self.sig[-1])
    #     tsel = np.arange(self.samples/self.sr * target_sr/rate)*rate/target_sr
    #     return Asig(interp_fn(tsel), target_sr, label=self.label+"_resampled")

    def resample(self, target_sr=44100, rate=1, kind='linear'):
        """
            Resample signal based on interpolation, can process multichannel
        """
        times = np.arange(self.samples )/self.sr
        tsel = np.arange(self.samples/self.sr * target_sr/rate)*rate/target_sr
        if self.channels == 1:
            interp_fn = scipy.interpolate.interp1d(times, self.sig, kind=kind, 
                    assume_sorted=True, bounds_error=False, fill_value=self.sig[-1])
            return Asig(interp_fn(tsel), target_sr, label=self.label+"_resampled")
        else:
            new_sig = np.ndarray(shape = (int(self.samples/self.sr * target_sr/rate), self.channels))
            for i in range(self.channels):
                interp_fn = scipy.interpolate.interp1d(times, self.sig[:,i], kind=kind, 
                        assume_sorted=True, bounds_error=False, fill_value=self.sig[-1, i])
                new_sig[:, i] = interp_fn(tsel)
            return Asig(new_sig, target_sr, label=self.label+"_resampled")

        # This part uses pyaudio for playing. 
    def _playpyaudio(self, device_index=1):
        """
            play function take signal and channels as arguments. 
            device_index needs to be set properly: currently this method is not robust, as you need to 
            manually adjust it for external soundcard output. 
        """
        try:
            self.device_index = device_index
            self.audiostream = PyaudioStream(bs=self.bs, sr=self.sr, device_index=self.device_index)
            self.audiostream.play(self.sig, chan=self.channels)
            return self
        except ImportError:
            raise ImportError("Can't play audio via Pyaudiostream")
        
    def play(self, rate=1, device_index=1):
        """
        Play audio using pyaudio. 1. Resample the data if needed. 
            @This forces the audio to be always played at 44100, it is not effective. 
        """
        # if not self.sr in [8000, 11025, 22050, 44100, 48000]:
        #     print("resample as sr is exotic")
        #     return self.resample(44100, rate).play()['play']
        # else:
        #     if rate is not 1:
        #         print("resample as rate!=1")
        #         return self.resample(44100, rate).play()['play']
            
        #         # self._['play'] = self.resample(44100, rate).play()['play']
        #     else:
        #         return self._playpyaudio(device_index = device_index)

        if not self.sr in [8000, 11025, 22050, 44100, 48000]:
            print("resample as sr is exotic")
            self._['play'] = self.resample(44100, rate).play()['play']
        else:
            if rate is not 1:
                print("resample as rate!=1")
                self._['play'] = self.resample(44100, rate).play()['play']
            
                # self._['play'] = self.resample(44100, rate).play()['play']
            else:
                self._['play'] = self._playpyaudio(device_index = device_index)
        return self

    def stop(self):
        """
            Stop playing
        """
        try:
            self._['play'].audiostream.stopPlaying()
        except KeyError:
            print ("No play no stop, nothing happened.")
        return self

    def route(self, out=0):
        """
            Route the signal to n channel starting with out (type int):
                out = 0: does nothing as the same signal is being routed to the same position
                out > 0: move the first channel of self.sig to out channel, other channels follow
                out < 0: negative slicing, if overslicing, do nothing. 
        """
        if type(out) is not int:
            print ("Warning: route: out needs to be integer, nothing happened")
            return self
        else:
            if out == 0:  #If 0 do nothing. 
                return self
            elif out > 0: 
                # not optimized method here
                new_sig = np.zeros((self.samples, out + self.channels))
                new_sig[:, out:out + self.channels] = self.sig
                return Asig(new_sig, self.sr, label=self.label)

            elif out < 0 and -out < self.channels :
                new_sig = self.sig[:, -out:]
                return Asig(new_sig, self.sr, label=self.label)
            else:
                print ("left shift over the total channel, nothing happened")
                return self

    def panning(self,  pan):
        """
            @TODO, need rework now route is in a separate method. 
            There are two possible ways of using pos:
            pos as a list: [0, 1, 0, 1] : they become the multiplier. 
        """
        if type(pan) is int:
            # How do I mapped a stereo signal to one channel. 
            if self.channels > 1 and self.channels > pan:
                print ("Warning: Assigning a multichannel signal to a single channel. Signal will be merged and averaged.")
                sig_merged = self.sig.mean(axis = 1) # L + R /2 method, but not good with phase issue 
                # Assign 
                sig_nchan = np.ndarray(shape = (len(sig_merged) , self.channels))
                sig_nchan[:, pan] = sig_merged
            else:
                raise ValueError("Integer pan needs to be smaller than ASig.channels (max output channels)")
        elif type(pan) is list: 
            """
                Several possibilities here:
                    1. sig is mono:
                        convert to multi channels and apply gain. 
                    2. sig's channels equals pan size 
                    3. sig's channels > pan size and
                    4. sig's channels < pan size 
            """
            if np.max(pan) > 1:
                print ("Warning: list value should be between 0 ~ 1.") 
            if self.channels == 1: # if mono sig. 
                sig_nchan = self.mono2nchanel(self.sig, len(pan))
                sig_nchan *= pan # apply panning. 
            elif self.channels == len(pan):
                sig_nchan = self.sig * pan
            else:
                raise ValueError ("pan size and signal channels don't match")
            self.sig =  sig_nchan
        return self

    def overwrite(self, sig, sr = None):
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

    #TODO not checked. 
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

    # TODO not checked. 
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

    def size(self): 
        # return samples and length in time:
        return self.sig.shape, self.sig.shape[0]/self.sr

    def mono2nchanel(self, x , chan):
        c = np.vstack([x]*chan)
        return c.transpose()


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

#TODO, check with multichannel 
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


class Aserver:
    """
    Aserver manages an pyaudio stream, using its aserver callback 
    to feed dispatched signals to output at the right time 
    """
    def __init__(self, sr=44100, bs=256, device=1, channels=2, format=pyaudio.paInt16):
        self.sr = sr
        self.bs = bs
        self.device = device
        self.pa = pyaudio.PyAudio()
        self.channels = channels
        self.device_dict = self.pa.get_device_info_by_index(self.device)
        self.max_out_chn = self.device_dict['maxOutputChannels']
        if self.max_out_chn < self.channels:
            print(f"Aserver: warning: {channels}>{self.max_out_chn} channels requested - truncated.")
            self.channels = self.max_out_chn
        self.format = format
        self.gain = 1.0
        self.asigs = []
        self.onsets = []
        self.curpos = []  # start of next frame to deliver
        self.pastream = None
        self.empty_buffer = np.zeros((self.bs, self.channels), dtype='int16')

    def __repr__(self):
        state = False
        if self.pastream: 
            state = self.pastream.is_active()
        return f"AServer: sr: {self.sr}, blocksize: {self.bs}, Stream Active: {state}"

    def boot(self):
        """ boot Aserver = start stream, setting its callback to this callback"""
        if not self.pastream is None and self.pastream.is_active():
            print("Aserver:boot: already running...")
            return -1
        self.pastream = self.pa.open(format=self.format, channels=self.channels, rate=self.sr,
            input=False, output=True, frames_per_buffer=self.bs, 
            output_device_index=self.device, stream_callback=self._play_callback)
        self.pastream.start_stream()
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
            print ("No stream...")
        print ("Aserver stopped.")
        self.pastream = None

    def __del__(self):
        self.pa.terminate()

    def play(self, asig, onset=0):
        """dispatch asigs or arrays for given onset"""
        if onset < 1e6:
            onset = time.time() + onset
        idx = np.searchsorted(self.onsets, onset)
        self.onsets.insert(idx, onset)
        # TODO: make sure that inserted asig has self.channels channels,
        # 'float32' dtype and correct sr
        # TODO: integrate out argument
        self.asigs.insert(idx, asig)
        self.curpos.insert(idx, 0)
        return self

    def _play_callback(self, in_data, frame_count, time_info, flag):
        """callback function, called from pastream thread when data needed"""
        tnow = time.time()
        if len(self.asigs) == 0 or self.onsets[0]>tnow:  # to shortcut computing
            return (self.empty_buffer, pyaudio.paContinue)

        data = np.zeros((self.bs, self.channels), dtype='float32')
        # iterate through all registered asigs, adding samples to play
        dellist = [] # memorize completed items for deletion
        for i, t in enumerate(self.onsets):
            if t > tnow: 
                # TODO: integrate more precise start with zero padding
                # at begin if (t + self.bs/self.sr > tnow)
                break  # since list is always onset-sorted
            a = self.asigs[i]
            c = self.curpos[i]
            tmpsig = a.sig[c:c+self.bs]
            n = tmpsig.shape[0]
            if n == self.bs:
                data += tmpsig.reshape(n, self.channels) 
                self.curpos[i] += self.bs
            else:  # must have reached end of asig 
                data[:n] += tmpsig.reshape(n, self.channels)
                dellist.append(i) # store for deletion
        # clean up list
        for i in dellist[::-1]:  # traverse backwards!
            del(self.asigs[i])
            del(self.onsets[i])
            del(self.curpos[i])
        outdata = (data * 32767 * self.gain).astype('int16')
        return (outdata, pyaudio.paContinue)
