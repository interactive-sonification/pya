import pyaudio
import numpy as np
import time
import scipy.signal as signal
from collections import deque
from math import ceil

class PyaudioStream():
    def __init__(self, bs = 256,  sr = 44100,  device_index  = 1):
        self.pa = pyaudio.PyAudio()
        self.fs = sr  # The divider helps us to use a smaller samping rate. 
        self.chunk = bs # The smaller the smaller latency 
        self.audioformat = pyaudio.paInt16
        self.recording = False
        self.bufflist = [] # Buffer for input audio tracks
        self.totalChunks = 0 # This is the len of bufflist. 
        self.masterVol = 1 # Master volume. 
        
        # self.il = [] ; self.ol = [] # out list
        # for i in range(self.pa.get_device_count()):
        #     if self.pa.get_device_info_by_index(i)['maxInputChannels'] > 0:
        #         self.il.append(self.pa.get_device_info_by_index(i))
        #     if self.pa.get_device_info_by_index(i)['maxOutputChannels'] > 0:
        #         self.ol.append(self.pa.get_device_info_by_index(i))        
        self.inputIdx = 0 
        self.maxInputChannels = self.pa.get_device_info_by_index(self.inputIdx)['maxInputChannels']
        self.outputIdx = device_index
        self.maxOutputChannels = self.pa.get_device_info_by_index(self.outputIdx)['maxOutputChannels']
        self.inVol = np.ones(self.maxInputChannels) # Current version has 6 inputs max. But should taylor this. 
        self.outVol = np.ones(self.maxOutputChannels)
        
    
    def printAudioDevices(self):
        for i in range(self.pa.get_device_count()):
            print (self.pa.get_device_info_by_index(i))


    def _playcallback(self, in_data, frame_count, time_info, flag):
        if (self.frameCount < self.totalChunks):
            # out_data = self._outputgain(self.play_data[self.frameCount])
            out_data = self.play_data[self.frameCount]
            self.frameCount += 1
            return bytes(out_data), pyaudio.paContinue
        else: # The last one . 
            out_data = np.zeros(self.chunk * self.outputChannels)
            return bytes(out_data), pyaudio.paComplete

    def _recordCallback(self, in_data, frame_count, time_info, flag):
        audio_data = (np.frombuffer(in_data, dtype = np.int16) * self.inVol[0]).astype(np.int16)
        self.bufflist.append(audio_data)
        # out_data = self._outputgain(middle_data) # Individual channel gain and master gain
        return audio_data, pyaudio.paContinue

    def record(self):
        # What is the chunk size ? is it scaled based on input channels or output channels. 
        try:  # restarting recording without closing stream, resulting an unstoppable stream. 
             # This can happen in Jupyter when you trigger record multple times without stopping it first. 
            self.stream.stop_stream()
            self.stream.close()
            # print ("Record stream closed. ")
        except AttributeError:
            pass
        print ("Start Recording")
        self.recording = True 
        self.bufflist = []
        self.stream = self.pa.open(
            format = self.audioformat,
            channels = self.outputChannels,
            rate = self.fs,
            input = True,
            output = True,
            input_device_index=self.inputIdx,
            output_device_index=self.outputIdx,
            frames_per_buffer = self.chunk,
            stream_callback=self._recordCallback)
        self.stream.start_stream()

    def stopRecording(self):
        try:  # To prevent self.stream not created before stop button pressed
            self.stream.stop_stream()
            self.stream.close()
            print ("Record stream closed. ")
        except AttributeError:
            print ("No stream, stop button did nothing. ")

    def stopPlaying(self):
        try: # To prevent self.playStream not created before stop button pressed
            self.playStream.stop_stream()
            self.playStream.close()
        except AttributeError:
            pass

    def toInt16(self, sig):
        """
        Conver the datatype to int16
        """
        if sig.dtype == np.dtype('float64'):
            sig = (sig * 32767).astype(np.int16)
        elif sig.dtype == np.dtype('float32'):
            sig = (sig * 32767).astype(np.int16)
        elif sig.dtype == np.dtype('int8'):
            sig = (sig * 256).astype(np.int16)
        elif sig.dtype == np.dtype('int16'): # Correct format
            pass
        else:
            msg = str(sig.dtype) + " is not a supported date type. Supports int16, float64 and int8."
            raise TypeError(msg)
        return sig

    def play(self, sig, chan = 1):
        """
            -> try to close any excessive stream. 
            -> check if signal channels more than device output channels, if so slice it. 
            -> Convert data into int16
            ->: Convert the sig in to certain chunks for playback: 
                This method needs to be changed to suit multiple sounds being played together. 
            -> Switch on the stream. 
        """
        self.outputChannels = chan
        self.stopPlaying() # Make sure there is no previous stream leftover. 
        if chan > self.maxOutputChannels:
            sig = sig[:, :self.maxOutputChannels]
            self.outputChannels = self.maxOutputChannels

        sig = self.toInt16(sig) # Make sure the data is int16. 
        sig_long = sig.reshape(sig.shape[0]*sig.shape[1]) if self.outputChannels > 1 else sig # Long format is only for multichannel

        # sig = self.mono2nchan(sig,self.outputChannels) # duplicate based on channels
        self.play_data = self.makechunk(sig_long, self.chunk*self.outputChannels) 

        self.totalChunks = len(self.play_data) # total chunks
        self.frameCount = 0 # Start from the beginning. 
        # This method will only work with pyqt, because if not it will only run 1 iteration.  
        self.playStream = self.pa.open(
            format = self.audioformat,
            channels = self.outputChannels, 
            rate = self.fs,
            input = False,
            output = True,
            output_device_index=self.outputIdx,
            frames_per_buffer = self.chunk,
            stream_callback=self._playcallback
           )
        self.playStream.start_stream()

    
    def makechunk(self, lst, chunk):
        """
            Return a list that is splice by the chunk
        """
        l = lst.copy() # Make a copy if not it will change the input. 
        l.resize((ceil(l.shape[0]/chunk),chunk), refcheck=False)
        l.reshape(l.shape[0] * l.shape[1])
        return l

    # def makechunk2(self, lst, chunk):
    #     result = []
    #     lst = np.pad(lst, (0, lst.shape[0]%chunk), 'constant')
    #     for i in np.arange(0, len(lst), chunk):
    #         temp = lst[i:i + chunk]
    #         result.append(temp)
    #     return result

    def mono2nchan(self, mono, channels = 2):
        # convert a mono signal to multichannel by duplicating it to each channel. 
        return np.repeat(mono, channels)# This is actually quite slow

        """
        
            c = np.vstack([b]*4)
            return c.transpose()
        """

    def _outputgain(self, sig):
        out_data =  self._multichannelgain(sig, \
                self.outputChannels, self.outVol)
        return bytes((out_data * self.masterVol).astype(np.int16))

    def _multichannelgain(self, data , channels,  glist):
        data = data.astype(np.float32)
        data_col = data.reshape([self.chunk, channels])
        data_col *= glist[:channels] # Only apply the list matching the channels. 
        return data.astype(np.int16)


class Soundserver(PyaudioStream):
    def __init__(self, bs = 256,  sr = 44100,  device_index  = 1):
        PyaudioStream.__init__(self, bs,  sr ,  device_index)
        self.outputChannels = self.maxOutputChannels # Make sure the output channels match the device max output
        self.emptybuffer = np.zeros(self.chunk * self.outputChannels).astype(np.int16)

    def unifySR(self, sig):
        """
        Check the sampling rate of each asig in the list, 
        if they are different, resample to the lowest sampling rate. 
        """
        srl = [s.sr for s in sig]
        if srl.count(srl[0]) == len(srl):
            return sig
        else:
            self.fs  = np.min(srl)
            for i in range(len(srl)): # resample asig that is > smallest sr. 
                if sig[i].sr != self.fs:
                    sig[i] = sig[i].resample(self.fs)
            return sig

    def _streamcallback(self, in_data, frame_count, time_info, flag):  
        if (self.framecount < self.len):
            out_data = self.play_data[self.framecount]
            self.framecount +=1
        else:
            out_data = self.emptybuffer
        return bytes(out_data), pyaudio.paContinue

    def open(self):
        """
            Only the server. It will have a constant callback
        """
        try:  
            self.serverStream.stop_stream()
            self.serverStream.close()
        except AttributeError:
            pass
        self.dataflag = False
        self.framecount = 0
        self.len = -1 
        self.serverStream = self.pa.open(
            format = self.audioformat,
            channels = self.outputChannels, 
            rate = self.fs,
            input = False,
            output = True,
            output_device_index=self.outputIdx,
            frames_per_buffer = self.chunk,
            stream_callback=self._streamcallback
           )
        self.serverStream.start_stream()
        

    def play(self, onset, siglist):
        """
            Play sequence: 
                onset: a list of timestamp for each sound to be play 
        """
        if len(onset) != len(siglist):
            raise AssertionError("Size of onset and signal lists need to be the same.")
        sig = self.unifySR(siglist) # Check if any difference in sampling rates
        sig = self.mixing(onset, sig)

        print ("Play Sound")
        

    def scale2channels(self, asigs):
        if asigs.channels == self.outputChannels:
            return asigs.sig# Dont do anything
        elif asigs.channels == 1:
            y = np.repeat(asigs.sig, self.outputChannels).reshape((len(asigs.sig), self.outputChannels))
            return y 
        elif asigs.channels > self.outputChannels:
            y = asigs.sig[:,:self.outputChannels] 
            return y
        elif asigs.channels < self.outputChannels:
            y = np.zeros(shape = (len(asigs.sig), self.outputChannels))
            y[:,:asigs.channels] = asigs.sig
            return y 


    def mixing(self, onset, sig):
        """
            What is the quickest way to blend all sigles. 
            1. mono signal needs to be scale to whatever 
            2. 
        """
        # maxlen only need to be check on one channel. 
        maxlen = np.max([o + len(s.sig) for o, s in zip(onset, sig)])
        # result =  np.zeros(maxlen) # This is wrong for multichannels. 
        sig_scaled = [self.scale2channels(s) for s in sig]
        result = np.zeros(shape = (maxlen, self.outputChannels))
        for i in range(len(onset)):
            result[onset[i]:onset[i] + len(sig_scaled[i]), :] += sig_scaled[i]
        return result

    def mix(self, onset, sig):
        siglengths = np.vectorize(len)(sig)
        maxlen = max(onset + siglengths)
        result = np.zeros(maxlen)
        for i in range(len(sig)):
            result[onset[i]: onset[i]+siglengths[i]] += sig[i]
        return result

    def closeserver(self):
        try: # To prevent self.playStream not created before stop button pressed
            self.serverStream.stop_stream()
            self.serverStream.close()
            print ("Play Stream Stopped. ")
        except AttributeError:
            print ("No stream, stop button did nothing. ")