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
            print ("Play Stream Stopped. ")
        except AttributeError:
            print ("No stream, stop button did nothing. ")

    def play(self, sig, chan = 1):
        """
            Step 1: try to close any excessive stream. 
            Step 2: check if there's any input sigal, if not play internal buffer, else do nothing
            Step 3: Convert data into int16
            Step 4: Convert to the correct channels (not implemented at this stage. )
            Step 5: Convert the sig in to certain chunks for playback: 
                This method needs to be changed to suit multiple sounds being played together. 
            Step 6: Switch on the stream. 
        """
        self.outputChannels = chan
        try:  
            self.playStream.stop_stream()
            self.playStream.close()
        except AttributeError:
            pass

        if chan > self.maxOutputChannels:
            sig = sig[:, :self.maxOutputChannels]
            self.outputChannels = self.maxOutputChannels

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

        sig_long = sig.reshape(sig.shape[0]*sig.shape[1]) if self.outputChannels > 1 else sig

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
        l = lst.copy() # Make a copy if not it will change the input. 
        l.resize((ceil(l.shape[0]/chunk),chunk), refcheck=False)
        l.reshape(l.shape[0] * l.shape[1])
        # l_flat =l.flatten()
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

# Again, not dealing with multichannel yet. 
class SequenceStream():
    def __init__(self, bs = 256,  sr = 44100, channels = 1):
        self.pa = pyaudio.PyAudio()
        self.outputChannels =  channels # Work on this later. 
        self.fs = sr; self.chunk = bs # The smaller the smaller latency 
        self.audioformat = pyaudio.paInt16
        self.outputIdx = 1
        self.emptybuffer = bytes(np.zeros(self.chunk * self.outputChannels))
        self.ringbuffer = deque([self.emptybuffer,self.emptybuffer, self.emptybuffer,self.emptybuffer], maxlen = 4)

    def openstream(self):
        try:  
            self.playStream.stop_stream()
            self.playStream.close()
        except AttributeError:
            pass
        self.dataflag = False
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

    def play(self, onset, sig):
        if onset is list and sig is list:
            self.play_data = self.mixing_function(onset, sig)
            self.play_data = self.makechunk(self.play_data, self.chunk)
            self.len = len(self.play_data)
            self.count = 0
            self.dataflag = True
        else: 
            raise TypeError("Onset and sig need to be list ")


    def mixing_function(self, onset,sig):
        maxlen = np.max([o + len(s) for o, s in zip(onset, sig)])
        result =  np.zeros(maxlen)
        for i in range(len(onset)):
            result[onset[i]:onset[i] + len(sig[i])] += sig[i] 
        return result

    def makechunk(self, lst, chunk):
        result = []
        # TODO. Find a faster solution for this. 
        for i in np.arange(0, len(lst), chunk):
            temp = lst[i:i + chunk]
            if len(temp) < chunk:
                temp = np.pad(temp, (0, chunk - len(temp)), 'constant')
            result.append(temp)
        return result

    
    def _playcallback(self, in_data, frame_count, time_info, flag):  
        # Create a ring buffer here. 

        if (self.dataflag):
            out_data = self.play_data(self.count)
            self.count +=1
            if self.count > self.len:
                self.dataflag = False
        # out_data = self.ringbuffer[0]
        # self.ringbuffer.rotate(-1)
        # # Here to decide what should give to append to the ring buffer. . 
        # self.ringbuffer[-1] = self.emptybuffer
        else:
            out_data = self.emptybuffer
        return out_data, pyaudio.paComplete


    def closestream(self):
        try: # To prevent self.playStream not created before stop button pressed
            self.playStream.stop_stream()
            self.playStream.close()
            print ("Play Stream Stopped. ")
        except AttributeError:
            print ("No stream, stop button did nothing. ")

