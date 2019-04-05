import pyaudio
import numpy as np
import time, threading
import scipy.signal as signal
from collections import deque
from math import ceil

class PyaudioStream():
    """Pyaudiostream class manages audio I/O via pyaudio."""
    def __init__(self, bs=256,  sr=44100,  device=1):
        self.pa = pyaudio.PyAudio()
        self.sr = sr  
        self.bs = bs 
        self.audioformat = pyaudio.paInt16
        self.recording = False
        self.bufflist = []  # buffer for input audio tracks
        self.totalChunks = 0  # this is the len of bufflist. 
        self.gain = 1 
        self.inputIdx = 0 
        self.maxInputChannels = self.pa.get_device_info_by_index(self.inputIdx)['maxInputChannels']
        self.outputIdx = device
        self.maxOutputChannels = self.pa.get_device_info_by_index(self.outputIdx)['maxOutputChannels']
        self.inVol = np.ones(self.maxInputChannels) # Current version has 6 inputs max. But should taylor this. 
        self.outVol = np.ones(self.maxOutputChannels)
        
    def printAudioDevices(self):
        for i in range(self.pa.get_device_count()):
            print (self.pa.get_device_info_by_index(i))

    def _playcallback(self, in_data, frame_count, time_info, flag):
        if (self.frameCount < self.totalChunks):
            out_data = self.play_data[self.frameCount]
            self.frameCount += 1
            return bytes(out_data), pyaudio.paContinue
        else: # The last one
            out_data = np.zeros(self.bs * self.outputChannels)
            return bytes(out_data), pyaudio.paComplete

    def _recordCallback(self, in_data, frame_count, time_info, flag):
        audio_data = (np.frombuffer(in_data, dtype = np.int16) * self.inVol[0]).astype(np.int16)
        self.bufflist.append(audio_data)
        return audio_data, pyaudio.paContinue

    def record(self):
        # What is the chunk size? is it scaled based on input channels or output channels. 
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
            rate = self.sr,
            input = True,
            output = True,
            input_device_index=self.inputIdx,
            output_device_index=self.outputIdx,
            frames_per_buffer = self.bs,
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
        Convert the data type to int16
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
            msg = f"{sig.dtype} is not a supported date type. Supports int16, float64 and int8."
            raise TypeError(msg)
        return sig

    def play(self, sig, chan=1):
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
        self.play_data = self.make_chunk(sig_long, self.bs*self.outputChannels) 

        self.totalChunks = len(self.play_data) # total chunks
        self.frameCount = 0 # Start from the beginning. 

        self.playStream = self.pa.open(
            format = self.audioformat,
            channels = self.outputChannels, 
            rate = self.sr,
            input = False,
            output = True,
            output_device_index = self.outputIdx,
            frames_per_buffer = self.bs,
            stream_callback = self._playcallback
           )
        self.playStream.start_stream()

    
    def make_chunk(self, lst, chunk):
        """
            Return a list that is splice by the chunk
            Make chunk will pad the final chunk with zeros to match the buffersize. Although this is not necessary, 
            as pyaudio to finish callback automatically if the final chunk is smaller than buffersize. This however is 
            useful for you want a always on stream.
        """
        l = lst.copy() # Make a copy if not it will change the input. 
        l.resize((ceil(l.shape[0]/chunk),chunk), refcheck=False)
        l.reshape(l.shape[0] * l.shape[1])
        return l

    def mono2nchan(self, mono, channels=2):
        # convert a mono signal to multichannel by duplicating it to each channel. 
        return np.repeat(mono, channels)# This is actually quite slow

    def _outputgain(self, sig):
        out_data =  self._multichannelgain(sig, self.outputChannels, self.outVol)
        return bytes((out_data * self.gain).astype(np.int16))

    # Currently not implemented here. 
    def _multichannelgain(self, data, channels, glist):
        data = data.astype(np.float32)
        data_col = data.reshape([self.bs, channels])
        data_col *= glist[:channels] # Only apply the list matching the channels. 
        return data.astype(np.int16)
