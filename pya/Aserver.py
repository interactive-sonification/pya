import copy
import time
import logging
import numpy as np
import pyaudio
from warnings import warn

from sys import platform

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class Aserver:
    """Pya audio server
    Based on pyaudio, works as a FIFO style audio stream pipeline,
    allowing Asig.play() to send audio segement into the stream.

    Examples:
    -----------
    from pya import *

    ser = Aserver()

    ser.boot()

    asine = Ugen().sin().play(server=ser)

    """

    default = None  # that's the default Aserver if Asigs play via it

    @staticmethod
    def startup_default_server(**kwargs):
        if Aserver.default is None:
            _LOGGER.info("Aserver startup_default_server: create and boot")
            Aserver.default = Aserver(**kwargs)  # using all default settings
            Aserver.default.boot()
            # print(Aserver.default)
            _LOGGER.info("Default server info: %s", Aserver.default)
        else:
            _LOGGER.info("Aserver default_server already set.")
        return Aserver.default

    @staticmethod
    def shutdown_default_server():
        if isinstance(Aserver.default, Aserver):
            Aserver.default.quit()
            del(Aserver.default)
            Aserver.default = None
        else:
            warn("Aserver:shutdown_default_server: no default_server to shutdown")

    def __init__(self, sr=44100, bs=256, device=None, channels=2, format=pyaudio.paFloat32):
        """Aserver manages an pyaudio stream, using its aserver callback
        to feed dispatched signals to output at the right time.

        Parameters
        ----------
        sr : int
            Sampling rate (Default value = 44100)
        bs : int
            block size or buffer size (Default value = 256)
        channels : int
            number of channel (Default value = 2)
        format : pyaudio.Format
             Audio data format(Default value = pyaudio.paFloat32)

        Returns
        -------
        _ : numpy.ndarray
            numpy array of the recorded audio signal.
        """
        # TODO check if channels is overwritten by the device.
        self.sr = sr
        self.bs = bs
        self.pa = pyaudio.PyAudio()
        self.channels = channels
        self._status = pyaudio.paComplete

        # Get audio devices to input_device and output_device
        self.input_devices = []
        self.output_devices = []
        for i in range(self.pa.get_device_count()):
            if self.pa.get_device_info_by_index(i)['maxInputChannels'] > 0:
                self.input_devices.append(self.pa.get_device_info_by_index(i))
            if self.pa.get_device_info_by_index(i)['maxOutputChannels'] > 0:
                self.output_devices.append(self.pa.get_device_info_by_index(i))

        if device is None:
            self.device = self.pa.get_default_output_device_info()['index']
        else:
            self.device = device

        self.device_dict = self.pa.get_device_info_by_index(self.device)
        # self.max_out_chn is not that useful: there can be multiple devices having the same mu
        self.max_out_chn = self.device_dict['maxOutputChannels']
        if self.max_out_chn < self.channels:
            warn(f"Aserver: warning: {channels}>{self.max_out_chn} channels requested - truncated.")
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
        self._stop = True
        if self.format == pyaudio.paInt16:
            self.dtype = 'int16'
            self.range = 32767
        if self.format not in [pyaudio.paInt16, pyaudio.paFloat32]:
            warn(f"Aserver: currently unsupported pyaudio format {self.format}")
        self.empty_buffer = np.zeros((self.bs, self.channels), dtype=self.dtype)

    def __repr__(self):
        state = False
        if self.pastream:
            state = self.pastream.is_active()
        msg = f"""AServer: sr: {self.sr}, blocksize: {self.bs},
         Stream Active: {state}, Device: {self.device_dict['name']}, Index: {self.device_dict['index']}"""
        return msg

    def get_devices(self):
        """Print available input and output device."""
        print("Input Devices: ")
        [print(f"Index: {i['index']}, Name: {i['name']},  Channels: {i['maxInputChannels']}")
         for i in self.input_devices]
        print("Output Devices: ")
        [print(f"Index: {i['index']}, Name: {i['name']}, Channels: {i['maxOutputChannels']}")
         for i in self.output_devices]
        return self.input_devices, self.output_devices

    def print_device_info(self):
        """Print device info"""
        print("Input Devices: ")
        [print(i) for i in self.input_devices]
        print("\n")
        print("Output Devices: ")
        [print(o) for o in self.output_devices]

    def set_device(self, idx, reboot=True):
        """Set audio device 

        Parameters
        ----------
        idx : int
            Index of the device
        reboot : bool
            If true the server will reboot. (Default value = True)
        """
        self.device = idx
        self.device_dict = self.pa.get_device_info_by_index(self.device)
        if reboot:
            self.quit()
            try:
                self.boot()
            except OSError:
                warn("Error: Invalid device. Server did not boot.")

    def boot(self):
        """boot Aserver = start stream, setting its callback to this callback."""
        if self.pastream is not None and self.pastream.is_active():
            _LOGGER.info("Aserver already running...")
            return -1
        self.pastream = self.pa.open(format=self.format, channels=self.channels, rate=self.sr,
                                     input=False, output=True, frames_per_buffer=self.bs,
                                     output_device_index=self.device, stream_callback=self._play_callback)
        self.boot_time = time.time()
        self.block_time = self.boot_time
        self.block_cnt = 0
        self.pastream.start_stream()
        _LOGGER.info("Server Booted")
        return self

    def quit(self):
        """Aserver quit server: stop stream and terminate pa"""
        if not self.pastream.is_active():
            _LOGGER.info("Aserver:quit: stream not active")
            return -1
        try:
            self.pastream.stop_stream()
            self.pastream.close()
            _LOGGER.info("Aserver stopped.")
        except AttributeError:
            _LOGGER.info("No stream found...")
        self.pastream = None

    def __del__(self):
        self.pa.terminate()

    def play(self, asig, onset=0, out=0, **kwargs):
        """Dispatch asigs or arrays for given onset."""
        self._stop = False
        self._status = pyaudio.paContinue

        sigid = id(asig)  # for copy check
        if asig.sr != self.sr:
            asig = asig.resample(self.sr)
        if onset < 1e6:
            rt_onset = time.time() + onset
        else:
            rt_onset = onset
        idx = np.searchsorted(self.srv_onsets, rt_onset)
        self.srv_onsets.insert(idx, rt_onset)
        if asig.sig.dtype != self.dtype:
            warn("Not the same type. ")
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
        # asig.channels = nchn
        # so now in callback safely copy to out:out+asig.sig.shape[1]
        self.srv_asigs.insert(idx, asig)
        self.srv_curpos.insert(idx, 0)
        self.srv_outs.insert(idx, out)
        if 'block' in kwargs and kwargs['block']:
            if onset > 0:  # here really omset and not rt_onset!
                warn("blocking inactive with play(onset>0)")
            else:
                time.sleep(asig.get_duration())
        return self

    def _play_callback(self, in_data, frame_count, time_info, flag):
        """callback function, called from pastream thread when data needed."""
        tnow = self.block_time
        self.block_time += self.block_duration
        self.block_cnt += 1
        # just curious - not needed but for time stability check
        self.timejitter = time.time() - self.block_time
        if self.timejitter > 3 * self.block_duration:
            _LOGGER.debug(f"Aserver late by {self.timejitter} seconds: block_time reset!")
            self.block_time = time.time()

        if not self.srv_asigs or self.srv_onsets[0] > tnow:  # to shortcut computing
            return (self.empty_buffer, pyaudio.paContinue)
        elif self._stop:
            self.srv_asigs.clear()
            self.srv_onsets.clear()
            self.srv_curpos.clear()
            self.srv_outs.clear()
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

    def stop(self):
        self._stop = True
