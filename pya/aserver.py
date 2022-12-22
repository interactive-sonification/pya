import copy
import time
import logging
import numpy as np
from warnings import warn


_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


class Aserver:
    """Pya audio server
    Based on pyaudio, works as a FIFO style audio stream pipeline,
    allowing Asig.play() to send audio segement into the stream.

    Examples:
    -----------
    >>> from pya import *
    >>> ser = Aserver()
    >>> ser.boot()
    AServer: sr: 44100, blocksize: ...,
             Stream Active: True, Device: ...
    >>> asine = Ugen().sine()
    >>> asine.play(server=ser)
    Asig('sine'): 1 x 44100 @ 44100Hz = 1.000s cn=['0']
    """

    default = None  # that's the default Aserver if Asigs play via it

    @staticmethod
    def startup_default_server(**kwargs):
        if Aserver.default is None:
            _LOGGER.info("Aserver startup_default_server: create and boot")
            Aserver.default = Aserver(**kwargs)  # using all default settings
            Aserver.default.boot()
            _LOGGER.info("Default server info: %s", Aserver.default)
        else:
            _LOGGER.info("Aserver default_server already set.")
        return Aserver.default

    @staticmethod
    def shutdown_default_server():
        if isinstance(Aserver.default, Aserver):
            Aserver.default.quit()
            del Aserver.default
            Aserver.default = None
        else:
            warn("Aserver:shutdown_default_server: no default_server to shutdown")

    def __init__(self, sr=44100, bs=None, device=None,
                 channels=None, backend=None, **kwargs):
        """Aserver manages an pyaudio stream, using its aserver callback
        to feed dispatched signals to output at the right time.

        Parameters
        ----------
        sr : int
            Sampling rate (Default value = 44100)
        bs : int
            Override block size or buffer size set by chosen backend
        device : int
            The device index based on pya.device_info(), default is None which will set 
            the default device from PyAudio
        channels : int
            number of channel (Default value = 2)
        kwargs : backend parameter

        Returns
        -------
        _ : numpy.ndarray
            numpy array of the recorded audio signal.
        """
        # TODO check if channels is overwritten by the device.
        self.sr = sr
        if backend is None:
            from .backend.PyAudio import PyAudioBackend
            self.backend = PyAudioBackend(**kwargs)
        else:
            self.backend = backend
        self.bs = bs or self.backend.bs
        # Get audio devices to input_device and output_device
        self.input_devices = []
        self.output_devices = []
        for i in range(self.backend.get_device_count()):
            if int(self.backend.get_device_info_by_index(i)['maxInputChannels']) > 0:
                self.input_devices.append(self.backend.get_device_info_by_index(i))
            if int(self.backend.get_device_info_by_index(i)['maxOutputChannels']) > 0:
                self.output_devices.append(self.backend.get_device_info_by_index(i))

        self._device = device or self.backend.get_default_output_device_info()['index']
        self.channels = channels or self.backend.get_device_info_by_index(self.device)['maxOutputChannels']

        self.gain = 1.0
        self.srv_onsets = []
        self.srv_asigs = []
        self.srv_curpos = []  # start of next frame to deliver
        self.srv_outs = []  # output channel offset for that asig
        self.stream = None
        self.boot_time = 0  # time.time() when stream starts
        self.block_cnt = 0  # nr. of callback invocations
        self.block_duration = self.bs / self.sr  # nominal time increment per callback
        self.block_time = 0  # estimated time stamp for current block
        self._stop = True
        self.empty_buffer = np.zeros((self.bs, self.channels),
                                     dtype=self.backend.dtype)
        self._is_active = False

    @property
    def device_dict(self):
        return self.backend.get_device_info_by_index(self._device)

    @property
    def max_out_chn(self):
        return int(self.device_dict['maxOutputChannels'])

    @property
    def max_in_chn(self):
        return int(self.device_dict['maxInputChannels'])

    @property
    def device_dict(self):
        return self.backend.get_device_info_by_index(self._device)

    @property
    def max_out_chn(self):
        return int(self.device_dict['maxOutputChannels'])

    @property
    def max_in_chn(self):
        return int(self.device_dict['maxInputChannels'])

    @property
    def device(self):
        return self._device

    @property
    def is_active(self) -> bool:
        return self.stream is not None and self.stream.is_active()

    @device.setter
    def device(self, val):
        self._device = val if val is not None else self.backend.get_default_output_device_info()['index']
        if self.max_out_chn < self.channels:
            warn(f"Aserver: warning: {self.channels}>{self.max_out_chn} channels requested - truncated.")
            self.channels = self.max_out_chn

    def __repr__(self):
        state = False
        msg = f"""AServer: sr: {self.sr}, blocksize: {self.bs},
         Stream Active: {self.is_active}, Device: {self.device_dict['name']}, Index: {self.device_dict['index']}"""
        return msg

    def get_devices(self, verbose=False):
        """Return (and optionally print) available input and output device"""
        if verbose:
            print("Input Devices: ")
            [print(f"Index: {i['index']}, Name: {i['name']},  Channels: {i['maxInputChannels']}")
             for i in self.input_devices]
            print("Output Devices: ")
            [print(f"Index: {i['index']}, Name: {i['name']}, Channels: {i['maxOutputChannels']}")
             for i in self.output_devices]
        return self.input_devices, self.output_devices

    def set_device(self, idx, reboot=True):
        """Set audio device

        Parameters
        ----------
        idx : int
            Index of the device
        reboot : bool
            If true the server will reboot. (Default value = True)
        """
        self._device = idx
        if reboot:
            try:
                self.quit()
            except AttributeError:
                _LOGGER.warning(" Reboot while no active stream")
            try:
                self.boot()
            except OSError:
                raise OSError("Error: Invalid device. Server did not boot.")

    def boot(self):
        """boot Aserver = start stream, setting its callback to this callback."""
        if self.is_active:
            _LOGGER.info("Aserver already running...")
            return -1
        self.boot_time = time.time()
        self.block_time = self.boot_time
        self.block_cnt = 0
        self.stream = self.backend.open(channels=self.channels, rate=self.sr,
                                        input_flag=False, output_flag=True,
                                        frames_per_buffer=self.bs,
                                        output_device_index=self.device,
                                        stream_callback=self._play_callback)
        self._is_active = self.stream.is_active()
        _LOGGER.info("Server Booted")
        return self

    def quit(self):
        """Aserver quit server: stop stream and terminate pa"""
        if not self.is_active:
            _LOGGER.info("Stream not active")
            return -1
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                _LOGGER.info("Aserver stopped.")
        except AttributeError:
            _LOGGER.info("No stream found...")
        self.stream = None
        return 0

    def play(self, asig, onset=0, out=0, **kwargs):
        """Dispatch asigs or arrays for given onset."""
        self._stop = False

        sigid = id(asig)  # for copy check
        if asig.sr != self.sr:
            asig = asig.resample(self.sr)
        if onset < 1e6:
            rt_onset = time.time() + onset
        else:
            rt_onset = onset
        idx = np.searchsorted(self.srv_onsets, rt_onset)
        self.srv_onsets.insert(idx, rt_onset)
        if asig.sig.dtype != self.backend.dtype:
            warn("Not the same type. ")
            if id(asig) == sigid:
                asig = copy.copy(asig)
            asig.sig = asig.sig.astype(self.backend.dtype)
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
                _LOGGER.warning("blocking inactive with play(onset>0)")
            else:
                time.sleep(asig.get_duration())
        return self

    def _play_callback(self, in_data, frame_count, time_info, flag):
        """callback function, called from pastream thread when data needed."""
        tnow = self.block_time
        self.block_time += self.block_duration
        # self.block_cnt += 1  # TODO this will get very large eventually
        # just curious - not needed but for time stability check
        self.timejitter = time.time() - self.block_time
        if self.timejitter > 3 * self.block_duration:
            msg = f"Aserver late by {self.timejitter} seconds: block_time reset!"
            _LOGGER.debug(msg)
            self.block_time = time.time()
        # to shortcut computing
        if not self.srv_asigs or self.srv_onsets[0] > tnow:
            return self.backend.process_buffer(self.empty_buffer)
        elif self._stop:
            self.srv_asigs.clear()
            self.srv_onsets.clear()
            self.srv_curpos.clear()
            self.srv_outs.clear()
            return self.backend.process_buffer(self.empty_buffer)
        data = np.zeros((self.bs, self.channels), dtype=self.backend.dtype)
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
            # .reshape(n, nch) not needed as moved to play
            data[io0:io0 + n, out:out + nch] += tmpsig
            self.srv_curpos[i] += n
            if self.srv_curpos[i] >= a.samples:
                dellist.append(i)  # store for deletion
        # clean up lists
        for i in dellist[::-1]:  # traverse backwards!
            del self.srv_asigs[i]
            del self.srv_onsets[i]
            del self.srv_curpos[i]
            del self.srv_outs[i]
        return self.backend.process_buffer(data * (self.backend.range * self.gain))

    def stop(self):
        self._stop = True

    def __enter__(self):
        return self.boot()

    def __exit__(self, exc_type, exc_value, traceback):
        self.quit()
        self.backend.terminate()

    def __del__(self):
        self.quit()
        self.backend.terminate()

    def __enter__(self):
        return self.boot()
