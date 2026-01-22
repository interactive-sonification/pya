from pya.asig import Asig
from .helper.backend import determine_backend
from .helper.helpers import RingBuffer
import copy
import logging
import time
from typing import Optional, Union
from warnings import warn

import numpy as np


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
    >>> ser.quit()  # Important to call quit() to close the stream when you are done. Or use context manager.
    """

    default = None  # that's the default Aserver if Asigs play via it

    @staticmethod
    def startup_default_server(input_flag=False, **kwargs):
        if Aserver.default is None:
            _LOGGER.info("Aserver startup_default_server: create and boot")
            Aserver.default = Aserver(**kwargs)  # using all default settings
            Aserver.default.input_flag = input_flag
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

    def __init__(self, sr: int = 44100, bs: Optional[int] = None,
                 device: Optional[int] = None, channels: Optional[int] = None, 
                 history_size: Optional[int] = 2**14,
                 backend=None, **kwargs):
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
            number of channel, default is the max output channels of the device
        history_size : int
            The size of the stored input and output history.
        kwargs : backend parameter

        Returns
        -------
        _ : numpy.ndarray
            numpy array of the recorded audio signal.
        """
        # TODO check if channels is overwritten by the device.
        self.sr = sr
        self.stream = None
        self.backend = determine_backend(**kwargs) if backend is None else backend
        self.bs = bs or self.backend.bs
        # Get audio devices to input_device and output_device
        self.input_devices = []
        self.output_devices = []
        for i in range(self.backend.get_device_count()):
            if int(self.backend.get_device_info_by_index(i)['maxInputChannels']) > 0:
                self.input_devices.append(self.backend.get_device_info_by_index(i))
            if int(self.backend.get_device_info_by_index(i)['maxOutputChannels']) > 0:
                self.output_devices.append(self.backend.get_device_info_by_index(i))

        self._device = self.backend.get_default_output_device_info()['index'] if device is None else device
        self._channels = channels or self.max_out_chn

        self.gain = 1.0
        self.srv_onsets = []
        self.srv_curpos = []  # start of next frame to deliver
        self.srv_asigs = []  # array of asigs or agens
        self.srv_outs = []  # output channel offset for that asig
        self.boot_time = 0  # time.time() when stream starts
        self.block_cnt = 0  # nr. of callback invocations
        self.block_duration = self.bs / self.sr  # nominal time increment per callback
        self.block_time = 0  # estimated time stamp for current block
        self._stop = True
        self.empty_buffer = np.zeros((self.bs, self.channels), dtype=self.backend.dtype)
        self._is_active = False

        assert(history_size >= bs)
        self.history_size = history_size
        self.output_history = RingBuffer((self.history_size, self._channels))
        self.input_history  = RingBuffer((self.history_size, self._channels))

        # TH: added for scope test
        self.scope = None 

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, val: int):
        """
        Set the number of channels. Aserver needs reboot.
        """
        if val > self.max_out_chn:
            raise ValueError(f"AServer: channels {val} > max {self.max_out_chn}")
        self._channels = val

    @property
    def device_dict(self):
        return self.backend.get_device_info_by_index(self._device)

    @property
    def max_out_chn(self) -> int:
        return int(self.device_dict['maxOutputChannels'])

    @property
    def max_in_chn(self) -> int:
        return int(self.device_dict['maxInputChannels'])

    @property
    def is_active(self) -> bool:
        return self.stream is not None and self.stream.is_active()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, val):
        self._device = val if val is not None else self.backend.get_default_output_device_info()['index']
        if self.max_out_chn < self.channels:
            warn(f"Aserver: warning: {self.channels}>{self.max_out_chn} channels requested - truncated.")
            self.channels = self.max_out_chn

    def __repr__(self):
        msg = f"""AServer: sr: {self.sr}, blocksize: {self.bs},
         Stream Active: {self.is_active}, Device: {self.device_dict['name']}, Index: {self.device_dict['index']}"""
        return msg

    def get_devices(self, verbose: bool = False):
        """Return (and optionally print) available input and output device"""
        if verbose:
            print("Input Devices: ")
            [print(f"Index: {i['index']}, Name: {i['name']},  Channels: {i['maxInputChannels']}")
             for i in self.input_devices]
            print("Output Devices: ")
            [print(f"Index: {i['index']}, Name: {i['name']}, Channels: {i['maxOutputChannels']}")
             for i in self.output_devices]
        return self.input_devices, self.output_devices

    def set_device(self, idx: int, reboot: bool = True):
        """Set audio device, an alternative way is to direct set the device property, i.e. Aserver.device = 1, 
        but that will not reboot the server.

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
        # for now to enable AudioIn use Aserver.default.input_flag = True
        input_flag = getattr(self, "input_flag", False)
        self.stream = self.backend.open(
            channels=self.channels,
            rate=self.sr,
            input_flag=input_flag,
            output_flag=True,
            frames_per_buffer=self.bs,
            output_device_index=self.device,
            stream_callback=self._play_callback,
        )
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

    def play(self, asig, onset: Union[int, float] = 0, out: int = 0, **kwargs):
        """Dispatch asigs or arrays for given onset.

        asig: pya.Asig
            An Asig object
        onset: int or float
            Time when the sound should play, 0 means asap
        out: int
            Output channel
        """
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

    def play_agen(self, agen, onset: Union[int, float] = 0, out: int = 0, **kwargs):
        """Dispatch agen for given onset.

        agen: pya.agen.AGen
            An AGen object
        onset: int or float
            Time when the sound should play, 0 means asap
        out: int
            Output channel
        """
        self._stop = False

        # as of now assume Agen sr == Aserver.sr

        if onset < 1e6:
            rt_onset = time.time() + onset
        else:
            rt_onset = onset

        idx = np.searchsorted(self.srv_onsets, rt_onset)
        self.srv_onsets.insert(idx, rt_onset)
        self.srv_asigs.insert(idx, agen) # ToDo refactor later: srv_asigs -> srv_aobjs (asigs or agens)
        self.srv_curpos.insert(idx, 0)
        self.srv_outs.insert(idx, out)
        if 'block' in kwargs and kwargs['block']:
            if onset > 0:  # here really onset and not rt_onset!
                _LOGGER.warning("blocking inactive with play(onset>0)")
            else:
                print("play_agen(): implement sleep until AGen end...")
                # time.sleep(asig.get_duration())
        return self

    def scope_gui(self, pos=(-400, 0), size=(400, 300), rate=25, mode="signal"):
        """Create and activate oscilloscope using pyagui Scope"""
        try:
            from pya.gui import Scope

            self.scope = Scope(self.bs, self._channels, pos=pos, size=size, rate=rate)
            self.scope.start()
            print("Aserver: Scope opened and started")
        except BaseException as e:
            _LOGGER.warning(
                "Scope is an optional feature. Requires additional package pyagui"
            )
            print(e)
            self.scope = None

    def _play_callback(self, in_data, frame_count, time_info, flag):
        """callback function, called from pastream thread when data needed."""
        # TODO input handling
        #in_samples = np.frombuffer(in_data, dtype=self.backend.dtype)
        #in_samples = in_samples.reshape(-1, self.channels)
        #self.input_history.insert(in_samples)

        tnow = self.block_time
        self.block_time += self.block_duration
        # self.block_cnt += 1  # TODO this will get very large eventually
        self.block_cnt = (self.block_cnt + 1) % 1000  # to enable check for updates
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
            a = self.srv_asigs[i]  # ATTENTION: a can be asig or agen
            c = self.srv_curpos[i]
            if t > tnow:  # first block: apply precise zero padding
                io0 = int((t - tnow) * self.sr)
            else:
                io0 = 0
            # here we need to take different action for asigs and agens
            if isinstance(a, Asig):
                tmpsig = a.sig[c:c + self.bs - io0]
            else: # can only be AGen
                # tmpsig = a.generate(self.bs-io0, c, 0) # take care for multichannel later
                tmpsig = [
                    a.generate(self.bs-io0, c, k)
                    for k in range(a.channels)
                ]
                min_len = min((s.shape[0] for s in tmpsig))
                tmpsig = np.stack([s[:min_len] for s in tmpsig], axis=1)

                # ToDo: more flexible channel handling
                if len(tmpsig.shape) == 1: # ToDo: dirty hack, improve later
                    tmpsig = np.expand_dims(tmpsig, axis=1)
            n, nch = tmpsig.shape
            out = self.srv_outs[i]
            # .reshape(n, nch) not needed as moved to play
            data[io0:io0 + n, out:out + nch] += tmpsig
            self.srv_curpos[i] += n
            # different delete conditions for AGen and Asig
            if isinstance(a, Asig):
                if self.srv_curpos[i] >= a.samples:
                    dellist.append(i)  # store for deletion
            else: # then it must be an AGen
                if n != self.bs - io0:
                    dellist.append(i)  # store for deletion
        # clean up lists
        for i in dellist[::-1]:  # traverse backwards!
            del self.srv_asigs[i]  # Asig or AGen
            del self.srv_onsets[i]
            del self.srv_curpos[i]
            del self.srv_outs[i]

        # data maintenance for scope, ScopeWidget and other services
        self.output_history.insert(data)

        if self.scope and self.scope.running:
            self.scope.set_data(data)

        return self.backend.process_buffer(data * (self.backend.range * self.gain))

    def stop(self):
        self._stop = True

    def __enter__(self):
        return self.boot()

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit"""
        _LOGGER.info("Exiting context manager. Cleaning up stream and backend")
        self.quit()
        self.backend.terminate()

    def __del__(self):
        """Backup cleanup, only if context manager wasn't used"""
        if hasattr(self, 'stream') and self.stream is not None:
            try:
                self.quit()
                self.backend.terminate()
            except:
                pass  # Ignore cleanup errors during shutdown

        # TH: added for scope test
        if self.scope:
            del(self.scope)
