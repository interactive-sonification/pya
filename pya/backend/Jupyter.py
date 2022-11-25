from .base import BackendBase, StreamBase

import asyncio
import threading
from functools import partial
from IPython.display import Javascript, HTML, display

try:
    import websockets
except ImportError:
    websockets = None


class JupyterBackend(BackendBase):

    dtype = 'float32'
    range = 1
    bs = 4096  # streaming introduces lack which has to be covered by the buffer

    def __init__(self, port=8765, proxy_suffix=None):
        if not websockets:
            raise Exception("JupyterBackend requires 'websockets' but it could not be imported. "
                            "Did you miss installing optional 'remote' requirements?")

        self.dummy_devices = [dict(maxInputChannels=0, maxOutputChannels=2, index=0, name="JupyterBackend")]
        self.port = port
        self.proxy_suffix = proxy_suffix
        if self.proxy_suffix is not None:
            self.bs = 1024 * 10  # probably running on binder; increase buffer size

    def get_device_count(self):
        return len(self.dummy_devices)

    def get_device_info_by_index(self, idx):
        return self.dummy_devices[idx]

    def get_default_input_device_info(self):
        return self.dummy_devices[0]

    def get_default_output_device_info(self):
        return self.dummy_devices[0]

    def open(self, *args, channels, rate, stream_callback=None, **kwargs):
        display(HTML("<div class=\"alert-info\">You are using the experimental Jupyter backend. "
                     "Note that this backend is not feature complete and does not support recording so far. "
                     "User experience may vary depending on the network latency.</div>"))
        stream = JupyterStream(channels=channels, rate=rate, stream_callback=stream_callback, port=self.port,
                               proxy_suffix=self.proxy_suffix)
        stream.start_stream()
        return stream

    def process_buffer(self, buffer):
        return buffer

    def terminate(self):
        pass


class JupyterStream(StreamBase):

    def __init__(self, channels, rate, stream_callback, port, proxy_suffix):
        self.rate = rate
        self.channels = channels
        self.stream_callback = stream_callback
        self.server = None
        self._is_active = False

        async def bridge(websocket):
            async for _ in websocket:
                buffer = self.stream_callback(None, None, None, None)
                # print(buffer)
                await websocket.send(buffer.reshape(-1, 1, order='F').tobytes())

        async def ws_runner():
            async with websockets.serve(bridge, "0.0.0.0", 8765):
                await asyncio.Future()

        def loop_thread(loop):
            # since ws_runner will block forever it will raise a runtime excepion
            # when we kill the event loop.
            try:
                loop.run_until_complete(ws_runner())
            except RuntimeError:
                pass

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=loop_thread, args=(self.loop,))
        # self.thread.daemon = True  # allow program to shutdown even if the thread is alive

        url_suffix = f':{port}' if proxy_suffix is None else proxy_suffix

        self.client = Javascript(
            f"""
var sampleRate = {self.rate};
var channels = {self.channels};
var urlSuffix = "{url_suffix}";
window.pya = {{ bufferThresh: 0.2 }}
            """
            r"""
var processedPackages = 0;
var latePackages = 0;
var badPackageRatio = 1;

function resolveProxy() {
  let reg = /\/notebooks.*ipynb/g
  let res = window.location.pathname.replace(reg, "");
  return res 
}

var protocol = (window.location.protocol == 'https:') ? 'wss://' : 'ws://'
var startTime = 0;
var context = new (window.AudioContext || window.webkitAudioContext)();

context.onstatechange = function() {
    console.log("PyaJSClient: AudioContext StateChange!")
    if (context.state == "running") {
        var ws = new WebSocket(protocol+window.location.hostname+resolveProxy()+urlSuffix);
        ws.binaryType = 'arraybuffer';
        window.ws = ws;

        ws.onopen = function() {
            console.log("PyaJSClient: Websocket connected.");
            startTime = context.currentTime;
            ws.send("G");
        };

        ws.onmessage = function (evt) {
            if (evt.data) {
                processedPackages++;
                var buf = new Float32Array(evt.data)
                var duration = buf.length / channels
                var buffer = context.createBuffer(channels, duration, sampleRate)
                for (let i = 0; i < channels; i++) {
                    updateChannel(buffer, buf.slice(i * duration, (i + 1) * duration), i)
                }
                var source = context.createBufferSource()
                source.buffer = buffer
                source.connect(context.destination)
                if (startTime > context.currentTime) {
                    source.start(startTime)
                    startTime += buffer.duration
                } else {
                    latePackages++;
                    badPackageRatio = latePackages / processedPackages
                    if (processedPackages > 50) {
                        console.log("PyaJSClient: Dropped sample ratio is " + badPackageRatio.toFixed(2))
                        if (badPackageRatio > 0.05) {
                            let tr = window.pya.bufferThresh
                            window.pya.bufferThresh = (tr > 0.01) ? tr - 0.03 : 0.01;
                            console.log("PyaJSClient: Decrease buffer delay to " + window.pya.bufferThresh.toFixed(2))
                        }
                        latePackages = 0;
                        processedPackages = 0;
                    }
                    source.start()
                    startTime = context.currentTime + buffer.duration
                }
                setTimeout(function() {ws.send("G")},
                    (startTime - context.currentTime) * 1000 * window.pya.bufferThresh)
            }
        };
    }
};

var updateChannel = function(buffer, data, channelId) {
    buffer.copyToChannel(data, channelId, 0)
}

// Fallback for browsers without copyToChannel Support
if (! AudioBuffer.prototype.copyToChannel) {
    console.log("PyaJSClient: AudioBuffer.copyToChannel not supported. Falling back...")
    updateChannel = function(buffer, data, channelId) {
        buffer.getChannelData(channelId).set(data);
    }
}

function resumeContext() {
    context.resume();
    var codeCells = document.getElementsByClassName("input_area")
    for (var i = 0; i < codeCells.length; i++) {
        codeCells[i].removeEventListener("focusin", resumeContext)
    }
}

if (context.state == "suspended") {
    console.log("PyaJSClient: AudioContext not running. Waiting for user input...")
    var codeCells = document.getElementsByClassName("input_area")
    for (var i = 0; i < codeCells.length; i++) {
        codeCells[i].addEventListener("focusin", resumeContext)
    }
}

console.log("PyaJSClient: Websocket client loaded.")
            """)

    @staticmethod
    def set_buffer_threshold(buffer_limit):
        display(Javascript(f"window.pya.bufferThresh = {1 - buffer_limit}"))

    def stop_stream(self):
        if self.thread.is_alive():
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join()

    def close(self):
        self.stop_stream()
        self.loop.close()

    def start_stream(self):
        if not self.thread.is_alive():
            self.thread.start()
        display(self.client)

    def is_active(self):
        return self.thread.is_alive()
