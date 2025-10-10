from multiprocessing import Pipe, shared_memory
import subprocess
import numpy as np


class Scope:

    def __init__(
        self,
        num_samples=256,
        num_channels=2,
        pos=(-320, -240),
        size=(320, 240),
        rate=24,
        mode="signal",
    ):
        """Scope is the GUI interface class to spawn and use ScopeGUI.
        ScopeGUI is a self-contained Qt app using pyqtgraph defined
        in this same file below.

        Args:
            num_samples (int, optional): _description_. Defaults to 256.
            num_channels (int, optional): _description_. Defaults to 1.
            pos_x (int, optional): _description_. Defaults to -320.
            pos_y (int, optional): _description_. Defaults to -240.
            width (int, optional): _description_. Defaults to 320.
            height (int, optional): _description_. Defaults to 240.
            mode (str, optional): "signal" or "spectrum"
        """
        from pathlib import Path
        import os

        scope_script = Path(__file__).resolve()
        self.running = False
        print("the scopecript is", scope_script)
        self.mode = mode

        self.parent_conn, self.child_conn = Pipe(duplex=True)
        self.fd = self.child_conn.fileno()
        env = os.environ.copy()
        env["QT_LOGGING_RULES"] = "*.debug=false;*.info=false"  # get rid of Qt logging
        self.proc = subprocess.Popen(
            [
                "python",
                scope_script,
                str(self.fd),
                str(pos[0]),
                str(pos[1]),
                str(size[0]),
                str(size[1]),
                self.mode,
            ],
            pass_fds=(self.fd,),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.set_points(num_samples, num_channels)
        self.render_framerate(rate)

    def cmd(self, cmd, **kwargs):
        argdict = dict(cmd=cmd, **kwargs)
        self.parent_conn.send(argdict)
        if self.parent_conn.poll(timeout=1):
            return self.parent_conn.recv()
        else:
            return -1

    def set_points(self, num_samples: int = 128, num_channels: int = 2):
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.shm_name = self.cmd(
            "set_points", num_samples=self.num_samples, num_channels=num_channels
        )
        self.shm = shared_memory.SharedMemory(name=str(self.shm_name))
        self.data = np.ndarray(
            (self.num_samples, self.num_channels), dtype=np.float32, buffer=self.shm.buf
        )

    def render_framerate(self, rate=30):
        self.rate = rate
        return self.cmd("render_framerate", rate=rate)

    def move(self, x, y):
        return self.cmd("move", x=x, y=y)

    def resize(self, width=400, height=300):
        return self.cmd("resize", width=width, heigth=height)

    def start(self):
        ret = self.cmd("start")
        if ret >= 0:
            self.running = True
        return ret

    def stop(self):
        ret = self.cmd("stop")
        if ret >= 0:
            self.running = False
        return ret

    def exit(self):
        if self.running:
            self.stop()
        return self.cmd("exit")

    def set_data(self, x):
        if x.shape != (self.num_samples, self.num_channels):
            print("Scope.set_data: shape mismatch")
        else:
            try:
                self.data[:] = x
            except ValueError:
                print("Scope.set_data: probably wrong argument shape")

    def set_mode(self, mode="signal"):
        self.mode = mode
        ret = self.cmd("mode", mode=mode)
        return ret

    def __del__(self):
        self.exit()
        if self.shm:
            self.shm.close()
            print("Scope: local shm closed\n")


# ScopeGUI is only used for the spawned in the Qt app for GUI rendering
# don't import this in the caller app


class _ScopeGUI:
    def __init__(self, conn, pos_x=0, pos_y=0, width=320, height=240, mode="signal"):
        """realtime view for Aserver output

        Args:
            conn (_type_): file handle to the subprocess
            pos_x (int, optional): window origin x (upper left) in pixels. Defaults to 0.
            pos_y (int, optional): window origin y (upper left) in pixels_. Defaults to 0.
            width (int, optional): window widths in pixels. Defaults to 320.
            height (int, optional): window height in pixels. Defaults to 240.
            mode (str, optional): mode, either "signal" or "spectrum".
        """
        self.conn = conn
        self.shm = None
        self.num_samples = 256
        self.num_channels = 2
        self.data_dtype = np.float32
        self.dtype_size = np.dtype(self.data_dtype).itemsize
        self.mode = mode

        self.shm_name = None
        self.running = False  # whether update_plot() should do anything
        self.update_rate = 30  # Hz

        # create qt app with pyqtgraph window
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(title="Shared Memory Plot")
        self.plot = self.win.addPlot()
        self.plot.enableAutoRange(axis="y", enable=False)
        self.plot.setYRange(-0.5, self.num_channels - 0.5)

        self.curves = [
            self.plot.plot(np.ones(self.num_samples) + k - 1, pen="y")
            for k in range(self.num_channels)
        ]

        self.win.resize(width, height)
        screen = self.app.primaryScreen()
        rect = screen.availableGeometry()
        if pos_x < 0:
            pos_x = rect.width() - self.win.frameGeometry().width()
        if pos_y < 0:
            pos_y = rect.height() - self.win.frameGeometry().height()
        self.win.move(pos_x, pos_y)
        self.win.show()
        self.win.raise_()

        # create timer to update the plot
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        self.update_timer.start(1000 // self.update_rate)
        atexit.register(self.cleanup)  # <- Ensures cleanup on hard exit

    def setup_shared_memory(self):
        if self.shm:
            self.shm.close()
            self.shm.unlink()
        self.shm = shared_memory.SharedMemory(
            create=True, size=self.num_samples * self.num_channels * self.dtype_size
        )
        self.shm_name = self.shm.name
        self.conn.send(self.shm_name)  # Tell the caller the name

    def update_plot(self):
        if self.shm and self.running:
            try:
                # get data from shared memory
                data = np.ndarray(
                    (self.num_samples, self.num_channels),
                    dtype=self.data_dtype,
                    buffer=self.shm.buf,
                )
                # plot data into view
                if self.mode == "signal":
                    for k in range(self.num_channels):
                        self.curves[k].setData(data[:, k] + k)
                elif self.mode == "spectrum":
                    spectrum = np.abs(np.fft.fft(data.transpose()))
                    nfreqs = data.shape[0] // 2
                    for k in range(self.num_channels):
                        self.curves[k].setData(spectrum[k, :nfreqs] / nfreqs + k)
            except Exception as e:
                print(f"Plot update failed: {e}")

    def handle_commands(self):
        try:
            while self.conn.poll():
                msg = self.conn.recv()
                if isinstance(msg, dict):
                    cmd = msg.get("cmd")
                    if cmd == "set_points":
                        self.num_samples = int(msg.get("num_samples", 256))
                        self.num_channels = int(msg.get("num_channels", 2))
                        self.setup_shared_memory()
                    elif cmd == "start":
                        self.running = True
                        self.conn.send(0)  # return value
                    elif cmd == "stop":
                        self.running = False
                        self.conn.send(0)  # return value
                    elif cmd == "move":
                        pos_x = int(msg.get("x", 0))
                        pos_y = int(msg.get("y", 0))
                        screen = self.app.primaryScreen()
                        rect = screen.availableGeometry()
                        if pos_x < 0:
                            pos_x = rect.width() - self.win.frameGeometry().width()
                        if pos_y < 0:
                            pos_y = rect.height() - self.win.frameGeometry().height()
                        self.win.move(pos_x, pos_y)
                        self.conn.send(0)  # return value
                    elif cmd == "mode":
                        self.mode = str(msg.get("mode", "signal"))
                        self.conn.send(0)
                    elif cmd == "raise":
                        self.win.raise_()
                        self.conn.send(0)  # return value
                    elif cmd == "resize":
                        width = int(msg.get("width", 320))
                        height = int(msg.get("height", 240))
                        self.win.resize(width, height)
                        self.conn.send(0)  # return value
                    elif cmd == "render_framerate":
                        rate = int(msg.get("rate", 30))
                        self.update_timer.setInterval(1000 // rate)
                        self.conn.send(1000 // rate)  # return value
                    elif cmd == "exit":
                        self.conn.send(0)  # return value
                        self.cleanup()
                        QtWidgets.QApplication.quit()
        except OSError as e:
            print("OS Error: handle obviously closed!", e)

    def cleanup(self):
        print("scopeGUI: cleanup")
        if self.shm:
            self.shm.close()
            self.shm.unlink()
            print("scopeGUI: shm closed and unlinked")
            self.shm = None
        self.conn.close()

    def run(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.handle_commands)
        timer.start(100)

        self.app.exec()


# Entry point when run as script
if __name__ == "__main__":
    import sys
    import numpy as np
    import pyqtgraph as pg
    from multiprocessing import shared_memory
    from multiprocessing.connection import Connection
    from pyqtgraph.Qt import QtWidgets, QtCore
    import atexit
    import platform

    fd, pos_x, pos_y, width, height = [int(x) for x in sys.argv[1:-1]]
    mode = sys.argv[-1]
    conn = Connection(fd)
    scope = _ScopeGUI(conn, pos_x, pos_y, width, height, mode)

    # rename process on non-window OS
    if platform.system() != "Windows":
        try:
            from setproctitle import setproctitle

            setproctitle("ScopeGUI")
        except ImportError:
            print("Optional: setproctitle not installed, skipping process renaming.")
    else:
        print("Process renaming not supported on Windows.")

    scope.run()
