# A Scope (signal & spectrum) for Jupyter notebooks (using %matplotlib widget and FuncAnimation)
# Currently this only displays AudioIn, eventually both input/output to be supported.
# ToDos: Multi-channel extension, Input/Output selection API / GUI

from matplotlib.animation import FuncAnimation
from ipywidgets import widgets
import matplotlib.pyplot as plt
from pya import Aserver
import numpy as np
from IPython.display import display


def get_audio_input_data(server=None):
    """get aserver input data as numpy array.

    Args:
        server (Aserver | None): the Aserver. Defaults to None.
        on None, Aserver.default will be used

    Returns:
        np.ndarray: the input data of shape (server.bs,  server.channels).

    ToDo: move function to Aserver.
    """
    if not server:
        server = Aserver.default
    num_channels = server.channels
    samples = np.frombuffer(server.latest_input, dtype=server.backend.dtype)
    samples = samples.reshape(-1, num_channels)
    return samples, server.sr


class ScopeWidget:

    def __init__(self, server=None, fps=5, figsize=(8, 3)):
        """ScopeWidget - a Scope/FreqScope view for Jupyter notebooks
        for interactive contexts using %matplotlib widget

        Args:
            server (Aserver, optional): pya Aserver. Defaults to None.
            fps (int, optional): frames per second for rendering. Defaults to 5.
            figsize (tuple, optional): figure size. Defaults to (8, 3).
        """
        self.server = server
        if not self.server:
            self.server = Aserver.default

        self.sr = self.server.sr
        self.fps = fps

        # it is assumed that %matplotlib widget is used
        self.fig = plt.figure(figsize=figsize)

        # signal plot on the left
        self.axsig = plt.subplot(1, 2, 1)
        (self.line2Dsig,) = self.axsig.plot([], [], "b-")
        plt.xlabel("time")
        plt.ylabel("signal [arb. units]")
        plt.grid()

        # spectrum plot on the right
        self.axspec = plt.subplot(1, 2, 2)
        (self.line2Dspec,) = self.axspec.plot([1, 22050], [1e2, 1e-5], "b-")
        plt.xlabel("frequency")
        plt.ylabel("E(w) [arb. units]")
        plt.grid()
        plt.tight_layout()
        self.axspec.set_xscale("log")
        self.axspec.set_yscale("log")

        @staticmethod
        def update(frame):
            sig, sr = get_audio_input_data(self.server)
            self.line2Dsig.set_data(np.linspace(0, 1.0, sig.shape[0]), sig[:, 0])
            spec = np.fft.rfft(sig[:, 0], axis=0)
            self.line2Dspec.set_data(np.linspace(0, 22050, spec.shape[0]), np.abs(spec))

        self.ani = FuncAnimation(
            self.fig,
            update,
            init_func=self.init_plot,
            blit=True,
            interval=int(1000 / self.fps),
            save_count=10,
        )
        plt.show()

        def quit(event):
            self.ani.event_source.stop()
            self.__del__()

        self.btn_quit = widgets.Button(description="Shutdown")
        self.btn_quit.on_click(quit)

        def pause(event):
            self.ani.pause()

        self.btn_pause = widgets.Button(description="Pause")
        self.btn_pause.on_click(pause)

        def resume(event):
            self.ani.resume()

        self.btn_resume = widgets.Button(description="Resume")
        self.btn_resume.on_click(resume)

        # self.wdg = interactive(self.set_fps, fps=(1, 50, 1))
        self.fps_wdg = widgets.IntSlider(
            description="fps",
            value=self.fps,
            min=1,
            max=50,
            step=1,
        )

        def on_fps_value_change(change):
            self.set_fps(change["new"])

        self.fps_wdg.observe(on_fps_value_change, names="value")

        display(
            widgets.HBox(
                [self.fps_wdg, self.btn_quit, self.btn_pause, self.btn_resume],
                layout=widgets.Layout(
                    display="flex",
                    flex_flow="row wrap",
                    align_items="stretch",
                    width="100%",
                ),
            )
        )

        # some interaction features / keybindings
        def on_key(event):
            if event.key == "l":
                ax = event.inaxes
                if ax is None:
                    return

                # get current xscale and yscale
                xscale = ax.get_xscale()
                yscale = ax.get_yscale()

                # advance to next state
                index = (xscale == "log") + 2 * (yscale == "log")
                index = (index + 1) % 4

                # set new scales - via bits of index
                new_xscale = ["linear", "log"][index % 2]  # LSB for x
                new_yscale = ["linear", "log"][index // 2]  # MSB for y
                ax.set_xscale(new_xscale)
                ax.set_yscale(new_yscale)
                ax.relim()
                ax.autoscale()

            if event.key == "a":
                ax = event.inaxes
                if ax is None:
                    return
                ax.relim()  # get data limits
                ax.autoscale()  # and apply these

        self.cid = self.fig.canvas.mpl_connect("key_press_event", on_key)

    def set_fps(self, fps=5):
        self.fps = fps
        self.ani._interval = int(1000 / fps)

    def __del__(self):
        del self.ani

    def init_plot(self):
        self.axsig.set_xlim(0, 1)
        self.axsig.set_ylim(-1.2, 1.2)
        self.axspec.set_xlim(10, self.sr)
        self.axspec.set_ylim(0.0001, 1)
