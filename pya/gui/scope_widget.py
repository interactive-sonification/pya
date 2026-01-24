# A Scope (signal & spectrum) for Jupyter notebooks (using %matplotlib widget and FuncAnimation)
# Currently this only displays AudioIn, eventually both input/output to be supported.
# ToDos: Multi-channel extension, Input/Output selection API / GUI

from matplotlib.animation import FuncAnimation
from ipywidgets import widgets
import matplotlib.pyplot as plt
from pya import Aserver
from pya.agen.lib import SinOsc, Line
import numpy as np
from IPython.display import display
from scipy.signal import get_window, find_peaks
from time import time

class ScopeWidget:

    def __init__(self, 
                 server=None, 
                 fps=25, 
                 mode="input", 
                 window : str = 'blackmanharris', 
                 window_size : int = 4096, 
                 figsize=(8, 3)
        ):
        """ScopeWidget - a Scope/FreqScope view for Jupyter notebooks
        for interactive contexts using %matplotlib widget

        Args:
            server (Aserver, optional): pya Aserver. Defaults to None.
            fps (int, optional): frames per second for rendering. Defaults to 20.
            mode (str, optional): "input" or "output"
            window_shape (str): type of the window function
            window_size: size of the analyzed window. Power of 2 recommended. Has to be <= server.history_size 
            figsize (tuple, optional): figure size. Defaults to (8, 3).
        """
        self.server = server
        if not self.server:
            self.server = Aserver.default

        self.sr = self.server.sr
        self.fps = fps
        self.last_frame_time = time()
        self._window = window
        self._window_size = window_size
        self.init_spec()

        self.spec_decay_seconds = 2.5 # seconds it takes for a bin to get from ylim[1] to ylim[0]
        self.spec_lin_xlim = (0,  self.sr//2 + 1)
        self.spec_log_xlim = (20, self.sr//2 + 1)
        self.spec_lin_ylim = (0, 1.118)
        self.spec_log_ylim = (1e-4, 3)
        self.spec_peak_prominance = 1
        self.spec_selected_idx = None
        self.last_mouse_ax_pos = None
        self.spec_hovered = False
        self.mouse_ax_pos = np.array([0, 0])
        self.mouse_spec_coord = np.array([0, 0])

        self.output = widgets.Output()
        with self.output:
            # it is assumed that %matplotlib widget is used
            self.fig = plt.figure(figsize=figsize)

            # signal plot on the left
            self.axsig = plt.subplot(1, 2, 1)
            (self.line2Dsig,) = self.axsig.plot(
                [0, self._window_size / self.sr], [0, 0], "b-"
            )
            plt.xlabel("time")
            plt.ylabel("signal [arb. units]")
            plt.grid()

            # spectrum plot on the right
            self.axspec = plt.subplot(1, 2, 2)
            (self.line2Dspec,) = self.axspec.plot(np.array(self.spec_log_xlim), np.array(self.spec_log_ylim), "b-")
            (self.line2Dspec_p,) = self.axspec.plot([0], [0], "bo", mec='white', mew=1.5, ms=7)
            self.line2Dspec_p.set_visible(False)
            self.peak_label = self.axspec.text(0, 0, "", transform=self.axspec.transAxes, 
                ha="center", va="bottom", #fontsize=12,
                bbox=dict(boxstyle="round", fc="w", ec="0.5"))
            self.peak_label.set_visible(False)
            self.axspec.scatter(np.array(0), np.array(0))
            plt.xlabel("frequency")
            plt.ylabel("E(w) [arb. units]")
            plt.grid()
            plt.tight_layout()
            self.axspec.set_xscale("log")
            self.axspec.set_yscale("log")
            self.axspec.figure.canvas.mpl_connect('axes_enter_event', self.on_mouse_enter)
            self.axspec.figure.canvas.mpl_connect('axes_leave_event', self.on_mouse_leave)
            self.axspec.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_hover)
            self.axspec.figure.canvas.mpl_connect('button_press_event', self.on_mouse_click)
            plt.show()

        @staticmethod
        def update(frame):
            sr = self.server.sr
            dt = time() - self.last_frame_time
            self.last_frame_time = time()

            if self.mode == "input":
                sig = self.server.input_history.unwrapped_copy(self._window_size)
            else:
                sig = self.server.output_history.unwrapped_copy(self._window_size)
            block_size = sig.shape[0]
            self.line2Dsig.set_data(
                np.linspace(0, self._window_size / sr, self._window_size, endpoint=False), sig[:, 0]
            )
            spec = np.fft.rfft(self.norm_window * sig[:, 0], axis=0)
            spec[1:-1] *= 2
            spec_decay_seconds = self.spec_decay_seconds if not self.spec_hovered else 3 * self.spec_decay_seconds + 3
            if self.axspec.get_yscale() == 'log':
                self.last_spec /= (self.spec_log_ylim[1] / self.spec_log_ylim[0]) ** (dt / spec_decay_seconds)
            else:
                self.last_spec -= dt / spec_decay_seconds * (self.spec_lin_ylim[1] - self.spec_lin_ylim[0])
            self.last_spec = np.maximum(np.abs(spec), self.last_spec)
            self.line2Dspec.set_data(
                self.spec_freqs, self.last_spec
            )

            # Find closest peak to mouse and display it
            if self.spec_hovered:
                mouse_moved = np.all(self.mouse_ax_pos != self.last_mouse_ax_pos)
                if mouse_moved:
                    peaks_idx = find_peaks(np.log(self.last_spec), prominence=self.spec_peak_prominance)[0]
                    peaks_coord = np.array([self.spec_freqs[peaks_idx], self.last_spec[peaks_idx]])
                    peaks_px = self.axspec.transData.transform(np.column_stack(peaks_coord))
                    peaks_ax_pos = self.axspec.transAxes.inverted().transform(peaks_px)

                    mouse_distances = np.linalg.norm(peaks_ax_pos - self.mouse_ax_pos, axis=1)
                    nearest_peak_idx = np.argmin(mouse_distances)

                    if mouse_distances[nearest_peak_idx] < 0.08:
                        self.spec_selected_idx = peaks_idx[nearest_peak_idx]
                    else:
                        self.spec_selected_idx = round(self.mouse_spec_coord[0] * (len(self.last_spec)-1) * 2/self.sr)
                    self.last_mouse_ax_pos = self.mouse_ax_pos

                if self.spec_selected_idx:
                    idx = self.spec_selected_idx
                    freq = self.spec_freqs[idx]
                    mag = self.last_spec[idx]
                    self.line2Dspec_p.set_data([freq], [mag])
                    self.line2Dspec_p.set_visible(True)

                    if not self.peak_label.get_visible() or mouse_moved:
                        selected_px = self.axspec.transData.transform(np.column_stack([freq, mag]))
                        selected_ax_pos = self.axspec.transAxes.inverted().transform(selected_px)
                        #self.peak_label.set_position(selected_ax_pos[0] + [0, 0.1])
                        self.peak_label.set_position(self.mouse_ax_pos + [0, 0.1])
                        self.peak_label.set_visible(True)
                    self.peak_label.set_text(f"{freq:{".1f" if freq < 100 else ".0f"}} Hz\n{mag:.1e}")
                else:
                    self.line2Dspec_p.set_visible(False)
                    self.peak_label.set_visible(False)
            else:
                if self.spec_selected_idx:
                    self.spec_selected_idx = None
                    self.line2Dspec_p.set_visible(False)
                    self.peak_label.set_visible(False)



        self.ani = FuncAnimation(
            self.fig,
            update,
            init_func=self.init_plot,
            blit=True,
            interval=int(1000 / self.fps),
            save_count=10,
        )
        # plt.show()

        def quit(event):
            self.ani.event_source.stop()
            plt.close(self.fig)
            self.__del__()

        layout = widgets.Layout(width="70px")

        self.btn_quit = widgets.Button(description="Quit", layout=layout)
        self.btn_quit.on_click(quit)

        def pause(event):
            self.ani.pause()

        self.btn_pause = widgets.Button(description="Pause", layout=layout)
        self.btn_pause.on_click(pause)

        def resume(event):
            self.ani.resume()

        self.btn_resume = widgets.Button(description="Resume", layout=layout)
        self.btn_resume.on_click(resume)

        self.fps_wdg = widgets.IntSlider(
            description="fps",
            value=self.fps,
            min=1,
            max=50,
            step=1,
        )

        def on_fps_value_change(change):
            self.fps = change["new"]

        self.fps_wdg.observe(on_fps_value_change, names="value")

        self.window_size_wdg = widgets.FloatLogSlider(
            description="Window Size",
            value=self._window_size,
            base=2, 
            min=8, 
            max=np.log2(self.server.history_size), 
            step=1, 
            readout_format=".0f"
        )

        def on_window_size_value_change(change):
            self.window_size = change["new"]

        self.window_size_wdg.observe(on_window_size_value_change, names="value")

        self.spec_decay_wdg = widgets.FloatLogSlider(
            description="Spec Decay [s]",
            value=self.spec_decay_seconds,
            base=10, 
            min=np.log10(0.05), 
            max=np.log10(20),
            step=0
        )

        def on_spec_decay_change(change):
            self.spec_decay_seconds = change["new"]

        self.spec_decay_wdg.observe(on_spec_decay_change, names="value")

        self.mode_wdg = widgets.Dropdown(
            options=["input", "output"],
            description="Source:",
            # intent=False,
            layout=widgets.Layout(width="160px"),
        )

        self.mode = mode  # at the moment a string ("input" or "output")

        def on_mode_change(change):
            self.mode = change["new"]

        self.mode_wdg.observe(on_mode_change, names="value")

        self.scope_widgets = widgets.VBox(
            [
                self.output,
                widgets.HBox(
                    [
                        self.fps_wdg,
                        self.window_size_wdg,
                        self.spec_decay_wdg,
                        self.mode_wdg,
                        self.btn_quit,
                        self.btn_pause,
                        self.btn_resume,
                    ],
                    layout=widgets.Layout(
                        display="flex",
                        flex_flow="row wrap",
                        align_items="stretch",
                        width="100%",
                    ),
                ),
            ]
        )
        self.show()  # display plot and widgets

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
                
                if ax == self.axspec:
                    new_xlim = [self.spec_lin_xlim, self.spec_log_xlim][index % 2] 
                    new_ylim = [self.spec_lin_ylim, self.spec_log_ylim][index // 2]
                    ax.set_xlim(new_xlim)
                    ax.set_ylim(new_ylim)
                else:
                    ax.relim()
                    ax.autoscale()

                ax.xaxis.set_major_formatter(plt.ScalarFormatter())

            if event.key == "a":
                ax = event.inaxes
                if ax is None:
                    return
                ax.relim()  # get data limits
                ax.autoscale()  # and apply these

            if event.key == "r":
                ax = event.inaxes
                if ax == self.axsig:
                    ax.set_xlim(0, self.server.bs / self.server.sr)
                    ax.set_yscale("linear")
                    ax.set_xscale("linear")
                    ax.set_ylim(-1.2, 1.2)

                elif ax == self.axspec:
                    ax.set_xlim(1, self.server.sr // 2)
                    ax.set_ylim(1e-5, 3.1)
                    ax.set_yscale("log")
                    ax.set_xscale("log")

        self.cid = self.fig.canvas.mpl_connect("key_press_event", on_key)

    def on_mouse_enter(self, event):
        if event.inaxes == self.axspec:
            self.spec_hovered = True

    def on_mouse_leave(self, event):
        if event.inaxes == self.axspec:
            self.spec_hovered = False

    def on_mouse_hover(self, event):
        if event.inaxes == self.axspec:
            self.mouse_spec_coord = np.array([event.xdata, event.ydata])
            self.mouse_ax_pos = self.axspec.transAxes.inverted().transform((event.x, event.y))

    def on_mouse_click(self, event):
        if event.inaxes == self.axspec:
            if event.button == 1: # left click
                (SinOsc(self.spec_freqs[self.spec_selected_idx]) * Line(0.1, 0, 0.2)).dup(2).play()

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value in ["input", "output"]:
            self._mode = value
            self.mode_wdg.value = value
        else:
            raise ValueError("Mode must be either 'input' or 'output'.")

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, value):
        self._fps = value
        try:  # works only after initialization of FuncAnimation
            self.ani._interval = int(1000 / value)
            self.fps_wdg.value = value
        except BaseException:
            pass

    def init_spec(self):
        self.temp_window = get_window(self._window, self._window_size)
        self.norm_window = self.temp_window / np.sum(self.temp_window)
        self.last_spec = np.ones(self._window_size//2 + 1) * 1e-12
        self.spec_freqs = np.linspace(0, self.sr // 2, self._window_size//2+1)

    @property
    def window(self):
        return self._window
    
    @window.setter
    def window(self, value: str):
        self._window = value
        self.init_spec()

    @property
    def window_size(self):
        return self._window_size
    
    @window_size.setter
    def window_size(self, value: str):
        self._window_size = int(value)
        self.init_spec()

    def __del__(self):
        del self.ani

    def show(self):
        """display the scope UI including plot and widgets"""
        display(self.scope_widgets)

    def init_plot(self):
        self.axsig.set_xlim(0, self.server.bs / self.sr)
        self.axsig.set_ylim(-1.2, 1.2)
        self.axspec.set_xlim(self.spec_log_xlim)
        self.axspec.set_ylim(self.spec_log_ylim)
        self.axspec.xaxis.set_major_formatter(plt.ScalarFormatter())
