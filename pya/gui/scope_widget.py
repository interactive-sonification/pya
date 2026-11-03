# A Scope (signal & spectrum) for Jupyter notebooks (using %matplotlib widget)
# Currently this only displays the Aserver output.
# ToDos: Multi-channel extension, input support

import io
from time import time

import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image, display
from matplotlib.backends.backend_webagg_core import FigureCanvasWebAggCore
from matplotlib.patches import Polygon
from matplotlib.ticker import EngFormatter, Formatter, FuncFormatter
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import correlate, find_peaks, get_window

from pya import Aserver
from pya.agen.lib import Line, SinOsc

MODES = ("output",)
CONTROLS = ("fps", "window_size", "spec_decay", "lock_period", "quit", "pause_resume")
DEFAULT_CONTROLS = ("window_size", "spec_decay", "lock_period", "quit")
WINDOW_SIZES = tuple(2 ** k for k in range(8, 17))
SPEC_DECAY_STEPS = (0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10)
UNIT_SEP = " "
HOLD_COLOR = "0.65"


def _check_mode(mode):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r} (input is not supported yet)")


def _check_controls(names):
    unknown = [name for name in names if name not in CONTROLS]
    if unknown:
        raise ValueError(f"unknown widgets {unknown}, available are {CONTROLS}")


def _close_widget_tree(widget):
    for child in getattr(widget, "children", ()):
        _close_widget_tree(child)
    for part in (getattr(widget, "layout", None), getattr(widget, "style", None)):
        if part is not None:
            part.close()
    widget.close()


def _steps_with(steps, value):
    return tuple(sorted({*steps, value})) if value is not None else steps


def _with_unit(value, unit, fmt="g"):
    return Formatter.fix_minus(f"{value:{fmt}}{UNIT_SEP}{unit}")


def _to_db(amplitude):
    return 20 * np.log10(amplitude)


def _inside_offscreen_run(y, low, high):
    """Marks the points whose neighbours lie on the same side outside of [low, high].
    All lines from and to them are invisible, so removing them does not change the plot."""
    hidden = np.zeros(len(y), bool)
    for outside in (y < low, y > high):
        hidden[1:-1] |= outside[:-2] & outside[1:-1] & outside[2:]
    return hidden


def _parabolic_peak(spec, idx):
    """Returns (fractional bin, amplitude) of the vertex of the parabola through the log amplitudes
    of bin idx and its neighbours, or the bin itself if it is no local maximum. Fitting the log
    amplitudes is much more accurate than the linear ones, since the main lobe of a windowed
    spectrum is nearly a parabola in dB."""
    if not 0 < idx < len(spec) - 1 or spec[idx] < max(spec[idx - 1], spec[idx + 1]):
        return float(idx), spec[idx]
    left, center, right = np.log(spec[idx - 1:idx + 2])
    curvature = left - 2 * center + right
    if curvature >= 0:
        return float(idx), spec[idx]
    offset = 0.5 * (left - right) / curvature
    return idx + offset, np.exp(center - 0.25 * (left - right) * offset)


def _minmax_envelope(x, y, columns, half_width, half_columns):
    """Returns the polygon around the minimum and maximum of y within each of at most `columns`
    buckets, widened by half_width vertically and half_columns buckets horizontally like a stroke,
    or None if there are less than two samples per bucket. Filling it renders like a line through
    all samples, but much faster than stroking that dense zigzag line or the jagged polygon outline."""
    bucket = -(-len(y) // max(columns, 1))
    if bucket < 2:
        return None
    starts = np.arange(0, len(y), bucket)
    xs = x[starts]
    size = 2 * half_columns + 1
    lower = minimum_filter1d(np.minimum.reduceat(y, starts), size, mode="nearest") - half_width
    upper = maximum_filter1d(np.maximum.reduceat(y, starts), size, mode="nearest") + half_width
    return np.column_stack([np.concatenate([xs, xs[::-1]]), np.concatenate([upper, lower[::-1]])])


class ScopeWidget:
    _open_scopes = {}
    spec_floor = 1e-12
    lock_min_correlation = 0.5
    lock_octave_tolerance = 0.8
    lock_analysis_size = 8192
    sig_marker_max_duration = 1e-3
    peak_close_distance = 0.02

    def __init__(self,
                 server=None,
                 fps=30,
                 mode="output",
                 window: str = "blackmanharris",
                 window_size: int = 8192,
                 figsize=(8, 3),
                 spec_decay: float | None = 0.5,
                 widgets=DEFAULT_CONTROLS,
                 num: int | str = "ScopeWidget",
        ):
        """ScopeWidget - a Scope/FreqScope view for Jupyter notebooks
        for interactive contexts using %matplotlib widget

        Args:
            server (Aserver, optional): pya Aserver. Defaults to Aserver.default.
            fps (int, optional): frames per second for rendering. Defaults to 30.
            mode (str, optional): source of the signal, currently only "output".
            window (str): type of the window function used for the spectrum.
            window_size (int): size of the analyzed window. Power of 2 recommended. Has to be <= server.history_size
            figsize (tuple, optional): figure size. Defaults to (8, 3).
            spec_decay (float | None, optional): seconds it takes for a spectrum bin to fall
                over the whole y range. None or 0 shows the plain spectrum of each frame. Defaults to 0.5.
            widgets (iterable of str, optional): controls shown below the plot, in this order.
                Available are "fps", "window_size", "spec_decay", "lock_period", "quit" and "pause_resume".
                Defaults to ("window_size", "spec_decay", "lock_period", "quit").
            num (int | str, optional): matplotlib figure number or label. A new ScopeWidget closes
                the open figure and scope with the same num, so only different nums give several
                scopes at once. Defaults to "ScopeWidget".
        """
        _check_mode(mode)
        _check_controls(widgets)
        previous = ScopeWidget._open_scopes.get(num)
        if previous is not None:
            previous.close()
        plt.close(num)
        self.num = num
        self.server = server or Aserver.default
        self.sr = self.server.sr
        self._fps = fps
        self._mode = mode
        self._window = window
        self._window_size = window_size
        self._spec_decay = spec_decay or None
        self._lock_period = False
        self._controls = {}

        self.spec_lin_xlim = (0, self.sr // 2 + 1)
        self.spec_log_xlim = (20, self.sr // 2 + 1)
        self.spec_lin_ylim = (0, 1.118)
        self.spec_log_ylim = (1e-4, 3)
        self.spec_peak_prominence = 1
        self.spec_peak_snap_distance = 0.08
        self.peak_label_offset_points = 40

        self.spec_hovered = False
        self.spec_selected_idx = None
        self.spec_selected_peak = None
        self.mouse_ax_pos = np.zeros(2)
        self.mouse_freq = 0.0
        self._selection_dirty = False
        self._sig_duration = None
        self._sig_envelope_xy = None
        self._spec_columns_key = None
        self._background = None
        self._closed = False
        self._background_view = None
        self._dirty = True
        self.last_frame_time = time()
        self._init_window()

        self.output = ipw.Output()
        with self.output:
            self._create_figure(figsize)
            plt.show()

        self.timer = self.fig.canvas.new_timer(interval=int(1000 / fps))
        self.timer.add_callback(self._update)
        self.timer.start()
        ScopeWidget._open_scopes[num] = self

        self.controls = self._create_controls(widgets)
        self.scope_widgets = ipw.VBox([self.output, self.controls])
        self.show()

    def _create_figure(self, figsize):
        self.fig, (self.axsig, self.axspec) = plt.subplots(1, 2, figsize=figsize, num=self.num)

        (self.line2Dsig,) = self.axsig.plot([], [], "b-")
        self.sig_envelope = self.axsig.add_patch(Polygon(
            np.zeros((0, 2)), closed=False, color=self.line2Dsig.get_color(), lw=0, visible=False,
        ))
        self.axsig.set_xlabel("time")
        self.axsig.set_ylabel("signal [arb. units]")
        self.axsig.grid()

        (self.line2Dhold,) = self.axspec.plot([], [], "-", color=HOLD_COLOR, visible=False)
        (self.line2Dspec,) = self.axspec.plot([], [], "b-")
        (self.line2Dspec_p,) = self.axspec.plot(
            [], [], "o", color=HOLD_COLOR, mec="white", mew=1.5, ms=7, visible=False,
        )
        self.peak_label = self.axspec.annotate(
            "", xy=(0, 0), xytext=(0, 0), textcoords="offset points", ha="center", visible=False,
            bbox=dict(boxstyle="round", fc="w", ec="0.5"),
        )
        self.axspec.set_xlabel("frequency")
        self.axspec.set_ylabel("amplitude A(f)")
        self.axspec.grid()
        self._reset_view(self.axsig)
        self._reset_view(self.axspec)
        self.fig.tight_layout()

        self._artists = (self.sig_envelope, self.line2Dsig, self.line2Dhold, self.line2Dspec, self.line2Dspec_p, self.peak_label)
        canvas = self.fig.canvas
        # The WebAgg based canvas of ipympl reports supports_blit = False, but its blit() sends the renderer buffer.
        self._blit = hasattr(canvas, "copy_from_bbox")
        for artist in self._artists:
            artist.set_animated(self._blit)
        # WebAgg frontends drop an image when the next one arrives before it is decoded. In the default
        # diff mode only changed pixels are sent, so every dropped image leaves artifacts. Hence full
        # images are sent, and only when the frontend requests them after a draw_idle, which it does
        # not while it waits for an image.
        self._frontend_requests_frames = isinstance(canvas, FigureCanvasWebAggCore)
        if self._frontend_requests_frames:
            canvas.handle_draw = self._render_requested_frame

        canvas.mpl_connect("draw_event", self._on_draw)
        canvas.mpl_connect("axes_enter_event", self._on_axes_enter)
        canvas.mpl_connect("axes_leave_event", self._on_axes_leave)
        canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        canvas.mpl_connect("button_press_event", self._on_mouse_click)
        canvas.mpl_connect("key_press_event", self._on_key)

    def _create_controls(self, names):
        children = []
        for name in names:
            if name == "quit":
                children.append(self._button("Quit", self.close))
            elif name == "pause_resume":
                children += [self._button("Pause", self.pause), self._button("Resume", self.resume)]
            else:
                self._controls[name] = getattr(self, f"_create_{name}_control")()
                children.append(self._controls[name])
        return ipw.HBox(
            children,
            layout=ipw.Layout(display="flex", flex_flow="row wrap", align_items="stretch", width="100%"),
        )

    def _bind(self, widget, attribute):
        widget.observe(lambda change: setattr(self, attribute, change["new"]), names="value")
        return widget

    @staticmethod
    def _button(description, callback):
        button = ipw.Button(description=description, layout=ipw.Layout(width="70px"))
        button.on_click(lambda _: callback())
        return button

    def _create_fps_control(self):
        return self._bind(ipw.IntSlider(description="FPS", value=self.fps, min=1, max=60), "fps")

    def _create_window_size_control(self):
        return self._bind(ipw.SelectionSlider(
            description="Window size",
            options=_steps_with(WINDOW_SIZES, self.window_size),
            value=self.window_size,
            style={"description_width": "initial"},
        ), "window_size")

    def _create_spec_decay_control(self):
        steps = _steps_with(SPEC_DECAY_STEPS, self.spec_decay)
        self._spec_decay_values = (None, *steps)
        slider = ipw.SelectionSlider(
            description="Spec decay",
            options=("off", *(_with_unit(step, "s") for step in steps)),
            index=self._spec_decay_values.index(self.spec_decay),
            style={"description_width": "initial"},
        )
        slider.observe(lambda change: setattr(self, "spec_decay", self._spec_decay_values[change["new"]]), names="index")
        return slider

    def _create_lock_period_control(self):
        return self._bind(ipw.Checkbox(
            description="Lock period", value=self.lock_period, indent=False, layout=ipw.Layout(width="110px"),
        ), "lock_period")

    def _sync_control(self, name, value):
        widget = self._controls.get(name)
        if widget is None:
            return
        if name == "spec_decay":
            if value in self._spec_decay_values:
                widget.index = self._spec_decay_values.index(value)
        else:
            options = getattr(widget, "options", None)
            if options is None or value in options:
                widget.value = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        _check_mode(value)
        self._mode = value

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, value):
        self._fps = value
        self.timer.interval = int(1000 / value)
        self._sync_control("fps", value)

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, value: str):
        self._window = value
        self._init_window()

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, value: int):
        self._window_size = int(value)
        self._init_window()
        self._sync_control("window_size", self._window_size)

    @property
    def spec_decay(self):
        return self._spec_decay

    @spec_decay.setter
    def spec_decay(self, value: float | None):
        self._spec_decay = value or None
        self._sync_control("spec_decay", self._spec_decay)

    @property
    def lock_period(self):
        return self._lock_period

    @lock_period.setter
    def lock_period(self, value: bool):
        self._lock_period = value
        self._sync_control("lock_period", value)

    def _init_window(self):
        n = self._window_size
        window = get_window(self._window, n)
        self.norm_window = window / np.sum(window)
        self._sig = np.zeros(n)
        self._sig_time = np.arange(n) / self.sr
        self._windowed = np.empty(n)
        self.spec_freqs = np.fft.rfftfreq(n, 1 / self.sr)
        self.last_spec = np.full(len(self.spec_freqs), self.spec_floor)
        self._spec_columns_key = None
        self._reset_hold()

    def _reset_hold(self):
        self._hold_spec = self.last_spec.copy()
        self._peaks_idx = None
        self._selection_dirty = True

    def show(self):
        """display the scope UI including plot and widgets"""
        display(self.scope_widgets)

    def pause(self):
        self.timer.stop()

    def resume(self):
        self.timer.start()

    def close(self):
        """Stops the scope and replaces the live figure and the controls by an image of the last frame."""
        if self._closed:
            return
        self._closed = True
        self.timer.stop()
        if ScopeWidget._open_scopes.get(self.num) is self:
            del ScopeWidget._open_scopes[self.num]

        last_frame = self._snapshot()
        plt.close(self.fig)
        self.output.outputs = ()
        self.output.append_display_data(last_frame)
        self.scope_widgets.children = (self.output,)
        _close_widget_tree(self.controls)

    def _snapshot(self):
        for artist in self._artists:
            artist.set_animated(False)
        png = io.BytesIO()
        self.fig.savefig(png, format="png", dpi=self.fig.dpi)
        return Image(data=png.getvalue(), width=round(self.fig.bbox.width / self.fig.canvas.device_pixel_ratio))

    def __del__(self):
        timer = getattr(self, "timer", None)
        if timer is not None:
            timer.stop()

    def _update(self):
        # Closing the figure in pyplot (plt.close, ipympl) emits no close_event, so it is checked here.
        if not plt.fignum_exists(self.fig.number):
            self.close()
            return
        now = time()
        dt, self.last_frame_time = now - self.last_frame_time, now

        sig = self._latest_signal()
        self._draw_signal(sig)

        np.multiply(self.norm_window, sig, out=self._windowed)
        spec = np.abs(np.fft.rfft(self._windowed))
        spec[1:-1] *= 2
        self._update_spectrum(spec, dt)

        if self.spec_hovered:
            self._update_hold(spec)
            self._update_selection()
            self._update_peak_color()
        self._draw_spectrum()

        if self._dirty or self._view_state() != self._background_view:
            if self._frontend_requests_frames:
                self.fig.canvas.draw_idle()
            else:
                self._render()

    def _latest_signal(self):
        """Returns the latest window_size output samples of channel 0, zero-padded where no audio exists."""
        sig = self._sig
        recent = self.server.get_output_history(len(sig), channel=0) if self.server.is_active else sig[:0]
        start = len(sig) - len(recent)
        sig[:start] = 0
        sig[start:] = recent
        return sig

    def _set_line_data(self, line, x, y):
        if not (np.array_equal(line.get_ydata(), y) and np.array_equal(line.get_xdata(), x)):
            line.set_data(x, y)
            self._dirty = True

    def _set_visible(self, artist, visible):
        if artist.get_visible() != visible:
            artist.set_visible(visible)
            self._dirty = True

    def _draw_signal(self, sig):
        locked = self._locked_period(sig) if self.lock_period else None
        if locked is None:
            t = self._sig_time
            duration = len(sig) / self.sr
        else:
            crossing, period = locked
            idx = np.arange(int(crossing), min(len(sig), int(np.ceil(crossing + period)) + 1))
            t, sig = (idx - crossing) / self.sr, sig[idx]
            duration = period / self.sr
        self._update_sig_xlim(duration)

        xmin, xmax = self.axsig.get_xlim()
        first, last = np.searchsorted(t, (xmin, xmax))
        visible = slice(max(first - 1, 0), last + 1)
        half_pixels = self.line2Dsig.get_linewidth() / 2 * self.fig.dpi / 72
        ymin, ymax = self.axsig.get_ylim()
        envelope = _minmax_envelope(
            t[visible], sig[visible], int(self.axsig.bbox.width),
            half_pixels * (ymax - ymin) / self.axsig.bbox.height, round(half_pixels - 0.5),
        )
        if envelope is None:
            self._set_line_data(self.line2Dsig, t[visible], sig[visible])
        elif not np.array_equal(self._sig_envelope_xy, envelope):
            self._sig_envelope_xy = envelope
            self.sig_envelope.set_xy(envelope)
            self._dirty = True
        self._set_visible(self.line2Dsig, envelope is None)
        self._set_visible(self.sig_envelope, envelope is not None)

        marker = "." if xmax - xmin <= self.sig_marker_max_duration else "None"
        if self.line2Dsig.get_marker() != marker:
            self.line2Dsig.set_marker(marker)
            self._dirty = True

    def _update_sig_xlim(self, duration):
        """Follows a changed duration only if all of it was shown or nothing would be shown anymore,
        so that a zoomed view is kept."""
        if duration == self._sig_duration:
            return
        xmin, xmax = self.axsig.get_xlim()
        showed_all = self._sig_duration is None or np.allclose((xmin, xmax), (0, self._sig_duration))
        if showed_all or xmin >= duration:
            self.axsig.set_xlim(0, duration)
        self._sig_duration = duration

    def _locked_period(self, sig):
        """Returns (crossing, period) of the latest full oscillation, where crossing is the
        linearly interpolated position of its rising zero crossing in samples,
        or None if the latest lock_analysis_size samples are not periodic within half of them."""
        segment = sig[-self.lock_analysis_size:]
        centered = segment - segment.mean()
        if not centered.any():
            return None
        n = len(centered)
        acf = correlate(centered, centered, mode="full", method="fft")[n - 1:n - 1 + n // 2]
        peaks, properties = find_peaks(acf, height=self.lock_min_correlation * acf[0])
        if len(peaks) == 0:
            return None
        # Take the shortest lag that is nearly as strong as the strongest one,
        # so that multiples of the period (lower octaves) are not chosen.
        heights = properties["peak_heights"]
        period = peaks[np.argmax(heights >= self.lock_octave_tolerance * heights.max())]

        rising = np.flatnonzero((centered[:-1] < 0) & (centered[1:] >= 0))
        rising = rising[rising < n - 1 - period]
        if len(rising) == 0:
            return None
        i = rising[-1]
        before, after = centered[i], centered[i + 1]
        return len(sig) - n + i + before / (before - after), period

    def _update_spectrum(self, spec, dt):
        """Without spec_decay the plain spectrum is shown, otherwise a peak meter that rises instantly."""
        if self.spec_decay:
            if self.axspec.get_yscale() == "log":
                low, high = self.spec_log_ylim
                self.last_spec *= (low / high) ** (dt / self.spec_decay)
            else:
                low, high = self.spec_lin_ylim
                self.last_spec -= dt / self.spec_decay * (high - low)
            np.maximum(self.last_spec, spec, out=self.last_spec)
        else:
            self.last_spec[:] = spec
        np.maximum(self.last_spec, self.spec_floor, out=self.last_spec)

    def _spectrum_columns(self):
        """Returns the first bin of each group of bins that share a pixel column (cached)."""
        ax = self.axspec
        key = (ax.viewLim.intervalx.tobytes(), ax.get_xscale(), ax.bbox.bounds, len(self.spec_freqs))
        if key != self._spec_columns_key:
            freqs = self.spec_freqs
            columns = np.floor(ax.transData.transform(np.column_stack([freqs, np.ones_like(freqs)]))[:, 0])
            self._spec_columns = np.flatnonzero(np.r_[True, columns[1:] != columns[:-1]] | (freqs == freqs[-1]))
            self._spec_columns_key = key
        return self._spec_columns

    def _decimated_spectrum(self, spec):
        """The maximum of each pixel column, so that peaks are kept."""
        starts = self._spectrum_columns()
        freqs, values = self.spec_freqs[starts], np.maximum.reduceat(spec, starts)
        visible = ~_inside_offscreen_run(values, *self.axspec.get_ylim())
        return freqs[visible], values[visible]

    def _draw_spectrum(self):
        self._set_line_data(self.line2Dspec, *self._decimated_spectrum(self.last_spec))
        if self.spec_hovered:
            freqs, values = self._decimated_spectrum(self._hold_spec)
            if self.spec_selected_peak is not None:
                freq, mag = self.spec_selected_peak
                position = np.searchsorted(freqs, freq)
                freqs, values = np.insert(freqs, position, freq), np.insert(values, position, mag)
            self._set_line_data(self.line2Dhold, freqs, values)

    def _update_hold(self, spec):
        if np.any(spec > self._hold_spec):
            np.maximum(self._hold_spec, spec, out=self._hold_spec)
            self._peaks_idx = None
            self._selection_dirty = True

    def _update_selection(self):
        if not self._selection_dirty:
            return
        self._selection_dirty = False
        self._dirty = True
        if self._peaks_idx is None:
            self._peaks_idx = find_peaks(np.log(self._hold_spec), prominence=self.spec_peak_prominence)[0]

        idx = self.spec_selected_idx = self._selected_bin()
        fractional_bin, mag = _parabolic_peak(self._hold_spec, idx)
        freq = fractional_bin * self.sr / self._window_size
        self.spec_selected_peak = freq, mag
        self.line2Dspec_p.set_data([freq], [mag])
        self.line2Dspec_p.set_visible(True)
        freq_text = _with_unit(freq, "Hz", ".1f" if freq < 100 else ".0f")
        self.peak_label.set_text(f"{freq_text}\n{self._format_amplitude(mag)}")
        self._place_peak_label(freq, mag)

    def _update_peak_color(self):
        """The peak point is drawn in the spectrum color while the current spectrum is visually
        close to its maximum, measured in axes fractions so that it follows the y scaling."""
        idx = self.spec_selected_idx
        freq = self.spec_freqs[idx]
        to_axes = self.axspec.transData - self.axspec.transAxes
        (_, current), (_, peak) = to_axes.transform([(freq, self.last_spec[idx]), (freq, self._hold_spec[idx])])
        close = abs(peak - current) < self.peak_close_distance
        color = self.line2Dspec.get_color() if close else HOLD_COLOR
        if self.line2Dspec_p.get_color() != color:
            self.line2Dspec_p.set_color(color)
            self._dirty = True

    def _place_peak_label(self, freq, mag):
        label = self.peak_label
        label.xy = (freq, mag)
        label.set_visible(True)
        label.set_position((0, self.peak_label_offset_points))
        label.set_verticalalignment("bottom")

        renderer = self.fig.canvas.get_renderer()
        label.update_positions(renderer)
        label.update_bbox_position_size(renderer)
        if label.get_bbox_patch().get_window_extent().y1 > self.axspec.bbox.y1:
            label.set_position((0, -self.peak_label_offset_points))
            label.set_verticalalignment("top")

    def _selected_bin(self):
        peaks = self._peaks_idx
        if len(peaks) > 0:
            peaks_data = np.column_stack([self.spec_freqs[peaks], self._hold_spec[peaks]])
            peaks_ax_pos = (self.axspec.transData - self.axspec.transAxes).transform(peaks_data)
            distances = np.linalg.norm(peaks_ax_pos - self.mouse_ax_pos, axis=1)
            nearest = np.argmin(distances)
            if distances[nearest] < self.spec_peak_snap_distance:
                return peaks[nearest]
        return int(np.clip(round(self.mouse_freq * self._window_size / self.sr), 0, len(self.spec_freqs) - 1))

    def _view_state(self):
        return (
            self.axsig.viewLim.bounds, self.axsig.get_xscale(), self.axsig.get_yscale(),
            self.axspec.viewLim.bounds, self.axspec.get_xscale(), self.axspec.get_yscale(),
            self.fig.bbox.bounds,
        )

    def _render(self):
        """Redraws only the changing artists over a cached background when the canvas allows it.
        A full draw is only needed when limits, scales or the figure size changed."""
        self._dirty = False
        canvas = self.fig.canvas
        if not self._blit:
            canvas.draw_idle()
        elif self._background is None or self._view_state() != self._background_view:
            canvas.draw()
        else:
            canvas.restore_region(self._background)
            self._draw_artists()
            canvas.blit(self.fig.bbox)

    def _render_requested_frame(self, event):
        self.fig.canvas._force_full = True
        self._render()

    def _on_draw(self, event):
        if not self._blit:
            return
        self._background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._background_view = self._view_state()
        self._draw_artists()

    def _draw_artists(self):
        for artist in self._artists:
            artist.axes.draw_artist(artist)

    def _on_axes_enter(self, event):
        if event.inaxes is self.axspec:
            self.spec_hovered = True
            self._reset_hold()
            self.line2Dhold.set_visible(True)

    def _on_axes_leave(self, event):
        if event.inaxes is self.axspec:
            self.spec_hovered = False
            self.spec_selected_idx = None
            self.spec_selected_peak = None
            for artist in (self.line2Dhold, self.line2Dspec_p, self.peak_label):
                artist.set_visible(False)
            self._dirty = True

    def _on_mouse_move(self, event):
        if event.inaxes is self.axspec:
            self.mouse_freq = event.xdata
            self.mouse_ax_pos = self.axspec.transAxes.inverted().transform((event.x, event.y))
            self._selection_dirty = True

    def _on_mouse_click(self, event):
        if event.inaxes is self.axspec and event.button == 1 and self.spec_selected_idx is not None:
            (SinOsc(self.spec_selected_peak[0]) * Line(0.1, 0, 0.2)).dup(2).play()

    def _on_key(self, event):
        """matplotlib's own key handler is connected first and already toggled the scale for the
        keymap.xscale ("k") and keymap.yscale ("l") keys, so only limits and units are adapted here."""
        ax = event.inaxes
        if ax is None:
            return
        if event.key in plt.rcParams["keymap.xscale"]:
            self._set_unit_formatters(ax)
            if ax is self.axspec:
                ax.set_xlim(self.spec_log_xlim if ax.get_xscale() == "log" else self.spec_lin_xlim)
            else:
                ax.relim(visible_only=True)
                ax.autoscale(axis="x")
        elif event.key in plt.rcParams["keymap.yscale"]:
            self._set_unit_formatters(ax)
            if ax is self.axspec:
                ax.set_ylim(self.spec_log_ylim if ax.get_yscale() == "log" else self.spec_lin_ylim)
            else:
                ax.relim(visible_only=True)
                ax.autoscale(axis="y")
        elif event.key == "a":
            ax.relim(visible_only=True)
            ax.autoscale()
        elif event.key == "r":
            self._reset_view(ax)
        self._selection_dirty = True
        self.fig.canvas.draw_idle()

    def _reset_view(self, ax):
        if ax is self.axsig:
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            ax.set_xlim(0, self._sig_duration or self._window_size / self.sr)
            ax.set_ylim(-1.2, 1.2)
        elif ax is self.axspec:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(self.spec_log_xlim)
            ax.set_ylim(self.spec_log_ylim)
        self._set_unit_formatters(ax)

    def _set_unit_formatters(self, ax):
        """Scale changes reset the tick formatters, so this has to be called after each of them."""
        if ax is self.axsig:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: _with_unit(t * 1e3, "ms")))
        elif ax is self.axspec:
            ax.xaxis.set_major_formatter(EngFormatter(unit="Hz", sep=UNIT_SEP))
            if ax.get_yscale() == "log":
                ax.yaxis.set_major_formatter(FuncFormatter(lambda a, _: _with_unit(_to_db(a), "dB")))

    def _format_amplitude(self, amplitude):
        if self.axspec.get_yscale() == "log":
            return _with_unit(_to_db(amplitude), "dB", ".1f")
        return f"{amplitude:.3g}"
