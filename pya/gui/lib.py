from typing import Callable
import matplotlib.pyplot as plt
import numpy as np

class MultiSliderPlot:

    def __init__(
            self, 
            data, 
            fig=None, 
            ax=None,
            bottom=0, 
            range=[0, 10],
            margin=0,
            figsize=(8, 3), 
            sticky=False, 
            positions=None, 
            clipping=True, 
            autoscale=False,
            on_motion_callback=None
        ):
        self.data = data
        self.fig = fig if fig is not None else plt.figure(figsize=figsize)
        self.bottom = bottom
        self.range = range
        self.margin = margin
        self.sticky = sticky
        self.positions = positions if positions is not None else np.arange(len(self.data))
        self.clipping = clipping
        self.autoscale = autoscale
        self.on_motion_callback = on_motion_callback
        self.slider_index = 0 

        self.selected_bar = None

        if ax is None:
            self.ax = self.fig.add_subplot(1, 1, 1)
        else: 
            self.ax = ax
        self.bars = self.ax.bar(self.positions, self.data, align='center', bottom=bottom, alpha=0.8)
        self.ax.set_ylim(self.range[0]-self.margin, self.range[1]+self.margin)
        plt.tight_layout()

        def on_press(event):
            if event.inaxes is not self.ax:
                return
            self.slider_index = np.argmin(np.abs(self.positions - event.xdata))
            self.selected_bar = self.bars[self.slider_index]
        
        def on_motion(event):
            if event is not None:
                if self.selected_bar is None:
                    return
                if event.inaxes is not self.ax:
                    return
                if self.sticky is False:
                    self.slider_index = np.argmin(np.abs(self.positions - event.xdata))
                    self.selected_bar = self.bars[self.slider_index]
                new_value = event.ydata
                if self.clipping:
                    new_value = np.clip(new_value, *self.range)
                self.selected_bar.set_height(abs(new_value))
                self.selected_bar.set_y(new_value if new_value < self.bottom else self.bottom)

                self.data[self.slider_index] = new_value
                if isinstance(self.on_motion_callback, Callable):
                    self.on_motion_callback(self)
            self.fig.canvas.draw()

        def on_release(event):
            self.selected_bar = None

        def on_key(event):
            if event.inaxes is not self.ax:
                return
            if event.key == 't': 
                self.sticky = not self.sticky
            if event.key == 'r':
                pass # self.data 
        
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', on_press)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', on_motion)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', on_release)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', on_key)

    def set_callback(self, fn):
        self.on_motion_callback = fn

    def set_data(self, data, positions=None):
        self.data = data
        if positions is not None: 
            self.positions = positions 
        for i, v in enumerate(self.data):
            self.bars[i].set_height(abs(v))
            self.bars[i].set_y(v if v < self.bottom else self.bottom)
        if self.autoscale:
            if self.autoscale == "abs":
                v_max = max(np.abs(self.data))
                v_min = -v_max
            else:
                v_min = min(self.data-self.margin)
                v_max = max(self.data+self.margin)
            if v_min != v_max:
                self.ax.set_ylim(v_min-self.margin, v_max+self.margin)

    def __del__(self):
        self.close()

    def close(self):
        self.fig.canvas.mpl_disconnect(self.cid_press)
        self.fig.canvas.mpl_disconnect(self.cid_motion)
        self.fig.canvas.mpl_disconnect(self.cid_release)
        self.fig.canvas.mpl_disconnect(self.cid_key)


def ctrl_gui(agen, **kwargs):
    """Render ipywidgets GUI for synth controls.
    This function can be used directly, or indirectly by setting the widget
    argument of the AGen's playx() method (see pya.gui.AGenPlayGUI).

    Parameters
    ----------
    - agen (AGen)
        this should be an AGen with `ctrl` attribute (which is currently added to
        AGen via the @asynth decorator.
    - kwargs
        to specify (min, max, step) for parameter names analogous to
        ipywidgets.interactive

    Returns
    -------
    the return value of ipywidgets.interactive()
    """
    from ipywidgets import interactive
    import inspect

    ix_dict = {}
    set_param_kwargs = {}
    sig_params = []

    for k, v in kwargs.items():
        if k in agen.ctrl._nodes:  # only kwargs that are agen nodes
            ix_dict[k] = v  # store ranges for interactive()
            gen = agen.ctrl[k]
            if isinstance(gen, (int, float, bool)):  # for default value extraction
                set_param_kwargs[k] = gen
                p = inspect.Parameter(
                    name=k,
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=gen,
                )
                sig_params.append(p)
        else:
            print(f"ctrlgui(): parameter {k} not in agen.ctrl nodes")

    def _set_params(**set_param_kwargs):  # ():
        for k, v in set_param_kwargs.items():
            agen.ctrl[k] = v

    _set_params.__signature__ = inspect.Signature(sig_params)

    return interactive(_set_params, **ix_dict)
