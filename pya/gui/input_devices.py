import threading
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.widgets as mwidgets


def qwerty_keyboard_controller(playfn, transpose=0, keys=""):
    """qwerty_keyboard controller, e.g. as AGen player.

    This function opens a Tk Window in which keybindings allow
    to call custom functions, e.g. to synthesize sound.
    The keybindings make sense for an English international kepboard,
    For a German keyboard, simply exchange y by z, the keystring below
    or pass your own custom key series for semitones

    Arguments:
    ----------

    playfn (Callable): function of a `note` argument

    transpose (int): added to the argument of playfn

    keys (string): sequence of keys to be used for semitones
        if keys=="", the default keymap is taken

    Example:
    --------
    from pya.agen.lib import PlayAsig, LFSaw, XLine
    from pyamapping import midi_to_cps

    def play_note_fn(note):
        (LFSaw(midi_to_cps(note+36)) * XLine(0.5, 0.05, 1.5)).play()

    qwerty_keyboard_controller(play_note_fn)
    """
    import tkinter as tk

    default_key_sequence = "zsxdcvgbhnjmq2w3er5t6y7ui9o0p[=]"  # English keyboard
    if keys == "":
        keys = default_key_sequence
    root = tk.Tk()
    root.title("Keyboard Piano")
    tk.Label(root, text="Press keys: " + keys).pack(pady=10)
    root.lift()
    root.attributes("-topmost", True)
    root.after_idle(root.attributes, "-topmost", False)

    def on_key(event):
        try:
            idx = keys.index(event.char)
            playfn(idx + transpose)
        except ValueError:
            pass

    root.bind("<Key>", on_key)
    root.mainloop()


class KeyboardControllerJupyter:
    """
    QWERTY keyboard controller for Jupyter.
    - Requires pynput

    Calls `note_on(note)` when a mapped key is pressed,
    and `note_off(note)` when it is released.
    """

    DEFAULT_KEYS = "zsxdcvgbhnjmq2w3er5t6y7ui9o0p[=]"

    def __init__(self, note_on, note_off, transpose=0, keys=""):
        """
        Parameters:
        -----------
        note_on : Callable[[int], None]
            Called when a key is pressed.
        note_off : Callable[[int], None]
            Called when a key is released.
        transpose : int
            Value added to the note index.
        keys : str
            Sequence of keys representing semitones.
        """

        from pynput import keyboard
        from ipywidgets import widgets
        from IPython.display import display

        self.keys = keys or self.DEFAULT_KEYS
        self.transpose = transpose
        self.note_on = note_on
        self.note_off = note_off
        self.pressed = set()  # track currently pressed keys

        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self.thread = threading.Thread(target=self.listener.start, daemon=True)
        self.thread.start()

        # Textarea used to capture focus + keystrokes
        self.wdg = widgets.Textarea(
            value="",
            placeholder="Click here and play the keyboard piano…",
            description="Piano:",
            disabled=False,
            layout={"width": "600px", "height": "30px"},
        )
        self.wdg_quit = widgets.Button(description="Quit")
        self.wdg_quit.on_click(lambda x: self.stop())
        display(self.wdg, self.wdg_quit)

    def _on_press(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            return  # special key (shift, ctrl, etc.) ignored
        if k in self.keys and k not in self.pressed:
            self.pressed.add(k)
            note = self.keys.index(k) + self.transpose
            self.note_on(note)

    def _on_release(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            return
        if k in self.pressed:
            self.pressed.remove(k)
            note = self.keys.index(k) + self.transpose
            self.note_off(note)

    def stop(self):
        """Stop the listener."""
        self.listener.stop()
        self.thread.join()


# matplotlib widget based qwerty controller
class KeyboardControllerMPL:
    """
    Interactive keyboard piano controller using matplotlib widgets.

    Only triggers note_on/note_off when the figure has focus.
    Provides a Quit button and temporarily disables conflicting key bindings.
    """

    DEFAULT_KEYS = "zsxdcvgbhnjmq2w3er5t6y7ui9o0p[=]"

    def __init__(self, note_on, note_off, transpose=0, keys="", fig=None):
        self.keys = keys or self.DEFAULT_KEYS
        self.transpose = transpose
        self.note_on = note_on
        self.note_off = note_off
        self.pressed = set()

        # Backup original keymaps
        self._original_keymaps = {
            k: list(mpl.rcParams[k]) for k in mpl.rcParams if k.startswith("keymap")
        }

        # Disable conflicting key bindings
        for keymap in [
            "keymap.quit",
            "keymap.save",
            "keymap.home",
            "keymap.fullscreen",
            "keymap.grid",
            "keymap.zoom",
            "keymap.pan",
        ]:
            mpl.rcParams[keymap] = []

        # Create figure
        if fig is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set_title("Keyboard Piano (focus here)")
            self.ax.axis("off")
        else:
            print("using given figure")
            self.fig = fig

        # Connect keyboard events
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)

        # Add a Quit button
        ax_quit = plt.axes([0.85, 0.01, 0.1, 0.05])
        self._btn_quit = mwidgets.Button(ax_quit, "Quit")
        self._btn_quit.on_clicked(lambda event: self.close())

    def _on_key_press(self, event):
        if event.key is None:
            return
        k = event.key.lower()
        if k in self.keys and k not in self.pressed:
            self.pressed.add(k)
            note = self.keys.index(k) + self.transpose
            self.note_on(note)

    def _on_key_release(self, event):
        if event.key is None:
            return
        k = event.key.lower()
        if k in self.pressed:
            self.pressed.remove(k)
            note = self.keys.index(k) + self.transpose
            self.note_off(note)

    def close(self):
        """Close the figure and restore original keymaps."""
        # Release all pressed notes
        for k in list(self.pressed):
            note = self.keys.index(k) + self.transpose
            self.note_off(note)
        self.pressed.clear()

        # Restore original keymaps
        for k, v in self._original_keymaps.items():
            mpl.rcParams[k] = v

        plt.close(self.fig)
