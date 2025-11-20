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
