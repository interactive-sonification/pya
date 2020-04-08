import matplotlib.pyplot as plt
import math


def gridplot(pya_objects, titles=None, col_wrap=None, figsize=None):
    """Create a grid plot of pya objects which have plot() methods, i.e. Asig, Aspec, Astft, Amfcc
    TODO add sharex, sharey.

    Parameters
    ----------
    pya_objects : iterable object
        A list of pya objects with the plot() method.
    col_wrap : int
        Wrap column at position. Can be considered as the column size.
    figsize : tuple
        width, height in inches. default size is (6.4, 4.8)

    Returns
    -------
    fig :
    """
    sig_len = len(pya_objects)
    fig = plt.figure(figsize=figsize)

    # Figure what the grid dimension is .
    if not col_wrap:
        col_wrap = sig_len

    ncol = col_wrap
    nrow = math.ceil(sig_len / ncol)

    if not titles:
        titles = [str(i) for i in range(sig_len)]

    for idx in range(sig_len):
        plt.subplot(nrow, ncol, idx + 1)
        pya_objects[idx].plot()
        plt.title(titles[idx])
    plt.tight_layout()
    return fig