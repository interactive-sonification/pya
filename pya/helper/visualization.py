from __future__ import absolute_import  # This allows the next line to work.
from .. import Asig, Amfcc, Astft, Aspec
import matplotlib.pyplot as plt
import math
import matplotlib.gridspec as grd


def gridplot(pya_objects, col_wrap=1, cbar_ratio=0.04, figsize=None):
    """Create a grid plot of pya objects which have plot() methods, i.e. Asig, Aspec, Astft, Amfcc.
    It takes a list of pya_objects and plot each object into a grid. You can mix different types of plots
    together.

    Examples
    --------
    # plot all 4 different pya objects in 1 column, amfcc and astft use pcolormesh so colorbar will
    # be displayed as well
    gridplot([asig, amfcc, aspec, astft], col_wrap=1, cbar_ratio=0.08, figsize=[10, 10]);

    Parameters
    ----------
    pya_objects : iterable object
        A list of pya objects with the plot() method.
    col_wrap : int, optional
        Wrap column at position. Can be considered as the column size. Default is 1, meaning 1 column.
    cbar_ratio : float, optional
        For each column create another column reserved for the colorbar. This is the ratio
        of the width relative to the plot. 0.04 means 4% of the width of the data plot.
    figsize : tuple, optional
        width, height of the entire image in inches. Default size is (6.4, 4.8)

    Returns
    -------
    fig : plt.figure()
        The plt.figure() object
    """
    nplots = len(pya_objects)

    if col_wrap > nplots:
        ncol = col_wrap
    elif col_wrap < 1:
        raise ValueError("col_wrap needs to an integer > 0")
    else:
        ncol = col_wrap

    nrow = math.ceil(nplots / ncol)
    ncol = ncol * 2  # Double the col for colorbars.

    even_weight = 100
    odd_weight = even_weight * cbar_ratio
    wratio = []  # Aspect ratio of width
    for i in range(ncol):
        wratio.append(odd_weight) if i % 2 else wratio.append(even_weight)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = grd.GridSpec(nrow, ncol, figure=fig, width_ratios=wratio, wspace=0.01)

    total_idx = ncol * nrow
    for i in range(total_idx):
        if not i % 2:  # The data plot is always at even index.
            idx = i // 2  # Index in the pya_objects list
            if idx < nplots:
                ax = plt.subplot(grid[i])
                # Title is object type + label
                title = pya_objects[idx].__repr__().split('(')[0] + ': ' + pya_objects[idx].label
                title = (title[:30] + "..." if len(title) > 30 else title)  # Truncate if str too long
                ax.set_title(title)
                if isinstance(pya_objects[idx], Asig.Asig):
                    pya_objects[idx].plot()
                elif isinstance(pya_objects[idx], Aspec.Aspec):
                    pya_objects[idx].plot()
                elif isinstance(pya_objects[idx], Astft.Astft):
                    pya_objects[idx].plot(show_bar=False)
                    next_ax = plt.subplot(grid[i + 1])
                    cb = plt.colorbar(pya_objects[idx].im, cax=next_ax)
                elif isinstance(pya_objects[idx], Amfcc.Amfcc):
                    pya_objects[idx].plot(show_bar=False, axis=ax)
                    next_ax = plt.subplot(grid[i + 1])
                    cb = plt.colorbar(pya_objects[idx].im, cax=next_ax)
    return fig