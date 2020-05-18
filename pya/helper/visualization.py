from __future__ import absolute_import
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.gridspec as grd
from mpl_toolkits.axes_grid1 import make_axes_locatable


def basicplot(data, ticks, channels, offset=0, scale=1,
              cn=None, ax=None, typ='plot', cmap='inferno',
              xlim=None, ylim=None, xlabel='', ylabel='',
              show_bar=False,
              **kwargs):
    """Basic version of the plot for pya, this can be directly used
    by Asig. Aspec/Astft/Amfcc will have different extra setting
    and type. 

    Parameters
    ----------
        data : numpy.ndarray
            data array
        channels : int
            number of channels
        axis : matplotlib.axes, optional
            Plot image on the matplotlib axis if it was given.
            Default is None, which use plt.gca()
        typ : str, optional
            Plot type.

    """
    ax = plt.gca() or ax
    if channels == 1 or (offset == 0 and scale == 1):
        # if mono signal or you would like to stack signals together
        # offset is the spacing between channel,
        # scale is can shrink the signal just for visualization purpose.
        # Plot everything on top of each other.
        if typ == 'plot':
            p = ax.plot(ticks, data, **kwargs)
            # return p, ax
        elif typ == 'spectrogram':
            # ticks is (times, freqs)
            p = ax.pcolormesh(ticks[0], ticks[1], data,
                              cmap=plt.get_cmap(cmap), **kwargs)
        elif typ == 'mfcc':
            p = ax.pcolormesh(data, cmap=plt.get_cmap(cmap), **kwargs)
    else:
        if typ == 'plot': 
            for idx, val in enumerate(data.T):
                p = ax.plot(ticks, idx * offset + val * scale, **kwargs)
                ax.set_xlabel(xlabel)
                if cn:
                    ax.text(0, (idx + 0.1) * offset, cn[idx])
        elif typ == 'spectrogram':
            for idx in range(data.shape[1]):
                p = ax.pcolormesh(ticks[0], idx * offset + scale * ticks[1], 
                                  data[:, idx, :], cmap=plt.get_cmap(cmap),
                                  **kwargs)
                if cn:
                    ax.text(0, (idx + 0.1) * offset, cn[idx])
            ax.set_yticklabels([])
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim([xlim[0], xlim[1]])
    if ylim:
        ax.set_ylim([ylim[0], ylim[1]])
    # Colorbar
    if show_bar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size="2%", pad=0.03)
        _ = plt.colorbar(p, cax=cax)  # Add
    return p, ax


def gridplot(pya_objects, colwrap=1, cbar_ratio=0.04, figsize=None):
    """Create a grid plot of pya objects which have plot() methods,
    i.e. Asig, Aspec, Astft, Amfcc.
    It takes a list of pya_objects and plot each object into a grid.
    You can mix different types of plots
    together.

    Examples
    --------
    # plot all 4 different pya objects in 1 column,
    amfcc and astft use pcolormesh so colorbar will
    # be displayed as well
    gridplot([asig, amfcc, aspec, astft], colwrap=2, 
              cbar_ratio=0.08, figsize=[10, 10]);

    Parameters
    ----------
    pya_objects : iterable object
        A list of pya objects with the plot() method.
    colwrap : int, optional
        Wrap column at position.
        Can be considered as the column size. Default is 1, meaning 1 column.
    cbar_ratio : float, optional
        For each column create another column reserved for the colorbar.
        This is the ratio of the width relative to the plot.
        0.04 means 4% of the width of the data plot.
    figsize : tuple, optional
        width, height of the entire image in inches. Default size is (6.4, 4.8)

    Returns
    -------
    fig : plt.figure()
        The plt.figure() object
    """
    from .. import Asig, Amfcc, Astft, Aspec
    nplots = len(pya_objects)

    if colwrap > nplots:
        ncol = colwrap
    elif colwrap < 1:
        raise ValueError("col_wrap needs to an integer > 0")
    else:
        ncol = colwrap

    nrow = math.ceil(nplots / ncol)
    ncol = ncol * 2  # Double the col for colorbars.

    even_weight = 100
    odd_weight = even_weight * cbar_ratio
    wratio = []  # Aspect ratio of width
    for i in range(ncol):
        wratio.append(odd_weight) if i % 2 else wratio.append(even_weight)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = grd.GridSpec(nrow, ncol,
                        figure=fig, width_ratios=wratio, wspace=0.01)

    total_idx = ncol * nrow
    for i in range(total_idx):
        if not i % 2:  # The data plot is always at even index.
            idx = i // 2  # Index in the pya_objects list
            if idx < nplots:
                ax = plt.subplot(grid[i])
                # Title is object type + label
                title = pya_objects[idx].__repr__().split('(')[0] + ': ' + pya_objects[idx].label
                # Truncate if str too long
                title = (title[:30] + "..." if len(title) > 30 else title)
                ax.set_title(title)
                if isinstance(pya_objects[idx], Asig):
                    pya_objects[idx].plot(ax=ax)
                elif isinstance(pya_objects[idx], Aspec):
                    pya_objects[idx].plot(ax=ax)
                elif isinstance(pya_objects[idx], Astft):
                    pya_objects[idx].plot(show_bar=False, ax=ax)
                    next_ax = plt.subplot(grid[i + 1])
                    _ = plt.colorbar(pya_objects[idx].im, cax=next_ax)
                elif isinstance(pya_objects[idx], Amfcc):
                    pya_objects[idx].plot(show_bar=False, ax=ax)
                    next_ax = plt.subplot(grid[i + 1])
                    _ = plt.colorbar(pya_objects[idx].im, cax=next_ax)
    return fig
