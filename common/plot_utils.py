#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 11:50:31 2018

@author: Raul Sanchez-Vazquez
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def add_bar_values(ax, fontsize=10, kind=None, rotate=None):
    if kind is None:
        kind = 'bar'

    if (rotate is None) and (kind == 'bar'):
        rotate = 45
    else:
        rotate = 0

    for bar in ax.patches:
        height = bar.get_height()
        width = bar.get_width()

        if kind == 'bar':
            x = bar.get_x() + (width / 2)
            y = height
            text = str(height)
        else:
            x = width
            y = bar.get_y() + (bar.get_height() / 2)
            text = str(width)

        ax.text(
            x=x, y=y,
            s=text,
            fontsize=10,
            rotation=rotate,
            horizontalalignment='left',
            verticalalignment='bottom'
            )
    return ax

def make_colormap(seq):
    """
    Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).

    Original code:
    https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale

    Parameters
    ----------
        seq : list

    Returns
    -------
        cmap : matplotlib.colors

    Example:
    --------
        transparent = (1., 1., 1., 0.)
        transparent_red = (1., .0, .0, .5)
        solid_red = (1., .0, .0, .1)

        seq = [
           transparent,
           transparent_red,
           0.66,
           transparent_red,
           solid_red,
           1.,
           solid_red
        ]

        rvb = make_colormap(seq)
        N = 1000
        array_dg = np.random.uniform(0, 10, size=(N, 2))
        colors = np.random.uniform(-2, 2, size=(N,))
        plt.scatter(array_dg[:, 0], array_dg[:, 1], c=colors, cmap=rvb)
        plt.colorbar()
        plt.show()

    """

    seq = [(None,) * 4, 0.0] + list(seq) + [1.0, (None,) * 4]
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha':[]}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1, alpha1 = seq[i - 1]
            r2, g2, b2, alpha2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
            cdict['alpha'].append([item, alpha1, alpha2])
    cmap = mcolors.LinearSegmentedColormap('CustomMap', cdict)

    return cmap
