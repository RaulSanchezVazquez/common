#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 11:50:31 2018

@author: Raul Sanchez-Vazquez
"""

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
