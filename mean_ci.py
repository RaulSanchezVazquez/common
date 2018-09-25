#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:10:50 2018

@author: lsanchez
"""

import numpy as np
import scipy.stats as stats

def mean_conf_interv(measurements, alpha=0.95):
    """
    """

    res = stats.t.interval(
        alpha,
        len(measurements) - 1,
        loc=np.mean(measurements),
        scale=stats.sem(measurements))

    return res