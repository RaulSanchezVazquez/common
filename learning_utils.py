#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 00:03:04 2018

@author: raulsanchez
"""
import numpy as np


def random_choice(GS_params):
    """
    Single random sample of parameters.

    Parameters
    ----------
        GS_params : : dict
            Dictionary of parameters an plaussible values to random sampling:
                {
                    'param_1': [param_val_1, param_val_2, ...],
                    'param_2': [param_val_x, param_val_y, ...]
                }

    Return
    ------
        params : dict
            The random sampling

    """
    params = {}
    for param_name, param_values in GS_params.items():

        n_params = len(param_values)
        param_idx = np.random.choice(range(n_params), size=1)[0]

        params[param_name] = param_values[param_idx]

    return params


def get_n_rand_params(search_space, size=10):
    """
    Parameters
    ----------
    search_space : dict
        Dictionary of parameters an plaussible values to random sampling:
            {
                'param_1': [param_val_1, param_val_2, ...],
                'param_2': [param_val_x, param_val_y, ...]
            }
    size : int
        Number of random samples

    Return
    ------
        param_list : list
            List of unique random sampling
    """

    param_list = []
    for it in range(size):
        params = random_choice(search_space)

        is_repeted = True
        while(is_repeted):
            params = random_choice(search_space)

            is_repeted = False
            for params_ in param_list:
                match_same = 0
                for k, v in params.items():
                    if params_[k] == v:
                        match_same += 1
                if len(params) == match_same:
                    is_repeted = True
        param_list.append(params)

    return param_list
