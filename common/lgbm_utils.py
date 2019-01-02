#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 10:12:13 2018

@author: Raul Sanchez-Vazquez
"""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from common.learning_utils import get_n_rand_params
from common.eval_utils import classification_report
from common import parallel

# Parameter distribution for random-search on binary classification task
params_dist_regression = {
    "num_leaves": [4, 8, 12, 16, 25, 32, 50, 64, 90, 100, 128, 256, 400, 512],
    "max_depth": [-1, 2, 4, 6, 8, 12, 16, 24, 32, 50, 64],
    "learning_rate": [0.01, 0.015, 0.025, 0.05, 0.1, 0.2],
    "max_bin": [8, 16, 32, 64, 128, 180, 256],
    "subsample_for_bin": [12000, 25000, 50000, 100000],
    'metric': ['l2', 'l1', 'mape'],
    'objective': ['regression', 'regression_l1', 'mape'],
    "min_child_weight": [2, 3, 5, 7, 11],
    "min_child_samples": [3, 5, 7, 10, 11, 12, 13, 14, 15],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "subsample_freq": [0, 1, 3, 5, 6, 7, 8, 9, 1013, 15],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0.0, 0.01, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 15],
    "reg_alpha": [0.0, 0.01, 0.25, 0.5, .65, 0.75, 1, 1.25, 1.5, 2, 3, 5, 7],
    "verbose":[-1]
}

# Parameter distribution for random-search on binary classification task
params_dist_binary = {
    "num_leaves": [4, 8, 12, 16, 25, 32, 50, 64, 90, 100, 128, 256, 400, 512],
    "max_depth": [-1, 2, 4, 6, 8, 12, 16, 24, 32, 50, 64],
    "learning_rate": [0.01, 0.015, 0.025, 0.05, 0.1, 0.2],
    "max_bin": [8, 16, 32, 64, 128, 180, 256],
    "subsample_for_bin": [12000, 25000, 50000, 100000],
    'metric': ['auc', 'binary_logloss'],
    'objective': ['binary'],
    "min_child_weight": [2, 3, 5, 7, 11],
    "min_child_samples": [3, 5, 7, 10, 11, 12, 13, 14, 15],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "subsample_freq": [0, 1, 3, 5, 6, 7, 8, 9, 1013, 15],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0.0, 0.01, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 15],
    "reg_alpha": [0.0, 0.01, 0.25, 0.5, .65, 0.75, 1, 1.25, 1.5, 2, 3, 5, 7],
    "verbose":[-1]
}


global booster_
def get_single_pred_leaf_outputs(params):
    """
    Retrieve a leaf outputs given tree_id and leaf_id.
    This is a helper function used in get_leaf_outputs()

    Parameters
    ----------
    params : tuple
        Tuple containing (tree_id, leaf_id)

    Return
    ------
    leaf_weight : float
        The corresponding leaf weigth
    """

    tree_id, leaf_id = params

    leaf_weight = booster_.get_leaf_output(
        tree_id=tree_id,
        leaf_id=leaf_id)

    return leaf_weight


def get_leaf_outputs(model, X):
    """
    Retrieves leaf outputs of a LGBM booster given a leaf-ids array

    Parameters
    ----------
    booster : lightgbm.basic.Booster
        LGBM booster model
    leaf_ids : np.array
        Leaf indexes obtained with model.predict(X, pred_leaf=True)
    Return
    ------
        leaf_outputs : np.array

    Example
    --------
        # model is a fitted lgbm.LGBMClassifier()
        booster = model.booster_
        leaf_ids = model.predict(X, pred_leaf=True)
        get_leaf_outputs(booster, leaf_ids)

        ground_truth = model.predict_proba(X)
        y_pred_hand = get_predictions_from_ouput_leafs(leaf_output)
    """
    global booster_
    booster_ = model.booster_
    leaf_index = model.predict(X, pred_leaf=True)

    n_trees = booster_.current_iteration()
    n_rows = X.shape[0]

    tree_index = np.arange(0, n_trees).reshape(1, -1).repeat(n_rows, 0)

    tree_leaf_idx = np.vstack([
        tree_index.flatten(),
        leaf_index.flatten()
    ]).T

    leaf_output = parallel.apply(
        get_single_pred_leaf_outputs,
        tree_leaf_idx,
        n_jobs=32)

    leaf_output = np.array(
        leaf_output
    ).reshape(n_rows, n_trees)

    return leaf_output

def get_predictions_from_ouput_leafs(leaf_outputs):
    """
    Computes predictions from raw outputs of the lgbm
    gradient boosting.

    Paramaters
    ----------
    leaf_outputs: np.array
        Array with shapes: (n_rows x leaf_outputs) for which to
        compute {0, 1} class probability.

    Return
    ------
    predic_proba : np.array
        Class 1 probability

    Example
    -------


    """
    predic_proba = 1 / ( 1 + np.exp(-1 * leaf_outputs.sum(1)))

    return predic_proba
