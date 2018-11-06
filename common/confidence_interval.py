#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:10:50 2018

@author: Raul Sanchez-Vazquez
"""

import numpy as np
from scipy import stats
import scipy.stats as stats
from scipy.stats import sem
from sklearn.metrics import roc_auc_score

from . import compare_auc_delong_xu

def mean(measurements, alpha=0.95):
    """
    """

    res = stats.t.interval(
        alpha,
        len(measurements) - 1,
        loc=np.mean(measurements),
        scale=stats.sem(measurements))

    return res

def AUC(y_true, y_pred, n_bootstraps=2000, bootstrap_seed=42):
    """
    Computes AUC with its respective confidence interval via boostrap.
    Original code:

    https://stackoverflow.com/questions/19124239/
    scikit-learn-roc-curve-with-confidence-intervals

    """

    auc = roc_auc_score(y_true, y_pred)
    rng = np.random.RandomState(bootstrap_seed)

    bootstrapped_scores = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    if auc < .5:
        auc = 1 - auc
        confidence_lower = 1 - confidence_lower
        confidence_upper = 1 - confidence_upper

        return auc, confidence_upper, confidence_lower

    return auc, confidence_lower, confidence_upper

def AUC_DeLong_Xu(y_true, y_score, alpha=0.95):
    """
    Computes the AUC and its respective confidence Interval.
    
    y_true : np.array
        np.Array containing the binary target
    y_score : np.array
        np.Array containing the prediction scores
    alpha : float
        Confidence level for the confidence interval

    Return
    ------
        auc : float
            Area Under the ROC Curve
        ci : tuple
            Confidence Interval of the AUC estimated via DeLong

    """

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    if y_score.shape[0] == 2:
        y_score = y_score[:, 1]

    auc, cov = compare_auc_delong_xu.delong_roc_variance(
        ground_truth=y_true,
        predictions=y_score,
        sample_weight=None)

    auc_std = np.sqrt(cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

    ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    return auc, ci
