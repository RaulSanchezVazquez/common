#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 23:44:10 2018

@author: raulsanchez
"""

def drop_high_corr(data, corr_th=.95):
    """
    Drops correlated features ath a correlatio threshold.
    The order that features are drop is prioritizing the ones tha have
    the less number of correlated features.
    """

    # Computes correlation
    corr = data.drop(['ID', 'target'], axis=1).corr()

    # finds high corr.
    is_very_corr = corr.apply(
        lambda x: x[x > corr_th].index.tolist(), axis=1)
    is_very_corr = is_very_corr[is_very_corr.apply(len) > 1]

    # Keep the ones with less corr. cnt
    priority_selection = is_very_corr.apply(len).sort_values()

    features_to_keep = set()
    features_to_drop = set()
    for feature in priority_selection.index:

        if not(feature in features_to_drop):
            features_to_keep.add(feature)

        drop_cols = set(is_very_corr.loc[feature])
        drop_cols = drop_cols - features_to_keep

        features_to_drop.update(drop_cols)

    # Drop correlated features
    return data.drop(list(features_to_drop), axis=1)
