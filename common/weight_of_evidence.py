#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 11:22:06 2018

@author: lsanchez
"""


import numpy as np

class WOES():
    def __init__(
            self,
            min_freq=60,
            verbose=False,
            low_freq_value='WOE_OTHER'):
        """
        """

        self.min_freq = min_freq

        self.categ_features = []
        self.woes = {}

        self.verbose = verbose
        self.low_freq_value = low_freq_value

    def fit(self, X, y):
        """
        """

        self.X = X.copy()
        self.y = y.copy()

        # get categ. features
        self.categ_features = self.X.select_dtypes('object').columns

        self.y_vc = self.y.value_counts()


        self.woes = {}
        for f in self.categ_features:
            if self.verbose:
                print(f)

            woe = self.get_feature_woes(f)

            self.woes[f] = woe


    def get_feature_woes(self, f):
        """
        """

        X_f = self.X[f]
        # Count categ. value frequency
        X_f_vc = X_f.value_counts()

        # Only allow categ. values with a minimum of frequency
        is_valid_woe_categ = (X_f_vc >= self.min_freq)
        value_min_one = X_f_vc[is_valid_woe_categ].index

        # Set alias for low freq. categ. values
        X_f[~X_f.isin(value_min_one)] = self.low_freq_value

        # build helper to slice data by their categ. value
        X_f = X_f.reset_index().set_index(f)
        X_f = X_f[X_f.columns[0]]

        woe = {}
        for unique_val in X_f.index.unique():
            val_idx = X_f.loc[unique_val].tolist()

            # Get local targets
            y_local = self.y.loc[val_idx]

            # Count target frequency
            y_local_vc = y_local.value_counts().reindex([0, 1]).fillna(0)

            # Get percentage
            percent = y_local_vc / self.y_vc

            woe[unique_val] = np.log(percent[0] / percent[1])

        return woe

    def transform(self, X_):
        """
        """
        X_ = X_.copy()
        for f in self.categ_features:
            woe = self.woes[f]
            valid_values = list(woe.keys())

            is_valid_value = X_[f].isin(valid_values)
            X_[f][~is_valid_value] = self.low_freq_value

            X_[f] = X_[f].apply(lambda x: woe[x])

        return X_