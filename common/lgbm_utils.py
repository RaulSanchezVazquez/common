#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 10:12:13 2018

@author: lsanchez
"""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from common.learning_utils import get_n_rand_params
from common.eval_utils import classification_report

# Random Grid-Search
GS_params_regression = {
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
    "reg_alpha": [0.0, 0.01, 0.25, 0.5, .65, 0.75, 1, 1.25, 1.5, 2, 3, 5, 7]
}

GS_params_binary = {
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
    "reg_alpha": [0.0, 0.01, 0.25, 0.5, .65, 0.75, 1, 1.25, 1.5, 2, 3, 5, 7]
}

def fit(
    X_train,
    y_train,
    log_path=None,
    iterations=50,
    n_splits=5,
    random_state=42):
    """
    """

    # Obtain random grid-search parameters
    params_gridsearch = get_n_rand_params(
        search_space=GS_params_binary,
        size=iterations)

    ks = KFold(n_splits=n_splits, random_state=random_state)

    # Perform evaluation rounds
    params_gridsearch_results = []
    progress_bar = tqdm(total=iterations)
    for it, params in enumerate(params_gridsearch):
        param_results = []
        start = time.time()
        for train_idx, test_idx in ks.split(X_train):
            dtrain = lgb.Dataset(
                X_train.iloc[train_idx],
                label=y_train.iloc[train_idx])

            dtest = lgb.Dataset(
                X_train.iloc[test_idx],
                label=y_train.iloc[test_idx])

            params['nthread'] = 16

            gbm = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=3000,
                valid_sets=dtest,
                early_stopping_rounds=50,
                verbose_eval=False)

            test_y_score = gbm.predict(
                X_train.iloc[test_idx],
                raw_score=True)
            test_y_pred = (test_y_score < .8).astype(int)

            test_y_score = np.vstack([
                test_y_score * -1,
                test_y_score
            ]).T

            fold_result = classification_report(
                y_true=y_train.iloc[test_idx],
                y_pred=test_y_pred,
                y_score=test_y_score)

            param_results.append(fold_result['AUC'].iloc[-1])

        progress_bar.update(1)
        params['eval_metric_result'] = pd.Series(param_results).mean()
        params['time'] = round(time.time() - start, 2)

        params_gridsearch_results.append(params)
        if not log_path is None:
            pd.DataFrame(
                params_gridsearch_results
            ).to_csv(log_path, index=False)

        print('[%s] %s' % (it + 1, params))
        print('[%s] %s' % (it + 1, params['eval_metric_result']))

    if not log_path is None:
        results = pd.read_csv(log_path)

        results['eval_metric_result'].plot(
            kind='area',
            use_index=False,
            grid=True,
            alpha=.2,
            ylim=(.5, .8))

        best_result = results.sort_values(
            'eval_metric_result'
        ).iloc[[-1]]

        return best_result.iloc[0]
