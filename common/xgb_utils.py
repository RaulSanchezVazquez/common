#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 14:01:42 2018

@author: raulsanchez
"""

import time

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from common.learning_utils import get_n_rand_params
from common.eval_utils import classification_report

# Random Grid-Search
GS_params = {
    "learning_rate": [0.01, 0.015, 0.025, 0.05, 0.1],
    "gamma": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    "max_depth": [3, 5, 7, 9, 12, 15, 17, 25],
    "min_child_weight": [2, 3, 5, 7, 9],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bylevel": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0, 0.01, 0.25, 0.5, 0.75, 1, 2, 3, 5, 7, 9, 11],
    "reg_alpha": [0, 0.01, 0.25, 0.5, 0.75, 1, 2, 3, 5, 7, 9, 11],
}

def fit(
    log_path,
    X_train,
    y_train,
    iterations=50,
    n_splits=5,
    random_state=42):
    """
    """

    # Obtain random grid-search parameters
    params_gridsearch = get_n_rand_params(
        search_space=GS_params,
        size=iterations)

    ks = KFold(n_splits=n_splits, random_state=random_state)

    # Perform evaluation rounds
    params_gridsearch_results = []
    progress_bar = tqdm(total=iterations)
    for it, params in enumerate(params_gridsearch):
        param_results = []
        start = time.time()
        for train_idx, test_idx in ks.split(X_train):
            dtrain = xgb.DMatrix(
                X_train.iloc[train_idx],
                label=y_train.iloc[train_idx])

            dtest = xgb.DMatrix(
                X_train.iloc[test_idx],
                label=y_train.iloc[test_idx])

            evallist = [(dtest, 'dtest')]

            params['nthread'] = 16
            params['objective'] = 'binary:logistic'

            bst = xgb.train(
                params,
                dtrain,
                early_stopping_rounds=3000,
                evals=evallist,
                verbose_eval=False)

            test_y_score = bst.predict(dtest)
            test_y_pred = (test_y_score < .8).astype(int)

            test_y_score = np.vstack([
                1 - test_y_score,
                test_y_score
            ]).T

            fold_result = classification_report(
                y_true=y_train.iloc[test_idx].astype(int),
                y_pred=test_y_pred,
                y_score=test_y_score)

            param_results.append(fold_result['AUC'].iloc[-1])

        params['eval_metric_result'] = pd.Series(param_results).mean()
        params['time'] = round(time.time() - start, 2)

        params_gridsearch_results.append(params)
        pd.DataFrame(params_gridsearch_results).to_csv(log_path, index=False)

        print('[%s] %s' % (it + 1, params))
        print('[%s] %s' % (it + 1, params['eval_metric_result']))
        progress_bar.update(1)

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
