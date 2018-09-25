#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:34:43 2018

@author: lsanchez
"""

import time

from pyfm import pylibfm
#from fastFM import als

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix
from tqdm import tqdm

from common.learning_utils import get_n_rand_params
from common.eval_utils import classification_report


# Random Grid-Search
GS_params = {
    'num_factors': [10, 30, 50, 100, 150, 200],
    'num_iter': [5], #10, 20, 30, 50],
    'verbose': [False],
    'task': ['classification'],
    'initial_learning_rate': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    'learning_rate_schedule': ['optimal', 'constant', 'invscaling'],
    'power_t': [5e-1, 1e-1, 5e-2, 1e-2],
    't0': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    'init_stdev': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
}

#GS_params = {
#    'n_iter': [2, 5, 10],
#    'init_stdev': [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4],
#    'rank': [2, 5, 10, 20, 30, 50, 100, 150, 300],
#    'l2_reg_w': [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4],
#    'l2_reg_V': [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4],
#}


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
        #cols = list(pd.Series(X_train.columns).sample(100))

        cols = X_train.columns[:100]
        for train_idx, test_idx in ks.split(X_train):

            dtest = csr_matrix(
                X_train[cols].iloc[train_idx].values
            ).astype(float)
            ytest = y_train.iloc[train_idx].values.astype(int)

            dtrain = csr_matrix(
                X_train[cols].iloc[test_idx].values
            ).astype(float)
            ytrain = y_train.iloc[test_idx].values.astype(int)

            model = pylibfm.FM(**params)
            model.fit(dtrain, ytrain)

#            model = als.FMRegression(**params)
#            model.fit(dtrain, ytrain)

            test_y_score = model.predict(dtest)
            test_y_pred = pd.Series(
                test_y_score < np.median(test_y_score)
            ).apply(lambda x: 1 if x else -1)

            test_y_score = np.vstack([
                1 - test_y_score,
                test_y_score
            ]).T

            fold_result = classification_report(
                y_true=ytest,
                y_pred=test_y_pred,
                y_score=test_y_score)

            param_results.append(fold_result['AUC'].iloc[-1])

        params['cols'] = list(cols)
        params['eval_metric_result'] = pd.Series(param_results).mean()

        if params['eval_metric_result'] < .5:
            params['eval_metric_result'] = 1 - params['eval_metric_result']

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