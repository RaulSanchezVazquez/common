#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:41:20 2018

@author: raulsanchez
"""

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from . import parallel


def dummy_dataframe(df):
    '''Returns a dataframe with one line and the same columns as df.
    All values on the dummy rows are np.nan when possible.
    More specifically, default values are as follow:
        np.nan
            When it is a float
        0
            When it is an int
        -1
            Default categorical code for pandas categorical data,
            the resulting dummy columns has all categories on it
            as the original pandas.DataFrame: df
        NaT
            When it is datetime
    Parameters
    ----------
    df : pandas.DataFrame
        pandas.DataFrame for whom to copy of column-names and dtypes

    Return
    ------
    df_dummy : pandas.DataFrame
        Dummy copy of the original df with the same dtypes
    '''

    df_dummy = pd.DataFrame(
        [[0] * df.shape[1]],
        columns=df.columns)

    for c in df.columns:
        df_dummy[c] = df_dummy[c].astype(str(df[c].dtype))

    # Copy category
    for c in df.select_dtypes('category').columns:
        categorical_dtype = CategoricalDtype(categories=df[c].cat.categories)
        df_dummy[c] = np.nan
        df_dummy[c] = df_dummy[c].astype(categorical_dtype)

    # Set datetimes
    for c in df.select_dtypes('datetime64[ns]').columns:
        df_dummy[c] = np.nan
        df_dummy[c] = pd.to_datetime(df_dummy[c])

    df.dtypes.value_counts()
    df_dummy.dtypes.value_counts()

    return df_dummy


def flatten_dataframe(df, separator="__"):
    """
    Flattens a pandas.DataFrame.
    Columns names of the flattered-DF are
    compused of the indexes and columns of the
    original-DF

    params:
    df          pandas.DataFrame

    returns:

    flatt_df    pandas.DataFrame
    """

    categ_cols = df.select_dtypes('category').columns
    new_cols = []
    new_cols_categ = dict([(x, []) for x in categ_cols])
    for row in df.index:
        for col in df.columns:
            new_c = "%s%s%s" % (col, separator, row)
            new_cols.append(new_c)
            if col in categ_cols:
                new_cols_categ[col].append(new_c)

    flatt_df = pd.DataFrame(
        [df.values.flatten()],
        columns=new_cols)

    for c, c_news in new_cols_categ.items():
        categorical_dtype = CategoricalDtype(categories=df[c].cat.categories)
        for c_n in c_news:
            flatt_df[c_n] = flatt_df[c_n].astype(categorical_dtype)

    return flatt_df


def clean_df(data, null_threshold=.95):
    """
    Basic cleanning operations on a pandas.DataFrame.
    """

    # DROP features with null
    null_count = data.isnull().sum()
    null_percent = null_count / data.shape[0]

    null_ignore = null_percent[
        null_percent >= null_threshold
    ]

    null_ignore = null_ignore.index.tolist()
    data = data.drop(null_ignore, axis=1)

    # DROP features filled with constant value
    constant_ignore = []
    for c in data.columns:
        nunique = data[c].nunique(dropna=False)
        if nunique == 1:
            constant_ignore.append(c)

    data = data.drop(constant_ignore, axis=1)

    return data


def cat_replace_others(param):
    """
    User for replace cardinal values
    that follow under the 'other' category
    when min. freq. threshold is defined when
    the fancy version of get_dummies
    is used.

    This function is intended to be executed
    in parallel (kind of map-reduce)

    See datautils.get_dummies

    """
    x, cat_others = param

    if x in cat_others:
        return "__other"
    else:
        return x


def get_dummies(
        feature_series,
        cat_th=.01,
        sep_dummy='/',
        sep_f_name='__'):
    """

    Returns a prety version of the
    pandas.get_dummies


    cat_th      min. threshold to consider to add the cat.
                to the one-hot. The threshold can be
                between (0, 1) in order to set a min
                percentage of sparsity or a plain int number
                on the range [1, #number_of_data_instances]

    sep_dummy   a custom sep. for binary cat.
    sep_f_name  a custom sep. for the feature-category

    """

    feature_index = feature_series.index
    feature_name = feature_series.name
    feature_vc = feature_series.value_counts(dropna=True)

    # Min. Freq. threshold
    if cat_th < 1:
        cat_th = int(cat_th * feature_series.shape[0])

    # Find low freq. cat. and replace for __other new cat.
    cat_others = feature_vc[feature_vc < cat_th].index.tolist()

    data_map = [(x, cat_others) for x in feature_series.tolist()]
    feature_series = parallel.apply(
        cat_replace_others,
        data_map,
        n_jobs=4)
    feature_series = pd.Series(feature_series)

    # Value count once more
    feature_vc = feature_series.value_counts()

    # Get and format dummy
    if len(feature_vc) <= 2:
        most_pop = feature_vc.index[0]

        # Get dummy
        dummies = (feature_series == most_pop).astype(int)
        dummies = pd.DataFrame(dummies)

        # Rename feature names
        if len(feature_vc) == 1:
            dummy_columns = "%s%s%s" % (
                feature_name,
                sep_f_name,
                "is_not_nan")
        else:
            dummy_columns = "%s%s%s" % (
                feature_name,
                sep_f_name,
                sep_dummy.join(feature_vc.index))

        dummies.columns = [dummy_columns]
    else:
        # Get dummy
        dummies = pd.get_dummies(feature_series)

        # Rename feature names
        dummy_columns = []
        for x in dummies.columns:
            dummy_columns.append(
                "%s%s%s" % (feature_name, sep_f_name, x))
        dummies.columns = dummy_columns

    # Reassign indexes
    dummies.index = feature_index

    # Correct data types (memory efficiency)
    for c in dummies.columns:
        dummies[c] = dummies[c].astype('int8')

    return dummies


def pd_read_csv_helper(params):
    """
    Helper function to read in parallel
    multiple csvs. It supports to ignore
    features and can be silenced.
    """
    try:
        if isinstance(params, str):
            params = {'data_path': params}

        if 'verbose' not in params:
            params['verbose'] = False

        if 'ignore' not in params:
            params['ignore'] = []

        if params['verbose']:
            print("Reading: %s" % params['data_path'])

        data = pd.read_csv(
            params['data_path'],
            compression='bz2',
            low_memory=False)

        if 'usecols' in params:
            usecols = params['usecols']
        else:
            usecols = list(set(data.columns) - set(params['ignore']))

        return data[usecols]
    except BaseException:
        print("ERROR!!")
        print(params)

        return None


def get_raw(dataset_path, data_label, data_sources):
    """
    Vertically concats csvs

    Saves to disk as hdf and returns pandas.DataFrame
    with the files concatenated.
    """
    print("[%s files] Merging..." % len(data_sources))

    dataset = pd.concat(
        parallel.apply(
            pd_read_csv_helper,
            data_sources,
            n_jobs=4
        ),
        axis=0)

    dataset.to_hdf(
        dataset_path,
        key=data_label)

    return dataset