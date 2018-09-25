#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 16:07:39 2018

@author: lsanchez
"""

import numpy as np
import pandas as pd


def expand_date(date_col, drop=True, time_expand=False):
    """
    date_col = pd.Series(pd.to_datetime([
        '3/11/2000',
        '3/12/2000',
        '3/13/2000']))

    date_expand(date_col)

           year  month  week  day  dayofweek  dayofyear is_month_end is_month_start  \
    0  2000      3    10   11          5         71        False          False
    1  2000      3    10   12          6         72        False          False
    2  2000      3    11   13          0         73        False          False

      is_quarter_end is_quarter_start is_year_end is_year_start  \
    0          False            False       False         False
    1          False            False       False         False
    2          False            False       False         False

             elapsed_days
    0  952732800000000000
    1  952819200000000000
    2  952905600000000000
    """

    date_attr = [
        'year',
        'month',
        'week',
        'day',
        'dayofweek',
        'dayofyear',
        'is_month_end',
        'is_month_start',
        'is_quarter_end',
        'is_quarter_start',
        'is_year_end',
        'is_year_start']

    if time_expand:
        date_attr += ['Hour', 'Minute', 'Second']

    expanded_date = []
    for attr in date_attr:
        expanded_date.append(getattr(
            date_col.dt,
            attr.lower()))

    expanded_date = pd.concat(
        expanded_date,
        axis=1)

    expanded_date.columns = date_attr
    expanded_date['elapsed_days'] = date_col.astype(np.int64)

    return expanded_date
