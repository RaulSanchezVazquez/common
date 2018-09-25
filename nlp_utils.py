#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 10:34:57 2018

@author: lsanchez
"""


from nltk.corpus import stopwords
import pandas as pd
import unidecode

from common import parallel


def spanish_remove_stop_words(sentence, forbidden_words=[]):
    """
    Parameters
    ----------
        sentence : str

    Return
    ------
        valid_words : list

    """

    forbidden_words += stopwords.words('spanish')
    forbidden_words += ['']

    valid_words = filter(
        lambda x: x not in forbidden_words,
        sentence.split(' '))

    valid_words = list(valid_words)

    return valid_words


def spanish_basic_sentence_cleanning(sentences, forbidden_words):
    """
    Cleans a pandas.Series containing spanish sentences:
        - Lowercase
        - Remove accents
        - Remove Stopwords

    Parameters
    ----------
        sentences : pandas.Series
            Column containing the sentences

    Return
    ------
        clean_sentences : pandas.Series
            Column with the cleant sentences
    """

    sentences = sentences.copy()

    # Lower and remove accents
    sentences = sentences.astype(str).apply(
        lambda x: unidecode.unidecode(x.lower()))

    # Remove Stop Words
    sentences = pd.Series(
        parallel.apply(
            spanish_remove_stop_words,
            sentences,
            n_jobs=8),
        index=sentences.index)

    return sentences
