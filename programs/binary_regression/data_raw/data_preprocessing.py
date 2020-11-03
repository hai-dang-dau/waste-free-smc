import pandas as pd
from typing import List
import numpy as np

toy = pd.DataFrame({'hi': [0, 1, 0, 0], 'ho': ['cat', 'dog', 'others', 'dog'], 'he': [9, 2, 8, 5]}) # toy dataframe, used for testing

def standardize(df: pd.DataFrame, catvars: List[str], tarvar: str):
    # tested when there is no missing data, no categorical variable and no intercept.
    """
    Standardize a dataframe for binary regression,
    using the recommendation in Chopin's Pima Indians article.
    :param catvars: list of categorical variables
    :param tarvar: name of target variable
    :return numpy array of format [target, intercept, binary predictors, non-binary predictors]
    """
    target = process_target(df[tarvar].values)
    df = df.drop(tarvar, axis=1)
    intercept = np.zeros(shape=len(df)) + 1
    df = remove_intercept(df)
    #todo: process categorical variables
    non_binaries = mean_zero_std_half(df.values)
    return np.c_[target, intercept, non_binaries]

def process_target(x):
    #tested
    """
    Given an array x of two values, return a corresponding [-1,1] array.
    """
    y = np.zeros(shape=x.shape, dtype=int) + 1
    negative = np.where(x == x[0])
    y[negative] = -1
    return y

def remove_intercept(df):
    #todo
    return df

def mean_zero_std_half(x):
    #tested
    """
    :param x: a numpy 2D array
    :return: a new numpy array where each column has mean 0 and std 0.5
    """
    return (x - x.mean(axis=0))/x.std(axis=0) * 0.5