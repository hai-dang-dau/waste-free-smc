from programs.common_diagnostics import common_diagnostics, LogLT, MeanFirstMarginal, get_variance_estimators_for_posterior_mean
import math
import numpy as np

diag_quantities = []
diag_quantities.extend(common_diagnostics)

def add_to_dq(something):
    diag_quantities.append(something)
    return something

def log(x: str):
    #tested
    x_str = ''
    for c in x:
        if c.isdigit():
            x_str += c
    x = int(x_str)
    return math.log(x)

log_no_latin_squares = [np.nan, np.log(1), np.log(2), np.log(12), np.log(576), np.log(161280), log('812,851,200'), log('61,479,419,904,000'), log('108,776,032,459,082,956,800'), log('5,524,751,496,156,892,842,531,225,600'), log('9,982,437,658,213,039,871,725,064,756,920,320,000'), log('776,966,836,171,770,144,107,444,346,734,230,682,311,065,600,000')]

@add_to_dq
class _LogLT(LogLT):
    @staticmethod
    def exact(des):
        try:
            return log_no_latin_squares[des.d]
        except IndexError:
            return np.nan

def dts(x):
    """
    :param x: a numpy array of permutation squares
    :return: a numpy array of the Difference of Two Squares at positions [0,0] and [1,0] (i.e. two first squares of the first column)
    """
    res = [sq.columns[0][0] - sq.columns[0][1] for sq in x]
    return np.array(res)

@add_to_dq
class _MeanFirstMarginal(MeanFirstMarginal):
    @staticmethod
    def name() -> str:
        return 'mean_diff_two_squares'

    @staticmethod
    def exact(des):
        return 0

    @staticmethod
    def func():
        return dts

diag_quantities.extend(get_variance_estimators_for_posterior_mean('var_mean_diff_two_squares', dts))