import pandas as pd
from programs.binary_regression.data_raw.data_preprocessing import standardize
import numpy as np

if __name__ == '__main__':
    x = pd.read_csv('./programs/binary_regression/data_raw/musk1.data', header=None)
    x = x.drop(labels=[0, 1], axis=1)
    x.columns = ['f' + str(i) for i in range(1, 167)] + ['class']
    x = standardize(df=x, catvars=[], tarvar='class')
    np.save(file='./programs/binary_regression/data/musk1.npy', arr=x)
