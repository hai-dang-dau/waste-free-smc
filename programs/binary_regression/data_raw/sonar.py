import pandas as pd
from programs.binary_regression.data_raw.data_preprocessing import standardize
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('./programs/binary_regression/data_raw/sonar.csv', header=None)
    df.columns = ['col' + str(i) for i in range(60)] + ['out']
    df = standardize(df=df, catvars=[], tarvar='out')
    np.save(file='./programs/binary_regression/data/sonar.npy', arr=df)