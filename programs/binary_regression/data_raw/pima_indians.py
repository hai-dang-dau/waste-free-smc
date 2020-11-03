import pandas as pd
from programs.binary_regression.data_raw.data_preprocessing import standardize
import numpy as np

if __name__ == '__main__':
    df = pd.read_csv('./programs/binary_regression/data_raw/pima_indians.csv')
    df = standardize(df=df, catvars=[], tarvar='Outcome')
    np.save(file='./programs/binary_regression/data/pima_indians.npy', arr=df)