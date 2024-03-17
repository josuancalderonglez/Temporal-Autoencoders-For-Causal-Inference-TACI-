import os
import sys
import numpy as np
import glob
import pickle
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
import copy
# from scipy import signal
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from pandas import DataFrame, concat
from sklearn.model_selection import train_test_split

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| Iteration: {iteration}/{total} Percentage: {percent}%  {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

def toc(tempBool=True): # This will be the main function through which we define both tic() and toc()
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        elapsed_time = "Elapsed time: " + str(tempTimeInterval) + " seconds"
        print(elapsed_time)
        return None
    else:
        return None

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def calculate_acf(max_lag=1000, paint=False):
    
    corr = np.empty((variables.shape[1], max_lag))

    print('Autocorrelation Time:')

    for ii in range(variables.shape[1]):
        for jj in range(max_lag):

            if jj == 0:
                corr[ii, jj] = 1
            else:
                corr[ii, jj] = np.corrcoef(variables[jj:, ii], pos_data[:-jj, ii])[0][1]
    
    if paint:
        plt.figure(figsize=(20,5))
        for ii in range(variables.shape[1]):
            plt.plot(corr[ii])
        plt.plot(np.zeros((max_lag)), '-k')
        plt.xlabel('Lags')
        plt.title('Autocorrelation Time')
        plt.show()
    
    return corr

def create_time_lagged_series(data, n_in, n_out, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def create_and_reshape_lagged_series(data, seq_length, shift):
    """Helper function to create a time-lagged series and reshape it."""
    raw_df = create_time_lagged_series(data.reshape(-1, 1), seq_length, 1, dropnan=True)
    reshaped_data = np.array(raw_df)[:, :-1]
    reshaped_data = np.expand_dims(reshaped_data, axis=-1)
    return reshaped_data[::shift]

def create_datasets(dataset, scaling_method='None', seq_length=1, shift=1, subsample=1, test_size=0.2, validation_size=0.1):
    
    """
    Create datasets by scaling, creating a time-lagged series, and splitting into train, validation, and test sets.

    Parameters:
    - dataset: Input dataset.
    - scaling_method: Method for scaling data.
    - seq_length: Sequence length for time-lagged series.
    - shift: Shift value for subsampling.
    - subsample: Rate for subsampling.
    - test_size: Proportion of dataset to include in the test split.
    - validation_size: Proportion of dataset to include in the validation split.

    Returns:
    - trainX, valX, testX: Training, validation, and test datasets.
    """
    
    # Scaling the dataset based on the provided method
    scalers = {
        'MinMax': MinMaxScaler(feature_range=(0, 1)),
        'Standard': StandardScaler(),
        'Robust': RobustScaler(),
        'None': None
    }
    scaler = scalers.get(scaling_method)
    if scaler:
        dataset_scaled = scaler.fit_transform(dataset.reshape(-1, 1))
    else:
        dataset_scaled = dataset.reshape(-1, 1)

    # Create and reshape time-lagged series for dataset and index
    full_data = create_and_reshape_lagged_series(dataset_scaled, seq_length, shift)
    full_data_ind = create_and_reshape_lagged_series(np.arange(len(dataset)).reshape(-1, 1), seq_length, shift)

    dataset_subsampled = full_data.copy()
    indx = full_data_ind.copy()

    # Apply subsampling
    if subsample > 1:
        dataset_subsampled = dataset_subsampled[::subsample]
        indx = indx[::subsample]

    # Splitting dataset into training and test sets
    if len(dataset) <= 100000:
        test_size_ = 0.3
    elif 100000 < len(dataset) <= 1000000:
        test_size_ = 0.2
    else:  # dataset_subsampled > 1000000
        test_size_ = 0.02

    X_train, X_temp, y_train, y_temp = train_test_split(dataset_subsampled, dataset_subsampled, test_size=test_size_, random_state=42, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, X_temp, test_size=0.5, random_state=42, shuffle=True)

    return X_train, X_val, X_test, full_data, indx

def generate_fourier_surrogate(data):
    
    ts = data.copy()
    ts_fourier = np.fft.rfft(ts)
    random_phases = np.exp(np.random.uniform(0, np.pi, len(ts) // 2 + 1) * 1.0j)
    ts_fourier_new = ts_fourier * random_phases
    surrogate = np.fft.irfft(ts_fourier_new, len(ts))

    return surrogate

def relative_directionality(scores_x, scores_y):

    if (scores_x)<0:
        scores_x = 0
    if (scores_y)<0:
        scores_y = 0

    if (scores_x + scores_y) == 0:
        directionality = 0
    else:
        directionality = (scores_y-scores_x)/(np.abs(scores_x+scores_y))

    return directionality