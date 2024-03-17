import os, re, sys, json, argparse, time, glob, pickle, copy, h5py, getopt

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Add other necessary imports
import tensorflow as tf
from tcn import tcn
from model_functions import temporal_unidirectional

def generate_combinations(brain_area, key1, key2):
    combinations = []
    for i in brain_area[key1]:
        for j in brain_area[key2]:
            combinations.append((i, j))
    return combinations

def main(noise, dropout_rate_tcn, dropout_rate_hidden, tol_default, seqs_len, 
         encoder_type, batch_size, epochs, scaling_method, window_len, 
         version):

    start = time.time()
    
    models_folder = '/scratch/jcalde2/Causation/RealDataExamples/MonkeyBrain/'

    # Load Data    
    filename = models_folder + 'Data/MainSession/Clean_data.npy'
    variables = np.load(filename)
        # filename = models_folder + 'Data/MainSession/Bad_channels.npy'
    # bad_chan = np.load(filename)
    # filename = models_folder + 'Data/MainSession/condition_index.npy'
    # condition_index = np.load(filename)
    # filename = models_folder + 'Data/MainSession/condition_label.pkl'
    # f = open(filename,"rb")
    # condition_label = pickle.load(f)
    # f.close()
    filename = models_folder + 'Data/MainSession/brain_area_updated.pkl'
    with open(filename, "rb") as f:
        brain_area = pickle.load(f)

    brain_area_keys = list(brain_area.keys())

    for key1 in brain_area_keys:
        for key2 in brain_area_keys:  # Avoid repeating pairs
            combinations = generate_combinations(brain_area, key1, key2)
            for vv1, vv2 in combinations:
                current_variables = variables[:, [vv1, vv2]]
                
                var1 = 0
                var2 = 1
                length_vars = len(current_variables)
                discard = 0

                directory = models_folder + 'Results/MainSession/' + key1 + '/' + key2 + '/' + \
                            'chan_' + str(vv1) + '/' + 'chan_' + str(vv2) + '/'

                temporal_unidirectional(directory, current_variables, var1, var2, noise, dropout_rate_tcn, dropout_rate_hidden,
                            tol_default, length_vars, discard, encoder_type, batch_size, epochs, scaling_method, 
                            seqs_len, window_len, version)

    end = time.time()
    print('\n\n Elapsed Time : ' + str(end - start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model with parameters")
    # Update arguments
    parser.add_argument("--noise", type=float, default=0.0, help="Noise Neural Network")
    parser.add_argument("--dropout_rate_tcn", type=float, default=0.0, help="TCN dropout rate")
    parser.add_argument("--dropout_rate_hidden", type=float, default=0.0, help="Hidden layer dropout rate")
    parser.add_argument("--tol_default", type=float, default=0.01, help="Default tolerance")
    parser.add_argument("--seqs_len", type=int, default=10, help="Length of Sequences")
    parser.add_argument("--encoder_type", type=int, default=2, help="Type of encoder")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs for training")
    parser.add_argument("--scaling_method", type=str, default='Standard', help="Scaling method for data")
    parser.add_argument("--window_len", type=int, default=100, help="Length of Windows")
    parser.add_argument("--version", type=int, default=None, help="Version number of the model")

    args = parser.parse_args()
    print(vars(args))

    main(args.noise, args.dropout_rate_tcn, args.dropout_rate_hidden, args.tol_default, 
         args.seqs_len, args.encoder_type, args.batch_size, args.epochs, args.scaling_method, 
         args.window_len, args.version)
