import os, re, sys, json, argparse, time, glob, pickle, copy, h5py, getopt

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Add other necessary imports
import tensorflow as tf
from tcn import tcn
from model_functions import temporal_unidirectional, temporal_bidirectional

def main(noise, dropout_rate_tcn, dropout_rate_hidden, tol_default, seqs_len, 
         encoder_type, batch_size, epochs, scaling_method, array_index, version):

    start = time.time()

    voles_names = ['PV20130829B_fastestHud1', 'PV20130125_fastestHud2', 'PV20130322A_fastestHud3',
            'PV20151006A_fastestHud4', 'PV20140405A_fastestHud5', 'PV20150919B_fastestHud6',
            'PV20130929B_fastestHud7', 'PV20150920_fastestHud8', 'PV20140405B_fastestHud9']
    variables_names = ['PFCdata', 'NAccdata']
    variables_types = ['matingSampleIDs', 'selfgroomingSampleIDs', 'huddlingSampleIDs']

    models_folder = '/scratch/jcalde2/Causation/RealDataExamples/Voles/'
        
    # Load Data    
    filename = models_folder + 'Data/MAT_Data/2_26_18_night.mat'
    mat_contents = loadmat(filename)
    name = voles_names[array_index]
    
    var1 = 0
    var2 = 1
    variable1 = mat_contents[name][variables_names[var1]][0,0][0]
    variable2 = mat_contents[name][variables_names[var2]][0,0][0]
    variables = np.vstack((variable1, variable2)).T
    
    length_vars = len(variables)
    discard = 0

    directory = models_folder + 'Results/General/Vole_' + str(array_index+1) + '/TCNI/Analysis/'

    window_lens = [500, 800, 1000, 1300, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    
    for window_len in window_lens:
        temporal_bidirectional(directory, variables, var1, var2, noise, dropout_rate_tcn, dropout_rate_hidden,
                    tol_default, length_vars, discard, encoder_type, batch_size, epochs, scaling_method, 
                    seqs_len, window_len, version)
            
    end = time.time()
    print('\n\n Elapsed Time : ' + str(end - start))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run model with parameters")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise Neural Network")
    parser.add_argument("--dropout_rate_tcn", type=float, default=0.0, help="TCN dropout rate")
    parser.add_argument("--dropout_rate_hidden", type=float, default=0.0, help="Hidden layer dropout rate")
    parser.add_argument("--tol_default", type=float, default=0.01, help="Default tolerance")
    parser.add_argument("--seqs_len", type=int, default=50, help="Lengh of Sequences")
    parser.add_argument("--encoder_type", type=int, default=2, help="Type of encoder")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs for training")
    parser.add_argument("--scaling_method", type=str, default='Standard', help="Scaling method for data")
    parser.add_argument("--array_index", type=int, required=True, help="Index for internal values combination")

    parser.add_argument("--version", type=int, default=None, help="Version number of the model")
                            
    args = parser.parse_args()
    print(vars(args))  
    sys.stdout.flush()  

    main(args.noise, args.dropout_rate_tcn, args.dropout_rate_hidden, args.tol_default, 
         args.seqs_len, args.encoder_type, args.batch_size, args.epochs, args.scaling_method, 
         args.array_index, args.version)