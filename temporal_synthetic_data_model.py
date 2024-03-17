import os, re, sys, json, argparse, time, glob, pickle, copy, h5py, getopt

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras import regularizers

# Add other necessary imports
from tcn import tcn
from artificial_data import generate_henon03_henon03_temporal_coupling
from model_functions import temporal_bidirectional

def main(noise, dropout_rate_tcn, dropout_rate_hidden, tol_default, coupling_constant, length_vars, discard, seqs_len, 
         encoder_type, batch_size, epochs, scaling_method, window_len, version):

    start = time.time()

    models_folder = '/scratch/jcalde2/Causation/SyntheticDataExamples/Temporal/'
    var1 = 0
    var2 = 2
    init_cond = [0.7, 0, 0.7, 0]

    # print('########################### ON_OFF_Alternating ###########################')

    # # Generate Variables
    # couplingx = np.zeros((length_vars))
    # t = np.linspace(0, 1, length_vars-discard)
    # pulse_func = signal.square(2 * np.pi * 1.49 * t)
    # pulse_func[pulse_func<=0] = 0
    # pulse_func = np.concatenate((np.ones((discard)), pulse_func))    
    # couplingy = pulse_func*coupling_constant

    # variables = generate_henon03_henon03_temporal_coupling(couplingx, couplingy, 
    #                 length_vars, discard, init_cond)

    # np.random.seed(0)
    # while np.isnan(variables).any():
    #     init_cond_ = np.random.random(4)
    #     variables = generate_henon03_henon03_temporal_coupling(couplingx, couplingy, 
    #                 length_vars, discard, init_cond)
        
    # directory = models_folder + 'ON_OFF_Alternating/Coupling_' + str(coupling_constant) + '/'

    # temporal_bidirectional(directory, variables, var1, var2, noise, dropout_rate_tcn, dropout_rate_hidden,
    #             tol_default, length_vars, discard, encoder_type, batch_size, epochs, scaling_method, 
    #             seqs_len, window_len, version)
            
    # print('########################### ON_OFF_Widths ###########################')

    # # Generate Variables
    # couplingx = np.zeros((length_vars))
    # step = (length_vars-discard)//10
    # couplingy = np.ones((length_vars))

    # couplingy[:] = coupling_constant*0
    # couplingy[1*step:] = coupling_constant*0
    # couplingy[2*step:] = coupling_constant
    # couplingy[3*step:] = coupling_constant*0
    # couplingy[4*step:] = coupling_constant*0
    # couplingy[5*step:] = coupling_constant
    # couplingy[6*step:] = coupling_constant*0
    # couplingy[7*step:] = coupling_constant*0
    # couplingy[8*step:] = coupling_constant
    # couplingy[9*step:] = coupling_constant
    # couplingy[10*step:] = coupling_constant*0

    # variables = generate_henon03_henon03_temporal_coupling(couplingx, couplingy, 
    #                 length_vars, discard, init_cond)

    # np.random.seed(0)
    # while np.isnan(variables).any():
    #     init_cond_ = np.random.random(4)
    #     variables = generate_henon03_henon03_temporal_coupling(couplingx, couplingy, 
    #                 length_vars, discard, init_cond)
        
    # directory = models_folder + 'ON_OFF_Widths/Coupling_' + str(coupling_constant) + '/'

    # temporal_bidirectional(directory, variables, var1, var2, noise, dropout_rate_tcn, dropout_rate_hidden,
    #             tol_default, length_vars, discard, encoder_type, batch_size, epochs, scaling_method, 
    #             seqs_len, window_len, version)
    
    # print('########################### ON_OFF_Flip ###########################')

    # # Generate Variables
    # couplingx = np.ones((length_vars))*coupling_constant
    # couplingx[:((length_vars-discard)//2+discard)] = couplingx[:(length_vars-discard)//2+discard]*0
    # couplingy = np.ones((length_vars))*coupling_constant
    # couplingy[(length_vars-discard)//2+discard:] = couplingy[(length_vars-discard)//2+discard:]*0

    # variables = generate_henon03_henon03_temporal_coupling(couplingx, couplingy, 
    #                 length_vars, discard, init_cond)

    # np.random.seed(0)
    # while np.isnan(variables).any():
    #     init_cond_ = np.random.random(4)
    #     variables = generate_henon03_henon03_temporal_coupling(couplingx, couplingy, 
    #                 length_vars, discard, init_cond)
        
    # directory = models_folder + 'ON_OFF_Flip/Coupling_' + str(coupling_constant) + '/'

    # temporal_bidirectional(directory, variables, var1, var2, noise, dropout_rate_tcn, dropout_rate_hidden,
    #             tol_default, length_vars, discard, encoder_type, batch_size, epochs, scaling_method, 
    #             seqs_len, window_len, version)
    
    print('########################### ON_OFF_Steps ###########################')

    # Generate Variables
    couplingx = np.zeros((length_vars))
    step = (length_vars-discard)//9
    couplingy = np.ones(((length_vars-discard)//2))
    for ii in range(5):
        couplingy[ii*step:] = coupling_constant/5*ii
    couplingy = np.concatenate((np.zeros((discard)), couplingy, couplingy[::-1]))   

    variables = generate_henon03_henon03_temporal_coupling(couplingx, couplingy, 
                    length_vars, discard, init_cond)

    np.random.seed(0)
    while np.isnan(variables).any():
        init_cond_ = np.random.random(4)
        variables = generate_henon03_henon03_temporal_coupling(couplingx, couplingy, 
                    length_vars, discard, init_cond)
        
    directory = models_folder + 'ON_OFF_Steps/Coupling_' + str(coupling_constant) + '/'

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
    parser.add_argument("--coupling_constant", type=float, default=0.5501, help="Coupling of variables")
    parser.add_argument("--length_vars", type=int, default=330000, help="Length of variables")
    parser.add_argument("--discard", type=int, default=30000, help="Number of points to discard")
    parser.add_argument("--seqs_len", type=int, default=10, help="Lengh of Sequences")
    parser.add_argument("--encoder_type", type=int, default=2, help="Type of encoder")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs for training")
    parser.add_argument("--scaling_method", type=str, default='Standard', help="Scaling method for data")
    parser.add_argument("--window_len", type=int, default=100, help="Length of Windows")
    parser.add_argument("--version", type=int, default=None, help="Version number of the model")
                            
    args = parser.parse_args()

    main(args.noise, args.dropout_rate_tcn, args.dropout_rate_hidden, args.tol_default, args.coupling_constant,
         args.length_vars, args.discard, args.seqs_len, args.encoder_type, args.batch_size, args.epochs, 
         args.scaling_method, args.window_len, args.version)
