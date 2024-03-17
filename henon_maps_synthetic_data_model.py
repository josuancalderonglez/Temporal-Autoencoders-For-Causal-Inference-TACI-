import os, re, sys, json, argparse, time, glob, pickle, copy, h5py, getopt

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from tensorflow.keras import regularizers

# Add other necessary imports
from tcn import tcn
from artificial_data import generate_henon03_henon03
from model_functions import synthetic_model_full_handler, synthetic_model_boot_plot

def main(noise, dropout_rate_tcn, dropout_rate_hidden, tol_default, length_vars, discard, encoder_type, batch_size, epochs, scaling_method, version):

    start = time.time()

    models_folder = '/scratch/jcalde2/Causation/SyntheticDataExamples/Henon_Maps/'
    
    # Check if 'version' is empty (None)
    if version is None:
        print("No version provided. Proceeding without a specific version.")

        nb_filters = 32
        kernel_size = 32
        tcn_dilations = []
        ii = 0
        while 2 ** ii <= nb_filters:
            tcn_dilations.append(2 ** ii)
            ii += 1
        conv_kernel_init = 'he_normal'
        latent_sample_rate = 2
        act_funct = 'elu'
        filters_conv1d = 16
        layer_nodes=128
        n_layers=4
        L1 = 0.0
        L2 = 0.0

        # 'Default'
        model_parameters_ = {
            'nb_filters' : nb_filters,
            'dilations' : tcn_dilations,
            'kernel_size' : kernel_size,
            'nb_stacks': 1,
            'ts_dimension': 1,
            'dropout_rate_tcn' : dropout_rate_tcn,
            'dropout_rate_hidden' : dropout_rate_hidden,
            'kernel_regularizer' : {'L1': L1, 'L2': L2},
            'conv_kernel_init' : conv_kernel_init,
            'padding': 'causal',
            'tcn_act_funct' : act_funct,
            'latent_sample_rate' : latent_sample_rate,
            'act_funct' : act_funct,
            'filters_conv1d' : filters_conv1d,
            'kernel_size_conv1d' : 1, 
            'activation_conv1d' : act_funct,
            'layer_nodes' : layer_nodes,
            'n_layers' : n_layers,
            'pad_encoder': False,
            'concatenate_layers': False,
            'model_summary': False,
            'encoder_type' : encoder_type,
            'batch_size' : batch_size,
            'epochs' : epochs,
            'noise' : noise,
            'verbose' : 0,
            'loss_funct' : 'mse',
            'shuffle' : True,
            'fulldata_testing': True,
            'scaling_method' : scaling_method,
            'surrogates_samples' : 100,
            'tol_default' : tol_default,
            'paint_fig' : False,
            'save_fig' : True}
                
        ssl = 10
        seq_length, shift, lag = ssl, ssl, ssl 
        if latent_sample_rate == 'seq_length':
            model_parameters_['latent_sample_rate'] = seq_length
        model_parameters_['model'] = 'generate_henon03_henon03'
        model_parameters_['directory'] = models_folder
        model_parameters_['method'] = 'odeint'
        model_parameters_['length_vars'] = length_vars
        model_parameters_['discard'] = discard
        model_parameters_['init_cond'] = [0.7, 0, 0.7, 0]
        model_parameters_['var1'] = 0
        model_parameters_['var2'] = 2    
        model_parameters_['couplings'] = np.round(np.arange(0.0, 0.804, 0.05) + 1e-4, 4).tolist()
        model_parameters_['seq_length'] = seq_length
        model_parameters_['shift'] = shift
        model_parameters_['lag'] = lag

        print('Noise = ' + str(model_parameters_['noise']))
        print('TCN Dropout = ' + str(model_parameters_['dropout_rate_tcn']))
        print('Dense Dropout = ' + str(model_parameters_['dropout_rate_hidden']))

        # Create New Folder
        version_pattern = re.compile(r'Version_(\d+)')
        folders = [f for f in os.listdir(model_parameters_['directory'] ) \
                if os.path.isdir(os.path.join(model_parameters_['directory'], f))]
        version_numbers = sorted([int(version_pattern.search(folder).group(1)) \
                for folder in folders if version_pattern.search(folder)])

        if not version_numbers:  # Check if the list is empty
            model_parameters_['directory'] = model_parameters_['directory'] + 'Version_0/'
            print('\nVersion_0')
        else:
            model_parameters_['directory'] = model_parameters_['directory'] + 'Version_' + \
                str(version_numbers[-1] + 1) + '/'
            print('\nVersion_' + str(version_numbers[-1] + 1))

        if not os.path.exists(model_parameters_['directory']):
            os.makedirs(model_parameters_['directory'])

        dicname = model_parameters_['directory'] + 'parameters.json'
        with open(dicname, 'w') as f:
            json.dump(model_parameters_, f, indent=4)

        dicname = model_parameters_['directory'] + 'parameters.txt'
        with open(dicname, 'w') as f: 
            for key, value in model_parameters_.items(): 
                f.write('%s:%s\n' % (key, value))
            
    else:
        print(f"Running with version {version}")
        model_parameters_={}
        model_parameters_['directory'] = models_folder + 'Version_' + str(version) + '/'
        parameters_dir = model_parameters_['directory'] + 'parameters.json'

        with open(parameters_dir, 'r') as f:
            model_parameters_ = json.load(f)
        model_parameters_['directory'] = models_folder + 'Version_' + str(version) + '/'

    function_name = model_parameters_['model']
    if function_name in globals():
        model_parameters_['model'] = globals()[function_name]
    else:
        print("Function not found.")

    synthetic_model_full_handler(model_parameters=model_parameters_)
    synthetic_model_boot_plot(model_parameters=model_parameters_)

    end = time.time()
    print('\n\n Elapsed Time : ' + str(end - start))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run model with parameters")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise Neural Network")
    parser.add_argument("--dropout_rate_tcn", type=float, default=0.0, help="TCN dropout rate")
    parser.add_argument("--dropout_rate_hidden", type=float, default=0.0, help="Hidden layer dropout rate")
    parser.add_argument("--tol_default", type=float, default=0.01, help="Default tolerance")
    parser.add_argument("--length_vars", type=int, default=330000, help="Length of variables")
    parser.add_argument("--discard", type=int, default=30000, help="Number of points to discard")
    parser.add_argument("--encoder_type", type=int, default=2, help="Type of encoder")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs for training")
    parser.add_argument("--scaling_method", type=str, default='Standard', help="Scaling method for data")
    parser.add_argument("--version", type=int, default=None, help="Version number of the model")
                            
    args = parser.parse_args()

    main(args.noise, args.dropout_rate_tcn, args.dropout_rate_hidden, args.tol_default, args.length_vars, 
         args.discard, args.encoder_type, args.batch_size, args.epochs, args.scaling_method, args.version)
