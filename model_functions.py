import os, re, sys, json, argparse, time, glob, pickle, copy, h5py, getopt

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from tcn import tcn
import tensorflow as tf
from sklearn.metrics import r2_score

try:
    from utils import printProgressBar, tic, toc, TicTocGenerator
    from utils import create_datasets, generate_fourier_surrogate
    from predictability import predictability_split, predictability_full
except ImportError:
    from .utils import printProgressBar, tic, toc, TicTocGenerator
    from .utils import create_datasets, generate_fourier_surrogate
    from .predictability import predictability_split, predictability_full

def tcni_pairwise(models_folder, parameters, variables, var1, var2, surrogate=False):

    if surrogate:
        parameters['encoder_type'] = 1
    else:
        parameters['encoder_type'] = 2

    parameters['pad_encoder'] = False
    parameters['concatenate_layers'] = False
    parameters['model_summary'] = False
    parameters['seq_length'] = 10
    parameters['shift'] = 10
    parameters['lag'] = 10

    seq_length, shift, lag = parameters['seq_length'] , parameters['shift'], parameters['lag']
    scaling_method = parameters['scaling_method']
    encoder_type = parameters['encoder_type']
    batch_size =  parameters['batch_size']

    # Generate Data
    trainX, valX, _, _, = create_datasets(dataset=variables[:-lag, var1], scaling_method=scaling_method,
                                            seq_length=seq_length, shift=shift)
    trainX_, valX_, _, _, = create_datasets(dataset=variables[lag:, var1], scaling_method=scaling_method,
                                                seq_length=seq_length, shift=shift)
    trainY, valY, _, _, = create_datasets(dataset=variables[:-lag, var2], scaling_method=scaling_method,
                                            seq_length=seq_length, shift=shift)
    trainY_, valY_, _, _, = create_datasets(dataset=variables[lag:, var2], scaling_method=scaling_method,
                                                seq_length=seq_length, shift=shift)

    if encoder_type == 1:
        trainX_sff, valX_sff, _, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var1]),
                                                scaling_method=scaling_method, seq_length=seq_length, shift=shift)
        trainY_sff, valY_sff, _, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var2]),
                                                scaling_method=scaling_method, seq_length=seq_length, shift=shift)
    else:
        trainX_sff, valX_sff, _ = [], [], []
        trainY_sff, valY_sff, _ = [], [], []
                        
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    filelist = glob.glob(os.path.join(models_folder, "*.hdf5"))

    filenameX = models_folder + 'auto_encoder_model_XYX.hdf5'
    filenameXP = models_folder + 'auto_encoder_model_XYXP.hdf5'
    filenameY = models_folder + 'auto_encoder_model_YXY.hdf5'
    filenameYP = models_folder + 'auto_encoder_model_YXYP.hdf5'
    plot_nameX = models_folder + 'Y to X training and validation.png'
    plot_nameY = models_folder + 'X to Y training and validation.png'

    if len(filelist) != 4:
        
        ######################################################## Train Models ########################################################
        

        ##############  Y to X  ##############
        auto_encoder_model_XYX, auto_encoder_model_XYXP = predictability(parameters, filenameX, filenameXP,
                                                                            trainX, valX, trainX_, valX_, trainY, valY,
                                                                            trainY_sff, valY_sff, save_plot=plot_nameX)

        # ##############  X to Y  ##############
        auto_encoder_model_YXY, auto_encoder_model_YXYP = predictability(parameters, filenameY, filenameYP,
                                                                            trainY, valY, trainY_, valY_, trainX, valX,
                                                                            trainX_sff, valX_sff, save_plot=plot_nameY)

    else:
        ######################################################## Load Models ########################################################
        auto_encoder_model_XYX = tf.keras.models.load_model(filenameX, custom_objects={'TCN': tcn.TCN})
        auto_encoder_model_XYXP = tf.keras.models.load_model(filenameXP, custom_objects={'TCN': tcn.TCN})
        auto_encoder_model_YXY = tf.keras.models.load_model(filenameY, custom_objects={'TCN': tcn.TCN})
        auto_encoder_model_YXYP = tf.keras.models.load_model(filenameYP, custom_objects={'TCN': tcn.TCN})

    ############################################################# Analysis ###########################################################
    default = 1e-10
    tol_default = parameters['tol_default']
    results = {}

    testX = valX.copy()
    testX_ = valX_.copy()
    test_surroX = valX_sff.copy()

    testY = valY.copy()
    testY_ = valY_.copy()
    test_surroY = valY_sff.copy()

    ##############  Y to X  ##############
    y_true = np.squeeze(testX_.copy())
    y_pred = np.squeeze(auto_encoder_model_XYX.predict([testX, testY], batch_size=batch_size, verbose=0))

    if encoder_type == 1:
        y_predP = np.squeeze(
            auto_encoder_model_XYXP.predict([testX, test_surroY ], batch_size=batch_size, verbose=0))
    elif encoder_type == 2:
        y_predP = np.squeeze(
            auto_encoder_model_XYXP.predict([testX, np.random.random(testY.shape)], batch_size=batch_size, verbose=0))

    ave_score = True

    rse1 = r2_score(y_true, y_pred)
    rse2 = r2_score(y_true, y_predP)

    if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=tol_default, atol=tol_default)):
        rse2 = default
        rse1 = rse2
        ave_score = False
    if rse2 < 0:
        rse2 = default
        ave_score = False

    if ave_score:
        score_x = ((rse1 - rse2) / (0.5 * np.abs(rse1 + rse2)))
    else:
        score_x = ((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2)))

    results['rse1_yx'] = rse1
    results['rse2_yx'] = rse2
    results['score_yx'] = score_x

    ##############  X to Y  ##############
    y_true = np.squeeze(testY_.copy())
    y_pred = np.squeeze(auto_encoder_model_YXY.predict([testY, testX], batch_size=batch_size, verbose=0))
    if encoder_type == 1:
        y_predP = np.squeeze(
            auto_encoder_model_YXYP.predict([testY, test_surroX], batch_size=batch_size, verbose=0))
    elif encoder_type == 2:
        y_predP = np.squeeze(
            auto_encoder_model_YXYP.predict([testY, np.random.random(testX.shape)], batch_size=batch_size, verbose=0))

    ### Raw ###
    ave_score = True

    rse1 = r2_score(y_true, y_pred)
    rse2 = r2_score(y_true, y_predP)

    if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=tol_default, atol=tol_default)):
        rse2 = default
        rse1 = rse2
        ave_score = False
    if rse2 < 0:
        rse2 = default
        ave_score = False

    if ave_score:
        score_y = ((rse1 - rse2) / (0.5 * np.abs(rse1 + rse2)))
    else:
        score_y = ((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2)))

    results['rse1_xy'] = rse1
    results['rse2_xy'] = rse2
    results['score_xy'] = score_y

    return score_x, score_y, results

def synthetic_model_split_handler(model_parameters):
    # tic()

    model = model_parameters['model']
    directory = model_parameters['directory']
    method = model_parameters['method']
    length_vars = model_parameters['length_vars']
    discard = model_parameters['discard']
    init_cond = model_parameters['init_cond']
    encoder_type = model_parameters['encoder_type']
    var1 = model_parameters['var1']
    var2 = model_parameters['var2']
    couplings = np.array(model_parameters['couplings'])
    seq_length = model_parameters['seq_length']
    shift = model_parameters['shift']
    lag = model_parameters['lag']
    scaling_method = model_parameters['scaling_method']
    batch_size = model_parameters['batch_size']
    epochs = model_parameters['epochs']
    noise = model_parameters['noise']

    if encoder_type == 0:
        encoder_type_name = 'Padded_Encoder'
    elif encoder_type == 1:
        encoder_type_name = 'Shuffle_encoder'
    elif encoder_type == 2:
        encoder_type_name = 'Random_Encoder'

    count = 0
    printProgressBar(count, len(couplings), prefix='Progress: ', suffix='Complete', length=50)
    for coupling_constant in couplings:

        models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
            epochs) + '/SeqLen_' + str(seq_length) + \
                        '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/Coupling_' + str(coupling_constant) + '/'
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        filelist = glob.glob(os.path.join(models_folder, "*.hdf5"))

        if len(filelist) != 4:

            # Generate Data
            variables = model(method, length_vars, init_cond, discard, coupling=coupling_constant)

            trainX, valX, _, _, _, = create_datasets(dataset=variables[:-lag, var1], scaling_method=scaling_method,
                                                  seq_length=seq_length, shift=shift)
            trainX_, valX_, _, _, _, = create_datasets(dataset=variables[lag:, var1], scaling_method=scaling_method,
                                                     seq_length=seq_length, shift=shift)
            trainY, valY, _, _, _, = create_datasets(dataset=variables[:-lag, var2], scaling_method=scaling_method,
                                                  seq_length=seq_length, shift=shift)
            trainY_, valY_, _, _, _, = create_datasets(dataset=variables[lag:, var2], scaling_method=scaling_method,
                                                     seq_length=seq_length, shift=shift)
            
            if encoder_type == 1:
                trainX_sff, valX_sff, _, _, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var1]),
                                                        scaling_method=scaling_method, seq_length=seq_length, shift=shift)
                trainY_sff, valY_sff, _, _, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var2]),
                                                        scaling_method=scaling_method, seq_length=seq_length, shift=shift)
            else:
                trainX_sff, valX_sff = [], []
                trainY_sff, valY_sff = [], []

            # Save Time Series
            filename = models_folder + 'variables.npy'
            np.save(filename, variables)

            # Train Models

            filenameX = models_folder + 'auto_encoder_model_XYX.hdf5'
            filenameXP = models_folder + 'auto_encoder_model_XYXP.hdf5'
            filenameY = models_folder + 'auto_encoder_model_YXY.hdf5'
            filenameYP = models_folder + 'auto_encoder_model_YXYP.hdf5'
            plot_nameX = models_folder + 'Y to X training and validation.png'
            plot_nameY = models_folder + 'X to Y training and validation.png'

            ##############  Y to X  ##############
            auto_encoder_model_XYX, auto_encoder_model_XYXP = predictability_split(model_parameters, filenameX, filenameXP,
                                                                             trainX, valX, trainX_, valX_, trainY, valY,
                                                                             trainY_sff, valY_sff, save_plot=plot_nameX)

            # ##############  X to Y  ##############
            auto_encoder_model_YXY, auto_encoder_model_YXYP = predictability_split(model_parameters, filenameY, filenameYP,
                                                                             trainY, valY, trainY_, valY_, trainX, valX,
                                                                             trainX_sff, valX_sff, save_plot=plot_nameY)

        count += 1
        printProgressBar(count, len(couplings), prefix='Progress: ', suffix='Complete', length=50)

    # toc()
    
def synthetic_model_full_handler(model_parameters):
    # tic()

    model = model_parameters['model']
    directory = model_parameters['directory']
    method = model_parameters['method']
    length_vars = model_parameters['length_vars']
    discard = model_parameters['discard']
    init_cond = model_parameters['init_cond']
    encoder_type = model_parameters['encoder_type']
    var1 = model_parameters['var1']
    var2 = model_parameters['var2']
    couplings = np.array(model_parameters['couplings'])
    seq_length = model_parameters['seq_length']
    shift = model_parameters['shift']
    lag = model_parameters['lag']
    scaling_method = model_parameters['scaling_method']
    batch_size = model_parameters['batch_size']
    epochs = model_parameters['epochs']
    noise = model_parameters['noise']

    if encoder_type == 0:
        encoder_type_name = 'Padded_Encoder'
    elif encoder_type == 1:
        encoder_type_name = 'Shuffle_encoder'
    elif encoder_type == 2:
        encoder_type_name = 'Random_Encoder'

    count = 0
    printProgressBar(count, len(couplings), prefix='Progress: ', suffix='Complete', length=50)
    for coupling_constant in couplings:

        models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
            epochs) + '/SeqLen_' + str(seq_length) + \
                        '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/Coupling_' + str(coupling_constant) + '/'
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        filelist = glob.glob(os.path.join(models_folder, "*.hdf5"))

        if len(filelist) != 4:

            # Generate Data
            variables = model(method, length_vars, init_cond, discard, coupling=coupling_constant)

            _, _, _, full_dataX, _, = create_datasets(dataset=variables[:-lag, var1], scaling_method=scaling_method,
                                                  seq_length=seq_length, shift=shift)
            _, _, _, full_dataX_, _, = create_datasets(dataset=variables[lag:, var1], scaling_method=scaling_method,
                                                     seq_length=seq_length, shift=shift)
            _, _, _, full_dataY, _, = create_datasets(dataset=variables[:-lag, var2], scaling_method=scaling_method,
                                                  seq_length=seq_length, shift=shift)
            _, _, _, full_dataY_, _, = create_datasets(dataset=variables[lag:, var2], scaling_method=scaling_method,
                                                     seq_length=seq_length, shift=shift)
            
            if encoder_type == 1:
                _, _, _, full_dataX_sff, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var1]),
                                                        scaling_method=scaling_method, seq_length=seq_length, shift=shift)
                _, _, _, full_dataY_sff, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var2]),
                                                        scaling_method=scaling_method, seq_length=seq_length, shift=shift)
            else:
                full_dataX_sff = []
                full_dataY_sff = []

            # Save Time Series
            filename = models_folder + 'variables.npy'
            np.save(filename, variables)

            # Train Models

            filenameX = models_folder + 'auto_encoder_model_XYX.hdf5'
            filenameXP = models_folder + 'auto_encoder_model_XYXP.hdf5'
            filenameY = models_folder + 'auto_encoder_model_YXY.hdf5'
            filenameYP = models_folder + 'auto_encoder_model_YXYP.hdf5'
            plot_nameX = models_folder + 'Y to X training and validation.png'
            plot_nameY = models_folder + 'X to Y training and validation.png'

            ##############  Y to X  ##############
            auto_encoder_model_XYX, auto_encoder_model_XYXP = predictability_full(model_parameters, filenameX, filenameXP,
                                                                             full_dataX, full_dataX_, full_dataX_sff, 
                                                                             full_dataY, full_dataY_, full_dataY_sff,
                                                                             save_plot=plot_nameX)

            # ##############  X to Y  ##############
            auto_encoder_model_YXY, auto_encoder_model_YXYP = predictability_full(model_parameters, filenameY, filenameYP,
                                                                             full_dataY, full_dataY_, full_dataY_sff,
                                                                             full_dataX, full_dataX_, full_dataX_sff, 
                                                                             save_plot=plot_nameY)

        count += 1
        printProgressBar(count, len(couplings), prefix='Progress: ', suffix='Complete', length=50)

    # toc()

def synthetic_model_boot_plot(model_parameters):
    # tic()

    directory = model_parameters['directory']
    encoder_type = model_parameters['encoder_type']
    var1 = model_parameters['var1']
    var2 = model_parameters['var2']
    couplings = np.array(model_parameters['couplings'])
    seq_length = model_parameters['seq_length']
    shift = model_parameters['shift']
    lag = model_parameters['lag']
    scaling_method = model_parameters['scaling_method']
    batch_size = model_parameters['batch_size']
    epochs = model_parameters['epochs']
    noise = model_parameters['noise']
    verbose = model_parameters['verbose']
    surrogates_samples = model_parameters['surrogates_samples']
    tol_default = model_parameters['tol_default']
    paint_fig = model_parameters['paint_fig']
    save_fig = model_parameters['save_fig']
    fulldata_testing = model_parameters['fulldata_testing']

    if encoder_type == 0:
        encoder_type_name = 'Padded_Encoder'
    elif encoder_type == 1:
        encoder_type_name = 'Shuffle_encoder'
    elif encoder_type == 2:
        encoder_type_name = 'Random_Encoder'

    models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
        epochs) + '/SeqLen_' + str(seq_length) + \
                    '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/'
    filelist = glob.glob(os.path.join(models_folder, "score*"))

    if len(filelist) == 0:

        default = 1e-4

        # score_raw = np.zeros((2, len(couplings)))
        # score_surro_mean = np.zeros((2, len(couplings)))
        # score_surro_std = np.zeros((2, len(couplings)))
        score_boot_mean = np.zeros((2, len(couplings)))
        score_boot_std = np.zeros((2, len(couplings)))

        count = 0
        printProgressBar(count, len(couplings), prefix='Progress: ', suffix='Complete', length=50)

        for coupling_constant in couplings:

            models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
                epochs) + '/SeqLen_' + str(seq_length) + \
                            '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/Coupling_' + str(coupling_constant) + '/'

            filelist = glob.glob(os.path.join(models_folder, "*.hdf5"))

            if len(filelist) == 4:
                
                # Load Models
                filenameX = models_folder + 'auto_encoder_model_XYX.hdf5'
                auto_encoder_model_XYX = tf.keras.models.load_model(filenameX, custom_objects={'TCN': tcn.TCN})

                filenameXP = models_folder + 'auto_encoder_model_XYXP.hdf5'
                auto_encoder_model_XYXP = tf.keras.models.load_model(filenameXP, custom_objects={'TCN': tcn.TCN})

                filenameY = models_folder + 'auto_encoder_model_YXY.hdf5'
                auto_encoder_model_YXY = tf.keras.models.load_model(filenameY, custom_objects={'TCN': tcn.TCN})

                filenameYP = models_folder + 'auto_encoder_model_YXYP.hdf5'
                auto_encoder_model_YXYP = tf.keras.models.load_model(filenameYP, custom_objects={'TCN': tcn.TCN})

                # Generate Data
                filename = models_folder + 'variables.npy'
                variables = np.load(filename)

                _, _, testX, full_dataX, _, = create_datasets(dataset=variables[:-lag, var1], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
                _, _, testX_, full_dataX_, _, = create_datasets(dataset=variables[lag:, var1], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
                _, _, testY, full_dataY, _, = create_datasets(dataset=variables[:-lag, var2], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
                _, _, testY_, full_dataY_, _, = create_datasets(dataset=variables[lag:, var2], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
                
                if fulldata_testing:
                    testX = full_dataX
                    testX_ = full_dataX_
                    testY = full_dataY
                    testY_ = full_dataY_

                ##############  Y to X  ##############
                # y_true = np.squeeze(testX_.copy())
                # y_pred = np.squeeze(auto_encoder_model_XYX.predict([testX, testY], batch_size=batch_size, verbose=0))
                # y_predP = np.squeeze(
                #     auto_encoder_model_XYXP.predict([testX, np.random.random(testY.shape)], batch_size=batch_size, verbose=0))

                # ### Raw ###
                # ave_score = True

                # rse1 = r2_score(y_true, y_pred)
                # rse2 = r2_score(y_true, y_predP)

                # if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=tol_default, atol=tol_default)):
                #     rse2 = default
                #     rse1 = rse2
                #     ave_score = False
                # if rse2 < 0:
                #     rse2 = default
                #     ave_score = False

                # if ave_score:
                #     score_raw[0, count] = ((rse1 - rse2) / (0.5 * np.abs(rse1 + rse2)))
                # else:
                #     score_raw[0, count] = ((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2)))

                ### Bootstraps ###
                score = []                
                for jj in range(surrogates_samples):

                    ave_score = True

                    indxs = np.random.choice(np.arange(testX.shape[0]), size=testX.shape[0], replace=True)
                    
                    boot_testX = np.squeeze(testX)[indxs]
                    boot_testY = np.squeeze(testY)[indxs]
                    boot_testX_ = np.squeeze(testX_)[indxs]

                    y_true_boot = boot_testX_.flatten()
                    y_pred_boot = auto_encoder_model_XYX.predict([boot_testX, boot_testY], batch_size=batch_size, verbose=0).flatten()
                    y_predP_boot = auto_encoder_model_XYXP.predict([boot_testX, np.random.random(boot_testY.shape)], 
                                                                   batch_size=batch_size, verbose=0).flatten()
                                        
                    rse1 = r2_score(y_true_boot, y_pred_boot)
                    rse2 = r2_score(y_true_boot, y_predP_boot)

                    if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=tol_default, atol=tol_default)):
                        rse2 = default
                        rse1 = rse2
                        ave_score = False
                    if rse2 < 0:
                        rse2 = default
                        ave_score = False

                    if ave_score:
                        score.append(((rse1 - rse2) / (0.5 * np.abs(rse1 + rse2))))
                    else:
                        score.append(((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2))))

                score_boot_mean[0, count] = np.mean(score)
                score_boot_std[0, count] = np.std(score)

                # ### Surrogates ###
                # score = []
                # for jj in range(surrogates_samples):

                #     ave_score = True

                #     _, test_surroY, full_data_surroY, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var2]), 
                #                                     scaling_method=scaling_method,seq_length=seq_length, shift=shift)
                    
                #     y_true = np.squeeze(testX_.copy())
                #     y_pred = np.squeeze(auto_encoder_model_XYX.predict([testX, testY], batch_size=batch_size, verbose=0))
                #     y_predP = np.squeeze(
                #         auto_encoder_model_XYXP.predict([testX, test_surroY ], batch_size=batch_size, verbose=0))

                #     rse1 = r2_score(y_true, y_pred)
                #     rse2 = r2_score(y_true, y_predP)

                #     if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=tol_default, atol=tol_default)):
                #         rse2 = default
                #         rse1 = rse2
                #         ave_score = False
                #     if rse2 < 0:
                #         rse2 = default
                #         ave_score = False

                #     if ave_score:
                #         score.append(((rse1 - rse2) / (0.5 * np.abs(rse1 + rse2))))
                #     else:
                #         score.append(((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2))))

                # score_surro_mean[0, count] = np.mean(score)
                # score_surro_std[0, count] = np.std(score)
            
                ##############  X to Y  ##############
                # y_true = np.squeeze(testY_.copy())
                # y_pred = np.squeeze(auto_encoder_model_YXY.predict([testY, testX], batch_size=batch_size, verbose=0))
                # y_predP = np.squeeze(
                #     auto_encoder_model_YXYP.predict([testY, np.random.random(testX.shape)], batch_size=batch_size, verbose=0))

                # ### Raw ###
                # ave_score = True

                # rse1 = r2_score(y_true, y_pred)
                # rse2 = r2_score(y_true, y_predP)

                # if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=tol_default, atol=tol_default)):
                #     rse2 = default
                #     rse1 = rse2
                #     ave_score = False
                # if rse2 < 0:
                #     rse2 = default
                #     ave_score = False

                # if ave_score:
                #     score_raw[1, count] = ((rse1 - rse2) / (0.5 * np.abs(rse1 + rse2)))
                # else:
                #     score_raw[1, count] = ((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2)))

                ### Bootstraps ###s
                score = []
                for jj in range(surrogates_samples):

                    ave_score = True

                    indxs = np.random.choice(np.arange(testX.shape[0]), size=testX.shape[0], replace=True)

                    boot_testX = np.squeeze(testX.copy())[indxs]
                    boot_testY = np.squeeze(testY.copy())[indxs]
                    boot_testY_ = np.squeeze(testY_.copy())[indxs]

                    y_true_boot = boot_testY_.copy().flatten()
                    y_pred_boot = auto_encoder_model_YXY.predict([boot_testY, boot_testX], batch_size=batch_size, verbose=0).flatten()
                    y_predP_boot = auto_encoder_model_YXYP.predict([boot_testY, np.random.random(boot_testX.shape)], 
                                                              batch_size=batch_size, verbose=0).flatten()

                    rse1 = r2_score(y_true_boot, y_pred_boot)
                    rse2 = r2_score(y_true_boot, y_predP_boot)

                    if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=tol_default, atol=tol_default)):
                        rse2 = default
                        rse1 = rse2
                        ave_score = False
                    if rse2 < 0:
                        rse2 = default
                        ave_score = False

                    if ave_score:
                        score.append(((rse1 - rse2) / (0.5 * np.abs(rse1 + rse2))))
                    else:
                        score.append(((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2))))

                score_boot_mean[1, count] = np.mean(score)
                score_boot_std[1, count] = np.std(score)
                
                # ### Surrogates ###
                # score = []
                # for jj in range(surrogates_samples):

                #     ave_score = True

                #     _, test_surroX, full_data_surroX, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var1]), 
                #                                     scaling_method=scaling_method,seq_length=seq_length, shift=shift)
                    
                #     y_true = np.squeeze(testY_.copy())
                #     y_pred = np.squeeze(auto_encoder_model_YXY.predict([testY, testX], batch_size=batch_size, verbose=0))
                #     y_predP = np.squeeze(
                #         auto_encoder_model_YXYP.predict([testY, test_surroX], batch_size=batch_size, verbose=0))

                #     if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=tol_default, atol=tol_default)):
                #         rse2 = default
                #         rse1 = rse2
                #         ave_score = False
                #     if rse2 < 0:
                #         rse2 = default
                #         ave_score = False

                #     if ave_score:
                #         score.append(((rse1 - rse2) / (0.5 * np.abs(rse1 + rse2))))
                #     else:
                #         score.append(((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2))))

                # score_surro_mean[1, count] = np.mean(score)
                # score_surro_std[1, count] = np.std(score)

            count += 1

            printProgressBar(count, len(couplings), prefix='Progress: ', suffix='Complete', length=50)

        models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
            epochs) + '/SeqLen_' + str(seq_length) + \
                        '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/'

        # Save Scores
        # filename = models_folder + 'score_raw.npy'
        # np.save(filename, score_raw)
        filename = models_folder + 'score_boot_mean.npy'
        np.save(filename, score_boot_mean)
        filename = models_folder + 'score_boot_std.npy'
        np.save(filename, score_boot_std)
        # filename = models_folder + 'score_surro_mean.npy'
        # np.save(filename, score_surro_mean)
        # filename = models_folder + 'score_surro_std.npy'
        # np.save(filename, score_surro_std)

    else:
        
        models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
            epochs) + '/SeqLen_' + str(seq_length) + \
                        '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/'

        # Load Scores
        # filename = models_folder + 'score_raw.npy'
        # score_raw = np.load(filename)
        filename = models_folder + 'score_boot_mean.npy'
        score_boot_mean = np.load(filename)
        filename = models_folder + 'score_boot_std.npy'
        score_boot_std = np.load(filename)
        # filename = models_folder + 'score_surro_mean.npy'
        # score_surro_mean = np.load(filename)
        # filename = models_folder + 'score_surro_std.npy'
        # score_surro_std = np.load(filename)

    # # Save Raw Figure
    # fig = plt.figure(figsize=(20, 5))
    # plt.plot(couplings, score_raw[0], 'ro-', label='Y to X')
    # plt.plot(couplings, score_raw[1], 'bo-',label='X to Y')
    # plt.plot(couplings, np.zeros((len(couplings))), '-k')
    # plt.xlabel('Couplings')
    # plt.ylabel('Inference Score')
    # plt.title('TCNI')
    # plt.legend()

    # if save_fig:
    #     save_plot = models_folder + 'TCNI.png'

    #     plt.savefig(save_plot, bbox_inches="tight", pad_inches=1, transparent=False,
    #                 facecolor="w", edgecolor='w', orientation='landscape')
    # if paint_fig:
    #     plt.show()
    # else:
    #     plt.close(fig)

    # Save Bootstraps Figure
    fig = plt.figure(figsize=(20, 5))
    plt.errorbar(couplings, score_boot_mean[0], yerr=score_boot_std[0], fmt='o-', color='red',
                    label='Y to X')
    plt.errorbar(couplings, score_boot_mean[1], yerr=score_boot_std[1], fmt='o-', color='blue',
                    label='X to Y')
    plt.plot(couplings, np.zeros((len(couplings))), '-k')
    plt.xlabel('Couplings')
    plt.ylabel('Inference Score')
    plt.title('TCNI with Bootstraps')
    plt.legend()

    if save_fig:
        save_plot = models_folder + 'TCNI_Bootstraps.png'

        plt.savefig(save_plot, bbox_inches="tight", pad_inches=1, transparent=False,
                    facecolor="w", edgecolor='w', orientation='landscape')

    if paint_fig:
        plt.show()
    else:
        plt.close(fig)

    # # Save Surrogates Figure
    # fig = plt.figure(figsize=(20, 5))
    # plt.errorbar(couplings, score_surro_mean[0], yerr=score_surro_std[0], fmt='o-', color='red',
    #              label='Y to X')
    # plt.errorbar(couplings, score_surro_mean[1], yerr=score_surro_std[1], fmt='o-', color='blue',
    #              label='X to Y')
    # plt.plot(couplings, np.zeros((len(couplings))), '-k')
    # plt.xlabel('Couplings')
    # plt.ylabel('Inference Score')
    # plt.title('TCNI with Surrogates')
    # plt.legend()

    # if save_fig:
    #     save_plot = models_folder + 'TCNI_Surrogates.png'

    #     plt.savefig(save_plot, bbox_inches="tight", pad_inches=1, transparent=False,
    #                 facecolor="w", edgecolor='w', orientation='landscape')

    # if paint_fig:
    #     plt.show()
    # else:
    #     plt.close(fig)

    # toc()

def temporal_unidirectional(directory, variables, var1, var2, noise, dropout_rate_tcn, dropout_rate_hidden, tol_default, 
                  length_vars, discard, encoder_type, batch_size, epochs, scaling_method, seqs_len, window_len, version):
    
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

    tic()

    seq_length, shift, lag = seqs_len, seqs_len, seqs_len 
    if latent_sample_rate == 'seq_length':
        model_parameters_['latent_sample_rate'] = seq_length
    model_parameters_['model'] = None
    model_parameters_['directory'] = directory
    model_parameters_['method'] = None
    model_parameters_['length_vars'] = length_vars
    model_parameters_['discard'] = discard
    model_parameters_['init_cond'] = None
    model_parameters_['var1'] = var1
    model_parameters_['var2'] = var2
    model_parameters_['couplings'] = None
    model_parameters_['seq_length'] = seq_length
    model_parameters_['shift'] = shift
    model_parameters_['lag'] = lag
    
    if not os.path.exists(model_parameters_['directory']):
        os.makedirs(model_parameters_['directory'])

    if version is None:
        print("No version provided. Proceeding without a specific version.")

        # Create New Folder
        version_pattern = re.compile(r'Version_(\d+)')
        folders = [f for f in os.listdir(model_parameters_['directory'] ) \
                if os.path.isdir(os.path.join(model_parameters_['directory'], f))]
        version_numbers = sorted([int(version_pattern.search(folder).group(1)) \
                for folder in folders if version_pattern.search(folder)])

        if not version_numbers:  # Check if the list is empty
            model_parameters_['directory'] = model_parameters_['directory'] + 'Version_0/'
            # print('\nVersion_0')
        else:
            model_parameters_['directory'] = model_parameters_['directory'] + 'Version_' + \
                str(version_numbers[-1] + 1) + '/'
            # print('\nVersion_' + str(version_numbers[-1] + 1))

    else:
        print(f"Running with version {version}")

        model_parameters_['directory'] = model_parameters_['directory'] + 'Version_' + str(version) + '/'

    if not os.path.exists(model_parameters_['directory']):
            os.makedirs(model_parameters_['directory'])

    print('Model Directory: ' + model_parameters_['directory'])

    dicname = model_parameters_['directory'] + 'parameters.json'
    with open(dicname, 'w') as f:
        json.dump(model_parameters_, f, indent=4)

    dicname = model_parameters_['directory'] + 'parameters.txt'
    with open(dicname, 'w') as f: 
        for key, value in model_parameters_.items(): 
            f.write('%s:%s\n' % (key, value))

    temporal_unidirectional_handler(model_parameters=model_parameters_, variables=variables)
    temporal_unidirectional_boot_plot(model_parameters=model_parameters_, window_len=window_len)
    
    toc()

def temporal_bidirectional(directory, variables, var1, var2, noise, dropout_rate_tcn, dropout_rate_hidden, tol_default, 
                  length_vars, discard, encoder_type, batch_size, epochs, scaling_method, seqs_len, window_len, version):
    
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

    tic()

    seq_length, shift, lag = seqs_len, seqs_len, seqs_len 
    if latent_sample_rate == 'seq_length':
        model_parameters_['latent_sample_rate'] = seq_length
    model_parameters_['model'] = None
    model_parameters_['directory'] = directory
    model_parameters_['method'] = None
    model_parameters_['length_vars'] = length_vars
    model_parameters_['discard'] = discard
    model_parameters_['init_cond'] = None
    model_parameters_['var1'] = var1
    model_parameters_['var2'] = var2
    model_parameters_['couplings'] = None
    model_parameters_['seq_length'] = seq_length
    model_parameters_['shift'] = shift
    model_parameters_['lag'] = lag
    
    if not os.path.exists(model_parameters_['directory']):
        os.makedirs(model_parameters_['directory'])

    if version is None:
        print("No version provided. Proceeding without a specific version.")

        # Create New Folder
        version_pattern = re.compile(r'Version_(\d+)')
        folders = [f for f in os.listdir(model_parameters_['directory'] ) \
                if os.path.isdir(os.path.join(model_parameters_['directory'], f))]
        version_numbers = sorted([int(version_pattern.search(folder).group(1)) \
                for folder in folders if version_pattern.search(folder)])

        if not version_numbers:  # Check if the list is empty
            model_parameters_['directory'] = model_parameters_['directory'] + 'Version_0/'
            # print('\nVersion_0')
        else:
            model_parameters_['directory'] = model_parameters_['directory'] + 'Version_' + \
                str(version_numbers[-1] + 1) + '/'
            # print('\nVersion_' + str(version_numbers[-1] + 1))

    else:
        print(f"Running with version {version}")

        model_parameters_['directory'] = model_parameters_['directory'] + 'Version_' + str(version) + '/'

    if not os.path.exists(model_parameters_['directory']):
            os.makedirs(model_parameters_['directory'])

    dicname = model_parameters_['directory'] + 'parameters.json'
    with open(dicname, 'w') as f:
        json.dump(model_parameters_, f, indent=4)

    dicname = model_parameters_['directory'] + 'parameters.txt'
    with open(dicname, 'w') as f: 
        for key, value in model_parameters_.items(): 
            f.write('%s:%s\n' % (key, value))

    temporal_bidirectional_handler(model_parameters=model_parameters_, variables=variables)
    temporal_bidirectional_boot_plot(model_parameters=model_parameters_, window_len=window_len)
    
    toc()

def temporal_unidirectional_handler(model_parameters, variables):
    # tic()

    directory = model_parameters['directory']
    encoder_type = model_parameters['encoder_type']
    var1 = model_parameters['var1']
    var2 = model_parameters['var2']
    seq_length = model_parameters['seq_length']
    shift = model_parameters['shift']
    lag = model_parameters['lag']
    scaling_method = model_parameters['scaling_method']
    epochs = model_parameters['epochs']
    noise = model_parameters['noise']

    if encoder_type == 0:
        encoder_type_name = 'Padded_Encoder'
    elif encoder_type == 1:
        encoder_type_name = 'Shuffle_encoder'
    elif encoder_type == 2:
        encoder_type_name = 'Random_Encoder'

    models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
        epochs) + '/SeqLen_' + str(seq_length) + \
                    '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/'
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    filelist = glob.glob(os.path.join(models_folder, "*.hdf5"))

    print('Training ...')

    if len(filelist) != 2:

        _, _, _, full_dataX, _, = create_datasets(dataset=variables[:-lag, var1], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
        _, _, _, full_dataX_, _, = create_datasets(dataset=variables[lag:, var1], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
        _, _, _, full_dataY, _, = create_datasets(dataset=variables[:-lag, var2], scaling_method=scaling_method,
                                                seq_length=seq_length, shift=shift)
        _, _, _, full_dataY_, _, = create_datasets(dataset=variables[lag:, var2], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
        
        if encoder_type == 1:
            _, _, _, full_dataX_sff, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var1]),
                                                    scaling_method=scaling_method, seq_length=seq_length, shift=shift)
            _, _, _, full_dataY_sff, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var2]),
                                                    scaling_method=scaling_method, seq_length=seq_length, shift=shift)
        else:
            full_dataX_sff = []
            full_dataY_sff = []

        # # Save Time Series
        filename = models_folder + 'variables.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(variables, file)

        # Train Models

        filenameX = models_folder + 'auto_encoder_model_XYX.hdf5'
        filenameXP = models_folder + 'auto_encoder_model_XYXP.hdf5'
        filenameY = models_folder + 'auto_encoder_model_YXY.hdf5'
        filenameYP = models_folder + 'auto_encoder_model_YXYP.hdf5'
        plot_nameX = models_folder + 'Y to X training and validation.png'
        plot_nameY = models_folder + 'X to Y training and validation.png'

        # ##############  X to Y  ##############
        auto_encoder_model_YXY, auto_encoder_model_YXYP = predictability_full(model_parameters, filenameY, filenameYP,
                                                                            full_dataY, full_dataY_, full_dataY_sff,
                                                                            full_dataX, full_dataX_, full_dataX_sff, 
                                                                            save_plot=plot_nameY)
    # toc()
        
def temporal_unidirectional_boot_plot(model_parameters, window_len):
    # tic()

    directory = model_parameters['directory']
    encoder_type = model_parameters['encoder_type']
    var1 = model_parameters['var1']
    var2 = model_parameters['var2']
    seq_length = model_parameters['seq_length']
    shift = model_parameters['shift']
    lag = model_parameters['lag']
    scaling_method = model_parameters['scaling_method']
    batch_size = model_parameters['batch_size']
    epochs = model_parameters['epochs']
    noise = model_parameters['noise']
    verbose = model_parameters['verbose']
    surrogates_samples = model_parameters['surrogates_samples']
    tol_default = model_parameters['tol_default']
    paint_fig = model_parameters['paint_fig']
    save_fig = model_parameters['save_fig']
    fulldata_testing = model_parameters['fulldata_testing']

    if encoder_type == 0:
        encoder_type_name = 'Padded_Encoder'
    elif encoder_type == 1:
        encoder_type_name = 'Shuffle_encoder'
    elif encoder_type == 2:
        encoder_type_name = 'Random_Encoder'

    models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(epochs) + \
                    '/SeqLen_' + str(seq_length) + '_Shift_' + str(shift) + '_Lag_' + str(lag) + \
                    '/Tolerance_' + str(tol_default) + '/window_length_' + str(window_len) + '/'
    
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    filelist = glob.glob(os.path.join(models_folder, "score*"))

    print('Calculating Scores ...')

    if len(filelist) == 0:

        default = 1e-4

        models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(epochs) + \
                    '/SeqLen_' + str(seq_length) + '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/'

        filelist = glob.glob(os.path.join(models_folder, "*.hdf5"))

        if len(filelist) == 2:

            # Load Models
            filenameY = models_folder + 'auto_encoder_model_YXY.hdf5'
            auto_encoder_model_YXY = tf.keras.models.load_model(filenameY, custom_objects={'TCN': tcn.TCN})

            filenameYP = models_folder + 'auto_encoder_model_YXYP.hdf5'
            auto_encoder_model_YXYP = tf.keras.models.load_model(filenameYP, custom_objects={'TCN': tcn.TCN})

            print('\t Models Loaded')

            # Generate Data
            filename = models_folder + 'variables.pkl'
            if os.path.exists(filename):  
                try:
                    with open(filename, 'rb') as file:
                        variables = pickle.load(file)
                except pickle.UnpicklingError:
                    pass
            
            else:
                filename = models_folder + 'variables.npy'
                if os.path.exists(filename):  
                    try:
                        variables = np.load(filename)
                    except Exception as e:
                        pass
            
            print('\t Data Loaded')

            _, _, testX, full_dataX, full_data_trackerX = create_datasets(dataset=variables[:-lag, var1], scaling_method=scaling_method,
                                                seq_length=seq_length, shift=shift)
            _, _, testX_, full_dataX_, full_data_trackerX_ = create_datasets(dataset=variables[lag:, var1], scaling_method=scaling_method,
                                                seq_length=seq_length, shift=shift)
            _, _, testY, full_dataY, full_data_trackerY = create_datasets(dataset=variables[:-lag, var2], scaling_method=scaling_method,
                                                seq_length=seq_length, shift=shift)
            _, _, testY_, full_dataY_, full_data_trackerY_ = create_datasets(dataset=variables[lag:, var2], scaling_method=scaling_method,
                                                seq_length=seq_length, shift=shift)
            
            if fulldata_testing:
                testX = full_dataX
                testX_ = full_dataX_
                testY = full_dataY
                testY_ = full_dataY_
            
            score_no_boot = np.zeros((1, len(testX_) // window_len))
            score_boot_mean = np.zeros((1, len(testX_) // window_len))
            score_boot_std = np.zeros((1, len(testX_) // window_len))
            intervals = np.zeros((len(testX_) // window_len))

            for ii in range(len(testX_) // window_len):

                interval = np.arange(ii * window_len, (ii + 1) * window_len)
                intervals[ii] = np.squeeze(full_data_trackerX_[interval]).flatten()[-1]

                #############  X to Y  ##############
                y_true = np.squeeze(testY_[interval]).flatten()
                y_pred = np.squeeze(auto_encoder_model_YXY.predict([testY[interval],
                                testX[interval]], batch_size=batch_size, verbose=verbose)).flatten()
                # if encoder_type == 1:
                #     y_predP = np.squeeze(auto_encoder_model_YXYP.predict([testY[interval],
                #                                                           full_dataX_sff[interval]],
                #                                                          batch_size=batch_size,
                #                                                          verbose=verbose)).flatten()
                if encoder_type == 2:
                    y_predP = np.squeeze(auto_encoder_model_YXYP.predict([testY[interval],
                                np.random.random(testX[interval].shape)], batch_size=batch_size, verbose=verbose)).flatten()

                ### Without Bootstraps ###

                rse1 = r2_score(y_true, y_pred)
                rse2 = r2_score(y_true, y_predP)

                if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=tol_default, atol=tol_default)):
                    rse2 = default
                    rse1 = rse2
                if rse2 < 0:
                    rse2 = default

                score_no_boot[0, ii] = ((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2)))

                # ### Bootstraps ###
                # score = []
                # for jj in range(surrogates_samples):

                #     ave_score = True

                #     indxs = np.random.choice(np.arange(len(interval)), size=len(interval), replace=True)

                #     boot_testX = np.squeeze(testX[interval])[indxs]
                #     boot_testY = np.squeeze(testY[interval])[indxs]
                #     boot_testY_ = np.squeeze(testY_[interval])[indxs]

                #     y_true_boot = boot_testY_.copy().flatten()
                #     y_pred_boot = auto_encoder_model_YXY.predict([boot_testY, boot_testX], batch_size=batch_size, verbose=0).flatten()
                #     y_predP_boot = auto_encoder_model_YXYP.predict([boot_testY, np.random.random(boot_testX.shape)], 
                #                                               batchs_size=batch_size, verbose=0).flatten()
                    
                #     rse1 = r2_score(y_true_boot, y_pred_boot)
                #     rse2 = r2_score(y_true_boot, y_predP_boot)

                #     if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=tol_default, atol=tol_default)):
                #         rse2 = default
                #         rse1 = rse2
                #         ave_score = False
                #     if rse2 < 0:
                #         rse2 = default
                #         ave_score = False

                #     if ave_score:
                #         score.append(((rse1 - rse2) / (0.5 * np.abs(rse1 + rse2))))
                #     else:
                #         score.append(((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2))))

                # score_boot_mean[0, ii] = np.mean(score)
                # score_boot_std[0, ii] = np.std(score)

            models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(epochs) + \
                    '/SeqLen_' + str(seq_length) + '_Shift_' + str(shift) + '_Lag_' + str(lag) + \
                    '/Tolerance_' + str(tol_default) + '/window_length_' + str(window_len) + '/'

            # Save Scores
            filename = models_folder + 'score_no_boot.npy'
            np.save(filename, score_no_boot)
            filename = models_folder + 'intervals.npy'
            np.save(filename, intervals)
            filename = models_folder + 'score_boot_mean.npy'
            np.save(filename, score_boot_mean)
            filename = models_folder + 'score_boot_std.npy'
            np.save(filename, score_boot_std)

            # # Save Figure
            # fig = plt.figure(figsize=(20, 5))
            # plt.errorbar(intervals, score_boot_mean[0], yerr=score_boot_std[0], fmt='o-', color='red',
            #              label='Bootstrap Y to X')
            # plt.errorbar(intervals, score_boot_mean[1], yerr=score_boot_std[1], fmt='o-', color='blue',
            #              label='Bootstrap X to Y')
            # plt.plot(intervals, np.zeros((len(full_dataX_) // window_len)), '-k')
            # plot_title = 'SeqLen_' + str(seq_length) + '_Shift_' + str(shift) + '_Lag_' + str(lag)
            # plt.title(plot_title)
            # plt.legend()

            # if save_fig:
            #     save_plot = models_folder + 'causal_inference_bootstraps.png'

            #     plt.savefig(save_plot, bbox_inches="tight", pad_inches=1, transparent=False,
            #                 facecolor="w", edgecolor='w', orientation='landscape')

            # if paint_fig:
            #     plt.show()
            # else:
            #     plt.close(fig)

            # Save Figure
            fig = plt.figure(figsize=(20, 5))
            plt.plot(intervals, score_no_boot[0], 'o-', color='red', label=' X to Y')
            plt.plot(intervals, np.zeros((len(full_dataX_) // window_len)), '-k')
            plot_title = 'SeqLen_' + str(seq_length) + '_Shift_' + str(shift) + '_Lag_' + str(lag)
            plt.title(plot_title)
            plt.legend()

            if save_fig:
                save_plot = models_folder + 'causal_inference.png'

                plt.savefig(save_plot, bbox_inches="tight", pad_inches=1, transparent=False,
                            facecolor="w", edgecolor='w', orientation='landscape')

            if paint_fig:
                plt.show()
            else:
                plt.close(fig)

        # toc()



def temporal_bidirectional_handler(model_parameters, variables):
    # tic()

    model = model_parameters['model']
    directory = model_parameters['directory']
    method = model_parameters['method']
    length_vars = model_parameters['length_vars']
    discard = model_parameters['discard']
    init_cond = model_parameters['init_cond']
    encoder_type = model_parameters['encoder_type']
    var1 = model_parameters['var1']
    var2 = model_parameters['var2']
    couplings = np.array(model_parameters['couplings'])
    seq_length = model_parameters['seq_length']
    shift = model_parameters['shift']
    lag = model_parameters['lag']
    scaling_method = model_parameters['scaling_method']
    batch_size = model_parameters['batch_size']
    epochs = model_parameters['epochs']
    noise = model_parameters['noise']

    if encoder_type == 0:
        encoder_type_name = 'Padded_Encoder'
    elif encoder_type == 1:
        encoder_type_name = 'Shuffle_encoder'
    elif encoder_type == 2:
        encoder_type_name = 'Random_Encoder'

    models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
        epochs) + '/SeqLen_' + str(seq_length) + \
                    '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/'
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    filelist = glob.glob(os.path.join(models_folder, "*.hdf5"))

    print('\n\nTraining ...')

    if len(filelist) != 4:

        _, _, _, full_dataX, _, = create_datasets(dataset=variables[:-lag, var1], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
        _, _, _, full_dataX_, _, = create_datasets(dataset=variables[lag:, var1], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
        _, _, _, full_dataY, _, = create_datasets(dataset=variables[:-lag, var2], scaling_method=scaling_method,
                                                seq_length=seq_length, shift=shift)
        _, _, _, full_dataY_, _, = create_datasets(dataset=variables[lag:, var2], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
        
        if encoder_type == 1:
            _, _, _, full_dataX_sff, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var1]),
                                                    scaling_method=scaling_method, seq_length=seq_length, shift=shift)
            _, _, _, full_dataY_sff, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var2]),
                                                    scaling_method=scaling_method, seq_length=seq_length, shift=shift)
        else:
            full_dataX_sff = []
            full_dataY_sff = []

        # Save Time Series
        filename = models_folder + 'variables.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(variables, file)


        # Train Models

        filenameX = models_folder + 'auto_encoder_model_XYX.hdf5'
        filenameXP = models_folder + 'auto_encoder_model_XYXP.hdf5'
        filenameY = models_folder + 'auto_encoder_model_YXY.hdf5'
        filenameYP = models_folder + 'auto_encoder_model_YXYP.hdf5'
        plot_nameX = models_folder + 'Y to X training and validation.png'
        plot_nameY = models_folder + 'X to Y training and validation.png'

        ##############  Y to X  ##############
        auto_encoder_model_XYX, auto_encoder_model_XYXP = predictability_full(model_parameters, filenameX, filenameXP,
                                                                            full_dataX, full_dataX_, full_dataX_sff, 
                                                                            full_dataY, full_dataY_, full_dataY_sff,
                                                                            save_plot=plot_nameX)

        # ##############  X to Y  ##############
        auto_encoder_model_YXY, auto_encoder_model_YXYP = predictability_full(model_parameters, filenameY, filenameYP,
                                                                            full_dataY, full_dataY_, full_dataY_sff,
                                                                            full_dataX, full_dataX_, full_dataX_sff, 
                                                                            save_plot=plot_nameY)

    print('Training Done')

    # toc()

def temporal_bidirectional_boot_plot(model_parameters, window_len):
    # tic()

    model = model_parameters['model']
    directory = model_parameters['directory']
    method = model_parameters['method']
    length_vars = model_parameters['length_vars']
    discard = model_parameters['discard']
    init_cond = model_parameters['init_cond']
    encoder_type = model_parameters['encoder_type']
    var1 = model_parameters['var1']
    var2 = model_parameters['var2']
    couplings = np.array(model_parameters['couplings'])
    seq_length = model_parameters['seq_length']
    shift = model_parameters['shift']
    lag = model_parameters['lag']
    scaling_method = model_parameters['scaling_method']
    batch_size = model_parameters['batch_size']
    epochs = model_parameters['epochs']
    noise = model_parameters['noise']
    verbose = model_parameters['verbose']
    surrogates_samples = model_parameters['surrogates_samples']
    tol_default = model_parameters['tol_default']
    paint_fig = model_parameters['paint_fig']
    save_fig = model_parameters['save_fig']
    fulldata_testing = model_parameters['fulldata_testing']

    if encoder_type == 0:
        encoder_type_name = 'Padded_Encoder'
    elif encoder_type == 1:
        encoder_type_name = 'Shuffle_encoder'
    elif encoder_type == 2:
        encoder_type_name = 'Random_Encoder'

    models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(epochs) + \
                    '/SeqLen_' + str(seq_length) + '_Shift_' + str(shift) + '_Lag_' + str(lag) + \
                    '/Tolerance_' + str(tol_default) + '/window_length_' + str(window_len) + '/'
    
    filelist = glob.glob(os.path.join(models_folder, "score*"))

    print('Calculating Scores ...')

    if len(filelist) == 0:        

        default = 1e-4

        models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
            epochs) + '/SeqLen_' + str(seq_length) + '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/'

        filelist = glob.glob(os.path.join(models_folder, "*.hdf5"))

        if len(filelist) == 4:

            # Load Models
            filenameX = models_folder + 'auto_encoder_model_XYX.hdf5'
            auto_encoder_model_XYX = tf.keras.models.load_model(filenameX, custom_objects={'TCN': tcn.TCN})

            filenameXP = models_folder + 'auto_encoder_model_XYXP.hdf5'
            auto_encoder_model_XYXP = tf.keras.models.load_model(filenameXP, custom_objects={'TCN': tcn.TCN})

            filenameY = models_folder + 'auto_encoder_model_YXY.hdf5'
            auto_encoder_model_YXY = tf.keras.models.load_model(filenameY, custom_objects={'TCN': tcn.TCN})

            filenameYP = models_folder + 'auto_encoder_model_YXYP.hdf5'
            auto_encoder_model_YXYP = tf.keras.models.load_model(filenameYP, custom_objects={'TCN': tcn.TCN})

            # Generate Data
            filename = models_folder + 'variables.pkl'
            if os.path.exists(filename):  
                try:
                    with open(filename, 'rb') as file:
                        variables = pickle.load(file)
                except pickle.UnpicklingError:
                    pass
            
            else:
                filename = models_folder + 'variables.npy'
                if os.path.exists(filename):  
                    try:
                        variables = np.load(filename)
                    except Exception as e:
                        pass

            print('\t Data Loaded')

            _, _, testX, full_dataX, full_data_trackerX = create_datasets(dataset=variables[:-lag, var1], scaling_method=scaling_method,
                                                seq_length=seq_length, shift=shift)
            _, _, testX_, full_dataX_, full_data_trackerX_ = create_datasets(dataset=variables[lag:, var1], scaling_method=scaling_method,
                                                seq_length=seq_length, shift=shift)
            _, _, testY, full_dataY, full_data_trackerY = create_datasets(dataset=variables[:-lag, var2], scaling_method=scaling_method,
                                                seq_length=seq_length, shift=shift)
            _, _, testY_, full_dataY_, full_data_trackerY_ = create_datasets(dataset=variables[lag:, var2], scaling_method=scaling_method,
                                                seq_length=seq_length, shift=shift)
            
            if fulldata_testing:
                testX = full_dataX
                testX_ = full_dataX_
                testY = full_dataY
                testY_ = full_dataY_
            
            if window_len > len(testX_):
                window_len = len(testX_)

            models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(epochs) + \
                    '/SeqLen_' + str(seq_length) + '_Shift_' + str(shift) + '_Lag_' + str(lag) + \
                    '/Tolerance_' + str(tol_default) + '/window_length_' + str(window_len) + '/'

            if not os.path.exists(models_folder):
                    os.makedirs(models_folder)

            filelist = glob.glob(os.path.join(models_folder, "score*"))

            if len(filelist) == 0:

                score_no_boot = np.zeros((2, len(testX_) // window_len))
                score_boot_mean = np.zeros((2, len(testX_) // window_len))
                score_boot_std = np.zeros((2, len(testX_) // window_len))
                intervals = np.zeros((len(testX_) // window_len))

                for ii in range(len(testX_) // window_len):

                    interval = np.arange(ii * window_len, (ii + 1) * window_len)
                    intervals[ii] = np.squeeze(full_data_trackerX_[interval]).flatten()[-1]

                    ##############  Y to X  ##############
                    y_true = np.squeeze(testX_[interval]).flatten()
                    y_pred = np.squeeze(auto_encoder_model_XYX.predict([testX[interval],
                                    testY[interval]], batch_size=batch_size, verbose=verbose)).flatten()
                    # if encoder_type == 1:
                    #     y_predP = np.squeeze(auto_encoder_model_XYXP.predict([testX[interval],
                    #                                                           full_dataY_sff[interval]],
                    #                                                          batch_size=batch_size,
                    #                                                          verbose=verbose)).flatten()
                    if encoder_type == 2:
                        y_predP = np.squeeze(auto_encoder_model_XYXP.predict([testX[interval],
                                        np.random.random(testY[interval].shape)], batch_size=batch_size, verbose=verbose)).flatten()

                    ### Without Bootstraps ###
                    rse1 = r2_score(y_true, y_pred)
                    rse2 = r2_score(y_true, y_predP)

                    if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=tol_default, atol=tol_default)):
                        rse2 = default
                        rse1 = rse2
                    if rse2 < 0:
                        rse2 = default

                    score_no_boot[0, ii] = ((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2)))

                    ### Bootstraps ###
                    score = []
                    for jj in range(surrogates_samples):

                        indxs = np.random.choice(np.arange(len(interval)), size=len(interval), replace=True)
                        
                        boot_testX = testX[interval][indxs]
                        boot_testY = testY[interval][indxs]
                        boot_testX_ = testX_[interval][indxs]
                                            
                        y_true_boot = np.squeeze(boot_testX_.flatten())
                        y_pred_boot = np.squeeze(auto_encoder_model_XYX.predict([boot_testX, boot_testY], batch_size=batch_size, verbose=0)).flatten()
                        y_predP_boot = np.squeeze(auto_encoder_model_XYXP.predict([boot_testX, np.random.random(boot_testY.shape)], 
                                                                    batch_size=batch_size, verbose=0)).flatten()
                        
                        rse1 = r2_score(y_true_boot, y_pred_boot)
                        rse2 = r2_score(y_true_boot, y_predP_boot)

                        if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=tol_default, atol=tol_default)):
                            rse2 = default
                            rse1 = rse2
                        if rse2 < 0:
                            rse2 = default

                        score.append(((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2))))

                    score_boot_mean[0, ii] = np.mean(score)
                    score_boot_std[0, ii] = np.std(score)
                    
                    #############  X to Y  ##############
                    y_true = np.squeeze(testY_[interval]).flatten()
                    y_pred = np.squeeze(auto_encoder_model_YXY.predict([testY[interval],
                                    testX[interval]], batch_size=batch_size, verbose=verbose)).flatten()
                    # if encoder_type == 1:
                    #     y_predP = np.squeeze(auto_encoder_model_YXYP.predict([testY[interval],
                    #                                                           full_dataX_sff[interval]],
                    #                                                          batch_size=batch_size,
                    #                                                          verbose=verbose)).flatten()
                    if encoder_type == 2:
                        y_predP = np.squeeze(auto_encoder_model_YXYP.predict([testY[interval],
                                    np.random.random(testX[interval].shape)], batch_size=batch_size, verbose=verbose)).flatten()

                    ### Without Bootstraps ###

                    rse1 = r2_score(y_true, y_pred)
                    rse2 = r2_score(y_true, y_predP)

                    if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=tol_default, atol=tol_default)):
                        rse2 = default
                        rse1 = rse2
                    if rse2 < 0:
                        rse2 = default

                    score_no_boot[1, ii] = ((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2)))

                    ### Bootstraps ###
                    score = []
                    for jj in range(surrogates_samples):

                        indxs = np.random.choice(np.arange(len(interval)), size=len(interval), replace=True)

                        boot_testX = testX[interval][indxs]
                        boot_testY = testY[interval][indxs]
                        boot_testY_ = testY_[interval][indxs]

                        y_true_boot = np.squeeze(boot_testY_.copy()).flatten()
                        y_pred_boot = np.squeeze(auto_encoder_model_YXY.predict([boot_testY, boot_testX], batch_size=batch_size, verbose=0)).flatten()
                        y_predP_boot = np.squeeze(auto_encoder_model_YXYP.predict([boot_testY, np.random.random(boot_testX.shape)], 
                                                                batch_size=batch_size, verbose=0)).flatten()
                        
                        rse1 = r2_score(y_true_boot, y_pred_boot)
                        rse2 = r2_score(y_true_boot, y_predP_boot)

                        if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=tol_default, atol=tol_default)):
                            rse2 = default
                            rse1 = rse2
                        if rse2 < 0:
                            rse2 = default

                        score.append(((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2))))

                    score_boot_mean[1, ii] = np.mean(score)
                    score_boot_std[1, ii] = np.std(score)

                print('\t Saving Scores')

                # Save Scores
                filename = models_folder + 'score_no_boot.npy'
                np.save(filename, score_no_boot)
                filename = models_folder + 'intervals.npy'
                np.save(filename, intervals)
                filename = models_folder + 'score_boot_mean.npy'
                np.save(filename, score_boot_mean)
                filename = models_folder + 'score_boot_std.npy'
                np.save(filename, score_boot_std)

                print('\t Generating Plots')
                # Save Figure
                fig = plt.figure(figsize=(20, 5))
                plt.errorbar(intervals, score_boot_mean[0], yerr=score_boot_std[0], fmt='o-', color='red',
                            label='Bootstrap Y to X')
                plt.errorbar(intervals, score_boot_mean[1], yerr=score_boot_std[1], fmt='o-', color='blue',
                            label='Bootstrap X to Y')
                plt.plot(intervals, np.zeros((len(full_dataX_) // window_len)), '-k')
                plot_title = 'SeqLen_' + str(seq_length) + '_Shift_' + str(shift) + '_Lag_' + str(lag)
                plt.title(plot_title)
                plt.legend()

                if save_fig:
                    save_plot = models_folder + 'causal_inference_bootstraps.png'

                    plt.savefig(save_plot, bbox_inches="tight", pad_inches=1, transparent=False,
                                facecolor="w", edgecolor='w', orientation='landscape')

                if paint_fig:
                    plt.show()
                else:
                    plt.close(fig)

                # Save Figure
                fig = plt.figure(figsize=(20, 5))
                plt.plot(intervals, score_no_boot[0], 'o-', color='red', label='Raw Y to X')
                plt.plot(intervals, score_no_boot[1], 'o-', color='blue', label='Raw X to Y')
                plt.plot(intervals, np.zeros((len(full_dataX_) // window_len)), '-k')
                plot_title = 'SeqLen_' + str(seq_length) + '_Shift_' + str(shift) + '_Lag_' + str(lag)
                plt.title(plot_title)
                plt.legend()

                if save_fig:
                    save_plot = models_folder + 'causal_inference_no_bootstraps.png'

                    plt.savefig(save_plot, bbox_inches="tight", pad_inches=1, transparent=False,
                                facecolor="w", edgecolor='w', orientation='landscape')

                if paint_fig:
                    plt.show()
                else:
                    plt.close(fig)

        # toc()

def temporal_bidirectional_training_handler(model_parameters, variables):
    # tic()

    model = model_parameters['model']
    directory = model_parameters['directory']
    method = model_parameters['method']
    length_vars = model_parameters['length_vars']
    discard = model_parameters['discard']
    init_cond = model_parameters['init_cond']
    encoder_type = model_parameters['encoder_type']
    var1 = model_parameters['var1']
    var2 = model_parameters['var2']
    couplings = np.array(model_parameters['couplings'])
    seq_length = model_parameters['seq_length']
    shift = model_parameters['shift']
    lag = model_parameters['lag']
    scaling_method = model_parameters['scaling_method']
    batch_size = model_parameters['batch_size']
    epochs = model_parameters['epochs']
    noise = model_parameters['noise']

    if encoder_type == 0:
        encoder_type_name = 'Padded_Encoder'
    elif encoder_type == 1:
        encoder_type_name = 'Shuffle_encoder'
    elif encoder_type == 2:
        encoder_type_name = 'Random_Encoder'

    models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
        epochs) + '/SeqLen_' + str(seq_length) + \
                    '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/'
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    filelist = glob.glob(os.path.join(models_folder, "*.hdf5"))

    if len(filelist) != 4:

        trainX, valX, _, _, _, = create_datasets(dataset=variables[:-lag, var1], scaling_method=scaling_method,
                                                seq_length=seq_length, shift=shift)
        trainX_, valX_, _, _, _, = create_datasets(dataset=variables[lag:, var1], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
        trainY, valY, _, _, _, = create_datasets(dataset=variables[:-lag, var2], scaling_method=scaling_method,
                                                seq_length=seq_length, shift=shift)
        trainY_, valY_, _, _, _, = create_datasets(dataset=variables[lag:, var2], scaling_method=scaling_method,
                                                    seq_length=seq_length, shift=shift)
        
        if encoder_type == 1:
            trainX_sff, valX_sff, _, _, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var1]),
                                                    scaling_method=scaling_method, seq_length=seq_length, shift=shift)
            trainY_sff, valY_sff, _, _, _, = create_datasets(dataset=generate_fourier_surrogate(variables[:-lag, var2]),
                                                    scaling_method=scaling_method, seq_length=seq_length, shift=shift)
        else:
            trainX_sff, valX_sff = [], []
            trainY_sff, valY_sff = [], []

        # Save Time Series
        filename = models_folder + 'variables.npy'
        np.save(filename, variables)

        # Train Models
        parameters = {
            'nb_filters': nb_filters,
            'dilations': tcn_dilations,
            'kernel_size': kernel_size,
            'nb_stacks': 1,
            'dropout_rate_tcn': dropout_rate_tcn,
            'dropout_rate_hidden': dropout_rate_hidden,
            'kernel_regularizer': kernel_regularizer,
            'conv_kernel_init': conv_kernel_init,
            'padding': 'causal',
            'tcn_act_funct' : tcn_act_funct,
            'latent_sample_rate': latent_sample_rate,
            'noise': noise,
            'act_funct': act_funct,
            'filters_conv1d': filters_conv1d,
            'kernel_size_conv1d' : kernel_size_conv1d, 
            'activation_conv1d': activation_conv1d,
            'ts_dimension': 1,
            'layer_nodes': layer_nodes,
            'n_layers': n_layers,
            'seq_length': seq_length,
            'pad_encoder': False,
            'encoder_type': encoder_type,
            'concatenate_layers': False,
            'batch_size': batch_size,
            'epochs': epochs,
            'verbose': verbose,
            'loss_funct': loss_funct,
            'shuffle': shuffle,
            'surrogates_samples': surrogates_samples,
            'paint_fig': paint_fig,
            'save_fig': save_fig,
            'model_summary': False}

        # Train Models

        filenameX = models_folder + 'auto_encoder_model_XYX.hdf5'
        filenameXP = models_folder + 'auto_encoder_model_XYXP.hdf5'
        filenameY = models_folder + 'auto_encoder_model_YXY.hdf5'
        filenameYP = models_folder + 'auto_encoder_model_YXYP.hdf5'
        plot_nameX = models_folder + 'Y to X training and validation.png'
        plot_nameY = models_folder + 'X to Y training and validation.png'

        ##############  Y to X  ##############
        auto_encoder_model_XYX, auto_encoder_model_XYXP = predictability(parameters, filenameX, filenameXP,
                                                                            trainX, valX, trainX_, valX_, trainY, valY,
                                                                            trainY_sff, valY_sff, save_plot=plot_nameX)

        # ##############  X to Y  ##############
        auto_encoder_model_YXY, auto_encoder_model_YXYP = predictability(parameters, filenameY, filenameYP,
                                                                            trainY, valY, trainY_, valY_, trainX, valX,
                                                                            trainX_sff, valX_sff, save_plot=plot_nameY)
    # toc()


def monkey_temporal_model_handler(model_parameters, variables):
    # tic()

    nb_filters = model_parameters['nb_filters']
    tcn_dilations = model_parameters['tcn_dilations']
    kernel_size = model_parameters['kernel_size']
    dropout_rate = model_parameters['dropout_rate']
    kernel_regularizer = model_parameters['kernel_regularizer']
    conv_kernel_init = model_parameters['conv_kernel_init']
    latent_sample_rate = model_parameters['latent_sample_rate']
    act_funct = model_parameters['act_funct']
    filters_conv1d = model_parameters['filters_conv1d']
    activation_conv1d = model_parameters['activation_conv1d']
    n_layers = model_parameters['n_layers']
    layer_nodes = model_parameters['layer_nodes']
    model = model_parameters['model']
    directory = model_parameters['directory']
    encoder_type = model_parameters['encoder_type']
    var1 = model_parameters['var1']
    var2 = model_parameters['var2']
    seq_length = model_parameters['seq_length']
    shift = model_parameters['shift']
    lag = model_parameters['lag']
    scaling_method = model_parameters['scaling_method']
    batch_size = model_parameters['batch_size']
    epochs = model_parameters['epochs']
    noise = model_parameters['noise']
    verbose = model_parameters['verbose']
    loss_funct = model_parameters['loss_funct']
    shuffle = model_parameters['shuffle']
    bootstrap_samples = model_parameters['bootstrap_samples']
    paint_fig = model_parameters['paint_fig']
    save_fig = model_parameters['save_fig']
    subsampling_freq = model_parameters['subsampling_freq']

    if encoder_type == 0:
        encoder_type_name = 'Padded_Encoder'
    elif encoder_type == 1:
        encoder_type_name = 'Shuffle_encoder'
    elif encoder_type == 2:
        encoder_type_name = 'Random_Encoder'

    models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
        epochs) + '/SeqLen_' + str(seq_length) + '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/'
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    filelist = glob.glob(os.path.join(models_folder, "*.hdf5"))

    # Generate Data
    trainX, valX, _, _ = create_alternating_datasets(dataset=variables[:-lag, var1],
                                                     scaling_method=scaling_method, seq_length=seq_length, shift=shift,
                                                     subsample=subsampling_freq)
    trainX_, valX_, _, _ = create_alternating_datasets(dataset=variables[lag:, var1],
                                                       scaling_method=scaling_method, seq_length=seq_length,
                                                       shift=shift, subsample=subsampling_freq)
    trainY, valY, _, _ = create_alternating_datasets(dataset=variables[:-lag, var2],
                                                     scaling_method=scaling_method, seq_length=seq_length, shift=shift,
                                                     subsample=subsampling_freq)
    trainY_, valY_, _, _ = create_alternating_datasets(dataset=variables[lag:, var2],
                                                       scaling_method=scaling_method, seq_length=seq_length,
                                                       shift=shift, subsample=subsampling_freq)

    if encoder_type == 1:
        ts = variables[:-lag, var1].copy()
        ts_fourier = np.fft.rfft(ts)
        random_phases = np.exp(np.random.uniform(0, np.pi, len(ts) // 2 + 1) * 1.0j)
        ts_fourier_new = ts_fourier * random_phases
        surr_var1 = np.fft.irfft(ts_fourier_new)

        del ts, ts_fourier, random_phases, ts_fourier_new

        ts = variables[:-lag, var2].copy()
        ts_fourier = np.fft.rfft(ts)
        random_phases = np.exp(np.random.uniform(0, np.pi, len(ts) // 2 + 1) * 1.0j)
        ts_fourier_new = ts_fourier * random_phases
        surr_var2 = np.fft.irfft(ts_fourier_new)

        trainX_sff, valX_sff, _, _ = create_alternating_datasets(dataset=surr_var1,
                                                                 scaling_method=scaling_method, seq_length=seq_length,
                                                                 shift=shift, subsample=subsampling_freq)
        trainY_sff, valY_sff, _, _ = create_alternating_datasets(dataset=surr_var2,
                                                                 scaling_method=scaling_method, seq_length=seq_length,
                                                                 shift=shift, subsample=subsampling_freq)

        del ts, ts_fourier, random_phases, ts_fourier_new

    parameters = {
        'nb_filters': nb_filters,
        'dilations': tcn_dilations,
        'kernel_size': kernel_size,
        'nb_stacks': 1,
        'dropout_rate': dropout_rate,
        'kernel_regularizer': kernel_regularizer,
        'conv_kernel_init': conv_kernel_init,
        'padding': 'causal',
        'latent_sample_rate': latent_sample_rate,
        'noise': noise,
        'act_funct': act_funct,
        'filters_conv1d': filters_conv1d,
        'activation_conv1d': activation_conv1d,
        'ts_dimension': 1,
        'layer_nodes': layer_nodes,
        'n_layers': n_layers,
        'seq_length': seq_length,
        'pad_encoder': False,
        'encoder_type': encoder_type,
        'concatenate_layers': False,
        'batch_size': batch_size,
        'epochs': epochs,
        'verbose': verbose,
        'loss_funct': loss_funct,
        'shuffle': shuffle,
        'bootstrap_samples': bootstrap_samples,
        'paint_fig': paint_fig,
        'save_fig': save_fig,
        'model_summary': False}

    # Train Models

    filenameX = models_folder + 'auto_encoder_model_XYX.hdf5'
    filenameXP = models_folder + 'auto_encoder_model_XYXP.hdf5'
    filenameY = models_folder + 'auto_encoder_model_YXY.hdf5'
    filenameYP = models_folder + 'auto_encoder_model_YXYP.hdf5'
    plot_nameX = models_folder + 'Y to X training and validation.png'
    plot_nameY = models_folder + 'X to Y training and validation.png'

    # ##############  X to Y  ##############
    auto_encoder_model_YXY, auto_encoder_model_YXYP = predictability(parameters, filenameY, filenameYP,
                                                                     trainY, valY, trainY_, valY_, trainX, valX,
                                                                     trainX_sff, valX_sff, save_plot=plot_nameY)

    # toc()


def monkey_temporal_model_boot_plot(model_parameters, variables, window_len, condition_index='', condition_label=''):
    # tic()

    model = model_parameters['model']
    directory = model_parameters['directory']
    method = model_parameters['method']
    length_vars = model_parameters['length_vars']
    discard = model_parameters['discard']
    init_cond = model_parameters['init_cond']
    encoder_type = model_parameters['encoder_type']
    var1 = model_parameters['var1']
    var2 = model_parameters['var2']
    couplings = model_parameters['couplings']
    seq_length = model_parameters['seq_length']
    shift = model_parameters['shift']
    lag = model_parameters['lag']
    scaling_method = model_parameters['scaling_method']
    batch_size = model_parameters['batch_size']
    epochs = model_parameters['epochs']
    noise = model_parameters['noise']
    verbose = model_parameters['verbose']
    bootstrap_samples = model_parameters['bootstrap_samples']
    paint_fig = model_parameters['paint_fig']
    save_fig = model_parameters['save_fig']
    subsampling_freq = model_parameters['subsampling_freq']

    if encoder_type == 0:
        encoder_type_name = 'Padded_Encoder'
    elif encoder_type == 1:
        encoder_type_name = 'Shuffle_encoder'
    elif encoder_type == 2:
        encoder_type_name = 'Random_Encoder'

    models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(
        epochs) + '/SeqLen_' + str(seq_length) + '_Shift_' + str(shift) + '_Lag_' + str(lag) + \
                    '/Scores_Bootstraps_Full_Data/window_length_' + str(window_len) + '/'
    filelist = glob.glob(os.path.join(models_folder, "score*"))

    default = 1e-10

    models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(epochs) + \
                    '/SeqLen_' + str(seq_length) + '_Shift_' + str(shift) + '_Lag_' + str(lag) + '/'

    _, _, full_dataX, _ = create_alternating_datasets(dataset=variables[:-lag, var1],
                                                      scaling_method=scaling_method, seq_length=seq_length, shift=shift,
                                                      subsample=subsampling_freq)
    _, _, full_dataY, full_data_trackerY = create_alternating_datasets(dataset=variables[:-lag, var2],
                                                                       scaling_method=scaling_method,
                                                                       seq_length=seq_length, shift=shift,
                                                                       subsample=subsampling_freq)
    _, _, full_dataY_, full_data_trackerY_ = create_alternating_datasets(dataset=variables[lag:, var2],
                                                                         scaling_method=scaling_method,
                                                                         seq_length=seq_length, shift=shift,
                                                                         subsample=subsampling_freq)

    filelist = glob.glob(os.path.join(models_folder, "*.hdf5"))

    if len(filelist) == 2:

        # Load Models
        filenameY = models_folder + 'auto_encoder_model_YXY.hdf5'
        auto_encoder_model_YXY = tf.keras.models.load_model(filenameY, custom_objects={'TCN': tcn.TCN})

        filenameYP = models_folder + 'auto_encoder_model_YXYP.hdf5'
        auto_encoder_model_YXYP = tf.keras.models.load_model(filenameYP, custom_objects={'TCN': tcn.TCN})

        score_no_boot = np.zeros((len(full_dataY_) // window_len))
        score_boot_mean = np.zeros((len(full_dataY_) // window_len))
        score_boot_std = np.zeros((len(full_dataY_) // window_len))
        intervals = np.zeros((len(full_dataY_) // window_len))

        # printProgressBar(0, len(testX_)//window_len, prefix='Progress: ', suffix='Complete', length=50)
        for ii in range(len(full_dataY_) // window_len):

            interval = np.arange(ii * window_len, (ii + 1) * window_len)
            intervals[ii] = np.squeeze(full_data_trackerY_[interval]).flatten()[-1]

            ##############  X to Y  ##############
            y_true = np.squeeze(full_dataY_[interval]).flatten()
            y_pred = np.squeeze(auto_encoder_model_YXY.predict([full_dataY[interval],
                                                                full_dataX[interval]], batch_size=batch_size,
                                                               verbose=verbose)).flatten()
            y_predP = np.squeeze(auto_encoder_model_YXYP.predict([full_dataY[interval],
                                                                  np.random.random(full_dataX[interval].shape)],
                                                                 batch_size=batch_size,
                                                                 verbose=verbose)).flatten()

            ### Without Bootstraps ###
            ave_score = True

            rse1 = r2_score(y_true, y_pred)
            rse2 = r2_score(y_true, y_predP)

            if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=default, atol=default)):
                rse2 = default
                rse1 = rse2
                ave_score = False
            if rse2 < 0:
                rse2 = default
                ave_score = False

            if ave_score:
                score_no_boot[ii] = ((rse1 - rse2) / (0.5 * np.abs(rse1 + rse2)))
            else:
                score_no_boot[ii] = ((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2)))

            ### Bootstraps ###
            if bootstrap_samples > 0:
                score = []
                for jj in range(bootstrap_samples):

                    ave_score = True

                    indxs = np.random.choice(np.arange(y_true.shape[0]), size=y_true.shape[0])
                    y_true_boot = y_true[indxs]
                    y_pred_boot = y_pred[indxs]
                    y_predP_boot = y_predP[indxs]

                    rse1 = r2_score(y_true_boot, y_pred_boot)
                    rse2 = r2_score(y_true_boot, y_predP_boot)

                    if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=default, atol=default)):
                        rse2 = default
                        rse1 = rse2
                        ave_score = False
                    if rse2 < 0:
                        rse2 = default
                        ave_score = False

                    if ave_score:
                        score.append(((rse1 - rse2) / (0.5 * np.abs(rse1 + rse2))))
                    else:
                        score.append(((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2))))

                score_boot_mean[ii] = np.mean(score)
                score_boot_std[ii] = np.std(score)

            # printProgressBar(ii+1, len(testX_)//window_len, prefix='Progress: ', suffix='Complete', length=50)

        models_folder = directory + encoder_type_name + '/Noise_' + str(noise) + '/Epochs_' + str(epochs) + \
                        '/SeqLen_' + str(seq_length) + '_Shift_' + str(shift) + '_Lag_' + str(lag) + \
                        '/Scores_Bootstraps_Full_Data/window_length_' + str(window_len) + '/'
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)

        # Save Scores
        filename = models_folder + 'score_no_boot.npy'
        np.save(filename, score_no_boot)
        if bootstrap_samples > 0:
            filename = models_folder + 'score_boot_mean.npy'
            np.save(filename, score_boot_mean)
            filename = models_folder + 'score_boot_std.npy'
            np.save(filename, score_boot_std)

        # Save Figure
        # fig = plt.figure(figsize=(20, 5))
        # plt.errorbar(intervals, score_boot_mean[0], yerr=score_boot_std[0], fmt='o-', color='red',
        #              label='Bootstrap Y to X')
        # plt.errorbar(intervals, score_boot_mean[1], yerr=score_boot_std[1], fmt='o-', color='blue',
        #              label='Bootstrap X to Y')
        # plt.plot(intervals, np.zeros((len(full_dataY_) // window_len)), '-k')
        # for kk in range(len(condition_index)):
        #     plt.vlines(x=condition_index[kk], ymin=0, ymax=2, colors='purple', ls='--', lw=2,
        #                label=condition_label[kk])
        # plot_title = 'SeqLen_' + str(seq_length) + '_Shift_' + str(shift) + '_Lag_' + str(lag)
        # if subsampling_freq > 1:
        #     plot_title = plot_title + ' Subsampled by ' + str(subsampling_freq)
        # plt.title(plot_title)
        # plt.legend()
        #
        # if save_fig:
        #     save_plot = models_folder + 'causal_inference_bootstraps.png'
        #
        #     plt.savefig(save_plot, bbox_inches="tight", pad_inches=1, transparent=False,
        #                 facecolor="w", edgecolor='w', orientation='landscape')
        #
        # if paint_fig:
        #     plt.show()
        # else:
        #     plt.close(fig)
        #
        # Save Figure
        fig = plt.figure(figsize=(20, 5))
        # plt.plot(intervals, score_no_boot[0], 'o-', color='red', label='Raw Y to X')
        plt.plot(intervals, score_no_boot, 'o-', color='blue', label=' X to Y')
        plt.plot(intervals, np.zeros((len(full_dataY_) // window_len)), '-k')

        if condition_index != '':
            for kk in range(len(condition_index)):
                plt.vlines(x=condition_index[kk], ymin=0, ymax=0.5, colors='purple', ls='--', lw=2)
                plt.text(x=condition_index[kk], y=0.25, s=condition_label[kk], fontsize='xx-small', ha='center',
                         va='center', rotation='vertical', backgroundcolor='white')
        # plt.xticks(intervals, intervals//1000)
        plt.xlabel('Time')
        plot_title = 'SeqLen_' + str(seq_length) + '_Shift_' + str(shift) + '_Lag_' + str(lag)
        if subsampling_freq > 1:
            plot_title = plot_title + ' Subsampled by ' + str(subsampling_freq)
        plt.title(plot_title)
        plt.legend()

        if save_fig:
            save_plot = models_folder + 'causal_inference_raw.png'

            plt.savefig(save_plot, bbox_inches="tight", pad_inches=1, transparent=False,
                        facecolor="w", edgecolor='w', orientation='landscape')

        if paint_fig:
            plt.show()
        else:
            plt.close(fig)

        # toc()

