

import os, sys, time, glob, shutil
import numpy as np
import pickle
import h5py
import copy
import matplotlib.pyplot as plt

try:
    from model_structure import Encoder, Bottleneck, Decoder
except ImportError:
    from .model_structure import Encoder, Bottleneck, Decoder
    
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def predictability_split(parameters, filenameX, filenameXP, trainX, valX, trainX_, valX_,
                   trainY, valY, trainY_sff, valY_sff, save_plot=''):
    ##################################### XY to X #####################################

    # Use Output Encoder Y
    parameters.update({'pad_encoder': False})

    encoder_XX = Encoder(parameters)
    encoder_YX = Encoder(parameters)
    bottleneck_XYX = Bottleneck(parameters)
    decoder_XYX = Decoder(parameters)

    encoder_input_X = Input(batch_shape=(None, parameters['seq_length'], parameters['ts_dimension']))
    encoder_input_Y = Input(batch_shape=(None, parameters['seq_length'], parameters['ts_dimension']))

    # Compile Autoencoder
    auto_encoder_model_XYX = Model(inputs=[encoder_input_X, encoder_input_Y],
                                   outputs=decoder_XYX(
                                       bottleneck_XYX([encoder_XX([encoder_input_X]), encoder_YX([encoder_input_Y])])),
                                   name='AutoEncoder_XYX')
    auto_encoder_model_XYX.compile(loss=parameters['loss_funct'], optimizer='Adam', metrics=parameters['loss_funct'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=parameters['verbose'], mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filenameX, monitor='val_loss', mode='min', verbose=parameters['verbose'], save_best_only=True,
                                 save_weights_only=False)

    historyXYX = auto_encoder_model_XYX.fit([trainX, trainY], trainX_, validation_data=([valX, valY], valX_),
                                            batch_size=parameters['batch_size'], epochs=parameters['epochs'],
                                            shuffle=parameters['shuffle'], use_multiprocessing=True,
                                            callbacks=[checkpoint, early_stopping, reduce_lr],
                                            verbose=parameters['verbose'])

    ##################################### XY to X updated #####################################

    if parameters['encoder_type'] == 0:
        padded_encoder = True
        shuffle_encoder = False
        random_encoder = False
    elif parameters['encoder_type'] == 1:
        padded_encoder = False
        shuffle_encoder = True
        random_encoder = False
    elif parameters['encoder_type'] == 2:
        padded_encoder = False
        shuffle_encoder = False
        random_encoder = True

    if padded_encoder:
        # Make Output Encoder Y to 0
        parameters.update({'pad_encoder': True})

        encoder_XXP = Encoder(parameters)
        encoder_YXP = Encoder(parameters)
        bottleneck_XYXP = Bottleneck(parameters)
        decoder_XYXP = Decoder(parameters)

        # Make Y Weights 0
        old_weigths = encoder_YXP.get_weights()
        new_weigths = old_weigths.copy()

        for ii in range(len(old_weigths[0])):
            new_weigths[ii] = np.zeros_like(old_weigths[ii])

        encoder_YXP.set_weights(new_weigths)

        # Freeze Encoder Layers
        for layer in encoder_YXP.layers[:]:
            layer.trainable = False

    elif shuffle_encoder:
        # Use Output Encoder Y
        parameters.update({'pad_encoder': False})

        encoder_XXP = Encoder(parameters)
        encoder_YXP = Encoder(parameters)
        bottleneck_XYXP = Bottleneck(parameters)
        decoder_XYXP = Decoder(parameters)

        trainY = trainY_sff.copy()
        valY = valY_sff.copy()

    elif random_encoder:
        # Use Output Encoder Y
        parameters.update({'pad_encoder': False})

        encoder_XXP = Encoder(parameters)
        encoder_YXP = Encoder(parameters)
        bottleneck_XYXP = Bottleneck(parameters)
        decoder_XYXP = Decoder(parameters)

        trainY = np.random.random(trainY.shape)
        valY = np.random.random(valY.shape)

    encoder_input_X = Input(batch_shape=(None, parameters['seq_length'], parameters['ts_dimension']))
    encoder_input_Y = Input(batch_shape=(None, parameters['seq_length'], parameters['ts_dimension']))

    # Compile Autoencoder
    auto_encoder_model_XYXP = Model(inputs=[encoder_input_X, encoder_input_Y],
                                    outputs=decoder_XYXP(bottleneck_XYXP(
                                        [encoder_XXP([encoder_input_X]), encoder_YXP([encoder_input_Y])])),
                                    name='AutoEncoder_XYXP')
    auto_encoder_model_XYXP.compile(loss=parameters['loss_funct'], optimizer='Adam', metrics=parameters['loss_funct'])

    # print("\n> Starting Training XYX ...")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=parameters['verbose'], mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filenameXP, monitor='val_loss', mode='min', verbose=parameters['verbose'], save_best_only=True,
                                 save_weights_only=False)

    historyXYXP = auto_encoder_model_XYXP.fit([trainX, trainY], trainX_, validation_data=([valX, valY], valX_),
                                              batch_size=parameters['batch_size'], epochs=parameters['epochs'],
                                              shuffle=parameters['shuffle'], use_multiprocessing=True,
                                              callbacks=[checkpoint, early_stopping, reduce_lr],
                                              verbose=parameters['verbose'])

    fig = plt.figure(figsize=(20, 5))
    gs = plt.GridSpec(1, 2, figure=fig)

    ax = fig.add_subplot(gs[:, 0])
    ax.semilogy(np.arange(len(historyXYX.history['loss'])),
                np.sqrt(historyXYX.history['loss']), 'k-', label='Training Loss')
    ax.semilogy(np.arange(len(historyXYX.history['val_loss'])),
                np.sqrt(historyXYX.history['val_loss']), 'k:', label='Validation Loss')
    ax.set_title('XY --> X shifted')

    ax = fig.add_subplot(gs[:, 1])
    ax.semilogy(np.arange(len(historyXYXP.history['loss'])),
                np.sqrt(historyXYXP.history['loss']), 'k-', label='Training Loss')
    ax.semilogy(np.arange(len(historyXYXP.history['val_loss'])),
                np.sqrt(historyXYXP.history['val_loss']), 'k:', label='Validation Loss')
    ax.set_title('XY padded --> X shifted')

    plt.legend()

    if save_plot != '':
        plt.savefig(save_plot, bbox_inches="tight", pad_inches=1, transparent=False,
                    facecolor="w", edgecolor='w', orientation='landscape')

    if parameters['paint_fig']:
        plt.show()
    else:
        plt.close(fig)

    return auto_encoder_model_XYX, auto_encoder_model_XYXP

def predictability_full(parameters, filenameX, filenameXP, 
                        full_dataX, full_dataX_, full_dataX_sff, 
                        full_dataY, full_dataY_, full_dataY_sff, save_plot=''):
    
    ##################################### XY to X #####################################

    # Use Output Encoder Y
    parameters.update({'pad_encoder': False})

    encoder_XX = Encoder(parameters)
    encoder_YX = Encoder(parameters)
    bottleneck_XYX = Bottleneck(parameters)
    decoder_XYX = Decoder(parameters)

    encoder_input_X = Input(batch_shape=(None, parameters['seq_length'], parameters['ts_dimension']))
    encoder_input_Y = Input(batch_shape=(None, parameters['seq_length'], parameters['ts_dimension']))

    # Compile Autoencoder
    auto_encoder_model_XYX = Model(inputs=[encoder_input_X, encoder_input_Y],
                                   outputs=decoder_XYX(
                                       bottleneck_XYX([encoder_XX([encoder_input_X]), encoder_YX([encoder_input_Y])])),
                                   name='AutoEncoder_XYX')
    auto_encoder_model_XYX.compile(loss=parameters['loss_funct'], optimizer='Adam', metrics=parameters['loss_funct'])

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-7, verbose=parameters['verbose'], mode='min')
    early_stopping = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filenameX, monitor='loss', mode='min', verbose=parameters['verbose'], save_best_only=True,
                                 save_weights_only=False)

    historyXYX = auto_encoder_model_XYX.fit([full_dataX, full_dataY], full_dataX_, 
                                            batch_size=parameters['batch_size'], epochs=parameters['epochs'],
                                            shuffle=parameters['shuffle'], use_multiprocessing=True,
                                            callbacks=[checkpoint, early_stopping, reduce_lr],
                                            verbose=parameters['verbose'])

    ##################################### XY to X updated #####################################

    if parameters['encoder_type'] == 0:
        padded_encoder = True
        shuffle_encoder = False
        random_encoder = False
    elif parameters['encoder_type'] == 1:
        padded_encoder = False
        shuffle_encoder = True
        random_encoder = False
    elif parameters['encoder_type'] == 2:
        padded_encoder = False
        shuffle_encoder = False
        random_encoder = True

    if padded_encoder:
        # Make Output Encoder Y to 0
        parameters.update({'pad_encoder': True})

        encoder_XXP = Encoder(parameters)
        encoder_YXP = Encoder(parameters)
        bottleneck_XYXP = Bottleneck(parameters)
        decoder_XYXP = Decoder(parameters)

        # Make Y Weights 0
        old_weigths = encoder_YXP.get_weights()
        new_weigths = old_weigths.copy()

        for ii in range(len(old_weigths[0])):
            new_weigths[ii] = np.zeros_like(old_weigths[ii])

        encoder_YXP.set_weights(new_weigths)

        # Freeze Encoder Layers
        for layer in encoder_YXP.layers[:]:
            layer.trainable = False

    elif shuffle_encoder:
        # Use Output Encoder Y
        parameters.update({'pad_encoder': False})

        encoder_XXP = Encoder(parameters)
        encoder_YXP = Encoder(parameters)
        bottleneck_XYXP = Bottleneck(parameters)
        decoder_XYXP = Decoder(parameters)

        full_dataY = full_dataY_sff.copy()

    elif random_encoder:
        # Use Output Encoder Y
        parameters.update({'pad_encoder': False})

        encoder_XXP = Encoder(parameters)
        encoder_YXP = Encoder(parameters)
        bottleneck_XYXP = Bottleneck(parameters)
        decoder_XYXP = Decoder(parameters)

        full_dataY = np.random.random(full_dataY.shape)

    encoder_input_X = Input(batch_shape=(None, parameters['seq_length'], parameters['ts_dimension']))
    encoder_input_Y = Input(batch_shape=(None, parameters['seq_length'], parameters['ts_dimension']))

    # Compile Autoencoder
    auto_encoder_model_XYXP = Model(inputs=[encoder_input_X, encoder_input_Y],
                                    outputs=decoder_XYXP(bottleneck_XYXP(
                                        [encoder_XXP([encoder_input_X]), encoder_YXP([encoder_input_Y])])),
                                    name='AutoEncoder_XYXP')
    auto_encoder_model_XYXP.compile(loss=parameters['loss_funct'], optimizer='Adam', metrics=parameters['loss_funct'])

    # print("\n> Starting Training XYX ...")
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-7, verbose=parameters['verbose'], mode='min')
    early_stopping = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filenameXP, monitor='loss', mode='min', verbose=parameters['verbose'], save_best_only=True,
                                 save_weights_only=False)

    historyXYXP = auto_encoder_model_XYXP.fit([full_dataX, full_dataY], full_dataX_,
                                              batch_size=parameters['batch_size'], epochs=parameters['epochs'],
                                              shuffle=parameters['shuffle'], use_multiprocessing=True,
                                              callbacks=[checkpoint, early_stopping, reduce_lr],
                                              verbose=parameters['verbose'])

    fig = plt.figure(figsize=(20, 5))
    gs = plt.GridSpec(1, 2, figure=fig)

    ax = fig.add_subplot(gs[:, 0])
    ax.semilogy(np.arange(len(historyXYX.history['loss'])),
                np.sqrt(historyXYX.history['loss']), 'k-', label='Training Loss')
    ax.set_title('XY --> X shifted')

    ax = fig.add_subplot(gs[:, 1])
    ax.semilogy(np.arange(len(historyXYXP.history['loss'])),
                np.sqrt(historyXYXP.history['loss']), 'k-', label='Training Loss')
    ax.set_title('XY padded --> X shifted')

    plt.legend()

    if save_plot != '':
        plt.savefig(save_plot, bbox_inches="tight", pad_inches=1, transparent=False,
                    facecolor="w", edgecolor='w', orientation='landscape')

    if parameters['paint_fig']:
        plt.show()
    else:
        plt.close(fig)

    return auto_encoder_model_XYX, auto_encoder_model_XYXP