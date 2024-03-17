import os, sys, time, glob, shutil
import numpy as np
import copy

from tcn import tcn
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Conv1D, Reshape, Flatten, GaussianNoise, Lambda, \
    Add, Multiply
from tensorflow.keras.layers import UpSampling1D, AveragePooling1D, Multiply, MaxPooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

############################################################################################
#########################################            #######################################
######################################### FULL MODEL #######################################
#########################################            #######################################
############################################################################################

######################################## ENCODER ########################################
def Encoder(parameters):
    
    encoder_inputs = Input(batch_shape=(None, parameters['seq_length'], parameters['ts_dimension']))
    
    if parameters['noise'] > 0:
        encoder = GaussianNoise(stddev=parameters['noise'])(encoder_inputs)
    else:
        encoder = encoder_inputs
    
    encoder = tcn.TCN(nb_filters=parameters['nb_filters'], kernel_size=parameters['kernel_size'], nb_stacks=parameters['nb_stacks'], 
                        dilations= parameters['dilations'], activation=parameters['tcn_act_funct'], padding=parameters['padding'], 
                        use_skip_connections=True, dropout_rate=parameters['dropout_rate_tcn'], return_sequences=True,
                        kernel_initializer=parameters['conv_kernel_init'])(encoder_inputs)

    # encoder = tcn.TCN(nb_filters=parameters['nb_filters'], kernel_size=parameters['kernel_size'], 
    #                     padding=parameters['padding'], 
    #                     dropout_rate=parameters['dropout_rate'], return_sequences=True)(encoder_inputs)
    
    encoder = Conv1D(filters=parameters['filters_conv1d'],
                     kernel_size=parameters['kernel_size_conv1d'], activation=parameters['activation_conv1d'], padding=parameters['padding'])(encoder)

    encoder = AveragePooling1D(pool_size=parameters['latent_sample_rate'], strides=None, padding='valid',
                               data_format='channels_last')(encoder)

    encoder = Activation(parameters['act_funct'])(encoder)

    encoder_ouputs = Flatten()(encoder)

    layer_nodes = parameters['layer_nodes']

    for ii in range(parameters['n_layers'] - 1):

        if layer_nodes < encoder_ouputs.shape[1]:

            encoder_ouputs = Dense(layer_nodes, activation=parameters['act_funct'])(encoder_ouputs)
            
            if parameters['dropout_rate_hidden'] > 0:
                encoder_ouputs = Dropout(parameters['dropout_rate_hidden'])(encoder_ouputs)

        layer_nodes = layer_nodes // 2

    if layer_nodes < encoder_ouputs.shape[1]:

        encoder_ouputs = Dense(layer_nodes, activation=parameters['act_funct'])(encoder_ouputs)

    Encoder_model = Model(inputs=[encoder_inputs], outputs=[encoder_ouputs])
    
    return Encoder_model

######################################## Bottleneck  ########################################
def Bottleneck(parameters):
    
    layer_nodes = parameters['layer_nodes'] // (2 ** (parameters['n_layers'] - 1))

    bottleneck_inputs_X = Input(batch_shape=(None, layer_nodes), name='Input_Bot_X')

    bottleneck_inputs_Y = Input(batch_shape=(None, layer_nodes), name='Input_Bot_Y')

    bottleneck_outputs = Multiply()([bottleneck_inputs_X, bottleneck_inputs_Y])

    Bottleneck_model = Model(inputs=[bottleneck_inputs_X, bottleneck_inputs_Y],
                             outputs=[bottleneck_outputs], name='Bottleneck')

    return Bottleneck_model

######################################## DECODER XY ########################################
def Decoder(parameters):
    
    layer_nodes = parameters['layer_nodes'] // (2 ** (parameters['n_layers'] - 1))

    decoder_inputs = Input(batch_shape=(None, layer_nodes), name='Input_Dec')

    decoder = decoder_inputs

    if parameters['seq_length'] // parameters['latent_sample_rate'] != 1:

        for ii in range(parameters['n_layers'] - 1):

            layer_nodes = layer_nodes * 2

            if layer_nodes < (parameters['seq_length'] // parameters['latent_sample_rate'] * (parameters['nb_filters']//2)):

                decoder = Dense(layer_nodes, activation=parameters['act_funct'])(decoder)

                if parameters['dropout_rate_hidden'] > 0:
                    decoder = Dropout(parameters['dropout_rate_hidden'])(decoder)

        decoder = Dense( parameters['seq_length'] // parameters['latent_sample_rate'] * (parameters['nb_filters']//2),
                        activation=parameters['act_funct'])(decoder)

    decoder = Reshape((parameters['seq_length'] // parameters['latent_sample_rate'], parameters['nb_filters']//2))(
        decoder)

    decoder = UpSampling1D(size=parameters['latent_sample_rate'])(decoder)

    # decoder = tcn.TCN(nb_filters=parameters['nb_filters'], kernel_size=parameters['kernel_size'], 
    #                 padding=parameters['padding'], 
    #                 dropout_rate=parameters['dropout_rate'], return_sequences=True)(decoder)

    decoder = tcn.TCN(nb_filters=parameters['nb_filters'], kernel_size=parameters['kernel_size'], nb_stacks=parameters['nb_stacks'], 
                      dilations= parameters['dilations'], activation=parameters['tcn_act_funct'], padding=parameters['padding'], 
                      use_skip_connections=True, dropout_rate=parameters['dropout_rate_tcn'], return_sequences=True,
                      kernel_initializer=parameters['conv_kernel_init'])(decoder)
    
    decoder_outputs = Dense(parameters['ts_dimension'], activation=parameters['act_funct'])(decoder)

    Decoder_model = Model(inputs=[decoder_inputs], outputs=[decoder_outputs], name='Decoder')
    
    return Decoder_model


# ######################################## ENCODER ########################################
# def Encoder(parameters):
#     if parameters['noise'] > 0:

#         encoder_inputs = Input(batch_shape=(None, parameters['seq_length'],
#                                             parameters['ts_dimension']))

#         encoder = GaussianNoise(stddev=parameters['noise'])(encoder_inputs)

#         encoder = tcn.TCN(nb_filters=parameters['nb_filters'], kernel_size=parameters['kernel_size'],
#                           nb_stacks=parameters['nb_stacks'], activation=parameters['act_funct'],
#                           padding=parameters['padding'], use_skip_connections=True,
#                           dropout_rate=parameters['dropout_rate'], return_sequences=True,
#                           kernel_initializer=parameters['conv_kernel_init'])(encoder)
#     else:

#         encoder_inputs = Input(batch_shape=(None, parameters['seq_length'],
#                                             parameters['ts_dimension']))

#         encoder = tcn.TCN(nb_filters=parameters['nb_filters'], kernel_size=parameters['kernel_size'],
#                           nb_stacks=parameters['nb_stacks'], activation=parameters['act_funct'],
#                           padding=parameters['padding'], use_skip_connections=True,
#                           dropout_rate=parameters['dropout_rate'], return_sequences=True,
#                           kernel_initializer=parameters['conv_kernel_init'])(encoder_inputs)

#     encoder = Conv1D(filters=parameters['filters_conv1d'],
#                      kernel_size=1, activation=parameters['activation_conv1d'], padding=parameters['padding'])(encoder)

#     encoder = AveragePooling1D(pool_size=parameters['latent_sample_rate'], strides=None, padding='valid',
#                                data_format='channels_last')(encoder)

#     encoder = Activation(parameters['act_funct'])(encoder)

#     encoder_ouputs = Flatten()(encoder)

#     layer_nodes = parameters['layer_nodes']

#     for ii in range(parameters['n_layers'] - 1):

#         if layer_nodes < encoder_ouputs.shape[1]:

#             if parameters['kernel_regularizer'] != 'None':
#                 encoder_ouputs = Dense(layer_nodes, kernel_regularizer=parameters['kernel_regularizer'],
#                                        activation=parameters['act_funct'])(encoder_ouputs)
#             else:
#                 encoder_ouputs = Dense(layer_nodes, activation=parameters['act_funct'])(encoder_ouputs)

#             if parameters['dropout_rate'] > 0:
#                 encoder_ouputs = Dropout(parameters['dropout_rate'])(encoder_ouputs)

#         layer_nodes = layer_nodes // 2

#     if layer_nodes < encoder_ouputs.shape[1]:
#         encoder_ouputs = Dense(layer_nodes, activation=parameters['act_funct'])(encoder_ouputs)

#     Encoder_model = Model(inputs=[encoder_inputs], outputs=[encoder_ouputs])

#     if parameters['model_summary']:
#         print('\n')
#         Encoder_model.summary()

#     return Encoder_model

# ######################################## Bottleneck  ########################################
# def Bottleneck(parameters):
#     layer_nodes = parameters['layer_nodes'] // (2 ** (parameters['n_layers'] - 1))

#     bottleneck_inputs_X = Input(batch_shape=(None, layer_nodes), name='Input_Bot_X')

#     bottleneck_inputs_Y = Input(batch_shape=(None, layer_nodes), name='Input_Bot_Y')

#     if parameters['pad_encoder']:

#         bottleneck_inputs_Y2 = Lambda(lambda x: x * 0)(bottleneck_inputs_Y)

#     else:

#         bottleneck_inputs_Y2 = bottleneck_inputs_Y

#     if parameters['concatenate_layers']:

#         bottleneck = Concatenate(axis=1)([bottleneck_inputs_X, bottleneck_inputs_Y])

#         if parameters['kernel_regularizer'] != 'None':
#             bottleneck_outputs = Dense(layer_nodes, kernel_regularizer=parameters['kernel_regularizer'],
#                                    activation=parameters['act_funct'])(bottleneck)
#         else:
#             bottleneck_outputs = Dense(layer_nodes, activation=parameters['act_funct'])(bottleneck)

#         if parameters['dropout_rate'] > 0:
#             bottleneck_outputs = Dropout(parameters['dropout_rate'])(bottleneck_outputs)

#     else:
#         # bottleneck_outputs = Multiply()([bottleneck_inputs_X, bottleneck_inputs_Y2])
#         bottleneck_outputs = Add()([bottleneck_inputs_X, bottleneck_inputs_Y2])

#     Bottleneck_model = Model(inputs=[bottleneck_inputs_X, bottleneck_inputs_Y],
#                              outputs=[bottleneck_outputs], name='Bottleneck')

#     if parameters['model_summary']:
#         print('\n')
#         Bottleneck_model.summary()

#     return Bottleneck_model

# ######################################## DECODER XY ########################################
# def Decoder(parameters):
#     layer_nodes = parameters['layer_nodes'] // (2 ** (parameters['n_layers'] - 1))

#     decoder_inputs = Input(batch_shape=(None, layer_nodes), name='Input_Dec')

#     decoder = decoder_inputs

#     if parameters['seq_length'] // parameters['latent_sample_rate'] != 1:
#         for ii in range(parameters['n_layers'] - 1):

#             layer_nodes = layer_nodes * 2

#             if layer_nodes < parameters['seq_length'] // parameters['latent_sample_rate'] * parameters['dilations'][-2]:

#                 if parameters['kernel_regularizer'] != 'None':
#                     decoder = Dense(layer_nodes, kernel_regularizer=parameters['kernel_regularizer'],
#                                            activation=parameters['act_funct'])(decoder)
#                 else:
#                     decoder = Dense(layer_nodes, activation=parameters['act_funct'])(decoder)

#                 if parameters['dropout_rate'] > 0:
#                     decoder = Dropout(parameters['dropout_rate'])(decoder)

#         if parameters['kernel_regularizer'] != 'None':
#             decoder = Dense(parameters['seq_length'] // parameters['latent_sample_rate'] * parameters['dilations'][-2],
#                             kernel_regularizer=parameters['kernel_regularizer'], activation=parameters['act_funct'])(decoder)
#         else:
#             decoder = Dense(parameters['seq_length'] // parameters['latent_sample_rate'] * parameters['dilations'][-2],
#                             activation=parameters['act_funct'])(decoder)

#         if parameters['dropout_rate'] > 0:
#             decoder = Dropout(parameters['dropout_rate'])(decoder)

#     decoder = Reshape((parameters['seq_length'] // parameters['latent_sample_rate'], parameters['dilations'][-2]))(
#         decoder)

#     decoder = UpSampling1D(size=parameters['latent_sample_rate'])(decoder)

#     decoder = tcn.TCN(nb_filters=parameters['nb_filters'], kernel_size=parameters['kernel_size'],
#                       nb_stacks=parameters['nb_stacks'],
#                       padding=parameters['padding'], use_skip_connections=True,
#                       dropout_rate=parameters['dropout_rate'], return_sequences=True,
#                       kernel_initializer=parameters['conv_kernel_init'], name='TCN-dec')(decoder)

#     decoder_outputs = Dense(parameters['ts_dimension'], activation=parameters['act_funct'])(decoder)

#     Decoder_model = Model(inputs=[decoder_inputs], outputs=[decoder_outputs], name='Decoder')

#     if parameters['model_summary']:
#         print('\n')
#         Decoder_model.summary()

#     return Decoder_model
