import os
import sys
import numpy as np
import glob
import pickle
import time
import pandas as pd

from scipy.integrate import ode
from scipy.integrate import odeint
from scipy.integrate import RK45

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from pandas import DataFrame, concat
from sklearn.model_selection import train_test_split

from tcn import tcn

from keras import backend as K
from tensorflow.keras.layers import Dense, Input, Conv1D, Reshape, Flatten, Lambda, Add, Activation
from tensorflow.keras.layers import UpSampling1D, AveragePooling1D, Multiply, MaxPooling1D, Concatenate
from tensorflow.keras.models import Model, Sequential

from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

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

def create_datasets(dataset, scaling_method, seq_length, shift, split_tr=None, split_vl=None):

    # only first coordinate is observable
    data = dataset.copy()
    if split_tr == None:
        split_train = int(len(data) * 0.65)
        split_valid = int(len(data) * 0.85)
    elif split_tr < 1:
        split_train = int(len(data) * split_tr)
        split_valid = int(len(data) * split_vl)
    else:
        split_train = int(split_tr)
        split_valid = int(split_vl)

    train_X, valid_X, test_X = data[:split_train], data[split_train:split_valid], data[split_valid:]
    # print('Shape of Training, Validation, and Testing before Series Lagged')
    # print(train_X.shape, valid_X.shape, test_X.shape)

    # Create Lagged Series of Training
    if scaling_method == 'MinMax':
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_X = scaler.fit_transform(train_X.reshape(-1, 1))
        del scaler
    elif scaling_method == 'Standard':
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X.reshape(-1, 1))
        del scaler
    elif scaling_method == 'None':
        print('No Scaling')
        train_X = train_X.reshape(-1, 1)

        # print('Shape of Training, Validation, and Testing after Series Lagged')
    raw_df = create_time_lagged_series(train_X.reshape(-1, 1), seq_length, 1, dropnan=True)
    raw_df = np.array(raw_df)
    trainX = raw_df[:, :-1]
    trainX = np.expand_dims(trainX, axis=-1)
    trainX = trainX[::shift]
    # print('Training Shape: ', trainX.shape)

    # Create Lagged Series of Validation
    if scaling_method == 'MinMax':
        scaler = MinMaxScaler(feature_range=(0, 1))
        valid_X = scaler.fit_transform(valid_X.reshape(-1, 1))
        del scaler
    elif scaling_method == 'Standard':
        scaler = StandardScaler()
        valid_X = scaler.fit_transform(valid_X.reshape(-1, 1))
        del scaler
    elif scaling_method == 'None':
        valid_X = valid_X.reshape(-1, 1)

    raw_df = create_time_lagged_series(valid_X.reshape(-1, 1), seq_length, 1, dropnan=True)
    raw_df = np.array(raw_df)
    valX = raw_df[:, :-1]
    valX = np.expand_dims(valX, axis=-1)
    valX = valX[::shift]
    # print('Validation Shape: ', valX.shape)

    # Create Lagged Series of Testing
    if scaling_method == 'MinMax':
        scaler = MinMaxScaler(feature_range=(0, 1))
        test_X = scaler.fit_transform(test_X.reshape(-1, 1))
        del scaler
    elif scaling_method == 'Standard':
        scaler = StandardScaler()
        test_X = scaler.fit_transform(test_X.reshape(-1, 1))
        del scaler
    elif scaling_method == 'None':
        test_X = test_X.reshape(-1, 1)

    raw_df = create_time_lagged_series(test_X.reshape(-1, 1), seq_length, 1, dropnan=True)
    raw_df = np.array(raw_df)
    testX = raw_df[:, :-1]
    testX = np.expand_dims(testX, axis=-1)
    testX = testX[::shift]
    # print('Testing Shape: ', testX.shape)

    return trainX, valX, testX

def generate_rossler_lorenz(method, length_vars, init_cond, discard, coupling):
    sigma, beta, rho, alpha_freq, C = 10, 8 / 3, 28, 6, coupling

    if init_cond is not None:
        X0 = init_cond
    else:
        X0 = np.random.rand(1, 6).flatten()

    if method == 'odeint':

        def rossler_lorenz_ode(X, t):
            x1, x2, x3, y1, y2, y3 = X

            dx1 = -alpha_freq * (x2 + x3)
            dx2 = alpha_freq * (x1 + 0.2 * x2)
            dx3 = alpha_freq * (0.2 + x3 * (x1 - 5.7))
            dy1 = sigma * (-y1 + y2)
            dy2 = rho * y1 - y2 - y1 * y3 + C * x2 ** 2
            dy3 = -(beta) * y3 + y1 * y2
            return [dx1, dx2, dx3, dy1, dy2, dy3]

        # Integrate the Rossler equations on the time grid t
        time = np.arange(0, length_vars / 100, 0.01)
        result = odeint(rossler_lorenz_ode, X0, time)

        variables = result[discard:, :]

        return variables

    elif method == 'rk45':

        def rossler_lorenz_ode(t, X):
            x1, x2, x3, y1, y2, y3 = X

            dx1 = -alpha_freq * (x2 + x3)
            dx2 = alpha_freq * (x1 + 0.2 * x2)
            dx3 = alpha_freq * (0.2 + x3 * (x1 - 5.7))
            dy1 = sigma * (-y1 + y2)
            dy2 = rho * y1 - y2 - y1 * y3 + C * x2 ** 2
            dy3 = -(beta) * y3 + y1 * y2
            return [dx1, dx2, dx3, dy1, dy2, dy3]

        # Integrate the Rossler equations on the time grid t
        integrator = RK45(rossler_lorenz_ode, 0, X0, length_vars, first_step=0.01)

        results = []
        for ii in range(length_vars):
            integrator.step()
            results.append(integrator.y)

        discard = 1000
        variables = np.array(results)[discard:, :].T

        return variables

######################################## ENCODERS X ########################################
def Encoder(input_dims, tcn_layer_nodes, tcn_dilations, n_layers, layer_nodes, activation_funcs):
    combined_input = Input(batch_shape=(None, 2, input_dims, 1))

    ######################################## ENCODER X ########################################

    encoder_inputs_X = Lambda(lambda x: x[:, 0], name='Input_Enc_X')(combined_input)

    encoder_X = tcn.TCN(nb_filters=tcn_layer_nodes, kernel_size=tcn_layer_nodes, nb_stacks=1, 
                activation=activation_funcs, padding='causal', use_skip_connections=True, dropout_rate=0.0,
                return_sequences=True, kernel_initializer='glorot_normal', name='TCN_Enc_X')(encoder_inputs_X)

    encoder_X = Conv1D(filters=tcn_dilations[-1], kernel_size=1,
                       activation=activation_funcs, padding='causal')(encoder_X)

    encoder_X = AveragePooling1D(pool_size=2, strides=None, padding='valid',
                                 data_format='channels_last')(encoder_X)

    encoder_X = Activation(activation_funcs)(encoder_X)
    
    encoder_ouputs_X = Flatten()(encoder_X)

    layer_nodes_X = layer_nodes

    for ii in range(n_layers - 1):
        encoder_ouputs_X = Dense(layer_nodes_X, activation=activation_funcs)(encoder_ouputs_X)

        layer_nodes_X = layer_nodes_X // 2

    encoder_ouputs_X = Dense(layer_nodes_X, activation=activation_funcs, name='Output_Enc_X')(encoder_ouputs_X)

    ######################################## ENCODERS Y ########################################
    encoder_inputs_Y = Lambda(lambda x: x[:, 1], name='Input_Enc_Y')(combined_input)

    encoder_Y = tcn.TCN(nb_filters=tcn_layer_nodes, kernel_size=tcn_layer_nodes, nb_stacks=1, 
                activation=activation_funcs, padding='causal', use_skip_connections=True, dropout_rate=0.0,
                return_sequences=True, kernel_initializer='glorot_normal', name='TCN_Enc_Y')(encoder_inputs_Y)

    encoder_Y = Conv1D(filters=tcn_dilations[-1], kernel_size=1,
                       activation=activation_funcs, padding='causal')(encoder_Y)

    encoder_Y = AveragePooling1D(pool_size=2, strides=None, padding='valid',
                                 data_format='channels_last')(encoder_Y)

    encoder_Y = Activation(activation_funcs)(encoder_Y)
    
    encoder_ouputs_Y = Flatten()(encoder_Y)

    layer_nodes_Y = layer_nodes

    for ii in range(n_layers - 1):
        encoder_ouputs_Y = Dense(layer_nodes_Y, activation=activation_funcs)(encoder_ouputs_Y)

        layer_nodes_Y = layer_nodes_Y // 2

    encoder_ouputs_Y = Dense(layer_nodes_Y, activation=activation_funcs, name='Output_Enc_Y')(encoder_ouputs_Y)

    ######################################## COMBINED OUTPUTS ########################################
    Encoder_model = Model(inputs=[combined_input], outputs=[encoder_ouputs_X, encoder_ouputs_Y])

    # Encoder_model.summary()

    return Encoder_model

######################################## Bottleneck  ########################################
def Bottleneck(n_layers, layer_nodes, activation_funcs):
    concatenate_layers = False

    layer_nodes = layer_nodes // (2 ** (n_layers - 1))

    bottleneck_inputs_X = Input(batch_shape=(None, layer_nodes), name='Input_Bot_X')

    bottleneck_inputs_Y = Input(batch_shape=(None, layer_nodes), name='Input_Bot_Y')

    if concatenate_layers:

        bottleneck = Concatenate(axis=1)([bottleneck_inputs_X, bottleneck_inputs_Y])

        bottleneck_outputs = Dense(layer_nodes, activation=activation_funcs)(bottleneck)

    else:
        bottleneck_outputs = Add()([bottleneck_inputs_X, bottleneck_inputs_Y])

    Bottleneck_model = Model(inputs=[bottleneck_inputs_X, bottleneck_inputs_Y],
                             outputs=[bottleneck_outputs], name='Bottleneck')

    # Bottleneck_model.summary()

    return Bottleneck_model

######################################## DECODER XY ########################################
def Decoder(input_dims, tcn_layer_nodes, tcn_dilations, n_layers, layer_nodes, activation_funcs):
    layer_nodes = layer_nodes // (2 ** (n_layers - 1))

    decoder_inputs = Input(batch_shape=(None, layer_nodes), name='Input_Dec')

    layer_nodes = layer_nodes * 2

    decoder = Dense(layer_nodes, activation=activation_funcs)(decoder_inputs)

    for ii in range(1, n_layers - 1):
        layer_nodes = layer_nodes * 2

        decoder = Dense(layer_nodes, activation=activation_funcs)(decoder)

    decoder = Dense(input_dims * tcn_layer_nodes // 4, activation=activation_funcs)(decoder)

    decoder = Reshape((input_dims // 2, tcn_layer_nodes // 2))(decoder)

    decoder = UpSampling1D(size=2)(decoder)

    decoder = tcn.TCN(nb_filters=tcn_layer_nodes, kernel_size=tcn_layer_nodes, nb_stacks=1, 
              activation=activation_funcs, padding='causal', use_skip_connections=True, dropout_rate=0.0,
              return_sequences=True, kernel_initializer='glorot_normal')(decoder)
    
    decoder_outputs = Dense(1, activation=activation_funcs, name='Output_Dec')(decoder)

    ######################################## COMBINED OUTPUTS ########################################
    combined_output = Lambda(lambda x: K.squeeze(x, axis=-1))(decoder_outputs)

    Decoder_model = Model(inputs=[decoder_inputs], outputs=[combined_output], name='Decoder')

    # Decoder_model.summary()

    return Decoder_model

def createmodel(input_dim, n_layer, layer_node, tcn_layer_node, activation_func):

    tcn_dilation = []
    ii = 0
    while 2 ** ii != tcn_layer_node:
        tcn_dilation.append(2 ** ii)
        ii += 1

    encoder = Encoder(input_dims=input_dim, tcn_layer_nodes=tcn_layer_node,
                      tcn_dilations=tcn_dilation, n_layers=n_layer, layer_nodes=layer_node,
                      activation_funcs=activation_func)
    bottleneck = Bottleneck(n_layers=n_layer, layer_nodes=layer_node,
                            activation_funcs=activation_func)
    decoder = Decoder(input_dims=input_dim, tcn_layer_nodes=tcn_layer_node,
                      tcn_dilations=tcn_dilation, n_layers=n_layer, layer_nodes=layer_node,
                      activation_funcs=activation_func)

    combined_input = Input(batch_shape=(None, 2, input_dim, 1))

    # Compile Autoencoder
    auto_encoder_model_XYX = Model(inputs=combined_input,
                                   outputs=decoder(bottleneck([encoder(combined_input)])), name='AutoEncoder_XYX')
    auto_encoder_model_XYX.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

    return auto_encoder_model_XYX


if __name__ == "__main__":

    start = time.time()

    directory_results = '/users/jcalde2/Causation/Scripts/'
    os.chdir(directory_results)

    method = 'odeint'
    length_vars = 101000
    discard = 1000
    init_cond = [0, 0, 0.4, 0.3, 0.3, 0.3]
    coupling = 0.3
    var1 = 1
    var2 = 4
    seq_length, shift, lag = 10, 10, 10
    variables = generate_rossler_lorenz(method, length_vars, init_cond, discard, coupling)

    # Generate Data
    trainX, valX, testX = create_datasets(dataset=variables[:-lag, var1], scaling_method='Standard', seq_length=seq_length,
                                          shift=shift)
    trainX_, valX_, testX_ = create_datasets(dataset=variables[lag:, var1], scaling_method='Standard',
                                             seq_length=seq_length, shift=shift)
    trainY, valY, testY = create_datasets(dataset=variables[:-lag, var2], scaling_method='Standard', seq_length=seq_length,
                                          shift=shift)
    trainY_, valY_, testY_ = create_datasets(dataset=variables[lag:, var2], scaling_method='Standard',
                                             seq_length=seq_length, shift=shift)

    print(trainX.shape)

    # Test model Structure
    model = createmodel(input_dim=seq_length, n_layer=4, layer_node=128, tcn_layer_node=32, activation_func='elu')
    del model

    # Wrap model scikeras.wrappers
    model = KerasRegressor(model=createmodel, verbose=0, input_dim=seq_length, n_layer=7, layer_node=1024,
            tcn_layer_node=128, activation_func='linear')

    # define the grid search parameter
    param_grid = []

    # activation_funcs = ['linear', 'relu', 'tanh', 'elu', 'selu']
    input_dims = [seq_length]
    activation_funcs = ['elu', 'selu']
    batch_sizes = [512]

    # n_layers = [1, 2, 3, 4]
    # layer_nodes = [64]
    # tcn_layer_nodes = [8]
    # param_grid.append(dict(input_dim=input_dims, n_layer=n_layers, layer_node=layer_nodes,
    #                        tcn_layer_node=tcn_layer_nodes, activation_func=activation_funcs,
    #                        batch_size=batch_sizes, epochs=[100]))

    n_layers = [3, 4]
    layer_nodes = [128]
    tcn_layer_nodes = [32, 64]
    param_grid.append(dict(input_dim=input_dims, n_layer=n_layers, layer_node=layer_nodes,
                           tcn_layer_node=tcn_layer_nodes, activation_func=activation_funcs,
                           batch_size=batch_sizes, epochs=[100]))

    n_layers = [4, 5]
    layer_nodes = [256]
    tcn_layer_nodes = [32, 64]
    param_grid.append(dict(input_dim=input_dims, n_layer=n_layers, layer_node=layer_nodes,
                           tcn_layer_node=tcn_layer_nodes, activation_func=activation_funcs,
                           batch_size=batch_sizes, epochs=[100]))

    # n_layers = [1, 2, 3, 4, 5, 6, 7]
    # layer_nodes = [512]
    # tcn_layer_nodes = [64]
    # param_grid.append(dict(input_dim=input_dims, n_layer=n_layers, layer_node=layer_nodes,
    #                        tcn_layer_node=tcn_layer_nodes, activation_func=activation_funcs,
    #                        batch_size=batch_sizes, epochs=[100]))

    # n_layers = [1, 2, 3, 4, 5, 6, 7, 8]
    # layer_nodes = [1024]
    # tcn_layer_nodes = [128]
    # param_grid.append(dict(input_dim=input_dims, n_layer=n_layers, layer_node=layer_nodes,
    #                        tcn_layer_node=tcn_layer_nodes, activation_func=activation_funcs,
    #                        batch_size=batch_sizes, epochs=[100]))
    #
    # n_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # layer_nodes = [2048]
    # tcn_layer_nodes = [256]
    # param_grid.append(dict(input_dim=input_dims, n_layer=n_layers, layer_node=layer_nodes,
    #                        tcn_layer_node=tcn_layer_nodes, activation_func=activation_funcs,
    #                        batch_size=batch_sizes, epochs=[100]))

    combined_training_input = np.stack([trainX, trainY], axis=1)
    combined_validation_input = np.stack([valX, valY], axis=1)

    X = np.concatenate((combined_training_input, combined_validation_input), axis=0)
    y = np.concatenate((np.squeeze(trainX_), np.squeeze(valX_)), axis=0)

    ps = PredefinedSplit(np.concatenate((np.zeros(len(combined_training_input)) - 1,
                                         np.ones(len(combined_validation_input)))))

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=ps, verbose=3)
    grid_result = grid.fit(X, y)

    # summarize results
    print("\n\n\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    print('\n\n')
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # directory_results = '/users/jcalde2/Causation/Scripts/'
    # filename = directory_results + 'GridOptimization.csv'
    # DataFrame(grid_result.cv_results_)[['mean_test_score', 'std_test_score', 'params']].to_csv(filename)

    end = time.time()
    print('\n\n Elapsed Time : ' + str(end - start))