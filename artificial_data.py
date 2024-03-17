import os, sys, time, glob, shutil
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting

from scipy.integrate import ode
from scipy.integrate import odeint
from scipy.integrate import RK45

from scipy import signal

################################################################
##################### Lorenz Attractors  ##################
################################################################

def generate_lorenz(method, length_vars, init_cond, discard, coupling):
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    if init_cond is not None:
        X0 = init_cond
    else:
        X0 = np.random.rand(3).flatten()  # Ensuring it's a flat array of length 3

    if method == 'odeint':
        def lorenz_ode(X, t):
            x, y, z = X
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            return [dx, dy, dz]

        # Integrate the Lorenz equations on the time grid t
        time = np.arange(0, length_vars / 100, 0.01)
        result = odeint(lorenz_ode, X0, time)

        variables = result[discard:, :]  # Discard transient dynamics

    return variables

################################################################
##################### Henon-Henon Attractors  ##################
################################################################

def generate_henon03_henon03(method, length_vars, init_cond, discard, coupling):
    b1, b2 = 0.3, 0.3
    variables = np.empty((length_vars, 4))

    variables[0, :] = init_cond

    for ii in range(length_vars - 1):
        variables[ii + 1, 0] = 1.4 - variables[ii, 0] ** 2 + b1 * variables[ii, 1]
        variables[ii + 1, 1] = variables[ii, 0]

        variables[ii + 1, 2] = 1.4 - (coupling * variables[ii, 0] * variables[ii, 2] \
                                      + (1 - coupling) * variables[ii, 2] ** 2) + b2 * variables[ii, 3]
        variables[ii + 1, 3] = variables[ii, 2]

    variables = variables[discard:]

    return variables


def henon03_henon03_plots():
    method = None
    length_vars = 101000
    discard = 1000
    init_cond = [0.7, 0, 0.7, 0]
    couplings = np.linspace(0.0, 0.8, 8)

    coupling = round(couplings[0], 2)
    paint = False
    variables = generate_henon03_henon03(method, length_vars, init_cond, discard, coupling)

    plt.figure(figsize=(20, 15))
    plt.subplot(3, 3, 1)
    plt.plot(variables[:, 0], variables[:, 1], '.k', markersize=1)
    plt.title('Driver')
    plt.xticks([])
    plt.yticks([])
    
    for ii in range(len(couplings)):
        coupling = round(couplings[ii], 2)
        paint = False
        variables = generate_henon03_henon03(method, length_vars, init_cond, discard, coupling)

        plt.subplot(3, 3, 2 + ii)
        plt.plot(variables[:, 2], variables[:, 3], '.k', markersize=1)
        plt.title('Response with C = ' + str(coupling))
        plt.xticks([])
        plt.yticks([])
        
    plt.show()

    return None


def generate_henon03_henon01(method, length_vars, init_cond, discard, coupling):
    b1, b2 = 0.3, 0.1
    variables = np.empty((length_vars, 4))

    variables[0, :] = init_cond

    for ii in range(length_vars - 1):
        variables[ii + 1, 0] = 1.4 - variables[ii, 0] ** 2 + b1 * variables[ii, 1]
        variables[ii + 1, 1] = variables[ii, 0]

        variables[ii + 1, 2] = 1.4 - (coupling * variables[ii, 0] * variables[ii, 2] \
                                      + (1 - coupling) * variables[ii, 2] ** 2) + b2 * variables[ii, 3]
        variables[ii + 1, 3] = variables[ii, 2]

    variables = variables[discard:]

    return variables


def generate_henon01_henon03(method, length_vars, init_cond, discard, coupling):
    b1, b2 = 0.1, 0.3
    variables = np.empty((length_vars, 4))

    variables[0, :] = init_cond

    for ii in range(length_vars - 1):
        variables[ii + 1, 0] = 1.4 - variables[ii, 0] ** 2 + b1 * variables[ii, 1]
        variables[ii + 1, 1] = variables[ii, 0]

        variables[ii + 1, 2] = 1.4 - (coupling * variables[ii, 0] * variables[ii, 2] \
                                      + (1 - coupling) * variables[ii, 2] ** 2) + b2 * variables[ii, 3]
        variables[ii + 1, 3] = variables[ii, 2]

    variables = variables[discard:]

    return variables


def generate_henon03_henon03_temporal_coupling(couplingx, couplingy, length_vars, discard, init_cond):
    variables = np.empty((length_vars, 4))

    variables[0, :] = init_cond

    for ii in range(length_vars - 1):
        variables[ii + 1, 0] = 1.4 - (couplingx[ii] * variables[ii, 0] * variables[ii, 2] \
                                      + (1 - couplingx[ii]) * variables[ii, 0] ** 2) + 0.3 * variables[ii, 1]
        variables[ii + 1, 1] = variables[ii, 0]

        variables[ii + 1, 2] = 1.4 - (couplingy[ii] * variables[ii, 0] * variables[ii, 2] \
                                      + (1 - couplingy[ii]) * variables[ii, 2] ** 2) + 0.3 * variables[ii, 3]
        variables[ii + 1, 3] = variables[ii, 2]

    return variables[discard:]

################################################################
################## Rossler-Lorenz Attractors  ##################
################################################################

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


def rossler_lorenz_plots():
    method = 'odeint'
    length_vars = 101000
    discard = 1000
    init_cond = [0, 0, 0.4, 0.3, 0.3, 0.3]
    couplings = np.linspace(0.0, 5.0, 8)

    coupling = round(couplings[0], 2)
    variables = generate_rossler_lorenz(method, length_vars, init_cond, discard, coupling)

    plt.figure(figsize=(20, 15))
    plt.subplot(3, 3, 1)
    plt.plot(variables[:, 0], variables[:, 1], '.k', markersize=1)
    plt.title('Driver')
    plt.xticks([])
    plt.yticks([])
    
    for ii in range(len(couplings)):
        coupling = round(couplings[ii], 2)
        variables = generate_rossler_lorenz(method, length_vars, init_cond, discard, coupling)

        plt.subplot(3, 3, 2 + ii)
        plt.plot(variables[:, 3], variables[:, 4], '.k', markersize=1)
        plt.title('Response with C = ' + str(coupling))
        plt.xticks([])
        plt.yticks([])
        
    plt.show()

    return None


################################################################
################## Lotka-Volterra Equations  ###################
################################################################

def generate_lotka_volterra(method, length_vars, init_cond, discard, coupling):
    alpha, beta, delta, gamma = 0.07, 0.07, 0.07, 0.07

    if init_cond is not None:
        X0 = init_cond
    else:
        X0 = np.random.rand(1, 2).flatten()

    def lotka_volterra_ode(X, t):
        x, y = X

        dx = x * (alpha - beta * y)
        dy = y * (-delta + gamma * x)

        return [dx, dy]

    # Integrate the equations on the time grid t
    time = np.arange(0, length_vars / 100, 0.01)
    result = odeint(lotka_volterra_ode, X0, time)

    variables = result[discard:, :]

    return variables


################################################################
################## Logistic Maps Equations  ####################
################################################################

def generate_two_species_model(method, length_vars, init_cond, discard, coupling):
    lag = 1
    X = np.empty((2, length_vars))

    if init_cond is not None:
        X[:, 0] = init_cond
    else:
        X[:, 0] = np.random.rand(1, 2).flatten()

    for ii in range(lag, length_vars):
        # equations
        X[0, ii] = X[0, ii - 1] * (3.8 - 3.8 * X[0, ii - 1] - coupling * X[1, ii - 1])
        X[1, ii] = X[1, ii - 1] * (3.5 - 3.5 * X[1, ii - 1] - (coupling * 5) * X[0, ii - 1])

    variables = np.array(X)[:, discard:].T

    return variables

################################################################
################## Autoregressive Processes  ###################
################################################################
def generate_ar_processes(method, length_vars, init_cond, discard, coupling):

    # Parameters
    T = length_vars  # Total points
    C = coupling
    variance = 0.1

    # Initialize x and y using normal distribution with zero mean and unit variance
    x = np.random.randn(T)
    y = np.random.randn(T)

    # Generate Gaussian noise for nx and ny
    nx = np.random.normal(0, np.sqrt(variance), T)
    ny = np.random.normal(0, np.sqrt(variance), T)

    # Generate the autoregressive processes
    for t in range(1, T):
        x[t] = 0.5 * x[t-1] + 0.1 * y[t-1] + nx[t]
        y[t] = C * x[t-1] + 0.7 * y[t-1] + ny[t]

    variables = np.column_stack([x,y])  
    variables = variables[discard:]
    
    return variables

def ar_processes_plots():
    method = None
    length_vars = 101000
    discard = 1000
    init_cond = None
    couplings = np.linspace(0, 0.6, 9)
    spacing = 1000

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    for ii, ax in enumerate(axes.flatten()):
        coupling = round(couplings[ii], 2)
        variables = generate_ar_processes(method, length_vars, init_cond, discard, coupling)
                
        ax.plot(variables[::spacing, 0], label='Driver')
        ax.plot(variables[::spacing, 1] - 3, label='Response')
        ax.set_title('Driver and Response with C = ' + str(coupling))
        ax.set_xticks([])
        ax.set_yticks([])

    # Create a single legend outside the plots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1))

    plt.tight_layout()
    plt.show()

    return None
        