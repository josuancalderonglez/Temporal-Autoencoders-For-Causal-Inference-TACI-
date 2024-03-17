# Temporal Autoencoders for Causal Inference (TACI)

## Description

In this project, we introduce an approach named Temporal Autoencoders for Causal Inference (TACI). Our methodology uses a two-headed Temporal Convolutional Network (TCN) autoencoder architecture to encode time series data x(t) and y(t). A third TCN is used for decoding a future trajectory of y(t) (shifted by a time, τ) from a compact latent space derived from the outputs of the first two autoencoders. This approach allows for the training of four distinct versions of the network, tailored to predict the future states of x(t) and y(t), incorporating both original and surrogate data. The ultimate goal is to utilize the variance explained (R² values) over a moving window to compute metrics (χx→y and χy→x) for assessing causal inference between the two variables.

## Installation Instructions

To use TACI, you will need to have Python installed on your system, along with the following packages: numpy, matplotlib, scikit-learn, pandas, scipy, h5py, tensorflow, and keras-tcn. You can install these packages using pip. Here's how you can set up your environment:

```bash
# It's recommended to create a virtual environment
python -m venv taci-env

# Activate the virtual environment
# On Windows
taci-env\Scripts\activate
# On macOS and Linux
source taci-env/bin/activate

# Install the required packages
pip install numpy matplotlib scikit-learn pandas scipy h5py tensorflow keras-tcn
