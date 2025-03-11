"""Implementation of the various activations used in this chapter"""

import numpy as np

def sigmoid(x:np.ndarray, threshold: float = 0.5):
    """Sigmoid activation function

    Arguments:
        x -- linear output from the model

    Returns:
        output
    """
    sigm = (1 / ( 1 + np.exp(-1*x)) > threshold).astype(np.int16)
    sigm = np.where(sigm == 0, -1, sigm)
    return sigm

