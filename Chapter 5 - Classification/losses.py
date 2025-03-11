"""Implements different loss functions required for classification operations in this chapter"""
import numpy as np

def missclassification_count(y_true: np.ndarray, y_pred: np.ndarray):
    """Returns the number of miss classified data points

    Arguments:
        y_true -- _description_
        y_pred -- _description_

    Returns:
        _description_
    """
    return np.sum(y_true != y_pred)

def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray):
    return max(0, 1 - y_true*y_pred)