"""Loads the different kinds of datasets for this chapter"""
from logzero import logger
import numpy as np
from sklearn.datasets import load_iris

def load_iris_dataset():
    dataset = load_iris()
    X, y = dataset["data"], dataset["target"]

    return np.array(X), np.array(y)

# Used for testing the different dataloaders
# if __name__ == "__main__":
#     IRIS
#     load_iris_dataset()