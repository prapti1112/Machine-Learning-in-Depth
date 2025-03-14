"""Loads the different kinds of datasets for this chapter"""
from pathlib import Path
from logzero import logger
import numpy as np
from sklearn.datasets import load_iris

def load_iris_dataset():
    dataset = load_iris()
    X, y = dataset["data"], dataset["target"]

    return np.array(X), np.array(y)


def load_binary_dataset(dataset_path: Path|str):
    """Loads a dataset and only takes the subset of the dataset which can be used for binary classification

    Arguments:
        dataset_path -- _description_

    Returns:
        dataset
    """
    if dataset_path == "iris":
        X, y = load_iris_dataset()
        logger.debug(f"X: {X.shape}, y: {y.shape}")        
        
        y = np.where(y == 0, -1, y)
        dataset = np.hstack((X, np.expand_dims(y, axis=1)))
        dataset = dataset[np.logical_or(dataset[:, -1] == -1, dataset[:, -1] == 1)]
        np.random.shuffle(dataset)
        
        
        return dataset
    else:
        logger.debug(f"Data loading for dataset: {dataset_path} is not supported yet!")



# Used for testing the different dataloaders
# if __name__ == "__main__":
#     IRIS
#     load_iris_dataset()