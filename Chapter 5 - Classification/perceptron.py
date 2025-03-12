"""Implementation of a perceptron for binary classification on linearly seperable data"""
from pathlib import Path
from logzero import logger

import numpy as np
from activations import sigmoid
from dataloaders import load_iris_dataset
from losses import missclassification_count

class Perceptron:
    def __init__(self, input_shape: list|tuple|None = None, lr: float = 1e-3, loss: str = "missclassification_count", *args, **kwargs):
        if input_shape:
            self.weights = np.random.randn(*[1, *input_shape][::-1]) + 1e-5
        self.bais = np.random.randn(1) + 1e-5
        self.learning_rate = lr
        self.loss_type = loss
    
    def forward(self, x):
        """Represent a forward pass of the model through the given data

        Arguments:
            x -- input

        Returns:
            prediction
        """
        if not isinstance(self.weights, np.ndarray):
            self.weights = np.random.randn(*x.shape) + 1e-5
        
        if x.shape[-1] != self.weights.shape[0]:
            logger.error(f"Mismatch between the input shape: {x.shape} and initialise weights: {self.weights.shape}")
            return
        
        x = np.matmul(x, self.weights) + self.bais
        return sigmoid(x)
    
    def backward(self, epoch, weight_update: np.ndarray, bais_update: np.ndarray):
        """Performs parameter updation using the gradient"""
        self.learning_rate = self.learning_rate * (1 / (epoch+1))
        self.weights += self.learning_rate * weight_update[:, np.newaxis]
        self.bais += self.learning_rate * bais_update



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

def train(dataset_path: Path|str, epochs: int, *args, **kwargs):
    """Train a perceptron for binary classification

    Arguments:
        dataset_path -- _description_
        epochs -- _description_
    """
    train_dataset = load_binary_dataset(dataset_path)
    loss = missclassification_count
    model = Perceptron(input_shape=(4,), loss=loss.__name__)

    # logger.debug(np.array(train_dataset))
    y_true = train_dataset[:, -1]
    for epoch in range(1, epochs+1):
        y_pred = model.forward(train_dataset[:, :-1])
        train_loss = loss(y_true, np.squeeze(y_pred))

        model.backward(epoch, np.sum(train_dataset[:, :-1]*y_true[:, np.newaxis], axis=0), np.sum(y_true))
        logger.info(f"Epochs {epoch}: Training loss: {train_loss}")
    
    return model

def infer(dataset_path: Path|str, model: Perceptron):
    # Taking a subset of Iris dataset for testing
    test_dataset = load_binary_dataset(dataset_path)
    num_rows_to_choose = int(0.3 * len(test_dataset))
    indices = np.random.choice(test_dataset, size=num_rows_to_choose, replace=False)
    test_dataset = test_dataset[indices]
    
    loss = missclassification_count

    y_pred = model.forward(test_dataset[:, :-1])
    test_loss = loss(test_dataset[:, -1], np.squeeze(y_pred))

    logger.info( f"Validation loss: {test_loss}" )
    

if __name__ == "__main__":
    classifier = train("iris", epochs=10)

    infer("iris", classifier)
