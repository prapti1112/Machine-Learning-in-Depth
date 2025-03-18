"""Implementation of a perceptron for binary classification on linearly seperable data"""
from pathlib import Path
from logzero import logger

import numpy as np
from activations import sigmoid
from dataloaders import load_binary_dataset
from losses import BinaryCrossentropyLoss

class LogisticRegression:
    def __init__(self, input_shape: list|tuple|None = None, lr: float = 1e-3, *args, **kwargs):
        if input_shape:
            self.weights = np.random.randn(*[1, *input_shape][::-1]) + 1e-5
        self.learning_rate, self.k, self.tau = lr, 1, 10
        self.lmbda = 1e-9
    
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
        
        x = np.matmul(x, self.weights)
        return sigmoid(x)
    
    def backward(self, epoch, weight_update: np.ndarray):
        """Performs parameter updation using the gradient"""
        self.learning_rate = ( self.tau + epoch ) ** (-1 * self.k)
        self.weights -= self.learning_rate * (np.expand_dims(weight_update, axis=1) + 2 * self.lmbda * self.weights )


def train(dataset_path: Path|str, epochs: int, lmbda:float = 1e-9 ,*args, **kwargs):
    """Train a perceptron for binary classification

    Arguments:
        dataset_path -- _description_
        epochs -- _description_
    """
    train_dataset = load_binary_dataset(dataset_path)
    loss = BinaryCrossentropyLoss
    model = LogisticRegression(input_shape=(4,), loss=loss.__name__)

    y_true = train_dataset[:, -1]
    for epoch in range(1, epochs+1):
        y_pred = model.forward(train_dataset[:, :-1])[:, 0]
        train_loss = loss(y_true, y_pred) + lmbda * (np.transpose(model.weights)@model.weights).flatten()[0]

        model.backward(epoch, np.transpose(train_dataset[:, :-1])@(y_true - y_pred))
        logger.info(f"Epochs {epoch}: Training loss: {train_loss}")
    
    return model

def infer(dataset_path: Path|str, model: LogisticRegression):
    # Taking a subset of Iris dataset for testing
    test_dataset = load_binary_dataset(dataset_path)
    num_rows_to_choose = int(0.3 * len(test_dataset))
    indices = np.random.choice(test_dataset, size=num_rows_to_choose, replace=False)
    test_dataset = test_dataset[indices]
    
    loss = BinaryCrossentropyLoss

    y_pred = model.forward(test_dataset[:, :-1])
    test_loss = loss(test_dataset[:, -1], np.squeeze(y_pred))

    logger.info( f"Validation loss: {test_loss}" )
    

if __name__ == "__main__":
    classifier = train("iris", epochs=10)

    infer("iris", classifier)
