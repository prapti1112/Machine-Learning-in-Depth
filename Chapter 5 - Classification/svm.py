"""Implements Support Vector Machine Model Training & Inference"""

import numpy as np
from logzero import logger
from cvxopt import matrix, solvers
from losses import missclassification_count
from dataloaders import load_binary_dataset

class SVM:
    def __init__(self, kernel, degree, gamma = "auto", coef0 = 0.0) -> None:
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.bias = 0

        self.X = self.y = None
    
    def _kernel(self, x1, x2):
        """SVM kernel to get similarity between two datapoints

        Arguments:
            x1 -- .
            x2 -- .

        Raises:
            Exception: Kernel not supported

        Returns:
            kernel_output
        """
        if self.kernel == "linear":
            return np.dot(x1, np.transpose(x2))
        elif self.kernel == "polynomial":
            return (np.dot(x1, np.transpose(x2)) * self._gamma() + self.coef0)**self.degree
        elif self.kernel == "rbf":
            return np.exp(-1 * self._gamma() * np.linalg.norm( x1[:, np.newaxis] - x2[:, np.newaxis], axis=2 ) ** 2 )
        else:
            raise Exception(f"Kernel type {self.kernel} is not support!!!")
    
    def _gamma(self):
        """Calculates multiplier gamma

        Returns:
            multiplier
        """
        if self.gamma == 'scale':
            return 1 / (self.X.shape[1] * np.var(self.X))
        elif self.gamma == 'auto':
            return 1 / self.X.shape[1]
        else:
            return self.gamma


    def fit(self, X, y):
        self.X, self.y = X, y
        y_ = np.array([ -1 if y_true < 0 else 1  for y_true in y ]).astype(float)
        num_samples = X.shape[0]

        K = self._kernel(X, X)
        P = matrix( np.outer(y_, y_) * K)
        q = matrix( -1 * np.ones(num_samples) )
        G = matrix( -1 * np.eye(num_samples) )
        h = matrix( np.zeros(num_samples) )
        A = matrix(y_, (1, num_samples))
        b = matrix(0.0)

        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(solution["x"]).flatten()

        sv_indices = alphas > 1e+5
        self.alphas, self.support_vectors, self.support_vector_labels = alphas[sv_indices], X[sv_indices], y_[sv_indices]

        K_sv = self._kernel(self.support_vectors, self.support_vectors)
        self.bias = np.mean(self.support_vector_labels - np.dot(self.alphas * self.support_vector_labels, K_sv))

    def predict(self, X):
        preds = self._kernel(X, self.support_vectors)
        return np.sign(np.dot(self.alphas * self.support_vector_labels, preds) + self.bias)


if __name__ == "__main__":
    # Simple example dataset
    # X = np.array([[3, 3], [4, 3], [1, 1], [1, 2]])
    # y = np.array([1, 1, -1, -1])
    
    train_dataset = load_binary_dataset(dataset_path="iris")
    X, y = train_dataset[:, :-1], train_dataset[:, -1]

    loss = missclassification_count
    clf = SVM(kernel='rbf', degree=0, gamma="auto")
    clf.fit(X, y)

    predictions = clf.predict(X)
    logger.debug(f"Predictions: {y}")
    logger.info(f"Prediction Loss: {loss(y, predictions)}")

    