import numpy as np
import pandas as pd
from typing import Optional

class Perceptron:
    """Perceptron classifier from scratch."""

    def __init__(self, learning_rate: float=0.01, n_iter: int=1000, tol: float=1e-4):
        """Initialize hyperparameters for Perceptron."""
        self.lr = learning_rate
        self.n_iter = n_iter
        self.tol = tol
        
        self.w = None 
        self.b = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Perceptron':
        """Train the Perceptron classifier."""
        self.w = np.zeros(X.shape[1])
        self.b = 0.0

        y = self._activation_func(method='step', y=y)

        for n in range(self.n_iter):
            linear_pred = self._linear_model(X, self.w, self.b)
            y_pred = self._activation_func(linear_pred)

            grad_w = np.dot(X.T, (y - y_pred))
            grad_b = np.sum(y - y_pred)

            self.w = self.w + self.lr * grad_w
            self.b = self.b + self.lr * grad_b

            if np.mean(np.abs(y - y_pred)) < self.tol:
                break

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the input data."""
        linear_pred = self._linear_model(X, self.w, self.b)
        y_pred = self._activation_func(linear_pred)
        return y_pred 

    def _linear_model(self, X: pd.DataFrame, w: np.ndarray, b: float) -> np.ndarray:
        """Calculate the linear model output."""
        return np.dot(X, w) + b

    def _activation_func(self, y: np.ndarray, method='step') -> np.ndarray:
        """Step activation function."""
        if method == 'step':
            return np.where(y > 0, 1, 0) 
        return 

    