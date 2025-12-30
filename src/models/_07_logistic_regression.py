import numpy as np
import pandas as pd
from typing import Optional

class LogisticRegression:
    """Logistic Regression classifier from scratch."""

    def __init__(self, learning_rate: float=0.01, n_iter: int=1000, tol: float=1e-4):
        """Initialize hyperparameters for Logistic Regression."""
        self.lr = learning_rate
        self.n_iter = n_iter
        self.tol = tol

        self.w = None 
        self.b = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LogisticRegression':
        """Train the Logistic Regression classifier."""
        self.maximum_likelihood_estimate(X, y, self.w, self.b)
        return self 

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the input data."""
        linear_pred = self._linear_model(X, self.w, self.b)
        y_pred = self._sigmoid(linear_pred)
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        return y_pred

    def maximum_likelihood_estimate(self, X: pd.DataFrame, y: pd.Series, w: np.ndarray, b: float) -> float:
        """Calculate the maximum likelihood estimate for Logistic Regression."""
        self.w = np.zeros(X.shape[1])
        self.b = 0.0

        def negative_log_likelihood(y: pd.Series, y_pred: np.ndarray) -> float:
            """Calculate the negative log-likelihood (= binary cross-entropy) for Logistic Regression."""
            return -1*np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        def gradient_descent(X: pd.DataFrame, y: pd.Series, w: np.ndarray, b: float) -> tuple[np.ndarray, float]:
            """Calculate the gradient for Logistic Regression."""
            linear_pred = self._linear_model(X, w, b)
            y_pred = self._sigmoid(linear_pred)

            grad_w = (1 / len(X)) * np.dot(X.T, (y_pred - y))
            grad_b = (1 / len(X)) * np.sum(y_pred - y)
            self.w = self.w - self.lr * grad_w
            self.b = self.b - self.lr * grad_b
            return self.w, self.b

        list_logL = [np.inf]
        for n in range(self.n_iter):
            self.w, self.b = gradient_descent(X, y, self.w, self.b)
            linear_pred = self._linear_model(X, self.w, self.b)
            y_pred = self._sigmoid(linear_pred)
            LogL = negative_log_likelihood(linear_pred, y_pred)
            list_logL.append(LogL)
            if np.abs(list_logL[-1] - list_logL[-2]) < self.tol:
                break

        return self

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply the sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def _linear_model(self, X: pd.DataFrame, w: np.ndarray, b: float) -> np.ndarray:
        """Calculate the linear combination of inputs and weights."""
        return np.dot(X, w) + b
