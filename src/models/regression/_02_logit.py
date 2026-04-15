import pandas as pd
import numpy as np 
from typing import Optional, Dict

# from utils import sigmoid

class Logit():
    """Logistic Regression Classifier from scratch."""
    
    def __init__(self, lr: float, max_iter: float, n_iter: int=100, tol: float=1e-4):
        """Initialize hyperparameters for the Logistic Regression."""
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol

        self.weights = None
        self.bias = None
        self.diagnostics = None

    def fit(self, X: pd.DataFrame, y: pd.Series, eps: float=1e-15) -> 'Logit':
        """Fit the Logistic Regression model to the training data."""
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.Series) else y
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        list_loss = [np.inf]
        for _ in range(self.n_iter):
            log_odds = X @ self.weights + self.bias
            y_pred = self._sigmoid(log_odds)
            resid = y_pred - y

            # GRADIENT DESCENT
            dw = (1/n_samples) * X.T @ resid
            db = (1/n_samples) * np.sum(resid)
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

            # CROSS-ENTROPY LOSS (= MLE)
            y_pred_c = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y * np.log(y_pred_c) + (1 - y) * np.log(1 - y_pred_c))
            list_loss.append(loss)

            if abs(list_loss[-2] - list_loss[-1]) < self.tol:
                break

        y_pred = self.sigmoid(X @ self.weights + self.bias)
        self.diagnostics = self.calc_diagnostics(X, y, y_pred)
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        
        
        pass

    def calc_diagnostics(self, X: pd.DataFrame, y: pd.Series, y_pred: np.ndarray, ) -> Dict[str, float]:
        LogL = self.log_likelihood()
        pass

    def coefficients(self) -> Optional[np.ndarray]:
        """Return the coefficients of the fitted model."""
        return self.weights

    @property
    def aic(self) -> Optional[float]:
        """Calculate AIC for the fitted model."""
        return 2 * len(self.weights) - 2 * self.diagnostics['log_likelihood']

    @property
    def bic(self) -> Optional[float]:
        """Calculate BIC for the fitted model."""
        return len(self.weights) * np.log(len(self.diagnostics['y'])) - 2 * self.diagnostics['log_likelihood']

    @property
    def log_likelihood(self) -> Optional[float]:
        """Calculate log-likelihood for the fitted model."""
        return self.diagnostics['log_likelihood']
    
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid function to convert log-odds to probabilities."""
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _log_likelihood(y: np.ndarray, p: np.ndarray, eps=1e-15) -> float:
        """Calculate log-likelihood for the given data and model parameters."""
        eps = 1e-15
        y_pred_c = np.clip(y_pred, eps, 1 - eps)
        return np.sum(y * np.log(y_pred_c) + (1 - y) * np.log(1 - y_pred_c))