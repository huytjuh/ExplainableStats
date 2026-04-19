import pandas as pd
import numpy as np 
from typing import Optional, Dict


class Multinomial():
    """Multinomial Logistic Regression Classifier from scratch."""
    
    def __init__(self, lr: float=0.01, n_iter: int=100, tol: float=1e-4):
        """Initialize hyperparameters for the Multinomial Logistic Regression."""
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol

        self.classes_: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None

        self.coef_: Optional[Dict[str, np.ndarray]] = None
        self.diagnostics: Optional[Dict[str, float]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Multinomial':
        """Fit the Multinomial Logistic Regression model to the training data.""" 
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.Series) else y

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        


        return self

    @staticmethod
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Softmax function to convert log-odds to probabilities."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
