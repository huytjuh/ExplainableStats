import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

class LatentClass(ABC):
    """Abstract Base class for Latent Class models using EM."""
    def __init__(self, n_components=3, max_iter: int=100, tol: float=1e-6):
        """Initialize LatentClass"""
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LatentClass':
        """Fit Latent Class models using EM."""
        self._initialize_weights(X, y)

        list_loss = [np.inf]
        for _ in range(self.max_iter):
            resp = self._e_step(X, y)
            self._m_step(X, y, resp)
 
            loss = self._log_likelihood(X, y)
            list_loss.append(loss)

            if abs(list_loss[-2] - list_loss[-1]) < self.tol:
                break

        return self

    def predict(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Predict Latent Class model"""
        proba = self.predict_proba(X, y)
        return np.argmax(proba, axis=1)

    @abstractmethod
    def _initialize_weights(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def _e_step(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def _m_step(self, X: pd.DataFrame, y: pd.Series, resp: np.ndarray) -> None:
        pass

    @abstractmethod
    def _log_likelihood(self, X: pd.DataFrame, y: pd.Series) -> float:
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        pass