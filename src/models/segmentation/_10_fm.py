import numpy as np
import pandas as pd
from typing import Optional, Dict

from abc import ABC, abstractmethod

class FiniteMixtureRegression(ABC):
    """Abstract Base class for Finite Mixture Regression using EM."""
    def __init__(self, n_components=3, max_iter: int=100, tol: float=1e-6):
        """Initialize LatentClass"""
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

        self.summary: Optional[Dict[str, np.ndarray]] = None
        self.diagnostics: Optional[Dict[str, np.ndarray]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FiniteMixtureRegression':
        """Fit Finite Mixture Regression using EM."""
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.Series) else y

        self._initialize_weights(X, y)

        list_loss = [np.inf]
        for _ in range(self.max_iter):
            resp = self._e_step(X, y)
            self._m_step(X, y, resp)
 
            loss = self._log_likelihood(X, y)
            list_loss.append(loss)

            if abs(list_loss[-2] - list_loss[-1]) < self.tol:
                break

        self.summary = self.summary(X, y)

        return self

    def predict(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Predict Finite Mixture Regression model"""
        proba = self.predict_proba(X, y)
        return np.argmax(proba, axis=1)

    @abstractmethod
    def _initialize_weights(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def _e_step(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        
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
        
    @abstractmethod
    def summary(self) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def diagnostics(self) -> Dict[str, np.ndarray]:
        pass