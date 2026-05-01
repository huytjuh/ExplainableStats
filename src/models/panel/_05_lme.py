import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple 

from statsmodels.api import sm

from models.panel._02_re import RandomEffects
from models.panel._03_fe import FixedEffects

class LinearMixedEffects():
    """Linear Mixed Effects for panel data from scratch"""

    def __init__(self, max_iter: int=100, tol: float=1e-6) -> None:
        """Initialize hyperparameters for the Linear Mixed Effects."""
        self.max_iter = max_iter
        self.tol = tol

        self.beta = None 
        self.beta_fe = None
        self.beta_re = None 

        self.sigma2 = None 
        self.sigma2_alpha = None

        self.coef_table: Optional[Dict[str, np.ndarray]] = None
        self.diagnostics: Optional[Dict[str, float]] = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, entity_col: pd.Series) -> 'LinearMixedEffects':
        """Fit the Linear Mixed Effects model to the training data."""
        X = np.asarray(X)
        y = np.asarray(y)
        entity_col = np.asarray(entity_col)
        n_samples, n_features = X.shape
        list_entities = np.unique(entity_col)
        n_entities = len(list_entities)
        entity_idx = np.searchsorted(list_entities, entity_col)

        X = sm.add_constant(X)
        self._initialize(X, y, entity_col)

        for _ in range(self.max_iter):
            break

        self._e_step()
        self._m_step()

        # INFERENCE & DIAGNOSTICS
        self._inference(X)
        self._diagnostics(X, y)

        return self

    def _initialize(self, X: np.ndarray, y: np.ndarray, entity_col: np.ndarray) -> None:
        """Initialize hyperparameters for the Linear Mixed Effects."""
        RE = RandomEffects()
        RE_fit = RE.fit(X, y, entity_col)
        self.beta_re = RE_fit.beta
        self.sigma2 = RE_fit.sigma2
        self.sigma2_alpha = RE_fit.sigma2_alpha

        FE = FixedEffects()
        FE_fit = FE.fit(X, y, entity_col)
        self.beta_fe = FE_fit.beta

        alpha0 = y.mean() - X.mean(axis=0) @ self.beta_re
        self.beta = np.concatenate([alpha0, self.beta_re])

    def _e_step(self):
        """E-step of the EM algorithm to estimate the random effects."""
        pass

    def _m_step(self):
        """M-step of the EM algorithm to update the fixed effects and variance components."""
        pass

    def _log_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate the log-likelihood of the fitted model."""
        return 0.0

    def predict(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Predict using the fitted Linear Mixed Effects model."""
        return 

    def _inference(self, X: np.ndarray) -> None:
        """Calculate inference for the fitted model."""
        pass
    
    def _diagnostics(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calculate diagnostics for the fitted model."""
        pass
