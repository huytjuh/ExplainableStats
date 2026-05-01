import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple 

from statsmodels.api import sm

from models.panel._01_ols_pooled import PooledOLS
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

        # INITIALIE M-STEP
        X = sm.add_constant(X)
        self._initialize(X, y, entity_col)

        # EM ALGORITHM
        # list_beta, list_sigma2, list_sigma2_alpha = [self.beta], [self.sigma2], [self.sigma2_alpha]
        list_loss = [-np.inf]
        for _ in range(self.max_iter):

            m_i, s_i = self._e_step(X, y, entity_col)
            self._m_step(X, y, entity_col, m_i, s_i)

            loss = self._log_likelihood(X, y)
            list_loss.append(loss)

            if abs(list_loss[-2] - list_loss[-1]) < self.tol:
                break

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

    def _e_step(self, X: np.ndarray, y: np.ndarray, entity_col: np.ndarray, eps: float=1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """E-step of the EM algorithm to estimate the random effects."""
        list_entities = np.unique(entity_col)
        n_entities = len(list_entities)
        entity_idx = np.searchsorted(list_entities, entity_col)

        resid = y - X @ self.beta
        N_i = np.bincount(entity_idx, minlength=n_entities)
        e_i = np.bincount(entity_idx, weights=resid, minlength=n_entities)
        denom = self.sigma2 + N_i * self.sigma2_alpha + eps

        m_i = self.sigma2_alpha * e_i / denom
        s_i = self.sigma2_alpha * self.sigma2 / denom

        return m_i, s_i

    def _m_step(self, X: np.ndarray, y: np.ndarray, entity_col: np.ndarray, m_i: np.ndarray, s_i: np.ndarray, eps: float=1e-6) -> None:
        """M-step of the EM algorithm to update the fixed effects and variance components."""
        list_entity = np.unique(entity_col)
        n_entities = len(list_entity)
        entity_idx = np.searchsorted(list_entity, entity_col)

        # UPDATE BETA
        y_tilde = y - m_i[entity_idx]
        OLS = PooledOLS()
        OLS_fit = OLS.fit(X, y_tilde, constant=False)
        self.beta = OLS_fit.beta

        # UPDATE SIGMA
        resid = y - X @ self.beta
        resid_tilde = resid - m_i[entity_idx]
        self.sigma2 = max(np.mean(resid_tilde**2 + s_i[entity_idx]), eps)
        self.sigma2_alpha = max(np.mean(m_i**2 + s_i), eps)

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
