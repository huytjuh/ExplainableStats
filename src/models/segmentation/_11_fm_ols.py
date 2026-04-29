import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

import statsmodels.api as sm
from scipy.special import logsumexp

from ._10_fm import FiniteMixtureRegression

class OLSComponent:
    """Component of Finite Mixture of Ordinary Least Squares."""
    def __init__(self, n_features: int, rng: int=42):
        self.rng = np.random.default_rng(rng)
        self.beta = self.rng.standard_normal(n_features)
        self.sigma2 = 1.0

        self.results = None

    def update(self, X: np.ndarray, y: np.ndarray, resp: np.ndarray) -> None:
        """Update beta using weighted least squares."""
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.Series) else y
        X = sm.add_constant(X)

        model = sm.WLS(y, sm.add_constant(X), weights=resp)
        res = model.fit()
        self.results = res

        self.beta = res.params
        self.sigma2 = np.sum(resp * res.resid**2) / np.sum(resp)

    def log_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """Log likelihood of OLS component."""
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.Series) else y
        X = sm.add_constant(X)

        y_pred = X @ self.beta
        resid = y - y_pred
        logL = -0.5 * len(y) * np.log(2 * np.pi * self.sigma2) - 0.5 * np.sum(resid**2) / self.sigma2
        return logL

class FiniteMixtureOLS(FiniteMixtureRegression):
    """Finite Mixture of Ordinary Least Squares using EM."""

    def _initialize_weights(self, X: pd.DataFrame, Y: pd.Series) -> None:
        """Initialize Finite Mixture OLS Model"""
        self.n_samples, self.n_features = X.shape
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.components_ = [OLSComponent(n_features=self.n_features) for k in range(self.n_components)]

    def _e_step(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Expectation step for Finite Mixture OLS Model"""
        resp = np.zeros((self.n_samples, self.n_components))
        for k in range(self.n_components):
            logL_k = self.components_[k].log_likelihood(X, y)
            resp[:, k] = self.weights_[k] * np.exp(logL_k)
        resp = resp / resp.sum(axis=1, keepdims=True)
        return resp

    def _m_step(self, X: np.ndarray, y: np.ndarray, resp: np.ndarray) -> None:
        """Maximization step for Finite Mixture OLS Model"""
        self.weights_ = resp.mean(axis=0)
        for k in range(self.n_components):
            self.components_[k].update(X, y, resp[:, k])

