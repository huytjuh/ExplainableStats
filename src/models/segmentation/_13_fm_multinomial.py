import numpy as np
import pandas as pd
from typing import Dict

from scipy.special import logsumexp
from scipy.stats import multinomial, norm

from ._10_fm import FiniteMixtureRegression

class MultinomialComponent:
    """Component of Finite Mixture of Multinomial Regressions."""
    def __init__(self, n_classes: int, rng: int=42):
        self.rng = np.random.default_rng(rng)
        self.theta = self.rng.dirichlet(alpha=np.ones(n_classes))

class FiniteMixtureMultinomialRegression(FiniteMixtureRegression):
    """Multinomial Finite Mixture of Multinomial Regressions using EM."""

    def _initialize_weights(self, X: pd.DataFrame, Y: pd.Series) -> None:
        """Initialize Multinomial Finite Mixture Model"""
        self.n_samples, self.n_classes = Y.shape
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.theta_ = np.random.dirichlet(alpha=np.ones(self.n_classes), size=self.n_components)

    def _e_step(self, X: pd.DataFrame, Y: pd.Series) -> np.ndarray:
        """Expectation step for Finite Mixture of Multinomial Regressions"""
        resp = np.zeros((self.n_samples, self.n_components))
        for k in range(self.n_components):
            resp[:, k] = self.weights_[k] * multinomial.pmf(Y, n=Y.sum(axis=1), p=self.theta_[k])
        resp = resp / resp.sum(axis=1, keepdims=True)
        return resp

    def _m_step(self, X: pd.DataFrame, Y: pd.Series, resp: np.ndarray) -> None:
        """Maximization step for Finite Mixture of Multinomial Regressions"""
        self.weights_ = resp.mean(axis=0)
        Y_weighted = resp.T @ Y
        self.theta_ = Y_weighted / Y_weighted.sum(axis=1, keepdims=True)

    def _log_likelihood(self, X: pd.DataFrame, Y: pd.Series) -> float:
        """Log likelihood of Finite Mixture of Multinomial Regressions"""
        LogL = np.zeros((self.n_samples, self.n_components))
        for k in range(self.n_components):
            LogL[:, k] = np.log(self.weights_[k]) + multinomial.logpmf(Y, n=Y.sum(axis=1), p=self.theta_[k])
        return logsumexp(LogL, axis=1).sum()
    
    def predict_proba(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Predict Finite Mixture of Multinomial Regressions model"""
        resp = self._e_step(X, y)
        return resp
    
    def summary(self, X: pd.DataFrame, Y: pd.Series, alpha: float=0.05) -> Dict[tuple, Dict[str, float]]:
        """Summary of Finite Mixture of Multinomial Regressions"""
        X = X.values if isinstance(X, pd.DataFrame) else X
        Y = Y.values if isinstance(Y, pd.Series) else Y

        resp = self.predict_proba(X, Y)
        Nk = resp.sum(axis=0)

        z_crit = norm.ppf(1 - alpha/2)
        dict_summary = {}
        for k in range(self.n_components):
            theta_k = self.theta_[k]
            
            Nk_k = Nk[k]
            se_k = np.sqrt(theta_k * (1 - theta_k) / Nk_k)
            
            z_k = theta_k / se_k
            p_k = 2 * (1 - norm.cdf(np.abs(z_k)))

            ci_k = np.column_stack([theta_k - z_crit * se_k, theta_k + z_crit * se_k])
            for c in range(self.n_classes):
                dict_summary[(k, c)] = {
                    'theta': theta_k[c],
                    'se': se_k[c],
                    'z_score': z_k[c],
                    'p_value': p_k[c],
                    'ci_95': ci_k[c]
                }   

        self.summary = dict_summary
        return self.summary
    
    def diagnostics(self, X: pd.DataFrame, Y: pd.Series) -> Dict[str, float]:
        """Diagnostics of Finite Mixture of Multinomial Regressions"""
        X = X.values if isinstance(X, pd.DataFrame) else X
        Y = Y.values if isinstance(Y, pd.Series) else Y

        K = self.n_components
        C = self.n_classes
        n_params = K * (C - 1) + (K - 1)

        logL0 = multinomial.logpmf(Y, n=Y.sum(axis=1), p=Y.mean(axis=0)).sum()
        logL1 = self._log_likelihood(X, Y)

        self.diagnostics = {
            'log_likelihood': logL1,
            'aic': 2 * n_params - 2 * logL1,
            'bic': np.log(self.n_samples) * n_params - 2 * logL1,
            'llr': -2 * (logL1 - logL0)
        }

        return self.diagnostics