import numpy as np
import pandas as pd
from typing import Dict

from scipy.special import logsumexp
from scipy.stats import poisson, norm

from ._10_fm import FiniteMixtureRegression

class PoissonRegressionComponent:
    """Component of Finite Mixture Poisson Regression."""
    def __init__(self, n_features: int, rng: int=42):
        self.rng = np.random.default_rng(rng)
        self.beta = self.rng.standard_normal(n_features)

    def _grad_hessian(self, X: pd.DataFrame, y: np.ndarray, resp: np.ndarray) -> tuple:
        """Gradient and Hessian of Poisson log-likelihood."""
        log_rate = X @ self.beta
        lambda_ = np.exp(log_rate)
        resid = y - lambda_

        grad = X.T @ resid
        lambda_weighted = resp * lambda_
        hessian = -(X.T - lambda_weighted) @ X
        return grad, hessian
    
    def _update(self, X: pd.DataFrame, y: np.ndarray, resp: np.ndarray) -> None:
        """Update beta using Newton-Raphson."""
        grad, hessian = self._grad_hessian(X, y, resp)
        self.beta = self.beta + np.linalg.solve(-hessian, grad)


class FiniteMixturePoissonRegression(FiniteMixtureRegression):
    """Finite Mixture Poisson Regression using EM."""

    def _initialize_weights(self, X: pd.DataFrame, Y: pd.Series) -> None:
        """Initialize Finite Mixture Poisson Regression Model"""
        self.n_samples, self.n_features = X.shape
        self.weights_ = np.ones(self.n_components) / self.n_components
        self.components_ = [PoissonRegressionComponent(n_features=self.n_features) for k in range(self.n_components)]

    def _e_step(self, X: pd.DataFrame, Y: pd.Series) -> np.ndarray:
        """Expectation step for Finite Mixture Poisson Regression Model"""
        resp = np.zeros((self.n_samples, self.n_components))
        for k in range(self.n_components):
            beta_k = self.components_[k].beta
            lambda_k = np.exp(X @ beta_k)
            resp[:, k] = self.weights_[k] * poisson.pmf(Y, mu=lambda_k)
        resp = resp / resp.sum(axis=1, keepdims=True)
        return resp
    
    def _m_step(self, X: pd.DataFrame, Y: pd.Series, resp: np.ndarray) -> None:
        """Maximization step for Finite Mixture Poisson Regression Model"""
        self.weights_ = resp.mean(axis=0)
        for k in range(self.n_components):
            self.components_[k]._update(X, Y, resp[:, k])

    def _log_likelihood(self, X: pd.DataFrame, Y: pd.Series) -> float:
        """Log likelihood of Finite Mixture of Poisson Regressions"""
        LogL = np.zeros((self.n_samples, self.n_components))
        for k in range(self.n_components):
            beta_k = self.components_[k].beta
            lambda_k = np.exp(X @ beta_k)
            LogL[:, k] = np.log(self.weights_[k]) + poisson.logpmf(Y, mu=lambda_k)
        return logsumexp(LogL, axis=1).sum()

    def predict_proba(self, X: pd.DataFrame, Y: pd.Series) -> np.ndarray:
        """Predict Finite Mixture Poisson Regression model"""
        resp = self._e_step(X, Y)
        return resp
    
    def summary(self, X: pd.DataFrame, Y: pd.Series, alpha: float=0.05, eps: float=1e-6) -> Dict[tuple, Dict[str, float]]:
        """Summary of Finite Mixture of Poisson Regressions"""
        X = X.values if isinstance(X, pd.DataFrame) else X
        Y = Y.values if isinstance(Y, pd.Series) else Y

        resp = self.predict_proba(X, Y)

        z_crit = norm.ppf(1 - alpha/2)
        dict_summary = {}
        for k in range(self.n_components):
            beta_k = self.components_[k].beta

            grad_k, H_k = self.components_[k]._grad_hessian(X, Y, resp[:, k])
            ridge = eps * np.eye(H_k.shape[0])
            cov_k = np.linalg.inv(-(H_k + ridge))
            se_k = np.sqrt(np.diag(cov_k))

            z_k = beta_k / se_k
            p_k = 2 * (1 - norm.cdf(np.abs(z_k)))

            ci_k = np.column_stack([beta_k - z_crit * se_k, beta_k + z_crit * se_k])
            for feature in range(self.n_features):
                dict_summary[(k, feature)] = {
                    'beta': beta_k[feature],
                    'se': se_k[feature],
                    'z_score': z_k[feature],
                    'p_value': p_k[feature],
                    'ci_lower': ci_k[feature, 0],
                    'ci_upper': ci_k[feature, 1]
                }
        return dict_summary