import pandas as pd
import numpy as np 
from typing import Optional

from scipy.stats import multivariate_normal

class GaussianMixtureModel():
    """Gaussian Mixture Model for clustering."""
    
    def __init__(self, n_components: int=3, n_iter: int=100, tol: float=1e-4):
        """Initialize hyperparameters for the Gaussian Mixture Model."""
        self.n_components = n_components
        self.tol = tol
        self.n_iter = n_iter

        self.weights_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None
        

    def fit(self, X: pd.DataFrame) -> 'GaussianMixtureModel':
        """Fit the Gaussian Mixture Model to the data."""
        X = X.values if isinstance(X, pd.DataFrame) else X
        n_samples, n_features = X.shape

        self.weights_ = np.ones(self.n_components) / self.n_components
        self.means_ = np.random.rand(self.n_components, n_features)
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        

        list_loss = [np.inf]
        for _ in range(self.n_iter):
            resp = self._e_step(X)
            self._m_step(X, resp)

            loss = self._log_likelihood(X)
            list_loss.append(loss)

            if abs(list_loss[-2] - list_loss[-1]) < self.tol:
                break

        return self

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """Expectation step within EM algorithm: Compute responsibilities."""
        n_samples, n_features = X.shape
        resp = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            resp[:, k] = self.weights_[k] * multivariate_normal(mean=self.means_[k], cov=self.covariances_[k]).pdf(X)
        resp = resp / resp.sum(axis=1, keepdims=True)
        return resp

    def _m_step(self, X: np.ndarray, resp: np.ndarray):
        """Maximization step within EM algorithm: Update parameters."""
        n_samples, n_features = X.shape

        Nk = resp.sum(axis=0)
        self.weights_ = Nk / n_samples 
        self.means_ = (resp.T @ X) / Nk[:, None]
        for k in range(self.n_components):
            X_centered = X - self.means_[k]
            X_centered_weighted = X_centered * resp[:, k][:, None]
            self.covariances_[k] = (X_centered_weighted.T @ X_centered) / Nk[k]

    def _log_likelihood(self, X: np.ndarray) -> float:
        """Calculate the log-likelihood of the data given the current parameters."""
        n_samples, n_features = X.shape
        prob = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            prob[:, k] = self.weights_[k] * multivariate_normal(mean=self.means_[k], cov=self.covariances_[k]).pdf(X)
        LogL = np.sum(np.log(np.sum(prob, axis=1)))
        return LogL