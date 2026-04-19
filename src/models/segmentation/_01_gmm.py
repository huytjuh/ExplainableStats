import pandas as pd
import numpy as np 
from typing import Optional

class GaussianMixtureModel():
    """Gaussian Mixture Model for clustering."""
    
    def __init__(self, n_components: int=1, tol: float=1e-4, max_iter: int=100):
        """Initialize hyperparameters for the Gaussian Mixture Model."""
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter

        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame) -> 'GaussianMixtureModel':
        """Fit the Gaussian Mixture Model to the data."""
        X = X.values if isinstance(X, pd.DataFrame) else X
        n_samples, n_features = X.shape

        # Initialize parameters
        self.means_ = np.random.rand(self.n_components, n_features)
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        self.weights_ = np.ones(self.n_components) / self.n_components

        log_likelihood_old = -np.inf
        for _ in range(self.max_iter):
            # E-step: Compute responsibilities
            responsibilities = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                responsibilities[:, k] = self.weights_[k] * self._gaussian(X, self.means_[k], self.covariances_[k])
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)

            # M-step: Update parameters
            Nk = responsibilities.sum(axis=0)
            for k in range(self.n_components):
                self.means_[k] = (responsibilities[:, k][:, np.newaxis] * X).sum(axis=0) / Nk[k]
                diff = X - self.means_[k]
                self.covariances_[k] = (responsibilities[:, k][:, np.newaxis, np.newaxis] * (diff[:, :, np.newaxis] * diff[:, np.newaxis, :])).sum(axis=0) / Nk[k]
                self.weights_[k] = Nk[k] / n_samples

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """Expectation step within EM algorithm: Compute responsibilities."""
        n_samples, n_features = X.shape
        responsibilities = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights_[k] * self._gaussian(X, self.means_[k], self.covariances_[k])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """Maximization step within EM algorithm: Update parameters."""
        n_samples, n_features = X.shape
        Nk = responsibilities.sum(axis=0)
        for k in range(self.n_components):
            self.means_[k] = (responsibilities[:, k][:, np.newaxis] * X).sum(axis=0) / Nk[k]
            diff = X - self.means_[k]
            self.covariances_[k] = (responsibilities[:, k][:, np.newaxis, np.newaxis] * (diff[:, :, np.newaxis] * diff[:, np.newaxis, :])).sum(axis=0) / Nk[k]
            self.weights_[k] = Nk[k] / n_samples