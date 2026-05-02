import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

from scipy import stats
import statsmodels.api as sm

class HBhyperparams:
    # GLOBAL ELASTICITY PRIOR
    mu0: float=-1.5
    sigma20: float=1.0

    # VARIANCE PRIOR
    beta_a: float=2.0
    beta_b: float=1.0
    eps_a: float=2.0
    eps_b: float=1.0
    gamma: float=100.0

class HierarchicalBayes:
    """Hierarchical Bayes model for panel data from scratch"""

    def __init__(self, hyperparams: Optional[HBhyperparams]=None, max_iter: int=100, n_burn: int=500, random_state: int=42) -> None:
        """Initialize hyperparameters for the Hierarchical Bayes model."""
        self.hyperparams = hyperparams or HBhyperparams()
        self.max_iter = max_iter
        self.n_burn = n_burn
        self.random_state = random_state

        self.alpha = None
        self.beta = None
        self.gamma = None

        self.mu_beta = None
        self.sigma2_beta = None
        self.sigma2_eps = None

        self.entities = None
        self.entity_idx = None

    def fit(self, X: pd.DataFrame, y: pd.Series, entity_col: pd.Series, time_col: pd.Series) -> 'HierarchicalBayes':
        """Fit the Hierarchical Bayes model to the training data."""
        X = np.asarray(X)
        y = np.asarray(y)
        entity_col = np.asarray(entity_col)
        time_col = np.asarray(time_col)

        self.entities = np.unique(entity_col)
        self.entity_idx = np.searchsorted(self.entities, entity_col)

        X_dm, y_dm, self.alpha = self._within_transform(X, y, entity_col)

        self._gibbs_sampler(X_dm, y_dm)

        return self
    
    def _gibbs_sampler(self, X: np.ndarray, y: np.ndarray, entity_col: np.ndarray, time_col: np.ndarray) -> None:
        """Run the Gibbs sampler for the Hierarchical Bayes model."""
        n_samples, n_features = X.shape
        n_entities = len(self.entities)

        beta = np.full(n_entities, self.hyperparams.mu0)
        mu_beta = self.hyperparams.mu0
        sigma2_beta = self.hyperparams.sigma20
        sigma2_eps = 1.0
        gamma = np.zeros(n_features)

        n_keep = self.max_iter - self.n_burn
        draws_beta = np.zeros((n_keep, n_features))
        draws_mu_beta = np.zeros(n_keep)
        draws_sigma2_beta = np.zeros(n_keep)
        draws_sigma2_eps = np.zeros(n_keep)
        draws_gamma = np.zeros((n_features, n_entities))

        for _ in range(self.max_iter):
            for i in range(n_entities):
                y_i = y[self.entity_idx == i]
                X_i = X[self.entity_idx == i]

                beta[i] = stats.norm.rvs(loc=mu_beta, scale=np.sqrt(sigma2_beta))


    def predict(self, X: pd.DataFrame, entity_col: pd.Series, time_col: pd.Series) -> np.ndarray:
        """Predict using the fitted Hierarchical Bayes model."""
        return np.zeros(len(X))

    @staticmethod
    def _within_transform(X: np.ndarray, y: np.ndarray, entity_col: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Within transformation to remove time-invariant unobserved heterogeneity."""
        n_samples, n_features = X.shape
        list_entities = np.unique(entity_col)
        n_entities = len(list_entities)
        entity_idx = np.searchsorted(list_entities, entity_col)

        N_i = np.bincount(entity_idx, minlength=n_entities)
        X_bar = np.column_stack([np.bincount(entity_idx, weights=X[:, col], minlength=n_entities) for col in range(n_features)]) / N_i[:, None]
        y_bar = np.bincount(entity_idx, weights=y, minlength=n_entities) / N_i

        X_dm = X - X_bar[entity_idx]
        y_dm = y - y_bar[entity_idx]
        alpha = {entity: y_bar[i] for i, entity in enumerate(list_entities)}

        return X_dm, y_dm, alpha
