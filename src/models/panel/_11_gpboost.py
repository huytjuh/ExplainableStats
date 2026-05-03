import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

from scipy import stats
import statsmodels.api as sm

from sklearn.ensemble import GradientBoostingRegressor

class GaussianProcessBoost:
    """Gaussian Process Boost for panel data from scratch.
    
    Model:
        y_it = f(x_it) + Z_it @ u_{g(i)} + eps_it

    where:
        - f(x) is a GradientBoostingRegressor
        - Z_it = [1, re_it] (intercept + one random-slope covariate)
        - u_g ~ N(0, sigma2_u I)
    """

    def __init__(self, n_estimators: int=100, learning_rate: float=0.1, max_depth: int=3, min_samples_leaf: int=2, n_outer_iter: int=10, random_state: int=42) -> None:
        """Initialize hyperparameters for the Gaussian Process Boost."""
        self.n_estimator = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_outer_iter = n_outer_iter
        self.random_state = random_state

        self.sigma2_u = 0.5
        self.sigma2_e = 1.0

        self.group = None
        self.group_idx = None

        self.gb: Optional[GradientBoostingRegressor] = None
        self.global_bias = None

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series, re: Optional[pd.Series]) -> 'GaussianProcessBoost':
        """Fit the Gaussian Process Boost model to the training data. """
        X = np.asarray(X)
        y = np.asarray(y)
        groups = np.asarray(groups)
        re = np.asarray(re) if re is not None else np.ones(len(X))
        n_samples, n_features = X.shape

        self.group = np.unique(groups)
        self.group_idx = np.searchsorted(self.group, groups)

        Z = sm.add_constant(re)

        self.gb = GradientBoostingRegressor(n_estimators=self.n_estimator, learning_rate=self.learning_rate, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=self.random_state)

        self.global_bias = y.mean()
        f = np.full(n_samples, self.global_bias)
        u = np.zeros(n_samples)

        for _ in range(self.n_outer_iter):
            resid = y - (f + u)

            u = self._update_random_effects(resid, Z)

            resid_gb = y - u

            GB_fit = self.gb.fit(X, resid_gb)
            f = GB_fit.predict(X)

        return self
    
    def _update_random_effects(self, resid: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Update the random effects."""
        n_groups = len(self.group)
        n_features_re = Z.shape[1]

        I = np.eye(n_features_re)
        u_group = np.zeros((n_groups, n_features_re))
        for i in range(n_groups):
            Z_i = Z[self.group_idx == i]
            resid_i = resid[self.group_idx == i]

            cov_post = Z_i.T @ Z_i / self.sigma2_e + I / self.sigma2_u
            cov_post = np.linalg.inv(cov_post)
            mean_post = cov_post @ (Z_i.T @ resid_i / self.sigma2_e)
            u_group[i] = mean_post

        return Z @ u_group
        

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the fitted Gaussian Process Boost model."""
        pass

    