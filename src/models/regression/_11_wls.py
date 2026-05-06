import pandas as pd
import numpy as np
from typing import Optional, Dict

import statsmodels.api as sm
from scipy.stats import norm


class WLS:
    """
    Weighted Least Squares (WLS) Regression from scratch.

    Motivation:
      Ordinary Least Squares (OLS) assumes homoskedasticity:
        Var(error_i | X_i) = constant for all i.
      When this is violated (heteroskedasticity), the variability of the errors
      changes with X. OLS coefficients remain unbiased, but they are no longer
      efficient and, more importantly, the usual standard errors, t-tests, and
      F-tests become unreliable.

    High-level steps:
      Step 0. Detect heteroskedasticity (e.g., White test, Breusch-Pagan test, residual plots).
      Step 1. Model or estimate how the error variance changes across observations.
              For example, estimate sigma_i^2 as a function of X.
      Step 2. Define weights w_i = 1 / sigma_i^2 (observations with higher variance get lower weight).
      Step 3. Run a weighted regression that minimizes the weighted sum of squared residuals.

    Layman's term:
      If the spread of the errors gets bigger for some observations than others,
      treating every data point as equally reliable (OLS) makes your uncertainty
      estimates wrong. With WLS, you downweight the noisy points and upweight the
      more precise points so that your coefficient estimates and their confidence
      intervals better reflect the true information in the data.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        """Initialize WLS regression."""
        self.alpha: float = alpha
        self.beta: Optional[np.ndarray] = None

        self.coef_table: Optional[Dict[str, np.ndarray]] = None
        self.diagnostics: Optional[Dict[str, float]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, weights: np.ndarray, add_constant: bool = True) -> "WLS":
        """
        Fit the WLS regression model.

        Parameters
        ----------
        X : DataFrame
            Regressors in the model.
        y : Series
            Dependent variable.
        weights : array-like, shape (n,)
            Observation weights, typically w_i = 1 / sigma_i^2, where sigma_i^2
            is the estimated error variance for observation i.
        add_constant : bool
            Whether to add an intercept term to X.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        weights = np.asarray(weights).reshape(-1)
        X = sm.add_constant(X)
        n_samples, n_features = X.shape

        X_weighted, y_weighted = self.weighted_transform(X, y)
        OLS = sm.OLS(y_weighted, X_weighted)
        OLS_fit = OLS.fit()
        self.beta = OLS_fit.params

        # Fitted values and residuals (on original scale)
        y_pred = X_weighted @ self.beta
        resid = y - y_pred

        # Inference and diagnostics
        self._inference(X, resid, weights)
        self._diagnostics(X, y, resid)

        return self

    def _weighted_transform(self, X: np.ndarray, y: np.ndarray,weights: np.ndarray) -> np.ndarray:
        """Apply weighted transformation to X and y. """
        weights2 = np.sqrt(weights).reshape(-1, 1)
        X_weighted = weights2 * X
        y_weighted = weights2 * y
        return X_weighted, y_weighted

    def predict(self, X: pd.DataFrame, add_constant: bool=True) -> np.ndarray:
        """
        Predict values for new input data using the fitted WLS model.
        """
        if self.beta is None:
            raise ValueError("Model is not fitted yet.")

        X_df = X.copy()
        if add_constant:
            X_df = sm.add_constant(X_df, has_constant="add")

        X_mat = np.asarray(X_df)
        return X_mat @ self.beta

    def _inference(self, X: np.ndarray, resid: np.ndarray, weights: np.ndarray) -> None:
        """
        Compute standard errors, t-statistics, and confidence intervals for WLS.

        Under correctly specified WLS weights:
          Var(beta_WLS) = (X' W X)^(-1) * sigma^2
          where sigma^2 can be estimated from the weighted residuals.
        """
        n, k = X.shape
        w = weights.reshape(-1, 1)
        u = resid  # shape (n, 1)

        # Weighted RSS and effective variance estimate
        # s2 = sum(w_i * u_i^2) / (n - k)
        rss_w = float((w * u**2).sum())
        s2 = rss_w / (n - k)

        # X' W X
        X_w = np.sqrt(w) * X
        XTX_inv = np.linalg.inv(X_w.T @ X_w)
        var_beta = s2 * XTX_inv
        se_beta = np.sqrt(np.diag(var_beta))

        t_stats = self.beta / se_beta

        # Normal critical values
        z_crit = norm.ppf(1 - self.alpha / 2)
        ci_lower = self.beta - z_crit * se_beta
        ci_upper = self.beta + z_crit * se_beta

        self.coef_table = {
            "beta": self.beta,
            "std_err": se_beta,
            "t_stat": t_stats,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    def _diagnostics(self, X: np.ndarray, y: np.ndarray, resid: np.ndarray) -> None:
        """
        Basic diagnostics for the WLS fit.

        Currently:
          - R-squared (on original y)
          - Residual sum of squares (unweighted)
          - Number of observations and parameters
        """
        y = y.reshape(-1, 1)
        y_mean = y.mean()
        ss_tot = float(((y - y_mean) ** 2).sum())
        ss_res = float((resid**2).sum())
        r2 = 1 - ss_res / ss_tot

        self.diagnostics = {
            "r2_wls": r2,
            "rss": ss_res,
            "n_obs": X.shape[0],
            "n_params": X.shape[1],
        }