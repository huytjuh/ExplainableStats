import pandas as pd
import numpy as np 
from typing import Optional, Dict

import statsmodels.api as sm
from scipy.stats import norm

class IV:
    """Instrumental Variable Regression from scratch.
    
    Step 0a. Use Durbin Watson test to check autocorrelation of residuals.
    Step 0b. Use <insert test> to check endogeneity
    Step 1. Instrumentally regress X on Z
    Step 2. Endogenosly regress Y on X and Z

    Layman's term:
    If endogeneity in the model which results in correlated errors and biased coefficients (why?), therefore use IV
    """

    def __init__(self, alpha: float=0.05) -> None:
        """Initialize IV regression."""
        self.alpha = None
        self.beta = None

        self.coef_table = Optional[Dict[str, np.ndarray]]
        self.diagnostics = Optional[Dict[str, float]]

    def fit(self, X: pd.DataFrame, y: pd.Series, Z: pd.DataFrame) -> 'IV':
        """Fit the IV regression model."""
        X = np.asarray(X)
        y = np.asarray(y)
        Z = np.asarray(Z)
        n_samples, n_features = X.shape

        # FIRST STAGE
        x = sm.add_constant(X)
        X_hat = self._2SLS(X, Z)

        # SECOND STAGE
        OLS_IV = sm.OLS(y, X_hat)
        OLS_IV_fit = OLS_IV.fit()
        self.beta = OLS_IV_fit.params

        y_pred = X_hat @ self.beta
        resid = y - y_pred

        self._inference(X, y, resid)
        self._diagnostics(X, y, resid)

        return self
    
    def _2SLS(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Endogenously regress Y on X and Z."""
        ZtZ_inv = np.linalg.inv(Z.T @ Z)
        P_Z = Z @ ZtZ_inv @ Z.T

        return P_Z @ X

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict values for the input data."""
        X = np.asarray(X)
        return X @ self.beta

    def _inference(self, X_hat: np.ndarray, resid: np.ndarray) -> None:
        """
        Calculate basic inference statistics (standard errors, t-stats, CI) for the fitted model.

        Uses homoskedastic OLS formulas on the second stage:
          var(beta_IV) = s2 * (X_hat' X_hat)^(-1)
          with s2 = RSS / (n - k)
        """
        n, k = X_hat.shape
        u = resid.reshape(-1, 1)
        rss = float(u.T @ u)
        s2 = rss / (n - k)  # error variance estimate

        XTX_inv = np.linalg.inv(X_hat.T @ X_hat)
        var_beta = s2 * XTX_inv
        se_beta = np.sqrt(np.diag(var_beta))

        t_stats = self.beta / se_beta

        # Normal approx for critical values
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
        Calculate simple diagnostics for the fitted model.

        Currently:
          - R-squared of second stage
          - Residual sum of squares (RSS)
          - Number of observations and parameters

        Can be extended with:
          - First-stage F-statistics (instrument relevance)
          - Overidentification tests (if #instruments > #endogenous regressors)
        """
        y = y.reshape(-1, 1)
        y_mean = y.mean()
        ss_tot = float(((y - y_mean) ** 2).sum())
        ss_res = float((resid.reshape(-1, 1) ** 2).sum())
        r2 = 1 - ss_res / ss_tot

        self.diagnostics = {
            "r2_iv_second_stage": r2,
            "rss_second_stage": ss_res,
            "n_obs": X.shape[0],
            "n_params": X.shape[1],
        }