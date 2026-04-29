import pandas as pd
import numpy as np
from typing import Optional, Dict, List

import statsmodels.api as sm
from scipy import stats

class PooledOLS():
    """Pooled OLS for pandel data from scratch"""

    def __init__(self):
        """Initialize hyperparameters for the Pooled OLS."""
        self.beta = None
        self.sigma2 = None

        self.coef_table: Optional[Dict[str, np.ndarray]] = None
        self.diagnostics: Optional[Dict[str, float]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'PooledOLS':
        """Fit the Pooled OLS model to the training data."""
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        X = sm.add_constant(X)
        self.beta = np.linalg.solve(X.T @ X, X.T @ y)
        y_pred = X @ self.beta
        resid = y - y_pred
        self.sigma2 = np.sum(resid**2) / (n_samples - n_features)

        self._inference(X)
        self._diagnostics(X, y, resid)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the target variable using the fitted model."""
        X = np.asarray(X)
        X = sm.add_constant(X)
        return X @ self.beta
    
    def _inference(self, X: np.ndarray, alpha: float=0.05) -> None:
        """Calculate inference statistics for the fitted model."""
        n_sample, n_feature = X.shape

        coef = self.beta
        var = self.sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(var))
        t_score = coef / se
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_score), df=n_sample - n_feature))
        t_crit = stats.t.ppf(1 - alpha/2, df=n_sample - n_feature)
        ci_95 = np.column_stack([coef - t_crit*se, coef + t_crit*se])

        self.coef_table = {
            'coef': coef,
            'se': se,
            't_score': t_score,
            'p_value': p_value,
            'ci_95': ci_95
        }

    def _diagnostics(self, X: np.ndarray, y: np.ndarray, resid: np.ndarray) -> None:
        """Calculate diagnostics for the fitted model."""
        n_sample, n_feature = X.shape

        resid0 = y - y.mean()
        sigma20 = np.sum(resid0**2) / (n_sample - 1)
        logL0 = -0.5*(n_sample * np.log(2*np.pi*sigma20)) - 0.5*np.sum(resid0**2)/sigma20

        logL1 = -0.5*(n_sample * np.log(2*np.pi*self.sigma2)) - 0.5*np.sum(resid**2)/self.sigma2
        
        llr_stat = 2 * (logL1 - logL0)
        llr_pval = stats.chi2.sf(llr_stat, df=n_feature-1)
        aic = 2 * (n_feature + 1) - 2 * logL1
        bic = (n_feature + 1) * np.log(n_sample) - 2 * logL1

        ssr = np.sum(resid**2)
        sst = np.sum((y - y.mean())**2)
        r2 = 1 - ssr/sst

        f_stat = (sst - ssr) / (n_feature - 1) / (ssr / (n_sample - n_feature))
        f_pval = stats.f.sf(f_stat, dfn=n_feature - 1, dfd=n_sample - n_feature)

        self.diagnostics = {
            'logL0': logL0,
            'logL1': logL1,
            'llr_stat': llr_stat,
            'llr_pval': llr_pval,
            'aic': aic,
            'bic': bic,
            'r2': r2,
            'f_stat': f_stat,
            'f_pval': f_pval
        }

