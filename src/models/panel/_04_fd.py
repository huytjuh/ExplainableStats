import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple

from scipy import stats

from models.panel._01_ols_pooled import PooledOLS

class FirstDifference():
    """First-Difference model for panel data from scratch"""

    def __init__(self) -> None:
        """Initialize hyperparameters for the First-Difference."""
        self.beta = None 
        self.sigma2 = None

        self.coef_table: Optional[Dict[str, np.ndarray]] = None
        self.diagnostics: Optional[Dict[str, float]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, entity_col: pd.Series, date_col: pd.Series) -> 'FirstDifference':
        """Fit the First-Difference model to the training data."""
        X = np.asarray(X)
        y = np.asarray(y)
        entity_col = np.asarray(entity_col)
        date_col = np.asarray(date_col)
        
        n_samples, n_features = X.shape 
        list_entities = np.unique(entity_col)
        n_entities = len(list_entities)

        # FIRST DIFFERENCE ESTIMATOR
        dX, dy = self._fd_transform(X, y, entity_col, date_col)

        # OLS ON FIRST-DIFFERENCED DATA
        OLS = PooledOLS()
        OLS_fit = OLS.fit(dX, dy, constant=False)
        self.beta = OLS_fit.beta
        y_pred = OLS_fit.predict(dX, constant=False)
        resid = dy - y_pred
        self.sigma2 = np.sum(resid**2) / max(n_samples - n_entities - n_features, 1)

        # INFERENCE & DIAGNOSTICS
        self._inference(dX, entity_col)
        self._diagnostics(dX, dy, resid, entity_col)

        # Breusch-Godfrey proxy: correlation between resid_t and resid_{t-1}

        return self

    def _fd_transform(self, X: np.ndarray, y: np.ndarray, entity_col: np.ndarray, date_col: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """First-difference transformation to remove time-invariant unobserved heterogeneity."""
        sorted_idx = np.lexsort((date_col, entity_col))
        X_sorted = X[sorted_idx]
        y_sorted = y[sorted_idx]
        entity_sorted = entity_col[sorted_idx]

        mask = np.concatenate(([False], entity_sorted[1:] == entity_sorted[:-1]))
        idx_curr = np.where(mask)[0]
        dX = X_sorted[idx_curr] - X_sorted[idx_curr - 1]
        dy = y_sorted[idx_curr] - y_sorted[idx_curr - 1]

        return dX, dy

    def predict(self, X: pd.DataFrame, entity_col: pd.Series, date_col: pd.Series) -> np.ndarray:
        """Predict the target variable using the fitted model."""
        X = np.asarray(X)
        entity_col = np.asarray(entity_col)
        date_col = np.asarray(date_col)

        dX, _ = self._fd_transform(X, np.zeros(len(X)), entity_col, date_col)
        return dX @ self.beta

    def _inference(self, X: np.ndarray, entity_col: np.ndarray, alpha: float=0.05) -> None:
        """Calculate inference statistics for the fitted model."""
        n_samples, n_features = X.shape
        list_entities = np.unique(entity_col)
        n_entities = len(list_entities)

        coef = self.beta 
        var = self.sigma2 * np.linalg.inv(X.T @ X)
        se = np.diag(var)**0.5
        t_score = coef / se
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_score), df=max(n_samples - n_entities - n_features, 1)))
        t_crit = stats.t.ppf(1 - alpha/2, df=max(n_samples - n_entities - n_features, 1))
        ci_95 = np.column_stack([coef - t_crit*se, coef + t_crit*se])
        self.coef_table = {
            'coef': coef,
            'se': se,
            't_score': t_score,
            'p_value': p_value,
            'ci_95': ci_95
        }

    def _diagnostics(self, X: np.ndarray, y: np.ndarray, resid: np.ndarray, entity_col: np.ndarray) -> None:
        """Calculate diagnostics for the fitted model."""
        n_samples, n_features = X.shape 
        list_entities = np.unique(entity_col)
        n_entities = len(list_entities)

        resid0 = y - y.mean()
        sigma20 = np.sum(resid0**2) / (n_samples - n_entities - 1)
        logL0 = -0.5 * n_samples * np.log(2 * np.pi * sigma20) - 0.5 * np.sum(resid**2) / sigma20

        logL1 = -0.5 * n_samples * np.log(2 * np.pi * self.sigma2) - 0.5 * np.sum(resid**2) / self.sigma2

        llr_stat = 2 * (logL1 - logL0)
        llr_pval = stats.chi2.sf(llr_stat, df=n_features)
        aic = 2 * n_features - 2 * logL1
        bic = n_features * np.log(n_samples - n_entities) - 2 * logL1
        
        ssr = np.sum(resid**2)
        sst = np.sum((y - y.mean())**2)
        r2 = 1 - ssr/sst
        r2_adj = 1 - (1 - r2) * (n_samples - n_entities - 1) / max(n_samples - n_features - n_entities, 1)
        
        f_stat = (sst - ssr) / n_features / (ssr / max(n_samples - n_features - n_entities, 1))
        f_pval = stats.f.sf(f_stat, dfn=n_features, dfd=max(n_samples - n_features - n_entities, 1))

        self.diagnostics = {
            'logL0': logL0,
            'logL1': logL1,
            'llr_stat': llr_stat,
            'llr_pval': llr_pval,
            'aic': aic,
            'bic': bic,
            'r2': r2,
            'r2_adj': r2_adj,
            'f_stat': f_stat,
            'f_pval': f_pval
        }