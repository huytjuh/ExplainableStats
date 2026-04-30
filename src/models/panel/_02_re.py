import pandas as pd
import numpy as np 
from typing import Optional, Dict, List, Tuple

import statsmodels.api as sm
from scipy import stats

from models.panel._01_ols_pooled import PooledOLS

class RandomEffects():
    """Random Effects model for panel data from scratch"""

    def __init__(self) -> None:
        """Initialize hyperparameters for the Random Effects."""
        self.beta = None 
        self.sigma2 = None                  # Within variance (= idiosyncratic))
        self.sigma2_alpha = None            # Between variance entity variance (= unobserved heterogeneity)
        self.theta = None

        self.coef_table: Optional[Dict[str, np.ndarray]] = None
        self.diagnostics: Optional[Dict[str, float]] = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, entity_col: pd.Series) -> 'RandomEffects':
        """Fit the Random Effects model to the training data."""
        X = np.asarray(X)
        y = np.asarray(y)
        entity_col = np.asarray(entity_col)
        n_samples, n_features = X.shape

        # POOLED OLS
        OLS = PooledOLS()
        OLS_fit = OLS.fit(X, y)
        y_pred_pooled = OLS_fit.predict(X)
        resid_pooled = y - y_pred_pooled

        # VARIANCE OF RANDOM EFFECTS
        self.sigma2, self.sigma2_alpha = self._variance_re(X, y, resid_pooled, entity_col)
        
        # GLS TRANSFORMATION BY QUASI-DEMEANING
        X_tilde, y_tilde = self._gls_transform(X, y, entity_col)

        # OLS ON TRANSFORMED DATA
        GLS = PooledOLS()
        GLS_fit = GLS.fit(X_tilde, y_tilde)
        self.beta = GLS_fit.beta
        y_pred = GLS_fit.predict(X_tilde)
        resid = y_tilde - y_pred

        # INFERENCE & DIAGNOSTICS
        X_tilde = sm.add_constant(X_tilde)
        self._inference(X_tilde)
        self._diagnostics(X_tilde, y_tilde, resid)
    
        return self
    
    def _variance_re(self, X: np.ndarray, y: np.ndarray, resid_pooled: np.ndarray, entity_col: np.ndarray, method: str='wallace-hussain') -> Tuple[float, float]:
        """Calculate the variance of the Random Effects model."""
        n_samples, n_features = X.shape
        list_entities = np.unique(entity_col)
        n_entities = len(list_entities)
        entity_idx = np.searchsorted(list_entities, entity_col)

        # NUMBER OF OBSERVATIONS PER ENTITY, MEAN RESIDUAL PER ENTITY, AND AVERAGE NUMBER OF OBSERVATIONS PER ENTITY
        N_i = np.bincount(entity_idx, minlength=n_entities)
        e_bar = np.bincount(entity_idx, weights=resid_pooled, minlength=n_entities) / N_i
        N_bar = n_entities / np.sum(1 / N_i)

        if method == 'wallace-hussain':
            resid_dm = resid_pooled - e_bar[entity_idx]         # DE-MEAN RESIDUALS OF POOLED OLS BY ENTITY

            # WITHIN SUM SQUARE RESIDUALS
            SSW = np.sum(resid_dm**2)
            sigma2 = SSW / max(n_samples - n_features - n_entities, 1)

            # BETWEEN SUM SQUARE RESIDUALS
            SSB = np.sum(N_i * e_bar**2)                                   
            sigma2_b = SSB / max(n_entities - n_features - 1, 1)
            sigma2_alpha = max(sigma2_b - sigma2/N_bar, 0)

            return sigma2, sigma2_alpha

    def _gls_transform(self, X: np.ndarray, y: np.ndarray, entity_col: np.ndarray, eps: float=1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Transform the GLS model to the Random Effects model."""
        n_samples, n_features = X.shape
        list_entities = np.unique(entity_col)
        n_entities = len(list_entities)
        entity_idx = np.searchsorted(list_entities, entity_col)

        N_i = np.bincount(entity_idx, minlength=n_entities)
        theta = 1 - (self.sigma2 / (self.sigma2 + N_i * self.sigma2_alpha + eps))**0.5
        self.theta = theta[entity_idx]

        X_bar = np.column_stack([np.bincount(entity_idx, weights=X[:, col], minlength=n_entities) for col in range(n_features)]) / N_i[:, None]
        y_bar = np.bincount(entity_idx, weights=y, minlength=n_entities) / N_i
        X_tilde = X - self.theta[:, None] * X_bar[entity_idx]
        y_tilde = y - self.theta * y_bar[entity_idx]

        return X_tilde, y_tilde

    def predict(self, X: pd.DataFrame, entity_col: pd.Series) -> np.ndarray:
        """Predict the target variable using the fitted model."""
        X = np.asarray(X)
        entity_col = np.asarray(entity_col)
        X_tilde, _ = self._transform_gls(X, np.zeros(len(X)), entity_col)
        X_tilde = sm.add_constant(X_tilde)
        return X_tilde @ self.beta
    
    def _inference(self, X: np.ndarray, alpha: float=0.05) -> None:
        """Calculate inference statistics for the fitted model."""
        n_samples, n_features = X.shape
        
        coef = self.beta
        var = self.sigma2 * np.linalg.inv(X.T @ X)
        se = np.diag(var)**0.5
        t_stat = coef / se 
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_samples - n_features))
        t_crit = stats.t.ppf(1 - alpha/2, df=n_samples - n_features)
        ci_95 = np.column_stack([coef - t_crit*se, coef + t_crit*se])

        self.coef_table = {
            'coef': coef,
            'se': se,
            't_stat': t_stat,
            'p_value': p_value,
            'ci_95': ci_95
        }

    def _diagnostics(self, X: np.ndarray, y: np.ndarray, resid: np.ndarray) -> None:
        """Calculate diagnostics for the fitted model."""
        n_samples, n_features = X.shape 

        resid0 = y - y.mean()
        sigma20 = np.sum(resid0**2) / (n_samples - 1)
        logL0 = -0.5 * (n_samples * np.log(2 * np.pi * sigma20)) - 0.5 * np.sum(resid0**2) / sigma20

        logL1 = -0.5 * (n_samples * np.log(2 * np.pi * self.sigma2)) - 0.5 * np.sum(resid**2) / self.sigma2

        llr_stat = 2 * (logL1 - logL0)
        llr_pval = stats.chi2.sf(llr_stat, df=n_features - 1)
        aic = 2 * n_features - 2 * logL1
        bic = n_features * np.log(n_samples) - 2 * logL1

        ssr = np.sum(resid**2)
        sst = np.sum((y - y.mean())**2)
        r2 = 1 - ssr/sst
        r2_adj = 1 - (1 - r2) * (n_samples - 1) / max(n_samples - n_features, 1)

        f_stat = (sst - ssr) / (n_features - 1) / (ssr / (n_samples - n_features))
        f_pval = stats.f.sf(f_stat, dfn=n_features - 1, dfd=n_samples - n_features)

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