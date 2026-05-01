import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple

import statsmodels.api as sm
from scipy import stats

from models.panel._01_ols_pooled import PooledOLS

class FixedEffects():
    """Fixed Effects model for panel data from scratch"""

    def __init__(self) -> None:
        """Initialize hyperparameters for the Fixed Effects."""
        self.beta = None 
        self.alpha = None
        self.sigma2 = None

        self.coef_table: Optional[Dict[str, np.ndarray]] = None
        self.diagnostics: Optional[Dict[str, float]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, entity_col: pd.Series) -> 'FixedEffects':
        """Fit the Fixed Effects model to the training data."""
        X = np.asarray(X)
        y = np.asarray(y)
        entity_col = np.asarray(entity_col)

        n_samples, n_features = X.shape
        list_entitites = np.unique(entity_col)
        n_entities = len(list_entitites)

        # WITHIN ESTIMATOR TO SUBTRACT TIME-INVARIANT UNOBSERVED HETEROGENEITY (= CUSTOMER BIAS: WEALTH, RISK APPETITE, ETC.)
        X_dm, y_dm, X_bar, y_bar = self._within_transform(X, y, entity_col)

        # OLS ON DEMEANED DATA 
        OLS = PooledOLS()
        OLS_fit = OLS.fit(X_dm, y_dm, constant=False)
        self.beta = OLS_fit.beta
        y_pred_dm = OLS_fit.predict(X_dm, constant=False)
        resid_dm = y_dm - y_pred_dm

        # VARIANCE OF FIXED EFFECTS
        self.sigma2 = np.sum(resid_dm**2) / max(n_samples - n_entities - n_features, 1)

        # ENTITY INTERCEPT
        self.alpha = y_bar - X_bar @ self.beta

        # INFERENCE & DIAGNOSTICS
        self._inference(X_dm, entity_col)
        self._diagnostics(X_dm, y_dm, resid_dm, entity_col)

        # HAUSMANN TEST: Compare FE vs RE by testing if the entity effects are correlated with the regressors (i.e. if the entity effects are truly random or not)

        return self

    def _within_transform(self, X: np.ndarray, y: np.ndarray, entity_col: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        return X_dm, y_dm, X_bar, y_bar

    def predict(self, X: pd.DataFrame, entity_col: pd.Series) -> np.ndarray:
        """Predict using the Fixed Effects model."""
        X = np.asarray(X)
        entity_col = np.asarray(entity_col)
        list_entities = np.unique(entity_col)
        entity_idx = np.searchsorted(list_entities, entity_col)
        return self.alpha[entity_idx] + X @ self.beta

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
        sigma20 = np.sum(resid0**2) / (n_samples - 1)
        logL0 = -0.5 * (n_samples * np.log(2 * np.pi * sigma20)) - 0.5 * np.sum(resid0**2) / sigma20

        logL1 = -0.5 * (n_samples * np.log(2 * np.pi * self.sigma2)) - 0.5 * np.sum(resid**2) / self.sigma2
        
        llr_stat = 2 * (logL1 - logL0)
        llr_pval = stats.chi2.sf(llr_stat, df=n_features)
        aic = 2 * (n_features + n_entities) - 2 * logL1
        bic = (n_features + n_entities) * np.log(n_samples) - 2 * logL1

        ssr = np.sum(resid**2)
        sst = np.sum((y - y.mean())**2)
        r2 = 1 - ssr/sst
        r2_adj = 1 - (1 - r2) * (n_samples - 1) / max(n_samples - n_features - n_entities, 1)

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