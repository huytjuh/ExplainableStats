import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

from scipy import stats
import statsmodels.api as sm

from models.panel._01_ols_pooled import PooledOLS
from models.panel._03_fe import FixedEffects

class ErrorCorrectionModel():
    """"Error Correction Model for panel data from scratch"""

    def __init__(self) -> None:
        """Initialize hyperparameters for the Error Correction Model."""
        self.beta_long = None
        self.alpha = None
        self.e = None

        self.beta_short = None
        self.sigma2 = None

        self.entities = None
        self.entity_idx = None

        self.coef_table: Optional[Dict[str, Dict[str, np.ndarray]]] = {}
        self.diagnostics: Optional[Dict[str, Dict[str, float]]] = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, entity_col: pd.Series, time_col: pd.Series) -> 'ErrorCorrectionModel':
        """Fit the Error Correction Model to the training data."""
        X = np.asarray(X)
        y = np.asarray(y)
        entity_col = np.asarray(entity_col)
        time_col = np.asarray(time_col)

        sort_idx = np.lexsort((time_col, entity_col))
        X = X[sort_idx]
        y = y[sort_idx]
        entity_col = entity_col[sort_idx]
        time_col = time_col[sort_idx]

        self.entities = np.unique(entity_col)
        self.entity_idx = np.searchsorted(self.entities, entity_col)

        self._long_run_regression(X, y, entity_col)
        self._short_run_regression(X, y, entity_col)

        return self
        
    def _long_run_regression(self, X: np.ndarray, y: np.ndarray, entity_col: np.ndarray) -> None:
        """Fit the long-run regression for the Error Correction Model."""
        FE = FixedEffects()
        FE_fit = FE.fit(X, y, entity_col)
        beta_fe = FE_fit.beta
        alpha_fe = FE_fit.alpha

        self.beta_long = beta_fe
        self.alpha = {entity: alpha_fe[i] for i, entity in enumerate(self.entities)}
        self.e = y - alpha_fe[self.entity_idx] - X @ self.beta_long

        self.coef_table['long_run'] = FE_fit.coef_table
        self.diagnostics['long_run'] = FE_fit.diagnostics

    def _short_run_regression(self, X: np.ndarray, y: np.ndarray, entity_col: np.ndarray) -> None:
        """Fit the short-run regression for the Error Correction Model."""
        mask = np.concatenate(([False], entity_col[1:] == entity_col[:-1]))
        idx_curr = np.where(mask)[0]
        dX = X[idx_curr] - X[idx_curr - 1]
        dy = y[idx_curr] - y[idx_curr - 1]
        e_lag = self.e[idx_curr - 1]

        X_ecm = sm.add_constant(np.column_stack([dX, e_lag]))
        OLS = PooledOLS()
        OLS_fit = OLS.fit(X_ecm, dy, constant=False)
        self.beta_short = OLS_fit.beta
        self.sigma2 = OLS_fit.sigma2

        self.coef_table['short_run'] = OLS_fit.coef_table
        self.diagnostics['short_run'] = OLS_fit.diagnostics

    def predict_short(self, X: pd.DataFrame, entity_col: pd.Series, time_col: pd.Series) -> np.ndarray:
        """Predict using the fitted Error Correction Model."""
        X = np.asarray(X)
        entity_col = np.asarray(entity_col)
        time_col = np.asarray(time_col)

        sort_idx = np.lexsort((time_col, entity_col))
        X = X[sort_idx]
        entity_col = entity_col[sort_idx]
        time_col = time_col[sort_idx]

        entity_idx = np.searchsorted(self.entities, entity_col)
        mask = np.concatenate(([False], entity_col[1:] == entity_col[:-1]))
        idx_curr = np.where(mask)[0]

        dX = X[idx_curr] - X[idx_curr - 1]
        alpha = self.alpha[entity_idx]
        e_hat = -alpha[entity_idx] - X @ self.beta_long
        e_lag = e_hat[idx_curr - 1]

        X_ecm = sm.add_constant(np.column_stack([dX, e_lag]), has_constant='add')

        return X_ecm @ self.beta_short

    def predict_long(self, X: pd.DataFrame, entity_col: pd.Series) -> np.ndarray:
        """Predict the long-run component using the fitted model."""
        X = np.asarray(X)
        entity_col = np.asarray(entity_col)
        X = sm.add_constant(X)

        alpha = self.alpha[entity_col]
        return alpha + X @ self.beta_long

    @property
    def speed_of_adjustment(self) -> float:
        """Return the speed of adjustment."""
        return float(self.beta_short[1])