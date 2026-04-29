import pandas as pd
import numpy as np 
from typing import Optional, Dict, List, Tuple

import statsmodels.api as sm

from _01_ols_pooled import PooledOLS

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
        X_tilde, y_tilde = self._transform_gls(X, y, entity_col)

        GLS = PooledOLS()
        GLS_fit = GLS.fit(X_tilde, y_tilde)
        y_pred = GLS_fit.predict(X_tilde)
        resid = y_tilde - y_pred
    
        return self
    
    def _variance_re(self, X: np.ndarray, y: np.ndarray, resid_pooled: np.ndarray, entity_col: np.ndarray, method: str='wallace-hussain') -> Tuple[float, float]:
        """Calculate the variance of the Random Effects model."""
        n_samples, n_features = X.shape
        list_entities = np.unique(entity_col)
        n_entities = len(list_entities)
        
        # NUMBER OF OBSERVATIONS PER ENTITY, MEAN RESIDUAL PER ENTITY, AND AVERAGE NUMBER OF OBSERVATIONS PER ENTITY
        T_i = np.array([np.sum(entity_col == entity) for entity in list_entities])
        e_bar = np.array([np.mean(resid_pooled[entity_col == entity]) for entity in list_entities])
        T_bar = n_entities / np.sum(1 / T_i)

        if method == 'wallace-hussain':
            resid_dm = resid_pooled.copy()

            # DE-MEAN RESIDUALS OF POOLED OLS BY ENTITY
            for entity in list_entities:
                resid_dm[entity_col == entity] -= e_bar[list_entities == entity]

            # WITHIN SUM SQUARE RESIDUALS
            SSW = np.sum(resid_dm**2)
            sigma2 = SSW / max(n_samples - n_features - n_entities, 1)

            # BETWEEN SUM SQUARE RESIDUALS
            SSB = np.sum(T_i * e_bar**2)                                   
            sigma2_b = SSB / max(n_entities - n_features - 1, 1)

            # SIGMA
            sigma2_alpha = max(sigma2_b - sigma2/T_bar, 0)
            return sigma2, sigma2_alpha

        if method == 'amemiya':
            sigma2_alpha = max((np.sum(T_i * e_bar**2) - np.sum(resid_pooled**2)) / (n_entities - n_features), 0)
            sigma2 = max((np.sum(resid_pooled**2) - np.sum(T_i * e_bar**2)) / (n_samples - n_features), 0)
            return sigma2, sigma2_alpha

    def _transform_gls(self, X: np.ndarray, y: np.ndarray, entity_col: np.ndarray, eps: float=1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Transform the GLS model to the Random Effects model."""
        X_tilde = X.copy()
        y_tilde = y.copy()
        list_entities = np.unique(entity_col)

        for entity in list_entities:
            idx = entity_col == entity
            n_i = np.sum(idx)
            theta_i = 1 - (self.sigma2 / (self.sigma2 + n_i * self.sigma2_alpha + eps))**0.5
            X_tilde[idx] = X[idx] - theta_i * X[idx].mean(axis=0)
            y_tilde[idx] = y[idx] - theta_i * y[idx].mean()

        n_samples, n_features = X.shape
        list_entities = np.unique(entity_col)
        n_entities = len(list_entities)

        pass

    def predict(self, X: pd.DataFrame, entity_col: pd.Series) -> np.ndarray:
        """Predict the target variable using the fitted model."""
        X = np.asarray(X)
        entity_col = np.asarray(entity_col)
        X_tilde, _ = self._transform_gls(X, np.zeros(X.shape[0]), entity_col)
        return X_tilde @ self.beta
    
    def _inference(self, X: np.ndarray, alpha: float=0.05) -> None:
        """Calculate inference statistics for the fitted model."""
        pass

    def _diagnostics(self, X: np.ndarray, y: np.ndarray, resid: np.ndarray) -> None:
        """Calculate diagnostics for the fitted model."""
        pass