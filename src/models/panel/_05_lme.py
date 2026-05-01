import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple 

from scipy import stats
import statsmodels.api as sm

from models.panel._01_ols_pooled import PooledOLS
from models.panel._02_re import RandomEffects
from models.panel._03_fe import FixedEffects

class LinearMixedEffects():
    """Linear Mixed Effects for panel data from scratch"""

    def __init__(self, max_iter: int=100, tol: float=1e-6) -> None:
        """Initialize hyperparameters for the Linear Mixed Effects."""
        self.max_iter = max_iter
        self.tol = tol

        self.beta = None 
        self.beta_fe = None
        self.beta_re = None 

        self.sigma2 = None 
        self.sigma2_alpha = None

        self.random_effects = None

        self.coef_table: Optional[Dict[str, np.ndarray]] = None
        self.diagnostics: Optional[Dict[str, float]] = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, entity_col: pd.Series) -> 'LinearMixedEffects':
        """Fit the Linear Mixed Effects model to the training data."""
        X = np.asarray(X)
        y = np.asarray(y)
        entity_col = np.asarray(entity_col)
        list_entities = np.unique(entity_col)
        X = sm.add_constant(X)

        # INITIALIE M-STEP
        self._initialize(X[:, 1:], y, entity_col)

        # EM ALGORITHM
        # list_beta, list_sigma2, list_sigma2_alpha = [self.beta], [self.sigma2], [self.sigma2_alpha]
        list_loss = [-np.inf]
        for _ in range(self.max_iter):

            m_i, s_i = self._e_step(X, y, entity_col)
            self._m_step(X, y, entity_col, m_i, s_i)

            loss = self._log_likelihood(X, y, entity_col)
            list_loss.append(loss)

            if abs(list_loss[-2] - list_loss[-1]) < self.tol:
                break

        m_i, _ = self._e_step(X, y, entity_col)
        self.random_effects = {entity: m_i[i] for i, entity in enumerate(list_entities)}
        y_pred = self.predict(X[:, 1:], entity_col)
        resid = y - y_pred

        # INFERENCE & DIAGNOSTICS
        self._inference(X)
        self._diagnostics(X, y, resid)

        return self

    def _initialize(self, X: np.ndarray, y: np.ndarray, entity_col: np.ndarray) -> None:
        """Initialize hyperparameters for the Linear Mixed Effects."""
        RE = RandomEffects()
        RE_fit = RE.fit(X, y, entity_col)
        self.beta_re = RE_fit.beta
        self.sigma2 = RE_fit.sigma2
        self.sigma2_alpha = RE_fit.sigma2_alpha

        FE = FixedEffects()
        FE_fit = FE.fit(X, y, entity_col)
        self.beta_fe = FE_fit.beta

        self.beta = self.beta_re

    def _e_step(self, X: np.ndarray, y: np.ndarray, entity_col: np.ndarray, eps: float=1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """E-step of the EM algorithm to estimate the random effects."""
        list_entities = np.unique(entity_col)
        n_entities = len(list_entities)
        entity_idx = np.searchsorted(list_entities, entity_col)

        resid = y - X @ self.beta
        N_i = np.bincount(entity_idx, minlength=n_entities)
        e_i = np.bincount(entity_idx, weights=resid, minlength=n_entities)
        denom = self.sigma2 + N_i * self.sigma2_alpha + eps

        m_i = self.sigma2_alpha * e_i / denom
        s_i = self.sigma2_alpha * self.sigma2 / denom

        return m_i, s_i

    def _m_step(self, X: np.ndarray, y: np.ndarray, entity_col: np.ndarray, m_i: np.ndarray, s_i: np.ndarray, eps: float=1e-6) -> None:
        """M-step of the EM algorithm to update the fixed effects and variance components."""
        list_entity = np.unique(entity_col)
        n_entities = len(list_entity)
        entity_idx = np.searchsorted(list_entity, entity_col)

        # UPDATE BETA
        y_tilde = y - m_i[entity_idx]
        OLS = PooledOLS()
        OLS_fit = OLS.fit(X, y_tilde, constant=False)
        self.beta = OLS_fit.beta

        # UPDATE SIGMA
        resid = y - X @ self.beta
        resid_tilde = resid - m_i[entity_idx]
        self.sigma2 = max(np.mean(resid_tilde**2 + s_i[entity_idx]), eps)
        self.sigma2_alpha = max(np.mean(m_i**2 + s_i), eps)

    def _log_likelihood(self, X: np.ndarray, y: np.ndarray, entity_col: np.ndarray) -> float:
        """Calculate the log-likelihood of the fitted model."""
        logL = 0 
        for entity in np.unique(entity_col):
            X_i = X[entity_col == entity]
            y_i = y[entity_col == entity]
            n_i = len(y_i)

            resid_i = y_i - X_i @ self.beta
            V_i = self.sigma2 * np.eye(n_i) + self.sigma2_alpha * np.ones((n_i, n_i))
            V_i_inv = np.linalg.inv(V_i)

            sign, logdet = np.linalg.slogdet(V_i)
            if sign <= 0:
                return -np.inf

            logL += -0.5 * n_i * np.log(2 * np.pi) -0.5 * logdet - 0.5 * resid_i.T @ V_i_inv @ resid_i

        return logL

    def predict(self, X: pd.DataFrame, entity_col: pd.Series) -> np.ndarray:
        """Predict using the fitted Linear Mixed Effects model."""
        X = np.asarray(X)
        entity_col = np.asarray(entity_col)
        X = sm.add_constant(X)

        y_hat = X @ self.beta
        u_hat = np.array([self.random_effects[entity] for entity in entity_col])

        return y_hat + u_hat

    def _inference(self, X: np.ndarray, alpha: float=0.05) -> None:
        """Calculate inference for the fitted model."""
        n_samples, n_features = X.shape

        coef = self.beta
        var = self.sigma2 * np.linalg.pinv(X.T @ X)
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

        logL1 = self._log_likelihood(X, y, np.zeros(len(y)))
        llr_stat = 2 * (logL1 - logL0)
        llr_pval = stats.chi2.sf(llr_stat, df=n_features)
        aic = 2 * (n_features + 2) - 2 * logL1
        bic = (n_features + 2) * np.log(n_samples) - 2 * logL1

        ssr = np.sum((resid)**2)
        sst = np.sum((y - y.mean())**2)
        r2 = 1 - ssr/sst
        r2_adj = 1 - (1 - r2) * (n_samples - 1) / max(n_samples - n_features - 2, 1)

        f_stat = (sst - ssr) / (n_features - 1) / (ssr / (n_samples - n_features - 2))
        f_pval = stats.f.sf(f_stat, dfn=n_features - 1, dfd=n_samples - n_features - 2)

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
