import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple 

from scipy.optimize import minimize

class CoxProportionalHazards:
    """Cox Proportional Hazards Model

    Hazard:
        h_i(t) = h_0(t) * exp(x_i' beta)

    Estimated by maximizing the Cox partial likelihood
    (implemented here as minimizing the negative partial log-likelihood)
    using the Breslow approximation for tied event times.
    
    """
    def __init__(self, max_iter: int=100, tol: float=1e-6) -> None:
        """Initialize hyperparameters for the Cox Proportional Hazards."""
        self.max_iter = max_iter
        self.tol = tol

        self.beta = None

        self.coef_table: Optional[Dict[str, np.ndarray]] = None
        self.diagnostics: Optional[Dict[str, float]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, time: pd.Series, event: pd.Series) -> 'CoxProportionalHazards':
        """Fit the Cox Proportional Hazards model to the training data."""
        X = np.asarray(X)
        y = np.asarray(X)
        time = np.asarray(time)
        event = np.asarray(event)
        n_samples, n_features = X.shape

        beta0 = np.zeros(n_features)
        opt = minimize(self._neg_loglik, x0=beta0, args=(X, time, event), method="BFGS", options={"maxiter": self.max_iter, "ftol": self.tol})
        self.beta = opt.x
        return self

    def _neg_log_likelihood(self, X: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
        """Negative Cox partial log-likelihood using Breslow approximation for ties.

        Formula:
            L(β) = ∏_t [ exp(∑_{i∈D_t} x_i'β) / (∑_{j∈R_t} exp(x_j'β))^d_t ]
        
        Negative log-likelihood:
            -logL(β) = -∑_t [∑_{i∈D_t} x_i'β - d_t * log(∑_{j∈R_t} exp(x_j'β))]
        
        Where:
        - D_t = individuals with event at time t
        - R_t = individuals at risk at time t (time >= t)
        - d_t = number of events at time t
        """
        eta = X @ self.beta
        exp_eta = np.exp(eta)
        logL = 0
        for t in np.unique(time[event == 1]):
            event_idx = (time == t) & (event == 1)
            risk_idx = (time >= t)
            d = event_idx.sum() 

            logL = logL + eta[event_idx].sum() - d * np.log(exp_eta[risk_idx].sum()) 

        return -logL

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the fitted Cox Proportional Hazards model."""
        X = np.asarray(X)
        return np.exp(X @ self.beta)

    def _inference(self, X: np.ndarray, alpha: float=0.05) -> Dict[str, np.ndarray]:
        """Calculate inference statistics for the fitted Cox Proportional Hazards model."""
        pass

    def _diagnostics(self, X: np.ndarray, y: np.ndarray, resid: np.ndarray) -> Dict[str, float]:
        """Calculate diagnostics for the fitted Cox Proportional Hazards model."""
        pass