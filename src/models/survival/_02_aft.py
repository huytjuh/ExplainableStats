import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

from scipy.optimize import minimize
from scipy.stats import norm

class AcceleratedFailureTime:
    """Log-Normal Accelerated Failure Time model.

    Model:
        log(T_i) = x_i' beta + sigma * epsilon_i
        epsilon_i ~ N(0, 1)

    So:
        T_i | x_i ~ LogNormal(mean=x_i' beta, sigma=sigma)

    Right-censored log-likelihood:
        For event_i = 1:
            contribution = log f(t_i | x_i)
        For event_i = 0:
            contribution = log S(t_i | x_i)

    where:
        z_i = (log t_i - x_i' beta) / sigma
        f(t_i | x_i) = (1 / (t_i * sigma)) * phi(z_i)
        S(t_i | x_i) = 1 - Phi(z_i)
    """

    def __init__(self, max_iter: int=100, tol: float=1e-6) -> None:
        """Initialize the Accelerated Failure Time model."""
        self.max_iter = max_iter
        self.tol = tol

        self.beta = None
        self.sigma = None

        self.coef_table: Optional[Dict[str, np.ndarray]] = None
        self.diagnostics: Optional[Dict[str, float]] = None

    def fit(self, X: pd.DataFrame, y_time: pd.Series, y_event: pd.Series) -> 'AcceleratedFailureTime':
        """Fit the Accelerated Failure Time model to the training data."""
        X = np.asarray(X)
        y_time = np.asarray(y_time)
        y_event = np.asarray(y_event)
        n_samples, n_features = X.shape

        beta0 = np.zeros(n_features)
        log_sigma0 = np.array([0.0])
        theta0 = np.concatenate([beta0, log_sigma0])
        opt = minimize(self._neg_loglik, x0=theta0, args=(X, y_time, y_event), method="BFGS", options={"maxiter": self.max_iter, "tol": self.tol})

        self.beta = opt.x[:-1]
        self.sigma = opt.x[-1]

        return self
    
    def _neg_log_likelihood(self, theta: np.ndarray, X: np.ndarray, y_time: np.ndarray, y_event: np.ndarray) -> float:
        """Negative log-likelihood for right-censored Log-Normal AFT model.
        
        Model:
            log(T_i) = x_i'β + σε_i,  where ε_i ~ N(0,1)
        
        Likelihood (mixture of PDF for events and survival function for censored):
            L(β, σ) = ∏_i [f(t_i | x_i)]^δ_i × [S(t_i | x_i)]^(1-δ_i)
        
        Where:
            z_i = (log(t_i) - x_i'β) / σ
            f(t_i | x_i) = (1 / (t_i × σ)) × φ(z_i)     [log-normal PDF]
            S(t_i | x_i) = 1 - Φ(z_i)                    [survival function]
            φ(·) = standard normal PDF
            Φ(·) = standard normal CDF
        
        Log-likelihood:
            logL(β, σ) = ∑_i δ_i × log f(t_i | x_i) + (1 - δ_i) × log S(t_i | x_i)
                    = ∑_i δ_i × [-log(t_i) - log(σ) + log φ(z_i)] + (1 - δ_i) × log[1 - Φ(z_i)]
        
        Negative log-likelihood:
            -logL(β, σ) = -∑_i [δ_i × (-log(t_i) - log(σ) + log φ(z_i)) + (1 - δ_i) × log(1 - Φ(z_i))]
        
        Parameters:
        - δ_i = y_event[i] ∈ {0,1}  (1 = event observed, 0 = right-censored)
        - t_i = y_time[i] > 0       (observed time)
        - x_i = X[i, :]             (covariate vector)
        """
        beta = theta[:-1]
        log_sigma = theta[-1]
        sigma = np.exp(log_sigma)

        z = (y_time - X @ beta) / sigma
        log_pdf = -np.log(y_time) - log_sigma - norm.logpdf(z)
        log_S = norm.logsf(z)

        logL = np.sum((y_event - 1) * log_pdf + y_event * log_S)

        return -logL
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the fitted Accelerated Failure Time model."""
        return X @ self.beta

    def _inference(self, X: np.ndarray, alpha: float=0.05) -> None:
        """Calculate inference statistics for the fitted Accelerated Failure Time model."""
        pass 

    def _diagnostics(self, X: np.ndarray, y: np.ndarray, resid: np.ndarray) -> Dict[str, float]:
        """Calculate diagnostics for the fitted Accelerated Failure Time model."""
        pass

