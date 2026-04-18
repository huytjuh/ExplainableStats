import pandas as pd
import numpy as np 
from typing import Optional, Dict

from scipy import stats

# from utils import sigmoid

class Logit():
    """Logistic Regression Classifier from scratch."""
    
    def __init__(self, lr: float=0.01, n_iter: int=100, tol: float=1e-4):
        """Initialize hyperparameters for the Logistic Regression."""
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol

        self.weights = None
        self.bias = None

        self.coef_ = None
        self.diagnostics = None

    def fit(self, X: pd.DataFrame, y: pd.Series, eps: float=1e-15) -> 'Logit':
        """Fit the Logistic Regression model to the training data."""
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.Series) else y
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        list_loss = [np.inf]
        for _ in range(self.n_iter):
            log_odds = X @ self.weights + self.bias
            y_pred = self._sigmoid(log_odds)
            resid = y_pred - y

            # GRADIENT DESCENT
            dw = (1/n_samples) * X.T @ resid
            db = (1/n_samples) * np.sum(resid)
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

            # CROSS-ENTROPY LOSS (= MLE)
            y_pred_c = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y * np.log(y_pred_c) + (1 - y) * np.log(1 - y_pred_c))
            list_loss.append(loss)

            if abs(list_loss[-2] - list_loss[-1]) < self.tol:
                break

        log_odds = X @ self.weights + self.bias
        y_pred = self._sigmoid(log_odds)

        self.coeff_ = self.calc_coefficients(X, y)
        self.diagnostics = self.calc_diagnostics(X, y, y_pred)
        
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for the input data."""
        log_odds = X @ self.weights + self.bias
        return self._sigmoid(log_odds)

    def predict(self, X: pd.DataFrame, threshold: float=0.5) -> np.ndarray:
        """Predict class labels for the input data."""
        return (self.predict_proba(X) > threshold).astype(int)

    def calc_coefficients(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Calculate coefficients, standard errors, z-scores, p-values, and confidence intervals for the fitted model."""
        y_pred = self.predict_proba(X)
        
        # COEFFICIENTS
        coef = np.concatenate(([self.bias], self.weights))
        
        # STANDARD ERRORS
        W = np.diag(y_pred * (1 - y_pred))                  # Diagonal matrix of weights
        I = X.T @ W @ X                                     # Fisher Information Matrix
        se = np.sqrt(np.diag(np.linalg.inv(I)))             # Standard errors from the inverse of the Fisher Information Matrix

        # Z-SCORES, P-VALUES, AND 95% CONFIDENCE INTERVALS
        z_score = coef / se
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
        ci_95 = np.column_stack([coef - 1.96 * se, coef + 1.96 * se])

        self.coef_ = {
            'coef': coef,
            'se': se,
            'z_score': z_score,
            'p_value': p_value,
            'ci_95': ci_95
        }

        return self.coef_

    def calc_diagnostics(self, X: pd.DataFrame, y: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        n_samples, n_features = X.shape
        LogL0 = self._log_likelihood(y, np.full(n_samples, np.mean(y))) 
        LogL1 = self._log_likelihood(y, y_pred)

        self.diagnostics = {
            'log_likelihood': LogL1,
            'aic': 2 * (n_features + 1) - 2 * LogL1,
            'bic': (n_features + 1) * np.log(n_samples) - 2 * LogL1,
            'llr': -2 * (LogL1 - LogL0),
            'R2_mcfadden': 1 - LogL1 / LogL0,                               # McFadden's R² and between 0.2 and 0.4 is considered a good fit
            'R2_cox_snell': 1 - np.exp( (2/n_samples) * (LogL0 - LogL1) )   # Cox and Snell's R² and between 0.3 and 0.5 is considered a good fit
        }
        self.diagnostics['llr_p_value'] = 1 - stats.chi2.cdf(self.diagnostics['llr'], df=n_features)    # p-value for the likelihood ratio test
        self.diagnostics['R2_nagelkerke'] = self.diagnostics['R2_cox_snell'] / (1 - np.exp( (2/n_samples) * LogL0 ))    # Nagelkerke's R² and between 0.4 and 0.7 is considered a good fit
        return self.diagnostics

    @property   
    def coeff_(self) -> Optional[np.ndarray]:
        """Return the coefficients of the fitted model."""
        return [self.bias] + self.weights

    @property
    def aic(self) -> Optional[float]:
        """Calculate AIC for the fitted model."""
        return self.diagnostics['aic']

    @property
    def bic(self) -> Optional[float]:
        """Calculate BIC for the fitted model."""
        return self.diagnostics['bic']

    @property
    def llr(self) -> Optional[float]:
        """Calculate log-likelihood for the fitted model."""
        return self.diagnostics['llr']

    @property
    def llr_p_value(self) -> Optional[float]:
        """Calculate p-value for the likelihood ratio test."""
        return self.diagnostics['llr_p_value']

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid function to convert log-odds to probabilities."""
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def _log_likelihood(y: np.ndarray, p: np.ndarray, eps=1e-15) -> float:
        """Calculate log-likelihood for the given data and model parameters."""
        p = np.clip(p, eps, 1 - eps)  # Avoid log(0)
        return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))