import pandas as pd
import numpy as np 
from typing import Optional, Dict

from scipy import stats

class Multinomial():
    """Multinomial Logistic Regression Classifier from scratch."""
    
    def __init__(self, lr: float=0.01, n_iter: int=100, tol: float=1e-4):
        """Initialize hyperparameters for the Multinomial Logistic Regression."""
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol

        self.classes_: Optional[np.ndarray] = None
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None

        self.coef_: Optional[Dict[str, np.ndarray]] = None
        self.diagnostics: Optional[Dict[str, float]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, eps: float=1e-15) -> 'Multinomial':
        """Fit the Multinomial Logistic Regression model to the training data.""" 
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.Series) else y

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        y_encoded = np.eye(n_classes)[y.astype(int)]  # One-hot encode the target variable

        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)
        
        list_loss = [np.inf]
        for _ in range(self.n_iter):
            log_odds = X @ self.W + self.b
            y_pred = self._softmax(log_odds)
            resid = y_pred - y_encoded

            # GRADIENT DESCENT
            dW = (1/n_samples) * X.T @ resid
            db = (1/n_samples) * np.sum(resid, axis=0)
            self.W = self.W - self.lr * dW
            self.b = self.b - self.lr * db

            # CROSS-ENTROPY LOSS (= MLE)
            y_pred_c = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(np.sum(y_encoded * np.log(y_pred_c), axis=1))
            list_loss.append(loss)

            if abs(list_loss[-2] - list_loss[-1]) < self.tol:
                break

        log_odds = X @ self.W + self.b
        y_pred = self._softmax(log_odds)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for the input data."""
        log_odds = X @ self.W + self.b
        return self._softmax(log_odds)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the input data."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def calc_coefficients(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Calculate coefficients, standard errors, z-scores, p-values, and confidence intervals for the fitted model."""
        y_pred = self.predict_proba(X)

        self.coef_ = {}
        for k in range(len(self.classes_)):
            coef = np.concatenate(([self.b[k]], self.W[:, k]))
            
            # STANDARD ERRORS
            W = np.diag(y_pred[:, k] * (1 - y_pred[:, k]))                  # Diagonal matrix of weights
            I = X.T @ W @ X                                                 # Fisher Information Matrix
            se = np.sqrt(np.diag(np.linalg.inv(I)))                         # Standard errors from the inverse of the Fisher Information Matrix

            # Z-SCORES, P-VALUES, AND 95% CONFIDENCE INTERVALS
            z_score = coef / se
            p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))
            ci_95 = np.column_stack([coef - 1.96 * se, coef + 1.96 * se])

            self.coef_[k] = {
                'coef': coef,
                'se': se,
                'z_score': z_score,
                'p_value': p_value,
                'ci_95': ci_95
            }

        return self.coef_
    
    def calc_diagnostics(self, X: pd.DataFrame, y: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        y_encoded = np.eye(n_classes)[y.astype(int)]  # One-hot encode the target variable

        LogL0 = self._log_likelihood(y_encoded, np.full(n_samples, np.mean(y_encoded, axis=0))) 
        LogL1 = self._log_likelihood(y_encoded, y_pred)

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

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """Softmax function to convert log-odds to probabilities."""
        return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True) 
