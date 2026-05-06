import pandas as pd
import numpy as np 
from typing import Optional, Dict

class IV:
    """Instrumental Variable Regression from scratch.
    
    Step 0a. Use Durbin Watson test to check autocorrelation of residuals.
    Step 0b. Use <insert test> to check endogeneity
    Step 1. Instrumentally regress X on Z
    Step 2. Endogenosly regress Y on X and Z

    Layman's term:
    If endogeneity in the model which results in correlated errors and biased coefficients (why?), therefore use IV
    """

    def __init__(self, alpha: float=0.05) -> None:
        """Initialize IV regression."""
        self.beta = None

        self.coef_table = Optional[Dict[str, np.ndarray]]
        self.diagnostics = Optional[Dict[str, float]]

    def fit(self, X: pd.DataFrame, y: pd.Series, Z: pd.DataFrame) -> 'IV':
        """Fit the IV regression model."""

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict values for the input data."""
        return

    def _inference(self, X: np.ndarray, alpha: float=0.05) -> None:
        """Calculate inference statistics for the fitted model."""


    def _diagnostics(self, X: np.ndarray, y: np.ndarray, resid: np.ndarray) -> None:
        """Calculate diagnostics for the fitted model."""