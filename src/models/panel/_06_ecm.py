import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

class ErrorCorrectionModel():
    """"Error Correction Model for panel data from scratch"""

    def __init__(self) -> None:
        """Initialize hyperparameters for the Error Correction Model."""
        self.coef_table: Optional[Dict[str, np.ndarray]] = None
        self.diagnostics: Optional[Dict[str, float]] = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series, entity_col: pd.Series, time_col: pd.Series) -> 'ErrorCorrectionModel':
        """Fit the Error Correction Model to the training data."""
        X = np.asarray(X)
        y = np.asarray(y)
        entity_col = np.asarray(entity_col)
        time_col = np.asarray(time_col)

        return self 

    def predict(self, X: pd.DataFrame, entity_col: pd.Series, time_col: pd.Series) -> np.ndarray:
        """Predict using the fitted Error Correction Model."""
        X = np.asarray(X)
        entity_col = np.asarray(entity_col)
        time_col = np.asarray(time_col)

        return np.zeros(len(X))

    def _inference(self, X: np.ndarray, alpha: float=0.05) -> None:
        """Calculate inference for the fitted model."""
        pass

    def _diagnostics(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calculate diagnostics for the fitted model."""
        pass