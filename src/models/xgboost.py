import pandas as pd 
import numpy as np 

from models.decision_tree import DecisionTree

class XGBoost:

    def __init__(self, learning_rate: float=0.1, n_estimators: int=10, max_depth: int=5, min_samples_split: int=2, min_samples_leaf: int=1):
        """Initialize hyperparameters for XGBoost."""
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators 
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoost':
        """Train the XGBoost classifier."""
        return

    def build_forest(self, X: pd.DataFrame, y: pd.Series) -> list:
        """Build the XGBoost ensemble by training multiple Decision Trees."""
        return

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the input data."""
        return