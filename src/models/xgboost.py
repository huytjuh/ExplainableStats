import pandas as pd
import numpy as np

from models.decision_tree import DecisionTree

def similarity_score(resid, proba):
    """Calculate similarity score for each tree."""
    SSR = np.sum(resid)**2
    return None

class XGBoost:

    def __init__(self, learning_rate: float=0.1, n_estimators: int=10, max_depth: int=5, min_samples_split: int=2, min_samples_leaf: int=1):
        """Initialize hyperparameters for XGBoost."""
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators 
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.list_proba = [0.5]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoost':
        """Train the XGBoost classifier."""
        self.build_forest(X, y)
        return self

    def build_forest(self, X: pd.DataFrame, y: pd.Series) -> list:
        """Build the XGBoost ensemble by training multiple Decision Trees."""
        proba = self.list_proba[0]
        resid = proba - y
        for n in range(self.n_estimators)[:2]:
            DT = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
            DT_fit = DT.fit(X, y)
            DT_pred = DT_fit.predict(X)
            
            proba = self.list_proba[n]
            resid = proba - DT_pred

            print(resid)


        return

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the input data."""
        return