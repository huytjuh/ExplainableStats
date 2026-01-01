import pandas as pd
import numpy as np

from models.supervised._01_decision_tree import DecisionTree

class XGBoost:
    """XGBoost classifier from scratch."""
    def __init__(self, learning_rate: float=0.1, n_estimators: int=10, max_depth: int=5, min_samples_split: int=5, min_samples_leaf: int=2, gamma: float=0.0, lmbda: float=1.0, min_child_weight: float=1.0):
        """Initialize hyperparameters for XGBoost."""
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators 
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.gamma = gamma
        self.lmbda = lmbda
        self.min_child_weight = min_child_weight

        self.list_trees: list[DecisionTree] = []
        self.base_score: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoost':
        """Train the XGBoost classifier."""
        self.list_trees = self.build_forest(X, y)
        return self

    def build_forest(self, X: pd.DataFrame, y: pd.Series) -> list:
        """Build the XGBoost ensemble by training multiple Decision Trees."""
        self.base_score = 0.5

        y_pred = np.full(len(y), self.base_score)
        resid = y - y_pred
        list_tree = []
        for n in range(self.n_estimators):
            DT = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, gamma=self.gamma, lmbda=self.lmbda, min_child_weight=self.min_child_weight)
            DT_fit = DT.fit(X, resid, split_criterion='gain')
            DT_pred = DT_fit.predict(X)

            y_pred = y_pred + self.learning_rate * DT_pred
            resid = y - y_pred
            list_tree.append(DT_fit)
            
        return list_tree

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the input data."""
        XGB_pred = np.full(len(X), self.base_score)
        for DT in self.list_trees:
            DT_pred = DT.predict(X)
            XGB_pred += self.learning_rate * DT_pred

        XGB_pred = 1 / (1 + np.exp(-XGB_pred))
        XGB_pred = np.where(XGB_pred > 0.5, 1, 0)
        return XGB_pred