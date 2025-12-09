import pandas as pd 
import numpy as np 

from models.decision_tree import DecisionTree

def amount_of_say(x, eps=1e-4):
    """Compute the AdaBoost 'amount of say' (alpha) from weighted errors"""
    num = 1 - np.sum(x) + eps
    denom = np.sum(x) + eps
    return 0.5*np.log(num/denom)

class AdaBoost:

    def __init__(self, n_estimators: int=10):
        """Initialize hyperparameters for AdaBoost."""
        self.n_estimators = n_estimators 
        self.list_sample_weights = []
        self.list_trees = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AdaBoost':
        """Train the AdaBoost classifier."""
        self.list_trees = self.build_forest(X, y)
        return self

    def build_forest(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Build the AdaBoost ensemble by training multiple Decision Trees."""
        if not self.list_sample_weights:
            self.list_sample_weights.append(np.ones(len(y)) / len(y))

        sample_weights = None
        DT = DecisionTree(max_depth=1)
        list_tree = []
        for n in range(self.n_estimators):
            DT_fit = DT.fit(X, y, sample_weights=sample_weights)
            DT_pred = DT_fit.predict(X)
            DT_resid = np.abs(y - DT_pred)

            alpha = amount_of_say(self.list_sample_weights[-1]*DT_resid)
            sample_weights = self.calculate_sample_weight(DT_resid, alpha)

            tree = {'DecisionTree': DT_fit, 'alpha': alpha}
            list_tree.append(tree)
        
        return list_tree

    def calculate_sample_weight(self, resid: np.ndarray, alpha: float) -> np.ndarray:
        """Calculate sample weights for AdaBoost."""
        list_sign = np.array([-1 if x==0 else 1 for x in resid])
        
        sample_weights = self.list_sample_weights[-1] * np.exp(alpha*list_sign)
        sample_weights_scaled = sample_weights / np.sum(sample_weights)

        self.list_sample_weights.append(sample_weights_scaled)
        return self.list_sample_weights[-1]

    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the input data."""
        df_pred = pd.DataFrame()
        for n in range(self.n_estimators):
            tree = self.list_trees[n]['DecisionTree']
            alpha = self.list_trees[n]['alpha']
            df_pred[f'Tree_{n}'] = alpha * tree.predict(X)

        AB_pred = df_pred.sum(axis=1) / np.sum([tree['alpha'] for tree in self.list_trees])
        AB_pred = np.where(AB_pred > 0.5, 1, 0)
        return AB_pred