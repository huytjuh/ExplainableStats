import pandas as pd 
import numpy as np 
from models.decision_tree import DecisionTree

def amount_of_say(x, eps=1e-4):
    num = 1 - np.sum(x)
    denom = np.sum(x)
    return 0.5*np.log((num + eps)/(denom + eps))

class AdaBoost:

    def __init__(self, n_estimators: int=10, max_depth: int=5):
        """Initialize hyperparameters for AdaBoost."""
        self.n_estimators = n_estimators 
        self.max_depth = max_depth 
        self.list_sample_weights = []
        self.list_trees = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AdaBoost':
        """Train the AdaBoost classifier."""
        self.list_trees = self.build_forest(X, y)
        return self

    def build_forest(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Build the AdaBoost ensemble by training multiple Decision Trees."""
        for n in range(self.n_estimators):
            DT = DecisionTree(max_depth=1)
            DT_fit = DT.fit(X, y)
            DT_pred = DT_fit.predict(X)
            DT_resid = np.abs(y - DT_pred)

            sample_weights = self.calculate_sample_weight(DT_resid)
        
        print(DT_fit.print_tree())
        return None

    def calculate_sample_weight(self, resid: pd.Series) -> np.ndarray:
        """Calculate sample weights for AdaBoost."""
        if not self.list_sample_weights:
            sample_weights = np.ones(len(resid)) / len(resid)
        else: 
            list_sign = [-1 if x==0 else 1 for x in resid]
            say = amount_of_say(self.list_sample_weights[-1]*resid)
            sample_weights = self.list_sample_weights[-1] * np.exp(list_sign*say)
            # print(self.list_sample_weights[-1] * resid)
            print(list_sign*say)
            # print(list_sign * resid)
            # print(amount_of_say(list_sign*resid))
            sample_weights = self.list_sample_weights[-1]
            # sample_weights = self.list_sample_weights[-1] * amount_of_say(list_sign*resid)

        # print(sample_weights)
        self.list_sample_weights.append(sample_weights)
        
        return self.list_sample_weights[-1]
    
        # amount_of_say = -1 if resid is 0 else 1
        # new_weights = self.list_sample_weights[-1] * np.exp(amount_of_say)
        # return None