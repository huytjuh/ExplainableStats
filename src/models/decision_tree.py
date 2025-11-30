import pandas as pd
import numpy as np 

def gini_index(y) -> float:
    """Calculate the Gini index for a list of classes."""
    classes, counts = np.unique(y, return_counts=True)
    impurity = 1.0
    total = len(y)
    for count in counts:
        prob = count / total
        impurity -= prob ** 2
    return impurity

class DecisionTree:
    """A simple Decision Tree classifier from scratch."""

    def __init__(self, max_depth: int=5, min_samples_split: int=2):
        """Initialize hyperparameters for the Decision Tree."""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the Decision Tree classifier."""
        outcome = self.calculate_candidate(X, y)
        return outcome

    def calculate_candidate(self, X: pd.DataFrame, y: pd.Series):
        list_df = [pd.DataFrame({'feature': col, 'threshold': X[col].unique().tolist()}) for col in X.columns]
        df = pd.concat(list_df, ignore_index=True).sort_values(['feature', 'threshold']).reset_index(drop=True)

        rows_numeric = pd.to_numeric(df['threshold'], errors='coerce').notna()
        df.loc[rows_numeric, 'threshold'] = df.loc[rows_numeric, :].groupby('feature').rolling(window=2).mean()['threshold'].values
        out = df
        return out

    def split_criterion():
        return 

    def train():
        return

    def predict():
        return