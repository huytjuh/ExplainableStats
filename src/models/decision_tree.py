import pandas as pd
import numpy as np 

def gini_index(y) -> float:
    """Calculate the Gini index for a list of classes."""
    list_class, list_class_counts = np.unique(y, return_counts=True)
    p = list_class_counts / len(y)
    gini_index = 1 - np.sum(p ** 2)
    return gini_index

class DecisionTree:
    """A simple Decision Tree classifier from scratch."""

    def __init__(self, max_depth: int=5, min_samples_split: int=5):
        """Initialize hyperparameters for the Decision Tree."""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the Decision Tree classifier."""
        outcome = self.find_best_split_criterion(X, y)
        return outcome

    def find_best_split_criterion(self, X: pd.DataFrame, y: pd.Series):
        # LIST OF ALL UNIQUE THRESHOLDS FOR EACH FEATURE
        list_df = [pd.DataFrame({'feature': col, 'threshold': X[col].unique().tolist()}) for col in X.columns]
        df = pd.concat(list_df, ignore_index=True).sort_values(['feature', 'threshold']).reset_index(drop=True)

        # DETERMINE IF THRESHOLDS ARE NUMERIC OR CATEGORICAL
        df['numeric_flg'] = pd.to_numeric(df['threshold'], errors='coerce').notna().astype(int)
        df.loc[df['numeric_flg'] == 1, 'threshold'] = df.loc[df['numeric_flg'] == 1, :].groupby('feature').rolling(window=2).mean()['threshold'].values
        df = df.dropna().reset_index(drop=True)

        # OBTAIN PREDICTIONS FOR LEFT AND RIGHT NODES
        for feature in df['feature'].unique():
            df.loc[(df['feature'] == feature) & (df['numeric_flg'] == 1), 'pred_left'] = df.loc[(df['feature'] == feature) & (df['numeric_flg'] == 1), 'threshold'].apply(lambda x: y[X[X[feature] <= x].index].tolist())
            df.loc[(df['feature'] == feature) & (df['numeric_flg'] == 1), 'pred_right'] = df.loc[(df['feature'] == feature) & (df['numeric_flg'] == 1), 'threshold'].apply(lambda x: y[X[X[feature] > x].index].tolist())
            
            df.loc[(df['feature'] == feature) & (df['numeric_flg'] == 0), 'pred_left'] = df.loc[(df['feature'] == feature) & (df['numeric_flg'] == 0), 'threshold'].apply(lambda x: y[X[X[feature] == x].index].tolist())
            df.loc[(df['feature'] == feature) & (df['numeric_flg'] == 0), 'pred_right'] = df.loc[(df['feature'] == feature) & (df['numeric_flg'] == 0), 'threshold'].apply(lambda x: y[X[X[feature] != x].index].tolist())
   
        # CALCULATE GINI INDEX FOR EACH CANDIDATE SPLIT
        df['N_left'], df['N_right'] = df['pred_left'].str.len(), df['pred_right'].str.len()
        df['gini_left'], df['gini_right'] = df['pred_left'].apply(lambda x: gini_index(x)), df['pred_right'].apply(lambda x: gini_index(x))
        df['gini_index'] = (df['N_left']*df['gini_left'] + df['N_right']*df['gini_right']) / (df['N_left'] + df['N_right'])
        
        # SELECT BEST SPLIT BASED ON MIN SAMPLES SPLIT AND GINI INDEX
        df['min_samples_split_flg'] = ((df['N_left'] >= self.min_samples_split) & (df['N_right'] >= self.min_samples_split)).astype(int)
        best_split_criterion = df.loc[df.loc[df['min_samples_split_flg'] == 1, 'gini_index'].argmin()]
        
        out = best_split_criterion[['feature', 'threshold', 'gini_index']].to_dict()
        return out

    def build_tree():
        return 

    def train():
        return

    def predict():
        return