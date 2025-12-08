import pandas as pd
import numpy as np 
from utils import gini_index

class DecisionTree:
    """A simple Decision Tree classifier from scratch."""

    def __init__(self, max_depth: int=5, min_samples_split: int=2, min_samples_leaf: int=1):
        """Initialize hyperparameters for the Decision Tree."""
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'DecisionTree':
        """Train the Decision Tree classifier."""
        self.tree = self.build_tree(X, y)
        return self

    def find_best_split_criterion(self, X: pd.DataFrame, y: pd.Series, eps: int=1e-4) -> dict:
        """Find the best feature and threshold to split the data."""
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

        # IF REGRESSION, CALCULATE MSE FOR EACH CANDIDATE SPLIT
        # TO BE IMPLEMENTED LATER
        
        # SELECT BEST SPLIT BASED ON MIN SAMPLES SPLIT AND GINI INDEX
        df['min_samples_split_flg'] = (df['N_left'] + df['N_right'] >= self.min_samples_split).astype(int)
        df['min_samples_leaf_flg'] = ((df['N_left'] >= self.min_samples_leaf) & (df['N_right'] >= self.min_samples_leaf)).astype(int)
        df['gini_gain'] = np.where(df['min_samples_split_flg'] + df['min_samples_leaf_flg'] == 2, gini_index(y) - df['gini_index'], np.nan)
        
        if pd.isna(df['gini_gain'].max()) or df['gini_gain'].max() < eps:
            return None
        best_split_criterion = df.loc[df['gini_gain'].idxmax()]
        return best_split_criterion[['feature', 'threshold', 'numeric_flg', 'gini_index']].to_dict()
    
    def create_leaf_node(self, y: pd.Series):
        """Create a leaf node by majority class for classification and average for regression."""
        pred = y.mode()[0]
        return pred

    def build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int=0) -> dict:
        """Recursively build the decision tree."""

        # STOPPING CRITERIA
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(y.unique()) == 1:
            return self.create_leaf_node(y)
        
        # FIND BEST SPLIT CRITERION
        split_criterion = self.find_best_split_criterion(X, y)
        if split_criterion is None:
            return self.create_leaf_node(y)
        
        # SPLIT DATA AND RECURSIVELY BUILD LEFT AND RIGHT SUBTREES
        feature, threshold, numeric_flg, gini_index = split_criterion.values()
        if numeric_flg == 1:
            idx_left, idx_right = X[feature] <= threshold, X[feature] > threshold
        else:
            idx_left, idx_right = X[feature] == threshold, X[feature] != threshold
        left_split = self.build_tree(X[idx_left], y[idx_left], depth + 1)
        right_split = self.build_tree(X[idx_right], y[idx_right], depth + 1)

        # CREATE AND RETURN TREE NODE
        tree = {'feature': feature, 
                'threshold': threshold, 
                'gini_index': gini_index,
                'left_split': left_split, 
                'right_split': right_split}
        return tree

    def predict(self, X: pd.DataFrame):
        """Predict class labels for the input data."""
        X['pred'] = X.apply(lambda x: self.predict_single(x), axis=1)
        return np.array(X['pred'])

    def predict_single(self, row: pd.Series):
        """Predict the class label for a single data point."""
        current_node = self.tree
        while not isinstance(current_node, (int, np.int64)):
            feature, threshold = current_node['feature'], current_node['threshold']
            if isinstance(threshold, (int, float)):
                current_node = current_node['left_split'] if row[feature] <= threshold else current_node['right_split']
            else:
                current_node = current_node['left_split'] if row[feature] == threshold else current_node['right_split']

        return current_node
    
    def missing_value_handler(self, X: pd.DataFrame):
        """Handle missing values in the dataset."""
        # TO BE IMPLEMENTED LATER
        pass

    def print_tree(self):
        """Print the structure of the decision tree."""
        if not self.tree:
            raise ValueError("The tree has not been fitted yet.")
        return self.tree