import pandas as pd 
import numpy as np 

from models.decision_tree import DecisionTree

class RandomForest:
    """A simple Random Forest classifier from scratch."""

    def __init__(self, n_estimators: int=1, max_depth: int=5, min_samples_split: int=2, min_samples_leaf: int=1, max_features=.8, random_state: int=42):
        """Initialize hyperparameters for the Random Forest."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_features = max_features
        self.trees = []

    def fit(self, X: pd.DataFrame, y:pd.Series) -> 'RandomForest':
        """Train the Random Forest classifier."""
        np.random.seed(self.random_state)
        self.max_features = int(np.sqrt(X.shape[1]) if self.max_features == 'sqrt'
                            else np.log2(X.shape[1]) if self.max_features == 'log2'
                            else X.shape[1]*self.max_features if self.max_features <= 1 
                            else self.max_features)
        
        for n in range(self.n_estimators):
            for n_features in range(2, self.max_features):
                X_bootstrap, y_bootstrap = self.bootstrap_sample(X, y)
                feature_subset = self.subset_features(X_bootstrap, n_features)
                X_bootstrap = X_bootstrap[feature_subset]
                DT = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
                # print(n_features)
                # X_bootstrap, y_bootstrap = self.bootstrap_sample(X, y)
            






        return self
    
    def bootstrap_sample(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        idx = np.random.choice(X.index, size=len(X), replace=True)
        X_bootstrap, y_bootstrap = X.loc[idx].reset_index(drop=True), y.loc[idx].reset_index(drop=True)
        return X_bootstrap, y_bootstrap
    
    def subset_features(self, X: pd.DataFrame, n_features: int) -> list:
        col = np.random.choice(X.shape[1], size=n_features, replace=False)
        return X.columns[col].tolist()
    
    def oob_score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calculate the out-of-bag (OOB) score for the Random Forest."""
        pass

    def find_best_n_features(self, X: pd.DataFrame, y: pd.Series, feature_subset: list, eps: int=1e-4) -> dict:
        """Find the best feature and threshold to split the data among a subset of features."""
        pass
    
    def predict(self, X: pd.DataFrame, proba=False) -> np.ndarray:
        """Predict class labels for samples in X."""
        pass

    def predict_single(self, x: pd.Series) -> any:
        """Predict class label for a single sample x."""
        pass

    def handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        pass

    def feature_importances(self) -> pd.Series:
        """Calculate feature importances based on the trained Random Forest."""
        pass

    