import pandas as pd 
import numpy as np 

from models.decision_tree import DecisionTree

class RandomForest:
    """A simple Random Forest classifier from scratch."""

    def __init__(self, n_estimators: int=10, max_depth: int=5, min_samples_split: int=2, min_samples_leaf: int=1, max_features='sqrt', random_state: int=42):
        """Initialize hyperparameters for the Random Forest."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_features = max_features
        self.list_features = None
        self.list_tree = None

    def fit(self, X: pd.DataFrame, y:pd.Series) -> 'RandomForest':
        """Train the Random Forest classifier."""
        np.random.seed(self.random_state)
        self.max_features = int(np.sqrt(X.shape[1]) if self.max_features == 'sqrt'
                            else np.log2(X.shape[1]) if self.max_features == 'log2'
                            else X.shape[1]*self.max_features if self.max_features <= 1 
                            else self.max_features)
     
        df_tree = pd.DataFrame(columns=['n_features', 'n_estimator', 'decision_tree', 'features', 'OOB_score'])
        for n_features in range(2, self.max_features + 1):
            result_tree = self.build_forest(X, y, n_features)
            temp = pd.DataFrame({'n_features': n_features, 
                                'n_estimator': list(range(1, len(result_tree[0])+1)), 
                                'decision_tree': result_tree[0],
                                'features': result_tree[1],
                                'OOB_score': result_tree[2]})
            df_tree = pd.concat([df_tree, temp], ignore_index=True)

        best_n_feature = self.find_best_n_features(df_tree)
        self.list_tree = df_tree.loc[df_tree['n_features'] == best_n_feature, 'decision_tree'].tolist()
        self.list_features = df_tree.loc[df_tree['n_features'] == best_n_feature, 'features'].tolist()
        return self
    
    def bootstrap_sample(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Generate a bootstrap sample from the dataset."""
        idx = np.random.choice(X.index, size=len(X), replace=True)
        X_bootstrap, y_bootstrap = X.loc[idx], y.loc[idx]
        return X_bootstrap, y_bootstrap
    
    def subset_features(self, X: pd.DataFrame, n_features: int) -> list:
        """Randomly select a subset of features."""
        col = np.random.choice(X.shape[1], size=n_features, replace=False)
        return X.columns[col].tolist()
    
    def build_forest(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> list:
        """Build the Random Forest by training multiple Decision Trees."""
        list_tree = []
        list_features = []
        list_oob_score = []
        for n in range(self.n_estimators):
            X_bootstrap, y_bootstrap = self.bootstrap_sample(X, y)
            feature_subset = self.subset_features(X_bootstrap, n_features)
            X_bootstrap = X_bootstrap[feature_subset]

            DT = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
            DT_fit = DT.fit(X_bootstrap, y_bootstrap)
            list_tree.append(DT_fit)
            list_features.append(feature_subset)

            X_oob, y_oob = X.drop(X_bootstrap.index), y.drop(y_bootstrap.index)
            X_oob = X_oob[feature_subset]
            OOB_score = self.oob_score(X_oob, y_oob, DT_fit)
            list_oob_score.append(OOB_score)

        return list_tree, list_features, list_oob_score

    def oob_score(self, X: pd.DataFrame, y: pd.Series, DT: 'DecisionTree') -> float:
        """Calculate the out-of-bag (OOB) score for the Random Forest."""
        if len(X) > 0:
            pred_oob = DT.predict(X)
            OOB_score = (y == pred_oob).mean()
        else:
            OOB_score = np.nan
        return OOB_score

    def find_best_n_features(self, df: pd.DataFrame) -> dict:
        """Find the best feature and threshold to split the data among a subset of features."""
        idx = df.groupby('n_features')['OOB_score'].mean().idxmax()
        return idx
    
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

    