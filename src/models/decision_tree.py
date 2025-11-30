import numpy as np
from typing import Optional, Dict, Any


def gini_index(y) -> float:
   """Calculate the Gini index for a list/array of classes.

   Parameters
   ----------
   y : array-like
      Class labels for a node.

   Returns
   -------
   float
      Gini impurity value.
   """
   classes, counts = np.unique(y, return_counts=True)
   impurity = 1.0
   total = len(y)
   for count in counts:
      prob = count / total
      impurity -= prob ** 2
   return impurity


class DecisionTree:
   """A simple Decision Tree classifier from scratch using Gini impurity.

   Supports numeric features and binary splits. The tree is represented
   as nested dictionaries with keys: 'feature', 'threshold', 'left', 'right',
   and 'value' for leaf nodes.
   """

   def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
      self.max_depth = max_depth
      self.min_samples_split = min_samples_split
      self.tree: Optional[Dict[str, Any]] = None

   def _split(self, X: np.ndarray, y: np.ndarray, feature: int, threshold: float):
      left_mask = X[:, feature] <= threshold
      right_mask = ~left_mask
      return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

   def calculate_candidate(self, X: np.ndarray, y: np.ndarray) -> Optional[Dict[str, Any]]:
      """Find the best split across all features.

      Returns a dict with keys: feature, threshold, impurity, left_idx, right_idx
      or None if no valid split found.
      """
      n_samples, n_features = X.shape
      if n_samples < 2:
         return None

      best = None
      best_impurity = float('inf')

      base_impurity = gini_index(y)

      for feature in range(n_features):
         values = np.unique(X[:, feature])
         if values.size == 1:
            continue

         # candidate thresholds: midpoints between sorted unique values
         thresholds = (values[:-1] + values[1:]) / 2.0
         for thr in thresholds:
            X_left, y_left, X_right, y_right = self._split(X, y, feature, thr)
            if len(y_left) < 1 or len(y_right) < 1:
               continue

            # weighted impurity after split
            n_left = len(y_left)
            n_right = len(y_right)
            impurity = (n_left * gini_index(y_left) + n_right * gini_index(y_right)) / (n_left + n_right)

            if impurity < best_impurity:
               best_impurity = impurity
               best = {
                  'feature': feature,
                  'threshold': float(thr),
                  'impurity': float(impurity),
               }

      # return None if no improvement possible
      if best is None or best_impurity >= base_impurity:
         return None
      return best

   def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Dict[str, Any]:
      # create a leaf node if stopping criteria met
      num_samples = len(y)
      num_labels = len(np.unique(y))

      if (depth >= self.max_depth) or (num_samples < self.min_samples_split) or (num_labels == 1):
         # leaf
         values, counts = np.unique(y, return_counts=True)
         return {'value': values[np.argmax(counts)].item()}

      candidate = self.calculate_candidate(X, y)
      if candidate is None:
         values, counts = np.unique(y, return_counts=True)
         return {'value': values[np.argmax(counts)].item()}

      feature = candidate['feature']
      threshold = candidate['threshold']
      X_left, y_left, X_right, y_right = self._split(X, y, feature, threshold)

      left_node = self._build_tree(X_left, y_left, depth + 1)
      right_node = self._build_tree(X_right, y_right, depth + 1)

      return {
         'feature': feature,
         'threshold': threshold,
         'left': left_node,
         'right': right_node
      }

   def fit(self, X: np.ndarray, y: np.ndarray):
      """Fit the decision tree to X, y.

      Accepts X as 2D array-like and y as 1D array-like.
      """
      X_arr = np.asarray(X)
      if X_arr.ndim == 1:
         X_arr = X_arr.reshape(-1, 1)
      y_arr = np.asarray(y)
      self.tree = self._build_tree(X_arr, y_arr, depth=0)
      return self

   # keep train as an alias for fit to match provided skeleton
   def train(self, X: np.ndarray, y: np.ndarray):
      return self.fit(X, y)

   def _predict_one(self, x: np.ndarray, node: Dict[str, Any]):
      if 'value' in node:
         return node['value']
      feature = node['feature']
      threshold = node['threshold']
      if x[feature] <= threshold:
         return self._predict_one(x, node['left'])
      else:
         return self._predict_one(x, node['right'])

   def predict(self, X: np.ndarray):
      """Predict class labels for samples in X."""
      if self.tree is None:
         raise ValueError('The tree has not been trained yet. Call fit or train first.')
      X_arr = np.asarray(X)
      if X_arr.ndim == 1:
         X_arr = X_arr.reshape(-1, 1)
      preds = [self._predict_one(x, self.tree) for x in X_arr]
      return np.array(preds)

   # convenience
   def split_criterion(self):
      """Expose split criterion name (for API completeness)."""
      return 'gini'
