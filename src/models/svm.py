import numpy as np
import pandas as pd

class SVM:
    """SVM classifier from scratch."""
    def __init__(self, kernel: str='linear'):
        self.kernel = kernel

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SVM':
        """Train the SVM classifier."""
        X = X.values
        y = y.values

        def objective_func(alpha: np.ndarray) -> float:
            K = self._kernel(X, X)
            return 0.5 * alpha.T @ K @ alpha - alpha.T @ y

        pass

    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        if self.kernel == 'linear':
            return x1 @ x2.T #np.dot(x1, x2.T)
        else:
            raise ValueError(f"Kernel {self.kernel} is not supported.")

    def decision_boundary():
        pass

