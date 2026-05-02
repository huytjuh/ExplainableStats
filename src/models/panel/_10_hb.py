import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

from scipy import stats
import statsmodels.api as sm

class HBhyperparams:
    # GLOBAL ELASTICITY PRIOR
    mu0: float=-1.5
    sigma0: float=0.1

    # VARIANCE PRIOR
    a_sigma: float=2.0
    b_sigma: float=0.5
    a_mu: float=2.0
    b_mu: float=0.5
    a_eps: float=3.0
    b_eps: float=1.0

class HierarchicalBayes:
    """Hierarchical Bayes model for panel data from scratch"""

    def __init__(self, hyperparams: Optional[HBhyperparams]=None, max_iter: int=100, n_burn: int=500, random_state: int=42) -> None:
        """Initialize hyperparameters for the Hierarchical Bayes model."""
        self.hyperparams = hyperparams or HBhyperparams()
        self.max_iter = max_iter
        self.n_burn = n_burn
        self.random_state = random_state

        self.alpha = None
        self.beta = None
        self.gamma = None
        self.sigma2_eps = None

        self.entities = None
        self.entity_idx = None

    def fit(self, X: pd.DataFrame, y: pd.Series, entity_col: pd.Series, time_col: pd.Series) -> 'HierarchicalBayes':
        """Fit the Hierarchical Bayes model to the training data."""
        return self
    
    def _gibbs_sampler(self, X: np.ndarray, y: np.ndarray, entity_col: np.ndarray, time_col: np.ndarray) -> None:
        """Run the Gibbs sampler for the Hierarchical Bayes model."""
        pass

    def predict(self, X: pd.DataFrame, entity_col: pd.Series, time_col: pd.Series) -> np.ndarray:
        """Predict using the fitted Hierarchical Bayes model."""
        return np.zeros(len(X))
    