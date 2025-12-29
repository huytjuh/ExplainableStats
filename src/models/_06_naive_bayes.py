import numpy as np
import pandas as pd
from typing import Optional

from scipy.stats import norm

class NaiveBayes:
    """Naive Bayes classifier from scratch."""

    def __init__(self, method: str='gaussian', alpha: Optional[float] = 1.0):
        """Initialize hyperparameters for Naive Bayes."""
        self.method = method
        self.alpha = alpha

        self.class_priors = {}
        self.conditional_probs = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBayes':
        """Fit the Naive Bayes model to the training data."""
        self._conditional_probabilities(X, y)
        self._prior_probabilities(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for the input data."""
        if self.method == 'gaussian':
            return self._posterior_probability(X)
        else:
            raise ValueError("Unsupported method. Use 'gaussian'.")

    def _prior_probabilities(self, y: np.ndarray) -> dict:
        """Calculate prior probabilities for each class."""
        list_class, list_class_counts = np.unique(y, return_counts=True)
        priors = list_class_counts / len(y)
        self.class_priors = dict(zip(list_class, priors))
        return self.class_priors

    def _conditional_probabilities(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Calculate conditional probabilities P(X|Y) for each feature given class."""
        list_class = np.unique(y)
        for _class in list_class:
            X_class = X[y == _class]
            if self.method == 'gaussian':
                X_mean = X_class.mean(axis=0)
                X_std = X_class.std(axis=0)
                self.conditional_probs[_class] = (X_mean, X_std)

        return self.conditional_probs

    def _posterior_probability(self, X: np.ndarray) -> np.ndarray:
        """Calculate posterior probability P(Y|X) for a given class."""
        df_posterior = pd.DataFrame()
        for _class in list(self.class_priors):
            mean, std = self.conditional_probs[_class]
            logL = np.log(norm.pdf(X, mean, std))

            prior = self.class_priors[_class]
            posterior = np.log(prior) + np.sum(logL, axis=1)
            df_posterior[_class] = posterior

        df_posterior['pred'] = df_posterior.idxmax(axis=1)
        return df_posterior['pred'].values
