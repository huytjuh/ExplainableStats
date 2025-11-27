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
        self.tree = None


