import pandas as pd
import numpy as np 

def gini_index(y) -> float:
    """Calculate the Gini index for a list of classes."""
    _, list_class_counts = np.unique(y, return_counts=True)
    p = list_class_counts / len(y)
    gini_index = 1 - np.sum(p ** 2)
    return gini_index

def weighted_gini_index(y, sample_weights) -> float:
    """Calculate the weighted Gini index for a list of classes."""
    list_class_counts = np.bincount(y, weights=sample_weights, minlength=len(np.unique(y)))
    p = list_class_counts / np.sum(sample_weights)
    gini_index = 1 - np.sum(p ** 2)
    return gini_index

def similarity_score(resid: list, lmbda: float=1.0) -> float:
    """Calculate similarity score for each tree."""
    resid = np.array(resid)
    return np.sum(resid)**2 / (len(resid) + lmbda)

def coverage_score(resid: list) -> float:
    """Calculate coverage score for each tree."""
    resid = np.array(resid)
    return len(resid)