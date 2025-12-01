import pandas as pd
import numpy as np 

def gini_index(y) -> float:
    """Calculate the Gini index for a list of classes."""
    _, list_class_counts = np.unique(y, return_counts=True)
    p = list_class_counts / len(y)
    gini_index = 1 - np.sum(p ** 2)
    return gini_index
