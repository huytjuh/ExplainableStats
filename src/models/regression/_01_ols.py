import pandas as pd
import numpy as np 
from typing import Optional

from utils import gini_index, weighted_gini_index, similarity_score, coverage_score

class OLS:
    def __init__(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test