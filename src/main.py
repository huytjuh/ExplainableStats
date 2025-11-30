import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

from models.decision_tree import DecisionTree

if __name__ == "__main__":

    data = pd.read_csv('data/heart.csv')

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    DT = DecisionTree(max_depth=5)

    print(DT)
