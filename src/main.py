import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

from models.decision_tree import DecisionTree
from models.random_forest import RandomForest

if __name__ == "__main__":

    data = pd.read_csv('data/heart.csv')

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # DT = DecisionTree(max_depth=5)
    # DT_fit = DT.fit(X_train, y_train)
    # DT_pred = DT_fit.predict(X_train)

    RF = RandomForest(n_estimators=5, max_depth=5)
    RF_fit = RF.fit(X_train, y_train)
    print(RF_fit)
    # df_eval = pd.DataFrame({'y_true': y_train, 'y_pred': DT_pred})
    # accuracy = (df_eval['y_true'] == df_eval['y_pred']).mean()
    # print(f"Accuracy of Decision Tree classifier: {accuracy:.4f}")
    # print("Predictions:", np.unique(DT_pred, return_counts=True))