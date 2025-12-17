import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split

from models.decision_tree import DecisionTree
from models.random_forest import RandomForest
from models.adaboost import AdaBoost
from models.xgboost import XGBoost

if __name__ == "__main__":

    data = pd.read_csv('data/heart.csv')

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # DT = DecisionTree(max_depth=5)
    # DT_fit = DT.fit(X_train, y_train)
    # DT_pred = DT_fit.predict(X_test)

    # df_eval = pd.DataFrame({'y_true': y_test.values, 'y_pred': DT_pred})
    # accuracy = (df_eval['y_true'] == df_eval['y_pred']).mean()
    # print(f"Accuracy of Decision Tree classifier: {accuracy:.4f}")
    # print("Predictions:", np.unique(DT_pred, return_counts=True))

    # RF = RandomForest(n_estimators=3, max_depth=5)
    # RF_fit = RF.fit(X_train, y_train)
    # RF_pred = RF_fit.predict(X_test)

    # df_eval = pd.DataFrame({'y_true': y_test.values, 'y_pred': RF_pred})
    # accuracy = (df_eval['y_true'] == df_eval['y_pred']).mean()
    # print(f"Accuracy of Random Forest classifier: {accuracy:.4f}")
    # print("Predictions:", np.unique(RF_pred, return_counts=True))

    # AB = AdaBoost(n_estimators=3)
    # AB_fit = AB.fit(X_train, y_train)
    # AB_pred = AB_fit.predict(X_test)

    # df_eval = pd.DataFrame({'y_true': y_test.values, 'y_pred': AB_pred})
    # accuracy = (df_eval['y_true'] == df_eval['y_pred']).mean()
    # print(f"Accuracy of AdaBoost classifier: {accuracy:.4f}")
    # print("Predictions:", np.unique(AB_pred, return_counts=True))

    XGB = XGBoost(n_estimators=10, learning_rate=0.5)
    XGB_fit = XGB.fit(X_train, y_train)
    XGB_pred = XGB_fit.predict(X_test)

    df_eval = pd.DataFrame({'y_true': y_test.values, 'y_pred': XGB_pred})
    accuracy = (df_eval['y_true'] == df_eval['y_pred']).mean()
    print(f"Accuracy of XGBoost classifier: {accuracy:.4f}")
    print("Predictions:", np.unique(XGB_pred, return_counts=True))