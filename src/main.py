import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from models.supervised._01_decision_tree import DecisionTree
from models.supervised._02_random_forest import RandomForest
from models.supervised._03_adaboost import AdaBoost
from models.supervised._04_xgboost import XGBoost
from models.supervised._05_svm import SVM
from models.supervised._06_naive_bayes import NaiveBayes
from models.supervised._07_logistic_regression import LogisticRegression
from models.supervised._11_perceptron import Perceptron

if __name__ == "__main__":

    data = pd.read_csv('data/heart.csv')

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_encoded, X_test_encoded = pd.get_dummies(X_train, drop_first=True).astype(float), pd.get_dummies(X_test, drop_first=True).astype(float)

    scaler = MinMaxScaler()
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train_encoded), scaler.fit_transform(X_test_encoded)

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

    # XGB = XGBoost(n_estimators=10, learning_rate=0.5)
    # XGB_fit = XGB.fit(X_train, y_train)
    # XGB_pred = XGB_fit.predict(X_test)

    # df_eval = pd.DataFrame({'y_true': y_test.values, 'y_pred': XGB_pred})
    # accuracy = (df_eval['y_true'] == df_eval['y_pred']).mean()
    # print(f"Accuracy of XGBoost classifier: {accuracy:.4f}")
    # print("Predictions:", np.unique(XGB_pred, return_counts=True))

    # SVM = SVM(kernel='rbf', C=1.0, learning_rate=0.1, n_iter=10000)
    # SVM_fit = SVM.fit(X_train_scaled, y_train)

    # NB = NaiveBayes(method='gaussian')
    # NB_fit = NB.fit(X_train_scaled, y_train)
    # NB_pred = NB_fit.predict(X_test_scaled)
    # df_eval = pd.DataFrame({'y_true': y_test.values, 'y_pred': NB_pred})
    # accuracy = (df_eval['y_true'] == df_eval['y_pred']).mean()
    # print(f"Accuracy of Naive Bayes classifier: {accuracy:.4f}")
    # print("Predictions:", np.unique(NB_pred, return_counts=True))

    # Logit = LogisticRegression(learning_rate=0.01, n_iter=1000)
    # Logit_fit = Logit.fit(X_train_scaled, y_train)
    # Logit_pred = Logit_fit.predict(X_test_scaled)
    # df_eval = pd.DataFrame({'y_true': y_test.values, 'y_pred': Logit_pred})
    # accuracy = (df_eval['y_true'] == df_eval['y_pred']).mean()
    # print(f"Accuracy of Logistic Regression classifier: {accuracy:.4f}")
    # print("Predictions:", np.unique(Logit_pred, return_counts=True))

    # PPN = Perceptron(learning_rate=0.01, n_iter=1000)
    # PPN_fit = PPN.fit(X_train_scaled, y_train)
    # PPN_pred = PPN_fit.predict(X_test_scaled)
    # df_eval = pd.DataFrame({'y_true': y_test.values, 'y_pred': PPN_pred})
    # accuracy = (df_eval['y_true'] == df_eval['y_pred']).mean()
    # print(f"Accuracy of Perceptron classifier: {accuracy:.4f}")
    # print("Predictions:", np.unique(PPN_pred, return_counts=True))