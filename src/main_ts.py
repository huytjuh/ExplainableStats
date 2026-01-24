import pandas as pd
import numpy as np 

from sklearn.model_selection import TimeSeriesSplit
from models.supervised._14_rnn import RNN

if __name__ == "__main__":

    data = pd.read_csv('data/stocks.csv')
    X = data[['Date', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']].set_index('Date')
    y = data[['Date', 'TSLA']].set_index('Date')

    X_train, X_test, y_train, y_test = X[:-50], X[-50:], y[:-50], y[-50:]

    print(X_train)

    RNN = RNN(learning_rate=0.01, epochs=1000, tol=1e-5)
    RNN_fit = RNN.fit(X_train, y_train)

    print(RNN_fit)


