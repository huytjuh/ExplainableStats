import pandas as pd
import numpy as np 

from sklearn.model_selection import TimeSeriesSplit
from models.timeseries._10_rnn import RNN
from models.timeseries._12_lstm import LSTM

def create_sequence(data, n_steps=14):
    return np.array([data[i:i+n_steps] for i in range(len(data) - n_steps)])


if __name__ == "__main__":

    data = pd.read_csv('data/stocks.csv')
    X = data[['Date', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']].set_index('Date')
    y = data[['Date', 'TSLA']].set_index('Date')

    X_train, X_test, y_train, y_test = X[:-50], X[-50:], y[:-50], y[-50:]

    n_steps = 14
    X_train_seq = create_sequence(X_train, n_steps)
    y_train_seq = y_train[n_steps:]

    # RNN = RNN(learning_rate=0.01, epochs=1000, tol=1e-5)
    # RNN_fit = RNN.fit(X_train_seq, y_train_seq)

    LSTM = LSTM(learning_rate=0.01, epochs=1000, tol=1e-5)
    LSTM_fit = LSTM.fit(X_train_seq, y_train_seq)


    print(LSTM_fit)


