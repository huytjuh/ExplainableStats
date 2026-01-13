import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":

    data = pd.read_csv('data/stocks.csv')

    y = data[['Date', 'AAPL']]

    print(y)

