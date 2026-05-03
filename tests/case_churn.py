import pandas as pd
import numpy as np
from typing import List

if __name__ == "__main__":
    data = pd.read_csv(r'data/transactions.csv')

    df = data.copy()
    df['log_balance_eur'] = np.log1p(df['balance_eur'])  # Add 1 to avoid log(0)
    df['d_log_balance_eur'] = df['log_balance_eur'] - df['log_balance_eur'].shift(1)

    for col in ['own_rate_pct', 'ecb_rate_pct', 'inflation_pct', 'rel_rate_pct']:
        df[f'd_{col}'] = df[col] - df[col].shift(1)
    df = df.dropna().reset_index(drop=True)

    X = df.loc[:, ['d_own_rate_pct', 'd_ecb_rate_pct', 'd_inflation_pct', 'd_rel_rate_pct']]
    y = df.loc[:, 'd_log_balance_eur']
