import pandas as pd
import numpy as np
from typing import List

from models.panel._01_ols_pooled import PooledOLS
from models.panel._02_re import RandomEffects

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

    # POOLED OLS
    # OLS = PooledOLS()
    # OLS_fit = OLS.fit(X, y)
    # OLS_res = OLS_fit.coef_table 
    # print(OLS_res)

    # RANDOM EFFECTS
    RE = RandomEffects()
    RE_fit = RE.fit(X, y, df['customer_id'])
    RE_res = RE_fit.coef_table
    print(RE_res)

    # FIXED EFFECTS

    # PANELOLS: TWO-WAY ENTITY FE + TIME FE

    # FIRST-DIFFERENCE 

    # ECM

    # WITHIN ESTIMATOR TO SUBTRACT TIME-INVARIANT UNOBSERVED HETEROGENEITY (= CUSTOMER BIAS: WEALTH, RISK APPETITE, ETC.)
    # def within_transform(df: pd.DataFrame, entity: str, features: List[str], target: str):
    #     cols = features + [target]
    #     entity_means = df.groupby(entity)[cols].transform('mean')
    #     for col in cols:
    #         df[f'{col}_dm'] = df[col] - entity_means[col]
    #     return df

    # list_features = ['d_own_rate_pct', 'd_ecb_rate_pct', 'd_inflation_pct', 'd_rel_rate_pct']
    # df = within_transform(df, 'customer_id', list_features, 'd_log_balance_eur')
    # X = df.loc[:, [f'{col}_dm' for col in list_features]]
    # y = df['d_log_balance_eur_dm']


    # PRICING ELASTICITY

    # ACQUISITION ELASTICITY

    # ATTRITION ELASTICITY 
