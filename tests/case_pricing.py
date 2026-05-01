import pandas as pd
import numpy as np
from typing import List

from models.panel._01_ols_pooled import PooledOLS
from models.panel._02_re import RandomEffects
from models.panel._03_fe import FixedEffects
from models.panel._04_fd import FirstDifference
from models.panel._05_lme import LinearMixedEffects
from models.panel._06_ecm import ErrorCorrectionModel

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
    # RE = RandomEffects()
    # RE_fit = RE.fit(X, y, df['customer_id'])
    # RE_res = RE_fit.coef_table
    # print(RE_res)

    # FIXED EFFECTS
    # FE = FixedEffects()
    # FE_fit = FE.fit(X, y, df['customer_id'])
    # FE_res = FE_fit.coef_table
    # print(FE_res)

    # FIRST-DIFFERENCE 
    # FD = FirstDifference()
    # FD_fit = FD.fit(X, y, df['customer_id'], df['date'])
    # FD_res = FD_fit.coef_table
    # print(FD_res)

    # LINEAR MIXED EFFECTS 
    # LME = LinearMixedEffects()
    # LME_fit = LME.fit(X, y, df['customer_id'])
    # LME_res = LME_fit.coef_table
    # print(LME_res)

    # ECM
    ECM = ErrorCorrectionModel()
    ECM_fit = ECM.fit(X, y, df['customer_id'], df['date'])
    ECM_res = ECM_fit.coef_table
    print(ECM_res)


    # HIERARCHICAL BAYES

    # GPBOOST

    # DML





    # PRICING ELASTICITY

    # ACQUISITION ELASTICITY

    # ATTRITION ELASTICITY 
