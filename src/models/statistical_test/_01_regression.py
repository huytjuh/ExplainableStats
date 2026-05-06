import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

import scipy as stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

class RegressionModelTest():
    """Sequential tests for regression models selection.

    White Test (Breush-Pagan):              Heteroskedasticity              -> WLS
    Variance Inflation Error:               Multicollinearity               -> Omit features
    Durbin-Watson test (Breush-Godfrey):    Autocorrelation                 -> IV / 2SLS / Newey West
    Jarque-Bera test:                       Normality                       -> Robust/Bootstrapped SE
    Ramsey RESET test:                      Functional form                 -> Add power/interaction terms
    Pesaran test (Panel):                   Cross-sectional dependence      -> 
    """
    def __init__(self, alpha: float=0.05) -> None:
        """Initialize the PanelModelTest class."""
        self.alpha = alpha

        self.results: Optional[Dict[str, np.ndarray]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, ols: 'OLS') -> 'RegressionModelTest':
        """Fit the PanelModelTest to the training data."""

        return self
    
    def vif(self, X: pd.DaraFrame) -> Dict[str, float]:
        """Variance Inflation Factor (VIF) = 1 / (1 - R2).

        H0: Features are uncorrelated                                   (= Full features)
        H1: Features are correlated                                     (= Omit features)
        
        Layman's term:
        Does a given predictor bring unique information, or is it nearly a copy of others?
        """
        n_samples, n_features = X.shape
        vifs = {}
        for i in range(n_features):
            y_i = X.iloc[:, i]
            X_others = X.drop(X.columns[i], axis=1)
            OLS = sm.OLS(y_i, X_others)
            OLS_fit = OLS.fit()
            beta = OLS_fit.params 

            r2 = OLS_fit.rsquared
            vif = 1 / (1 - r2)
            vifs[X.columns[i]] = vif

        self.results['vif'] = {
            'vif': vifs,
            'conclusion': (
                "Severe multicollinearity (= OLS coefficients become unstable & hard to interpret) -> Omit features" if max(vifs.values()) > 5
                else
                "No severe multicollinearity -> Full features"
            )
        }

        return vifs
    
    def durbin_wu_hausman_test(self, resid: np.ndarray) -> Tuple[float, float]:
        """Durbin-Wu-Hausman test for endogeneity. 

        """
        return
    
    def durbin_watson_test(self, resid: np.ndarray) -> Tuple[float, float]:
        """Durbin-Watson test for autocorrelation. DW statistic = sum((e_t - e_{t-1})^2) / sum(e_t^2)

        H0: p = 0 | No first order autocorrelation                  (= OLS)
        H1: p ≠ 0 | AR(1) autocorrelation                           (= IV / 2SLS)

        Layman's term:
        Are today's errors independent of yesterday's, or do they tend to move together?
        """
        resid_diff = np.diff(resid)

        DW_stat = np.sum(resid_diff**2) / np.sum(resid**2)
        DW_pval = stats.chi2.sf(DW_stat, df=1)

        self.results['durbin_watson_test'] = {
            'stat': DW_stat, 
            'pval': DW_pval,
            'Autocorrelation (= Coefficients biased & dubbel counting over time) -> IV / 2SLS' if DW_pval < self.alpha 
            else 'conclusion': 'No autocorrelation -> IV / 2SLS'
        }

        return DW_stat, DW_pval

    def white_test(self, X: pd.DataFrame, resid: pd.Series) -> Tuple[float, float]:
        """Tests for heteroskedasticity by regressing squared residuals on the predictors, their squares, and cross-products.

        H0: Var(epsilon_i | X_i) = sigma2 | Constant variance over time     (= OLS)
        H1: Var(epsilon_i | X_i) ≠ sigma2 | Varying variance over time      (= WLS)

        Layman's term:
        Is the model's error variance constant, or does it get larger as variables change?
        """
        X = np.asarray(X)
        resid = np.asarray(resid)
        n_samples, n_features = X.shape

        resid2 = resid**2
        X_aug = np.hstack([X, X**2])
        OLS = sm.OLS(resid, X_aug)
        OLS_fit = OLS.fit()
        beta_aux = OLS_fit.params

        num = np.sum((resid2 - (X_aug @ beta_aux))**2)
        denom = np.sum((resid2 - np.mean(resid2))**2)
        r2 = 1 - num / denom
        
        white_stat = n_samples * r2
        white_pval = stats.chi2.sf(white_stat, df=n_features)

        self.results['white_test'] = {
            'white_stat': white_stat, 
            'white_pval': white_pval,
            'conclusion': 'Heteroskedasticity (= SE & pval wrong) -> WLS / Robust SE' if white_pval < self.alpha 
            else 'Homoskedasticity -> OLS'
        }
        
        return white_stat, white_pval

    def jarque_bera_test(self, resid: np.ndarray) -> Tuple[float, float]:
        """Jarque-Bera test for normality. JB = n/6 * (S^2 + 1/4 * (K-3)^2).
        
        H0: residuals are normal distributed                        (= Standard SE)
        H1: residuals are not normal distributed                    (= Robust SE)

        Layman's term:
        Do the model's mistakes follow a classic bell curve, or are they skewed / heavy-tailed?
        """
        n_samples = len(resid)

        skewness = np.mean((resid - np.mean(resid))**3) / np.std(resid)**3
        kurtosis = np.mean((resid - np.mean(resid))**4) / np.std(resid)**4

        JB_stat = (n_samples / 6) * (skewness**2 + 1/4 * (kurtosis - 3)**2)
        JB_pval = stats.chi2.sf(JB_stat, df=2)

        self.results['jarque_bera_test'] = {
            'stat': JB_stat, 
            'pval': JB_pval,
            'conclusion': 'Residuals non-normal (= t-test, pval, ci inaccurate) -> Robust SE' if JB_pval < self.alpha else
            'Normality -> OLS'
        }

        return JB_stat, JB_pval

    def ramsey_reset_test(self, y: pd.Series, resid: np.ndarray, fitted: Optional[np.ndarray] = None, powers: Tuple[int, ...] = (2, 3)) -> Tuple[float, float]:
        """Tests misspecification using powers of fitted values.

        Procedure:
        1. Baseline: y = X * Beta + epsilon
        2. Augmented: y = X * Beta + gamma_1 * y_hat^2 + gamma_2 * y_hat^3 + u
        3. Test H0: gamma_1 = gamma_2 = 0 via F-test.

        H0: Model correctly specified                               (= Full specification)
        H1: Model misspecified (omitted terms or variables)         (= Add power/interaction terms)

        Layman's term:
        Did we miss important curves, making a simple straight-line model insufficient?
        """
        y = np.asarray(y)
        X = np.asarray(X)
        X_const = sm.add_constant(X)

        # Baseline OLS
        base_model = sm.OLS(y, X_const)
        base_fit = base_model.fit()
        if fitted is None:
            fitted = base_fit.fittedvalues

        # Build augmented regressors: X + powers of fitted values
        extra_terms = [fitted**p for p in powers]
        X_aug = np.column_stack([X_const] + extra_terms)

        aug_model = sm.OLS(y, X_aug)
        aug_fit = aug_model.fit()

        ssr0 = np.sum(base_fit.resid**2)
        ssr1 = np.sum(aug_fit.resid**2)
        n = len(y)
        k0 = X_const.shape[1]       # baseline parameters
        q = len(powers)             # number of added terms
        k1 = k0 + q

        # F-stat: ((SSR0 - SSR1) / q) / (SSR1 / (n - k1))
        num = (ssr0 - ssr1) / q
        denom = ssr1 / (n - k1)
        reset_stat = num / denom
        reset_pval = stats.f.sf(reset_stat, q, n - k1)  # [web:22][web:57]

        self.results["ramsey_reset_test"] = {
            "stat": reset_stat,
            "pval": reset_pval,
            "conclusion": "Functional form problems (= under- or overpredictions) -> add nonlinear terms / interactions" if reset_pval < self.alpha
            else "No serious misspecification",
        }

        return reset_stat, reset_pval

    def pesaran_test(self, resid: np.ndarray) -> Tuple[float, float]:
        """Pesaran CD test for cross-sectional dependence.

        Input: resid (T x N matrix, where T=time, N=cross-sections)
        CD = sqrt(2 / (N * (N - 1))) * sum(rho_ij) * sqrt(T), where rho_ij are correlations.

        H0: Cov(epsilon_it, epsilon_jt) = 0 for all i != j                      (= OLS)
        H1: Cov(epsilon_it, epsilon_jt) ≠ 0                                     (= Panel OLS/FE/RE/LME)
    
        Layman's term:
        Do different units (e.g., countries) experience common shocks so their errors move together?
        """
        resid = np.asarray(resid)
        if resid.ndim != 2:
            raise ValueError("resid must be 2D array of shape (T, N).")

        T, N = resid.shape

        rhos = []
        for i in range(N):
            for j in range(i + 1, N):
                rij = np.corrcoef(resid[:, i], resid[:, j])[0, 1]
                rhos.append(rij)

        rhos = np.asarray(rhos)
        cd_stat = np.sqrt(2.0 / (N * (N - 1))) * rhos.sum() * np.sqrt(T)  # [web:52][web:61][web:65]
        cd_pval = 2.0 * (1.0 - stats.norm.cdf(np.abs(cd_stat)))

        self.results["pesaran_test"] = {
            "stat": cd_stat,
            "pval": cd_pval,
            "conclusion": "Cross-sectional dependence (= heterogeneity) -> panel OLS/FE/RE/LME" if cd_pval < self.alpha
            else "No cross-sectional dependence" 
        }

        return cd_stat, cd_pval
