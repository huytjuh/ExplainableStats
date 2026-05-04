import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

from scipy import stats

from models.panel._01_ols_pooled import PooledOLS
from models.panel._02_re import RandomEffects
from models.panel._03_fe import FixedEffects
from models.panel._04_fd import FirstDifference
from models.panel._05_lme import LinearMixedEffects
from models.panel._06_ecm import ErrorCorrectionModel

class PanelModelTest:
    """Sequential Panel Tests for Panel Data.

    F-test:                 PooledOLS vs FE
    Breusch-Pagan test:     PooledOLS vs RE
    Hausman test:           RE vs FE
    Wooldridge test:        FE vs FD
    Likelihood Ratio Test:  RE vs MLE
    Engler test:            PooledOLS vs ECM
    BIC over k:             Heterogeneity -> hidden segments -> Finite Mixture Models
    Data sparsity:          MLE vs Hierarchical Bayes
    Nonlinearity:           MLE vs Gaussian Process Boost
    """

    def __init__(self, alpha: float=0.05) -> None:
        """Initialize the PanelModelTest class."""
        self.alpha = alpha

        self.results: Optional[Dict[str, np.ndarray]] = None

    def fit(self, pooled: 'PooledOLS', re: 'RandomEffects', fe: 'FixedEffects', fd: 'FirstDifference', lme: 'LinearMixedEffects', ecm: 'ErrorCorrectionModel', entity: np.ndarray, time: np.ndarray) -> 'PanelModelTest':
        """Fit the PanelModelTest to the training data."""

        return self
    
    def f_test_entity_effects(self, pooled: 'PooledOLS', fe: 'FixedEffects') -> Tuple[float, float]:
        """Test the null hypothesis that the entity effects are zero.
        
        H0: alpha_i = 0 | Entity effects exists             (= PooledOLS)
        H1: alpha_i ≠ 0 | Entity effects do not exists      (= FixedEffects)

        Layman's terms:
        Check if the residuals of FixedEffects are significantly different from the residuals of PooledOLS
        """
        n_samples = pooled.n_samples
        n_entities = pooled.n_entities
        n_features = pooled.n_features

        ssr0 = np.sum(pooled.resid_**2)
        ssr1 = np.sum(fe.resid_**2)

        dof_num = n_entities - 1
        dof_denom = n_samples - n_entities - n_features

        num = (ssr0 - ssr1) / dof_num
        denom = ssr1 / dof_denom

        f_stat = num / denom
        f_pval = stats.f.sf(f_stat, dfn=dof_num, dfd=dof_denom)

        self.results['f_test_entity_effects'] = {
            'stat': f_stat, 
            'pval': f_pval,
            'conclusion': 'Entity effects exists -> FixedEffects' if f_pval < self.alpha else 'Entity effects do not exists -> PooledOLS'
        }
    
        return f_stat, f_pval

    def breusch_pagan_test(self, pooled: 'PooledOLS', y: pd.Series, entity: pd.Series) -> Tuple[float, float]:
        """Test the null hypothesis that the random effects are zero.
        
        H0: sigma2 = 0 | Random effects exists                 (= PooledOLS)
        H1: sigma2 ≠ 0 | Random effects do not exists          (= RandomEffects)

        Layman's term:
        Check if variance across entities is significantly different from the variance of PooledOLS
        """
        n_samples = pooled.n_samples
        n_entities = pooled.n_entities
        n_features = pooled.n_features
        avg_sample_entity = n_samples / n_entities

        resid = pooled.resid_
        ssr0 = np.sum(resid**2)
        ssr1 = sum([np.sum(resid[entity == e])**2 for e in entity])

        bp_stat = (n_entities * avg_sample_entity) / (2 * (avg_sample_entity - 1)) * (ssr1 / ssr0 - 1)**2
        bp_pval = stats.chi2.sf(bp_stat, df=1)

        self.results['breusch_pagan_test'] = {
            'stat': bp_stat, 
            'pval': bp_pval,
            'conclusion': 'Random effects exists -> RandomEffects' if bp_pval < self.alpha else 'Random effects does not exists -> PooledOLS'
        }

        return bp_stat, bp_pval

    def hausman_test(self, re: 'RandomEffects', fe: 'FixedEffects') -> Tuple[float, float]:
        """Test the null hypothesis that the Random Effects coefficient are consistent.
        
        H0: cov(alpha_i, X) = 0 | Random effects consistent -> RE more efficient    (= RandomEffects)
        H1: cov(alpha_i, X) ≠ 0 | Random effects inconsistent -> RE invalid         (= FixedEffects)

        Layman's terms:
        Check if FE coefficients are significantly different from RE coefficients. If they are the same then we favor RE due to more efficient (= more observations)
        """
        n_features = fe.n_features
        beta0 = re.beta
        beta1 = fe.beta
        var0 = re.var
        var1 = fe.var

        diff_beta = beta1 - beta0
        diff_var = var1 - var0

        hausman_stat = float(diff_beta @ np.linalg.inv(diff_var) @ diff_beta)
        hausman_pval = stats.chi2.sf(hausman_stat, df=n_features)

        self.results['hausman_test'] = {
            'stat': hausman_stat, 
            'pval': hausman_pval,
            'conclusion': 'RE inconsistent -> FixedEffects' if hausman_pval < self.alpha else 'RE consistent & more efficient -> RandomEffects'
        }

        return hausman_stat, hausman_pval
    
    def wooldridge_test(self, fe: 'FixedEffects', entity: pd.Series) -> Tuple[float, float]:
        """Test the null hypothesis that there is no autocorrelation.
        
        H0: rho = 0 | no AR(1)                              (= FixedEffects)
        H1: rho ≠ 0 | AR(1)                                 (= FirstDifference)

        Layman's terms:
        Check if FD coefficients are significantly different from FE coefficients. If they are the same then we favor FE due to more efficient (= more observations)
        """
        entities = np.unique(entity)
        list_rho = []
        for e in entities:
            r = fe.resid_[entity == e]
            if len(r) > 2:
                rho = np.corrcoef(r[:-1], r[1:])[0, 1]
                list_rho.append(rho)

        rho_mean = np.mean(list_rho)
        se = np.std(list_rho) / np.sqrt(len(list_rho))

        wooldridge_stat = rho_mean / se
        wooldridge_pval = stats.norm.sf(wooldridge_stat)

        self.results['wooldridge_test'] = {
            'stat': wooldridge_stat, 
            'pval': wooldridge_pval,
            'conclusion': 'AR(1) -> FirstDifference' if wooldridge_pval < self.alpha else 'no AR(1) -> FixedEffects'
        }

        return wooldridge_stat, wooldridge_pval
    
    def llr_random_slope(self, re: 'RandomEffects', lme: 'LinearMixedEffects') -> Tuple[float, float]:
        """Test the null hypothesis that the Random Effects coefficient are consistent.
        
        H0: cov(alpha_i, X) = 0 | Random effects consistent -> RE more efficient    (= RandomEffects)    
        H1: cov(alpha_i, X) ≠ 0 | Random effects inconsistent -> RE invalid         (= LinearMixedEffects)

        Layman's term:
        TO-DO:
        """
        logL0 = re.diagnostics['logL1']
        logL1 = lme.diagnostics['logL1']

        dof = len(lme.beta) - len(re.beta)
        llr_stat = 2 * (logL1 - logL0)
        llr_pval = stats.chi2.sf(llr_stat, df=dof)

        self.results['llr_random_slope'] = {
            'stat': llr_stat, 
            'pval': llr_pval,
            'conclusion': 'TO-DO -> LinearMixedEffects' if llr_pval < self.alpha else 'TO-DO -> RandomEffects'
        }