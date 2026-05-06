import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

from scipy import stats
from statsmodels.tsa.stattools import adfuller

from models.panel._01_ols_pooled import PooledOLS
from models.panel._02_re import RandomEffects
from models.panel._03_fe import FixedEffects
from models.panel._04_fd import FirstDifference
from models.panel._05_lme import LinearMixedEffects
from models.panel._06_ecm import ErrorCorrectionModel
from models.segmentation._10_fmr import FiniteMixtureRegression

class PanelModelTest:
    """Sequential Panel Tests for Panel Data.

    F-test:                 Entity effects              -> PooledOLS vs FE
    Breusch-Pagan test:     Random effects              -> PooledOLS vs RE
    Hausman test:           RE consistent               -> RE vs FE
    Wooldridge test:        Serial correlation          -> FE vs FD
    Likelihood Ratio Test:  Random slope                -> RE vs MLE
    Engler test:            Cointegration               -> PooledOLS vs ECM
    BIC over k:             Hidden heterogeneity        -> Finite Mixture Models
    Data sparsity:                                      -> MLE vs Hierarchical Bayes
    Nonlinearity:                                       -> MLE vs Gaussian Process Boost
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
        
        H0: alpha_i = 0 | No Entity Effects                     (= PooledOLS)
        H1: alpha_i ≠ 0 | Entity effects present                (= FixedEffects)

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
            'conclusion': 'Entity effects exists -> FixedEffects' if f_pval < self.alpha 
            else 'Entity effects do not exists -> PooledOLS'
        }
    
        return f_stat, f_pval

    def breusch_pagan_test(self, pooled: 'PooledOLS', y: pd.Series, entity: pd.Series) -> Tuple[float, float]:
        """Test the null hypothesis that the random effects are zero.
        
        H0: sigma2 = 0 | No random effects                     (= PooledOLS)
        H1: sigma2 > 0 | Random effects present                (= RandomEffects)

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
            'conclusion': 'Random effects exists -> RandomEffects' if bp_pval < self.alpha 
            else 'Random effects does not exists -> PooledOLS'
        }

        return bp_stat, bp_pval

    def hausman_test(self, re: 'RandomEffects', fe: 'FixedEffects') -> Tuple[float, float]:
        """Test the null hypothesis that the Random Effects coefficient are consistent.
        
        H0: cov(alpha_i, X) = 0 | Random effects consistent / uncorrelated -> RE more efficient    (= RandomEffects)
        H1: cov(alpha_i, X) ≠ 0 | Random effects inconsistent / correlated -> RE invalid         (= FixedEffects)

        Layman's terms:
        Check if FE coefficients are significantly different from RE coefficients. If they do not differ more than sample noise, 
        then RE due to more efficient (= more observations)
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
            'conclusion': 'RE inconsistent -> FixedEffects' if hausman_pval < self.alpha 
            else 'RE consistent & more efficient -> RandomEffects'
        }

        return hausman_stat, hausman_pval
    
    def wooldridge_test(self, fe: 'FixedEffects', entity: pd.Series) -> Tuple[float, float]:
        """Test the null hypothesis for AR(1) serial correlation in FE residuals.
        
        H0: p = 0 | No first order serial correlation                         (= FixedEffects)
        H1: p ≠ 0 | Serial correlation AR(1)                                  (= FirstDifference)

        Layman's terms:
        For each entity, look at how today's residual relates to yesterday's residual.
        If they are correlated, you likely have AR(1) serial correlation.
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
            'conclusion': 'AR(1) serial correlation -> FirstDifference' if wooldridge_pval < self.alpha 
            else 'no AR(1) -> FixedEffects'
        }

        return wooldridge_stat, wooldridge_pval
    
    def llr_random_slope(self, re: 'RandomEffects', lme: 'LinearMixedEffects') -> Tuple[float, float]:
        """Likelihood-ratio test for adding random slopes (RE vs LME).
        
        H0: Random intercept sufficient                 (= RandomEffects)
        H1: Random slopes significantly better          (= LinearMixedEffects)

        Layman's terms:
        Compare log-likelihoods of the simple random-intercept model and a richer random-slope mixed model. 
        If the richer model fits much better relative to the extra parameters, include random slopes.
        """
        logL0 = re.diagnostics['logL1']
        logL1 = lme.diagnostics['logL1']

        dof = len(lme.beta) - len(re.beta)
        llr_stat = 2 * (logL1 - logL0)
        llr_pval = stats.chi2.sf(llr_stat, df=dof)

        self.results['llr_random_slope'] = {
            'stat': llr_stat, 
            'pval': llr_pval,
            'conclusion': 'Random slopes significantly better -> LinearMixedEffects' if llr_pval < self.alpha 
            else 'Random intercept sufficient -> RandomEffects'
        }

        return llr_stat, llr_pval

    def engle_granger_cointegration(self, pooled: 'PooledOLS', entity: pd.Series, max_lag=1) -> Tuple[float, float]:
        """Engle-Granger residual-based cointegration test for panel data.

        Step 1: Use PooledOLS to estimate the long-run (levels) relationship.
                Obtain residuals resid = y_it - x_it'β_hat.

        Step 2: For each entity i, test resid_i for a unit root (e.g. ADF test).

        H0 (no cointegration): residuals contain a unit root (non-stationary)       (= PooledOLS)
                                -> pooled levels regression is spurious
        H1 (cointegration):    residuals are stationary                             (= ECM)

        Layman's terms:
        Check if deviations from the long-run relation are mean-reverting.
        If yes, there is a stable long-run equilibrium and an ECM is preferred.
        """
        resid = pooled.resid_
        entities = np.unique(entity)
        list_adf = []
        for e in entities:
            if len(resid[entity == e]) > max_lag + 1:
                adf_stat, adf_pval, _ = adfuller(resid[entity == e], maxlag=max_lag, regression='c', autolag=None)
                list_adf.append((adf_stat, adf_pval))

        engle_stat = np.mean([stat for stat, pval in list_adf])
        engle_pval = np.mean([pval < self.alpha for stat, pval in list_adf])

        self.results['engle_granger_cointegration'] = {
            'stat': engle_stat, 
            'pval': engle_pval,
            'conclusion': 'cointegration -> ECM' if engle_pval < self.alpha 
            else 'No cointegration -> PooledOLS'
        }

        return engle_stat, engle_pval

    def segment_validity_check(self, fmr: Dict[int, 'FiniteMixtureRegression']) -> Tuple[float, float]:
        """Compare BIC across K-component finite mixture models.
    
        H0: K = 1 no hidden segments -> No mixture      (= PooledOLS)
        H1: K > 1 hidden segments -> Mixture            (= FiniteMixtureRegression)
        """
        bic_values = [] 
        for k, model in enumerate(fmr):
            logL = model.diagnostics['logL1']
            bic = -2 * logL + model.n_features * np.log(model.n_samples)
            bic_values.append(bic)

        best_k = min(bic_values, key=bic_values.get)

        self.results['segment_validity_check'] = {
            'best_k': best_k,
            'bic': bic_values,
            'conclusion': f'Hidden segments k={best_k} -> FiniteMixtureRegression' if best_k > 1 
            else 'No hidden segments k=1 -> PooledOLS'
        }

        return best_k, bic_values

    def data_sparsity_check(self, entity: pd.Seres) -> Tuple[float, float]:
        """Assess data sparsity across entities for deciding MLE vs Hierarchical Bayes.
        
        H0: Panel dense enough                          (= MLE)
        H1: Panel sparse (many small-n entities)        (= Hierarchical Bayes)
        """
        entities, entity_counts = np.unique(entity, return_counts=True)

        self.results['data_sparsity_check'] = {
            'entities': entities,
            'entity_counts': entity_counts,
            'conclusion': 'Panel dense enough -> LinearMixedEffects' if np.mean(entity_counts) > 2 
            else 'Panel sparse -> Hierarchical Bayes'
        }

        return entities, entity_counts


    def nonlinear_check(self, y: pd.Series, y_linear: pd.Series, y_nonlinear: pd.Series) -> Tuple[float, float]:
        """Check for nonlinearity via predictive performance gap.
    
        Inputs are predictions on a held-out set from:
        - a linear model (e.g. MLE)
        - a nonlinear model (e.g. Gradient Boosting)

        H0: No substantial nonlinear structure (performance similar) -> linear MLE ok
        H1: Significant performance gain from nonlinear model -> use nonlinear model
            (e.g. Gradient Boosting, GP, etc.)
        """
        rmse_linear = np.sqrt(np.mean((y - y_linear)**2))
        rmse_nonlinear = np.sqrt(np.mean((y - y_nonlinear)**2))

        relative_improvement = (rmse_linear - rmse_nonlinear) / rmse_linear
        nonlinear_flg = relative_improvement > self.alpha

        self.results['nonlinear_check'] = {
            'rmse_linear': rmse_linear,
            'rmse_nonlinear': rmse_nonlinear,
            'nonlinear_flg': nonlinear_flg,
            'relative_improvement': relative_improvement,
            'conclusion': 'Nonlinear -> GPBoost' if nonlinear_flg 
            else 'Linear -> LinearMixedEffects'
        }

        return nonlinear_flg, relative_improvement