from pymatchingtools.__init__ import *
from pymatchingtools.utils import *

import pandas as pd
import numpy as np

from scipy import stats
from scipy.stats import ecdf, ks_2samp
from scipy.stats._result_classes import ECDFResult
import matplotlib.pyplot as plt
import seaborn as sns

def standardized_mean_difference(control: pd.Series, treatment: pd.Series, method='cohen_d', index_method='mean', eps=2**-53) -> float:
    """
    calculate standardized mean difference
    
    Parameters
    ----------
    control: pd.Series
        Data representing the control group
    treatment: pd.Series
        Data representing the treatment group
    method: str (optional), 
        The ways to calculate standardised_mean_difference. Default is 'cohen_d'. Support {'cohen_d', 'hedges_g', 'glass_delta'}
    index_method: str (optional)
        One way of calculating, using either the mean or the median. Default is 'mean'. Support {'mean', 'median'}
    eps: float (optional)
        An small value, Avoid cases where the denominator is 0 and cannot be calculated

    Returns
    ----------
    smd: float
        The result of standardized mean difference
    """

    if index_method == 'mean':
        m1 = np.mean(control)
        m2 = np.mean(treatment)
    elif index_method == 'median':
        m1 = np.median(control)
        m2 = np.median(treatment)

    else:
        raise Exception('index method wrong')


    if method == 'cohen_d':
        pool_std = np.std(pd.concat([control, treatment], axis=0)) + eps
        smd = (m1 - m2) * 1.0 / pool_std
    elif method == 'hedges_g':
        sample_std = np.std(pd.concat([control, treatment], axis=0), ddof=1)
        smd = (m1 - m2) * 1.0 / sample_std + eps
    elif method == 'glass_delta':
        sample_std = np.std(control, ddof=1)
        smd = (m1 - m2) * 1.0 / sample_std + eps
    else:
        raise Exception('method wrong')
    
    return smd

def variance_ratio(control: pd.Series, treatment: pd.Series, eps=2**-53):
    """
    calculate Variation ratio, a simple measure of statistical dispersion in nominal distributions
    
    Parameters
    ----------
    control: pd.Series
        Data representing the control group
    treatment: pd.Series
        Data representing the treatment group
    eps: float (optional)
        An small value, Avoid cases where the denominator is 0 and cannot be calculated

    Returns
    ----------
    f: float
        The result of Variation ratio
    """
    f = np.var(treatment, ddof=1)/ (np.var(control, ddof=1) + eps)
    return f

def EmpiricalCDF(control: pd.Series, plot: bool=False) -> ECDFResult:
    """
    get empirical cumulative distribution function
    
    Parameters
    ----------
    control: pd.Series
        Data representing the control group
    plot: bool (optional)
        Whether or not to draw empirical cumulative distribution function

    Returns
    ----------
    res: float
        The result of empirical cumulative distribution function
    """
    res = ecdf(control)

    if plot == True:
        ax = plt.subplot()
        res.cdf.plot(ax)
        ax.set_ylabel('Empirical CDF')
        plt.show()
    
    return res

def ks_boot_test(
    control: pd.Series, treatment: pd.Series, n_boots=1000,
    alternative: str='two_sided', eps=2**-53
):
    """
    calculate Kolmogorov Smirnov Boost Test
    
    Parameters
    ----------
    control: pd.Series
        Data representing the control group
    treatment: pd.Series
        Data representing the treatment group
    alternative: str (optional), 
        Defines the null and alternative hypotheses. Default is 'two-sided'.
        Support {'two-sided', 'less', 'greater'}
    index_method: str (optional)
        One way of calculating, using either the mean or the median.
    eps: float (optional)
        An small value, Avoid cases where the denominator is 0 and cannot be calculated

    Returns
    ----------
    ks_boot_p_value: float
        The result of ks test p-value
    """
    if type(control) == pd.Series and type(treatment) == pd.Series:
        control_array = control.values
        treatment_array = treatment.values

    treatment_obs_num = len(treatment_array)
    control_obs_num = len(control_array)
    w = np.concatenate([control_array, treatment_array])
    obs_num = len(w)

    cut_point = treatment_obs_num

    boot_cnt = 0
    stats_list = []

    tol = np.sqrt(eps)

    if n_boots < 10:
        print("At least 10 'nboots' must be run; seting 'nboots' to 10")
    elif n_boots < 500:
        print("For publication quality p-values it is recommended that 'nboots'\n be set equal to at least 500 (preferably 1000)")

    fs_ks, _ = ks_2samp(treatment_array, control_array, alternative=alternative)


    # print(f'ks.boot: {alternative} test')

    for _ in range(n_boots):
        sample_w = np.random.choice(w, obs_num, replace=True)

        x1_tmp = sample_w[: cut_point]
        x2_tmp = sample_w[cut_point:]

        s_ks, _ = ks_2samp(x1_tmp, x2_tmp, alternative=alternative)

        stats_list.append(s_ks)

        if s_ks > fs_ks - tol:
            boot_cnt += 1
    
    ks_boot_p_value = boot_cnt * 1.0 / n_boots

    return ks_boot_p_value




def average(values: pd.Series):

    return np.sum(values) / len(values)

