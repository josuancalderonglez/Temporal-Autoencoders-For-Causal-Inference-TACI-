
"""
Statistical tools for time series analysis
"""
from __future__ import annotations

from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import deprecate_kwarg
from statsmodels.compat.python import Literal, lzip
from statsmodels.compat.scipy import _next_regular

from typing import Tuple
import warnings

import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy.fft import fft, ifft

from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tools.sm_exceptions import (
    CollinearityWarning,
    InfeasibleTestError,
    InterpolationWarning,
    MissingDataError,
    ValueWarning,
)
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tools.validation import (
    array_like,
    bool_like,
    dict_like,
    float_like,
    int_like,
    string_like,
)
from statsmodels.tsa._bds import bds
from statsmodels.tsa._innovations import innovations_algo, innovations_filter
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
from statsmodels.tsa.tsatools import add_trend, lagmat, lagmat2ds


__all__ = [
    "acovf",
    "acf",
    "pacf",
    "pacf_yw",
    "pacf_ols",
    "ccovf",
    "ccf",
    "q_stat",
    "coint",
    "arma_order_select_ic",
    "adfuller",
    "kpss",
    "bds",
    "pacf_burg",
    "innovations_algo",
    "innovations_filter",
    "levinson_durbin_pacf",
    "levinson_durbin",
    "zivot_andrews",
    "range_unit_root_test",
]

SQRTEPS = np.sqrt(np.finfo(np.double).eps)

def _autolag(
    mod,
    endog,
    exog,
    startlag,
    maxlag,
    method,
    modargs=(),
    fitargs=(),
    regresults=False,
):
    """
    Returns the results for the lag length that maximizes the info criterion.

    Parameters
    ----------
    mod : Model class
        Model estimator class
    endog : array_like
        nobs array containing endogenous variable
    exog : array_like
        nobs by (startlag + maxlag) array containing lags and possibly other
        variables
    startlag : int
        The first zero-indexed column to hold a lag.  See Notes.
    maxlag : int
        The highest lag order for lag length selection.
    method : {"aic", "bic", "t-stat"}
        aic - Akaike Information Criterion
        bic - Bayes Information Criterion
        t-stat - Based on last lag
    modargs : tuple, optional
        args to pass to model.  See notes.
    fitargs : tuple, optional
        args to pass to fit.  See notes.
    regresults : bool, optional
        Flag indicating to return optional return results

    Returns
    -------
    icbest : float
        Best information criteria.
    bestlag : int
        The lag length that maximizes the information criterion.
    results : dict, optional
        Dictionary containing all estimation results

    Notes
    -----
    Does estimation like mod(endog, exog[:,:i], *modargs).fit(*fitargs)
    where i goes from lagstart to lagstart+maxlag+1.  Therefore, lags are
    assumed to be in contiguous columns from low to high lag length with
    the highest lag in the last column.
    """
    # TODO: can tcol be replaced by maxlag + 2?
    # TODO: This could be changed to laggedRHS and exog keyword arguments if
    #    this will be more general.

    results = {}
    method = method.lower()
    for lag in range(startlag, startlag + maxlag + 1):
        mod_instance = mod(endog, exog[:, :lag], *modargs)
        results[lag] = mod_instance.fit()

    if method == "aic":
        icbest, bestlag = min((v.aic, k) for k, v in results.items())
    elif method == "bic":
        icbest, bestlag = min((v.bic, k) for k, v in results.items())
    elif method == "t-stat":
        # stop = stats.norm.ppf(.95)
        stop = 1.6448536269514722
        # Default values to ensure that always set
        bestlag = startlag + maxlag
        icbest = 0.0
        for lag in range(startlag + maxlag, startlag - 1, -1):
            icbest = np.abs(results[lag].tvalues[-1])
            bestlag = lag
            if np.abs(icbest) >= stop:
                # Break for first lag with a significant t-stat
                break
    else:
        raise ValueError(f"Information Criterion {method} not understood.")

    if not regresults:
        return icbest, bestlag
    else:
        return icbest, bestlag, results

# this needs to be converted to a class like HetGoldfeldQuandt,
# 3 different returns are a mess
# See:
# Ng and Perron(2001), Lag length selection and the construction of unit root
# tests with good size and power, Econometrica, Vol 69 (6) pp 1519-1554
# TODO: include drift keyword, only valid with regression == "c"
# just changes the distribution of the test statistic to a t distribution
# TODO: autolag is untested

def adfuller(x, maxlag: int | None = None, regression="c", autolag="AIC", store=False, regresults=False, ):
    """
    Augmented Dickey-Fuller unit root test.

    The Augmented Dickey-Fuller test can be used to test for a unit root in a
    univariate process in the presence of serial correlation.

    Parameters
    ----------
    x : array_like, 1d
        The data series to test.
    maxlag : {None, int}
        Maximum lag which is included in test, default value of
        12*(nobs/100)^{1/4} is used when ``None``.
    regression : {"c","ct","ctt","n"}
        Constant and trend order to include in regression.

        * "c" : constant only (default).
        * "ct" : constant and trend.
        * "ctt" : constant, and linear and quadratic trend.
        * "n" : no constant, no trend.

    autolag : {"AIC", "BIC", "t-stat", None}
        Method to use when automatically determining the lag length among the
        values 0, 1, ..., maxlag.

        * If "AIC" (default) or "BIC", then the number of lags is chosen
        to minimize the corresponding information criterion.
        * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
        lag until the t-statistic on the last lag length is significant
        using a 5%-sized test.
        * If None, then the number of included lags is set to maxlag.
    store : bool
        If True, then a result instance is returned additionally to
        the adf statistic. Default is False.
    regresults : bool, optional
        If True, the full regression results are returned. Default is False.

    Returns
    -------
    adf : float
        The test statistic.
    pvalue : float
        MacKinnon's approximate p-value based on MacKinnon (1994, 2010).
    usedlag : int
        The number of lags used.
    nobs : int
        The number of observations used for the ADF regression and calculation
        of the critical values.
    critical values : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 %
        levels. Based on MacKinnon (2010).
    icbest : float
        The maximized information criterion if autolag is not None.
    resstore : ResultStore, optional
        A dummy class with results attached as attributes.

    Notes
    -----
    The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
    root, with the alternative that there is no unit root. If the pvalue is
    above a critical size, then we cannot reject that there is a unit root.

    The p-values are obtained through regression surface approximation from
    MacKinnon 1994, but using the updated 2010 tables. If the p-value is close
    to significant, then the critical values should be used to judge whether
    to reject the null.

    The autolag option and maxlag for it are described in Greene.

    See the notebook `Stationarity and detrending (ADF/KPSS)
    <../examples/notebooks/generated/stationarity_detrending_adf_kpss.html>`__
    for an overview.

    References
    ----------
    .. [1] W. Green.  "Econometric Analysis," 5th ed., Pearson, 2003.

    .. [2] Hamilton, J.D.  "Time Series Analysis".  Princeton, 1994.

    .. [3] MacKinnon, J.G. 1994.  "Approximate asymptotic distribution functions for
        unit-root and cointegration tests.  `Journal of Business and Economic
        Statistics` 12, 167-76.

    .. [4] MacKinnon, J.G. 2010. "Critical Values for Cointegration Tests."  Queen"s
        University, Dept of Economics, Working Papers.  Available at
        http://ideas.repec.org/p/qed/wpaper/1227.html
    """
    x = array_like(x, "x")
    maxlag = int_like(maxlag, "maxlag", optional=True)
    regression = string_like(
        regression, "regression", options=("c", "ct", "ctt", "n")
    )
    autolag = string_like(
        autolag, "autolag", optional=True, options=("aic", "bic", "t-stat")
    )
    store = bool_like(store, "store")
    regresults = bool_like(regresults, "regresults")

    if x.max() == x.min():
        raise ValueError("Invalid input, x is constant")

    if regresults:
        store = True

    trenddict = {None: "n", 0: "c", 1: "ct", 2: "ctt"}
    if regression is None or isinstance(regression, int):
        regression = trenddict[regression]
    regression = regression.lower()
    nobs = x.shape[0]

    ntrend = len(regression) if regression != "n" else 0
    if maxlag is None:
        # from Greene referencing Schwert 1989
        maxlag = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
        # -1 for the diff
        maxlag = min(nobs // 2 - ntrend - 1, maxlag)
        if maxlag < 0:
            raise ValueError(
                "sample size is too short to use selected "
                "regression component"
            )
    elif maxlag > nobs // 2 - ntrend - 1:
        raise ValueError(
            "maxlag must be less than (nobs/2 - 1 - ntrend) "
            "where n trend is the number of included "
            "deterministic regressors"
        )
    xdiff = np.diff(x)
    xdall = lagmat(xdiff[:, None], maxlag, trim="both", original="in")
    nobs = xdall.shape[0]

    xdall[:, 0] = x[-nobs - 1 : -1]  # replace 0 xdiff with level of x
    xdshort = xdiff[-nobs:]

    if store:
        from statsmodels.stats.diagnostic import ResultsStore

        resstore = ResultsStore()
    if autolag:
        if regression != "n":
            fullRHS = add_trend(xdall, regression, prepend=True)
        else:
            fullRHS = xdall
        startlag = fullRHS.shape[1] - xdall.shape[1] + 1
        # 1 for level
        # search for lag length with smallest information criteria
        # Note: use the same number of observations to have comparable IC
        # aic and bic: smaller is better

        if not regresults:
            icbest, bestlag = _autolag(
                OLS, xdshort, fullRHS, startlag, maxlag, autolag
            )
        else:
            icbest, bestlag, alres = _autolag(
                OLS,
                xdshort,
                fullRHS,
                startlag,
                maxlag,
                autolag,
                regresults=regresults,
            )
            resstore.autolag_results = alres

        bestlag -= startlag  # convert to lag not column index

        # rerun ols with best autolag
        xdall = lagmat(xdiff[:, None], bestlag, trim="both", original="in")
        nobs = xdall.shape[0]
        xdall[:, 0] = x[-nobs - 1 : -1]  # replace 0 xdiff with level of x
        xdshort = xdiff[-nobs:]
        usedlag = bestlag
    else:
        usedlag = maxlag
        icbest = None
    if regression != "n":
        resols = OLS(
            xdshort, add_trend(xdall[:, : usedlag + 1], regression)
        ).fit()
    else:
        resols = OLS(xdshort, xdall[:, : usedlag + 1]).fit()

    adfstat = resols.tvalues[0]
    #    adfstat = (resols.params[0]-1.0)/resols.bse[0]
    # the "asymptotically correct" z statistic is obtained as
    # nobs/(1-np.sum(resols.params[1:-(trendorder+1)])) (resols.params[0] - 1)
    # I think this is the statistic that is used for series that are integrated
    # for orders higher than I(1), ie., not ADF but cointegration tests.

    # Get approx p-value and critical values
    pvalue = mackinnonp(adfstat, regression=regression, N=1)
    critvalues = mackinnoncrit(N=1, regression=regression, nobs=nobs)
    critvalues = {
        "1%": critvalues[0],
        "5%": critvalues[1],
        "10%": critvalues[2],
    }
    if store:
        resstore.resols = resols
        resstore.maxlag = maxlag
        resstore.usedlag = usedlag
        resstore.adfstat = adfstat
        resstore.critvalues = critvalues
        resstore.nobs = nobs
        resstore.H0 = (
            "The coefficient on the lagged level equals 1 - " "unit root"
        )
        resstore.HA = "The coefficient on the lagged level < 1 - stationary"
        resstore.icbest = icbest
        resstore._str = "Augmented Dickey-Fuller Test Results"
        return adfstat, pvalue, critvalues, resstore
    else:
        if not autolag:
            return adfstat, pvalue, usedlag, nobs, critvalues
        else:
            return adfstat, pvalue, usedlag, nobs, critvalues, icbest

@deprecate_kwarg("unbiased", "adjusted")

def acovf(x, adjusted=False, demean=True, fft=True, missing="none", nlag=None):
    """
    Estimate autocovariances.

    Parameters
    ----------
    x : array_like
        Time series data. Must be 1d.
    adjusted : bool, default False
        If True, then denominators is n-k, otherwise n.
    demean : bool, default True
        If True, then subtract the mean x from each element of x.
    fft : bool, default True
        If True, use FFT convolution.  This method should be preferred
        for long time series.
    missing : str, default "none"
        A string in ["none", "raise", "conservative", "drop"] specifying how
        the NaNs are to be treated. "none" performs no checks. "raise" raises
        an exception if NaN values are found. "drop" removes the missing
        observations and then estimates the autocovariances treating the
        non-missing as contiguous. "conservative" computes the autocovariance
        using nan-ops so that nans are removed when computing the mean
        and cross-products that are used to estimate the autocovariance.
        When using "conservative", n is set to the number of non-missing
        observations.
    nlag : {int, None}, default None
        Limit the number of autocovariances returned.  Size of returned
        array is nlag + 1.  Setting nlag when fft is False uses a simple,
        direct estimator of the autocovariances that only computes the first
        nlag + 1 values. This can be much faster when the time series is long
        and only a small number of autocovariances are needed.

    Returns
    -------
    ndarray
        The estimated autocovariances.

    References
    ----------
    .. [1] Parzen, E., 1963. On spectral analysis with missing observations
        and amplitude modulation. Sankhya: The Indian Journal of
        Statistics, Series A, pp.383-392.
    """
    adjusted = bool_like(adjusted, "adjusted")
    demean = bool_like(demean, "demean")
    fft = bool_like(fft, "fft", optional=False)
    missing = string_like(
        missing, "missing", options=("none", "raise", "conservative", "drop")
    )
    nlag = int_like(nlag, "nlag", optional=True)

    x = array_like(x, "x", ndim=1)

    missing = missing.lower()
    if missing == "none":
        deal_with_masked = False
    else:
        deal_with_masked = has_missing(x)
    if deal_with_masked:
        if missing == "raise":
            raise MissingDataError("NaNs were encountered in the data")
        notmask_bool = ~np.isnan(x)  # bool
        if missing == "conservative":
            # Must copy for thread safety
            x = x.copy()
            x[~notmask_bool] = 0
        else:  # "drop"
            x = x[notmask_bool]  # copies non-missing
        notmask_int = notmask_bool.astype(int)  # int

    if demean and deal_with_masked:
        # whether "drop" or "conservative":
        xo = x - x.sum() / notmask_int.sum()
        if missing == "conservative":
            xo[~notmask_bool] = 0
    elif demean:
        xo = x - x.mean()
    else:
        xo = x

    n = len(x)
    lag_len = nlag
    if nlag is None:
        lag_len = n - 1
    elif nlag > n - 1:
        raise ValueError("nlag must be smaller than nobs - 1")

    if not fft and nlag is not None:
        acov = np.empty(lag_len + 1)
        acov[0] = xo.dot(xo)
        for i in range(lag_len):
            acov[i + 1] = xo[i + 1 :].dot(xo[: -(i + 1)])
        if not deal_with_masked or missing == "drop":
            if adjusted:
                acov /= n - np.arange(lag_len + 1)
            else:
                acov /= n
        else:
            if adjusted:
                divisor = np.empty(lag_len + 1, dtype=np.int64)
                divisor[0] = notmask_int.sum()
                for i in range(lag_len):
                    divisor[i + 1] = notmask_int[i + 1 :].dot(
                        notmask_int[: -(i + 1)]
                    )
                divisor[divisor == 0] = 1
                acov /= divisor
            else:  # biased, missing data but npt "drop"
                acov /= notmask_int.sum()
        return acov

    if adjusted and deal_with_masked and missing == "conservative":
        d = np.correlate(notmask_int, notmask_int, "full")
        d[d == 0] = 1
    elif adjusted:
        xi = np.arange(1, n + 1)
        d = np.hstack((xi, xi[:-1][::-1]))
    elif deal_with_masked:
        # biased and NaNs given and ("drop" or "conservative")
        d = notmask_int.sum() * np.ones(2 * n - 1)
    else:  # biased and no NaNs or missing=="none"
        d = n * np.ones(2 * n - 1)

    if fft:
        nobs = len(xo)
        n = _next_regular(2 * nobs + 1)
        Frf = np.fft.fft(xo, n=n)
        acov = np.fft.ifft(Frf * np.conjugate(Frf))[:nobs] / d[nobs - 1 :]
        acov = acov.real
    else:
        acov = np.correlate(xo, xo, "full")[n - 1 :] / d[n - 1 :]

    if nlag is not None:
        # Copy to allow gc of full array rather than view
        return acov[: lag_len + 1].copy()
    return acov

def q_stat(x, nobs):
    """
    Compute Ljung-Box Q Statistic.

    Parameters
    ----------
    x : array_like
        Array of autocorrelation coefficients.  Can be obtained from acf.
    nobs : int, optional
        Number of observations in the entire sample (ie., not just the length
        of the autocorrelation function results.

    Returns
    -------
    q-stat : ndarray
        Ljung-Box Q-statistic for autocorrelation parameters.
    p-value : ndarray
        P-value of the Q statistic.

    See Also
    --------
    statsmodels.stats.diagnostic.acorr_ljungbox
        Ljung-Box Q-test for autocorrelation in time series based
        on a time series rather than the estimated autocorrelation
        function.

    Notes
    -----
    Designed to be used with acf.
    """
    x = array_like(x, "x")
    nobs = int_like(nobs, "nobs")

    ret = (
        nobs
        * (nobs + 2)
        * np.cumsum((1.0 / (nobs - np.arange(1, len(x) + 1))) * x ** 2)
    )
    chi2 = stats.chi2.sf(ret, np.arange(1, len(x) + 1))
    return ret, chi2

# NOTE: Changed unbiased to False
# see for example
# http://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm
def acf(
    x,
    adjusted=False,
    nlags=None,
    qstat=False,
    fft=True,
    alpha=None,
    bartlett_confint=True,
    missing="none",
):
    """
    Calculate the autocorrelation function.

    Parameters
    ----------
    x : array_like
    The time series data.
    adjusted : bool, default False
    If True, then denominators for autocovariance are n-k, otherwise n.
    nlags : int, optional
        Number of lags to return autocorrelation for. If not provided,
        uses min(10 * np.log10(nobs), nobs - 1). The returned value
        includes lag 0 (ie., 1) so size of the acf vector is (nlags + 1,).
    qstat : bool, default False
        If True, returns the Ljung-Box q statistic for each autocorrelation
        coefficient.  See q_stat for more information.
    fft : bool, default True
        If True, computes the ACF via FFT.
    alpha : scalar, default None
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        Bartlett"s formula.
    bartlett_confint : bool, default True
        Confidence intervals for ACF values are generally placed at 2
        standard errors around r_k. The formula used for standard error
        depends upon the situation. If the autocorrelations are being used
        to test for randomness of residuals as part of the ARIMA routine,
        the standard errors are determined assuming the residuals are white
        noise. The approximate formula for any lag is that standard error
        of each r_k = 1/sqrt(N). See section 9.4 of [2] for more details on
        the 1/sqrt(N) result. For more elementary discussion, see section 5.3.2
        in [3].
        For the ACF of raw data, the standard error at a lag k is
        found as if the right model was an MA(k-1). This allows the possible
        interpretation that if all autocorrelations past a certain lag are
        within the limits, the model might be an MA of order defined by the
        last significant autocorrelation. In this case, a moving average
        model is assumed for the data and the standard errors for the
        confidence intervals should be generated using Bartlett's formula.
        For more details on Bartlett formula result, see section 7.2 in [2].
    missing : str, default "none"
        A string in ["none", "raise", "conservative", "drop"] specifying how
        the NaNs are to be treated. "none" performs no checks. "raise" raises
        an exception if NaN values are found. "drop" removes the missing
        observations and then estimates the autocovariances treating the
        non-missing as contiguous. "conservative" computes the autocovariance
        using nan-ops so that nans are removed when computing the mean
        and cross-products that are used to estimate the autocovariance.
        When using "conservative", n is set to the number of non-missing
        observations.

    Returns
    -------
    acf : ndarray
        The autocorrelation function for lags 0, 1, ..., nlags. Shape
        (nlags+1,).
    confint : ndarray, optional
        Confidence intervals for the ACF at lags 0, 1, ..., nlags. Shape
        (nlags + 1, 2). Returned if alpha is not None.
    qstat : ndarray, optional
        The Ljung-Box Q-Statistic for lags 1, 2, ..., nlags (excludes lag
        zero). Returned if q_stat is True.
    pvalues : ndarray, optional
        The p-values associated with the Q-statistics for lags 1, 2, ...,
        nlags (excludes lag zero). Returned if q_stat is True.

    Notes
    -----
    The acf at lag 0 (ie., 1) is returned.

    For very long time series it is recommended to use fft convolution instead.
    When fft is False uses a simple, direct estimator of the autocovariances
    that only computes the first nlag + 1 values. This can be much faster when
    the time series is long and only a small number of autocovariances are
    needed.

    If adjusted is true, the denominator for the autocovariance is adjusted
    for the loss of data.

    References
    ----------
    .. [1] Parzen, E., 1963. On spectral analysis with missing observations
    and amplitude modulation. Sankhya: The Indian Journal of
    Statistics, Series A, pp.383-392.
    .. [2] Brockwell and Davis, 1987. Time Series Theory and Methods
    .. [3] Brockwell and Davis, 2010. Introduction to Time Series and
    Forecasting, 2nd edition.
    """
    adjusted = bool_like(adjusted, "adjusted")
    nlags = int_like(nlags, "nlags", optional=True)
    qstat = bool_like(qstat, "qstat")
    fft = bool_like(fft, "fft", optional=False)
    alpha = float_like(alpha, "alpha", optional=True)
    missing = string_like(
        missing, "missing", options=("none", "raise", "conservative", "drop")
    )
    x = array_like(x, "x")
    # TODO: should this shrink for missing="drop" and NaNs in x?
    nobs = x.shape[0]
    if nlags is None:
        nlags = min(int(10 * np.log10(nobs)), nobs - 1)

    avf = acovf(x, adjusted=adjusted, demean=True, fft=fft, missing=missing)
    acf = avf[: nlags + 1] / avf[0]
    if not (qstat or alpha):
        return acf
    _alpha = alpha if alpha is not None else 0.05
    if bartlett_confint:
        varacf = np.ones_like(acf) / nobs
        varacf[0] = 0
        varacf[1] = 1.0 / nobs
        varacf[2:] *= 1 + 2 * np.cumsum(acf[1:-1] ** 2)
    else:
        varacf = 1.0 / len(x)
    interval = stats.norm.ppf(1 - _alpha / 2.0) * np.sqrt(varacf)
    confint = np.array(lzip(acf - interval, acf + interval))
    if not qstat:
        return acf, confint
    qstat, pvalue = q_stat(acf[1:], nobs=nobs)  # drop lag 0
    if alpha is not None:
        return acf, confint, qstat, pvalue
    else:
        return acf, qstat, pvalue

def pacf_yw(x, nlags=None, method="adjusted"):
    """
    Partial autocorrelation estimated with non-recursive yule_walker.

    Parameters
    ----------
    x : array_like
        The observations of time series for which pacf is calculated.
    nlags : int, optional
        Number of lags to return autocorrelation for. If not provided,
        uses min(10 * np.log10(nobs), nobs - 1).
    method : {"adjusted", "mle"}, default "adjusted"
        The method for the autocovariance calculations in yule walker.

    Returns
    -------
    ndarray
        The partial autocorrelations, maxlag+1 elements.

    See Also
    --------
    statsmodels.tsa.stattools.pacf
        Partial autocorrelation estimation.
    statsmodels.tsa.stattools.pacf_ols
        Partial autocorrelation estimation using OLS.
    statsmodels.tsa.stattools.pacf_burg
        Partial autocorrelation estimation using Burg"s method.

    Notes
    -----
    This solves yule_walker for each desired lag and contains
    currently duplicate calculations.
    """
    x = array_like(x, "x")
    nlags = int_like(nlags, "nlags", optional=True)
    nobs = x.shape[0]
    if nlags is None:
        nlags = min(int(10 * np.log10(nobs)), nobs - 1)

    method = string_like(method, "method", options=("adjusted", "mle"))
    pacf = [1.0]
    with warnings.catch_warnings():
        warnings.simplefilter("once", ValueWarning)
        for k in range(1, nlags + 1):
            pacf.append(yule_walker(x, k, method=method)[0][-1])
    return np.array(pacf)

def pacf_burg(x, nlags=None, demean=True):
    """
    Calculate Burg"s partial autocorrelation estimator.

    Parameters
    ----------
    x : array_like
        Observations of time series for which pacf is calculated.
    nlags : int, optional
        Number of lags to return autocorrelation for. If not provided,
        uses min(10 * np.log10(nobs), nobs - 1).
    demean : bool, optional
        Flag indicating to demean that data. Set to False if x has been
        previously demeaned.

    Returns
    -------
    pacf : ndarray
        Partial autocorrelations for lags 0, 1, ..., nlag.
    sigma2 : ndarray
        Residual variance estimates where the value in position m is the
        residual variance in an AR model that includes m lags.

    See Also
    --------
    statsmodels.tsa.stattools.pacf
        Partial autocorrelation estimation.
    statsmodels.tsa.stattools.pacf_yw
        Partial autocorrelation estimation using Yule-Walker.
    statsmodels.tsa.stattools.pacf_ols
        Partial autocorrelation estimation using OLS.

    References
    ----------
    .. [1] Brockwell, P.J. and Davis, R.A., 2016. Introduction to time series
        and forecasting. Springer.
    """
    x = array_like(x, "x")
    if demean:
        x = x - x.mean()
    nobs = x.shape[0]
    p = nlags if nlags is not None else min(int(10 * np.log10(nobs)), nobs - 1)
    if p > nobs - 1:
        raise ValueError("nlags must be smaller than nobs - 1")
    d = np.zeros(p + 1)
    d[0] = 2 * x.dot(x)
    pacf = np.zeros(p + 1)
    u = x[::-1].copy()
    v = x[::-1].copy()
    d[1] = u[:-1].dot(u[:-1]) + v[1:].dot(v[1:])
    pacf[1] = 2 / d[1] * v[1:].dot(u[:-1])
    last_u = np.empty_like(u)
    last_v = np.empty_like(v)
    for i in range(1, p):
        last_u[:] = u
        last_v[:] = v
        u[1:] = last_u[:-1] - pacf[i] * last_v[1:]
        v[1:] = last_v[1:] - pacf[i] * last_u[:-1]
        d[i + 1] = (1 - pacf[i] ** 2) * d[i] - v[i] ** 2 - u[-1] ** 2
        pacf[i + 1] = 2 / d[i + 1] * v[i + 1 :].dot(u[i:-1])
    sigma2 = (1 - pacf ** 2) * d / (2.0 * (nobs - np.arange(0, p + 1)))
    pacf[0] = 1  # Insert the 0 lag partial autocorrel

    return pacf, sigma2

@deprecate_kwarg("unbiased", "adjusted")
def pacf_ols(x, nlags=None, efficient=True, adjusted=False):
    """
    Calculate partial autocorrelations via OLS.

    Parameters
    ----------
    x : array_like
        Observations of time series for which pacf is calculated.
    nlags : int, optional
        Number of lags to return autocorrelation for. If not provided,
        uses min(10 * np.log10(nobs), nobs - 1).
    efficient : bool, optional
        If true, uses the maximum number of available observations to compute
        each partial autocorrelation. If not, uses the same number of
        observations to compute all pacf values.
    adjusted : bool, optional
        Adjust each partial autocorrelation by n / (n - lag).

    Returns
    -------
    ndarray
        The partial autocorrelations, (maxlag,) array corresponding to lags
        0, 1, ..., maxlag.

    See Also
    --------
    statsmodels.tsa.stattools.pacf
        Partial autocorrelation estimation.
    statsmodels.tsa.stattools.pacf_yw
        Partial autocorrelation estimation using Yule-Walker.
    statsmodels.tsa.stattools.pacf_burg
        Partial autocorrelation estimation using Burg"s method.

    Notes
    -----
    This solves a separate OLS estimation for each desired lag using method in
    [1]_. Setting efficient to True has two effects. First, it uses
    `nobs - lag` observations of estimate each pacf.  Second, it re-estimates
    the mean in each regression. If efficient is False, then the data are first
    demeaned, and then `nobs - maxlag` observations are used to estimate each
    partial autocorrelation.

    The inefficient estimator appears to have better finite sample properties.
    This option should only be used in time series that are covariance
    stationary.

    OLS estimation of the pacf does not guarantee that all pacf values are
    between -1 and 1.

    References
    ----------
    .. [1] Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
    Time series analysis: forecasting and control. John Wiley & Sons, p. 66
    """
    x = array_like(x, "x")
    nlags = int_like(nlags, "nlags", optional=True)
    efficient = bool_like(efficient, "efficient")
    adjusted = bool_like(adjusted, "adjusted")
    nobs = x.shape[0]
    if nlags is None:
        nlags = min(int(10 * np.log10(nobs)), nobs - 1)

    pacf = np.empty(nlags + 1)
    pacf[0] = 1.0
    if efficient:
        xlags, x0 = lagmat(x, nlags, original="sep")
        xlags = add_constant(xlags)
        for k in range(1, nlags + 1):
            params = lstsq(xlags[k:, : k + 1], x0[k:], rcond=None)[0]
            pacf[k] = np.squeeze(params[-1])
    else:
        x = x - np.mean(x)
        # Create a single set of lags for multivariate OLS
        xlags, x0 = lagmat(x, nlags, original="sep", trim="both")
        for k in range(1, nlags + 1):
            params = lstsq(xlags[:, :k], x0, rcond=None)[0]
            # Last coefficient corresponds to PACF value (see [1])
            pacf[k] = np.squeeze(params[-1])

    if adjusted:
        pacf *= nobs / (nobs - np.arange(nlags + 1))

    return pacf

def pacf(x, nlags=None, method="ywadjusted", alpha=None):
    """
    Partial autocorrelation estimate.

    Parameters
    ----------
    x : array_like
        Observations of time series for which pacf is calculated.
    nlags : int, optional
        Number of lags to return autocorrelation for. If not provided,
        uses min(10 * np.log10(nobs), nobs // 2 - 1). The returned value
        includes lag 0 (ie., 1) so size of the pacf vector is (nlags + 1,).
    method : str, default "ywunbiased"
        Specifies which method for the calculations to use.

        - "yw" or "ywadjusted" : Yule-Walker with sample-size adjustment in
        denominator for acovf. Default.
        - "ywm" or "ywmle" : Yule-Walker without adjustment.
        - "ols" : regression of time series on lags of it and on constant.
        - "ols-inefficient" : regression of time series on lags using a single
        common sample to estimate all pacf coefficients.
        - "ols-adjusted" : regression of time series on lags with a bias
        adjustment.
        - "ld" or "ldadjusted" : Levinson-Durbin recursion with bias
        correction.
        - "ldb" or "ldbiased" : Levinson-Durbin recursion without bias
        correction.
        - "burg" :  Burg"s partial autocorrelation estimator.

    alpha : float, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        1/sqrt(len(x)).

    Returns
    -------
    pacf : ndarray
        The partial autocorrelations for lags 0, 1, ..., nlags. Shape
        (nlags+1,).
    confint : ndarray, optional
        Confidence intervals for the PACF at lags 0, 1, ..., nlags. Shape
        (nlags + 1, 2). Returned if alpha is not None.

    See Also
    --------
    statsmodels.tsa.stattools.acf
        Estimate the autocorrelation function.
    statsmodels.tsa.stattools.pacf
        Partial autocorrelation estimation.
    statsmodels.tsa.stattools.pacf_yw
        Partial autocorrelation estimation using Yule-Walker.
    statsmodels.tsa.stattools.pacf_ols
        Partial autocorrelation estimation using OLS.
    statsmodels.tsa.stattools.pacf_burg
        Partial autocorrelation estimation using Burg"s method.

    Notes
    -----
    Based on simulation evidence across a range of low-order ARMA models,
    the best methods based on root MSE are Yule-Walker (MLW), Levinson-Durbin
    (MLE) and Burg, respectively. The estimators with the lowest bias included
    included these three in addition to OLS and OLS-adjusted.

    Yule-Walker (adjusted) and Levinson-Durbin (adjusted) performed
    consistently worse than the other options.
    """
    nlags = int_like(nlags, "nlags", optional=True)
    methods = (
        "ols",
        "ols-inefficient",
        "ols-adjusted",
        "yw",
        "ywa",
        "ld",
        "ywadjusted",
        "yw_adjusted",
        "ywm",
        "ywmle",
        "yw_mle",
        "lda",
        "ldadjusted",
        "ld_adjusted",
        "ldb",
        "ldbiased",
        "ld_biased",
        "burg"
    )
    x = array_like(x, "x", maxdim=2)
    method = string_like(method, "method", options=methods)
    alpha = float_like(alpha, "alpha", optional=True)

    nobs = x.shape[0]
    if nlags is None:
        nlags = min(int(10 * np.log10(nobs)), nobs // 2 - 1)
    if nlags >= x.shape[0] // 2:
        raise ValueError(
            "Can only compute partial correlations for lags up to 50% of the "
            f"sample size. The requested nlags {nlags} must be < "
            f"{x.shape[0] // 2}."
        )

    if method in ("ols", "ols-inefficient", "ols-adjusted"):
        efficient = "inefficient" not in method
        adjusted = "adjusted" in method
        ret = pacf_ols(x, nlags=nlags, efficient=efficient, adjusted=adjusted)
    elif method in ("yw", "ywa", "ywadjusted", "yw_adjusted"):
        ret = pacf_yw(x, nlags=nlags, method="adjusted")
    elif method in ("ywm", "ywmle", "yw_mle"):
        ret = pacf_yw(x, nlags=nlags, method="mle")
    elif method in ("ld", "lda", "ldadjusted", "ld_adjusted"):
        acv = acovf(x, adjusted=True, fft=False)
        ld_ = levinson_durbin(acv, nlags=nlags, isacov=True)
        ret = ld_[2]
    elif method == "burg":
        ret, _ = pacf_burg(x, nlags=nlags, demean=True)
    # inconsistent naming with ywmle
    else:  # method in ("ldb", "ldbiased", "ld_biased")
        acv = acovf(x, adjusted=False, fft=False)
        ld_ = levinson_durbin(acv, nlags=nlags, isacov=True)
        ret = ld_[2]

    if alpha is not None:
        varacf = 1.0 / len(x)  # for all lags >=1
        interval = stats.norm.ppf(1.0 - alpha / 2.0) * np.sqrt(varacf)
        confint = np.array(lzip(ret - interval, ret + interval))
        confint[0] = ret[0]  # fix confidence interval for lag 0 to varpacf=0
        return ret, confint
    else:
        return ret

@deprecate_kwarg("unbiased", "adjusted")
def ccovf(x, y, adjusted=True, demean=True, fft=True):
    """
    Calculate the crosscovariance between two series.

    Parameters
    ----------
    x, y : array_like
    The time series data to use in the calculation.
    adjusted : bool, optional
    If True, then denominators for crosscovariance is n-k, otherwise n.
    demean : bool, optional
        Flag indicating whether to demean x and y.
    fft : bool, default True
        If True, use FFT convolution.  This method should be preferred
        for long time series.

    Returns
    -------
    ndarray
        The estimated crosscovariance function.
    """
    x = array_like(x, "x")
    y = array_like(y, "y")
    adjusted = bool_like(adjusted, "adjusted")
    demean = bool_like(demean, "demean")
    fft = bool_like(fft, "fft", optional=False)

    n = len(x)
    if demean:
        xo = x - x.mean()
        yo = y - y.mean()
    else:
        xo = x
        yo = y
    if adjusted:
        d = np.arange(n, 0, -1)
    else:
        d = n

    method = "fft" if fft else "direct"
    return correlate(xo, yo, "full", method=method)[n - 1 :] / d

@deprecate_kwarg("unbiased", "adjusted")
def ccf(x, y, adjusted=True, fft=True):
    """
    The cross-correlation function.

    Parameters
    ----------
    x, y : array_like
        The time series data to use in the calculation.
    adjusted : bool
        If True, then denominators for cross-correlation is n-k, otherwise n.
    fft : bool, default True
        If True, use FFT convolution.  This method should be preferred
        for long time series.

    Returns
    -------
    ndarray
        The cross-correlation function of x and y.

    Notes
    -----
    If adjusted is true, the denominator for the autocovariance is adjusted.
    """
    x = array_like(x, "x")
    y = array_like(y, "y")
    adjusted = bool_like(adjusted, "adjusted")
    fft = bool_like(fft, "fft", optional=False)

    cvf = ccovf(x, y, adjusted=adjusted, demean=True, fft=fft)
    return cvf / (np.std(x) * np.std(y))

# moved from sandbox.tsa.examples.try_ld_nitime, via nitime
# TODO: check what to return, for testing and trying out returns everything
def levinson_durbin(s, nlags=10, isacov=False):
    """
    Levinson-Durbin recursion for autoregressive processes.

    Parameters
    ----------
    s : array_like
        If isacov is False, then this is the time series. If iasacov is true
        then this is interpreted as autocovariance starting with lag 0.
    nlags : int, optional
        The largest lag to include in recursion or order of the autoregressive
        process.
    isacov : bool, optional
        Flag indicating whether the first argument, s, contains the
        autocovariances or the data series.

    Returns
    -------
    sigma_v : float
        The estimate of the error variance.
    arcoefs : ndarray
        The estimate of the autoregressive coefficients for a model including
        nlags.
    pacf : ndarray
        The partial autocorrelation function.
    sigma : ndarray
        The entire sigma array from intermediate result, last value is sigma_v.
    phi : ndarray
        The entire phi array from intermediate result, last column contains
        autoregressive coefficients for AR(nlags).

    Notes
    -----
    This function returns currently all results, but maybe we drop sigma and
    phi from the returns.

    If this function is called with the time series (isacov=False), then the
    sample autocovariance function is calculated with the default options
    (biased, no fft).
    """
    s = array_like(s, "s")
    nlags = int_like(nlags, "nlags")
    isacov = bool_like(isacov, "isacov")

    order = nlags

    if isacov:
        sxx_m = s
    else:
        sxx_m = acovf(s, fft=False)[: order + 1]  # not tested

    phi = np.zeros((order + 1, order + 1), "d")
    sig = np.zeros(order + 1)
    # initial points for the recursion
    phi[1, 1] = sxx_m[1] / sxx_m[0]
    sig[1] = sxx_m[0] - phi[1, 1] * sxx_m[1]
    for k in range(2, order + 1):
        phi[k, k] = (
            sxx_m[k] - np.dot(phi[1:k, k - 1], sxx_m[1:k][::-1])
        ) / sig[k - 1]
        for j in range(1, k):
            phi[j, k] = phi[j, k - 1] - phi[k, k] * phi[k - j, k - 1]
        sig[k] = sig[k - 1] * (1 - phi[k, k] ** 2)

    sigma_v = sig[-1]
    arcoefs = phi[1:, -1]
    pacf_ = np.diag(phi).copy()
    pacf_[0] = 1.0
    return sigma_v, arcoefs, pacf_, sig, phi  # return everything

def levinson_durbin_pacf(pacf, nlags=None):
    """
    Levinson-Durbin algorithm that returns the acf and ar coefficients.

    Parameters
    ----------
    pacf : array_like
        Partial autocorrelation array for lags 0, 1, ... p.
    nlags : int, optional
        Number of lags in the AR model.  If omitted, returns coefficients from
        an AR(p) and the first p autocorrelations.

    Returns
    -------
    arcoefs : ndarray
        AR coefficients computed from the partial autocorrelations.
    acf : ndarray
        The acf computed from the partial autocorrelations. Array returned
        contains the autocorrelations corresponding to lags 0, 1, ..., p.

    References
    ----------
    .. [1] Brockwell, P.J. and Davis, R.A., 2016. Introduction to time series
        and forecasting. Springer.
    """
    pacf = array_like(pacf, "pacf")
    nlags = int_like(nlags, "nlags", optional=True)
    pacf = np.squeeze(np.asarray(pacf))

    if pacf[0] != 1:
        raise ValueError(
            "The first entry of the pacf corresponds to lags 0 "
            "and so must be 1."
        )
    pacf = pacf[1:]
    n = pacf.shape[0]
    if nlags is not None:
        if nlags > n:
            raise ValueError(
                "Must provide at least as many values from the "
                "pacf as the number of lags."
            )
        pacf = pacf[:nlags]
        n = pacf.shape[0]

    acf = np.zeros(n + 1)
    acf[1] = pacf[0]
    nu = np.cumprod(1 - pacf ** 2)
    arcoefs = pacf.copy()
    for i in range(1, n):
        prev = arcoefs[: -(n - i)].copy()
        arcoefs[: -(n - i)] = prev - arcoefs[i] * prev[::-1]
        acf[i + 1] = arcoefs[i] * nu[i - 1] + prev.dot(acf[1 : -(n - i)][::-1])
    acf[0] = 1
    return arcoefs, acf

def breakvar_heteroskedasticity_test(



    resid, subset_length=1 / 3, alternative="two-sided", use_f=True
):
    r"""
    Test for heteroskedasticity of residuals

    Tests whether the sum-of-squares in the first subset of the sample is
    significantly different than the sum-of-squares in the last subset
    of the sample. Analogous to a Goldfeld-Quandt test. The null hypothesis
    is of no heteroskedasticity.

    Parameters
    ----------
    resid : array_like
        Residuals of a time series model.
        The shape is 1d (nobs,) or 2d (nobs, nvars).
    subset_length : {int, float}
        Length of the subsets to test (h in Notes below).
        If a float in 0 < subset_length < 1, it is interpreted as fraction.
        Default is 1/3.
    alternative : str, 'increasing', 'decreasing' or 'two-sided'
        This specifies the alternative for the p-value calculation. Default
        is two-sided.
    use_f : bool, optional
        Whether or not to compare against the asymptotic distribution
        (chi-squared) or the approximate small-sample distribution (F).
        Default is True (i.e. default is to compare against an F
        distribution).

    Returns
    -------
    test_statistic : {float, ndarray}
        Test statistic(s) H(h).
    p_value : {float, ndarray}
        p-value(s) of test statistic(s).

    Notes
    -----
    The null hypothesis is of no heteroskedasticity. That means different
    things depending on which alternative is selected:

    - Increasing: Null hypothesis is that the variance is not increasing
        throughout the sample; that the sum-of-squares in the later
        subsample is *not* greater than the sum-of-squares in the earlier
        subsample.
    - Decreasing: Null hypothesis is that the variance is not decreasing
        throughout the sample; that the sum-of-squares in the earlier
        subsample is *not* greater than the sum-of-squares in the later
        subsample.
    - Two-sided: Null hypothesis is that the variance is not changing
        throughout the sample. Both that the sum-of-squares in the earlier
        subsample is not greater than the sum-of-squares in the later
        subsample *and* that the sum-of-squares in the later subsample is
        not greater than the sum-of-squares in the earlier subsample.

    For :math:`h = [T/3]`, the test statistic is:

    .. math::

        H(h) = \sum_{t=T-h+1}^T  \tilde v_t^2
        \Bigg / \sum_{t=1}^{h} \tilde v_t^2

    This statistic can be tested against an :math:`F(h,h)` distribution.
    Alternatively, :math:`h H(h)` is asymptotically distributed according
    to :math:`\chi_h^2`; this second test can be applied by passing
    `use_f=False` as an argument.

    See section 5.4 of [1]_ for the above formula and discussion, as well
    as additional details.

    References
    ----------
    .. [1] Harvey, Andrew C. 1990. *Forecasting, Structural Time Series*
            *Models and the Kalman Filter.* Cambridge University Press.
    """
    squared_resid = np.asarray(resid, dtype=float) ** 2
    if squared_resid.ndim == 1:
        squared_resid = squared_resid.reshape(-1, 1)
    nobs = len(resid)

    if 0 < subset_length < 1:
        h = int(np.round(nobs * subset_length))
    elif type(subset_length) is int and subset_length >= 1:
        h = subset_length

    numer_resid = squared_resid[-h:]
    numer_dof = (~np.isnan(numer_resid)).sum(axis=0)
    numer_squared_sum = np.nansum(numer_resid, axis=0)
    for i, dof in enumerate(numer_dof):
        if dof < 2:
            warnings.warn(
                "Early subset of data for variable %d"
                " has too few non-missing observations to"
                " calculate test statistic." % i,
                stacklevel=2,
            )
            numer_squared_sum[i] = np.nan

    denom_resid = squared_resid[:h]
    denom_dof = (~np.isnan(denom_resid)).sum(axis=0)
    denom_squared_sum = np.nansum(denom_resid, axis=0)
    for i, dof in enumerate(denom_dof):
        if dof < 2:
            warnings.warn(
                "Later subset of data for variable %d"
                " has too few non-missing observations to"
                " calculate test statistic." % i,
                stacklevel=2,
            )
            denom_squared_sum[i] = np.nan

    test_statistic = numer_squared_sum / denom_squared_sum

    # Setup functions to calculate the p-values
    if use_f:
        from scipy.stats import f

        pval_lower = lambda test_statistics: f.cdf(  # noqa:E731
            test_statistics, numer_dof, denom_dof
        )
        pval_upper = lambda test_statistics: f.sf(  # noqa:E731
            test_statistics, numer_dof, denom_dof
        )
    else:
        from scipy.stats import chi2

        pval_lower = lambda test_statistics: chi2.cdf(  # noqa:E731
            numer_dof * test_statistics, denom_dof
        )
        pval_upper = lambda test_statistics: chi2.sf(  # noqa:E731
            numer_dof * test_statistics, denom_dof
        )

    # Calculate the one- or two-sided p-values
    alternative = alternative.lower()
    if alternative in ["i", "inc", "increasing"]:
        p_value = pval_upper(test_statistic)
    elif alternative in ["d", "dec", "decreasing"]:
        test_statistic = 1.0 / test_statistic
        p_value = pval_upper(test_statistic)
    elif alternative in ["2", "2-sided", "two-sided"]:
        p_value = 2 * np.minimum(
            pval_lower(test_statistic), pval_upper(test_statistic)
        )
    else:
        raise ValueError("Invalid alternative.")

    if len(test_statistic) == 1:
        return test_statistic[0], p_value[0]

    return test_statistic, p_value

def generate_fourier_surrogate(data):
    
    ts = data.copy()
    ts_fourier = np.fft.rfft(ts)
    random_phases = np.exp(np.random.uniform(0, np.pi, len(ts) // 2 + 1) * 1.0j)
    ts_fourier_new = ts_fourier * random_phases
    surrogate = np.fft.irfft(ts_fourier_new, len(ts))

    return surrogate

def grangercausalitytests(x, maxlag, addconst=True, verbose=None):
    """
    Four tests for granger non causality of 2 time series.

    All four tests give similar results. `params_ftest` and `ssr_ftest` are
    equivalent based on F test which is identical to lmtest:grangertest in R.

    Parameters
    ----------
    x : array_like
        The data for testing whether the time series in the second column Granger
        causes the time series in the first column. Missing values are not
        supported.
    maxlag : {int, Iterable[int]}
        If an integer, computes the test for all lags up to maxlag. If an
        iterable, computes the tests only for the lags in maxlag.
    addconst : bool
        Include a constant in the model.
    verbose : bool
        Print results. Deprecated

        .. deprecated: 0.14

           verbose is deprecated and will be removed after 0.15 is released



    Returns
    -------
    dict
        All test results, dictionary keys are the number of lags. For each
        lag the values are a tuple, with the first element a dictionary with
        test statistic, pvalues, degrees of freedom, the second element are
        the OLS estimation results for the restricted model, the unrestricted
        model and the restriction (contrast) matrix for the parameter f_test.

    Notes
    -----
    TODO: convert to class and attach results properly

    The Null hypothesis for grangercausalitytests is that the time series in
    the second column, x2, does NOT Granger cause the time series in the first
    column, x1. Grange causality means that past values of x2 have a
    statistically significant effect on the current value of x1, taking past
    values of x1 into account as regressors. We reject the null hypothesis
    that x2 does not Granger cause x1 if the pvalues are below a desired size
    of the test.

    The null hypothesis for all four test is that the coefficients
    corresponding to past values of the second time series are zero.

    `params_ftest`, `ssr_ftest` are based on F distribution

    `ssr_chi2test`, `lrtest` are based on chi-square distribution

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Granger_causality

    .. [2] Greene: Econometric Analysis

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from statsmodels.tsa.stattools import grangercausalitytests
    >>> import numpy as np
    >>> data = sm.datasets.macrodata.load_pandas()
    >>> data = data.data[["realgdp", "realcons"]].pct_change().dropna()

    All lags up to 4

    >>> gc_res = grangercausalitytests(data, 4)

    Only lag 4

    >>> gc_res = grangercausalitytests(data, [4])
    """
    x = array_like(x, "x", ndim=2)
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN or inf values.")
    addconst = bool_like(addconst, "addconst")
    if verbose is not None:
        verbose = bool_like(verbose, "verbose")
        warnings.warn(
            "verbose is deprecated since functions should not print results",
            FutureWarning,
        )
    else:
        verbose = True  # old default

    try:
        maxlag = int_like(maxlag, "maxlag")
        if maxlag <= 0:
            raise ValueError("maxlag must be a positive integer")
        lags = np.arange(1, maxlag + 1)
    except TypeError:
        lags = np.array([int(lag) for lag in maxlag])
        maxlag = lags.max()
        if lags.min() <= 0 or lags.size == 0:
            raise ValueError(
                "maxlag must be a non-empty list containing only "
                "positive integers"
            )

    if x.shape[0] <= 3 * maxlag + int(addconst):
        raise ValueError(
            "Insufficient observations. Maximum allowable "
            "lag is {0}".format(int((x.shape[0] - int(addconst)) / 3) - 1)
        )

    resli = {}

    for mlg in lags:
        result = {}
        if verbose:
            print("\nGranger Causality")
            print("number of lags (no zero)", mlg)
        mxlg = mlg

        # create lagmat of both time series
        dta = lagmat2ds(x, mxlg, trim="both", dropex=1)

        # add constant
        if addconst:
            dtaown = add_constant(dta[:, 1 : (mxlg + 1)], prepend=False)
            dtajoint = add_constant(dta[:, 1:], prepend=False)
            if (
                dtajoint.shape[1] == (dta.shape[1] - 1)
                or (dtajoint.max(0) == dtajoint.min(0)).sum() != 1
            ):
                raise InfeasibleTestError(
                    "The x values include a column with constant values and so"
                    " the test statistic cannot be computed."
                )
        else:
            raise NotImplementedError("Not Implemented")
            # dtaown = dta[:, 1:mxlg]
            # dtajoint = dta[:, 1:]

        # Run ols on both models without and with lags of second variable
        res2down = OLS(dta[:, 0], dtaown).fit()
        res2djoint = OLS(dta[:, 0], dtajoint).fit()

        # print results
        # for ssr based tests see:
        # http://support.sas.com/rnd/app/examples/ets/granger/index.htm
        # the other tests are made-up

        # Granger Causality R Squared Ratio Test
        if res2djoint.model.k_constant:
            tss = res2djoint.centered_tss
        else:
            tss = res2djoint.uncentered_tss
        if (
            tss == 0
            or res2djoint.ssr == 0
            or np.isnan(res2djoint.rsquared)
            or (res2djoint.ssr / tss) < np.finfo(float).eps
            or res2djoint.params.shape[0] != dtajoint.shape[1]
        ):
            raise InfeasibleTestError(
                "The Granger causality test statistic cannot be compute "
                "because the VAR has a perfect fit of the data."
            )  
        default = 1e-4
        ave_score = True
        
        rse1 = res2djoint.rsquared
        rse2 = res2down.rsquared

        if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=default, atol=default)):
            rse2 = default
            rse1 = rse2
            ave_score = False
        if rse2 < 0:
            rse2 = default
            ave_score = False

        if ave_score:
            fgc0 = ((rse1 - rse2) / (0.5 * np.abs(rse1 + rse2)))
        else:
            fgc0 = ((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2)))
            
        if verbose:
            print(
                "ssr based Ratio test:         F=%-8.4f, p=%-8.4f, df_denom=%d,"
                " df_num=%d"
                % (
                    fgc0,
                    fgc0,
                    rse1,
                    rse2,
                )
            )
        result["ssr_ratio"] = (
            fgc0,
            fgc0,
            rse1,
            rse2,
        )

        # Granger Causality test using ssr (F statistic)
        if res2djoint.model.k_constant:
            tss = res2djoint.centered_tss
        else:
            tss = res2djoint.uncentered_tss
        if (
            tss == 0
            or res2djoint.ssr == 0
            or np.isnan(res2djoint.rsquared)
            or (res2djoint.ssr / tss) < np.finfo(float).eps
            or res2djoint.params.shape[0] != dtajoint.shape[1]
        ):
            raise InfeasibleTestError(
                "The Granger causality test statistic cannot be compute "
                "because the VAR has a perfect fit of the data."
            )
        fgc1 = (
            (res2down.ssr - res2djoint.ssr)
            / res2djoint.ssr
            / mxlg
            * res2djoint.df_resid
        )
        if verbose:
            print(
                "ssr based F test:         F=%-8.4f, p=%-8.4f, df_denom=%d,"
                " df_num=%d"
                % (
                    fgc1,
                    stats.f.sf(fgc1, mxlg, res2djoint.df_resid),
                    res2djoint.df_resid,
                    mxlg,
                )
            )
        result["ssr_ftest"] = (
            fgc1,
            stats.f.sf(fgc1, mxlg, res2djoint.df_resid),
            res2djoint.df_resid,
            mxlg,
        )

        # Granger Causality test using ssr (ch2 statistic)
        fgc2 = res2down.nobs * (res2down.ssr - res2djoint.ssr) / res2djoint.ssr
        if verbose:
            print(
                "ssr based chi2 test:   chi2=%-8.4f, p=%-8.4f, "
                "df=%d" % (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg)
            )
        result["ssr_chi2test"] = (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg)

        # likelihood ratio test pvalue:
        lr = -2 * (res2down.llf - res2djoint.llf)
        if verbose:
            print(
                "likelihood ratio test: chi2=%-8.4f, p=%-8.4f, df=%d"
                % (lr, stats.chi2.sf(lr, mxlg), mxlg)
            )
        result["lrtest"] = (lr, stats.chi2.sf(lr, mxlg), mxlg)

        # F test that all lag coefficients of exog are zero
        rconstr = np.column_stack(
            (np.zeros((mxlg, mxlg)), np.eye(mxlg, mxlg), np.zeros((mxlg, 1)))
        )
        ftres = res2djoint.f_test(rconstr)
        if verbose:
            print(
                "parameter F test:         F=%-8.4f, p=%-8.4f, df_denom=%d,"
                " df_num=%d"
                % (ftres.fvalue, ftres.pvalue, ftres.df_denom, ftres.df_num)
            )
        result["params_ftest"] = (
            np.squeeze(ftres.fvalue)[()],
            np.squeeze(ftres.pvalue)[()],
            ftres.df_denom,
            ftres.df_num,
        )

        resli[mxlg] = (result, [res2down, res2djoint, rconstr])

    return resli

def grangercausalitytestssurr(x, x_surr=[], maxlag=4, addconst=True, verbose=None):

    x = array_like(x, "x", ndim=2)
    if not np.isfinite(x).all():
        raise ValueError("x contains NaN or inf values.")
    addconst = bool_like(addconst, "addconst")
    # if verbose is not None:
    #     verbose = bool_like(verbose, "verbose")
    #     warnings.warn(
    #         "verbose is deprecated since functions should not print results",
    #         FutureWarning,
    #     )
    # else:
    #     verbose = True  # old default

    try:
        maxlag = int_like(maxlag, "maxlag")
        if maxlag <= 0:
            raise ValueError("maxlag must be a positive integer")
        lags = np.arange(1, maxlag + 1)
    except TypeError:
        lags = np.array([int(lag) for lag in maxlag])
        maxlag = lags.max()
        if lags.min() <= 0 or lags.size == 0:
            raise ValueError(
                "maxlag must be a non-empty list containing only "
                "positive integers"
            )

    if x.shape[0] <= 3 * maxlag + int(addconst):
        raise ValueError(
            "Insufficient observations. Maximum allowable "
            "lag is {0}".format(int((x.shape[0] - int(addconst)) / 3) - 1)
        )

    resli = {}

    for mlg in lags:
        result = {}
        if verbose:
            print("\nGranger Causality")
            print("number of lags (no zero)", mlg)
        mxlg = mlg

        # create lagmat of both time series
        dta = lagmat2ds(x, mxlg, trim="both", dropex=1)
        if len(x_surr)>0:
            dta_surr = lagmat2ds(x_surr, mxlg, trim="both", dropex=1)

        # add constant
        if addconst:
            if len(x_surr)>0:
                dtaown = add_constant(dta_surr[:, 1:], prepend=False)
            else:
                dtaown = add_constant(dta[:, 1 : (mxlg + 1)], prepend=False)
            dtajoint = add_constant(dta[:, 1:], prepend=False)
            if (
                dtajoint.shape[1] == (dta.shape[1] - 1)
                or (dtajoint.max(0) == dtajoint.min(0)).sum() != 1
            ):
                raise InfeasibleTestError(
                    "The x values include a column with constant values and so"
                    " the test statistic cannot be computed."
                )
        else:
            raise NotImplementedError("Not Implemented")
            # dtaown = dta[:, 1:mxlg]
            # dtajoint = dta[:, 1:]

        # Run ols on both models without and with lags of second variable
        res2down = OLS(dta[:, 0], dtaown).fit()
        res2djoint = OLS(dta[:, 0], dtajoint).fit()

        # print results
        # for ssr based tests see:
        # http://support.sas.com/rnd/app/examples/ets/granger/index.htm
        # the other tests are made-up

        
        # Granger Causality R Squared Ratio Test
        if res2djoint.model.k_constant:
            tss = res2djoint.centered_tss
        else:
            tss = res2djoint.uncentered_tss
        if (
            tss == 0
            or res2djoint.ssr == 0
            or np.isnan(res2djoint.rsquared)
            or (res2djoint.ssr / tss) < np.finfo(float).eps
            or res2djoint.params.shape[0] != dtajoint.shape[1]
        ):
            raise InfeasibleTestError(
                "The Granger causality test statistic cannot be compute "
                "because the VAR has a perfect fit of the data."
            )  
        default = 1e-10
        ave_score = True
        
        rse1 = res2djoint.rsquared
        rse2 = res2down.rsquared

        if (rse1 < 0) or (rse1 < rse2) or (np.allclose(rse1, rse2, rtol=default, atol=default)):
            rse2 = default
            rse1 = rse2
            ave_score = False
        if rse2 < 0:
            rse2 = default
            ave_score = False

        if ave_score:
            fgc0 = ((rse1 - rse2) / (0.5 * np.abs(rse1 + rse2)))
        else:
            fgc0 = ((rse1 - rse2) / (1.0 * np.abs(rse1 + rse2)))
            
        if verbose:
            print(
                "ssr based Ratio test:         F=%-8.4f, p=%-8.4f, df_denom=%d,"
                " df_num=%d"
                % (
                    fgc0,
                    fgc0,
                    rse1,
                    rse2,
                )
            )
        result["ssr_ratio"] = (
            fgc0,
            fgc0,
            rse1,
            rse2,
        )
        
        # Granger Causality test using ssr (F statistic)
        if res2djoint.model.k_constant:
            tss = res2djoint.centered_tss
        else:
            tss = res2djoint.uncentered_tss
        if (
            tss == 0
            or res2djoint.ssr == 0
            or np.isnan(res2djoint.rsquared)
            or (res2djoint.ssr / tss) < np.finfo(float).eps
            or res2djoint.params.shape[0] != dtajoint.shape[1]
        ):
            raise InfeasibleTestError(
                "The Granger causality test statistic cannot be compute "
                "because the VAR has a perfect fit of the data."
            )
        fgc1 = (
            (res2down.ssr - res2djoint.ssr)
            / res2djoint.ssr
            / mxlg
            * res2djoint.df_resid
        )
        if verbose:
            print(
                "ssr based F test:         F=%-8.4f, p=%-8.4f, df_denom=%d,"
                " df_num=%d"
                % (
                    fgc1,
                    stats.f.sf(fgc1, mxlg, res2djoint.df_resid),
                    res2djoint.df_resid,
                    mxlg,
                )
            )
        result["ssr_ftest"] = (
            fgc1,
            stats.f.sf(fgc1, mxlg, res2djoint.df_resid),
            res2djoint.df_resid,
            mxlg,
        )

        # Granger Causality test using ssr (ch2 statistic)
        fgc2 = res2down.nobs * (res2down.ssr - res2djoint.ssr) / res2djoint.ssr
        if verbose:
            print(
                "ssr based chi2 test:   chi2=%-8.4f, p=%-8.4f, "
                "df=%d" % (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg)
            )
        result["ssr_chi2test"] = (fgc2, stats.chi2.sf(fgc2, mxlg), mxlg)

        # likelihood ratio test pvalue:
        lr = -2 * (res2down.llf - res2djoint.llf)
        if verbose:
            print(
                "likelihood ratio test: chi2=%-8.4f, p=%-8.4f, df=%d"
                % (lr, stats.chi2.sf(lr, mxlg), mxlg)
            )
        result["lrtest"] = (lr, stats.chi2.sf(lr, mxlg), mxlg)

        # F test that all lag coefficients of exog are zero
        rconstr = np.column_stack(
            (np.zeros((mxlg, mxlg)), np.eye(mxlg, mxlg), np.zeros((mxlg, 1)))
        )
        ftres = res2djoint.f_test(rconstr)
        if verbose:
            print(
                "parameter F test:         F=%-8.4f, p=%-8.4f, df_denom=%d,"
                " df_num=%d"
                % (ftres.fvalue, ftres.pvalue, ftres.df_denom, ftres.df_num)
            )
        result["params_ftest"] = (
            np.squeeze(ftres.fvalue)[()],
            np.squeeze(ftres.pvalue)[()],
            ftres.df_denom,
            ftres.df_num,
        )

        resli[mxlg] = (result, [res2down, res2djoint, rconstr])
    
    return resli

def granger_pairwise(variables, var1, var2, maxlag=100, test='ssr_chi2test', surrogate=False, randomized=False):
        
        x_orig = variables[:, var1]
        y_orig = variables[:, var2] 

        x = np.column_stack([x_orig, y_orig])
        x_surr=[]

        if randomized:
            surrogate=False
            y_surr = np.zeros(y_orig.shape)
            x_surr = np.column_stack([x_orig, y_surr])
            
        if surrogate:
            y_surr = generate_fourier_surrogate(y_orig)
            x_surr = np.column_stack([x_orig, y_surr])

        if (randomized or surrogate) == False:
            test_result = grangercausalitytests(x, maxlag=maxlag, verbose=False)
        else:            
            test_result = grangercausalitytestssurr(x, x_surr, maxlag=maxlag, verbose=False)
        
        if test == 'ssr_ratio':
            if len(test_result) == 1:
                p_values = round(test_result[maxlag[0]][0][test][1],4)
                ave_p_value = p_values
            else:    
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                ave_p_value = np.mean(p_values)
            # if ave_p_value>0.0:
            #     score_x = 1
            # else: 
            #     score_x = 0
            score_x = ave_p_value
        else:
            if len(test_result) == 1:
                p_values = round(test_result[maxlag[0]][0][test][1],4)
                min_p_value = p_values
            else:
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                min_p_value = np.min(p_values)
            if min_p_value<0.05:
                score_x = 1
            else: 
                score_x = 0
            
        x_orig = variables[:, var2]
        y_orig = variables[:, var1] 

        x = np.column_stack([x_orig, y_orig])
        x_surr=[]

        if randomized:
            surrogate=False
            y_surr = np.zeros(y_orig.shape)
            x_surr = np.column_stack([x_orig, y_surr])
            
        if surrogate:
            y_surr = generate_fourier_surrogate(y_orig)
            x_surr = np.column_stack([x_orig, y_surr])
        
        if (randomized and surrogate) == False:
            test_result = grangercausalitytests(x, maxlag=maxlag, verbose=False)
        else:            
            test_result = grangercausalitytestssurr(x, x_surr, maxlag=maxlag, verbose=False)

        if test == 'ssr_ratio':
            if len(test_result) == 1:
                p_values = round(test_result[maxlag[0]][0][test][1],4)
                ave_p_value = p_values
            else:    
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                ave_p_value = np.mean(p_values)
            # if ave_p_value>0.0:
            #     score_y = 1
            # else: 
            #     score_y = 0
            score_y = ave_p_value
        else:
            if len(test_result) == 1:
                p_values = round(test_result[maxlag[0]][0][test][1],4)
                min_p_value = p_values
            else:
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                min_p_value = np.min(p_values)
            if min_p_value<0.05:
                score_y = 1
            else: 
                score_y = 0
        
        return score_x, score_y

def granger_temporal_bidirectional(variables, maxlag=100, test='ssr_chi2test', surrogate=False, randomized=False):
        
        x_orig = variables[:, 0]
        y_orig = variables[:, 1] 

        x = np.column_stack([x_orig, y_orig])

        if (randomized or surrogate):
            y_surr = variables[:, 3] 
            x_surr = np.column_stack([x_orig, y_surr])
            test_result = grangercausalitytestssurr(x, x_surr, maxlag=maxlag, verbose=False)
        else:       
            test_result = grangercausalitytests(x, maxlag=maxlag, verbose=False)

        if test == 'ssr_ratio':
            if len(test_result) == 1:
                p_values = round(test_result[maxlag[0]][0][test][1],4)
                ave_p_value = p_values
            else:    
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                ave_p_value = np.mean(p_values)
            # if ave_p_value>0.0:
            #     score_x = 1
            # else: 
            #     score_x = 0
            score_x = ave_p_value
        else:
            if len(test_result) == 1:
                p_values = round(test_result[maxlag[0]][0][test][1],4)
                min_p_value = p_values
            else:
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                min_p_value = np.min(p_values)
            if min_p_value<0.05:
                score_x = 1
            else: 
                score_x = 0
            
        x_orig = variables[:, 1]
        y_orig = variables[:, 0] 

        x = np.column_stack([x_orig, y_orig])

        if (randomized or surrogate):
            y_surr = variables[:, 2] 
            x_surr = np.column_stack([x_orig, y_surr])
            test_result = grangercausalitytestssurr(x, x_surr, maxlag=maxlag, verbose=False)
        else:   
            test_result = grangercausalitytests(x, maxlag=maxlag, verbose=False)

        if test == 'ssr_ratio':
            if len(test_result) == 1:
                p_values = round(test_result[maxlag[0]][0][test][1],4)
                ave_p_value = p_values
            else:    
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                ave_p_value = np.mean(p_values)
            # if ave_p_value>0.0:
            #     score_y = 1
            # else: 
            #     score_y = 0
            score_y = ave_p_value
        else:
            if len(test_result) == 1:
                p_values = round(test_result[maxlag[0]][0][test][1],4)
                min_p_value = p_values
            else:
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                min_p_value = np.min(p_values)
            if min_p_value<0.05:
                score_y = 1
            else: 
                score_y = 0
        
        return score_x, score_y

def granger_temporal_unidirectional(variables, maxlag=100, test='ssr_chi2test', surrogate=False, randomized=False):
        
        x_orig = variables[:, 0]
        y_orig = variables[:, 1] 

        x = np.column_stack([x_orig, y_orig])

        if (randomized or surrogate):
            y_surr = variables[:, 3] 
            x_surr = np.column_stack([x_orig, y_surr])
            test_result = grangercausalitytestssurr(x, x_surr, maxlag=maxlag, verbose=False)
        else:       
            test_result = grangercausalitytests(x, maxlag=maxlag, verbose=False)
        
        if test == 'ssr_ratio':
            if len(test_result) == 1:
                p_values = round(test_result[maxlag[0]][0][test][1],4)
                ave_p_value = p_values
            else:    
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                ave_p_value = np.mean(p_values)
            # if ave_p_value>0.0:
            #     score_x = 1
            # else: 
            #     score_x = 0
            score_x = ave_p_value
        else:
            if len(test_result) == 1:
                p_values = round(test_result[maxlag[0]][0][test][1],4)
                min_p_value = p_values
            else:
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                min_p_value = np.min(p_values)
            if min_p_value<0.05:
                score_x = 1
            else: 
                score_x = 0
        
        return score_x