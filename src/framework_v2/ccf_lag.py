"""Cross-correlation lag identification (Phase A-1).

In the v1 paper version the lag was a free variable in the
Nelder-Mead grid search. The optimiser absorbed it into (k, z), so
τ_rmse converged to 0 for every Siheung well and the ablation showed
no measurable effect of the lag identification module. This module
fixes the lag *before* parameter optimisation by computing the
cross-correlation peak between rainfall and the head increment dh/dt,
which is the physical phase delay (Cuthbert et al., 2019). The
optimisation then varies only (k, z) — or (k, z, Sy) when used with
the Bayesian Sy prior of `sy_prior.py`.
"""
from __future__ import annotations

import numpy as np


def ccf_peak_lag(
    rain_m: np.ndarray,
    gw_m: np.ndarray,
    max_lag_days: int = 30,
    min_samples: int = 30,
) -> int:
    """Return the lag (days, ≥ 0) at which the cross-correlation between
    daily rainfall and the head increment dh/dt is maximised.

    NaN values are replaced with 0 in both signals before computing the
    correlation, which is appropriate for sparse rainfall (most days are
    dry → 0 is the natural value).

    Parameters
    ----------
    rain_m : ndarray of shape (n,)
        Daily rainfall (m/day or any consistent unit).
    gw_m : ndarray of shape (n,)
        Daily observed head (m). NaN allowed.
    max_lag_days : int, default 30
        Upper bound of the lag search.
    min_samples : int, default 30
        Minimum sample size required at a given lag for the correlation
        to be considered reliable.

    Returns
    -------
    lag_days : int
        The lag (in days) with the maximum Pearson correlation. Always
        ≥ 0; returns 0 if no positive lag improves on lag = 0.
    """
    rain = np.asarray(rain_m, dtype=float)
    gw = np.asarray(gw_m, dtype=float)

    dh = np.diff(gw)
    rain_aligned = rain[1:]
    dh = np.where(np.isnan(dh), 0.0, dh)
    rain_aligned = np.where(np.isnan(rain_aligned), 0.0, rain_aligned)

    best_lag = 0
    best_cc = -np.inf

    for lag in range(0, max_lag_days + 1):
        if lag == 0:
            r, d = rain_aligned, dh
        else:
            r = rain_aligned[:-lag]
            d = dh[lag:]
        if len(r) < min_samples:
            continue
        # Pearson correlation; treat constant arrays as no signal
        if np.std(r) == 0 or np.std(d) == 0:
            continue
        cc = float(np.corrcoef(r, d)[0, 1])
        if np.isnan(cc):
            continue
        if cc > best_cc:
            best_cc = cc
            best_lag = lag

    return best_lag


def ccf_curve(
    rain_m: np.ndarray,
    gw_m: np.ndarray,
    max_lag_days: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the full cross-correlation curve for diagnostic plotting.

    Returns
    -------
    lags : ndarray
        Lag values from 0 to max_lag_days inclusive.
    correlations : ndarray
        Pearson correlation at each lag (NaN where invalid).
    """
    rain = np.asarray(rain_m, dtype=float)
    gw = np.asarray(gw_m, dtype=float)

    dh = np.diff(gw)
    rain_aligned = rain[1:]
    dh = np.where(np.isnan(dh), 0.0, dh)
    rain_aligned = np.where(np.isnan(rain_aligned), 0.0, rain_aligned)

    lags = np.arange(0, max_lag_days + 1)
    correlations = np.full_like(lags, np.nan, dtype=float)
    for i, lag in enumerate(lags):
        if lag == 0:
            r, d = rain_aligned, dh
        else:
            r = rain_aligned[:-lag]
            d = dh[lag:]
        if len(r) < 30 or np.std(r) == 0 or np.std(d) == 0:
            continue
        cc = float(np.corrcoef(r, d)[0, 1])
        correlations[i] = cc

    return lags, correlations


__all__ = ["ccf_peak_lag", "ccf_curve"]
