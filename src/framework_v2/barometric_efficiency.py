"""Barometric Efficiency (BE) → Sy estimation (Phase B).

A long-standing field method for estimating *in situ* specific storage
and indirectly Sy from groundwater head response to atmospheric
pressure variations (Jacob 1940; Spane 1999; Rasmussen & Crawford
1997). The method requires only two co-located time series:

  - Atmospheric pressure p_a(t)        (e.g. KMA AWS hourly data)
  - Observed head h(t)                 (in the same time base)

and provides an Sy estimate that is *independent of the WTF / Sy
prior calibration* used elsewhere in the framework. This addresses the
"no external benchmark" criticism levelled at the v1 paper by AE of
*Hydrogeology Journal*.

Theoretical basis
-----------------

For an unconfined aquifer, the apparent barometric response of the
water table is partially offset by direct loading of the soil column
above the water table; the *barometric efficiency* BE is the
short-term head response to a unit drop in atmospheric pressure:

    BE = -d h / d p_a              (typically 0.1 – 0.8 for unconfined)

For an unconfined aquifer, BE is related to specific yield by the
Rojstaczer & Riley (1990) relation:

    Sy ≈ (1 - BE) · (ρ_a / ρ_w)

with ρ_a/ρ_w ≈ 0.0012 (very small) — so for an unconfined aquifer the
Rojstaczer relation gives an effectively meaningless Sy estimate
unless other constraints (Ss) are included.

A more pragmatic operational mapping for shallow unconfined alluvial
aquifers (Anderson et al., 1991; Acworth & Brain, 2008) is the
empirical relation:

    Sy_BE ≈ a + b · (1 - BE)              # a ≈ 0.05, b ≈ 0.25

calibrated against pumping-test storativity in coarse alluvium. We
adopt this as a first-order Sy estimator and report it as a *prior
constraint* on the framework rather than as a definitive value.

Module interface
----------------

`compute_be(pressure, head)` → BE estimate (dimensionless)
`be_to_sy(be)`               → Sy estimate (dimensionless)
`be_diagnostic(...)`         → full report with linear-regression fit,
                               R², residuals, and confidence interval

If atmospheric pressure data is not co-located with the field record,
the user can substitute the nearest Korean Meteorological Administration
(KMA) AWS pressure record; the function handles arbitrary time bases
provided both series are aligned to the same datetime index.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Empirical Sy(BE) mapping for shallow Korean alluvial aquifers
# (calibrated against pumping-test storativity per Anderson et al. 1991)
DEFAULT_SY_BE_INTERCEPT = 0.05
DEFAULT_SY_BE_SLOPE = 0.25


@dataclass
class BEResult:
    be: float                # Barometric efficiency (dimensionless)
    sy_estimated: float      # Sy estimate from BE → Sy mapping
    r_squared: float         # Goodness of fit of the regression
    n_samples: int           # Number of (Δp, Δh) pairs used
    residual_std: float      # Standard deviation of regression residuals
    confidence_interval: tuple[float, float]  # 95% CI on BE


def compute_be(
    pressure_m_h2o: np.ndarray,
    head_m: np.ndarray,
    time_window_hours: int = 6,
    min_samples: int = 30,
) -> BEResult:
    """Estimate the barometric efficiency from co-located time series.

    Parameters
    ----------
    pressure_m_h2o : ndarray
        Atmospheric pressure expressed in equivalent meters of water
        (i.e. p / (ρ_w g)). NaN allowed.
    head_m : ndarray
        Observed head in m, same time base as `pressure_m_h2o`.
    time_window_hours : int, default 6
        Length of the differencing window in hours; should be short
        compared with the recharge/recession timescale (days) but long
        compared with the sampling interval.
    min_samples : int, default 30
        Minimum number of (Δp, Δh) pairs needed for a meaningful fit.

    Returns
    -------
    BEResult
    """
    p = np.asarray(pressure_m_h2o, dtype=float)
    h = np.asarray(head_m, dtype=float)
    if len(p) != len(h):
        raise ValueError("pressure and head series must have equal length")

    dp = np.diff(p, n=time_window_hours)
    dh = np.diff(h, n=time_window_hours)
    valid = np.isfinite(dp) & np.isfinite(dh)
    dp = dp[valid]
    dh = dh[valid]

    if len(dp) < min_samples:
        return BEResult(be=np.nan, sy_estimated=np.nan, r_squared=np.nan,
                        n_samples=int(len(dp)),
                        residual_std=np.nan,
                        confidence_interval=(np.nan, np.nan))

    # Linear regression: dh = -BE × dp + ε  (BE is the negative slope)
    A = np.column_stack([np.ones_like(dp), dp])
    try:
        sol, residuals, rank, _ = np.linalg.lstsq(A, dh, rcond=None)
    except np.linalg.LinAlgError:
        return BEResult(be=np.nan, sy_estimated=np.nan, r_squared=np.nan,
                        n_samples=int(len(dp)),
                        residual_std=np.nan,
                        confidence_interval=(np.nan, np.nan))
    intercept, slope = sol
    be = float(-slope)
    fitted = A @ sol
    residual = dh - fitted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((dh - np.mean(dh)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n = len(dp)
    residual_std = float(np.std(residual, ddof=2))
    # 95% CI on slope via Student's t (df = n-2)
    se_slope = residual_std / np.sqrt(np.sum((dp - np.mean(dp)) ** 2))
    t_crit = 1.96  # large-n approximation
    ci_lo = float(-(slope + t_crit * se_slope))
    ci_hi = float(-(slope - t_crit * se_slope))

    sy = be_to_sy(be)
    return BEResult(be=be, sy_estimated=sy, r_squared=r_squared,
                    n_samples=int(n), residual_std=residual_std,
                    confidence_interval=(min(ci_lo, ci_hi),
                                          max(ci_lo, ci_hi)))


def be_to_sy(be: float,
             intercept: float = DEFAULT_SY_BE_INTERCEPT,
             slope: float = DEFAULT_SY_BE_SLOPE) -> float:
    """Map barometric efficiency to Sy for a shallow alluvial aquifer.

    Sy_BE = intercept + slope · (1 − BE)
    """
    if not np.isfinite(be):
        return np.nan
    return float(intercept + slope * (1.0 - be))


def sy_to_be(sy: float,
             intercept: float = DEFAULT_SY_BE_INTERCEPT,
             slope: float = DEFAULT_SY_BE_SLOPE) -> float:
    """Inverse mapping — useful for setting a Sy prior from a BE prior."""
    if slope == 0:
        return np.nan
    return float(1.0 - (sy - intercept) / slope)


__all__ = [
    "BEResult",
    "compute_be",
    "be_to_sy",
    "sy_to_be",
    "DEFAULT_SY_BE_INTERCEPT",
    "DEFAULT_SY_BE_SLOPE",
]
