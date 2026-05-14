"""Stochastic Korean-monsoon rainfall generator.

Generates daily rainfall time series with seasonality patterns characteristic
of the Korean peninsula (East Asian summer monsoon, June–September concentration).

The generator uses a two-state hidden process (wet/dry days) modulated by a
seasonal occurrence probability and a magnitude distribution. This is sufficient
for synthetic-truth benchmarks of WTF estimators where the recharge response is
the validation target, not the precipitation generation realism itself.
"""
from __future__ import annotations

import numpy as np


# Monthly occurrence probability of a wet day (climatology, Korean peninsula)
# Source: rough fit to Korea Meteorological Administration ASOS records,
# 1991–2020 normals for inland stations.
_MONTHLY_WET_PROB = np.array([
    0.20,  # Jan
    0.21,  # Feb
    0.25,  # Mar
    0.30,  # Apr
    0.35,  # May
    0.45,  # Jun  ← monsoon onset
    0.55,  # Jul  ← peak monsoon
    0.50,  # Aug
    0.35,  # Sep
    0.20,  # Oct
    0.20,  # Nov
    0.20,  # Dec
])

# Mean wet-day rainfall (mm) by month, log-normal scale parameter
# Tuned so that annual total ~950 mm matches Yeongcheon-class inland sites.
_MONTHLY_WET_MEAN_MM = np.array([
    3.0,   # Jan
    4.0,   # Feb
    5.0,   # Mar
    6.0,   # Apr
    8.0,   # May
    12.0,  # Jun
    18.0,  # Jul  ← heaviest wet days
    16.0,  # Aug
    10.0,  # Sep
    5.0,   # Oct
    4.0,   # Nov
    3.0,   # Dec
])

_MONTHLY_WET_SIGMA_LOG = 0.9  # log-normal sigma (heavy tail)


def _day_of_year_to_month(doy: np.ndarray) -> np.ndarray:
    """Map 1-based day-of-year (1..365 or 1..366) to 0-based month index."""
    # Cumulative non-leap month boundaries: J F M A M J J A S O N D
    cum = np.array([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
    months = np.searchsorted(cum, doy - 1, side="right")
    return np.clip(months, 0, 11)


def generate_rainfall(
    n_days: int,
    start_day_of_year: int = 1,
    annual_total_mm: float | None = 950.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a daily rainfall time series.

    Parameters
    ----------
    n_days : int
        Number of days to simulate.
    start_day_of_year : int, default 1
        Day-of-year of the first sample (1 = 1 Jan).
    annual_total_mm : float or None, default 950.0
        If given, rescales the output so that the *expected* annual total
        matches this value. Passing ``None`` leaves the raw stochastic
        magnitudes unmodified.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    rainfall : ndarray of shape (n_days,)
        Daily rainfall amounts in millimetres.

    Notes
    -----
    The output is *expected* to follow Korean monsoon climatology in a
    coarse sense (wet-season concentration in June–September). It is *not*
    intended as a high-fidelity weather generator; its purpose is to provide
    reproducible, plausible forcing for synthetic-truth benchmarks of WTF
    estimators.
    """
    rng = np.random.default_rng(seed)

    doy = ((np.arange(n_days) + start_day_of_year - 1) % 365) + 1
    months = _day_of_year_to_month(doy)

    wet_prob = _MONTHLY_WET_PROB[months]
    wet_mean = _MONTHLY_WET_MEAN_MM[months]

    is_wet = rng.uniform(size=n_days) < wet_prob

    # Log-normal magnitude for wet days
    mu_log = np.log(wet_mean) - 0.5 * _MONTHLY_WET_SIGMA_LOG ** 2
    magnitudes = rng.lognormal(
        mean=mu_log,
        sigma=_MONTHLY_WET_SIGMA_LOG,
        size=n_days,
    )

    rainfall = np.where(is_wet, magnitudes, 0.0)

    # Rescale to target annual total (preserves day-to-day stochastic pattern)
    if annual_total_mm is not None and n_days > 0:
        n_years = n_days / 365.0
        target = annual_total_mm * n_years
        current = rainfall.sum()
        if current > 0:
            rainfall *= target / current

    return rainfall


__all__ = ["generate_rainfall"]
