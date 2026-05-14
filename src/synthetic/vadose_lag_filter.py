"""Linear-reservoir vadose-zone delay filter.

Maps a daily rainfall input series to an effective recharge series with a
prescribed mean travel time. This imposes a *known ground-truth lag* that the
WTF framework's lag-identification step (cross-correlation of rainfall and
groundwater response) should recover.

The model is a discrete linear reservoir (exponential impulse response):

    R(t) = recharge_fraction * sum_{k=0}^{∞} w_k · P(t - k),

with weights w_k = (1 / tau_lag) · exp(-k / tau_lag), normalised so that
sum_k w_k = 1. The first-moment travel time of the response is `tau_lag`
days. This is a deliberately simple model: its purpose is to provide a
single, identifiable lag parameter that downstream evaluation can compare
against the truth.

A small "field-capacity" buffer is optionally subtracted from the input
before convolution to mimic vadose-zone storage. By default this buffer is
zero, keeping the model linear; non-zero buffers introduce a mild
event-size threshold.
"""
from __future__ import annotations

import numpy as np


def _impulse_response(tau_lag_days: float, n_terms: int) -> np.ndarray:
    """Discrete exponential impulse response, normalised to unit sum."""
    if tau_lag_days <= 0:
        # Zero-lag = identity response
        w = np.zeros(max(1, n_terms))
        w[0] = 1.0
        return w
    k = np.arange(n_terms)
    w = (1.0 / tau_lag_days) * np.exp(-k / tau_lag_days)
    return w / w.sum()


def apply_lag_filter(
    rainfall_mm: np.ndarray,
    tau_lag_days: float,
    recharge_fraction: float = 0.12,
    field_capacity_mm: float = 0.0,
    n_kernel_terms: int | None = None,
) -> np.ndarray:
    """Apply a linear-reservoir lag filter to a daily rainfall series.

    Parameters
    ----------
    rainfall_mm : ndarray of shape (n_days,)
        Daily rainfall input in millimetres.
    tau_lag_days : float
        First-moment travel time of the impulse response in days. The
        WTF framework should recover this value (up to discretisation
        error) via cross-correlation of rainfall and groundwater response.
    recharge_fraction : float, default 0.12
        Fraction of rainfall that ultimately becomes recharge (steady-state
        gain). Plausible for inland Korean alluvium under HSG-C/D dominance.
    field_capacity_mm : float, default 0.0
        Per-event soil-moisture deficit absorbed before percolation. Zero
        keeps the filter linear; small positive values (~3-5 mm) introduce
        a mild event-size threshold.
    n_kernel_terms : int or None
        Number of impulse-response terms to keep. Defaults to a length
        sufficient to capture 99.9% of the response energy.

    Returns
    -------
    recharge_mm : ndarray of shape (n_days,)
        Effective recharge time series in millimetres per day, lagged
        relative to rainfall by `tau_lag_days`.
    """
    rainfall = np.asarray(rainfall_mm, dtype=float)

    if field_capacity_mm > 0:
        rainfall = np.maximum(rainfall - field_capacity_mm, 0.0)

    if n_kernel_terms is None:
        # Capture ~99.9% of exponential mass
        n_kernel_terms = int(np.ceil(8.0 * max(tau_lag_days, 1.0))) + 1

    kernel = _impulse_response(tau_lag_days, n_kernel_terms)

    # Convolve (causal: only past rainfall affects current recharge)
    convolved = np.convolve(rainfall, kernel, mode="full")[: len(rainfall)]

    return recharge_fraction * convolved


__all__ = ["apply_lag_filter"]
