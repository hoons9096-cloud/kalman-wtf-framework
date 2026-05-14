"""Observation step: sensor noise, occasional outliers, and random data gaps.

Takes a "physically true" groundwater head series (including pumping
disturbance, if any) and produces a *measured* series with:

1. Gaussian sensor noise of fixed standard deviation
2. Occasional large-magnitude outliers (e.g. transducer spikes)
3. Random missing values (NaN) representing sensor downtime
"""
from __future__ import annotations

import numpy as np


def apply_observation_process(
    h_signal_m: np.ndarray,
    sensor_noise_std_m: float = 0.02,
    outlier_prob: float = 0.005,
    outlier_magnitude_m: float = 0.5,
    gap_prob: float = 0.02,
    seed: int | None = None,
) -> np.ndarray:
    """Apply sensor noise + outliers + random gaps to a head series.

    Parameters
    ----------
    h_signal_m : ndarray of shape (n_days,)
        Underlying physical head series (may already include pumping).
    sensor_noise_std_m : float, default 0.02
        Standard deviation of zero-mean Gaussian sensor noise (m).
    outlier_prob : float, default 0.005
        Probability that a given day's measurement is replaced by an
        outlier of magnitude ±`outlier_magnitude_m` (Bernoulli).
    outlier_magnitude_m : float, default 0.5
        Magnitude of injected outliers (sign uniform-random).
    gap_prob : float, default 0.02
        Probability that a given day's measurement is missing (NaN).
        Independent of outliers.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    h_obs_m : ndarray of shape (n_days,)
        Observed head series with noise, outliers, and NaN gaps.
    """
    rng = np.random.default_rng(seed)
    h_obs = np.asarray(h_signal_m, dtype=float).copy()

    # Sensor noise
    if sensor_noise_std_m > 0:
        h_obs += rng.normal(0.0, sensor_noise_std_m, size=h_obs.shape)

    # Outliers (replace with signed spike)
    if outlier_prob > 0:
        is_outlier = rng.uniform(size=h_obs.shape) < outlier_prob
        sign = rng.choice([-1.0, 1.0], size=h_obs.shape)
        h_obs[is_outlier] += sign[is_outlier] * outlier_magnitude_m

    # Gaps
    if gap_prob > 0:
        is_gap = rng.uniform(size=h_obs.shape) < gap_prob
        h_obs[is_gap] = np.nan

    return h_obs


__all__ = ["apply_observation_process"]
