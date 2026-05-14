"""Statistical-threshold pumping detection.

Python port of `remove_outliers()` in the MATLAB Filter-WTF framework.

A day t is flagged as pumping-induced if its observed head drop
``dh = h(t) - h(t-1)`` is more negative than

    threshold = mean(drops) - (1/sensitivity) * 3 * std(drops)

where ``drops`` is the subset of all *negative* first-differences in the
series. Flagged days have their head replaced by NaN, so that the Kalman
filter in `kalman_wtf.run_model_core` re-imputes them from the linear-
reservoir prediction (the "natural recession" reconstruction).

The ``sensitivity`` parameter is the same UI slider value as in the
MATLAB GUI (default 2.0; higher = less aggressive flagging).
"""
from __future__ import annotations

import numpy as np


def remove_outliers(ho_raw: np.ndarray, sensitivity: float = 2.0) -> np.ndarray:
    """Mask pumping-induced drawdowns by NaN.

    Parameters
    ----------
    ho_raw : ndarray
        Raw observed head series (m).
    sensitivity : float, default 2.0
        UI sensitivity factor. Higher => less aggressive (fewer flags).

    Returns
    -------
    ho_clean : ndarray
        Same shape, with flagged days replaced by NaN.

    Notes
    -----
    The threshold uses *only the negative* first-differences (drops);
    rises are never flagged. This avoids flagging recharge-driven rises
    as pumping artefacts.
    """
    ho_clean = np.asarray(ho_raw, dtype=float).copy()
    dh = np.diff(ho_clean)
    drops = dh[dh < 0]

    if len(drops) == 0:
        return ho_clean

    mean_drop = float(np.mean(drops))
    std_drop = float(np.std(drops))
    threshold = mean_drop - (1.0 / sensitivity) * 3.0 * std_drop

    bad_idx = np.flatnonzero(dh < threshold)
    # bad_idx points to index of dh; flag the *next* observation (i+1)
    flag_idx = bad_idx + 1
    flag_idx = flag_idx[flag_idx < len(ho_clean)]
    ho_clean[flag_idx] = np.nan

    return ho_clean


__all__ = ["remove_outliers"]
