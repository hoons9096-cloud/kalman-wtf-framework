"""Van Genuchten (1980) soil-water retention curve.

Python port of `vg.m` from the MATLAB Filter-WTF framework.

The retention curve gives volumetric water content θ as a function of matric
pressure head h:

    θ(h) = θ_r + (θ_s − θ_r) / (1 + (α · h)^n)^m,    h > 0
    θ(h) = θ_s,                                       h ≤ 0   (saturated)

with m = 1 − 1/n.
"""
from __future__ import annotations

import numpy as np


def vg(
    h: np.ndarray | float,
    theta_s: float,
    theta_r: float,
    alpha: float,
    n: float,
) -> np.ndarray:
    """Van Genuchten retention θ(h).

    Parameters
    ----------
    h : array_like or float
        Matric pressure head magnitude (>= 0 for unsaturated, <= 0 for saturated).
    theta_s, theta_r : float
        Saturated and residual volumetric water content.
    alpha, n : float
        Van Genuchten shape parameters (alpha in 1/m, n dimensionless > 1).

    Returns
    -------
    theta : ndarray
        Volumetric water content at each h. Same shape as input.

    Notes
    -----
    This is a faithful port of the MATLAB `vg.m` (vectorised, point-wise
    operators). The boundary at h = 0 is set to the saturated value
    `theta_s`; only strictly positive h trigger the unsaturated branch.
    """
    h_arr = np.atleast_1d(np.asarray(h, dtype=float))
    m_exp = 1.0 - 1.0 / n

    theta = np.empty_like(h_arr)

    sat_mask = h_arr <= 0.0
    unsat_mask = ~sat_mask

    theta[sat_mask] = theta_s

    if np.any(unsat_mask):
        h_val = h_arr[unsat_mask]
        denom = (1.0 + (alpha * h_val) ** n) ** m_exp
        theta[unsat_mask] = theta_r + (theta_s - theta_r) / denom

    return theta if h_arr.ndim > 0 else float(theta)


__all__ = ["vg"]
