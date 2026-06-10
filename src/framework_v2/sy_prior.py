"""Bayesian Sy prior penalty (Phase A-2).

In the v1 paper version the Nelder-Mead optimisation minimised pure
head-fit RMSE without any constraint on the resulting effective Sy.
For 10 of 12 USDA texture classes at SH-22 this caused the optimiser
to settle at a degenerate minimum with Sy ≈ 0.022 and Rch ≈ 0.84 %,
because a very low Sy paired with a slow recession reproduces the
observed head curvature with marginally better RMSE than any physically
plausible (k, Sy) pair.

This module introduces a Gaussian penalty drawing the operational Sy
toward the **field-effective range** documented by Healy and Cook
(2002, §4.2) and Moon et al. (2004) for Korean shallow alluvial
aquifers:

    Sy ~ N(μ = 0.07, σ = 0.03)        # field-effective prior

The penalised loss is

    L_total = RMSE + λ · ((Sy_op - μ) / σ)²

where λ is the prior strength. λ = 0.05 m² was tuned to (a) leave the
data-fit term dominant in well-identified cases (sn = 1, 3 at SH-22),
(b) break the fine-texture degeneracy in the remaining classes, and
(c) keep the framework reproducible — λ is exposed as an argument so
that sensitivity to the prior can be examined.
"""
from __future__ import annotations

import numpy as np


# Field-effective Sy prior (Healy & Cook 2002; Moon et al. 2004)
DEFAULT_SY_PRIOR_MEAN = 0.07
DEFAULT_SY_PRIOR_STD = 0.03
DEFAULT_PRIOR_STRENGTH = 10.0  # λ tuned on synthetic S1/S2 (recovery 0.32 → 0.98)


def sy_prior_penalty(
    sy_operational: float,
    prior_mean: float = DEFAULT_SY_PRIOR_MEAN,
    prior_std: float = DEFAULT_SY_PRIOR_STD,
    strength: float = DEFAULT_PRIOR_STRENGTH,
) -> float:
    """Gaussian penalty on the operational specific yield.

    Parameters
    ----------
    sy_operational : float
        The framework's effective Sy (time-mean of the filpor_tr
        series, or Σ recharge / Σ positive Δh, depending on the caller).
    prior_mean, prior_std : float
        Field-effective Sy prior distribution (default: Healy & Cook 2002).
    strength : float
        Penalty coefficient λ. Larger → stronger prior pull.

    Returns
    -------
    penalty : float
        ≥ 0; added to the head-fit RMSE to give the total loss.
    """
    if sy_operational <= 0:
        # Hard penalty for unphysical values
        return 10.0
    z = (sy_operational - prior_mean) / prior_std
    return float(strength * z * z)


def estimate_operational_sy(
    recharge_m_per_day: np.ndarray,
    h_simulated: np.ndarray,
) -> float:
    """Compute the operational Sy used by the prior.

    Operational Sy = Σ recharge / Σ (positive head rises)

    This is the WTF identity (Healy & Cook 2002 eq. 4) and is the
    quantity whose lab-to-field gap drives the 25 % recovery bias in
    the v1 framework.
    """
    rch_total = float(np.nansum(recharge_m_per_day))
    dh = np.diff(h_simulated)
    pos_dh = float(np.nansum(np.maximum(dh, 0.0)))
    if pos_dh < 1e-9:
        return 0.0
    return rch_total / pos_dh


__all__ = [
    "sy_prior_penalty",
    "estimate_operational_sy",
    "DEFAULT_SY_PRIOR_MEAN",
    "DEFAULT_SY_PRIOR_STD",
    "DEFAULT_PRIOR_STRENGTH",
]
