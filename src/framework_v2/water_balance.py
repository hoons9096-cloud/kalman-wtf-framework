"""Water-balance scale constraint: identifying the non-identifiable Sy.

`free_sy_inversion` establishes that head data fix only the line

        R = Sy * U'          (U' = U * 1000 / n_years, mm/yr)

in the (Sy, recharge) plane — the recharge magnitude is free along it
because Sy is non-identifiable from head alone.  This module adds a
*second, orthogonal* constraint that pins the position on the line
without using any head information:

  Constraint A  (literature Sy prior):
        Sy ~ N(mu_Sy, sigma_Sy^2)              -> vertical band
  Constraint B  (catchment water balance):
        R ~ N(c * P, (sigma_c * P)^2)          -> horizontal band
        where c is the recharge coefficient (recharge / precipitation)
        from independent climate/water-balance knowledge, P the annual
        precipitation.

Both constraints live on the same line, so each maps to a Gaussian
statement about Sy:

        Sy_A = mu_Sy            +/- sigma_Sy
        Sy_B = (c * P) / U'     +/- (sigma_c * P) / U'

Their precision-weighted (Bayesian Gaussian) combination gives the
joint estimate ``Sy_joint`` and hence ``R_joint = Sy_joint * U'``.

Two payoffs:
  * The water-balance constraint supplies the scale the head data
    cannot, turning a prior-dominated guess into an identified estimate.
  * The *consistency* between the two constraints (how many sigma apart
    Sy_A and Sy_B sit) is a built-in cross-validation of the WTF result
    — large disagreement flags a violated assumption (wrong Sy class,
    rejected recharge, mis-estimated U), exactly the diagnostic a
    referee asks for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .free_sy_inversion import (
    DEFAULT_SY_PRIOR_MEAN,
    DEFAULT_SY_PRIOR_STD,
    FreeSyResult,
)

# Regional recharge-coefficient prior (recharge / precipitation).
# Korean shallow alluvial aquifers: commonly 0.08-0.20 of precipitation
# (e.g. Moon et al. 2004; Kim et al. 2014). Centred, deliberately wide.
DEFAULT_RCH_COEF_MEAN = 0.12
DEFAULT_RCH_COEF_STD = 0.05


@dataclass
class ConstrainedRecharge:
    U_annual_mm: float          # U' = identifiable head-input rate (mm/yr)
    annual_rain_mm: float
    # Constraint A — Sy prior alone
    sy_prior_mean: float
    sy_prior_std: float
    rch_sy_prior_mm: float
    # Constraint B — water balance alone
    sy_wb_mean: float
    sy_wb_std: float
    rch_wb_mm: float
    # Joint (precision-weighted) estimate on the line
    sy_joint: float
    sy_joint_std: float
    rch_joint_mm: float
    rch_joint_lo_mm: float
    rch_joint_hi_mm: float
    consistency_sigma: float    # |Sy_A - Sy_B| / sqrt(var_A + var_B)


def _gaussian_combine(m1: float, s1: float, m2: float, s2: float
                      ) -> tuple[float, float]:
    """Precision-weighted combination of two Gaussian estimates."""
    p1, p2 = 1.0 / s1 ** 2, 1.0 / s2 ** 2
    var = 1.0 / (p1 + p2)
    return (m1 * p1 + m2 * p2) * var, float(np.sqrt(var))


def constrain_recharge(
    res: FreeSyResult,
    annual_rain_mm: float,
    rch_coef_mean: float = DEFAULT_RCH_COEF_MEAN,
    rch_coef_std: float = DEFAULT_RCH_COEF_STD,
) -> ConstrainedRecharge:
    """Combine the head-derived line (from `invert_free_sy`) with the
    literature Sy prior and a catchment water-balance recharge prior.

    Parameters
    ----------
    res : FreeSyResult
        Output of `invert_free_sy` (provides U and the Sy prior).
    annual_rain_mm : float
        Catchment annual precipitation (mm/yr).
    rch_coef_mean, rch_coef_std : float
        Recharge-coefficient prior (recharge / precipitation).
    """
    U_annual = res.U_head_m * 1000.0 / res.n_years      # mm/yr per unit Sy
    if U_annual <= 0:
        U_annual = 1e-9

    # Constraint A: literature Sy prior
    syA, syA_s = res.sy_prior_mean, res.sy_prior_std
    rch_A = syA * U_annual

    # Constraint B: water-balance recharge -> Sy
    rch_B = rch_coef_mean * annual_rain_mm
    rch_B_s = rch_coef_std * annual_rain_mm
    syB = rch_B / U_annual
    syB_s = rch_B_s / U_annual

    # Joint Sy on the line, then recharge
    sy_j, sy_j_s = _gaussian_combine(syA, syA_s, syB, syB_s)
    rch_j = sy_j * U_annual

    consistency = abs(syA - syB) / np.sqrt(syA_s ** 2 + syB_s ** 2)

    return ConstrainedRecharge(
        U_annual_mm=float(U_annual), annual_rain_mm=float(annual_rain_mm),
        sy_prior_mean=float(syA), sy_prior_std=float(syA_s),
        rch_sy_prior_mm=float(rch_A),
        sy_wb_mean=float(syB), sy_wb_std=float(syB_s),
        rch_wb_mm=float(rch_B),
        sy_joint=float(sy_j), sy_joint_std=float(sy_j_s),
        rch_joint_mm=float(rch_j),
        rch_joint_lo_mm=float((sy_j - sy_j_s) * U_annual),
        rch_joint_hi_mm=float((sy_j + sy_j_s) * U_annual),
        consistency_sigma=float(consistency),
    )


__all__ = [
    "ConstrainedRecharge",
    "constrain_recharge",
    "DEFAULT_RCH_COEF_MEAN",
    "DEFAULT_RCH_COEF_STD",
]
