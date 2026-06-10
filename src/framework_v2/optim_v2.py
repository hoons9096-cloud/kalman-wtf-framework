"""Improved parameter optimisation (Phase A integration).

Combines the v1 framework with:
  - CCF-fixed lag from `ccf_lag.py`
  - Bayesian Sy prior from `sy_prior.py`

The lag is no longer a free Nelder-Mead variable; it is held fixed at
the CCF peak τ_cc. The objective adds a Gaussian Sy prior penalty so
that the optimiser cannot escape into the degenerate near-zero-Sy
basin that affects 10 of 12 USDA texture classes in the v1 results.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from framework.kalman_wtf import apply_lag, run_model_core

from .ccf_lag import ccf_peak_lag
from .sy_prior import (
    DEFAULT_PRIOR_STRENGTH,
    DEFAULT_SY_PRIOR_MEAN,
    DEFAULT_SY_PRIOR_STD,
    estimate_operational_sy,
    sy_prior_penalty,
)


@dataclass
class OptimV2Result:
    lag_days: int
    lag_source: str   # "ccf" if fixed from CCF, "free" if optimised
    k: float
    z: float
    sn: int
    sy_operational: float
    rmse_pure: float
    rmse_kalman: float
    cc: float
    annual_rch_mm: float
    prior_penalty: float
    total_loss: float


def calculate_loss_with_prior(
    k: float,
    z: float,
    lag: int,
    sn: int,
    po_m: np.ndarray,
    ho_m: np.ndarray,
    r_cutoff_m: float = 0.02,
    prior_mean: float = DEFAULT_SY_PRIOR_MEAN,
    prior_std: float = DEFAULT_SY_PRIOR_STD,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> tuple[float, float, float, float]:
    """Compute the penalised loss for a single (k, z, lag, sn) point.

    Returns (total_loss, rmse_pure, sy_operational, prior_penalty).
    """
    if k >= 0 or z <= 0:
        return 1e6, 1e6, 0.0, 1e6

    po_shift = apply_lag(po_m, lag)
    try:
        core = run_model_core(k=k, z=z, sn=sn, po_m=po_shift, ho_m=ho_m,
                              r_cutoff_m=r_cutoff_m)
    except Exception:
        return 1e6, 1e6, 0.0, 1e6

    mask = ~np.isnan(ho_m) & ~np.isnan(core.h_pure_wtf_m)
    if mask.sum() < 5:
        return 1e6, 1e6, 0.0, 1e6
    rmse_pure = float(np.sqrt(np.nanmean(
        (core.h_pure_wtf_m[mask] - ho_m[mask]) ** 2)))

    sy_op = estimate_operational_sy(core.recharge_m_per_day,
                                    core.h_pure_wtf_m)
    penalty = sy_prior_penalty(sy_op, prior_mean, prior_std, prior_strength)

    return rmse_pure + penalty, rmse_pure, sy_op, penalty


def run_optimization_v2(
    po_m: np.ndarray,
    ho_m: np.ndarray,
    sn: int = 2,
    r_cutoff_m: float = 0.02,
    start_k: float = -0.015,
    start_z: float = 3.0,
    k_bounds: tuple[float, float] = (-0.5, -0.0001),
    z_bounds: tuple[float, float] = (0.5, 5.0),
    tol_x: float = 1.0e-3,
    use_ccf_lag: bool = True,
    use_sy_prior: bool = True,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    max_lag_days: int = 14,
    lag_step: int = 1,
    nm_maxiter: int = 120,
) -> OptimV2Result:
    """Phase A optimisation: CCF-fixed lag + Sy prior.

    Parameters
    ----------
    use_ccf_lag : bool
        If True, lag is fixed at the CCF peak. If False, the v1
        free-lag behaviour is reproduced (grid search 0..max_lag_days).
    use_sy_prior : bool
        If True, add the Bayesian Sy penalty to the objective.
    prior_strength : float
        λ for the Sy prior; 0 disables the penalty term even if
        use_sy_prior=True.
    """
    # --- Step 1: Identify lag ----------------------------------------
    if use_ccf_lag:
        lag = ccf_peak_lag(po_m, ho_m, max_lag_days=max_lag_days)
        lag_source = "ccf"
    else:
        # Fall back to v1 grid behaviour (informative for ablation)
        lag = 0
        lag_source = "free"

    # --- Step 2: Optimise (k, z) at the fixed lag --------------------
    lam = prior_strength if use_sy_prior else 0.0

    def obj(x):
        total, _, _, _ = calculate_loss_with_prior(
            x[0], x[1], lag, sn, po_m, ho_m, r_cutoff_m,
            prior_strength=lam,
        )
        return total

    if use_ccf_lag:
        candidate_lags = [lag]
    else:
        candidate_lags = list(range(0, max_lag_days + 1, lag_step))

    best_total = np.inf
    best_lag = lag
    best_k = start_k
    best_z = start_z

    for cand_lag in candidate_lags:
        def obj_cand(x, _lag=cand_lag):
            total, _, _, _ = calculate_loss_with_prior(
                x[0], x[1], _lag, sn, po_m, ho_m, r_cutoff_m,
                prior_strength=lam,
            )
            return total
        res = minimize(obj_cand, x0=[start_k, start_z],
                       method="Nelder-Mead",
                       options={"xatol": tol_x, "fatol": 1e-4,
                                "maxiter": nm_maxiter, "disp": False})
        if res.fun < best_total:
            best_total = float(res.fun)
            best_lag = cand_lag
            best_k = max(min(res.x[0], k_bounds[1]), k_bounds[0])
            best_z = max(min(res.x[1], z_bounds[1]), z_bounds[0])

    # --- Step 3: Final run with optimal parameters -------------------
    total_final, rmse_pure, sy_op, penalty = calculate_loss_with_prior(
        best_k, best_z, best_lag, sn, po_m, ho_m, r_cutoff_m,
        prior_strength=lam,
    )

    po_shift = apply_lag(po_m, best_lag)
    core = run_model_core(k=best_k, z=best_z, sn=sn,
                          po_m=po_shift, ho_m=ho_m, r_cutoff_m=r_cutoff_m)

    mask = ~np.isnan(ho_m) & ~np.isnan(core.h_kalman_m)
    rmse_kal = float(np.sqrt(np.nanmean(
        (core.h_kalman_m[mask] - ho_m[mask]) ** 2)))
    if mask.sum() >= 2:
        cc = float(np.corrcoef(core.h_kalman_m[mask], ho_m[mask])[0, 1])
    else:
        cc = 0.0

    n_days = len(ho_m)
    years = n_days / 365.25
    annual_rch_mm = float(np.nansum(core.recharge_m_per_day) * 1000.0 / years)

    return OptimV2Result(
        lag_days=int(best_lag),
        lag_source=lag_source,
        k=float(best_k),
        z=float(best_z),
        sn=int(sn),
        sy_operational=float(sy_op),
        rmse_pure=float(rmse_pure),
        rmse_kalman=rmse_kal,
        cc=cc,
        annual_rch_mm=annual_rch_mm,
        prior_penalty=float(penalty),
        total_loss=float(total_final),
    )


def sn_sweep_v2(
    po_m: np.ndarray,
    ho_m: np.ndarray,
    sn_grid: tuple[int, ...] = tuple(range(1, 13)),
    **kwargs,
) -> tuple[OptimV2Result, list[OptimV2Result]]:
    """Sweep sn = 1..12 and return the RMSE-optimal configuration plus
    the full sweep table (for the sn-sensitivity diagnostic).
    """
    results = []
    best = None
    for sn in sn_grid:
        r = run_optimization_v2(po_m=po_m, ho_m=ho_m, sn=sn, **kwargs)
        results.append(r)
        # Selection criterion: prefer total_loss (RMSE + prior) under v2
        if best is None or r.total_loss < best.total_loss:
            best = r
    return best, results


__all__ = [
    "OptimV2Result",
    "calculate_loss_with_prior",
    "run_optimization_v2",
    "sn_sweep_v2",
]
