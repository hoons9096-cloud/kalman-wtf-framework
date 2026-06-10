"""MAP (maximum-a-posteriori) formulation of the WTF recharge inversion.

Why this module exists
----------------------
Recharge from the water-table-fluctuation (WTF) method is the product
Rch = Sy · Δh.  Head data constrain Δh well but constrain Sy only
weakly: a low-Sy / slow-recession aquifer and a high-Sy / fast-recession
aquifer reproduce the *same* observed head curve almost equally well.
The (k, Sy) likelihood therefore has a long, flat ridge — the recharge
estimate is **non-identifiable from head data alone** (the equifinality
that drives WTF recharge uncertainty in practice).

The earlier `optim_v2.calculate_loss_with_prior` added an *ad-hoc*
weighted penalty,  ``L = RMSE + λ·z²``, which mixes a metres-valued
data term with a dimensionless penalty, so the weight λ had hidden units
and had to be tuned by hand.  This module replaces that with a proper
maximum-a-posteriori (MAP) objective in which **both terms are negative
log-likelihoods** and the regularisation weight is *not a free
parameter*:

    J(θ) =  1/(2σ_obs²) · Σ_t (h_model(t) − h_obs(t))²        (data NLL)
          + 1/(2σ_Sy²)  · (Sy_op(θ) − μ_Sy)²                  (prior NLL)

where

    σ_obs  — observation-noise std, estimated from dry-day head
             increments (data-driven; see `estimate_obs_noise`);
    μ_Sy,
    σ_Sy   — field-effective specific-yield prior (mean, std) taken
             from the hydrogeologic literature for the aquifer type
             (Healy & Cook 2002; Johnson 1967).

Both terms are dimensionless, so the *effective* Tikhonov weight on the
sum-of-squares data misfit,

    λ_eff = σ_obs² / σ_Sy²,

is fully determined by two independently estimated quantities — the
measurement noise and the prior width — with **no calibration against
the recovered quantity**.  `lcurve.py` provides an independent
data-only check that this physically-determined weight lands at the
corner of the misfit trade-off curve.

The prior is placed on the *operational* Sy
(Sy_op = Σ recharge / Σ positive Δh; the WTF identity, Healy & Cook 2002
eq. 4) rather than on a raw model parameter, because Sy_op is the
physically meaningful, literature-comparable quantity whose lab-to-field
gap drives the recharge bias.
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
    DEFAULT_SY_PRIOR_MEAN,
    DEFAULT_SY_PRIOR_STD,
    estimate_operational_sy,
)


# ---------------------------------------------------------------------------
# Observation-noise estimate (data-driven σ_obs)
# ---------------------------------------------------------------------------

def estimate_obs_noise(
    ho_m: np.ndarray,
    po_m: np.ndarray,
    r_cutoff_m: float = 0.002,
    floor_m: float = 0.001,
) -> float:
    """Estimate observation-noise std σ_obs from dry-day head increments.

    On rain-free days the true head follows the slow recession, so the
    lag-1 difference is noise-dominated:  Var(Δh) ≈ 2 σ_obs².  A robust
    scale (median absolute deviation) avoids contamination by residual
    pumping spikes.  Returns σ̂_obs in metres, floored at ``floor_m``.
    """
    ho = np.asarray(ho_m, dtype=float)
    po = np.asarray(po_m, dtype=float)
    dh = np.diff(ho)
    dry = po[:-1] <= r_cutoff_m
    dh_dry = dh[dry & np.isfinite(dh)]
    if len(dh_dry) < 30:
        dh_dry = dh[np.isfinite(dh)]
    if len(dh_dry) == 0:
        return floor_m
    mad = float(np.median(np.abs(dh_dry - np.median(dh_dry))))
    sigma_dh = 1.4826 * mad
    return max(sigma_dh / np.sqrt(2.0), floor_m)


def effective_lambda(sigma_obs: float, sigma_sy: float) -> float:
    """Effective Tikhonov weight on the sum-of-squares data misfit,
    λ_eff = σ_obs² / σ_Sy².

    Exposed for reporting and for the L-curve cross-check (the L-curve
    corner is expected to coincide with this value).
    """
    return float(sigma_obs ** 2 / sigma_sy ** 2)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MapResult:
    lag_days: int
    lag_source: str
    k: float
    z: float
    sn: int
    sy_operational: float
    sigma_obs: float            # data-driven observation-noise std (m)
    sy_prior_mean: float
    sy_prior_std: float
    lambda_eff: float           # σ_obs² / σ_Sy²
    rmse_pure: float            # data-misfit RMSE (m), for reporting
    rmse_kalman: float
    cc: float
    annual_rch_mm: float
    nll_data: float             # data negative-log-likelihood term
    nll_prior: float            # prior negative-log-likelihood term
    neg_log_posterior: float    # J(θ) = nll_data + nll_prior


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def map_objective(
    k: float,
    z: float,
    lag: int,
    sn: int,
    po_m: np.ndarray,
    ho_m: np.ndarray,
    sigma_obs: float,
    sy_prior_mean: float = DEFAULT_SY_PRIOR_MEAN,
    sy_prior_std: float = DEFAULT_SY_PRIOR_STD,
    r_cutoff_m: float = 0.02,
) -> tuple[float, float, float, float, float]:
    """Negative log-posterior for one (k, z, lag, sn) point.

    Returns
    -------
    (J, nll_data, nll_prior, sy_op, rmse_pure)
        J = nll_data + nll_prior is the quantity to minimise.
    """
    if k >= 0 or z <= 0:
        big = 1e12
        return big, big, big, 0.0, 1e6

    po_shift = apply_lag(po_m, lag)
    try:
        core = run_model_core(k=k, z=z, sn=sn, po_m=po_shift, ho_m=ho_m,
                              r_cutoff_m=r_cutoff_m)
    except Exception:
        big = 1e12
        return big, big, big, 0.0, 1e6

    mask = ~np.isnan(ho_m) & ~np.isnan(core.h_pure_wtf_m)
    if mask.sum() < 5:
        big = 1e12
        return big, big, big, 0.0, 1e6

    resid = core.h_pure_wtf_m[mask] - ho_m[mask]
    sse = float(np.sum(resid ** 2))
    rmse_pure = float(np.sqrt(sse / mask.sum()))

    sy_op = estimate_operational_sy(core.recharge_m_per_day,
                                    core.h_pure_wtf_m)

    # Data negative log-likelihood (Gaussian, known σ_obs; constant
    # 0.5·N·log(2π σ²) dropped — irrelevant to the argmin and to the
    # L-curve trade-off).
    nll_data = sse / (2.0 * sigma_obs ** 2)

    # Prior negative log-likelihood. Unphysical Sy ≤ 0 gets a hard wall.
    if sy_op <= 0:
        nll_prior = 1e6
    else:
        z_sy = (sy_op - sy_prior_mean) / sy_prior_std
        nll_prior = 0.5 * z_sy * z_sy

    return nll_data + nll_prior, nll_data, nll_prior, sy_op, rmse_pure


# ---------------------------------------------------------------------------
# Single-sn MAP inversion
# ---------------------------------------------------------------------------

def run_map_inversion(
    po_m: np.ndarray,
    ho_m: np.ndarray,
    sn: int = 2,
    sigma_obs: float | None = None,
    sy_prior_mean: float = DEFAULT_SY_PRIOR_MEAN,
    sy_prior_std: float = DEFAULT_SY_PRIOR_STD,
    r_cutoff_m: float = 0.02,
    start_k: float = -0.015,
    start_z: float = 3.0,
    k_bounds: tuple[float, float] = (-0.5, -0.0001),
    z_bounds: tuple[float, float] = (0.5, 5.0),
    use_ccf_lag: bool = True,
    max_lag_days: int = 14,
    lag_step: int = 1,
    tol_x: float = 1.0e-3,
    nm_maxiter: int = 120,
) -> MapResult:
    """MAP inversion at a fixed soil class ``sn``.

    Parameters
    ----------
    sigma_obs : float or None
        Observation-noise std. If None, estimated from dry-day head
        increments via `estimate_obs_noise` (the default, data-driven
        path). Pass a value to override (e.g. for sensitivity sweeps).
    sy_prior_mean, sy_prior_std : float
        Field-effective Sy prior (literature). σ_Sy together with
        σ_obs fixes the regularisation weight; no λ is tuned.
    use_ccf_lag : bool
        If True the lag is fixed at the rainfall→dh/dt cross-correlation
        peak; otherwise a coarse grid (0..max_lag_days, step lag_step)
        is searched and the best-posterior lag retained.
    """
    if sigma_obs is None:
        sigma_obs = estimate_obs_noise(ho_m, po_m)

    if use_ccf_lag:
        lag0 = ccf_peak_lag(po_m, ho_m, max_lag_days=max_lag_days)
        candidate_lags = [lag0]
        lag_source = "ccf"
    else:
        candidate_lags = list(range(0, max_lag_days + 1, lag_step))
        lag_source = "grid"

    best_J = np.inf
    best = (candidate_lags[0], start_k, start_z)
    for cand_lag in candidate_lags:
        def obj(x, _lag=cand_lag):
            J, *_ = map_objective(
                x[0], x[1], _lag, sn, po_m, ho_m, sigma_obs,
                sy_prior_mean, sy_prior_std, r_cutoff_m)
            return J
        res = minimize(obj, x0=[start_k, start_z], method="Nelder-Mead",
                       options={"xatol": tol_x, "fatol": 1e-4,
                                "maxiter": nm_maxiter, "disp": False})
        if res.fun < best_J:
            best_J = float(res.fun)
            bk = max(min(res.x[0], k_bounds[1]), k_bounds[0])
            bz = max(min(res.x[1], z_bounds[1]), z_bounds[0])
            best = (cand_lag, bk, bz)

    best_lag, best_k, best_z = best

    J, nll_data, nll_prior, sy_op, rmse_pure = map_objective(
        best_k, best_z, best_lag, sn, po_m, ho_m, sigma_obs,
        sy_prior_mean, sy_prior_std, r_cutoff_m)

    po_shift = apply_lag(po_m, best_lag)
    core = run_model_core(k=best_k, z=best_z, sn=sn,
                          po_m=po_shift, ho_m=ho_m, r_cutoff_m=r_cutoff_m)
    mask = ~np.isnan(ho_m) & ~np.isnan(core.h_kalman_m)
    rmse_kal = float(np.sqrt(np.nanmean(
        (core.h_kalman_m[mask] - ho_m[mask]) ** 2)))
    cc = (float(np.corrcoef(core.h_kalman_m[mask], ho_m[mask])[0, 1])
          if mask.sum() >= 2 else 0.0)
    years = len(ho_m) / 365.25
    annual_rch_mm = float(np.nansum(core.recharge_m_per_day) * 1000.0 / years)

    return MapResult(
        lag_days=int(best_lag), lag_source=lag_source,
        k=float(best_k), z=float(best_z), sn=int(sn),
        sy_operational=float(sy_op),
        sigma_obs=float(sigma_obs),
        sy_prior_mean=float(sy_prior_mean), sy_prior_std=float(sy_prior_std),
        lambda_eff=effective_lambda(sigma_obs, sy_prior_std),
        rmse_pure=float(rmse_pure), rmse_kalman=rmse_kal, cc=cc,
        annual_rch_mm=annual_rch_mm,
        nll_data=float(nll_data), nll_prior=float(nll_prior),
        neg_log_posterior=float(J),
    )


def map_sn_sweep(
    po_m: np.ndarray,
    ho_m: np.ndarray,
    sn_grid: tuple[int, ...] = tuple(range(1, 13)),
    **kwargs,
) -> tuple[MapResult, list[MapResult]]:
    """Sweep sn = 1..12 and return the MAP-optimal configuration (lowest
    negative log-posterior) plus the full sweep for diagnostics."""
    results: list[MapResult] = []
    best: MapResult | None = None
    # σ_obs is a property of the record, not of sn — estimate once and
    # reuse so every sn shares the same likelihood scaling.
    sigma_obs = kwargs.pop("sigma_obs", None)
    if sigma_obs is None:
        sigma_obs = estimate_obs_noise(ho_m, po_m)
    for sn in sn_grid:
        r = run_map_inversion(po_m=po_m, ho_m=ho_m, sn=sn,
                              sigma_obs=sigma_obs, **kwargs)
        results.append(r)
        if best is None or r.neg_log_posterior < best.neg_log_posterior:
            best = r
    return best, results


__all__ = [
    "MapResult",
    "estimate_obs_noise",
    "effective_lambda",
    "map_objective",
    "run_map_inversion",
    "map_sn_sweep",
]
