"""Kalman-smoothed recharge inference (Phase A-4, S4/noise fix).

Problem: `run_model_core` infers recharge from day-to-day differences
of the *raw* observed head. Under high observation noise (synthetic
scenario S4, σ_obs = 0.05 m) the daily difference is noise-dominated
(σ_dh = √2 · 0.05 ≈ 0.07 m), corrupting both the fillable-porosity
trigger (`th = max(dh, 0.001)`) and the per-day recharge estimate.
v2's Sy prior cannot repair this because the corruption happens
upstream of the Sy computation: S4 recovery stalls at 0.47 while the
noise-free scenarios reach 0.95+.

Fix: two-stage iterated smoothing that reuses the framework's own
Kalman filter as a pre-smoother —

  Stage 1: optimise (k, z, sn) on the raw head (standard v2 path)
           and compute the Kalman-filtered trajectory h_kal.
  Stage 2: re-run the same optimisation with h_kal substituted for
           the raw observations, so the recharge-inference first pass
           differentiates a de-noised signal.

The substitution is conservative: h_kal tracks the observations
closely wherever they are reliable (Kalman gain balances process vs
observation noise) and reverts to the process model across gaps and
outliers. One iteration suffices; further iterations change recovery
by < 1 % in testing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from framework.kalman_wtf import apply_lag, run_model_core

from .optim_v2 import OptimV2Result, run_optimization_v2, sn_sweep_v2


def estimate_obs_noise(ho_m: np.ndarray, po_m: np.ndarray,
                       r_cutoff_m: float = 0.002) -> float:
    """Estimate the observation-noise std from dry-day head increments.

    On rain-free days the true head follows the slow recession, so the
    lag-1 difference is noise-dominated: Var(Δh) ≈ 2 σ_obs². A robust
    scale estimate (median absolute deviation) avoids contamination by
    residual pumping spikes.

    Returns σ̂_obs in metres (floored at 1 mm).
    """
    ho = np.asarray(ho_m, dtype=float)
    po = np.asarray(po_m, dtype=float)
    dh = np.diff(ho)
    dry = po[:-1] <= r_cutoff_m
    dh_dry = dh[dry & np.isfinite(dh)]
    if len(dh_dry) < 30:
        dh_dry = dh[np.isfinite(dh)]
    # MAD → std of Δh, then σ_obs = σ_Δh / √2
    mad = float(np.median(np.abs(dh_dry - np.median(dh_dry))))
    sigma_dh = 1.4826 * mad
    return max(sigma_dh / np.sqrt(2.0), 0.001)


def sn_sweep_v2_smoothed(
    po_m: np.ndarray,
    ho_m: np.ndarray,
    sn_grid: tuple[int, ...] = tuple(range(1, 13)),
    n_iterations: int = 1,
    **kwargs,
) -> tuple[OptimV2Result, list[OptimV2Result]]:
    """Two-stage sn-sweep: optimise on raw head, smooth, re-optimise.

    Parameters
    ----------
    po_m, ho_m : ndarray
        Rainfall (m) and observed head (m).
    sn_grid : tuple of int
        USDA soil classes to sweep.
    n_iterations : int, default 1
        Number of smooth-and-reoptimise passes after the initial raw
        fit. 1 is sufficient (see module docstring).
    **kwargs
        Forwarded to `run_optimization_v2` / `sn_sweep_v2`
        (use_ccf_lag, use_sy_prior, lag_step, nm_maxiter, ...).

    Returns
    -------
    (best, sweep) : the final-stage optimum and full sn sweep.
    """
    # --- Stage 1: raw-head optimisation --------------------------------
    best, sweep = sn_sweep_v2(po_m=po_m, ho_m=ho_m, sn_grid=sn_grid,
                              **kwargs)

    ho_current = np.asarray(ho_m, dtype=float)

    # Noise-adaptive observation variance: with the default R = 0.1 m²
    # (σ ≈ 0.32 m) the filter over-smooths low-noise records and
    # destroys genuine recharge rises (S1 recovery 0.96 → 0.09 in
    # testing). Estimating σ̂_obs from dry-day increments makes the
    # smoothing proportional to the actual noise level.
    sigma_obs = estimate_obs_noise(ho_m, po_m)
    R_adaptive = sigma_obs ** 2

    for _ in range(n_iterations):
        # Kalman-filtered trajectory under the current optimum
        po_shift = apply_lag(po_m, best.lag_days)
        core = run_model_core(k=best.k, z=best.z, sn=best.sn,
                              po_m=po_shift, ho_m=ho_current,
                              R=R_adaptive)
        h_smooth = core.h_kalman_m

        # --- Stage 2: re-optimise on the smoothed head ----------------
        best, sweep = sn_sweep_v2(po_m=po_m, ho_m=h_smooth,
                                  sn_grid=sn_grid, **kwargs)
        ho_current = h_smooth

    return best, sweep


__all__ = ["sn_sweep_v2_smoothed"]
