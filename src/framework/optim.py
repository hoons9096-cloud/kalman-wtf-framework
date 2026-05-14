"""Parameter optimisation: lag grid-search + Nelder-Mead over (k, z).

Python port of ``run_optimization()`` + ``calculate_pure_error()`` from
the MATLAB Filter-WTF framework.

For each candidate lag in a discrete grid (``0..max_lag``), the pure-WTF
RMSE between observation and the open-loop simulation is minimised over
``(k, z)`` using the Nelder-Mead simplex (``scipy.optimize.minimize``).
The (lag, k, z) triple with the lowest pure-WTF RMSE is returned.

The pure-WTF (not Kalman) RMSE is used as the objective by design: the
Kalman filter's strong correction can mask poor parameter choices, so
optimising against the *open-loop* error forces the framework to recover
the actual physical parameters that explain the observed head dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from .kalman_wtf import apply_lag, run_model_core


@dataclass
class OptimResult:
    """Output of `run_optimization`."""
    lag_days: int
    k: float
    z: float
    rmse_pure: float
    rmse_kalman: float
    cc_kalman: float


def calculate_pure_error(
    k: float,
    z: float,
    lag: int,
    sn: int,
    po_m: np.ndarray,
    ho_m: np.ndarray,
    r_cutoff_m: float,
) -> float:
    """RMSE between observed head and the *pure-WTF* (open-loop) simulation.

    Used as the objective function during Nelder-Mead optimisation of
    (k, z). Returns 1e9 for parameters outside the physical search bounds
    (matches MATLAB calculate_pure_error penalty).
    """
    if k >= 0 or k < -0.6:
        return 1.0e9
    if z < 0.1 or z > 30:
        return 1.0e9

    po_shifted = apply_lag(po_m, int(round(lag)))
    try:
        res = run_model_core(k=k, z=z, sn=sn, po_m=po_shifted, ho_m=ho_m,
                             r_cutoff_m=r_cutoff_m)
    except ValueError:
        return 1.0e9

    err = float(np.sqrt(np.nanmean((res.h_pure_wtf_m - ho_m) ** 2)))
    if not np.isfinite(err):
        return 1.0e9
    return err


def run_optimization(
    po_m: np.ndarray,
    ho_m: np.ndarray,
    sn: int = 2,
    r_cutoff_m: float = 0.02,
    start_k: float = -0.015,
    start_z: float = 3.0,
    lag_grid: tuple[int, ...] = tuple(range(0, 15)),
    k_bounds: tuple[float, float] = (-0.5, -0.0001),
    z_bounds: tuple[float, float] = (0.5, 5.0),
    tol_x: float = 1.0e-3,
) -> OptimResult:
    """Optimise (lag, k, z) by lag grid + Nelder-Mead over (k, z).

    Parameters
    ----------
    po_m, ho_m : ndarray
        Rainfall (m) and observed head (m), same length.
    sn : int, default 2
        USDA soil class (default 2 = Sandy Loam).
    r_cutoff_m : float, default 0.02
        Rainfall threshold (m) below which days are treated as dry.
    start_k, start_z : float
        Nelder-Mead initial guesses.
    lag_grid : tuple of int
        Discrete lag values (days) to grid-search.
    k_bounds, z_bounds : tuples
        Hard clamps applied to the optimised values *after* convergence
        (matches MATLAB feedback clamp on z ≤ 5 m).
    tol_x : float
        Nelder-Mead xatol.

    Returns
    -------
    OptimResult
        Best (lag, k, z) plus pure-WTF and Kalman-filtered RMSE/CC.
    """
    best_lag = 0
    best_k = start_k
    best_z = start_z
    min_err = 1.0e9

    for try_lag in lag_grid:
        obj = lambda x: calculate_pure_error(
            x[0], x[1], try_lag, sn, po_m, ho_m, r_cutoff_m
        )
        opt = minimize(obj, x0=[start_k, start_z],
                       method="Nelder-Mead",
                       options={"xatol": tol_x, "disp": False})
        err = float(opt.fun)
        if err < min_err:
            min_err = err
            best_lag = try_lag
            best_k = max(min(opt.x[0], k_bounds[1]), k_bounds[0])
            best_z = max(min(opt.x[1], z_bounds[1]), z_bounds[0])

    # Final Kalman-filtered run with optimal parameters
    po_shift = apply_lag(po_m, best_lag)
    final = run_model_core(k=best_k, z=best_z, sn=sn,
                           po_m=po_shift, ho_m=ho_m,
                           r_cutoff_m=r_cutoff_m)

    rmse_kalman = float(np.sqrt(np.nanmean((final.h_kalman_m - ho_m) ** 2)))
    # CC (Pearson, ignoring NaN-aligned pairs)
    mask = ~np.isnan(ho_m) & ~np.isnan(final.h_kalman_m)
    if mask.sum() >= 2:
        cc = float(np.corrcoef(final.h_kalman_m[mask], ho_m[mask])[0, 1])
    else:
        cc = 0.0

    return OptimResult(
        lag_days=int(best_lag),
        k=float(best_k),
        z=float(best_z),
        rmse_pure=float(min_err),
        rmse_kalman=rmse_kalman,
        cc_kalman=cc,
    )


__all__ = ["OptimResult", "calculate_pure_error", "run_optimization"]
