"""Core Kalman-WTF framework: state-space recharge inference.

Python port of `run_model_core()` in `filter_wtf_kalman_gui_v14_fixed.m`.

The framework treats groundwater head h(t) as the state of a linear
reservoir with discrete-time dynamics

    h(t+1) = (h(t) - h_base) * exp(k) + h_base + u(t) + w(t)
    y(t)   = h(t) + v(t)

where ``k < 0`` is the recession constant (per day), ``h_base`` is the
physical lower bound (set automatically as ``min(h_obs) - 2 m``, clipped
to non-negative), ``u(t)`` is the recharge-driven head increment, and
``w, v`` are zero-mean Gaussian process and observation noise.

The recharge series ``R(t)`` is inferred per-step from observed head
rises using

    R(t) = n_f(t) * k * (h(t+1) - h(t)*exp(k)) / (exp(k) - 1)

with ``n_f(t)`` the time-varying fillable porosity from
``fillable_porosity.filpor_tr``. A second pass runs the Kalman update,
producing both a "pure WTF" (open-loop) and a "Kalman-corrected" head
trajectory.

Calling pattern
---------------
    from framework.kalman_wtf import run_model_core
    result = run_model_core(k=-0.015, z=3.0, sn=2, po_m=rain_m, ho_m=gw_obs,
                            r_cutoff_m=0.02)
    # result is a dataclass with fields: recharge_m_per_day, h_kalman_m,
    # h_pure_wtf_m, h_base_m, n_f_avg, f_n
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .fillable_porosity import filpor_tr


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ModelCoreResult:
    """Per-step output of `run_model_core`."""
    recharge_m_per_day: np.ndarray   # recharge inferred per step (m/day)
    h_kalman_m: np.ndarray           # Kalman-corrected head trajectory (m)
    h_pure_wtf_m: np.ndarray         # open-loop "pure WTF" trajectory (m)
    h_base_m: float                  # physical base level used (m)
    n_f_avg: float                   # average fillable porosity (rainy days)
    f_n: float                       # self-calibration factor (recharge/rain/Sy_avg)
    sy_series: np.ndarray            # per-step fillable porosity series


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_lag(po: np.ndarray, lag: int) -> np.ndarray:
    """Shift rainfall by ``lag`` days (zero-pad at start)."""
    if lag <= 0:
        return po.copy()
    out = np.zeros_like(po)
    out[lag:] = po[:-lag]
    return out


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_model_core(
    k: float,
    z: float,
    sn: int,
    po_m: np.ndarray,
    ho_m: np.ndarray,
    r_cutoff_m: float = 0.02,
    Q: float = 0.005,
    R: float = 0.1,
    P0: float = 1.0,
    initial_n_dry: int = 30,
) -> ModelCoreResult:
    """Run the core Kalman-WTF model.

    Parameters
    ----------
    k : float
        Recession constant (per day). Must be < 0.
    z : float
        Vadose-zone depth (m), used by `filpor_tr`.
    sn : int
        USDA soil class (1..12).
    po_m : ndarray of shape (n,)
        Rainfall series in metres (i.e. mm/1000). Already lag-shifted if
        desired (see `apply_lag`).
    ho_m : ndarray of shape (n,)
        Observed (or pumping-filtered) head series in metres. NaN entries
        are propagated to the Kalman step as "missing observation".
    r_cutoff_m : float, default 0.02
        Rainfall threshold below which a day is treated as dry for the
        dry-period counter feeding `filpor_tr`.
    Q : float, default 0.005
        Process-noise variance.
    R : float, default 0.1
        Observation-noise variance.
    P0 : float, default 1.0
        Initial state-error covariance.
    initial_n_dry : int, default 30
        Starting value of the dry-period counter (matches MATLAB).

    Returns
    -------
    ModelCoreResult
        See class docstring.
    """
    po = np.asarray(po_m, dtype=float).copy()
    ho = np.asarray(ho_m, dtype=float).copy()
    nn = len(ho)

    if k >= 0:
        raise ValueError("k must be negative (recession constant)")

    # Physical base level
    h_base = float(np.nanmin(ho)) - 2.0
    if h_base < 0:
        h_base = 0.0

    # --- First pass: infer recharge from observed head rises ---
    n_dry = initial_n_dry
    sy_series = np.zeros(nn)
    rech = np.zeros(nn)

    for ii in range(nn - 1):
        ho_i = ho[ii]
        ho_ip1 = ho[ii + 1]

        if np.isnan(ho_i) or np.isnan(ho_ip1):
            dh = 0.0
        else:
            dh = ho_ip1 - ho_i

        th = max(dh, 0.001)
        if po[ii] <= r_cutoff_m:
            th = 0.001

        n_f = filpor_tr(sn, z, n_dry, th)

        if po[ii] < r_cutoff_m:
            n_dry += 1
        else:
            n_dry = 1

        sy_series[ii] = n_f

        if np.isnan(ho_i) or np.isnan(ho_ip1):
            rech[ii] = 0.0
        else:
            h1 = ho_i - h_base
            h2 = ho_ip1 - h_base
            denom = np.exp(k) - 1.0
            if abs(denom) < 1.0e-6:
                r_val = 0.0
            else:
                r_val = n_f * k * (h2 - h1 * np.exp(k)) / denom
            rech[ii] = r_val if po[ii] > 0 else 0.0

    # Self-calibration factor f_n
    nr = int(np.sum(po > r_cutoff_m))
    if nr == 0:
        nr = 1
    n_f_avg = float(np.sum(sy_series) / nr)
    total_rech_positive = float(np.sum(rech + np.abs(rech)) / 2.0)
    total_po = float(np.sum(po))
    if total_po > 0 and n_f_avg > 0:
        f_n = (total_rech_positive / total_po) / n_f_avg
    else:
        f_n = 0.0
    if not np.isfinite(f_n):
        f_n = 0.0

    # --- Second pass: Kalman filter + pure WTF simulation ---
    hs_kf = np.zeros(nn)
    hs_pure = np.zeros(nn)

    # Initial state from first non-NaN observation
    first_valid = np.flatnonzero(~np.isnan(ho))
    if len(first_valid) == 0:
        raise ValueError("ho_m has no valid (non-NaN) values")
    start_val = ho[first_valid[0]]
    hs_kf[0] = start_val
    hs_pure[0] = start_val

    P_cov = P0
    exp_k = np.exp(k)

    for ii in range(1, nn):
        u_input = po[ii - 1] * f_n / k * (exp_k - 1.0)
        hs_pure[ii] = (hs_pure[ii - 1] - h_base) * exp_k + h_base + u_input

        h_pred = (hs_kf[ii - 1] - h_base) * exp_k + h_base + u_input
        P_pred = exp_k ** 2 * P_cov + Q

        if np.isnan(ho[ii]):
            hs_kf[ii] = h_pred
            P_cov = P_pred
        else:
            K_gain = P_pred / (P_pred + R)
            hs_kf[ii] = h_pred + K_gain * (ho[ii] - h_pred)
            P_cov = (1.0 - K_gain) * P_pred

    return ModelCoreResult(
        recharge_m_per_day=rech,
        h_kalman_m=hs_kf,
        h_pure_wtf_m=hs_pure,
        h_base_m=h_base,
        n_f_avg=n_f_avg,
        f_n=f_n,
        sy_series=sy_series,
    )


__all__ = ["ModelCoreResult", "apply_lag", "run_model_core"]
