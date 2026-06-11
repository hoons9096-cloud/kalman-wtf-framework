"""Model-based (Kalman/RTS) estimation of the head-equivalent recharge input.

The crude estimator in `free_sy_inversion` forms the recession-corrected
input from a 7-day moving average of the raw head. That de-biasing is a
blunt instrument: a flat window trades noise suppression against
attenuation of genuine rises, and the rectifier ``max(u,0)`` still
accumulates positive observation noise, inflating the input integral
``U`` (the ~2x over-estimation diagnosed on the soil-heterogeneous
benchmark).

This module replaces the moving average with a *recession-aware*
Rauch–Tung–Striebel (RTS) smoother (Rauch et al., 1965) of the
linear-reservoir state. Modelling the head deficit ``x = h - h_b`` as

    x_{t+1} = (1-k) x_t + u_t + w_t,     w_t ~ N(0, Q)      (recharge enters Q)
    y_t     = x_t + v_t,                 v_t ~ N(0, R)

the forward Kalman filter and backward RTS pass return the
minimum-variance smoothed head, which suppresses white observation noise
*without* the lag/attenuation penalty of a flat window because it
exploits the known recession dynamics. The recession-corrected input is
then formed from the smoothed trajectory. ``R`` is estimated from dry-day
increments; ``Q`` is set from the rain-day signal so the smoother tracks
real rises while rejecting noise.

This is the EnKF/state-estimation leg of the benchmark: it improves the
*identifiable* component ``U`` (the reducible uncertainty), independently
of the non-identifiable specific-yield scale.
"""
from __future__ import annotations

import numpy as np

from .free_sy_inversion import estimate_h_base, estimate_recession_k
from .map_inversion import estimate_obs_noise


def rts_smooth_deficit(
    x_obs: np.ndarray,
    k: float,
    Q: float,
    R: float,
    P0: float = 1.0,
) -> np.ndarray:
    """Rauch–Tung–Striebel smoother for x_{t+1} = (1-k)x_t + noise(Q),
    y_t = x_t + noise(R). NaN observations are treated as missing
    (prediction only). Returns the smoothed deficit series."""
    n = len(x_obs)
    a = 1.0 - k
    xf = np.zeros(n)      # filtered mean
    Pf = np.zeros(n)      # filtered variance
    xp = np.zeros(n)      # predicted mean
    Pp = np.zeros(n)      # predicted variance

    # init
    x0 = x_obs[np.isfinite(x_obs)][0] if np.any(np.isfinite(x_obs)) else 0.0
    xf[0] = x0
    Pf[0] = P0
    xp[0] = xf[0]
    Pp[0] = Pf[0]
    for t in range(1, n):
        xp[t] = a * xf[t - 1]
        Pp[t] = a * a * Pf[t - 1] + Q
        if np.isfinite(x_obs[t]):
            Kt = Pp[t] / (Pp[t] + R)
            xf[t] = xp[t] + Kt * (x_obs[t] - xp[t])
            Pf[t] = (1.0 - Kt) * Pp[t]
        else:
            xf[t] = xp[t]
            Pf[t] = Pp[t]

    # RTS backward pass
    xs = xf.copy()
    Ps = Pf.copy()
    for t in range(n - 2, -1, -1):
        if Pp[t + 1] <= 0:
            continue
        Ct = a * Pf[t] / Pp[t + 1]
        xs[t] = xf[t] + Ct * (xs[t + 1] - xp[t + 1])
        Ps[t] = Pf[t] + Ct * Ct * (Ps[t + 1] - Pp[t + 1])
    return xs


def estimate_U_kalman(
    po_m: np.ndarray,
    ho_m: np.ndarray,
    r_cutoff_m: float = 0.002,
    Q_scale: float = 1.0,
) -> tuple[float, float, float]:
    """Estimate the annual head-equivalent recharge input U' (mm yr⁻¹)
    via an RTS-smoothed head, plus the recession constant and h_base.

    Q (process variance) is set from the rain-day increment variance in
    excess of the observation-noise floor, so the smoother follows genuine
    recharge rises; R (observation variance) from dry-day increments.
    """
    ho = np.asarray(ho_m, dtype=float)
    po = np.asarray(po_m, dtype=float)
    n_years = len(ho) / 365.25

    h_base = estimate_h_base(ho)
    sigma_obs = estimate_obs_noise(ho, po)
    R = sigma_obs ** 2

    # Process variance from rain-day signal above the noise floor.
    dh = np.diff(ho)
    rain = po[: len(dh)] > r_cutoff_m
    dh_rain = dh[rain & np.isfinite(dh)]
    var_rain = float(np.var(dh_rain)) if len(dh_rain) > 5 else 4.0 * R
    Q = max(var_rain - 2.0 * R, 0.25 * R) * Q_scale

    # recession constant on the (lightly) smoothed head is robust; estimate
    # on raw and reuse for the smoother dynamics.
    k, _ = estimate_recession_k(ho, po, h_base)

    x_obs = ho - h_base
    xs = rts_smooth_deficit(x_obs, k=k, Q=Q, R=R)
    hs = xs + h_base

    u = np.full(len(hs), np.nan)
    for i in range(len(hs) - 1):
        u[i] = (hs[i + 1] - hs[i]) + k * (hs[i] - h_base)
    keep = (po[: len(u)] > r_cutoff_m) & np.isfinite(u)
    U_total = float(np.sum(np.where(keep, np.maximum(u, 0.0), 0.0)))
    U_annual = U_total * 1000.0 / n_years
    return U_annual, float(k), float(h_base)


__all__ = ["rts_smooth_deficit", "estimate_U_kalman"]
