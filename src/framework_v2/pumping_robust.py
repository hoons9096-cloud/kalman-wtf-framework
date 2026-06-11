"""Pumping-robust estimation of the head-equivalent recharge input.

Diagnosis (see `notebooks/v2/pumping_benchmark.py`): on a
soil-heterogeneous synthetic ensemble the dominant bias in the WTF input
integral ``U`` is **pumping recovery mis-attributed as recharge** (a
~2.5x over-estimate), with observation noise a secondary contributor
(~1.3x). A moving-average smoother cannot fix this because pumping
recovery is a real, smooth head rise.

The fix exploits the physical asymmetry between recharge and pumping
(Nimmo et al., 2015, episodic master recession; Cuthbert, 2014):

  * genuine recharge raises the recession **baseline** permanently —
    after the rise the head recedes from a higher level;
  * a pumping episode is a drawdown that **recovers to the same
    baseline** — its net effect on the recession envelope is zero.

We therefore track the recession baseline and **reconstruct the head
along the recession through any below-baseline excursion** (pumping
drawdown + its recovery), keeping observed values only where the head is
at or above the receding baseline (genuine recharge). The standard
recession-corrected input is then formed from the *reconstructed* head,
so pumping recovery contributes nothing while the unbiased clean-data
behaviour is preserved.

The estimator is deliberately conservative: a genuine recharge pulse that
arrives during a drawdown is partially absorbed into the reconstructed
recession, producing a mild (~15%) low bias — an honest, characterised
trade for removing the dominant pumping bias.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .free_sy_inversion import estimate_h_base, estimate_recession_k


def _nan_smooth(h: np.ndarray, w: int, kind: str = "mean") -> np.ndarray:
    """NaN-aware centred moving filter. ``kind='median'`` is
    edge-preserving: it suppresses observation noise like the mean but
    does not attenuate genuine step rises, which removes most of the
    conservative bias of the reconstruction (full-regime recovery RMSE
    0.30 -> 0.16 on the soil-heterogeneous benchmark)."""
    if w <= 1:
        return np.asarray(h, dtype=float)
    h = np.asarray(h, dtype=float)
    n = len(h)
    out = np.full(n, np.nan)
    half = w // 2
    stat = np.median if kind == "median" else np.mean
    for i in range(n):
        seg = h[max(0, i - half):min(n, i + half + 1)]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = stat(seg)
    return out


@dataclass
class PumpingRobustResult:
    U_annual_mm: float
    k: float
    h_base: float
    h_reconstructed: np.ndarray
    n_years: float


def reconstruct_recession_baseline(h_def: np.ndarray, k: float) -> np.ndarray:
    """Reconstruct the deficit series along the recession through any
    below-baseline (pumping/noise) excursion.

    Where the (smoothed) head sits at or above the receding baseline it is
    kept and the baseline rises to it (genuine recharge); where it falls
    below, the baseline continues to recede and the reconstruction follows
    the recession line, so the subsequent recovery is not counted as a
    rise.
    """
    n = len(h_def)
    hat = np.full(n, np.nan)
    valid = np.flatnonzero(np.isfinite(h_def))
    if len(valid) == 0:
        return hat
    i0 = valid[0]
    baseline = h_def[i0]
    hat[i0] = baseline
    a = 1.0 - k
    for t in range(i0 + 1, n):
        pred = a * baseline
        x = h_def[t]
        if np.isfinite(x) and x >= pred:
            hat[t] = x
            baseline = x
        else:
            hat[t] = pred
            baseline = pred
    return hat


def _rain_recent_mask(po: np.ndarray, r_cutoff_m: float,
                      lag_window: int) -> np.ndarray:
    """True where rain fell on the day or within the preceding
    ``lag_window`` days. With a vadose-zone delay, recharge-driven rises
    arrive days after the rain; a same-day gate silently discards them.
    Widening the gate is safe in combination with the recession-baseline
    reconstruction, which already rejects pumping recoveries and
    below-baseline noise on the admitted days."""
    rain = po > r_cutoff_m
    if lag_window <= 0:
        return rain
    out = rain.copy()
    for s in range(1, lag_window + 1):
        out[s:] |= rain[:-s]
    return out


def estimate_U_pumping_robust(
    po_m: np.ndarray,
    ho_m: np.ndarray,
    smooth_window: int = 5,
    r_cutoff_m: float = 0.002,
    rain_lag_window: int = 0,
    smoother: str = "median",
) -> PumpingRobustResult:
    """Pumping-robust annual head-equivalent recharge input U' (mm yr⁻¹).

    ``rain_lag_window`` widens the rain gate to admit vadose-lagged
    recharge arriving after the causative rainfall. The default (0) keeps
    the strict same-day gate of the characterised point estimator; the
    gate-width systematic is marginalised in `posterior.recharge_posterior`
    rather than baked into the point estimate. ``smoother='median'``
    (default) is edge-preserving and roughly halves the input-recovery
    error relative to the moving mean.
    """
    raw = np.asarray(ho_m, dtype=float)
    po = np.asarray(po_m, dtype=float)
    n = len(raw)
    n_years = n / 365.25

    h_base = estimate_h_base(raw)
    hs = _nan_smooth(raw, smooth_window, kind=smoother)
    k, _ = estimate_recession_k(hs, po, h_base)

    hat = reconstruct_recession_baseline(hs - h_base, k)

    u = np.full(n, np.nan)
    for t in range(n - 1):
        if np.isfinite(hat[t]) and np.isfinite(hat[t + 1]):
            u[t] = (hat[t + 1] - hat[t]) + k * hat[t]
    rain = _rain_recent_mask(po[:n], r_cutoff_m, rain_lag_window)
    U_total = float(np.nansum(np.where(rain & np.isfinite(u),
                                       np.maximum(u, 0.0), 0.0)))
    return PumpingRobustResult(
        U_annual_mm=U_total * 1000.0 / n_years,
        k=float(k), h_base=float(h_base),
        h_reconstructed=hat + h_base, n_years=float(n_years),
    )


__all__ = [
    "PumpingRobustResult",
    "reconstruct_recession_baseline",
    "estimate_U_pumping_robust",
]
