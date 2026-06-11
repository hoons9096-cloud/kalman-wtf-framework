"""Monte Carlo posterior for WTF recharge: full uncertainty propagation.

The Gaussian precision-weighted fusion of `water_balance` propagates only
the specific-yield uncertainty into the recharge band. This module
replaces it with a Monte Carlo posterior that propagates, jointly:

  1. **Recession-constant uncertainty.** The pooled k is a length-weighted
     median over dry-spell tail fits; its sampling uncertainty is
     estimated by weighted bootstrap resampling of the per-spell decay
     rates (`recession_k_segments`).
  2. **Input (U) uncertainty.** For each k draw the pumping-robust
     reconstruction is recomputed, so the k uncertainty propagates
     nonlinearly into U. The residual processing sensitivity is included
     by drawing the smoothing window from a small ensemble (3/5/7 days),
     treating the processing choice as model uncertainty.
  3. **Scale (Sy) uncertainty.** Given each U draw, the two head-free
     constraints — the literature prior Sy ~ N(mu, sigma^2) and the
     water-balance prior R ~ N(cP, (sigma_c P)^2), i.e. Sy ~ N(cP/U',
     (sigma_c P/U')^2) — are conjugate Gaussians; Sy is drawn from their
     product (the per-draw fused Gaussian).

The recharge draw is R = Sy * U'. Reported quantiles therefore reflect
recession, processing, and scale uncertainty simultaneously, and the
posterior is exact under the stated Gaussian constraints rather than a
first-order approximation. The known ~15 % conservative bias of the
recession-baseline reconstruction (characterised on the clean synthetic
regime; Section 4.3.1 of the manuscript) can optionally be corrected by
an inflation factor with its own uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .free_sy_inversion import (
    DEFAULT_SY_PRIOR_MEAN,
    DEFAULT_SY_PRIOR_STD,
    estimate_h_base,
    recession_k_segments,
)
from .pumping_robust import (
    _nan_smooth,
    _rain_recent_mask,
    reconstruct_recession_baseline,
)
from .water_balance import DEFAULT_RCH_COEF_MEAN, DEFAULT_RCH_COEF_STD

# Optional reconstruction conservative-bias correction (characterised on
# the clean synthetic regime). Disabled by default: the lag-gate
# marginalisation below already brackets the gating systematics, and a
# synthetic-calibrated inflation factor risks re-importing circularity.
DEFAULT_BIAS_FACTOR_MEAN = 1.18
DEFAULT_BIAS_FACTOR_STD = 0.06
SMOOTH_ENSEMBLE = (3, 5, 7)
# Rain-gate lag windows marginalised as structural uncertainty: the vadose
# delay that maps rainfall to water-table arrival is uncertain (CCF *peak*
# estimates are unreliable; Section 2.5), so the admitted-day window is
# drawn from an ensemble spanning no-lag to a generous shallow-alluvium
# lag. The ensemble is *evidence-weighted*: each window's probability is
# the incremental positive rain->input cross-correlation mass it admits,
# so a fast-responding well concentrates the draws on short gates (CI
# tightens) while a slow/ambiguous response keeps them spread (CI stays
# honest). A floor keeps every window reachable.
LAG_GATE_ENSEMBLE = (0, 7, 14, 21, 28)
LAG_GATE_WEIGHT_FLOOR = 0.05


@dataclass
class RechargePosterior:
    rch_draws_mm: np.ndarray     # posterior recharge draws (mm/yr)
    sy_draws: np.ndarray
    U_draws_mm: np.ndarray
    k_draws: np.ndarray
    rch_median_mm: float
    rch_ci68_mm: tuple[float, float]
    rch_ci95_mm: tuple[float, float]
    sy_median: float
    consistency_sigma: float     # at the median U (diagnostic)


def _weighted_bootstrap_k(k_segs: np.ndarray, weights: np.ndarray,
                          n_draws: int, rng: np.random.Generator
                          ) -> np.ndarray:
    """Bootstrap distribution of the length-weighted-median recession k."""
    if len(k_segs) == 0:
        return np.full(n_draws, 0.005)
    p = weights / weights.sum()
    n = len(k_segs)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        idx = rng.choice(n, size=n, replace=True, p=p)
        ks = k_segs[idx]
        ws = weights[idx]
        order = np.argsort(ks)
        cw = np.cumsum(ws[order])
        draws[i] = ks[order][np.searchsorted(cw, 0.5 * cw[-1])]
    return np.maximum(draws, 1e-4)


def _lag_gate_weights(h_smooth: np.ndarray, po: np.ndarray, h_base: float,
                      k: float, r_cutoff_m: float) -> np.ndarray:
    """Evidence weights for the lag-gate ensemble from the positive
    rain->input cross-correlation profile of the reconstructed series."""
    hat = reconstruct_recession_baseline(h_smooth - h_base, k)
    n = len(hat)
    u = np.zeros(n)
    for t in range(n - 1):
        if np.isfinite(hat[t]) and np.isfinite(hat[t + 1]):
            u[t] = max((hat[t + 1] - hat[t]) + k * hat[t], 0.0)
    r = np.where(np.isfinite(po[:n]), po[:n], 0.0)
    max_lag = LAG_GATE_ENSEMBLE[-1] + 6
    c = np.zeros(max_lag + 1)
    for L in range(max_lag + 1):
        a = r[:-L] if L else r
        b = u[L:] if L else u
        if len(a) > 30 and a.std() > 0 and b.std() > 0:
            c[L] = max(float(np.corrcoef(a, b)[0, 1]), 0.0)
    cum = np.cumsum(c)
    total = max(cum[-1], 1e-12)
    # incremental mass admitted by each successive window
    w = []
    prev = 0.0
    for L in LAG_GATE_ENSEMBLE:
        w.append(max(cum[L] / total - prev, 0.0))
        prev = cum[L] / total
    w = np.asarray(w) + LAG_GATE_WEIGHT_FLOOR
    return w / w.sum()


def _U_given_k(h_smooth: np.ndarray, po: np.ndarray, h_base: float,
               k: float, n_years: float, r_cutoff_m: float,
               rain_lag_window: int = 14) -> float:
    """Pumping-robust annual input for a given recession constant."""
    hat = reconstruct_recession_baseline(h_smooth - h_base, k)
    n = len(hat)
    u = np.full(n, np.nan)
    for t in range(n - 1):
        if np.isfinite(hat[t]) and np.isfinite(hat[t + 1]):
            u[t] = (hat[t + 1] - hat[t]) + k * hat[t]
    rain = _rain_recent_mask(po[:n], r_cutoff_m, rain_lag_window)
    U = float(np.nansum(np.where(rain & np.isfinite(u),
                                 np.maximum(u, 0.0), 0.0)))
    return U * 1000.0 / n_years


def recharge_posterior(
    po_m: np.ndarray,
    ho_m: np.ndarray,
    annual_rain_mm: float,
    n_draws: int = 1000,
    seed: int = 0,
    sy_prior_mean: float = DEFAULT_SY_PRIOR_MEAN,
    sy_prior_std: float = DEFAULT_SY_PRIOR_STD,
    rch_coef_mean: float = DEFAULT_RCH_COEF_MEAN,
    rch_coef_std: float = DEFAULT_RCH_COEF_STD,
    bias_factor_mean: float = DEFAULT_BIAS_FACTOR_MEAN,
    bias_factor_std: float = DEFAULT_BIAS_FACTOR_STD,
    correct_bias: bool = False,
    r_cutoff_m: float = 0.002,
) -> RechargePosterior:
    """Monte Carlo recharge posterior with joint k / U / Sy propagation
    and marginalisation over the structural processing choices
    (smoothing window, rain-gate lag window)."""
    rng = np.random.default_rng(seed)
    raw = np.asarray(ho_m, dtype=float)
    po = np.asarray(po_m, dtype=float)
    n_years = len(raw) / 365.25
    h_base = estimate_h_base(raw)

    # Pre-smooth once per ensemble member (the expensive part).
    # Median filtering is edge-preserving: it removes the rise-attenuation
    # bias of the moving mean while suppressing noise equally well.
    smooths = {w: _nan_smooth(raw, w, kind="median") for w in SMOOTH_ENSEMBLE}

    # Per-spell recession rates on the default smoothing (k is robust to
    # the window choice; the window enters U directly instead).
    k_segs, weights, _ = recession_k_segments(smooths[5], po, h_base,
                                              r_cutoff_m=r_cutoff_m)
    k_draws = _weighted_bootstrap_k(k_segs, weights, n_draws, rng)

    win_draws = rng.choice(SMOOTH_ENSEMBLE, size=n_draws)
    k_ref = float(np.median(k_draws))
    gate_w = _lag_gate_weights(smooths[5], po, h_base, k_ref, r_cutoff_m)
    lag_draws = rng.choice(LAG_GATE_ENSEMBLE, size=n_draws, p=gate_w)
    if correct_bias:
        bias = rng.normal(bias_factor_mean, bias_factor_std, size=n_draws)
        bias = np.clip(bias, 1.0, None)   # never deflate below the raw value
    else:
        bias = np.ones(n_draws)

    U_draws = np.empty(n_draws)
    for i in range(n_draws):
        U_draws[i] = bias[i] * _U_given_k(smooths[int(win_draws[i])], po,
                                          h_base, k_draws[i], n_years,
                                          r_cutoff_m,
                                          rain_lag_window=int(lag_draws[i]))
    U_draws = np.maximum(U_draws, 1e-9)

    # Conjugate fusion per draw: Sy | U ~ product of the two Gaussians.
    syB = rch_coef_mean * annual_rain_mm / U_draws
    syB_s = rch_coef_std * annual_rain_mm / U_draws
    pA = 1.0 / sy_prior_std ** 2
    pB = 1.0 / syB_s ** 2
    mu_post = (sy_prior_mean * pA + syB * pB) / (pA + pB)
    sd_post = np.sqrt(1.0 / (pA + pB))
    sy_draws = rng.normal(mu_post, sd_post)
    sy_draws = np.clip(sy_draws, 1e-3, 0.6)

    rch = sy_draws * U_draws

    U_med = float(np.median(U_draws))
    syB_med = rch_coef_mean * annual_rain_mm / U_med
    syB_s_med = rch_coef_std * annual_rain_mm / U_med
    consistency = abs(sy_prior_mean - syB_med) / float(
        np.hypot(sy_prior_std, syB_s_med))

    q = np.percentile(rch, [2.5, 16, 50, 84, 97.5])
    return RechargePosterior(
        rch_draws_mm=rch, sy_draws=sy_draws, U_draws_mm=U_draws,
        k_draws=k_draws,
        rch_median_mm=float(q[2]),
        rch_ci68_mm=(float(q[1]), float(q[3])),
        rch_ci95_mm=(float(q[0]), float(q[4])),
        sy_median=float(np.median(sy_draws)),
        consistency_sigma=float(consistency),
    )


__all__ = [
    "RechargePosterior",
    "recharge_posterior",
    "DEFAULT_BIAS_FACTOR_MEAN",
    "DEFAULT_BIAS_FACTOR_STD",
]
