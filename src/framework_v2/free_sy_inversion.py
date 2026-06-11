"""Free-Sy WTF inversion with recession-corrected recharge input.

This module replaces the broken `filpor_tr` specific-yield engine (which
returns its floor value on ~99.8 % of days for realistic head rises) and
the raw-increment WTF identity (which inflates the apparent Sy by the
recession factor).  It makes the identifiability structure of the WTF
method explicit.

Model
-----
Linear-reservoir head dynamics (matching the synthetic generator
`synthetic.aquifer_state.simulate_aquifer`):

    h(t+1) - h_base = (1 - k) (h(t) - h_base) + u(t)

where ``k`` is the recession constant (per day) and ``u(t) = R(t)/Sy``
is the **head-equivalent recharge input** (metres of head added by
recharge on day t).

Two facts drive everything:

1.  ``k`` is identifiable from dry-period head decline alone
    (independent of Sy): on rain-free days u(t) = 0, so
    h(t+1)-h_base = (1-k)(h(t)-h_base).

2.  Given ``k``, the head-equivalent input is recovered as the
    recession-corrected rise

        u(t) = [h(t+1) - h_base] - (1-k) [h(t) - h_base]
             =  Δh(t) + k (h(t) - h_base)

    and its positive part summed over the record, ``U = Σ max(u,0)``,
    is **identifiable** from head data.

The recharge is then

        R_annual = Sy · U · 1000 / n_years   [mm/yr]

so recharge is the product of an *identifiable* head-rise integral ``U``
and a *non-identifiable* specific yield ``Sy``.  Head data constrain
``U`` but say nothing about ``Sy``: the head-fit residual is exactly
invariant to ``Sy`` (the equifinality).  ``Sy`` must come from a prior
(literature / barometric / pumping test) or an external scale
constraint (water balance).  This is the crux the paper formalises.

Why this fixes the v2 pathologies
---------------------------------
* No `filpor_tr` → no floor collapse, no negative fillable porosity.
* Recession-corrected ``u`` (vs raw Δh) removes the apparent-Sy
  inflation, so the Sy that reproduces the true recharge is the
  generator's actual mean daily Sy, not the recession-biased operational
  value.  Recharge accuracy and Sy accuracy become consistent (no more
  "right recharge from wrong Sy" compensating error).
* Recharge depends on the lag only through ``U``, which is far less
  lag-sensitive than the old (k, z, sn) optimisation, removing most of
  the configuration fragility.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Field-effective Sy prior (Healy & Cook 2002; Johnson 1967) — same as
# the rest of framework_v2, kept here so this module is self-contained.
DEFAULT_SY_PRIOR_MEAN = 0.07
DEFAULT_SY_PRIOR_STD = 0.03


@dataclass
class FreeSyResult:
    k: float                    # recession constant (per day), data-driven
    h_base: float               # reservoir base level (m)
    U_head_m: float             # Σ positive head-equivalent recharge input (m)
    sy_prior_mean: float
    sy_prior_std: float
    n_years: float
    annual_rch_mm: float        # MAP recharge = sy_post * U * 1000 / yr
    annual_rch_lo_mm: float     # ±1σ band from the Sy prior
    annual_rch_hi_mm: float
    sy_used: float              # Sy applied (= prior mean unless constrained)
    rmse_recession_m: float     # fit quality of the recession-only model


def estimate_h_base(ho_m: np.ndarray, drop_m: float = 2.0) -> float:
    """Reservoir base level: min observed head minus a buffer, floored at 0
    (matches `framework.kalman_wtf.run_model_core`)."""
    hb = float(np.nanmin(ho_m)) - drop_m
    return max(hb, 0.0)


def _dry_runs(po: np.ndarray, r_cutoff_m: float,
              min_run: int) -> list[tuple[int, int]]:
    """Return [start, end) index spans of consecutive rain-free days of
    length >= min_run."""
    runs = []
    i = 0
    n = len(po)
    while i < n:
        if po[i] <= r_cutoff_m:
            j = i
            while j < n and po[j] <= r_cutoff_m:
                j += 1
            if j - i >= min_run:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def estimate_recession_k(
    ho_m: np.ndarray,
    po_m: np.ndarray,
    h_base: float,
    r_cutoff_m: float = 0.002,
    min_above_base_m: float = 0.05,
    min_run: int = 8,
) -> tuple[float, float]:
    """Estimate the recession constant ``k`` from dry-period head decline.

    On rain-free days u(t)=0, so (h(t+1)-h_base) = (1-k)(h(t)-h_base).
    The estimate is Sy-independent, but only *genuine* recession may be
    used: during a vadose-lagged system the early part of a dry spell
    can still be rising from delayed recharge.  We therefore (a) restrict
    to sustained dry runs (>= ``min_run`` days) and (b) within each run
    use only the monotonically declining tail after the run's head peak.
    The slope (1-k) is a robust least-squares-through-origin fit of the
    next-day deficit on the current deficit over those declining pairs.

    Returns
    -------
    (k, rmse) : recession constant (per day) and the RMSE of the
        recession-only one-step prediction on the declining pairs (m).
    """
    k_segs, weights, rmse = recession_k_segments(
        ho_m, po_m, h_base, r_cutoff_m=r_cutoff_m,
        min_above_base_m=min_above_base_m, min_run=min_run)
    if len(k_segs) == 0:
        return 0.005, np.nan
    # length-weighted median of per-segment decay rates
    order = np.argsort(k_segs)
    ks = np.asarray(k_segs)[order]
    ws = np.asarray(weights, dtype=float)[order]
    cw = np.cumsum(ws)
    k = float(ks[np.searchsorted(cw, 0.5 * cw[-1])])
    return max(k, 1e-4), rmse


def recession_k_segments(
    ho_m: np.ndarray,
    po_m: np.ndarray,
    h_base: float,
    r_cutoff_m: float = 0.002,
    min_above_base_m: float = 0.05,
    min_run: int = 8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-dry-spell recession-rate estimates (k_segs, weights, rmse).

    Exposes the spell-to-spell spread of the multi-day log-linear fits so
    that the *uncertainty* of the pooled recession constant can be
    propagated (e.g. by bootstrap resampling in the posterior module).
    Weights are the per-spell sample counts used in the regression.
    """
    ho = np.asarray(ho_m, dtype=float)
    po = np.asarray(po_m, dtype=float)

    # Daily recession change (k * deficit) is comparable to observation
    # noise for slow systems, so a day-to-day slope is unidentifiable.
    # Instead fit each recession tail over its full multi-day length:
    #   deficit(t) = deficit_0 (1-k)^t  =>  log deficit linear in t.
    k_segs, weights, sse, ssn = [], [], 0.0, 0
    for (lo, hi) in _dry_runs(po, r_cutoff_m, min_run):
        seg = ho[lo:hi]
        if np.all(np.isnan(seg)):
            continue
        peak = int(np.nanargmax(seg))               # local recession start
        tail = ho[lo + peak:hi] - h_base
        t = np.arange(len(tail), dtype=float)
        ok = np.isfinite(tail) & (tail > min_above_base_m)
        if ok.sum() < 5:
            continue
        tt, dd = t[ok], tail[ok]
        # log-linear regression: log(deficit) = log(d0) + t*log(1-k)
        A = np.column_stack([np.ones_like(tt), tt])
        sol, *_ = np.linalg.lstsq(A, np.log(dd), rcond=None)
        slope_log = sol[1]
        k_seg = 1.0 - float(np.exp(slope_log))
        if -0.05 < k_seg < 0.5:                     # reject noise-only fits
            k_segs.append(k_seg)
            weights.append(int(ok.sum()))
            pred = np.exp(A @ sol)
            sse += float(np.sum((dd - pred) ** 2)); ssn += int(ok.sum())
    rmse = float(np.sqrt(sse / ssn)) if ssn else np.nan
    return np.asarray(k_segs), np.asarray(weights, dtype=float), rmse


def head_equivalent_input(
    ho_m: np.ndarray,
    k: float,
    h_base: float,
) -> np.ndarray:
    """Per-day head-equivalent recharge input u(t) = Δh + k(h-h_base).

    NaN-bridged days contribute 0. Only the positive part is recharge.
    """
    ho = np.asarray(ho_m, dtype=float)
    u = np.full(len(ho), np.nan)
    for i in range(len(ho) - 1):
        if np.isfinite(ho[i]) and np.isfinite(ho[i + 1]):
            u[i] = (ho[i + 1] - ho[i]) + k * (ho[i] - h_base)
    return u


def _smooth(ho_m: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware centred moving average. Observation noise inflates the
    positive-part sum Σ max(u,0); averaging over `window` days reduces
    the noise std by ~sqrt(window) while the vadose-lagged recharge
    rises (already gradual over many days) are largely preserved."""
    if window <= 1:
        return np.asarray(ho_m, dtype=float)
    ho = np.asarray(ho_m, dtype=float)
    n = len(ho)
    out = np.full(n, np.nan)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = ho[lo:hi]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


def invert_free_sy(
    po_m: np.ndarray,
    ho_m: np.ndarray,
    sy_prior_mean: float = DEFAULT_SY_PRIOR_MEAN,
    sy_prior_std: float = DEFAULT_SY_PRIOR_STD,
    sy_override: float | None = None,
    r_cutoff_m: float = 0.002,
    smooth_window: int = 7,
) -> FreeSyResult:
    """Recession-corrected WTF inversion with a free / prior-supplied Sy.

    Parameters
    ----------
    sy_override : float or None
        If given, use this Sy (e.g. from a water-balance constraint or a
        sensitivity sweep) instead of the prior mean.
    smooth_window : int, default 7
        Centred moving-average window applied before computing the
        head-equivalent input, to de-bias the noise inflation of
        Σ max(u,0). 1 disables smoothing.
    """
    ho_raw = np.asarray(ho_m, dtype=float)
    po = np.asarray(po_m, dtype=float)
    n_years = len(ho_raw) / 365.25

    h_base = estimate_h_base(ho_raw)
    ho = _smooth(ho_raw, smooth_window)
    k, rmse_rec = estimate_recession_k(ho, po, h_base, r_cutoff_m=r_cutoff_m)

    u = head_equivalent_input(ho, k, h_base)
    # recharge only where it rains (u on dry days is recession noise)
    rain_mask = po > r_cutoff_m
    u_pos = np.where(rain_mask[: len(u)] & np.isfinite(u),
                     np.maximum(u, 0.0), 0.0)
    U = float(np.nansum(u_pos))

    sy = sy_override if sy_override is not None else sy_prior_mean
    rch = sy * U * 1000.0 / n_years
    rch_lo = max(sy_prior_mean - sy_prior_std, 0.0) * U * 1000.0 / n_years
    rch_hi = (sy_prior_mean + sy_prior_std) * U * 1000.0 / n_years

    return FreeSyResult(
        k=float(k), h_base=float(h_base), U_head_m=U,
        sy_prior_mean=float(sy_prior_mean), sy_prior_std=float(sy_prior_std),
        n_years=float(n_years),
        annual_rch_mm=float(rch),
        annual_rch_lo_mm=float(rch_lo), annual_rch_hi_mm=float(rch_hi),
        sy_used=float(sy), rmse_recession_m=float(rmse_rec),
    )


def implied_sy(U_head_m: float, annual_rch_true_mm: float,
               n_years: float) -> float:
    """The Sy that would reproduce the *true* annual recharge given the
    identifiable head-rise integral U.  Used to check whether the
    consistent Sy matches the literature prior (it should, once the
    recession correction is applied)."""
    if U_head_m <= 0:
        return np.nan
    return annual_rch_true_mm * n_years / (U_head_m * 1000.0)


__all__ = [
    "FreeSyResult",
    "estimate_h_base",
    "estimate_recession_k",
    "recession_k_segments",
    "head_equivalent_input",
    "invert_free_sy",
    "implied_sy",
    "DEFAULT_SY_PRIOR_MEAN",
    "DEFAULT_SY_PRIOR_STD",
]
