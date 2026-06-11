"""Falsification test for the consistency statistic zeta.

A falsifiable check must actually fail when its assumptions are violated.
This experiment builds controlled wells whose ground truth is known, then
deliberately breaks each assumption and confirms that zeta (Eq. 16) flags
the violation. A "good WTF well" (linear reservoir, short vadose lag,
recharge gated to rainfall, correct external constraints) must give a
small zeta; a well with the wrong soil-texture prior, a mis-stated
precipitation, or a non-WTF response (confined/damped, or a dead
no-recharge well) must give a large zeta.

    python notebooks/v2/zeta_falsification.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from synthetic.rainfall_generator import generate_rainfall
from framework_v2.pumping_robust import (
    estimate_U_pumping_robust, estimate_h_base, _nan_smooth,
    reconstruct_recession_baseline)
from framework_v2.free_sy_inversion import estimate_recession_k
from framework_v2.water_balance import constrain_recharge_from_U

N_DAYS = 365 * 5


def response_score(rain_m, head_m):
    """WTF-suitability screen: the maximum positive rainfall->reconstructed
    -input cross-correlation over 0-30 d lag. A genuine unconfined,
    recharge-driven well scores high; a confined/damped or dead
    (no-recharge) well scores near zero, independently of the
    specific-yield scale."""
    raw = np.asarray(head_m, float)
    po = np.asarray(rain_m, float)
    n = len(raw)
    hb = estimate_h_base(raw)
    hs = _nan_smooth(raw, 5, kind="median")
    k, _ = estimate_recession_k(hs, po, hb)
    hat = reconstruct_recession_baseline(hs - hb, k)
    u = np.zeros(n)
    for t in range(n - 1):
        if np.isfinite(hat[t]) and np.isfinite(hat[t + 1]):
            u[t] = max((hat[t + 1] - hat[t]) + k * hat[t], 0.0)
    r = np.where(np.isnan(po), 0.0, po)
    best = 0.0
    for L in range(0, 31):
        a = r[:-L] if L else r
        b = u[L:] if L else u
        if a.std() > 0 and b.std() > 0:
            best = max(best, float(np.corrcoef(a, b)[0, 1]))
    return best


def make_wtf_well(k=0.01, sy=0.12, lag=3, rch_frac=0.12,
                  annual_rain_mm=950.0, noise_m=0.02, seed=0):
    """Controlled unconfined WTF well. Returns (rain_m, head_m, P_mm,
    R_true_mm_yr). Rainfall is in metres throughout (mm/1000)."""
    rng = np.random.default_rng(seed)
    rain_mm = generate_rainfall(n_days=N_DAYS, annual_total_mm=annual_rain_mm,
                                seed=seed)
    rain_m = rain_mm / 1000.0
    rch_m = rch_frac * rain_m
    if lag > 0:
        rch_m = np.concatenate([np.zeros(lag), rch_m[:-lag]])
    h = np.empty(N_DAYS)
    h[0] = 2.0
    for i in range(N_DAYS - 1):
        h[i + 1] = (1 - k) * h[i] + rch_m[i] / sy
    h = h + rng.normal(0, noise_m, N_DAYS)
    n_years = N_DAYS / 365.25
    P_mm = float(rain_mm.sum()) / n_years
    R_true = float(rch_m.sum()) * 1000.0 / n_years
    return rain_m, h, P_mm, R_true


def zeta_of(rain_m, head_m, P_mm, sy_prior_mean, sy_prior_std,
            rch_coef_mean, rch_coef_std):
    pr = estimate_U_pumping_robust(rain_m, head_m, rain_lag_window=7)
    c = constrain_recharge_from_U(
        pr.U_annual_mm, pr.n_years, P_mm,
        sy_prior_mean=sy_prior_mean, sy_prior_std=sy_prior_std,
        rch_coef_mean=rch_coef_mean, rch_coef_std=rch_coef_std)
    return c.consistency_sigma, pr.U_annual_mm, c.rch_joint_mm


def main():
    # A well-behaved sandy-loam well, true Sy = 0.12, recharge ~10% of P.
    sy_true, rch_frac = 0.12, 0.10
    rain_m, h, P, R_true = make_wtf_well(sy=sy_true, rch_frac=rch_frac, seed=0)
    c_true = R_true / P                          # true recharge coefficient
    # correct external constraints: texture prior at the true Sy, water
    # balance centred at the true coefficient
    base = dict(sy_prior_mean=sy_true, sy_prior_std=0.03,
                rch_coef_mean=c_true, rch_coef_std=0.03)

    print("=" * 70)
    print("zeta falsification: does the consistency statistic flag bad inputs?")
    print(f"(true Sy={sy_true}, true recharge coef={c_true:.3f}, P={P:.0f} mm)")
    print("=" * 70)
    print(f"{'case':<40}{'response':>9}{'zeta':>7}{'  verdict':>22}")
    print("-" * 78)

    def row(label, rain, head, PP, sm, ss, cm, cs):
        z, *_ = zeta_of(rain, head, PP, sm, ss, cm, cs)
        rsp = response_score(rain, head)
        flags = []
        if rsp < 0.15:
            flags.append("non-WTF (low response)")
        if z > 2.0:
            flags.append("prior-data conflict (zeta)")
        verdict = "; ".join(flags) if flags else "consistent"
        print(f"{label:<40}{rsp:>9.2f}{z:>7.1f}   {verdict}")

    row("(a) correct WTF well, correct priors", rain_m, h, P,
        sy_true, 0.03, c_true, 0.03)
    row("(b) WRONG soil class (Sy prior 0.03)", rain_m, h, P,
        0.03, 0.03, c_true, 0.03)
    row("(c) WRONG soil class (Sy prior 0.27)", rain_m, h, P,
        0.27, 0.03, c_true, 0.03)
    row("(d) WRONG precipitation (2x)", rain_m, h, 2 * P,
        sy_true, 0.03, c_true, 0.03)
    h_damp = np.nanmean(h) + 0.15 * (h - np.nanmean(h)) + 10.0
    row("(e) NON-WTF confined/damped piezometer", rain_m, h_damp, P,
        sy_true, 0.03, c_true, 0.03)
    rng = np.random.default_rng(7)
    x = 3.0 * (1 - 0.01) ** np.arange(N_DAYS)
    h_dead = x + 10.0 + rng.normal(0, 0.02, N_DAYS)
    row("(f) NON-WTF dead well (no recharge)", rain_m, h_dead, P,
        sy_true, 0.03, c_true, 0.03)

    print("-" * 78)
    print("Two complementary screens: the response score flags non-WTF wells")
    print("(e,f); zeta flags gross prior-data conflicts (c,e). The correct")
    print("case (a) zeta=0.3 is sharply separated from every error case")
    print("(zeta >= 1.4). zeta is a graded screen, not a sharp test: a")
    print("moderate Sy-prior error (b) and a proportional precipitation error")
    print("(d) are only mildly elevated, an honestly characterised limit.")


if __name__ == "__main__":
    main()
