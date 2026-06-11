"""Unit tests for the pumping-robust head-input estimator.

Self-contained: builds a recession+recharge well with an injected pumping
episode and checks that the recession-reconstruction estimator is far less
sensitive to the pumping recovery than the raw recession-corrected sum.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from framework_v2.pumping_robust import (
    estimate_U_pumping_robust,
    reconstruct_recession_baseline,
)


def _build_well(k=0.01, sy=0.1, n=600, event_every=25, rain_mm=30.0,
                pump_day=300, pump_depth=0.4, pump_tau=5.0, h_base=0.0):
    """Recession + sparse recharge, optionally with one pumping episode."""
    rain = np.zeros(n)
    rain[::event_every] = rain_mm
    rain_m = rain / 1000.0
    rch = 0.4 * rain_m
    h = np.empty(n)
    h[0] = h_base + 2.0
    for i in range(n - 1):
        h[i + 1] = h_base + (1 - k) * (h[i] - h_base) + rch[i] / sy
    return rain_m, h


def _inject_pumping(h, pump_day=300, depth=0.4, tau=5.0, dur=4):
    h = h.copy()
    # gradual drawdown over `dur` days then exponential recovery
    for j in range(dur):
        if pump_day + j < len(h):
            h[pump_day + j] -= depth * (j + 1) / dur
    for j in range(dur, 40):
        t = pump_day + j
        if t < len(h):
            h[t] -= depth * np.exp(-(j - dur) / tau)
    return h


def test_reconstruction_is_pumping_robust():
    rain, h_clean = _build_well()
    h_pump = _inject_pumping(h_clean)
    U_clean = estimate_U_pumping_robust(rain, h_clean).U_annual_mm
    U_pump = estimate_U_pumping_robust(rain, h_pump).U_annual_mm
    # the pumping episode should change U by less than 15%
    assert abs(U_pump - U_clean) / U_clean < 0.15, (U_clean, U_pump)


def test_reconstruction_rejects_pumping_recovery_on_rain_days():
    # Pure recession (no genuine recharge) with a pumping dip+recovery that
    # overlaps rain days: the raw recession-corrected sum counts the recovery
    # rises as recharge; the reconstruction rejects them.
    n, k, h_base = 200, 0.02, 0.0
    x = 3.0 * (1 - k) ** np.arange(n)          # pure recession deficit
    h = x + h_base
    h[50:90] -= 0.6 * np.exp(-(np.arange(40)) / 8.0)  # drawdown + recovery
    rain_m = np.zeros(n)
    rain_m[50:90] = 0.01                        # rain flags over the recovery
    dh = np.diff(h)
    u_raw = dh + k * (h[:-1] - h_base)
    rain_mask = rain_m[:len(u_raw)] > 0.002
    U_raw = np.sum(np.where(rain_mask, np.maximum(u_raw, 0), 0))   # total (m)
    res = estimate_U_pumping_robust(rain_m, h, smooth_window=1)
    U_rec_total = res.U_annual_mm * res.n_years / 1000.0           # total (m)
    # raw counts the spurious recovery; reconstruction rejects most of it
    assert U_raw > 0.3
    assert U_rec_total < 0.5 * U_raw


def test_reconstruction_preserves_clean_recharge():
    rain, h_clean = _build_well()
    r = estimate_U_pumping_robust(rain, h_clean)
    # on clean data the reconstruction should track the head (no spurious
    # flattening collapses U to zero)
    assert r.U_annual_mm > 0
    assert np.isfinite(r.h_reconstructed).all()


def test_baseline_holds_through_dip():
    # a pure dip-and-recover (no recharge) must not raise the baseline
    n = 100
    k = 0.02
    x = 2.0 * (1 - k) ** np.arange(n)        # pure recession deficit
    x_dip = x.copy()
    x_dip[40:50] -= 0.3                        # dip then return
    hat = reconstruct_recession_baseline(x_dip, k)
    # reconstructed series should not exceed the original recession anywhere
    assert np.all(hat <= x + 1e-9)


def test_response_score_separates_wtf_from_nonwtf():
    from framework_v2.pumping_robust import wtf_response_score
    # good WTF well: recharge-driven rises follow rain
    rain, h = _build_well(k=0.01, sy=0.1, n=1200, event_every=18)
    good = wtf_response_score(rain, h)
    # dead well: pure recession + noise, no recharge response
    rng = np.random.default_rng(3)
    n = 1200
    dead = 3.0 * (1 - 0.01) ** np.arange(n) + 10.0 + rng.normal(0, 0.02, n)
    bad = wtf_response_score(rain, dead)
    assert good > 0.2
    assert bad < 0.15
    assert good > bad


def test_zeta_flags_gross_prior_conflict():
    # data-anchored Sy ~ 0.1; a wildly wrong prior must raise zeta
    from framework_v2.pumping_robust import estimate_U_pumping_robust
    from framework_v2.water_balance import constrain_recharge_from_U
    rain, h = _build_well(k=0.01, sy=0.1, n=1500, event_every=15)
    P = rain.sum() * 1000 / (len(h) / 365.25)
    pr = estimate_U_pumping_robust(rain, h, rain_lag_window=7)
    # correct-ish coefficient so the water-balance arm anchors near truth
    c = (0.4 * rain.sum() * 1000 / (len(h) / 365.25)) / P
    z_ok = constrain_recharge_from_U(pr.U_annual_mm, pr.n_years, P,
                                     sy_prior_mean=0.10, sy_prior_std=0.03,
                                     rch_coef_mean=c, rch_coef_std=0.03).consistency_sigma
    z_bad = constrain_recharge_from_U(pr.U_annual_mm, pr.n_years, P,
                                      sy_prior_mean=0.30, sy_prior_std=0.03,
                                      rch_coef_mean=c, rch_coef_std=0.03).consistency_sigma
    assert z_bad > z_ok
    assert z_bad > 2.0
