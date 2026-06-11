"""Unit tests for the free-Sy recession-corrected WTF inversion.

Self-contained: each test builds a tiny linear-reservoir well in-place,
so no external (gitignored) benchmark data is required.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from framework_v2.free_sy_inversion import (
    estimate_recession_k,
    head_equivalent_input,
    invert_free_sy,
    implied_sy,
)


def make_reservoir(n_days=1500, k=0.01, sy=0.08, h_base=0.0,
                   event_every=20, event_rain_mm=30.0, seed=0):
    """Forward-simulate h(t+1)=h_base+(1-k)(h-h_base)+R/Sy with sparse
    rainfall events. Returns (rain_m, head_m)."""
    rng = np.random.default_rng(seed)
    rain = np.zeros(n_days)
    rain[::event_every] = event_rain_mm           # mm
    rain_m = rain / 1000.0
    recharge_m = 0.4 * rain_m                      # 40% of rain becomes recharge
    h = np.empty(n_days)
    h[0] = h_base + 2.0
    for i in range(n_days - 1):
        h[i + 1] = h_base + (1 - k) * (h[i] - h_base) + recharge_m[i] / sy
    return rain_m, h


def test_recession_k_recovered_clean():
    rain_m, h = make_reservoir(k=0.012, sy=0.08)
    k_est, _ = estimate_recession_k(h, rain_m, h_base=0.0)
    assert abs(k_est - 0.012) < 0.004, k_est


def test_head_equivalent_input_zero_on_pure_recession():
    # No rain at all -> pure recession -> u(t) = dh + k(h-h_base) ~ 0
    n = 500
    k = 0.01
    h = 5.0 * (1 - k) ** np.arange(n)              # exact recession to 0
    rain_m = np.zeros(n)
    u = head_equivalent_input(h, k=k, h_base=0.0)
    assert np.nanmax(np.abs(u)) < 1e-6


def test_recharge_scales_linearly_with_sy():
    rain_m, h = make_reservoir(k=0.01, sy=0.08)
    r1 = invert_free_sy(rain_m, h, sy_override=0.05, smooth_window=1)
    r2 = invert_free_sy(rain_m, h, sy_override=0.10, smooth_window=1)
    # Same U, double Sy -> double recharge (the equifinality line)
    assert abs(r1.U_head_m - r2.U_head_m) < 1e-9
    assert abs(r2.annual_rch_mm / r1.annual_rch_mm - 2.0) < 1e-6


def test_implied_sy_matches_true_sy():
    sy_true = 0.08
    rain_m, h = make_reservoir(k=0.01, sy=sy_true)
    r = invert_free_sy(rain_m, h, smooth_window=1)
    rch_true = sy_true * r.U_head_m * 1000.0 / r.n_years
    sy_imp = implied_sy(r.U_head_m, rch_true, r.n_years)
    assert abs(sy_imp - sy_true) < 1e-6


def test_band_brackets_point_estimate():
    rain_m, h = make_reservoir()
    r = invert_free_sy(rain_m, h, sy_prior_mean=0.07, sy_prior_std=0.03)
    assert r.annual_rch_lo_mm <= r.annual_rch_mm <= r.annual_rch_hi_mm
