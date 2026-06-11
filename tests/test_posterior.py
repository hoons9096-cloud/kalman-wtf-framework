"""Unit tests for the Monte Carlo recharge posterior."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from framework_v2.posterior import recharge_posterior
from framework_v2.free_sy_inversion import recession_k_segments


def _well(n=900, k=0.01, sy=0.08, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    rain = np.zeros(n)
    rain[::20] = 30.0
    rain_m = rain / 1000.0
    rch = 0.4 * rain_m
    h = np.empty(n)
    h[0] = 2.0
    for i in range(n - 1):
        h[i + 1] = (1 - k) * h[i] + rch[i] / sy
    return rain_m, h + rng.normal(0, noise, n)


def test_posterior_shapes_and_order():
    rain, h = _well()
    P = rain.sum() * 1000 / (len(h) / 365.25)
    p = recharge_posterior(rain, h, P, n_draws=200, seed=1)
    assert len(p.rch_draws_mm) == 200
    lo95, hi95 = p.rch_ci95_mm
    lo68, hi68 = p.rch_ci68_mm
    assert lo95 <= lo68 <= p.rch_median_mm <= hi68 <= hi95
    assert np.all(p.sy_draws > 0)
    assert np.all(p.U_draws_mm > 0)


def test_posterior_reproducible():
    rain, h = _well()
    P = rain.sum() * 1000 / (len(h) / 365.25)
    a = recharge_posterior(rain, h, P, n_draws=150, seed=7)
    b = recharge_posterior(rain, h, P, n_draws=150, seed=7)
    assert a.rch_median_mm == b.rch_median_mm
    assert np.array_equal(a.rch_draws_mm, b.rch_draws_mm)


def test_lag_gate_weights_valid_and_evidence_concentrated():
    from framework_v2.posterior import (
        _lag_gate_weights, LAG_GATE_ENSEMBLE)
    from framework_v2.pumping_robust import _nan_smooth
    # fast-responding well (rise on the rain day) with *aperiodic* rain
    # (periodic forcing would alias the CCF profile): evidence should
    # concentrate the gate distribution on the shortest window
    rng = np.random.default_rng(11)
    n, k, sy = 900, 0.01, 0.08
    rain = np.zeros(n)
    rain[rng.choice(n - 10, size=45, replace=False)] = 30.0
    rain_m = rain / 1000.0
    rch = 0.4 * rain_m
    h = np.empty(n)
    h[0] = 2.0
    for i in range(n - 1):
        h[i + 1] = (1 - k) * h[i] + rch[i] / sy
    h = h + rng.normal(0, 0.005, n)
    rain = rain_m
    hs = _nan_smooth(h, 5, kind="median")
    w = _lag_gate_weights(hs, rain, h_base=0.0, k=0.01, r_cutoff_m=0.002)
    assert len(w) == len(LAG_GATE_ENSEMBLE)
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.all(w > 0)
    assert w[0] == w.max()          # same-day gate dominates


def test_k_segments_exposed():
    rain, h = _well(noise=0.0)
    ks, ws, rmse = recession_k_segments(h, rain, h_base=0.0)
    assert len(ks) == len(ws) and len(ks) > 0
    # clean reservoir: per-spell estimates near the true k
    assert abs(np.median(ks) - 0.01) < 0.005
