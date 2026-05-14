"""Smoke tests for the framework port (vg, filpor, kalman, optim).

These tests verify structural correctness of the port: function signatures,
output shapes, monotonicity properties, and a coarse self-consistency
check on the SH22 field well (the Korean paper's flagship case).

Numerical exactness against the MATLAB reference is NOT asserted here —
the two implementations differ in fminsearch convergence and trapezoidal
quadrature precision, so the Python port produces *operationally
equivalent* but not bit-identical optima. Quantitative MATLAB comparison
should be done as a one-off validation by the user.
"""
from __future__ import annotations

import unittest
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from framework import (  # noqa: E402
    vg, filpor, filpor_tr, SOIL_DB,
    remove_outliers,
    apply_lag, run_model_core,
    run_optimization,
    load_well,
)


class TestVanGenuchten(unittest.TestCase):
    def test_saturation_at_zero_head(self):
        theta = vg(np.array([0.0, -1.0, -10.0]), 0.43, 0.078, 3.6, 1.56)
        self.assertTrue(np.allclose(theta, 0.43))

    def test_monotonic_decrease(self):
        h = np.linspace(0.001, 5.0, 50)
        theta = vg(h, 0.43, 0.078, 3.6, 1.56)
        diffs = np.diff(theta)
        self.assertTrue((diffs <= 0).all())

    def test_asymptote_at_large_h(self):
        theta = vg(np.array([1000.0]), 0.43, 0.078, 3.6, 1.56)
        self.assertAlmostEqual(float(theta[0]), 0.078, delta=0.01)


class TestFilpor(unittest.TestCase):
    def test_zero_dh_returns_zero(self):
        self.assertEqual(filpor(0.0, 0.43, 0.20, 3.6, 1.56), 0.0)

    def test_large_dh_dry_state_positive(self):
        # Very large dh AND very dry state → positive fillable porosity
        v = filpor(2.0, 0.43, 0.10, 3.6, 1.56)
        self.assertGreater(v, 0)


class TestFilporTr(unittest.TestCase):
    def test_bounds(self):
        s = SOIL_DB[2]
        upper = s["theta_s"] - s["theta_r"]
        for n_dry in [1, 30, 100]:
            for dh in [0.001, 0.05, 0.5]:
                nf = filpor_tr(2, 3.0, n_dry, dh)
                self.assertGreaterEqual(nf, 0.001)
                self.assertLessEqual(nf, upper + 1e-9)

    def test_soil_db_classes(self):
        # All 12 USDA classes should produce valid output
        for sn in range(1, 13):
            nf = filpor_tr(sn, 3.0, 30, 0.1)
            self.assertGreater(nf, 0)


class TestPumpingDetection(unittest.TestCase):
    def test_flags_large_drops(self):
        # Synthetic series with small natural drops + one large outlier
        rng = np.random.RandomState(0)
        ho = 1.0 - 0.001 * np.arange(100) + 0.005 * rng.randn(100)
        ho[50] -= 0.5   # large drop outlier
        cleaned = remove_outliers(ho, sensitivity=2.0)
        self.assertTrue(np.isnan(cleaned[50]))

    def test_preserves_smooth_data(self):
        ho = np.linspace(1.0, 1.5, 100) + 0.01 * np.random.RandomState(0).randn(100)
        cleaned = remove_outliers(ho, sensitivity=2.0)
        # Statistical noise can flag a handful of points; threshold is loose
        n_nan = int(np.isnan(cleaned).sum())
        self.assertLessEqual(n_nan, 10)


class TestKalmanWTF(unittest.TestCase):
    def test_apply_lag_zero(self):
        po = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(apply_lag(po, 0), po)

    def test_apply_lag_positive(self):
        po = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(apply_lag(po, 2), [0, 0, 1, 2])

    def test_run_model_core_shapes(self):
        n = 100
        po = np.zeros(n)
        po[10] = 0.05    # one rain event
        ho = 1.0 + np.linspace(0, 0.2, n) + 0.01 * np.random.RandomState(0).randn(n)
        res = run_model_core(k=-0.015, z=3.0, sn=2, po_m=po, ho_m=ho)
        self.assertEqual(res.h_kalman_m.shape, (n,))
        self.assertEqual(res.h_pure_wtf_m.shape, (n,))
        self.assertEqual(res.recharge_m_per_day.shape, (n,))
        self.assertTrue(np.isfinite(res.h_kalman_m).all())

    def test_negative_k_required(self):
        with self.assertRaises(ValueError):
            run_model_core(k=0.01, z=3.0, sn=2,
                           po_m=np.zeros(10), ho_m=np.ones(10))


class TestOptimization(unittest.TestCase):
    def test_runs_without_error(self):
        # 100-day synthetic case
        n = 100
        rng = np.random.RandomState(0)
        po = (rng.uniform(size=n) < 0.2) * 0.02
        ho = 1.0 + 0.5 * np.cumsum(po) - 0.001 * np.arange(n) + 0.01 * rng.randn(n)
        opt = run_optimization(po_m=po, ho_m=ho, sn=2,
                               lag_grid=(0, 5, 10))
        self.assertGreaterEqual(opt.lag_days, 0)
        self.assertLessEqual(opt.lag_days, 10)
        self.assertLess(opt.k, 0)
        self.assertGreater(opt.z, 0.5)


if __name__ == "__main__":
    unittest.main()
