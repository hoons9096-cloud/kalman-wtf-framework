"""Smoke tests for the synthetic-well generator.

Each test verifies a single physical or contractual property of the
synthetic-well output. Failures here usually indicate a regression in the
benchmark itself (not the framework under test).
"""
from __future__ import annotations

import unittest
import sys
import pathlib

# Make `src/` importable when running `python -m unittest discover`
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from synthetic import (  # noqa: E402
    SCENARIOS,
    generate_synthetic_well,
    generate_rainfall,
    apply_lag_filter,
)


class TestRainfall(unittest.TestCase):
    def test_nonnegative(self):
        rain = generate_rainfall(n_days=365 * 3, seed=0)
        self.assertTrue((rain >= 0).all())

    def test_annual_total_close_to_target(self):
        rain = generate_rainfall(n_days=365 * 3, annual_total_mm=950.0, seed=0)
        annual = rain.sum() / 3.0
        # Allow ~5% deviation
        self.assertAlmostEqual(annual, 950.0, delta=50.0)

    def test_seasonality(self):
        """Wet-season months (Jun-Sep) should carry > 50% of the annual total."""
        rain = generate_rainfall(n_days=365, start_day_of_year=1, seed=0)
        # Days 152-273 = Jun 1 to Sep 30 (rough)
        wet_total = rain[151:273].sum()
        self.assertGreater(wet_total / rain.sum(), 0.5)


class TestLagFilter(unittest.TestCase):
    def test_zero_lag_pass_through(self):
        rain = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
        out = apply_lag_filter(rain, tau_lag_days=0.0, recharge_fraction=1.0)
        np.testing.assert_array_almost_equal(out, rain)

    def test_first_moment_matches_tau(self):
        """An impulse rainfall delivers recharge with mean lag ≈ tau."""
        n = 200
        rain = np.zeros(n)
        rain[0] = 100.0
        for tau in (5.0, 14.0, 30.0):
            out = apply_lag_filter(rain, tau_lag_days=tau, recharge_fraction=1.0)
            # First moment of the response
            k = np.arange(n)
            mean_lag = (k * out).sum() / out.sum()
            self.assertAlmostEqual(mean_lag, tau, delta=1.0)

    def test_total_recharge_equals_fraction(self):
        rain = np.ones(500)
        out = apply_lag_filter(rain, tau_lag_days=10.0, recharge_fraction=0.12)
        # After many days the integrated output should be ~fraction * integrated input
        # (transient ramp-up adds small loss at start; check steady-state)
        steady_out = out[100:].sum()
        steady_in = rain[100:].sum()
        self.assertAlmostEqual(steady_out / steady_in, 0.12, delta=0.005)


class TestScenarioGeneration(unittest.TestCase):
    def test_all_scenarios_generate_without_error(self):
        for sid in SCENARIOS:
            well = generate_synthetic_well(scenario_id=sid, seed=42)
            self.assertEqual(len(well.h_observed_m), well.config.n_days)

    def test_S1_baseline_shape_and_finiteness(self):
        well = generate_synthetic_well(scenario_id="S1", seed=0)
        n = well.config.n_days
        # Shape consistency
        for arr in (
            well.rainfall_mm, well.recharge_mm, well.sy_true,
            well.h_true_m, well.h_with_pumping_m, well.h_observed_m,
        ):
            self.assertEqual(arr.shape, (n,))

        # Truth fields must be finite (only h_observed may contain NaN)
        for arr in (well.rainfall_mm, well.recharge_mm, well.sy_true,
                    well.h_true_m, well.h_with_pumping_m):
            self.assertTrue(np.isfinite(arr).all())

    def test_S1_pumping_events_recorded(self):
        well = generate_synthetic_well(scenario_id="S1", seed=0)
        # ~10 events/yr × 5 yr = 50 expected, allow Poisson spread
        self.assertGreaterEqual(len(well.pumping_truth.events), 30)
        self.assertLessEqual(len(well.pumping_truth.events), 80)

    def test_S2_long_lag(self):
        """S2 should produce a smoother recharge series than S1 (longer lag)."""
        s1 = generate_synthetic_well("S1", seed=0)
        s2 = generate_synthetic_well("S2", seed=0)
        # Day-to-day variance should be lower under longer lag
        self.assertLess(np.std(np.diff(s2.recharge_mm)),
                        np.std(np.diff(s1.recharge_mm)))

    def test_S5_fixed_sy(self):
        well = generate_synthetic_well("S5", seed=0)
        self.assertTrue(np.allclose(well.sy_true, 0.10))

    def test_h_with_pumping_below_h_true_at_event_peaks(self):
        well = generate_synthetic_well("S1", seed=0)
        diff = well.h_with_pumping_m - well.h_true_m
        # Pumping subtracts from head, so diff should be <= 0 everywhere
        self.assertTrue((diff <= 1e-9).all())

    def test_h_observed_has_some_gaps(self):
        well = generate_synthetic_well("S1", seed=0)
        # gap_prob default 0.02 over 1825 days → ~36 NaNs expected
        n_nan = int(np.isnan(well.h_observed_m).sum())
        self.assertGreater(n_nan, 5)
        self.assertLess(n_nan, 200)

    def test_base_level_constraint(self):
        well = generate_synthetic_well("S1", seed=0)
        self.assertTrue((well.h_true_m >= well.config.h_base_m - 1e-9).all())


if __name__ == "__main__":
    unittest.main()
