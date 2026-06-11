"""Unit tests for the water-balance dual-constraint identification."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from framework_v2.free_sy_inversion import FreeSyResult
from framework_v2.water_balance import constrain_recharge, _gaussian_combine


def _fake_result(U_head_m, n_years=5.0, sy_mean=0.07, sy_std=0.03):
    return FreeSyResult(
        k=0.005, h_base=0.0, U_head_m=U_head_m,
        sy_prior_mean=sy_mean, sy_prior_std=sy_std, n_years=n_years,
        annual_rch_mm=sy_mean * U_head_m * 1000 / n_years,
        annual_rch_lo_mm=0.0, annual_rch_hi_mm=0.0,
        sy_used=sy_mean, rmse_recession_m=0.01,
    )


def test_gaussian_combine_equal_inputs():
    m, s = _gaussian_combine(0.10, 0.02, 0.10, 0.02)
    assert abs(m - 0.10) < 1e-12
    assert s < 0.02                      # combined std is tighter


def test_gaussian_combine_precision_weighting():
    # Tighter estimate dominates
    m, _ = _gaussian_combine(0.05, 0.001, 0.20, 0.10)
    assert abs(m - 0.05) < 0.005


def test_joint_lies_between_constraints():
    # U chosen so the water-balance Sy sits above the literature prior
    res = _fake_result(U_head_m=5.0)               # U' = 1000 mm/yr
    c = constrain_recharge(res, annual_rain_mm=1000.0,
                           rch_coef_mean=0.12, rch_coef_std=0.05)
    lo, hi = sorted([c.sy_prior_mean, c.sy_wb_mean])
    assert lo <= c.sy_joint <= hi


def test_consistency_zero_when_constraints_agree():
    # Pick U so that water-balance Sy == literature prior exactly
    # Sy_wb = (c*P)/U' ; want = 0.07 with c=0.07, P=1000 -> U' = 1000
    res = _fake_result(U_head_m=5.0)               # U' = 1000 mm/yr
    c = constrain_recharge(res, annual_rain_mm=1000.0,
                           rch_coef_mean=0.07, rch_coef_std=0.03)
    assert c.consistency_sigma < 1e-6
    assert abs(c.sy_joint - 0.07) < 1e-9


def test_recharge_equals_sy_joint_times_U():
    res = _fake_result(U_head_m=4.0)
    c = constrain_recharge(res, annual_rain_mm=950.0)
    assert abs(c.rch_joint_mm - c.sy_joint * c.U_annual_mm) < 1e-6
