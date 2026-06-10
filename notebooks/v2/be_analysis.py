"""Barometric Efficiency analysis for the 5 SH wells (Phase B).

Pressure source: NOAA Global Hourly (ISD) — Incheon ASOS, WMO 47112.
Downloaded without authentication from:
    https://www.ncei.noaa.gov/data/global-hourly/access/<YEAR>/47112099999.csv
Station pressure parsed from the MA1 field (3-hourly synoptic, ~8 obs/day).

RESULT (2026-06-10, daily-scale attempt):
    BE estimates are NEGATIVE for 4 of 5 wells (physically impossible;
    BE ∈ [0, 1]) both for the full record and for rain-free windows.
    Diagnosis: at daily resolution the meteorological covariance
    (low pressure ↔ rain ↔ rising head) and the dry-season covariance
    (high pressure ↔ drought ↔ falling head) dominate the true
    barometric response, which is an hours-scale signal lost in daily
    averaging.

CONCLUSION:
    Daily data cannot support BE-based Sy estimation. To activate this
    module, the *hourly* (or finer) raw logger records for the Siheung
    wells are required. The municipal loggers most likely recorded at
    10-min or 1-h intervals before daily aggregation; the raw files
    should be requested from Siheung-si.

Usage (when hourly head data becomes available):
    python notebooks/v2/be_analysis.py <hourly_head_csv>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from framework.io import load_well
from framework_v2.barometric_efficiency import be_to_sy

PRESSURE_DAILY = ROOT / "data" / "kma_pressure" / "incheon_daily_mean.csv"
PRESSURE_HOURLY = ROOT / "data" / "kma_pressure" / "incheon_hourly_clean.csv"
FIELD_DIR = Path("/Users/choejeonghun/Dropbox/GW/Hybrid_final")
WELLS = ["SH08", "SH11", "SH22", "SH23", "SH28"]


def load_daily_pressure() -> pd.Series:
    pr = pd.read_csv(PRESSURE_DAILY, parse_dates=["datetime"])
    pr["p_m_h2o"] = pr["p_station_hpa"] * 0.0101972
    return pr.set_index(pr["datetime"].dt.normalize())["p_m_h2o"]


def daily_be(well: str, dry_only: bool = True,
             dry_window: int = 3, dry_thresh_mm: float = 1.0):
    """Daily-scale BE regression. KNOWN TO FAIL — kept for documentation."""
    pr = load_daily_pressure()
    w = load_well(FIELD_DIR / f"{well}.txt", name=well)
    dates = pd.DatetimeIndex(w.dates).normalize()
    head = pd.Series(w.gw_m, index=dates)
    rain = pd.Series(w.rain_m * 1000, index=dates)

    common = head.index.intersection(pr.index)
    h = head.loc[common].values
    p = pr.loc[common].values
    r = rain.loc[common].values

    n = len(r)
    mask = np.ones(n, dtype=bool)
    if dry_only:
        for i in range(n):
            lo, hi = max(0, i - dry_window), min(n, i + dry_window + 1)
            if np.nansum(r[lo:hi]) > dry_thresh_mm:
                mask[i] = False

    dh = np.diff(h)
    dp = np.diff(p)
    pair = mask[1:] & mask[:-1] & np.isfinite(dh) & np.isfinite(dp)
    dh, dp = dh[pair], dp[pair]
    if len(dh) < 30:
        return None

    A = np.column_stack([np.ones_like(dp), dp])
    sol, *_ = np.linalg.lstsq(A, dh, rcond=None)
    be = float(-sol[1])
    fitted = A @ sol
    ss_res = float(np.sum((dh - fitted) ** 2))
    ss_tot = float(np.sum((dh - dh.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"well": well, "n_pairs": int(len(dh)), "be": be,
            "r2": r2, "sy_be": be_to_sy(be) if 0 <= be <= 1 else np.nan}


if __name__ == "__main__":
    print("Daily-scale BE (KNOWN-FAIL diagnostic — see module docstring)")
    print(f"{'Well':>6} {'n':>5} {'BE':>7} {'R²':>6} {'Sy_BE':>7}")
    for well in WELLS:
        res = daily_be(well)
        if res is None:
            print(f"{well:>6}  insufficient data")
            continue
        print(f"{res['well']:>6} {res['n_pairs']:>5} {res['be']:>7.3f} "
              f"{res['r2']:>6.3f} {res['sy_be']:>7.4f}")
    print("\nNegative BE = meteorological confounding dominates at daily scale.")
    print("Hourly head data required — request raw logger files from Siheung-si.")
