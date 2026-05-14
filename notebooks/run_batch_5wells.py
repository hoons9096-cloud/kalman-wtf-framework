"""Batch runner for the 5 Siheung wells — reproduces Tables 2 and 3.

For each of the 5 SH wells (SH08, SH11, SH22, SH23, SH28):
  1. Load 3-column txt (date / head / rainfall).
  2. Remove pumping outliers (σ-threshold).
  3. Sweep sn = 1..12 and run Nelder-Mead over (lag, k, z) for each.
  4. Report the RMSE-optimal sn + (lag, k, z, Sy, Rch%, RMSE, CC).

For SH-22 specifically, the full sn-sweep is also written separately
(Table 3 in the manuscript).

Outputs:
  - data/batch_results.csv   (Table 2: one row per well)
  - data/sh22_sn_sweep.csv   (Table 3: sn=1..12 for SH-22)

Usage:
  python notebooks/run_batch_5wells.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from framework.io import load_well
from framework.pumping_detection import remove_outliers
from framework.optim import run_optimization
from framework.kalman_wtf import run_model_core, apply_lag

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
WELLS = ["SH08", "SH11", "SH22", "SH23", "SH28"]
FIELD_DIR = Path("/Users/choejeonghun/Dropbox/GW/Hybrid_final")
OUT_DIR = ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOIL_NAMES = {
    1: "Sand", 2: "Sandy Loam", 3: "Loamy Sand", 4: "Silt Loam",
    5: "Silt", 6: "Clay", 7: "Silty Clay", 8: "Sandy Clay",
    9: "Silty Clay Loam", 10: "Clay Loam", 11: "Sandy Clay Loam",
    12: "Loam",
}


def ccf_peak(rain_m: np.ndarray, gw_m: np.ndarray, max_lag: int = 30) -> int:
    """CCF argmax between rainfall and head increment dh/dt (τ_cc)."""
    dh = np.diff(gw_m)
    rain_aligned = rain_m[1:]
    dh = np.where(np.isnan(dh), 0.0, dh)
    rain_aligned = np.where(np.isnan(rain_aligned), 0.0, rain_aligned)
    best_lag, best_cc = 0, -1.0
    for lag in range(0, max_lag + 1):
        if lag == 0:
            r, d = rain_aligned, dh
        else:
            r, d = rain_aligned[:-lag], dh[lag:]
        if len(r) < 30:
            continue
        cc = np.corrcoef(r, d)[0, 1]
        if np.isnan(cc):
            continue
        if cc > best_cc:
            best_cc, best_lag = float(cc), lag
    return best_lag


def analyse_well(name: str, path: Path) -> dict:
    """Sweep sn = 1..12 and return the best configuration + per-sn detail."""
    w = load_well(path, name=name)
    rain_m = w.rain_m
    rain_mm = rain_m * 1000.0
    gw_m = w.gw_m
    n_days = w.n_days
    years = n_days / 365.25
    annual_rain_mm = float(rain_mm.sum() / years)

    # τ_cc (physical phase lag, from raw signals)
    tau_cc = ccf_peak(rain_m, gw_m, max_lag=30)

    # Pumping outlier removal
    gw_clean = remove_outliers(gw_m, sensitivity=2.0)

    # Count flagged points as "pumping events"
    n_pump = int(np.sum(np.isnan(gw_clean) & ~np.isnan(gw_m)))

    # sn sweep
    detail = []
    best = None
    for sn in range(1, 13):
        out = run_optimization(po_m=rain_m, ho_m=gw_clean, sn=sn,
                               lag_grid=tuple(range(0, 15)))
        po_shift = apply_lag(rain_m, out.lag_days)
        core = run_model_core(k=out.k, z=out.z, sn=sn,
                              po_m=po_shift, ho_m=gw_clean)
        sy_avg = float(np.nanmean(core.sy_series))
        rch_m_per_day = core.recharge_m_per_day
        annual_rch_mm = float(np.nansum(rch_m_per_day) * 1000.0 / years)
        rch_pct = 100.0 * annual_rch_mm / annual_rain_mm
        # CC (Pearson) between observed and Kalman trajectory
        mask = ~np.isnan(gw_clean) & ~np.isnan(core.h_kalman_m)
        if mask.sum() >= 2:
            cc = float(np.corrcoef(core.h_kalman_m[mask],
                                   gw_clean[mask])[0, 1])
        else:
            cc = 0.0
        row = dict(
            well=name, sn=sn, soil=SOIL_NAMES[sn],
            lag_day=out.lag_days, tau_cc=tau_cc,
            k=out.k, z=out.z, sy_avg=sy_avg,
            rch_pct=rch_pct, rmse_pure=out.rmse_pure,
            rmse_kalman=out.rmse_kalman, cc=cc,
            n_days=n_days, n_pump=n_pump,
            annual_rain_mm=annual_rain_mm,
        )
        detail.append(row)
        if best is None or out.rmse_pure < best["rmse_pure"]:
            best = row
    return {"best": best, "detail": detail}


# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
print("=" * 78)
print("Batch runner — 5 Siheung wells (Table 2 reproduction)")
print("=" * 78)
print(f"{'Well':>6} {'sn':>3} {'Soil':<16} {'τ_cc':>5} {'lag':>4} "
      f"{'k':>9} {'Sy':>7} {'Rch%':>7} {'RMSE':>7} {'CC':>5}")
print("-" * 78)

all_best = []
sh22_detail = None

for well in WELLS:
    path = FIELD_DIR / f"{well}.txt"
    if not path.exists():
        print(f"  ⚠ skipped: {path} not found")
        continue
    res = analyse_well(well, path)
    b = res["best"]
    all_best.append(b)
    if well == "SH22":
        sh22_detail = res["detail"]
    print(f"{b['well']:>6} {b['sn']:>3} {b['soil']:<16} "
          f"{b['tau_cc']:>5d} {b['lag_day']:>4d} {b['k']:>9.5f} "
          f"{b['sy_avg']:>7.4f} {b['rch_pct']:>7.2f} "
          f"{b['rmse_pure']:>7.4f} {b['cc']:>5.3f}")

# Write CSVs
df_best = pd.DataFrame(all_best)
out_best = OUT_DIR / "batch_results.csv"
df_best.to_csv(out_best, index=False)
print(f"\n  ✓ Table 2 written: {out_best}")

if sh22_detail is not None:
    df_sweep = pd.DataFrame(sh22_detail)
    out_sweep = OUT_DIR / "sh22_sn_sweep.csv"
    df_sweep.to_csv(out_sweep, index=False)
    print(f"  ✓ Table 3 written: {out_sweep}")

    print("\n" + "=" * 78)
    print("SH-22 sn-sweep (Table 3 reproduction)")
    print("=" * 78)
    print(f"{'sn':>3} {'Soil':<18} {'lag':>4} {'k':>9} {'z':>5} "
          f"{'Sy':>7} {'Rch%':>7} {'RMSE':>7} {'CC':>5}")
    for r in sh22_detail:
        print(f"{r['sn']:>3} {r['soil']:<18} {r['lag_day']:>4d} "
              f"{r['k']:>9.5f} {r['z']:>5.2f} {r['sy_avg']:>7.4f} "
              f"{r['rch_pct']:>7.2f} {r['rmse_pure']:>7.4f} "
              f"{r['cc']:>5.3f}")
