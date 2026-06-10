"""Apply the v2 framework (Sy prior) to the 5 Siheung field wells.

Produces the v2 counterpart of paper Table 2 and checks the resulting
Rch% against the external plausibility constraints (Korean alluvial
literature 8–25 %, Siheung catchment water balance ≤ ~18 %).

Usage:
    python notebooks/v2/field_v2.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from framework.io import load_well
from framework.pumping_detection import remove_outliers
from framework_v2.optim_v2 import sn_sweep_v2
from framework_v2.ccf_lag import ccf_peak_lag

FIELD_DIR = Path("/Users/choejeonghun/Dropbox/GW/Hybrid_final")
WELLS = ["SH08", "SH11", "SH22", "SH23", "SH28"]

SOIL_NAMES = {
    1: "Sand", 2: "Sandy Loam", 3: "Loamy Sand", 4: "Silt Loam",
    5: "Silt", 6: "Clay", 7: "Silty Clay", 8: "Sandy Clay",
    9: "Silty Clay Loam", 10: "Clay Loam", 11: "Sandy Clay Loam",
    12: "Loam",
}

# v1 paper values (Table 2) for comparison
V1_TABLE2 = {
    "SH08": dict(sn=1, rch_pct=27.47),
    "SH11": dict(sn=1, rch_pct=9.47),
    "SH22": dict(sn=3, rch_pct=19.30),
    "SH23": dict(sn=1, rch_pct=13.12),
    "SH28": dict(sn=1, rch_pct=6.86),
}


def main():
    print("=" * 86)
    print("Field application — v2 (Sy prior λ=10, free lag grid step 4)")
    print("=" * 86)
    print(f"{'Well':>6} {'sn':>3} {'Soil':<15} {'τcc':>4} {'lag':>4} "
          f"{'k':>9} {'z':>5} {'Sy_op':>7} {'Rch%':>6} {'RMSE':>7} "
          f"{'v1 Rch%':>8} {'Δ':>6}")
    print("-" * 86)

    rows = []
    for well in WELLS:
        t0 = time.time()
        w = load_well(FIELD_DIR / f"{well}.txt", name=well)
        gw_clean = remove_outliers(w.gw_m, sensitivity=2.0)
        tau_cc = ccf_peak_lag(w.rain_m, w.gw_m)

        best, sweep = sn_sweep_v2(
            po_m=w.rain_m, ho_m=gw_clean,
            use_ccf_lag=False, use_sy_prior=True,
            lag_step=4, nm_maxiter=120,
        )

        years = w.n_days / 365.25
        annual_rain_mm = float(w.rain_m.sum() * 1000.0 / years)
        rch_pct = 100.0 * best.annual_rch_mm / annual_rain_mm

        v1 = V1_TABLE2[well]
        delta = rch_pct - v1["rch_pct"]

        print(f"{well:>6} {best.sn:>3} {SOIL_NAMES[best.sn]:<15} "
              f"{tau_cc:>4d} {best.lag_days:>4d} {best.k:>9.5f} "
              f"{best.z:>5.2f} {best.sy_operational:>7.4f} {rch_pct:>6.2f} "
              f"{best.rmse_pure:>7.4f} {v1['rch_pct']:>8.2f} {delta:>+6.1f}"
              f"   ({time.time()-t0:.0f}s)")

        rows.append(dict(
            well=well, sn=best.sn, soil=SOIL_NAMES[best.sn],
            tau_cc=tau_cc, lag=best.lag_days, k=best.k, z=best.z,
            sy_op=best.sy_operational, rch_pct=rch_pct,
            rmse_pure=best.rmse_pure, rmse_kalman=best.rmse_kalman,
            cc=best.cc, v1_rch_pct=v1["rch_pct"], delta_pp=delta,
            annual_rain_mm=annual_rain_mm,
        ))

    df = pd.DataFrame(rows)
    out = ROOT / "data" / "field_v2_results.csv"
    df.to_csv(out, index=False)

    mean_rch = df["rch_pct"].mean()
    print("-" * 86)
    print(f"5-well mean Rch%: {mean_rch:.1f}  "
          f"(v1: {np.mean([v['rch_pct'] for v in V1_TABLE2.values()]):.1f})")
    print(f"Plausibility: Korean alluvial literature 8–25 %, "
          f"catchment P−ET−Q upper bound ≈ 18 % (± 5 pp)")
    print(f"\n  ✓ written: {out}")


if __name__ == "__main__":
    main()
