"""SH-22 ablation study — Table 6.

Computes the component contribution of the framework by selectively
disabling lag identification, outlier filtering, and dynamic Sy
(approximated by fixing sn) while keeping the rest of the pipeline.
Reports head-fit RMSE, Sy_avg, and Rch% for each configuration.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from framework.io import load_well
from framework.pumping_detection import remove_outliers
from framework.optim import run_optimization
from framework.kalman_wtf import run_model_core, apply_lag

SH22 = Path("/Users/choejeonghun/Dropbox/GW/Hybrid_final/SH22.txt")

print("Loading SH-22...")
w = load_well(SH22, name="SH22")
po_mm = w.rain_m * 1000.0
ho_m = w.gw_m
n_days = w.n_days
total_rain_m = po_mm.sum() / 1000.0
days_per_year = 365.25
years = n_days / days_per_year
annual_rain_mm = total_rain_m * 1000.0 / years

def report(name, lag, sn, k, z, h_pure, h_kalman, ho_used):
    """Compute summary stats for a configuration."""
    mask = ~np.isnan(ho_used) & ~np.isnan(h_kalman)
    rmse_k = float(np.sqrt(np.nanmean((h_kalman[mask] - ho_used[mask]) ** 2)))
    rmse_p = float(np.sqrt(np.nanmean((h_pure[mask] - ho_used[mask]) ** 2)))
    # Recharge from positive head rises × Sy(t)  (matches batch_runner)
    # Simplification here: use the Sy_avg × Σ positive Δh form
    dh = np.diff(h_pure)
    pos_dh = np.maximum(dh, 0.0).sum()
    rch_mm = pos_dh * sn_to_sy.get(sn, 0.05) * 1000.0  # use approximate Sy_lookup
    rch_pct = 100 * rch_mm / years / annual_rain_mm
    print(f"  {name:35s}  lag={lag:>2d}  sn={sn:>2d}  k={k:>+.5f}  "
          f"z={z:.2f}  RMSE_pure={rmse_p:.4f}  RMSE_kal={rmse_k:.4f}  "
          f"Rch%={rch_pct:6.2f}")
    return dict(name=name, lag=lag, sn=sn, k=k, z=z,
                rmse_pure=rmse_p, rmse_kalman=rmse_k, rch_pct=rch_pct)

# Approximate Sy lookup by sn (from MATLAB filpor central values)
sn_to_sy = {1: 0.073, 2: 0.022, 3: 0.046, 4: 0.022, 5: 0.022,
            6: 0.022, 7: 0.022, 8: 0.022, 9: 0.022, 10: 0.022,
            11: 0.022, 12: 0.022}

results = []

# === Config A: Pure WTF — no lag, no outlier removal, no dynamic Sy
#     Use sn=1 (Sand, conventional default), single Nelder-Mead with lag=0
print("\n=== Ablation on SH-22 ===")
po_m = po_mm / 1000.0
po_shift = apply_lag(po_m, 0)
out_A = run_optimization(po_m=po_shift, ho_m=ho_m, sn=1,
                         lag_grid=(0,))
core_A = run_model_core(k=out_A.k, z=out_A.z, sn=1,
                       po_m=po_shift, ho_m=ho_m)
results.append(report("A: Pure WTF (lag=0, sn=1, raw)",
                     0, 1, out_A.k, out_A.z,
                     core_A.h_pure_wtf_m, core_A.h_kalman_m, ho_m))

# === Config B: + outlier removal (still no lag, sn=1)
ho_clean = remove_outliers(ho_m, sensitivity=2.0)
po_shift = apply_lag(po_m, 0)
out_B = run_optimization(po_m=po_shift, ho_m=ho_clean, sn=1,
                         lag_grid=(0,))
core_B = run_model_core(k=out_B.k, z=out_B.z, sn=1,
                       po_m=po_shift, ho_m=ho_clean)
results.append(report("B: + outlier removal",
                     0, 1, out_B.k, out_B.z,
                     core_B.h_pure_wtf_m, core_B.h_kalman_m, ho_clean))

# === Config C: + lag identification (sn=1)
out_C = run_optimization(po_m=po_m, ho_m=ho_clean, sn=1,
                         lag_grid=tuple(range(0, 15)))
po_shift = apply_lag(po_m, out_C.lag_days)
core_C = run_model_core(k=out_C.k, z=out_C.z, sn=1,
                       po_m=po_shift, ho_m=ho_clean)
results.append(report("C: + lag identification (sn=1)",
                     out_C.lag_days, 1, out_C.k, out_C.z,
                     core_C.h_pure_wtf_m, core_C.h_kalman_m, ho_clean))

# === Config D: Full framework (sn-sweep automatically selects sn=3)
best_rmse = 1e9
best_sn = 1
best_out = out_C
for sn in range(1, 13):
    out = run_optimization(po_m=po_m, ho_m=ho_clean, sn=sn,
                           lag_grid=tuple(range(0, 15)))
    if out.rmse_pure < best_rmse:
        best_rmse = out.rmse_pure
        best_sn = sn
        best_out = out
po_shift = apply_lag(po_m, best_out.lag_days)
core_D = run_model_core(k=best_out.k, z=best_out.z, sn=best_sn,
                       po_m=po_shift, ho_m=ho_clean)
results.append(report(f"D: Full framework (sn-sweep)",
                     best_out.lag_days, best_sn, best_out.k, best_out.z,
                     core_D.h_pure_wtf_m, core_D.h_kalman_m, ho_clean))

# Save to CSV
df_out = pd.DataFrame(results)
out_csv = ROOT / "data" / "ablation_sh22.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(out_csv, index=False)
print(f"\n  ✓ written: {out_csv}")
