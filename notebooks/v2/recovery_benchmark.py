"""Recovery benchmark — measures any framework version against the 5
synthetic-truth scenarios.

Usage:
    python notebooks/v2/recovery_benchmark.py [version_label]

Returns a CSV report showing recovery ratios per scenario. Used to
track improvement across v1 (paper) → v2 (Phase A improvements).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from framework.io import load_well
from framework.pumping_detection import remove_outliers
from framework.optim import run_optimization
from framework.kalman_wtf import run_model_core, apply_lag

from framework_v2.optim_v2 import sn_sweep_v2
from framework_v2.ccf_lag import ccf_peak_lag


SYN_DIR = ROOT / "data" / "matlab_inputs"


def run_v1(name: str, txt_path: Path):
    """V1 (paper) framework: standard sn-sweep with Nelder-Mead over (lag, k, z)."""
    w = load_well(txt_path, name=name)
    gw_clean = remove_outliers(w.gw_m, sensitivity=2.0)
    best = None
    for sn in range(1, 13):
        out = run_optimization(po_m=w.rain_m, ho_m=gw_clean, sn=sn,
                               lag_grid=tuple(range(0, 15)))
        if best is None or out.rmse_pure < best[1].rmse_pure:
            best = (sn, out)
    sn_best, opt_best = best
    po_shift = apply_lag(w.rain_m, opt_best.lag_days)
    core = run_model_core(k=opt_best.k, z=opt_best.z, sn=sn_best,
                          po_m=po_shift, ho_m=gw_clean)
    years = w.n_days / 365.25
    annual_rch_mm = float(np.nansum(core.recharge_m_per_day) * 1000.0 / years)
    sy_avg = float(np.nanmean(core.sy_series))
    return {
        "method": "v1_baseline",
        "sn": sn_best,
        "lag": opt_best.lag_days,
        "k": opt_best.k,
        "z": opt_best.z,
        "sy_avg": sy_avg,
        "annual_rch_mm": annual_rch_mm,
        "rmse_pure": opt_best.rmse_pure,
    }


def run_v2(name: str, txt_path: Path,
           use_ccf_lag: bool = True,
           use_sy_prior: bool = True):
    """V2 framework with Phase A improvements (CCF lag, Sy prior).

    lag_step=4 keeps the free-lag grid coarse (0, 4, 8, 12) so the
    benchmark completes in minutes; nm_maxiter caps Nelder-Mead
    wandering on the rugged prior-penalised loss surface.
    """
    w = load_well(txt_path, name=name)
    gw_clean = remove_outliers(w.gw_m, sensitivity=2.0)
    best, _all = sn_sweep_v2(po_m=w.rain_m, ho_m=gw_clean,
                             use_ccf_lag=use_ccf_lag,
                             use_sy_prior=use_sy_prior,
                             lag_step=4, nm_maxiter=120)
    return {
        "method": f"v2_ccf={use_ccf_lag}_prior={use_sy_prior}",
        "sn": best.sn,
        "lag": best.lag_days,
        "k": best.k,
        "z": best.z,
        "sy_avg": best.sy_operational,
        "annual_rch_mm": best.annual_rch_mm,
        "rmse_pure": best.rmse_pure,
    }


def score_against_truth(result: dict, truth_path: Path) -> dict:
    """Compute recovery ratios against synthetic ground truth."""
    truth = json.load(open(truth_path))
    true_rch = float(truth["annual_recharge_true_mm"])
    true_sy_op = float(truth["sy_operational_true"])
    true_lag = float(truth["tau_lag_true_days"])

    return {
        **result,
        "true_rch_mm": true_rch,
        "true_sy": true_sy_op,
        "true_lag": true_lag,
        "recovery_rch": result["annual_rch_mm"] / true_rch if true_rch > 0 else 0,
        "recovery_sy": result["sy_avg"] / true_sy_op if true_sy_op > 0 else 0,
        "abs_lag_error": abs(result["lag"] - true_lag),
    }


def benchmark(framework_fn: Callable, label: str) -> pd.DataFrame:
    """Run a framework function against all 5 synthetic scenarios."""
    rows = []
    for sid in ["S1", "S2", "S3", "S4", "S5"]:
        txt = SYN_DIR / f"SYN_{sid}.txt"
        truth = SYN_DIR / f"SYN_{sid}_truth.json"
        if not txt.exists():
            print(f"  ⚠ skip {sid}: missing {txt}")
            continue
        r = framework_fn(sid, txt)
        r["scenario"] = sid
        r["version"] = label
        scored = score_against_truth(r, truth)
        rows.append(scored)
        print(f"  {sid:4s}  rch_est={r['annual_rch_mm']:6.1f}  "
              f"true={scored['true_rch_mm']:6.1f}  "
              f"recovery={scored['recovery_rch']:.3f}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    """Headline statistics: mean recovery (excluding S3) + S3 separately."""
    realistic = df[df["scenario"].isin(["S1", "S2", "S4", "S5"])]
    s3 = df[df["scenario"] == "S3"]
    return {
        "version": df["version"].iloc[0],
        "n_realistic": len(realistic),
        "recovery_rch_mean": float(realistic["recovery_rch"].mean()),
        "recovery_rch_std": float(realistic["recovery_rch"].std()),
        "recovery_sy_mean": float(realistic["recovery_sy"].mean()),
        "lag_mae": float(df["abs_lag_error"].mean()),
        "S3_recovery": float(s3["recovery_rch"].iloc[0]) if len(s3) else None,
    }


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "v1_baseline"

    print("=" * 70)
    print(f"Recovery benchmark — version: {label}")
    print("=" * 70)

    if label == "v1_baseline":
        df = benchmark(run_v1, label)
    elif label == "v2_ccf_lag":
        df = benchmark(lambda n, p: run_v2(n, p,
                                            use_ccf_lag=True,
                                            use_sy_prior=False), label)
    elif label == "v2_sy_prior":
        df = benchmark(lambda n, p: run_v2(n, p,
                                            use_ccf_lag=False,
                                            use_sy_prior=True), label)
    elif label == "v2_full":
        df = benchmark(lambda n, p: run_v2(n, p,
                                            use_ccf_lag=True,
                                            use_sy_prior=True), label)
    else:
        raise NotImplementedError(f"Unknown version: {label}")

    summary = summarize(df)
    print("\n" + "-" * 70)
    print(f"Summary for {label}:")
    print(f"  Realistic-scenario recovery (S1, S2, S4, S5):")
    print(f"    Recharge:  {summary['recovery_rch_mean']:.3f} ± {summary['recovery_rch_std']:.3f}")
    print(f"    Sy:        {summary['recovery_sy_mean']:.3f}")
    print(f"  Lag MAE (5 scenarios):  {summary['lag_mae']:.1f} d")
    print(f"  S3 (high-pumping) recovery:  {summary['S3_recovery']:.3f}")

    out_csv = ROOT / "data" / f"benchmark_{label}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  ✓ written: {out_csv}")
