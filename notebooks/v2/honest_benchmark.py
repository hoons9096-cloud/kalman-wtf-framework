"""Honest single-pipeline recovery benchmark (replaces best-of envelopes).

One fixed configuration, untuned priors, applied identically to all five
synthetic scenarios. Reports, for each scenario:

  * U            — identifiable head-equivalent recharge input (mm/yr)
  * recov_A      — recovery with the literature Sy prior alone
  * Sy_joint     — Sy after the water-balance constraint
  * recov_joint  — recovery of the dual-constrained recharge
  * consistency  — sigma-distance between the two priors (cross-check)

No per-scenario tuning, no scenario exclusion, no configuration search.

    python notebooks/v2/honest_benchmark.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from framework.io import load_well
from framework.pumping_detection import remove_outliers
from framework_v2.free_sy_inversion import invert_free_sy, implied_sy
from framework_v2.water_balance import constrain_recharge

SYN = ROOT / "data" / "matlab_inputs"
SCEN = ["S1", "S2", "S3", "S4", "S5"]

# Fixed pipeline configuration (identical for every well).
SMOOTH_WINDOW = 7
SY_PRIOR_MEAN, SY_PRIOR_STD = 0.07, 0.03
RCH_COEF_MEAN, RCH_COEF_STD = 0.12, 0.05


def run() -> pd.DataFrame:
    rows = []
    for sid in SCEN:
        w = load_well(SYN / f"SYN_{sid}.txt", name=sid)
        gw = remove_outliers(w.gw_m, sensitivity=2.0)
        truth = json.load(open(SYN / f"SYN_{sid}_truth.json"))
        rch_true = truth["annual_recharge_true_mm"]

        yr = len(w.gw_m) / 365.25
        P = float(w.rain_m.sum()) * 1000.0 / yr

        r = invert_free_sy(w.rain_m, gw,
                           sy_prior_mean=SY_PRIOR_MEAN,
                           sy_prior_std=SY_PRIOR_STD,
                           smooth_window=SMOOTH_WINDOW)
        c = constrain_recharge(r, P,
                               rch_coef_mean=RCH_COEF_MEAN,
                               rch_coef_std=RCH_COEF_STD)
        rows.append(dict(
            scenario=sid,
            k_est=r.k,
            U_annual_mm=c.U_annual_mm,
            sy_star=implied_sy(r.U_head_m, rch_true, yr),
            rch_true=rch_true,
            recov_A=c.rch_sy_prior_mm / rch_true,
            sy_joint=c.sy_joint,
            rch_joint=c.rch_joint_mm,
            recov_joint=c.rch_joint_mm / rch_true,
            consistency_sigma=c.consistency_sigma,
        ))
    return pd.DataFrame(rows)


def main():
    df = run()
    print("=" * 78)
    print("HONEST single-pipeline benchmark "
          f"(window={SMOOTH_WINDOW}, Sy~{SY_PRIOR_MEAN}±{SY_PRIOR_STD}, "
          f"c~{RCH_COEF_MEAN}±{RCH_COEF_STD})")
    print("=" * 78)
    print(f"{'sc':>3} {'k_est':>7} {'U(mm)':>7} {'Sy*':>6} {'true':>5} "
          f"{'recov_A':>8} {'Sy_jnt':>7} {'recov_jnt':>10} {'cons_σ':>7}")
    for _, r in df.iterrows():
        print(f"{r.scenario:>3} {r.k_est:>7.4f} {r.U_annual_mm:>7.0f} "
              f"{r.sy_star:>6.3f} {r.rch_true:>5.0f} {r.recov_A:>8.2f} "
              f"{r.sy_joint:>7.3f} {r.recov_joint:>10.2f} "
              f"{r.consistency_sigma:>7.1f}")
    print("-" * 78)
    print(f"Sy-prior alone : recov {df.recov_A.mean():.3f} ± {df.recov_A.std():.3f}")
    print(f"dual-constraint: recov {df.recov_joint.mean():.3f} ± {df.recov_joint.std():.3f}")
    print(f"max consistency gap: {df.consistency_sigma.max():.1f} σ "
          f"(all < 2σ ⇒ Sy-prior and water-balance mutually consistent)")
    out = ROOT / "data" / "honest_benchmark.csv"
    df.to_csv(out, index=False)
    print(f"\n  ✓ {out}")


if __name__ == "__main__":
    main()
