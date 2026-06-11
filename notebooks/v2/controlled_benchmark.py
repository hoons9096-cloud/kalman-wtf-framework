"""Controlled recovery measurement (settles the 'is accuracy alive?' question).

Runs the v2 MAP/prior pipeline on all five synthetic scenarios under
two lag-handling configurations, holding everything else fixed, and
reports recharge recovery and Sy recovery side by side.

    python notebooks/v2/controlled_benchmark.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from framework.io import load_well
from framework.pumping_detection import remove_outliers
from framework_v2.optim_v2 import sn_sweep_v2

SYN = ROOT / "data" / "matlab_inputs"
SCEN = ["S1", "S2", "S3", "S4", "S5"]
CONFIGS = {
    "ccf_lag":  dict(use_ccf_lag=True,  use_sy_prior=True, lag_step=4, nm_maxiter=120),
    "grid_lag": dict(use_ccf_lag=False, use_sy_prior=True, lag_step=4, nm_maxiter=120),
}


def run_one(sid: str, cfg: dict) -> dict:
    w = load_well(SYN / f"SYN_{sid}.txt", name=sid)
    gw = remove_outliers(w.gw_m, sensitivity=2.0)
    truth = json.load(open(SYN / f"SYN_{sid}_truth.json"))
    best, _ = sn_sweep_v2(po_m=w.rain_m, ho_m=gw, **cfg)
    rch_true = truth["annual_recharge_true_mm"]
    sy_true = truth["sy_operational_true"]
    return dict(
        sn=best.sn, lag=best.lag_days, sy_op=best.sy_operational,
        rch=best.annual_rch_mm, rch_true=rch_true,
        recov_rch=best.annual_rch_mm / rch_true,
        recov_sy=best.sy_operational / sy_true,
        rmse=best.rmse_pure,
    )


def main():
    rows = {}
    for cname, cfg in CONFIGS.items():
        print(f"\n=== config: {cname} ===")
        print(f"{'scen':>4} {'sn':>3} {'lag':>4} {'Sy_op':>7} {'rch':>7} "
              f"{'true':>7} {'recov_rch':>10} {'recov_sy':>9} {'rmse':>7}")
        recs = []
        for sid in SCEN:
            r = run_one(sid, cfg)
            recs.append(r)
            print(f"{sid:>4} {r['sn']:>3} {r['lag']:>4} {r['sy_op']:>7.4f} "
                  f"{r['rch']:>7.1f} {r['rch_true']:>7.1f} "
                  f"{r['recov_rch']:>10.3f} {r['recov_sy']:>9.3f} {r['rmse']:>7.4f}",
                  flush=True)
        rr = np.array([r["recov_rch"] for r in recs])
        print(f"  mean recov_rch (S1-S5) = {rr.mean():.3f} ± {rr.std():.3f}  "
              f"[realistic S1,S2,S4,S5 = {np.mean([recs[i]['recov_rch'] for i in (0,1,3,4)]):.3f}]")
        rows[cname] = recs


if __name__ == "__main__":
    main()
