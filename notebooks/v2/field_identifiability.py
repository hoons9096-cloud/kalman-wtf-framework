"""Apply the free-Sy + water-balance identification to the Siheung wells.

Real wells have no ground truth, so we report the identified recharge
with its Sy-prior band, the recharge coefficient, and the Sy-prior /
water-balance consistency (the built-in cross-validation). The field
txt files are kept local (see .gitignore: data/field/).

    python notebooks/v2/field_identifiability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from framework.io import load_well
from framework.pumping_detection import remove_outliers
from framework_v2.free_sy_inversion import invert_free_sy
from framework_v2.water_balance import constrain_recharge

FIELD = ROOT / "data" / "field"
WELLS = ["SH08", "SH11", "SH22", "SH23", "SH28"]
V1_RCH_PCT = {"SH08": 27.5, "SH11": 9.5, "SH22": 19.3, "SH23": 13.1, "SH28": 6.9}

SMOOTH_WINDOW = 7


def run() -> pd.DataFrame:
    rows = []
    for name in WELLS:
        w = load_well(FIELD / f"{name}.txt", name=name)
        gw = remove_outliers(w.gw_m, sensitivity=2.0)
        yr = w.n_days / 365.25
        P = float(np.nansum(w.rain_m)) * 1000.0 / yr
        r = invert_free_sy(w.rain_m, gw, smooth_window=SMOOTH_WINDOW)
        c = constrain_recharge(r, P)
        rows.append(dict(
            well=name, n_days=w.n_days, P_mm=P, k=r.k,
            U_mm=c.U_annual_mm, sy_joint=c.sy_joint,
            rch_mm=c.rch_joint_mm, rch_lo=c.rch_joint_lo_mm,
            rch_hi=c.rch_joint_hi_mm, rch_pct=100 * c.rch_joint_mm / P,
            consistency_sigma=c.consistency_sigma,
            v1_rch_pct=V1_RCH_PCT[name],
        ))
    return pd.DataFrame(rows)


def main():
    df = run()
    print("=" * 84)
    print(f"Siheung 5 wells — free-Sy + water-balance identification "
          f"(window={SMOOTH_WINDOW})")
    print("=" * 84)
    print(f"{'well':>5} {'days':>4} {'P':>5} {'k':>7} {'U_mm':>6} "
          f"{'Sy_jnt':>7} {'Rch_mm':>7} {'band':>13} {'Rch%':>6} "
          f"{'cons_σ':>7} {'v1_%':>5}")
    for _, r in df.iterrows():
        print(f"{r.well:>5} {r.n_days:>4} {r.P_mm:>5.0f} {r.k:>7.4f} "
              f"{r.U_mm:>6.0f} {r.sy_joint:>7.3f} {r.rch_mm:>7.0f} "
              f"[{r.rch_lo:5.0f},{r.rch_hi:5.0f}] {r.rch_pct:>6.1f} "
              f"{r.consistency_sigma:>7.1f} {r.v1_rch_pct:>5.1f}")
    print("-" * 84)
    print(f"5-well mean Rch% = {df.rch_pct.mean():.1f} "
          f"(catchment plausible 8-25%);  "
          f"max consistency = {df.consistency_sigma.max():.1f} σ")
    out = ROOT / "data" / "field_identifiability.csv"
    df.to_csv(out, index=False)
    print(f"\n  ✓ {out}")


if __name__ == "__main__":
    main()
