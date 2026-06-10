"""Quick test of event-based WTF on synthetic + 5 SH wells."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from framework.io import load_well
from framework.pumping_detection import remove_outliers
from framework_v2.event_wtf import event_based_recharge


def test_synthetic():
    print("\n" + "=" * 70)
    print("Event-based WTF on synthetic scenarios")
    print("=" * 70)

    base = ROOT / "data" / "matlab_inputs"
    print(f"{'Sc':>4} {'Sy_used':>8} {'n_events':>9} {'rch_event':>10} "
          f"{'true_rch':>9} {'recovery':>9}")
    print("-" * 60)

    for sid in ["S1", "S2", "S3", "S4", "S5"]:
        truth = json.load(open(base / f"SYN_{sid}_truth.json"))
        true_rch = float(truth["annual_recharge_true_mm"])
        true_sy = float(truth["sy_operational_true"])

        w = load_well(base / f"SYN_{sid}.txt", name=sid)
        rain_mm = w.rain_m * 1000.0
        gw_clean = remove_outliers(w.gw_m, sensitivity=2.0)

        # Use truth Sy for now (best case)
        rch_with_truth, events = event_based_recharge(
            rain_mm, gw_clean, sy=true_sy)
        recovery = rch_with_truth / true_rch if true_rch > 0 else 0
        print(f"{sid:>4} {true_sy:>8.4f} {len(events):>9d} "
              f"{rch_with_truth:>10.2f} {true_rch:>9.2f} {recovery:>9.3f}")


def test_field():
    print("\n" + "=" * 70)
    print("Event-based WTF on Siheung wells (Sy from field WTF = 0.046)")
    print("=" * 70)

    sy_field = 0.046  # SH-22 framework Sy

    field_dir = Path("/Users/choejeonghun/Dropbox/GW/Hybrid_final")
    print(f"{'Well':>6} {'Sy':>6} {'n_events':>9} {'rch_event':>10} {'rch_mm/yr':>10}")
    print("-" * 50)
    for well in ["SH08", "SH11", "SH22", "SH23", "SH28"]:
        path = field_dir / f"{well}.txt"
        if not path.exists():
            continue
        w = load_well(path, name=well)
        rain_mm = w.rain_m * 1000.0
        gw_clean = remove_outliers(w.gw_m, sensitivity=2.0)
        annual_rch, events = event_based_recharge(
            rain_mm, gw_clean, sy=sy_field)
        years = w.n_days / 365.25
        annual_rain = float(rain_mm.sum() / years)
        rch_pct = 100 * annual_rch / annual_rain
        print(f"{well:>6} {sy_field:>6.3f} {len(events):>9d} "
              f"{annual_rch:>10.2f} {rch_pct:>10.2f}")


if __name__ == "__main__":
    test_synthetic()
    test_field()
