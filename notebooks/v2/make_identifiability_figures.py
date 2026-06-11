"""Regenerate the three identifiability figures (Fig 1-3) for the paper.

    python notebooks/v2/make_identifiability_figures.py

Writes notebooks/figures/fig1_identifiability_S1.png,
       notebooks/figures/fig2_dual_constraint_S1.png,
       notebooks/figures/fig3_sensitivity.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from framework.io import load_well
from framework.pumping_detection import remove_outliers
from framework_v2.free_sy_inversion import invert_free_sy, implied_sy
from framework_v2.water_balance import constrain_recharge

SYN = ROOT / "data" / "matlab_inputs"
FIG = ROOT / "notebooks" / "figures"
SC = ["S1", "S2", "S3", "S4", "S5"]


def _load(s):
    w = load_well(SYN / f"SYN_{s}.txt", name=s)
    gw = remove_outliers(w.gw_m, sensitivity=2.0)
    rt = json.load(open(SYN / f"SYN_{s}_truth.json"))["annual_recharge_true_mm"]
    return w, gw, rt


def fig1():
    w, gw, rt = _load("S1")
    r = invert_free_sy(w.rain_m, gw, smooth_window=7)
    yr = r.n_years
    sy = np.linspace(0.02, 0.20, 100)
    sy_star = implied_sy(r.U_head_m, rt, yr)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sy, sy * r.U_head_m * 1000 / yr, "b-", lw=2,
            label="Rch = Sy x U  (all fit head equally)")
    ax.axhline(rt, color="k", ls="--", lw=1.5, label=f"true recharge ({rt:.0f})")
    ax.axvspan(0.04, 0.10, color="orange", alpha=0.2, label="Sy prior 0.07±0.03")
    ax.plot([sy_star], [rt], "r*", ms=18, label=f"consistent Sy* = {sy_star:.3f}")
    ax.set_xlabel("assumed specific yield Sy (-)")
    ax.set_ylabel("annual recharge (mm/yr)")
    ax.set_title("WTF non-identifiability (S1): head fixes the line, not the point")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig1_identifiability_S1.png", dpi=130)


def fig2():
    w, gw, rt = _load("S1")
    yr = len(w.gw_m) / 365.25
    P = float(w.rain_m.sum()) * 1000 / yr
    r = invert_free_sy(w.rain_m, gw, smooth_window=7)
    c = constrain_recharge(r, P)
    Up = r.U_head_m * 1000 / yr
    sy = np.linspace(0.02, 0.20, 100)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(sy, sy * Up, "b-", lw=2, label=f"head: R = Sy·U (U={Up:.0f})")
    ax.axvspan(c.sy_prior_mean - c.sy_prior_std, c.sy_prior_mean + c.sy_prior_std,
               color="orange", alpha=0.20)
    ax.axvline(c.sy_prior_mean, color="orange", lw=1, label="A: Sy prior 0.07±0.03")
    band = c.rch_wb_mm * 0.05 / 0.12
    ax.axhspan(c.rch_wb_mm - band, c.rch_wb_mm + band, color="green", alpha=0.15)
    ax.axhline(c.rch_wb_mm, color="green", lw=1, label="B: water balance c·P")
    ax.errorbar([c.sy_joint], [c.rch_joint_mm], xerr=c.sy_joint_std,
                yerr=(c.rch_joint_hi_mm - c.rch_joint_lo_mm) / 2, fmt="o",
                color="purple", ms=9, capsize=4,
                label=f"JOINT: {c.rch_joint_mm:.0f} mm/yr")
    ax.axhline(rt, color="k", ls="--", lw=1.3, label=f"truth {rt:.0f}")
    ax.plot([implied_sy(r.U_head_m, rt, yr)], [rt], "k*", ms=15)
    ax.set_xlabel("specific yield Sy (-)"); ax.set_ylabel("annual recharge (mm/yr)")
    ax.set_title("Two orthogonal constraints identify the non-identifiable Sy (S1)")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3); ax.set_ylim(0, 200)
    fig.tight_layout(); fig.savefig(FIG / "fig2_dual_constraint_S1.png", dpi=130)


def fig3():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    sy_grid = np.linspace(0.03, 0.15, 25)
    for s in SC:
        w, gw, rt = _load(s)
        r = invert_free_sy(w.rain_m, gw, smooth_window=7)
        Up = r.U_head_m * 1000 / r.n_years
        ax1.plot(sy_grid, sy_grid * Up / rt, lw=1.5, label=s)
    ax1.axhline(1.0, color="k", ls="--", lw=1)
    ax1.axvspan(0.04, 0.10, color="orange", alpha=0.15)
    ax1.set_xlabel("assumed Sy prior mean"); ax1.set_ylabel("recharge recovery")
    ax1.set_title("A. recovery linear in Sy (the equifinality)")
    ax1.legend(fontsize=8, ncol=5, loc="upper left"); ax1.grid(alpha=0.3)
    wins = list(range(1, 20, 2))
    for s in SC:
        w, gw, rt = _load(s)
        us = [implied_sy(invert_free_sy(w.rain_m, gw, smooth_window=ww).U_head_m,
                         rt, len(gw) / 365.25) for ww in wins]
        ax2.plot(wins, us, marker="o", ms=3, lw=1.3, label=s)
    ax2.axhspan(0.04, 0.10, color="orange", alpha=0.15, label="Sy prior band")
    ax2.set_xlabel("smoothing window (days)")
    ax2.set_ylabel("truth-consistent Sy*")
    ax2.set_title("B. U (thus Sy*) depends on noise processing (~±30%)")
    ax2.legend(fontsize=8, ncol=2); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "fig3_sensitivity.png", dpi=130)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig1(); fig2(); fig3()
    print(f"  ✓ figures written to {FIG}")
