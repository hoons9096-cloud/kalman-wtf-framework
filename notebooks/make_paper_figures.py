"""Publication-quality figures for the Kalman-WTF manuscript.

Produces five PNG figures in `notebooks/figures/`:

  fig1_concept.png         — framework flowchart
  fig2_5well_overview.png  — 5 SH wells: rainfall + GW + Kalman reconstruction
  fig3_SH22_flagship.png   — paper flagship: pumping isolation before/after
  fig4_synthetic_recovery — synthetic truth recovery (lag/Sy/recharge)
  fig5_multi_well_bars    — per-well recharge comparison (raw vs filtered)

Usage:
    cd ~/git/kalman-wtf-framework
    python notebooks/make_paper_figures.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from framework import (load_well, remove_outliers, apply_lag,
                       run_model_core, run_optimization)
from framework.fillable_porosity import SOIL_DB


# === Publication style ===
plt.rcParams.update({
    "font.family": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.4,
    "savefig.dpi": 300,
    "figure.dpi": 110,
    "axes.unicode_minus": False,
})

# Consistent palette
COLORS = {
    "obs_raw":   "#7f8c8d",   # grey
    "obs_clean": "#e74c3c",   # red dots
    "kalman":    "#2980b9",   # blue
    "pure":      "#27ae60",   # green
    "rain":      "#3498db",   # rain bars
    "truth":     "#16a085",   # synthetic truth
    "framework": "#c0392b",   # framework output
    "highlight": "#f39c12",   # pumping events
}

FIELD = Path("/Users/choejeonghun/Dropbox/GW/Hybrid_final")
OUT = ROOT / "notebooks" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Per-well optimal sn (from earlier sweep)
WELL_SN = {"SH08": 1, "SH11": 1, "SH22": 3, "SH23": 1, "SH28": 1}


def run_well(name: str, sn: int):
    """Run optimization + final model on one field well."""
    well = load_well(FIELD / f"{name}.txt", name)
    ho_clean = remove_outliers(well.gw_m, sensitivity=2.0)
    opt = run_optimization(po_m=well.rain_m, ho_m=ho_clean, sn=sn,
                           lag_grid=tuple(range(0, 15, 1)))
    po_shift = apply_lag(well.rain_m, opt.lag_days)
    res = run_model_core(k=opt.k, z=opt.z, sn=sn,
                         po_m=po_shift, ho_m=ho_clean)
    rech_pos = (res.recharge_m_per_day + np.abs(res.recharge_m_per_day)) / 2
    rain_mm = well.rain_m.sum() * 1000.0
    rch_pct = rech_pos.sum() * 1000.0 / rain_mm * 100 if rain_mm > 0 else np.nan

    # conventional (no filter, no lag)
    res_c = run_model_core(k=opt.k, z=opt.z, sn=sn,
                           po_m=well.rain_m, ho_m=well.gw_m)
    rech_c = (res_c.recharge_m_per_day + np.abs(res_c.recharge_m_per_day)) / 2
    rch_c_pct = rech_c.sum() * 1000.0 / rain_mm * 100 if rain_mm > 0 else np.nan

    return {
        "well": well, "ho_clean": ho_clean, "opt": opt, "res": res,
        "rch_pct": rch_pct, "rch_c_pct": rch_c_pct,
        "sn": sn, "rain_mm": rain_mm,
    }


# ============================================================
# FIGURE 1 — Conceptual framework flowchart
# ============================================================
def fig1_concept():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    def box(x, y, w, h, label, color="#3498db", text_color="white", fontsize=10, fw="bold"):
        bbox = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                              linewidth=1.2, edgecolor=color, facecolor=color, alpha=0.85)
        ax.add_patch(bbox)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight=fw)

    def arrow(x1, y1, x2, y2, label=None, color="#2c3e50"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        if label:
            ax.text((x1+x2)/2 + 0.1, (y1+y2)/2 + 0.15, label, fontsize=8,
                    color=color, style="italic")

    # Inputs (left)
    box(0.1, 5.0, 2.2, 0.8, "Rainfall P(t)", color="#5dade2")
    box(0.1, 3.7, 2.2, 0.8, "GW level h(t)", color="#5dade2")

    # Preprocessing (mid-left)
    box(2.9, 4.35, 2.4, 0.9, "(1) Outlier filter\n(pumping detection)",
        color="#e67e22", fontsize=9)

    # Lag identification (mid)
    box(5.9, 5.0, 2.4, 0.8, "(2) Cross-correlation\nlag identification",
        color="#16a085", fontsize=9)
    box(5.9, 3.7, 2.4, 0.8, "(3) Dynamic Sy(t)\n(van Genuchten)",
        color="#16a085", fontsize=9)

    # Core (right-mid)
    box(8.9, 4.35, 2.6, 0.9, "(4) State-space\nKalman filter",
        color="#c0392b", fontsize=10)

    # Output (right)
    box(9.4, 2.5, 2.2, 0.8, "Recharge R(t)", color="#27ae60")
    box(9.4, 1.5, 2.2, 0.8, "h_kalman(t)", color="#27ae60")
    box(9.4, 0.5, 2.2, 0.8, "Sy(t)", color="#27ae60")

    # Optimization box (bottom)
    box(4.5, 0.7, 4.0, 0.8, "(5) Nelder-Mead optimisation over (k, z, lag)\n[ Objective: pure-WTF RMSE ]",
        color="#7f8c8d", fontsize=9, fw="normal")

    # Arrows
    arrow(2.3, 5.4, 2.9, 5.0, "P(t)")
    arrow(2.3, 4.1, 2.9, 4.65, "h(t)")
    arrow(5.3, 4.8, 5.9, 5.4)
    arrow(5.3, 4.45, 5.9, 4.1)
    arrow(8.3, 5.4, 8.9, 5.0)
    arrow(8.3, 4.1, 8.9, 4.65)
    arrow(11.5, 4.5, 11.5, 3.3, color="#27ae60")
    arrow(11.5, 3.5, 11.5, 2.5, color="#27ae60")
    arrow(11.5, 2.3, 11.5, 1.5, color="#27ae60")
    # Optimization feedback arrows
    ax.annotate("", xy=(6.5, 1.5), xytext=(6.5, 3.7),
                arrowprops=dict(arrowstyle="-|>", color="#7f8c8d", lw=1.0, linestyle="--"))
    ax.text(6.6, 2.4, "optimise\n(k, z, lag)", fontsize=8, color="#7f8c8d", style="italic")

    # Title
    ax.text(6.0, 6.2, "Phase-aligned Kalman-WTF framework", ha="center",
            fontsize=13, fontweight="bold")

    # Legend
    legend_y = 0.05
    items = [
        ("Inputs", "#5dade2"),
        ("Pre-processing", "#e67e22"),
        ("Physics modules", "#16a085"),
        ("Inference core", "#c0392b"),
        ("Outputs", "#27ae60"),
    ]
    x0 = 0.5
    for lbl, c in items:
        ax.add_patch(mpatches.Rectangle((x0, legend_y), 0.25, 0.18, color=c, alpha=0.85))
        ax.text(x0 + 0.32, legend_y + 0.09, lbl, fontsize=8, va="center")
        x0 += 1.9

    plt.tight_layout()
    plt.savefig(OUT / "fig1_concept.png", bbox_inches="tight")
    plt.close()
    print("  ✓ fig1_concept.png")


# ============================================================
# FIGURE 2 — 5-well overview (raw + Kalman + rainfall)
# ============================================================
def fig2_5well_overview():
    wells_data = {name: run_well(name, WELL_SN[name])
                  for name in ["SH08", "SH11", "SH22", "SH23", "SH28"]}

    fig, axes = plt.subplots(5, 1, figsize=(11, 11.5), sharex=False)
    for ax, (name, d) in zip(axes, wells_data.items()):
        well = d["well"]
        ho_clean = d["ho_clean"]
        res = d["res"]
        n = well.n_days
        days = np.arange(n)

        ax.plot(days, well.gw_m, "-", color=COLORS["obs_raw"], lw=0.8,
                label="Observed (raw)", alpha=0.8)
        ax.plot(days, ho_clean, ".", color=COLORS["obs_clean"], ms=2.0,
                label="Observed (filtered)")
        ax.plot(days, res.h_kalman_m, "-", color=COLORS["kalman"], lw=1.6,
                label="Kalman reconstruction")
        ax.set_ylabel("Head (m)")
        ax.set_xlim(0, n)
        n_pump = int((np.diff(well.gw_m) < -0.3).sum())
        title = (f"{name} (n={n} d, {n_pump} pumping events, "
                 f"sn={d['sn']} [{SOIL_DB[d['sn']]['name']}], "
                 f"lag={d['opt'].lag_days}d, "
                 f"R={d['rch_pct']:.1f}% vs raw {d['rch_c_pct']:.1f}%)")
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)

        # rainfall on twin axis
        ax2 = ax.twinx()
        ax2.bar(days, well.rain_m * 1000, width=1.0, color=COLORS["rain"],
                alpha=0.25, edgecolor="none")
        ax2.set_ylabel("Rain (mm/d)", color=COLORS["rain"])
        ax2.tick_params(axis="y", colors=COLORS["rain"])
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(COLORS["rain"])
        # match y-axis scale across wells
        ax2.set_ylim(0, 100)

    axes[-1].set_xlabel("Day")
    fig.suptitle("Multi-well application of the Kalman-WTF framework (5 SH wells)",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(OUT / "fig2_5well_overview.png", bbox_inches="tight")
    plt.close()
    print("  ✓ fig2_5well_overview.png")
    return wells_data


# ============================================================
# FIGURE 3 — SH22 flagship pumping isolation (multi-panel)
# ============================================================
def fig3_SH22_flagship():
    sn = WELL_SN["SH22"]
    well = load_well(FIELD / "SH22.txt", "SH22")
    ho_clean = remove_outliers(well.gw_m, sensitivity=2.0)
    opt_clean = run_optimization(po_m=well.rain_m, ho_m=ho_clean, sn=sn,
                                 lag_grid=tuple(range(0, 15, 1)))
    opt_raw = run_optimization(po_m=well.rain_m, ho_m=well.gw_m, sn=sn,
                               lag_grid=tuple(range(0, 15, 1)))
    res_clean = run_model_core(k=opt_clean.k, z=opt_clean.z, sn=sn,
                               po_m=apply_lag(well.rain_m, opt_clean.lag_days),
                               ho_m=ho_clean)
    res_raw = run_model_core(k=opt_raw.k, z=opt_raw.z, sn=sn,
                             po_m=apply_lag(well.rain_m, opt_raw.lag_days),
                             ho_m=well.gw_m)

    rech_pos_c = (res_clean.recharge_m_per_day +
                  np.abs(res_clean.recharge_m_per_day)) / 2
    rech_pos_r = (res_raw.recharge_m_per_day +
                  np.abs(res_raw.recharge_m_per_day)) / 2
    rain_mm = well.rain_m.sum() * 1000.0
    rch_c_pct = rech_pos_c.sum() * 1000.0 / rain_mm * 100
    rch_r_pct = rech_pos_r.sum() * 1000.0 / rain_mm * 100

    pumping_mask = np.isnan(ho_clean) & ~np.isnan(well.gw_m)
    pumping_days = np.where(pumping_mask)[0]

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1.3, 1.0], hspace=0.42, wspace=0.25)

    n = well.n_days
    days = np.arange(n)

    # (a) Raw observation + pumping events highlighted
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.plot(days, well.gw_m, "-", color=COLORS["obs_raw"], lw=0.9,
              label="Raw observation")
    for d in pumping_days:
        ax_a.axvspan(d - 0.5, d + 0.5, color=COLORS["highlight"], alpha=0.25)
    if len(pumping_days) > 0:
        ax_a.plot([], [], color=COLORS["highlight"], lw=8, alpha=0.4,
                  label=f"Detected pumping ({len(pumping_days)} days)")
    ax_a.set_ylabel("Head (m)")
    ax_a.set_title(f"(a) SH22 raw groundwater level with {len(pumping_days)} pumping-suspect events detected by σ-threshold filter",
                   loc="left", fontsize=10)
    ax_a.legend(loc="lower left", fontsize=9, frameon=False)
    ax_a.set_xlim(0, n)

    # (b) Filtered + Kalman reconstruction
    ax_b = fig.add_subplot(gs[1, :])
    ax_b.plot(days, well.gw_m, "-", color=COLORS["obs_raw"], lw=0.6, alpha=0.5,
              label="Raw")
    ax_b.plot(days, ho_clean, ".", color=COLORS["obs_clean"], ms=2.5,
              label="Filtered observation")
    ax_b.plot(days, res_clean.h_kalman_m, "-", color=COLORS["kalman"], lw=1.8,
              label="Kalman reconstruction (natural recession)")
    ax_b.set_ylabel("Head (m)")
    ax_b.set_title(f"(b) Pumping events masked → Kalman filter reconstructs natural recession from physical state model",
                   loc="left", fontsize=10)
    ax_b.legend(loc="lower left", fontsize=9, frameon=False, ncol=3)
    ax_b.set_xlim(0, n)

    # (c) Recharge ratio bar chart
    ax_c1 = fig.add_subplot(gs[2, 0])
    bars = ax_c1.bar(["Without filter\n(raw)", "With pumping filter\n(framework)"],
                     [rch_r_pct, rch_c_pct],
                     color=[COLORS["obs_raw"], COLORS["framework"]],
                     edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, [rch_r_pct, rch_c_pct]):
        ax_c1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                   f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax_c1.set_ylabel("Annual recharge ratio (% of P)")
    ax_c1.set_title(f"(c) Recharge isolation effect: ΔR = {rch_r_pct - rch_c_pct:.1f} pp",
                    loc="left", fontsize=10)
    ax_c1.set_ylim(0, max(rch_r_pct, rch_c_pct) * 1.25)
    ax_c1.grid(axis="y", linestyle=":", alpha=0.4)

    # (d) Recharge time series
    ax_c2 = fig.add_subplot(gs[2, 1])
    ax_c2.fill_between(days, 0, rech_pos_r * 1000, color=COLORS["obs_raw"],
                       alpha=0.6, label=f"Raw ({rch_r_pct:.1f}%)", step="mid")
    ax_c2.fill_between(days, 0, rech_pos_c * 1000, color=COLORS["framework"],
                       alpha=0.7, label=f"Framework ({rch_c_pct:.1f}%)", step="mid")
    ax_c2.set_xlabel("Day"); ax_c2.set_ylabel("Recharge (mm/d)")
    ax_c2.legend(loc="upper right", fontsize=9, frameon=False)
    ax_c2.set_title("(d) Daily recharge: pumping artefacts removed",
                    loc="left", fontsize=10)
    ax_c2.set_xlim(0, n)

    fig.suptitle("Figure 3 — Pumping isolation flagship case (well SH22)",
                 fontsize=13, fontweight="bold", y=0.997)
    plt.savefig(OUT / "fig3_SH22_flagship.png", bbox_inches="tight")
    plt.close()
    print("  ✓ fig3_SH22_flagship.png")


# ============================================================
# FIGURE 4 — Synthetic truth recovery
# ============================================================
def fig4_synthetic_recovery():
    inputs_dir = ROOT / "data" / "matlab_inputs"
    scenarios = ["S1", "S2", "S3", "S4", "S5"]

    truths, frameworks = {}, {}
    for sid in scenarios:
        truth_path = inputs_dir / f"SYN_{sid}_truth.json"
        if not truth_path.exists():
            continue
        with open(truth_path) as f:
            truths[sid] = json.load(f)

        well = load_well(inputs_dir / f"SYN_{sid}.txt", f"SYN_{sid}")
        ho_clean = remove_outliers(well.gw_m, sensitivity=2.0)
        opt = run_optimization(po_m=well.rain_m, ho_m=ho_clean, sn=1,
                               lag_grid=tuple(range(0, 32, 1)))
        po_shift = apply_lag(well.rain_m, opt.lag_days)
        res = run_model_core(k=opt.k, z=opt.z, sn=1,
                             po_m=po_shift, ho_m=ho_clean)
        rech_pos = (res.recharge_m_per_day + np.abs(res.recharge_m_per_day)) / 2
        rch_est_pct = rech_pos.sum() * 1000.0 / (well.rain_m.sum() * 1000.0) * 100
        rch_true_pct = (truths[sid]["annual_recharge_true_mm"] /
                        (well.rain_m.sum() * 1000.0 / (well.n_days/365.0))) * 100

        frameworks[sid] = {
            "lag_est": opt.lag_days,
            "lag_true": truths[sid]["tau_lag_true_days"],
            "rch_est": rch_est_pct,
            "rch_true": rch_true_pct,
            "n_events_true": len(truths[sid]["pumping_events"]),
            "sy_avg_est": res.n_f_avg,
            # Operational Sy = total_recharge / sum_positive_rises (WTF identity)
            "sy_true_op": float(truths[sid].get(
                "sy_operational_true",
                np.mean(truths[sid]["sy_true"])
            )),
        }

    fig, axes_grid = plt.subplots(2, 2, figsize=(12, 8.5))
    axes = axes_grid.flatten()

    # (a) Lag recovery
    ax = axes[0]
    sids = list(frameworks.keys())
    x = np.arange(len(sids))
    w = 0.35
    true_lags = [frameworks[s]["lag_true"] for s in sids]
    est_lags = [frameworks[s]["lag_est"] for s in sids]
    ax.bar(x - w/2, true_lags, w, color=COLORS["truth"], label="True τ")
    ax.bar(x + w/2, est_lags, w, color=COLORS["framework"], label="Framework estimate")
    for i, (t, e) in enumerate(zip(true_lags, est_lags)):
        ax.text(i - w/2, t + 0.5, f"{t:.0f}", ha="center", fontsize=8)
        ax.text(i + w/2, e + 0.5, f"{e:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(sids)
    ax.set_ylabel("Lag time (days)")
    ax.set_title("(a) Lag recovery", loc="left")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, max(max(true_lags), max(est_lags)) + 5)

    # (b) Recharge ratio recovery
    ax = axes[1]
    true_rch = [frameworks[s]["rch_true"] for s in sids]
    est_rch = [frameworks[s]["rch_est"] for s in sids]
    ax.bar(x - w/2, true_rch, w, color=COLORS["truth"], label="True R")
    ax.bar(x + w/2, est_rch, w, color=COLORS["framework"], label="Framework estimate")
    for i, (t, e) in enumerate(zip(true_rch, est_rch)):
        ax.text(i - w/2, t + 0.05, f"{t:.1f}", ha="center", fontsize=8)
        ax.text(i + w/2, e + 0.05, f"{e:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(sids)
    ax.set_ylabel("Recharge ratio (% of P)")
    ax.set_title("(b) Annual recharge recovery", loc="left")
    ax.legend(loc="upper right", frameon=False)

    # (c) Operational Sy recovery (WTF-identity definition: R / Σ Δh)
    ax = axes[2]
    true_sy = [frameworks[s]["sy_true_op"] for s in sids]
    est_sy = [frameworks[s]["sy_avg_est"] for s in sids]
    ax.bar(x - w/2, true_sy, w, color=COLORS["truth"],
           label="True Sy_eff (= R/ΣΔh)")
    ax.bar(x + w/2, est_sy, w, color=COLORS["framework"],
           label="Framework Sy_avg")
    for i, (t, e) in enumerate(zip(true_sy, est_sy)):
        ax.text(i - w/2, t + 0.003, f"{t:.3f}", ha="center", fontsize=7)
        ax.text(i + w/2, e + 0.003, f"{e:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(sids)
    ax.set_ylabel("Effective specific yield")
    ax.set_title("(c) Operational Sy recovery", loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=8)

    # (d) Recovery ratio consistency — calibratable bias signal
    ax = axes[3]
    rch_ratio = [frameworks[s]["rch_est"] / frameworks[s]["rch_true"]
                 for s in sids]
    sy_ratio = [frameworks[s]["sy_avg_est"] / frameworks[s]["sy_true_op"]
                for s in sids]
    lag_ratio = [frameworks[s]["lag_est"] / max(frameworks[s]["lag_true"], 1)
                 for s in sids]

    w2 = 0.27
    xs = np.arange(len(sids))
    ax.bar(xs - w2, rch_ratio, w2, color="#e74c3c",
           label=f"Recharge (mean {np.mean(rch_ratio):.2f} ± {np.std(rch_ratio):.2f})")
    ax.bar(xs, sy_ratio, w2, color="#3498db",
           label=f"Sy (mean {np.mean(sy_ratio):.2f} ± {np.std(sy_ratio):.2f})")
    ax.bar(xs + w2, lag_ratio, w2, color="#27ae60",
           label=f"Lag (mean {np.mean(lag_ratio):.2f} ± {np.std(lag_ratio):.2f})")
    ax.axhline(1.0, color="black", linestyle="--", lw=1.0, alpha=0.6,
               label="Perfect recovery (=1)")
    # Annotate consistency band for recharge (the main bias)
    ax.axhspan(min(rch_ratio), max(rch_ratio), color="#e74c3c", alpha=0.08)
    ax.set_xticks(xs); ax.set_xticklabels(sids)
    ax.set_ylabel("Recovery ratio (estimate / truth)")
    ax.set_title("(d) Recovery-ratio consistency across scenarios",
                 loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_ylim(0, max(max(rch_ratio), max(sy_ratio), max(lag_ratio)) * 1.3)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    # Footer note about calibratable bias
    ax.text(0.5, -0.18,
            (f"Recharge bias is {np.mean(rch_ratio):.2f} ± {np.std(rch_ratio):.2f}: "
             "consistent across scenarios → calibratable by an independent reference (CMB, lysimeter)."),
            transform=ax.transAxes, ha="center", fontsize=8.5,
            style="italic", color="#7f8c8d")

    fig.suptitle("Figure 4 — Synthetic-truth recovery accuracy (5 scenarios with known ground truth)",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT / "fig4_synthetic_recovery.png", bbox_inches="tight")
    plt.close()
    print("  ✓ fig4_synthetic_recovery.png")


# ============================================================
# FIGURE 5 — Multi-well recharge bar chart
# ============================================================
def fig5_multi_well_bars():
    wells_data = {name: run_well(name, WELL_SN[name])
                  for name in ["SH08", "SH11", "SH22", "SH23", "SH28"]}
    names = list(wells_data.keys())
    raw_pct = [wells_data[n]["rch_c_pct"] for n in names]
    framework_pct = [wells_data[n]["rch_pct"] for n in names]
    n_pump = [int((np.diff(wells_data[n]["well"].gw_m) < -0.3).sum())
              for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [1.5, 1]})

    # Left: bar chart per well
    x = np.arange(len(names))
    w = 0.38
    b1 = ax1.bar(x - w/2, raw_pct, w, color=COLORS["obs_raw"], label="Without pumping filter")
    b2 = ax1.bar(x + w/2, framework_pct, w, color=COLORS["framework"], label="Framework (with filter)")
    for i, (r, f) in enumerate(zip(raw_pct, framework_pct)):
        ax1.text(i - w/2, r + 0.5, f"{r:.1f}%", ha="center", fontsize=9)
        ax1.text(i + w/2, f + 0.5, f"{f:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax1.text(i, max(r, f) + 4.5, f"({n_pump[i]} drops)", ha="center",
                 fontsize=8, color="#7f8c8d", style="italic")
    ax1.set_xticks(x); ax1.set_xticklabels(names)
    ax1.set_ylabel("Annual recharge ratio (% of P)")
    ax1.set_title("(a) Per-well recharge ratios — pumping-filter effect", loc="left")
    ax1.legend(loc="upper right", frameon=False)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    ax1.set_ylim(0, max(raw_pct) * 1.25)

    # Right: reduction (pp) vs. n_pump
    ax2.scatter(n_pump,
                [r - f for r, f in zip(raw_pct, framework_pct)],
                s=90, c=COLORS["framework"], edgecolors="black", zorder=5)
    for nm, n, rdc in zip(names, n_pump,
                          [r - f for r, f in zip(raw_pct, framework_pct)]):
        ax2.annotate(nm, (n, rdc), xytext=(7, 4), textcoords="offset points",
                     fontsize=9)
    ax2.set_xlabel("Pumping suspects per record (drops > 0.3 m)")
    ax2.set_ylabel("Recharge reduction (pp)")
    ax2.set_title("(b) Reduction scales with pumping intensity", loc="left")
    ax2.grid(linestyle=":", alpha=0.4)
    ax2.axhline(0, color="black", lw=0.6)

    fig.suptitle("Figure 5 — Multi-well analysis: framework's recharge-isolation effect across 5 SH wells",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "fig5_multi_well_bars.png", bbox_inches="tight")
    plt.close()
    print("  ✓ fig5_multi_well_bars.png")


if __name__ == "__main__":
    print("=== Generating publication figures ===")
    print(f"Output dir: {OUT}")
    fig1_concept()
    fig2_5well_overview()
    fig3_SH22_flagship()
    fig4_synthetic_recovery()
    fig5_multi_well_bars()
    print("\nDone.")
