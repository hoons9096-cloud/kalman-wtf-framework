"""Fig. 7 — Recharge plausibility plot.

Overlays the framework-calibrated Rch% for the 5 Siheung wells against
two external constraints:

  - Published Korean alluvial-aquifer recharge ranges (Moon 2004,
    Lee 2007, MOLIT 2017).
  - Top-down Siheung catchment water-balance upper bound (P-ET-Q,
    ~18% ± 5 pp).

A *hypothetical* synthetic-derived extrapolation bracket is shown as a
secondary annotation, deliberately rendered with reduced visual weight
to emphasise that it is a diagnostic of the divergence between
synthetic and field-scale constraints rather than a candidate
recharge value.

Output: notebooks/figures/fig7_plausibility.png (300 dpi, English).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- Framework Rch% (Table 2) ----------------------------------------
wells = ["SH-08", "SH-11", "SH-22", "SH-23", "SH-28"]
rch = np.array([27.47, 9.47, 19.30, 13.12, 6.86])  # %

# Hypothetical multiplicative extrapolation using synthetic recovery
# ratio (0.25 ± 0.06). Shown as a diagnostic, NOT as a recommended
# corrected value — see caption + §4.2.1 + §5.2.
rch_lo = rch / (0.25 + 0.06)
rch_hi = rch / (0.25 - 0.06)

# --- External constraints (Table 5) ----------------------------------
moon = (15, 25)
lee  = (8, 22)
molit = (10, 20)
catchment_max = 18
cb_lo, cb_hi = catchment_max - 5, catchment_max + 5

# --- Plot ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.8))

x = np.arange(len(wells))
xlim = (-0.5, len(wells) - 0.5)
y_cap = 60.0  # cap main axis at 60 % to keep the literature/water-balance
              # bands as the visual centre; off-scale extrapolations
              # are indicated with arrows.

# Literature bands — these are the *primary* visual anchors
ax.fill_between([xlim[0], xlim[1]], lee[0], lee[1],
                color="#bbdefb", alpha=0.55, zorder=1,
                label=f"Lee et al. (2007) NGMN unconfined  ({lee[0]}–{lee[1]} %)")
ax.fill_between([xlim[0], xlim[1]], molit[0], molit[1],
                color="#90caf9", alpha=0.55, zorder=2,
                label=f"MOLIT (2017) national alluvial    ({molit[0]}–{molit[1]} %)")
ax.fill_between([xlim[0], xlim[1]], moon[0], moon[1],
                color="#42a5f5", alpha=0.55, zorder=3,
                label=f"Moon et al. (2004) Korean alluvial ({moon[0]}–{moon[1]} %)")

# Catchment water-balance upper bound
ax.fill_between([xlim[0], xlim[1]], cb_lo, cb_hi,
                color="none", hatch="///", edgecolor="#d32f2f", linewidth=0,
                alpha=0.32, zorder=4,
                label=f"Siheung catchment P−ET−Q  (~{catchment_max} %, ± 5 pp)")
ax.axhline(catchment_max, color="#d32f2f", linestyle="--", linewidth=1.4,
           zorder=5)

# Framework estimates — the *primary* result the figure is about
ax.bar(x, rch, width=0.45, color="#2e7d32", edgecolor="black",
       linewidth=0.8, zorder=6, label="Framework estimate (Table 2)")
for xi, r in zip(x, rch):
    ax.annotate(f"{r:.1f} %", xy=(xi, r), xytext=(0, 4),
                textcoords="offset points", ha="center",
                fontsize=9.5, fontweight="bold", zorder=7)

# Hypothetical extrapolation — *secondary* annotation, deliberately
# soft so it does not visually dominate. Rendered as thin grey
# vertical lines with subtle caps; SH-08 is off-scale and gets an
# explicit "off-scale" arrow.
extrap_color = "#9e9e9e"
for xi, lo, hi, r in zip(x, rch_lo, rch_hi, rch):
    hi_plot = min(hi, y_cap - 1.5)
    lo_plot = min(lo, y_cap - 1.5)
    # Vertical extrapolation line
    ax.plot([xi + 0.30, xi + 0.30], [lo_plot, hi_plot],
            color=extrap_color, linewidth=1.0, alpha=0.85, zorder=6)
    # Caps
    ax.plot([xi + 0.24, xi + 0.36], [lo_plot, lo_plot],
            color=extrap_color, linewidth=0.9, alpha=0.85, zorder=6)
    if hi <= y_cap - 1.5:
        ax.plot([xi + 0.24, xi + 0.36], [hi, hi],
                color=extrap_color, linewidth=0.9, alpha=0.85, zorder=6)
    else:
        # Off-scale arrow
        ax.annotate("",
                    xy=(xi + 0.30, y_cap - 0.5),
                    xytext=(xi + 0.30, y_cap - 6),
                    arrowprops=dict(arrowstyle="->", color=extrap_color,
                                    lw=1.0, alpha=0.85), zorder=6)
        ax.text(xi + 0.42, y_cap - 4,
                f"→ {hi:.0f}%", color=extrap_color,
                fontsize=8, va="center", ha="left",
                style="italic", alpha=0.9, zorder=6)

# Phantom legend entry for the extrapolation
ax.plot([], [], color=extrap_color, linewidth=1.0,
        label="Synthetic-scale extrapolation (diagnostic only)")

# Mark SH-08 with an asterisk to flag the extreme extrapolation
ax.annotate("*", xy=(0, rch[0]), xytext=(8, 2),
            textcoords="offset points", fontsize=15, color="#d32f2f",
            fontweight="bold", zorder=8)

# Axes
ax.set_xticks(x)
ax.set_xticklabels(wells, fontsize=11)
ax.set_ylabel("Annual recharge / annual rainfall (%)", fontsize=11)
ax.set_title(
    "Recharge plausibility: framework estimates vs Korean literature "
    "and catchment water balance",
    fontsize=11.5, fontweight="bold", pad=10,
)
ax.set_xlim(*xlim)
ax.set_ylim(0, y_cap)
ax.grid(True, axis="y", linestyle=":", alpha=0.4)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.94, ncol=1)

# Footnote OUTSIDE the axes (below x-axis) so it does not occlude
# the SH-08 off-scale arrow at top-left.
fig.text(
    0.02, -0.02,
    "* SH-08's hypothetical extrapolation reaches ~ 145 %, off-scale. "
    "The grey vertical bars are a diagnostic of the divergence between "
    "synthetic-derived correction (Rch / 0.25 ± 0.06) and field-scale "
    "constraints; they are NOT proposed as recharge estimates.",
    fontsize=8, va="top", ha="left", wrap=True,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
              edgecolor="#bbbbbb", alpha=0.94),
)

fig.tight_layout()
out = OUT / "fig7_plausibility.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {out.name}")
