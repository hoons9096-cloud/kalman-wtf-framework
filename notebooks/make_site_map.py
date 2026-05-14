"""Generate site location map for the 5 SH wells (Siheung, Gyeonggi-do, Korea).

Reads the municipal soil-type Excel (TM coordinates EPSG:5186-ish) and produces
fig6_site_map.png:
  - Main panel: all 30 monitoring wells in Siheung as grey context, the 5
    study wells (SH-08/11/22/23/28) highlighted with topsoil-texture color
    and labeled.
  - Inset panel: Korean Peninsula schematic with a star at Siheung.

Output: notebooks/figures/fig6_site_map.png (300 dpi, English labels).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXCEL_PATH = Path(
    "/Users/choejeonghun/Dropbox/GW/Hybrid-WTF/sh/시흥시 자동관측정_토양타입.xlsx"
)
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STUDY_WELLS = ["SH-08", "SH-11", "SH-22", "SH-23", "SH-28"]

# Color by topsoil texture (USDA-style)
TEX_COLOR = {
    "Loam": "#d4a373",
    "Silt loam": "#a98467",
    "Silty clay loam": "#6c584c",
    "Sandy loam": "#e9c46a",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_excel(EXCEL_PATH)
df.columns = [str(c).strip() for c in df.columns]
df = df.rename(columns={"공번": "well", "주소": "addr"})
df["topsoil"] = df["Unnamed: 11"].astype(str).str.strip()

study = df[df["well"].isin(STUDY_WELLS)].copy()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(9.0, 7.0))
ax = fig.add_axes([0.10, 0.10, 0.78, 0.82])

# Context: all 30 wells in grey
ax.scatter(
    df["TMX"] / 1000.0, df["TMY"] / 1000.0,
    s=22, c="#cccccc", edgecolor="#888888", linewidth=0.4,
    zorder=2, label="Other monitoring wells (n=25)",
)

# Highlighted study wells
for _, row in study.iterrows():
    tex = row["topsoil"]
    color = TEX_COLOR.get(tex, "#888888")
    ax.scatter(
        row["TMX"] / 1000.0, row["TMY"] / 1000.0,
        s=180, c=color, edgecolor="black", linewidth=1.2,
        zorder=5,
    )
    # Label with offset
    ax.annotate(
        row["well"],
        xy=(row["TMX"] / 1000.0, row["TMY"] / 1000.0),
        xytext=(8, 6), textcoords="offset points",
        fontsize=11, fontweight="bold", zorder=6,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.6, alpha=0.9),
    )

# Legend for topsoil textures
legend_handles = [
    plt.scatter([], [], s=120, c=col, edgecolor="black", linewidth=1.0, label=tex)
    for tex, col in TEX_COLOR.items() if tex in study["topsoil"].values
]
legend_handles.append(
    plt.scatter([], [], s=22, c="#cccccc", edgecolor="#888888",
                linewidth=0.4, label="Other wells")
)
ax.legend(
    handles=legend_handles, loc="lower right", fontsize=9,
    frameon=True, fancybox=True, title="Topsoil texture",
    title_fontsize=9,
)

# Axes
ax.set_xlabel("Easting (km, Korean TM 2010)", fontsize=11)
ax.set_ylabel("Northing (km, Korean TM 2010)", fontsize=11)
ax.set_title(
    "Siheung-si, Gyeonggi-do, Republic of Korea — 5 study wells",
    fontsize=12, fontweight="bold", pad=10,
)
ax.grid(True, linestyle=":", alpha=0.45)
ax.set_aspect("equal", adjustable="box")

# Scale bar (2 km)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
x0 = xlim[0] + 0.05 * (xlim[1] - xlim[0])
y0 = ylim[0] + 0.05 * (ylim[1] - ylim[0])
ax.plot([x0, x0 + 2.0], [y0, y0], color="black", lw=3, zorder=10)
ax.text(x0 + 1.0, y0 + 0.15, "2 km", ha="center", va="bottom", fontsize=10, zorder=10)

# North arrow
nx = xlim[0] + 0.93 * (xlim[1] - xlim[0])
ny = ylim[0] + 0.88 * (ylim[1] - ylim[0])
ax.annotate(
    "N", xy=(nx, ny + 0.6), xytext=(nx, ny - 0.6),
    ha="center", fontsize=12, fontweight="bold",
    arrowprops=dict(arrowstyle="->", lw=1.8, color="black"),
)

# ---------------------------------------------------------------------------
# Inset: Korean Peninsula schematic
# ---------------------------------------------------------------------------
inset = fig.add_axes([0.13, 0.66, 0.18, 0.24])
# Very schematic outline (rough polygon of S. Korea)
korea = np.array([
    [126.4, 37.7], [127.4, 38.6], [128.6, 38.6], [129.4, 37.5],
    [129.5, 35.9], [128.5, 35.0], [126.6, 34.5], [126.3, 35.5],
    [126.6, 36.8], [126.4, 37.7],
])
inset.fill(korea[:, 0], korea[:, 1], color="#e8e8e8", edgecolor="black", lw=0.9)
# Siheung approx WGS84: 37.38 N, 126.80 E
inset.plot(126.80, 37.38, marker="*", color="red", markersize=14,
           markeredgecolor="black", markeredgewidth=0.6, zorder=5)
inset.annotate("Siheung", xy=(126.80, 37.38), xytext=(127.4, 37.6),
               fontsize=8, fontweight="bold",
               arrowprops=dict(arrowstyle="-", lw=0.6))
inset.set_xlim(125.5, 130.0)
inset.set_ylim(34.0, 39.0)
inset.set_xticks([])
inset.set_yticks([])
inset.set_title("Republic of Korea", fontsize=8)
for s in inset.spines.values():
    s.set_linewidth(0.8)

# ---------------------------------------------------------------------------
out = OUT_DIR / "fig6_site_map.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ {out.name}")
print(f"Saved to: {out}")
