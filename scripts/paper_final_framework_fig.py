"""Simple infographic overview of the methodology (one clean flow incl. the
perturbed training data). Pure drawing. Output: 00_overview/fig_framework.{png,pdf}
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path as SysPath

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib import rcParams

ROOT = SysPath(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_final_2026_05_28" / "00_overview"
OUT.mkdir(parents=True, exist_ok=True)
rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                 "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300})

PETROL = "#2a7d83"; PETROL_DK = "#16545a"; OCHRE = "#c08a3e"
TXT = "#1d2b2d"; BODY = "#5c6a6c"; ARR = "#6b7574"


def card(ax, cx, cy, w, h, title, body, strip, striph=4.4, tfs=10.5, bfs=8.6):
    x, y = cx - w / 2, cy - h / 2
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.3",
                         fc="white", ec="#c9d0d0", lw=0.8, zorder=4)
    box.set_path_effects([pe.withSimplePatchShadow(offset=(2.0, -2.0),
                          shadow_rgbFace="0.45", alpha=0.22, rho=0.4)])
    ax.add_patch(box)
    st = Rectangle((x, y + h - striph), w, striph, fc=strip, ec="none", zorder=5)
    st.set_clip_path(box)
    ax.add_patch(st)
    ax.text(cx, y + h - striph / 2, title, ha="center", va="center", fontsize=tfs,
            fontweight="bold", color="white", zorder=6)
    ax.text(cx, y + (h - striph) / 2 - 0.3, body, ha="center", va="center", fontsize=bfs,
            color=BODY, zorder=6, linespacing=1.35)


def main():
    fig, ax = plt.subplots(figsize=(14, 4.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 31); ax.axis("off")

    cy, h, w = 20, 15, 14.2
    cxs = [8.6, 24.1, 39.6, 55.1, 70.6, 86.1]
    spec = [("Demand data", "daily parcel\nvolumes", PETROL),
            ("Perturbation", "vary demand,\ndays & stops", PETROL),
            ("VROOM / Valhalla", "exact route\ncost", PETROL),
            ("ML surrogate", "$\\tilde C=\\alpha\\,\\hat C_{\\mathrm{Dag}}+g(\\mathbf{x})$\nDaganzo + LightGBM", PETROL_DK),
            ("Scheduling", "best weekly\npattern (of 39)", PETROL),
            ("Outcome", "cost savings\n& service KPIs", PETROL)]
    for cx, (t, b, s) in zip(cxs, spec):
        card(ax, cx, cy, w, h, t, b, s, tfs=9.6 if t == "Schedule optimisation" else 10.5)
    for i in range(5):
        ax.add_patch(FancyArrowPatch((cxs[i] + w / 2, cy), (cxs[i + 1] - w / 2, cy),
                     arrowstyle="-|>", mutation_scale=16, lw=2.1, color=ARR,
                     shrinkA=0, shrinkB=0, zorder=3))

    # bracket under the three training-data boxes
    x0, x1 = cxs[0] - w / 2, cxs[2] + w / 2
    yb = cy - h / 2 - 2.0
    ax.plot([x0, x0, x1, x1], [yb + 1.2, yb, yb, yb + 1.2], color=OCHRE, lw=1.4, zorder=2)
    ax.text((x0 + x1) / 2, yb - 1.4, "perturbed, solver-labelled training data",
            ha="center", va="top", fontsize=9, color="#9a7333", fontweight="bold", zorder=2)

    fig.savefig(OUT / "fig_framework.png", dpi=300)
    fig.savefig(OUT / "fig_framework.pdf")
    plt.close(fig)
    print("  ✓ fig_framework ->", OUT / "fig_framework.png")


if __name__ == "__main__":
    main()
