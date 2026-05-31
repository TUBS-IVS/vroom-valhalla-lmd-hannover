"""Percent-normalized delivery-frequency mix vs share willing, per penalty P,
for BOTH cost-optimal (init) and fleet-balanced output as two stacked 1xN rows.

Same palette/style as paper_plot_mix_vs_share_per_P.py. Purpose: test whether a
single-row-per-variant layout (init row + balanced row) keeps the 8 panels
legible or gets too small.

Saves results/paper_final_2026_05_28/05_optimization/
    fig_SM_mix_pct_init_vs_balanced.{png,pdf}
"""
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
BAL = ROOT / "results" / "overnight_2026_05_29_path2"
OUT = ROOT / "results" / "paper_final_2026_05_30" / "05_optimization"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})
FREQ_COLOR = {2: "#1d3557", 3: "#2a9d8f", 4: "#e9c46a", 5: "#f4a261", 6: "#e76f51"}
KEEP_P = [0.0, 0.5, 1.0, 10.0]   # 4 most informative penalties


def _stack(ax, sub, col, P):
    agg = sub.groupby(["share_willing", col]).size().reset_index(name="cnt")
    piv = (agg.pivot(index="share_willing", columns=col, values="cnt")
           .fillna(0).sort_index())
    piv = piv.div(piv.sum(axis=1), axis=0) * 100
    x = piv.index.values * 100
    bottom = np.zeros(len(piv))
    for sz in (2, 3, 4, 5, 6):
        if sz not in piv.columns:
            continue
        h = piv[sz].values
        ax.fill_between(x, bottom, bottom + h, color=FREQ_COLOR[sz], alpha=0.92)
        bottom += h
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.2)


def main():
    sched = pd.read_csv(BAL / "tab_chosen_schedules.csv")
    avail = sorted(sched.penalty.unique())
    P_VALUES = [p for p in KEEP_P
                if any(np.isclose(p, a) for a in avail)]
    n = len(P_VALUES)

    fig, axes = plt.subplots(2, n, figsize=(2.75 * n, 4.6),
                             sharey=True, sharex=True)
    rows = [("schedule_size_init", "Cost-optimal (init)"),
            ("schedule_size_balanced", "Fleet-balanced")]
    for ri, (col, label) in enumerate(rows):
        for ci, P in enumerate(P_VALUES):
            ax = axes[ri, ci]
            _stack(ax, sched[np.isclose(sched.penalty, P)], col, P)
            if ri == 0:
                ax.set_title(f"Service penalty $P = {P:g}$ €/p/d", fontsize=9.5)
        axes[ri, 0].set_ylabel(label, fontsize=9.5)
    fig.tight_layout(rect=[0.05, 0.16, 1, 1], pad=0.4, w_pad=0.3, h_pad=0.6)

    # Place the shared axis labels relative to the actual content edges (not
    # fixed figure coords): centred on the panel grid, the y-label snug to the
    # left of the row labels, and the legend directly below the x-label.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bb_tl = axes[0, 0].get_position()
    bb_br = axes[-1, -1].get_position()
    cx = (bb_tl.x0 + bb_br.x1) / 2.0
    cy = (bb_br.y0 + bb_tl.y1) / 2.0
    # x-label just below the bottom-row tick labels
    tb = axes[-1, 0].get_tightbbox(r).transformed(inv)
    xlab_y = tb.y0 - 0.030
    fig.text(cx, xlab_y, r"Willingness-to-wait share $\theta$ [%]",
             ha="center", va="top", fontsize=11)
    # y-label snug to the left of the row labels (uses the real left extent)
    lb = axes[0, 0].get_tightbbox(r).transformed(inv)
    fig.text(lb.x0 - 0.004, cy, "Share of postal-code areas [%]", rotation=90,
             ha="right", va="center", fontsize=10)
    # framed legend directly below the x-label (top anchored under it)
    handles = [Patch(facecolor=FREQ_COLOR[s], label=f"{s} day/wk") for s in (2, 3, 4, 5, 6)]
    fig.legend(handles=handles, title="Delivery days per week", loc="upper center",
               ncol=5, frameon=True, framealpha=0.9, edgecolor="0.8",
               fontsize=9, title_fontsize=9, bbox_to_anchor=(cx, xlab_y - 0.055),
               handlelength=1.4, columnspacing=1.3, borderpad=0.4)
    fig.savefig(OUT / "fig_SM_mix_pct_init_vs_balanced.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_SM_mix_pct_init_vs_balanced.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved fig_SM_mix_pct_init_vs_balanced ({n} panels/row, P={[f'{p:g}' for p in P_VALUES]})")


if __name__ == "__main__":
    main()
