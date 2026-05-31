"""Single-row variant of fig_SM_mix that shows ALL eight service-penalty
levels in the Path-2 sweep, instead of the prior 2x4 init-vs-balanced
panel. Because frequency-preserving balancing leaves the per-cell
delivery-day count unchanged, the cost-optimal mix already represents
the final pipeline output for this view; the second row is therefore
redundant and is dropped to fit eight panels on one row.

Output: results/EWGT_Results/fig_SM_mix_pct_8P.{png,pdf}
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
OUT = ROOT / "results" / "EWGT_Results"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})
FREQ_COLOR = {2: "#1d3557", 3: "#2a9d8f", 4: "#e9c46a",
              5: "#f4a261", 6: "#e76f51"}


def _stack(ax, sub, col):
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
        ax.fill_between(x, bottom, bottom + h, color=FREQ_COLOR[sz],
                         alpha=0.92)
        bottom += h
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.2)


def main():
    sched = pd.read_csv(BAL / "tab_chosen_schedules.csv")
    sched["plz"] = sched.plz.astype(str)

    P_VALUES = sorted(sched.penalty.unique())
    P_VALUES = [p for p in P_VALUES if not np.isclose(p, 0.4)]
    n = len(P_VALUES)
    print(f"Rendering {n} penalty levels: {P_VALUES}")

    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(2.75 * ncols, 2.3 * nrows),
                              sharey=True, sharex=True)
    for idx, P in enumerate(P_VALUES):
        r_, c_ = divmod(idx, ncols)
        ax = axes[r_, c_]
        _stack(ax, sched[np.isclose(sched.penalty, P)],
                "schedule_size_init")
        ax.set_title(rf"Service penalty $P = {P:g}$ €/p/d",
                      fontsize=9.5)
    # hide any unused
    for idx in range(n, nrows * ncols):
        r_, c_ = divmod(idx, ncols)
        axes[r_, c_].set_visible(False)

    fig.tight_layout(rect=[0.05, 0.16, 1, 1], pad=0.4,
                      w_pad=0.3, h_pad=0.6)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bb_tl = axes[0, 0].get_position()
    bb_br = axes[-1, -1].get_position()
    cx = (bb_tl.x0 + bb_br.x1) / 2.0
    cy = (bb_br.y0 + bb_tl.y1) / 2.0
    tb = axes[-1, 0].get_tightbbox(r).transformed(inv)
    xlab_y = tb.y0 - 0.030
    fig.text(cx, xlab_y, r"Willingness-to-wait share $\theta$ [%]",
             ha="center", va="top", fontsize=11)
    lb = axes[0, 0].get_tightbbox(r).transformed(inv)
    fig.text(lb.x0 - 0.004, cy, "Share of postal-code areas [%]",
             rotation=90, ha="right", va="center", fontsize=10)
    handles = [Patch(facecolor=FREQ_COLOR[s], label=f"{s} day/wk")
               for s in (2, 3, 4, 5, 6)]
    fig.legend(handles=handles, title="Delivery days per week",
                loc="upper center", ncol=5,
                frameon=True, framealpha=0.9, edgecolor="0.8",
                fontsize=9, title_fontsize=9,
                bbox_to_anchor=(cx, xlab_y - 0.055),
                handlelength=1.4, columnspacing=1.3, borderpad=0.4)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_SM_mix_pct_8P.{ext}",
                     bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'fig_SM_mix_pct_8P.png'}")
    print(f"saved {OUT/'fig_SM_mix_pct_8P.pdf'}")


if __name__ == "__main__":
    main()
