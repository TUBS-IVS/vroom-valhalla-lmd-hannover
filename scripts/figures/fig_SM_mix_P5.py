"""Update of fig_SM_mix_pct_init_vs_balanced for the Path-2 EWGT results,
keeping the exact same layout/styling as the 2026-05-28 version but reading
from the new Path-2 data with system-smoothed final schedules.

Layout: 2 rows x 4 columns, P in {0, 0.5, 1, 10}, theta on x-axis,
share of postal-code areas (%) stacked by delivery-day count {2..6}.

Top row: cost-optimal CD-init schedules (schedule_size_init).
Bottom row: final pipeline output (system-smoothed if available, else per-hub
            balanced; both preserve frequency by construction).

Saves results/EWGT_Results/05_optimization/
    fig_SM_mix_pct_init_vs_balanced_P5.{png,pdf}
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
KEEP_P = [0.0, 0.5, 1.0, 5.0]


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
        ax.fill_between(x, bottom, bottom + h, color=FREQ_COLOR[sz],
                         alpha=0.92)
        bottom += h
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.2)


def main():
    sched = pd.read_csv(BAL / "tab_chosen_schedules.csv")
    sched["plz"] = sched.plz.astype(str)
    # If system-smoothed schedules exist, use them as the "final" row.
    sm_path = BAL / "_tab_chosen_with_system_smoothing.csv"
    if sm_path.exists():
        sm = pd.read_csv(sm_path); sm["plz"] = sm.plz.astype(str)
        merged = sched.merge(
            sm[["penalty", "share_willing", "provider", "plz",
                 "schedule_size_system_smoothed"]],
            on=["penalty", "share_willing", "provider", "plz"], how="left")
        # Override the balanced column with smoothed where available
        merged["schedule_size_balanced"] = (
            merged.schedule_size_system_smoothed
                  .fillna(merged.schedule_size_balanced).astype(int))
        sched = merged
        print("Using system-smoothed schedules as the final row")

    avail = sorted(sched.penalty.unique())
    P_VALUES = [p for p in KEEP_P
                if any(np.isclose(p, a) for a in avail)]
    n = len(P_VALUES)

    fig, axes = plt.subplots(2, n, figsize=(2.75 * n, 4.6),
                              sharey=True, sharex=True)
    rows = [("schedule_size_init", "Cost-optimal"),
            ("schedule_size_balanced", "Fleet-balanced")]
    for ri, (col, label) in enumerate(rows):
        for ci, P in enumerate(P_VALUES):
            ax = axes[ri, ci]
            _stack(ax, sched[np.isclose(sched.penalty, P)], col, P)
            if ri == 0:
                ax.set_title(rf"Service penalty $P = {P:g}$ €/p/d",
                              fontsize=9.5)
        axes[ri, 0].set_ylabel(label, fontsize=9.5)
    fig.tight_layout(rect=[0.05, 0.16, 1, 1], pad=0.4, w_pad=0.3, h_pad=0.6)

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
        fig.savefig(OUT / f"fig_SM_mix_pct_init_vs_balanced_P5.{ext}",
                     bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'fig_SM_mix_pct_init_vs_balanced.png'}")
    print(f"saved {OUT/'fig_SM_mix_pct_init_vs_balanced.pdf'}")


if __name__ == "__main__":
    main()
