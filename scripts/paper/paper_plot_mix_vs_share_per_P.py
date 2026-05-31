"""Stacked area: delivery-frequency mix vs share_willing for each penalty P.

Multi-panel version of fig04 — one panel per service-penalty level so the
reader sees how the mix evolves with both share AND penalty.
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

ROOT = Path(__file__).resolve().parents[1]
OVERNIGHT = ROOT / "results" / "overnight_2026_05_27"

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

FREQ_COLOR = {
    1: "#9d2226", 2: "#1d3557", 3: "#2a9d8f",
    4: "#e9c46a", 5: "#f4a261", 6: "#e76f51",
}


def main():
    chosen = pd.read_csv(OVERNIGHT / "tab_chosen_schedules.csv")
    pen_values = sorted(chosen.penalty.unique())
    n = len(pen_values)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.5),
                              sharey=True, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, P in zip(axes, pen_values):
        sub = chosen[np.isclose(chosen.penalty, P)]
        agg = sub.groupby(["share_willing", "schedule_size"]).size().reset_index(name="cnt")
        pivot = (agg.pivot(index="share_willing", columns="schedule_size", values="cnt")
                  .fillna(0).sort_index())
        pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100   # %
        x = pivot.index.values * 100
        bottom = np.zeros(len(pivot))
        for sz in (2, 3, 4, 5, 6):
            if sz not in pivot.columns:
                continue
            h = pivot[sz].values
            ax.fill_between(x, bottom, bottom + h,
                             color=FREQ_COLOR[sz], alpha=0.92,
                             label=f"{sz} day/wk" if P == pen_values[-1] else None)
            bottom += h
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Share willing to wait [%]")
        ax.set_title(f"$P = {P:g}$ €/parcel/day")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Delivery-frequency mix [%]")
    axes[-1].legend(title="Delivery days/week", loc="upper left",
                    bbox_to_anchor=(1.02, 1.0), frameon=True)
    fig.suptitle("Delivery-frequency mix shifts with willingness, "
                  "for varying service penalty $P$",
                  fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OVERNIGHT / "fig06_schedule_mix_vs_share_per_P.png")
    fig.savefig(OVERNIGHT / "fig06_schedule_mix_vs_share_per_P.pdf")
    plt.close(fig)
    print(f"Saved fig06_schedule_mix_vs_share_per_P.png ({n} panels)")


if __name__ == "__main__":
    main()
