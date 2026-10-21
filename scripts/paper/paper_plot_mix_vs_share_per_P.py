"""Stacked area: delivery-frequency mix vs share_willing for each penalty P.

Multi-panel version of fig04 — one panel per service-penalty level so the
reader sees how the mix evolves with both share AND penalty.

Status B (Task 19): 74_-legacy's tab_chosen_schedules.csv has no unsuffixed
schedule_size column; schedule_size_balanced (the operator-polished/final
plan) is aliased to schedule_size, same convention as
paper_plot_maps_per_P.py.
"""
from pathlib import Path
import argparse
import sys
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paper_v6_common as V6  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
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
    global OVERNIGHT
    ap = argparse.ArgumentParser(description=__doc__)
    V6.add_v6_cli_args(ap, needs_legacy=True)
    args = ap.parse_args()
    v6_mode = args.legacy_dir is not None or args.rev_dir is not None
    if args.legacy_dir is not None:
        in_dir = Path(args.legacy_dir)
    elif args.rev_dir is not None:
        in_dir, _ = V6.run_legacy_adapter(
            args.rev_dir, Path(args.out_dir or OVERNIGHT) / "_legacy")
    else:
        in_dir = OVERNIGHT
    out_dir = Path(args.out_dir) if args.out_dir is not None else OVERNIGHT
    out_dir.mkdir(parents=True, exist_ok=True)
    OVERNIGHT = out_dir
    src_note = ("B: 74_-legacy tab_chosen_schedules.csv (schedule_size_balanced)"
               if v6_mode else "tab_chosen_schedules.csv (historical path)")

    chosen = pd.read_csv(in_dir / "tab_chosen_schedules.csv")
    if "schedule_size" not in chosen.columns:
        chosen = chosen.rename(columns={"schedule_size_balanced": "schedule_size"})
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
    V6.add_provenance_footer(fig, plan="operator-polished (balanced)",
                             script="paper_plot_mix_vs_share_per_P.py",
                             source=src_note)
    V6.savefig_pair(fig, OVERNIGHT / "fig06_schedule_mix_vs_share_per_P.png",
                    OVERNIGHT / "fig06_schedule_mix_vs_share_per_P.pdf")
    plt.close(fig)
    print(f"Saved fig06_schedule_mix_vs_share_per_P.png ({n} panels)")


if __name__ == "__main__":
    main()
