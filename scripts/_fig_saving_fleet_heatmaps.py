"""Three compact heatmaps over the (P, theta) grid for the paper:
  (a) cost saving [%] of the cost-optimal (init) plan vs daily baseline,
  (b) cost saving [%] of the fleet-balanced plan,
  (c) peak-fleet reduction [%] achieved by fleet balancing.

Same agg logic as paper_final_v2.load_data (reads tab_balancing_summary.csv +
tab_chosen_schedules.csv). Paper style (serif / dejavuserif mathtext), sized to
sit on the text width of a Transportation Research Procedia single column.

Saves results/paper_final_2026_05_28/05_optimization/
    fig_saving_fleet_heatmaps.{png,pdf}
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
BAL = ROOT / "results" / "overnight_2026_05_29_path2"
OUT = ROOT / "results" / "paper_final_2026_05_30" / "05_optimization"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 9,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 5, "ytick.labelsize": 5,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})


def load_agg():
    s = pd.read_csv(BAL / "tab_balancing_summary.csv")
    sched = pd.read_csv(BAL / "tab_chosen_schedules.csv")
    agg = s.groupby(["penalty", "share_willing"], as_index=False).agg(
        init_cost_eur=("init_cost_eur", "sum"),
        bal_cost_eur=("balanced_cost_eur", "sum"),
        max_fleet_before=("max_fleet_before", "sum"),
        max_fleet_after=("max_fleet_after", "sum"),
    )
    baseline = float(agg[agg.share_willing == 0.0].bal_cost_eur.max())
    agg["saving_init_pct"] = 100 * (baseline - agg.init_cost_eur) / baseline
    agg["saving_bal_pct"] = 100 * (baseline - agg.bal_cost_eur) / baseline
    agg["fleet_red_pct"] = (100 * (agg.max_fleet_before - agg.max_fleet_after)
                            / agg.max_fleet_before.clip(lower=1))
    return agg, baseline


def _heat(ax, mat, cmap, vmin, vmax, title):
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([f"{x*100:.0f}" for x in mat.columns])
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([f"{p:g}" for p in mat.index])
    ax.set_xlabel(r"Willingness-to-wait share $\theta$ [%]")
    ax.set_title(title, fontsize=7.5)
    thr = vmin + (vmax - vmin) * 0.55
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if v < thr else "black", fontsize=4.0)
    return im


def main():
    agg, baseline = load_agg()
    agg = agg[~np.isclose(agg.penalty, 0.4)]   # 0.4 is the fine-sweep sweet spot, shown separately
    sav_init = agg.pivot(index="penalty", columns="share_willing", values="saving_init_pct")
    sav_bal = agg.pivot(index="penalty", columns="share_willing", values="saving_bal_pct")
    fleet = agg.pivot(index="penalty", columns="share_willing", values="fleet_red_pct")

    s_vmin = min(sav_init.values.min(), sav_bal.values.min())
    s_vmax = max(sav_init.values.max(), sav_bal.values.max())
    f_vmax = float(np.ceil(fleet.values.max() / 5) * 5)

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    im_a = _heat(axes[0], sav_init, "viridis", s_vmin, s_vmax, "(a) Cost saving [%], cost-optimal")
    _heat(axes[1], sav_bal, "viridis", s_vmin, s_vmax, "(b) Cost saving [%], fleet-balanced")
    im_c = _heat(axes[2], fleet, "magma", 0.0, f_vmax, "(c) Peak-fleet reduction [%]")
    axes[0].set_ylabel(r"Service penalty $P$ [€/p/d]")

    cb1 = fig.colorbar(im_a, ax=axes[1], fraction=0.046, pad=0.03)
    cb1.set_label("Cost saving [%]", fontsize=8)
    cb1.ax.tick_params(labelsize=7)
    cb2 = fig.colorbar(im_c, ax=axes[2], fraction=0.046, pad=0.03)
    cb2.set_label("Peak-fleet reduction [%]", fontsize=8)
    cb2.ax.tick_params(labelsize=7)

    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "fig_saving_fleet_heatmaps.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_saving_fleet_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved fig_saving_fleet_heatmaps (baseline={baseline/1e3:.0f} k€/wk; "
          f"saving range [{s_vmin:.1f},{s_vmax:.1f}]%; "
          f"fleet-red max {fleet.values.max():.1f}% at "
          f"P={fleet.max(axis=1).idxmax():g})")


if __name__ == "__main__":
    main()
