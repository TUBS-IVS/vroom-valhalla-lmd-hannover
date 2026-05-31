"""How much smoother is the fleet after balancing? Read tab_fleet_per_hub.csv
(per-hub, per-day, before/after) and quantify the smoothing across the grid:

  - Peak-day fleet reduction (%) per (P, theta)
  - Mo-Sa spread reduction (%) per (P, theta)
  - Coefficient of variation reduction
  - Concrete Mo-Sa fleet pattern at the sweet-spot per provider (init vs balanced)

Outputs (under results/overnight_2026_05_29_path2/):
  _fleet_smoothing.csv        per-cell aggregate metrics
  _fig_fleet_smoothing.png    2-panel heatmap (peak red %, spread red %) + Mo-Sa example
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
OUT = ROOT / "results" / "overnight_2026_05_29_path2"

rcParams.update({
    "font.family": "serif", "font.size": 9,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "savefig.bbox": "tight", "savefig.dpi": 180, "pdf.fonttype": 42,
})


def main():
    f = pd.read_csv(OUT / "tab_fleet_per_hub.csv")

    # Aggregate per (P, theta): sum across hubs to get provider-level daily fleet
    # and then across providers to get system-level daily fleet
    sys_per_day = (f.groupby(["penalty", "share_willing", "day"])
                    .agg(fleet_before=("fleet_before", "sum"),
                         fleet_after=("fleet_after", "sum"))
                    .reset_index())
    cells = (sys_per_day.groupby(["penalty", "share_willing"])
             .apply(lambda g: pd.Series({
                 "peak_b": g.fleet_before.max(),
                 "peak_a": g.fleet_after.max(),
                 "min_b": g.fleet_before.min(),
                 "min_a": g.fleet_after.min(),
                 "mean_b": g.fleet_before.mean(),
                 "mean_a": g.fleet_after.mean(),
                 "std_b": g.fleet_before.std(),
                 "std_a": g.fleet_after.std(),
             })).reset_index())
    cells["peak_red_pct"] = 100 * (cells.peak_b - cells.peak_a) / cells.peak_b.clip(lower=1)
    cells["spread_b"] = cells.peak_b - cells.min_b
    cells["spread_a"] = cells.peak_a - cells.min_a
    cells["spread_red_pct"] = (100 * (cells.spread_b - cells.spread_a)
                                / cells.spread_b.clip(lower=1))
    cells["cv_b"] = cells.std_b / cells.mean_b.clip(lower=1)
    cells["cv_a"] = cells.std_a / cells.mean_a.clip(lower=1)
    cells["cv_red_pct"] = 100 * (cells.cv_b - cells.cv_a) / cells.cv_b.clip(lower=1e-6)
    cells.to_csv(OUT / "_fleet_smoothing.csv", index=False)

    print("=== Aggregated smoothing metrics (sample cells) ===")
    print(cells[["penalty", "share_willing", "peak_b", "peak_a", "peak_red_pct",
                 "spread_b", "spread_a", "spread_red_pct", "cv_red_pct"]]
          .round(1).to_string(index=False))

    # 3-panel figure: peak heatmap, spread heatmap, Mo-Sa example
    piv_peak = cells.pivot(index="penalty", columns="share_willing",
                            values="peak_red_pct")
    piv_spread = cells.pivot(index="penalty", columns="share_willing",
                              values="spread_red_pct")

    fig = plt.figure(figsize=(11, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.28)
    ax_pk = fig.add_subplot(gs[0, 0])
    ax_sp = fig.add_subplot(gs[0, 1])
    ax_ms = fig.add_subplot(gs[0, 2])

    def heat(ax, mat, vmax, title, cb_label):
        im = ax.imshow(mat.values, aspect="auto", cmap="magma", vmin=0,
                        vmax=vmax)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f"{x*100:.0f}" for x in mat.columns])
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"{p:g}" for p in mat.index])
        ax.set_xlabel(r"$\theta$ [%]")
        ax.set_title(title, fontsize=8.5)
        thr = vmax * 0.6
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v < thr else "black", fontsize=5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cb_label, fontsize=7)
        cb.ax.tick_params(labelsize=6)

    pk_max = float(np.ceil(piv_peak.values.max() / 5) * 5) if piv_peak.values.size else 30
    sp_max = float(np.ceil(piv_spread.values.max() / 5) * 5) if piv_spread.values.size else 50
    heat(ax_pk, piv_peak, pk_max, "(a) Peak-Fahrzeug-Reduktion [%]", "Peak red %")
    ax_pk.set_ylabel(r"$P$ [€/p/d]")
    heat(ax_sp, piv_spread, sp_max, "(b) Mo-Sa Spread-Reduktion [%]", "Spread red %")

    # Mo-Sa example: pick the lowest available P at theta=1 (max smoothing case)
    target_th = 1.0
    available_p = sorted(cells.penalty.unique())
    example_p = next((p for p in available_p
                       if cells[(np.isclose(cells.penalty, p))
                                & (np.isclose(cells.share_willing, target_th))].size),
                      available_p[0])
    msa = sys_per_day[(np.isclose(sys_per_day.penalty, example_p))
                       & (np.isclose(sys_per_day.share_willing, target_th))]
    days = ["Mo", "Di", "Mi", "Do", "Fr", "Sa"]
    if len(msa) == 6:
        x = np.arange(6)
        w = 0.4
        ax_ms.bar(x - w / 2, msa.fleet_before.values, w,
                  color="#e76f51", edgecolor="black", linewidth=0.4,
                  label="init")
        ax_ms.bar(x + w / 2, msa.fleet_after.values, w,
                  color="#2a9d8f", edgecolor="black", linewidth=0.4,
                  label="balanced")
        ax_ms.set_xticks(x); ax_ms.set_xticklabels(days)
        ax_ms.set_ylabel("Fahrzeuge gesamt")
        ax_ms.set_title(f"(c) Mo–Sa System-Flotte, $P={example_p:g}$, "
                        rf"$\theta={target_th:g}$", fontsize=8.5)
        ax_ms.legend(loc="lower right", fontsize=8)
        ax_ms.grid(axis="y", alpha=0.3)

    fig.savefig(OUT / "_fig_fleet_smoothing.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {OUT / '_fleet_smoothing.csv'}")
    print(f"saved {OUT / '_fig_fleet_smoothing.png'}")


if __name__ == "__main__":
    main()
