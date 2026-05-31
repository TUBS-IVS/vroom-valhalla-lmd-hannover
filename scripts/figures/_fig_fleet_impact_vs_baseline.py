"""Fleet-impact heatmaps relative to the daily baseline (not relative to init).

Three panels over the (P, theta) grid:
  (a) Cumulative peak-fleet reduction baseline -> balanced [%]
      = how much of the daily-baseline peak the full Path-2 pipeline (CD-init +
      per-hub balancing) removes.
  (b) Cumulative coefficient of variation (CV) of the Mo-Sa system fleet,
      baseline -> balanced [%]
      = how much smoother the weekly fleet looks after the full pipeline.
  (c) Combined operational metric: total weekly fleet (sum of Mo-Sa) reduction
      baseline -> balanced [%]
      = total transport-capacity demand savings.

Output:
  results/paper_final_2026_05_30/06_balancing/fig_fleet_impact_vs_baseline.{png,pdf}
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
PATH2 = ROOT / "results" / "overnight_2026_05_29_path2"
OUT = ROOT / "results" / "paper_final_2026_05_30" / "06_balancing"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 9,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 5, "ytick.labelsize": 5,
    "savefig.bbox": "tight", "savefig.dpi": 220, "pdf.fonttype": 42,
})


def main():
    f = pd.read_csv(PATH2 / "tab_fleet_per_hub.csv")

    # System fleet per (P, theta) per day (sum over providers / hubs)
    sys_per_day = (f.groupby(["penalty", "share_willing", "day"])
                    .agg(fleet_before=("fleet_before", "sum"),
                         fleet_after=("fleet_after", "sum"))
                    .reset_index())

    # Baseline fleet (per day) = fleet_before at share=0 (all daily)
    base_pivot = (sys_per_day[np.isclose(sys_per_day.share_willing, 0.0)]
                   .groupby("day").fleet_before.mean())  # average across P (same at share=0)
    baseline_by_day = np.array([base_pivot.loc[d] for d in range(6)])
    baseline_peak = float(baseline_by_day.max())
    baseline_total = float(baseline_by_day.sum())
    baseline_cv = float(baseline_by_day.std() / baseline_by_day.mean())
    print(f"Baseline (daily delivery, theta=0):")
    print(f"  Mo-Sa fleet: {baseline_by_day.astype(int).tolist()}")
    print(f"  peak={baseline_peak:.0f}, total={baseline_total:.0f}, CV={baseline_cv:.3f}")

    rows = []
    for (P, sh), g in sys_per_day.groupby(["penalty", "share_willing"]):
        g = g.sort_values("day")
        fa = g.fleet_after.values
        peak_a = fa.max()
        total_a = fa.sum()
        cv_a = fa.std() / fa.mean() if fa.mean() > 0 else 0.0
        rows.append({
            "penalty": P, "share_willing": sh,
            "peak_a": peak_a, "total_a": total_a, "cv_a": cv_a,
            "peak_red_pct": 100 * (baseline_peak - peak_a) / baseline_peak,
            "cv_red_pct": 100 * (baseline_cv - cv_a) / baseline_cv,
            "total_red_pct": 100 * (baseline_total - total_a) / baseline_total,
        })
    cells = pd.DataFrame(rows)
    cells.to_csv(OUT / "fleet_impact_vs_baseline.csv", index=False)

    # Drop P=0.4 if present (not in our grid)
    cells = cells[~np.isclose(cells.penalty, 0.4)]

    piv_peak = cells.pivot(index="penalty", columns="share_willing",
                            values="peak_red_pct")
    piv_cv = cells.pivot(index="penalty", columns="share_willing",
                          values="cv_red_pct")
    piv_total = cells.pivot(index="penalty", columns="share_willing",
                             values="total_red_pct")

    def heat(ax, mat, cmap, title, label, vmin=None, vmax=None,
             annot_fmt="{:.1f}"):
        if vmin is None: vmin = float(np.nanmin(mat.values))
        if vmax is None: vmax = float(np.nanmax(mat.values))
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap,
                        vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f"{x*100:.0f}" for x in mat.columns])
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"{p:g}" for p in mat.index])
        ax.set_xlabel(r"Willingness-to-wait share $\theta$ [%]")
        ax.set_title(title, fontsize=8.5)
        thr = vmin + (vmax - vmin) * 0.55
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, annot_fmt.format(v), ha="center", va="center",
                        color="white" if v < thr else "black", fontsize=4.2)
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=6)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.0))
    pk_max = float(np.ceil(piv_peak.values.max() / 5) * 5)
    cv_max = float(np.ceil(piv_cv.values.max() / 10) * 10)
    tot_max = float(np.ceil(piv_total.values.max() / 5) * 5)
    pk_min = float(np.floor(min(0, piv_peak.values.min()) / 5) * 5)
    heat(axes[0], piv_peak, "magma",
          "(a) Peak-Flotten-Reduktion [%] vs daily baseline",
          "Peak reduction %", vmin=pk_min, vmax=pk_max)
    axes[0].set_ylabel(r"Service penalty $P$ [€/p/d]")
    heat(axes[1], piv_cv, "viridis",
          "(b) Mo-Sa Coefficient of Variation Reduktion [%]",
          "CV reduction %", vmin=0, vmax=cv_max)
    heat(axes[2], piv_total, "magma",
          "(c) Wochen-Gesamtflotte Reduktion [%]",
          "Total fleet reduction %", vmin=0, vmax=tot_max)

    fig.tight_layout(w_pad=0.6)
    fig.savefig(OUT / "fig_fleet_impact_vs_baseline.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_fleet_impact_vs_baseline.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {OUT/'fig_fleet_impact_vs_baseline.png'}")
    print(f"\nKey numbers at sweet-spot P=0.5, theta=1.0:")
    sweet = cells[(np.isclose(cells.penalty, 0.5)) &
                   (np.isclose(cells.share_willing, 1.0))].iloc[0]
    print(f"  Peak reduction vs baseline:  {sweet.peak_red_pct:.1f} %")
    print(f"  CV reduction vs baseline:    {sweet.cv_red_pct:.1f} %")
    print(f"  Total fleet reduction:       {sweet.total_red_pct:.1f} %")


if __name__ == "__main__":
    main()
