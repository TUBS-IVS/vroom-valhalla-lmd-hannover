"""Fleet-impact heatmaps (peak / CV / total weekly fleet reduction) vs daily
baseline, rendered WIDER for the EWGT paper, flat into results/EWGT_Results/.

Same calculation as scripts/_fig_fleet_impact_vs_baseline.py, but with the
EWGT label conventions (English, "postal-code area" phrasing) and the wider
13.5" layout used by the other EWGT 3-panel figure.
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
OUT = ROOT / "results" / "EWGT_Results"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    f = pd.read_csv(PATH2 / "tab_fleet_per_hub.csv")

    sys_per_day = (f.groupby(["penalty", "share_willing", "day"])
                    .agg(fleet_before=("fleet_before", "sum"),
                         fleet_after=("fleet_after", "sum"))
                    .reset_index())

    base_pivot = (sys_per_day[np.isclose(sys_per_day.share_willing, 0.0)]
                   .groupby("day").fleet_before.mean())
    baseline_by_day = np.array([base_pivot.loc[d] for d in range(6)])
    baseline_peak = float(baseline_by_day.max())
    baseline_total = float(baseline_by_day.sum())
    baseline_cv = float(baseline_by_day.std() / baseline_by_day.mean())
    print(f"Baseline (daily delivery, theta=0):")
    print(f"  Mo-Sa fleet: {baseline_by_day.astype(int).tolist()}")
    print(f"  peak={baseline_peak:.0f}, total={baseline_total:.0f}, "
          f"CV={baseline_cv:.3f}")

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
            "total_red_pct":
                100 * (baseline_total - total_a) / baseline_total,
        })
    cells = pd.DataFrame(rows)
    cells = cells[~np.isclose(cells.penalty, 0.4)]

    piv_peak = cells.pivot(index="penalty", columns="share_willing",
                            values="peak_red_pct")
    piv_cv = cells.pivot(index="penalty", columns="share_willing",
                          values="cv_red_pct")
    piv_total = cells.pivot(index="penalty", columns="share_willing",
                             values="total_red_pct")

    def heat(ax, mat, cmap, title, vmin=None, vmax=None):
        if vmin is None:
            vmin = float(np.nanmin(mat.values))
        if vmax is None:
            vmax = float(np.nanmax(mat.values))
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap,
                        vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f"{x*100:.0f}" for x in mat.columns], fontsize=8)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"{p:g}" for p in mat.index], fontsize=8)
        ax.set_xlabel(r"$\theta$ [%]", fontsize=10)
        ax.set_title(title, fontsize=11)
        thr = vmin + (vmax - vmin) * 0.55
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v < thr else "black", fontsize=6.5)
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("[%]", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        return im

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    pk_max = float(np.ceil(piv_peak.values.max() / 5) * 5)
    cv_max = float(np.ceil(piv_cv.values.max() / 10) * 10)
    tot_max = float(np.ceil(piv_total.values.max() / 5) * 5)
    pk_min = float(np.floor(min(0, piv_peak.values.min()) / 5) * 5)

    heat(axes[0], piv_peak, "magma",
          "(a) Peak-fleet reduction [%]",
          vmin=pk_min, vmax=pk_max)
    axes[0].set_ylabel(r"$P$ [€/p/d]", fontsize=10)
    heat(axes[1], piv_cv, "viridis",
          "(b) Coefficient of variation reduction [%]",
          vmin=0, vmax=cv_max)
    heat(axes[2], piv_total, "magma",
          "(c) Total weekly fleet reduction [%]",
          vmin=0, vmax=tot_max)

    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / "fig_B1_fleet_impact_vs_baseline.png",
                 bbox_inches="tight")
    fig.savefig(OUT / "fig_B1_fleet_impact_vs_baseline.pdf",
                 bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {OUT/'fig_B1_fleet_impact_vs_baseline.png'} (13.5\" wide)")
    sweet = cells[(np.isclose(cells.penalty, 0.5)) &
                   (np.isclose(cells.share_willing, 1.0))].iloc[0]
    print(f"\nSweet-spot P=0.5, theta=1.0:")
    print(f"  Peak reduction vs baseline:  {sweet.peak_red_pct:.1f} %")
    print(f"  CV reduction vs baseline:    {sweet.cv_red_pct:.1f} %")
    print(f"  Total fleet reduction:       {sweet.total_red_pct:.1f} %")


if __name__ == "__main__":
    main()
