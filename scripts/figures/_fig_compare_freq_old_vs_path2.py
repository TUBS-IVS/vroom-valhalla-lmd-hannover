"""Compare mean init schedule_size as a function of (P, theta) between the old
penalty-blind buggy run (per-PLZ argmin on unbundled total_cost_mx) and the
new Path-2 run (optimize_cd_ml on hub-bundled + penalty). Only cells already
completed in the new run are shown for the new heatmap (rest left NaN).

Output: results/overnight_2026_05_29_path2/_fig_freq_compare_old_vs_path2.png
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
OLD = ROOT / "results" / "overnight_2026_05_27_balanced" / "_prebugfix_penaltyblind_2026_05_29"
NEW = ROOT / "results" / "overnight_2026_05_29_path2"
OUT = NEW

rcParams.update({
    "font.family": "serif", "font.size": 9,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "savefig.bbox": "tight", "savefig.dpi": 200, "pdf.fonttype": 42,
})


def mean_freq(df, col):
    return (df.groupby(["penalty", "share_willing"])[col]
              .mean()
              .reset_index()
              .pivot(index="penalty", columns="share_willing", values=col))


def main():
    old = pd.read_csv(OLD / "tab_chosen_schedules.csv")
    new = pd.read_csv(NEW / "tab_chosen_schedules.csv")
    piv_old = mean_freq(old, "schedule_size_init")
    piv_new = mean_freq(new, "schedule_size_init")

    # Align rows / cols: union over both
    rows = sorted(set(piv_old.index) | set(piv_new.index))
    cols = sorted(set(piv_old.columns) | set(piv_new.columns))
    piv_old = piv_old.reindex(index=rows, columns=cols)
    piv_new = piv_new.reindex(index=rows, columns=cols)
    diff = piv_new - piv_old

    vmin = min(np.nanmin(piv_old.values), np.nanmin(piv_new.values))
    vmax = max(np.nanmax(piv_old.values), np.nanmax(piv_new.values))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))

    def heat(ax, mat, cmap, vmin_, vmax_, title, fmt="{:.2f}"):
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap,
                        vmin=vmin_, vmax=vmax_)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f"{x*100:.0f}" for x in mat.columns])
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"{p:g}" for p in mat.index])
        ax.set_xlabel(r"Willingness-to-wait share $\theta$ [%]")
        ax.set_title(title, fontsize=8.5)
        thr = vmin_ + (vmax_ - vmin_) * 0.55
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center", color="0.5", fontsize=5)
                    continue
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color="white" if v < thr else "black", fontsize=4.5)
        return im

    im_a = heat(axes[0], piv_old, "viridis_r", vmin, vmax,
                "(a) OLD: per-PLZ argmin, ungebündelt (Buggy-Lauf)")
    im_b = heat(axes[1], piv_new, "viridis_r", vmin, vmax,
                "(b) NEU: CD_ML, gebündelt+Penalty (Pfad 2)")
    d_abs = max(abs(np.nanmin(diff.values)), abs(np.nanmax(diff.values)))
    im_c = heat(axes[2], diff, "RdBu_r", -d_abs, d_abs,
                "(c) DIFF (NEU − ALT), blau = Pfad 2 senkt Freq",
                fmt="{:+.2f}")
    axes[0].set_ylabel(r"Service penalty $P$ [€/p/d]")

    cb1 = fig.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.03)
    cb1.set_label("Mean delivery freq [d/wk]", fontsize=8)
    cb1.ax.tick_params(labelsize=6)
    cb2 = fig.colorbar(im_c, ax=axes[2], fraction=0.046, pad=0.03)
    cb2.set_label(r"$\Delta$ freq (Path 2 − old)", fontsize=8)
    cb2.ax.tick_params(labelsize=6)

    fig.tight_layout(w_pad=0.6)
    out_p = OUT / "_fig_freq_compare_old_vs_path2.png"
    fig.savefig(out_p, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {out_p}")
    # quick text summary
    print("\nSummary stats:")
    print(f"  OLD mean freq range: {np.nanmin(piv_old.values):.2f} -> {np.nanmax(piv_old.values):.2f}")
    print(f"  NEW mean freq range: {np.nanmin(piv_new.values):.2f} -> {np.nanmax(piv_new.values):.2f}")
    print(f"  Cells where NEW < OLD (Pfad 2 senkt Frequenz): "
          f"{int(np.nansum(diff.values < -0.05))} / {int(np.sum(~np.isnan(diff.values)))}")
    print(f"  Cells where NEW > OLD: "
          f"{int(np.nansum(diff.values > 0.05))} / {int(np.sum(~np.isnan(diff.values)))}")
    print(f"  Mean |diff|: {np.nanmean(np.abs(diff.values)):.3f}")


if __name__ == "__main__":
    main()
