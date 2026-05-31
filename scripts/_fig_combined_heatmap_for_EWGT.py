"""Six-panel combined heatmap over the $(P, \\theta)$ grid for the EWGT
paper. Top row holds the cost-side metrics (saving of the cost-optimal
selection, saving of the fleet-balanced pipeline output, and the
parcels-weighted mean customer waiting time). Bottom row holds the
fleet-side metrics (peak fleet reduction, weekly CV reduction, and total
weekly fleet reduction). All saving and reduction columns are reported
relative to the daily-delivery baseline at $\\theta = 0$; the total-fleet
panel uses a diverging colour map centred at zero because some operating
points raise the weekly vehicle stock instead of lowering it.

Same TRPro styling as fig_SM_mix_pct_8P (serif, dejavuserif mathtext,
size 11, ticks 9, line width and savefig.dpi=300).

Output: results/EWGT_Results/fig_grid_heatmap_6.{png,pdf}
"""
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
PATH2 = ROOT / "results" / "overnight_2026_05_29_path2"
OUT = ROOT / "results" / "EWGT_Results"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 13,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 13, "axes.titlesize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

BASE_TOTAL = 1909747.75  # daily-delivery weekly cost across all providers


def heat(ax, mat, cmap, title, vmin=None, vmax=None, fmt="{:.1f}",
          norm=None, cbar_label="[%]", text_color=None,
          invert_thr=False):
    if norm is None:
        if vmin is None:
            vmin = float(np.nanmin(mat.values))
        if vmax is None:
            vmax = float(np.nanmax(mat.values))
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap,
                        vmin=vmin, vmax=vmax)
        thr = vmin + (vmax - vmin) * (0.55 if not invert_thr else 0.65)
        if text_color is not None:
            def color_for(v):
                return text_color
        elif invert_thr:
            # light-low, dark-high colormap (e.g. YlOrRd):
            # white text on dark high values, black text on light low values
            def color_for(v):
                return "white" if v > thr else "black"
        else:
            def color_for(v):
                return "white" if v < thr else "black"
    else:
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap, norm=norm)
        thr_lo = norm.vmin + (norm.vmax - norm.vmin) * 0.30
        thr_hi = norm.vmin + (norm.vmax - norm.vmin) * 0.70
        if text_color is not None:
            def color_for(v):
                return text_color
        else:
            def color_for(v):
                return "white" if (v < thr_lo or v > thr_hi) else "black"
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([f"{x*100:.0f}" for x in mat.columns], fontsize=10)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([f"{p:g}" for p in mat.index], fontsize=10)
    ax.set_title(title, fontsize=12.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color=color_for(v), fontsize=11)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=11)
    cb.ax.tick_params(labelsize=10)
    return im


def main():
    s = pd.read_csv(PATH2 / "tab_balancing_summary.csv")
    s = s[~np.isclose(s.penalty, 0.4)].copy()
    f = pd.read_csv(PATH2 / "tab_fleet_per_hub.csv")
    f = f[~np.isclose(f.penalty, 0.4)].copy()
    sched = pd.read_csv(PATH2 / "tab_chosen_schedules.csv")
    sched["plz"] = sched.plz.astype(str)
    sched = sched[~np.isclose(sched.penalty, 0.4)].copy()

    # Aggregate costs per (P, theta)
    ag = s.groupby(["penalty", "share_willing"], as_index=False).agg(
        init=("init_cost_eur", "sum"),
        bal=("balanced_cost_eur", "sum"))
    ag["init_sav"] = 100 * (BASE_TOTAL - ag.init) / BASE_TOTAL
    ag["bal_sav"]  = 100 * (BASE_TOTAL - ag.bal)  / BASE_TOTAL
    pivI = ag.pivot(index="penalty", columns="share_willing",
                     values="init_sav")
    pivB = ag.pivot(index="penalty", columns="share_willing",
                     values="bal_sav")

    # Avg waiting time, parcels-weighted across cells per (P, theta)
    sched["wait_x_par"] = sched.avg_wait_d_balanced * sched.weekly_parcels
    wg = sched.groupby(["penalty", "share_willing"], as_index=False).agg(
        num=("wait_x_par", "sum"),
        den=("weekly_parcels", "sum"))
    wg["avg_wait_d"] = wg.num / wg.den
    pivW = wg.pivot(index="penalty", columns="share_willing",
                      values="avg_wait_d")

    # System fleet per day
    sys_day = (f.groupby(["penalty", "share_willing", "day"])
                .agg(fb=("fleet_before", "sum"),
                     fa=("fleet_after", "sum")).reset_index())
    # Baseline at theta=0 averaged over P
    base_day = (sys_day[np.isclose(sys_day.share_willing, 0.0)]
                  .groupby("day").fb.mean())
    base = np.array([base_day.loc[d] for d in range(6)])
    base_peak = float(base.max())
    base_total = float(base.sum())
    base_cv = float(base.std() / base.mean())
    print(f"Baseline Mo-Sa: peak={base_peak:.0f} total={base_total:.0f} "
          f"cv={base_cv:.3f}")

    rows = []
    for (P, sh), g in sys_day.groupby(["penalty", "share_willing"]):
        g = g.sort_values("day")
        fa = g.fa.values
        peak_a = float(fa.max()) if len(fa) else float("nan")
        total_a = float(fa.sum()) if len(fa) else float("nan")
        cv_a = float(fa.std() / fa.mean()) if fa.mean() > 0 else 0.0
        rows.append(dict(penalty=P, share_willing=sh,
                          peak_red=100 * (base_peak - peak_a) / base_peak,
                          cv_red=100 * (base_cv - cv_a) / base_cv,
                          total_red=100 * (base_total - total_a)
                                     / base_total))
    cells = pd.DataFrame(rows)
    pivPK = cells.pivot(index="penalty", columns="share_willing",
                          values="peak_red")
    pivCV = cells.pivot(index="penalty", columns="share_willing",
                          values="cv_red")
    pivTOT = cells.pivot(index="penalty", columns="share_willing",
                           values="total_red")

    # ---- figure ----
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.0))

    # Top row
    v_cost_max = float(np.ceil(max(pivI.values.max(),
                                     pivB.values.max()) / 5) * 5)
    heat(axes[0, 0], pivI, "viridis",
          "(a) Cost saving, cost-optimal selection\n"
          "(before fleet balancing)",
          vmin=0, vmax=v_cost_max, cbar_label="Saving [%]")
    heat(axes[0, 1], pivB, "viridis",
          "(b) Cost saving, full pipeline\n"
          "(after fleet balancing and smoothing)",
          vmin=0, vmax=v_cost_max, cbar_label="Saving [%]")
    w_max = float(np.ceil(pivW.values.max() * 10) / 10)
    heat(axes[0, 2], pivW, "YlOrRd",
          "(c) Mean additional customer wait per parcel\n"
          "(full pipeline output)",
          vmin=0, vmax=w_max, fmt="{:.2f}", cbar_label="Wait [d]",
          invert_thr=True)

    # Bottom row
    pk_max = float(np.ceil(pivPK.values.max() / 5) * 5)
    pk_min = float(np.floor(min(0, pivPK.values.min()) / 5) * 5)
    heat(axes[1, 0], pivPK, "magma",
          "(d) Peak-fleet reduction\n"
          "(full pipeline output)",
          vmin=pk_min, vmax=pk_max, cbar_label="Reduction [%]")
    cv_max = float(np.ceil(pivCV.values.max() / 10) * 10)
    heat(axes[1, 1], pivCV, "magma",
          "(e) Mo--Sa coefficient of variation reduction\n"
          "(full pipeline output)",
          vmin=0, vmax=cv_max, cbar_label="Reduction [%]")
    # Total fleet: diverging on signed change (positive = more fleet,
    # negative = less fleet). Red = increase (bad), blue = reduction (good).
    pivCHG = -pivTOT  # flip sign so positive = increase
    tot_abs = max(abs(pivCHG.values.min()), abs(pivCHG.values.max()))
    tot_abs = float(np.ceil(tot_abs))
    norm_tot = TwoSlopeNorm(vmin=-tot_abs, vcenter=0.0, vmax=tot_abs)
    heat(axes[1, 2], pivCHG, "RdBu_r",
          "(f) Total weekly fleet change [%]\n"
          "(full pipeline output)",
          norm=norm_tot, cbar_label="Change [%]")

    # outer axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Willingness-to-wait share $\theta$ [%]",
                       fontsize=12.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Service penalty $P$ [€/p/d]", fontsize=12.5)

    fig.tight_layout(pad=0.6, w_pad=1.4, h_pad=1.8,
                      rect=[0, 0.03, 1, 1])
    fig.text(0.5, 0.01,
             "All cost savings, fleet reductions and the wait metric are "
             r"reported relative to the daily-delivery baseline at $\theta=0$.",
             ha="center", va="bottom", fontsize=11)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_grid_heatmap_6.{ext}",
                     bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'fig_grid_heatmap_6.png'}")


if __name__ == "__main__":
    main()
