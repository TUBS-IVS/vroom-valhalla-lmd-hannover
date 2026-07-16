"""Six-panel combined heatmap over the $(P, \\theta)$ grid, Stage-3 (system-
smoothed) revision of scripts/figures/fig_combined_heatmap.py.

Copied from the frozen submission figure with these substitutions:
  - Panel (a): UNCHANGED -- cost-optimal selection, Stage 1
    (init_cost_eur, RUN_DIR/tab_balancing_summary.csv). Intentional: this
    panel documents the pre-balancing selection and is not affected by the
    Stage-3 recomputation.
  - Panel (b): saving from results/revision_2026_07/tab_costs_smoothed.csv
    (`bal` <- per-cell sum of `total_stage3_eur`), instead of the Stage-2
    `balanced_cost_eur` (tab_balancing_summary.csv).
  - Panel (c): wait from tab_wait_smoothed.csv (`avg_wait_d_stage3`),
    already parcels-weighted per (P, theta) -- no re-aggregation needed.
  - Panels (d,e,f): fleet from tab_fleet_per_hub_smoothed.csv, with
    `fa` <- `fleet_stage3` (Stage 3, after system smoothing) and the
    baseline (theta=0, mean over P) taken from `fleet_stage2` (Stage 2,
    per-hub balanced -- gate-identical to the submission's `fleet_before`
    mean at theta=0, since nothing is optimized when nobody is willing to
    wait). The baseline CV is asserted to reproduce the submission's 0.135.

Output: results/revision_2026_07/figures/fig5_grid_heatmap_6_smoothed.{png,pdf}
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _stage3_common as C  # noqa: E402

PATH2 = C.RUN_DIR
OUT = C.OUT_DIR / "figures"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 13,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 13, "axes.titlesize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

BASE_TOTAL = C.BASE_TOTAL  # daily-delivery weekly cost across all providers
SUFFIX = "(after per-hub balancing and system smoothing)"


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
    # ---- Panel (a): unchanged, Stage 1 cost-optimal selection ----
    s = pd.read_csv(PATH2 / "tab_balancing_summary.csv")
    s = s[~np.isclose(s.penalty, 0.4)].copy()
    ag = s.groupby(["penalty", "share_willing"], as_index=False).agg(
        init=("init_cost_eur", "sum"))
    ag["init_sav"] = 100 * (BASE_TOTAL - ag.init) / BASE_TOTAL
    pivI = ag.pivot(index="penalty", columns="share_willing",
                     values="init_sav")

    # ---- Panel (b): Stage-3 cost saving ----
    costs = pd.read_csv(C.OUT_DIR / "tab_costs_smoothed.csv")
    costs = costs[~np.isclose(costs.penalty, 0.4)].copy()
    agb = costs.groupby(["penalty", "share_willing"], as_index=False).agg(
        bal=("total_stage3_eur", "sum"))
    agb["bal_sav"] = 100 * (BASE_TOTAL - agb.bal) / BASE_TOTAL
    pivB = agb.pivot(index="penalty", columns="share_willing",
                      values="bal_sav")

    # ---- Panel (c): Stage-3 wait, already parcels-weighted per cell ----
    w = pd.read_csv(C.OUT_DIR / "tab_wait_smoothed.csv")
    w = w[~np.isclose(w.penalty, 0.4)].copy()
    pivW = w.pivot(index="penalty", columns="share_willing",
                   values="avg_wait_d_stage3")

    # ---- Panels (d,e,f): Stage-3 fleet vs Stage-2 baseline ----
    f = pd.read_csv(C.OUT_DIR / "tab_fleet_per_hub_smoothed.csv")
    f = f[~np.isclose(f.penalty, 0.4)].copy()
    sys_day = (f.groupby(["penalty", "share_willing", "day"])
                .agg(fb=("fleet_stage2", "sum"),
                     fa=("fleet_stage3", "sum")).reset_index())
    base_day = (sys_day[np.isclose(sys_day.share_willing, 0.0)]
                  .groupby("day").fb.mean())
    base = np.array([base_day.loc[d] for d in range(C.N_DAYS)])
    base_peak = float(base.max())
    base_total = float(base.sum())
    base_cv = float(base.std() / base.mean())
    print(f"Baseline Mo-Sa (Stage 2 @ theta=0, mean over P): peak={base_peak:.0f} "
          f"total={base_total:.0f} cv={base_cv:.3f}")
    assert abs(round(base_cv, 3) - 0.135) < 1e-9, (
        f"baseline CV {base_cv:.3f} != submission value 0.135 -- "
        "Stage-2 fleet baseline drifted, investigate before trusting fig5")

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

    # Ledger check: print panel (e) CV-reduction values at the three
    # paper-text reference cells.
    print("\nPanel (e) Mo-Sa CV reduction [%] at reference cells:")
    for P, sh in [(0.5, 1.0), (0.0, 1.0), (0.0, 0.8)]:
        v = pivCV.loc[P, sh] if (P in pivCV.index and sh in pivCV.columns) else float("nan")
        print(f"  (P={P}, theta={sh}): {v:.1f}%")
    print(f"  grid-max cell: {pivCV.stack().idxmax()} = {pivCV.stack().max():.1f}%")

    # ---- figure ----
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.0))

    v_cost_max = float(np.ceil(max(pivI.values.max(),
                                     pivB.values.max()) / 5) * 5)
    heat(axes[0, 0], pivI, "viridis",
          "(a) Cost saving, cost-optimal selection\n"
          "(before fleet balancing)",
          vmin=0, vmax=v_cost_max, cbar_label="Saving [%]")
    heat(axes[0, 1], pivB, "viridis",
          f"(b) Cost saving, full pipeline\n{SUFFIX}",
          vmin=0, vmax=v_cost_max, cbar_label="Saving [%]")
    w_max = float(np.ceil(pivW.values.max() * 10) / 10)
    heat(axes[0, 2], pivW, "YlOrRd",
          f"(c) Mean additional customer wait per parcel\n{SUFFIX}",
          vmin=0, vmax=w_max, fmt="{:.2f}", cbar_label="Wait [d]",
          invert_thr=True)

    pk_max = float(np.ceil(pivPK.values.max() / 5) * 5)
    pk_min = float(np.floor(min(0, pivPK.values.min()) / 5) * 5)
    heat(axes[1, 0], pivPK, "magma",
          f"(d) Peak-fleet reduction\n{SUFFIX}",
          vmin=pk_min, vmax=pk_max, cbar_label="Reduction [%]")
    cv_max = float(np.ceil(pivCV.values.max() / 10) * 10)
    heat(axes[1, 1], pivCV, "magma",
          f"(e) Mo--Sa coefficient of variation reduction\n{SUFFIX}",
          vmin=0, vmax=cv_max, cbar_label="Reduction [%]")
    pivCHG = -pivTOT
    tot_abs = max(abs(pivCHG.values.min()), abs(pivCHG.values.max()))
    tot_abs = float(np.ceil(tot_abs))
    norm_tot = TwoSlopeNorm(vmin=-tot_abs, vcenter=0.0, vmax=tot_abs)
    heat(axes[1, 2], pivCHG, "RdBu_r",
          f"(f) Total weekly fleet change [%]\n{SUFFIX}",
          norm=norm_tot, cbar_label="Change [%]")

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
        fig.savefig(OUT / f"fig5_grid_heatmap_6_smoothed.{ext}",
                     bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'fig5_grid_heatmap_6_smoothed.png'}")


if __name__ == "__main__":
    main()
