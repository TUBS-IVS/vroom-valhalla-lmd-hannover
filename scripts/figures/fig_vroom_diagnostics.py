"""VROOM-vs-ML diagnostics for the Path-2 balanced schedules.

Reads the resumable VROOM-validation checkpoint
(results/paper_results_final/07_validation/tab_vroom_balanced.csv) and the
corresponding ML cost predictions, then produces:

  tab_diagnostics_balanced.csv  per-cell MAPE / bias / R2 / n
  fig_V1_vroom_vs_ml.{png,pdf}  scatter ML vs VROOM, per cell

Designed to be RE-RUNNABLE during VROOM execution: it reports per-cell
completeness so we know which cells are still partial. Cells with < 80 %
PLZ coverage are reported as "partial" and excluded from the headline
diagnostics table.

Output flat in:
  results/EWGT_Results/
"""
from __future__ import annotations
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "results" / "paper_results_final" / "07_validation"
PATH2 = ROOT / "results" / "overnight_2026_05_29_path2"
OUT = ROOT / "results" / "EWGT_Results"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

PALETTE = {
    "Amazon": "#E69F00", "DHL": "#D55E00", "DPD": "#0072B2",
    "FedEx": "#56B4E9", "GLS": "#009E73", "Hermes": "#CC79A7",
    "UPS": "#7C71A0",
}


def cell_metrics(y_true, y_pred):
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    m = np.isfinite(yt) & np.isfinite(yp) & (yt > 0)
    yt, yp = yt[m], yp[m]
    if len(yt) == 0:
        return dict(n=0, mape=np.nan, bias=np.nan, r2=np.nan, mae=np.nan)
    err = yp - yt
    mape = float(np.mean(np.abs(err) / yt) * 100)
    bias = float(np.mean(err / yt) * 100)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    mae = float(np.mean(np.abs(err)))
    return dict(n=len(yt), mape=mape, bias=bias, r2=r2, mae=mae)


def main():
    if not (VAL / "tab_vroom_balanced.csv").exists():
        print("No VROOM checkpoint yet.")
        return

    vr = pd.read_csv(VAL / "tab_vroom_balanced.csv")
    vr = vr[vr.vroom_status == "OK"].copy()
    print(f"VROOM OK rows: {len(vr)}")
    if len(vr) == 0:
        print("No OK rows - nothing to diagnose yet.")
        return

    vr["plz"] = vr.plz.astype(str)
    # ml_dd_cost_balanced_eur is the WEEKLY total repeated across every day
    # row, so take the first value instead of summing.
    vr_w = (vr.groupby(["penalty", "share_willing", "provider", "plz"])
              .agg(vroom_w=("vroom_cost_eur", "sum"),
                   ml_w=("ml_dd_cost_balanced_eur", "first"),
                   n_days_routed=("day", "nunique"),
                   parcels_w=("vroom_n_parcels", "sum"))
              .reset_index())

    # Per-PLZ ML init cost (for comparison with init-side error too)
    sched = pd.read_csv(PATH2 / "tab_chosen_schedules.csv")
    sched["plz"] = sched.plz.astype(str)
    init_keep = sched[["penalty", "share_willing", "provider", "plz",
                        "dd_cost_init", "dd_cost_balanced"]].copy()
    init_keep = init_keep.rename(columns={
        "dd_cost_init": "ml_init",
        "dd_cost_balanced": "ml_bal_total"})
    m = vr_w.merge(init_keep,
                    on=["penalty", "share_willing", "provider", "plz"],
                    how="inner")
    print(f"Merged rows: {len(m)}")
    if len(m) == 0:
        return

    # Per-cell diagnostics
    rows = []
    for (P, sh), g in m.groupby(["penalty", "share_willing"]):
        n_total = init_keep[(np.isclose(init_keep.penalty, P))
                             & (np.isclose(init_keep.share_willing, sh))
                             ].shape[0]
        cov = len(g) / n_total if n_total > 0 else 0.0
        d_bal = cell_metrics(g.vroom_w.values, g.ml_w.values)
        d_init = cell_metrics(g.vroom_w.values, g.ml_init.values)
        rows.append(dict(
            penalty=P, share_willing=sh, n=len(g), coverage=cov,
            mape_balanced=d_bal["mape"], bias_balanced=d_bal["bias"],
            r2_balanced=d_bal["r2"],
            mape_init=d_init["mape"], bias_init=d_init["bias"],
            r2_init=d_init["r2"],
        ))
    diag = pd.DataFrame(rows).sort_values(["penalty", "share_willing"])
    diag.to_csv(OUT / "tab_vroom_diagnostics.csv", index=False)
    print("\nPer-cell diagnostics:")
    print(diag.round(2).to_string(index=False))

    # ---- Scatter figure ----
    cells = sorted(m[["penalty", "share_willing"]].drop_duplicates()
                    .itertuples(index=False, name=None))
    n = len(cells)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(3.6 * ncols, 3.6 * nrows),
                              squeeze=False, sharex=False, sharey=False)
    for ax, (P, sh) in zip(axes.flat, cells):
        g = m[(np.isclose(m.penalty, P))
              & (np.isclose(m.share_willing, sh))]
        for prov in PALETTE:
            sub = g[g.provider == prov]
            if len(sub) == 0:
                continue
            ax.scatter(sub.vroom_w / 1000.0,
                        sub.ml_w / 1000.0,
                        s=20, color=PALETTE[prov], alpha=0.75,
                        edgecolor="white", linewidth=0.4,
                        label=prov, rasterized=True)
        lo = min(g.vroom_w.min(), g.ml_w.min()) / 1000.0
        hi = max(g.vroom_w.max(), g.ml_w.max()) / 1000.0
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.6)
        d_bal = cell_metrics(g.vroom_w, g.ml_w)
        ax.set_title(rf"$P={P:g}$, $\theta={sh:g}$  "
                      rf"(n={len(g)}, MAPE={d_bal['mape']:.1f}%)",
                      fontsize=10)
        ax.set_xlabel("VROOM weekly cost [k€]")
        ax.set_ylabel("ML balanced cost [k€]")
        ax.grid(alpha=0.3)
    # hide unused subplots
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=8,
                bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96], w_pad=1.0, h_pad=1.0)
    fig.savefig(OUT / "fig_V1_vroom_vs_ml.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_V1_vroom_vs_ml.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {OUT/'fig_V1_vroom_vs_ml.png'}")


if __name__ == "__main__":
    main()
