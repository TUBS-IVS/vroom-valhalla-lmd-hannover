"""Multi-panel regime heatmap of the LGB residual share of VROOM cost.

Generalises fig_H4 from one (parcels × area) heatmap to a 3×3 grid covering
all pairwise combinations of the four numeric drivers
  (weekly_parcels, n_stops, area_km2, hub_dist_km)
plus three categorical breakdowns
  (raumtyp_3, provider, schedule_size).

Single figure 'fig_lgb_residual_regime_grid' with consistent color scale so
panels are directly comparable.

Output (results/overnight_2026_05_27/diagnosis_v2/interpretation/):
  fig_lgb_residual_regime_grid.{png,pdf}
  tab_lgb_residual_regime_grid.csv     long-form table for the paper appendix
"""
from __future__ import annotations
import sys
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

BASE = ROOT / "results" / "overnight_2026_05_27"
OUT = BASE / "diagnosis_v2" / "interpretation"
OUT.mkdir(parents=True, exist_ok=True)


def load_raumtyp():
    plz_rt = pd.read_csv(ROOT / "data/geodata/plz_raumtyp.csv", dtype={"plz": str})
    cl_rt = pd.read_csv(ROOT / "data/geodata/cluster_raumtyp.csv", dtype={"cluster_id": str})
    cl_rt = cl_rt.rename(columns={"cluster_id": "plz"})
    rt = pd.concat([plz_rt, cl_rt], ignore_index=True)
    rt["plz"] = rt["plz"].astype(str).str.zfill(5)
    return rt.drop_duplicates(subset=["plz"], keep="first")[["plz", "raumtyp_3"]]


def load_data():
    res = pd.read_csv(OUT / "tab_residual_breakdown.csv")
    res["plz"] = res.plz.astype(str).str.zfill(5)
    res = res.drop(columns=[c for c in res.columns if c.startswith("raumtyp")],
                    errors="ignore")
    rt = load_raumtyp()
    df = res.merge(rt, on="plz", how="left")
    df["raumtyp_3"] = df.raumtyp_3.fillna("unknown")
    df["raumtyp_3"] = pd.Categorical(df.raumtyp_3,
                                      categories=["urban", "suburban", "rural"],
                                      ordered=True)

    # Numeric bins
    df["parcel_bin"] = pd.cut(df.weekly_parcels,
                              bins=[0, 1500, 3500, 7500, 200000],
                              labels=["<1.5k", "1.5–3.5k", "3.5–7.5k", "≥7.5k"])
    df["stops_bin"] = pd.cut(df.n_stops,
                             bins=[0, 80, 160, 280, 5000],
                             labels=["<80", "80–160", "160–280", "≥280"])
    df["area_bin"] = pd.cut(df.area_km2,
                            bins=[0, 5, 15, 40, 500],
                            labels=["<5 km²", "5–15 km²", "15–40 km²", "≥40 km²"])
    df["hubdist_bin"] = pd.cut(df.hub_dist_km,
                                bins=[0, 5, 10, 20, 100],
                                labels=["0–5 km", "5–10 km", "10–20 km", "≥20 km"])
    return df


def heatmap_panel(ax, df, xcol, ycol, xlabel, ylabel, vmin, vmax,
                   show_ylabel=True, show_xlabel=True):
    agg = df.groupby([ycol, xcol], observed=True).agg(
        residual_pct=("lgb_residual_pct", "mean"),
        n=("provider", "count"),
    ).reset_index()
    pivot_v = agg.pivot(index=ycol, columns=xcol, values="residual_pct")
    pivot_n = agg.pivot(index=ycol, columns=xcol, values="n")

    im = ax.imshow(pivot_v.values, aspect="auto", cmap="RdBu_r",
                    vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot_v.columns)))
    ax.set_xticklabels(pivot_v.columns, rotation=0, fontsize=8)
    ax.set_yticks(range(len(pivot_v.index)))
    ax.set_yticklabels(pivot_v.index, fontsize=8)
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    for i in range(pivot_v.shape[0]):
        for j in range(pivot_v.shape[1]):
            v = pivot_v.values[i, j]
            n = pivot_n.values[i, j] if not np.isnan(pivot_n.values[i, j]) else 0
            if not np.isnan(v):
                ax.text(j, i,
                          f"{v:+.0f}%\nn={int(n)}",
                          ha="center", va="center",
                          color="white" if abs(v) > vmax * 0.55 else "black",
                          fontsize=7.5)
    return im, pivot_v, pivot_n


def main():
    print("=" * 70)
    print("LGB residual regime grid (multi-panel)")
    print("=" * 70)
    df = load_data()
    print(f"  cells: {len(df)}")
    print(f"  raumtyp coverage: "
          f"urban={int((df.raumtyp_3 == 'urban').sum())}, "
          f"suburban={int((df.raumtyp_3 == 'suburban').sum())}, "
          f"rural={int((df.raumtyp_3 == 'rural').sum())}")

    # Common color scale
    vmax = 50.0
    vmin = -vmax

    panels = [
        # (xcol, ycol, xlabel, ylabel)
        ("area_bin",    "parcel_bin",  "Area [km²]",      "Weekly parcels"),
        ("hubdist_bin", "parcel_bin",  "Hub distance",    "Weekly parcels"),
        ("stops_bin",   "parcel_bin",  "Drop-sites",       "Weekly parcels"),
        ("area_bin",    "hubdist_bin", "Area [km²]",      "Hub distance"),
        ("stops_bin",   "hubdist_bin", "Drop-sites",       "Hub distance"),
        ("stops_bin",   "area_bin",    "Drop-sites",       "Area [km²]"),
        ("parcel_bin",  "raumtyp_3",   "Weekly parcels",   "Raumtyp"),
        ("hubdist_bin", "raumtyp_3",   "Hub distance",     "Raumtyp"),
        ("schedule_size", "provider",  "Schedule size [d/wk]", "Provider"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    im_last = None
    long_rows = []
    for k, (xcol, ycol, xlab, ylab) in enumerate(panels):
        ax = axes[k // 3, k % 3]
        try:
            im, pv, pn = heatmap_panel(ax, df, xcol, ycol, xlab, ylab,
                                        vmin, vmax,
                                        show_ylabel=(k % 3 == 0),
                                        show_xlabel=(k // 3 == 2))
            im_last = im
            for i, yi in enumerate(pv.index):
                for j, xi in enumerate(pv.columns):
                    v = pv.values[i, j]
                    n = pn.values[i, j]
                    if not np.isnan(v):
                        long_rows.append({
                            "y_feature": ycol, "x_feature": xcol,
                            "y_bin": str(yi), "x_bin": str(xi),
                            "mean_lgb_resid_pct": float(v),
                            "n_cells": int(n),
                        })
        except Exception as e:
            ax.text(0.5, 0.5, f"error: {e}", ha="center", va="center",
                     transform=ax.transAxes)
        ax.set_title(f"{ylab}  ×  {xlab}", fontsize=11, loc="left",
                      fontweight="normal")

    # Single shared colorbar
    cbar = fig.colorbar(im_last, ax=axes.ravel().tolist(),
                         shrink=0.7, pad=0.02, location="right")
    cbar.set_label("Mean LGB residual / VROOM cost  [%]", fontsize=11)

    fig.suptitle(
        "Where does the LGB residual correct Daganzo's physics, and by how much?\n"
        "Mean residual (% of VROOM cost) across binned feature pairs  •  "
        f"n = {len(df)} (provider × PLZ) cells  •  P = 0.5 €/parcel/day, share = 100%",
        fontsize=14, y=0.995)

    fig.savefig(OUT / "fig_lgb_residual_regime_grid.png")
    fig.savefig(OUT / "fig_lgb_residual_regime_grid.pdf")
    plt.close(fig)
    print(f"  -> fig_lgb_residual_regime_grid.{{png,pdf}}")

    pd.DataFrame(long_rows).to_csv(
        OUT / "tab_lgb_residual_regime_grid.csv", index=False)
    print(f"  -> tab_lgb_residual_regime_grid.csv  ({len(long_rows)} rows)")

    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
