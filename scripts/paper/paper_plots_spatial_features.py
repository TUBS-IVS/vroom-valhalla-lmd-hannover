"""Three spatial + feature analysis plots from the overnight outputs.

  fig07a_raumtyp_summary.{png,pdf}        — saving% & schedule-size mix per
                                              raumtyp_3 at operating point
  fig07b_plz_choropleth.{png,pdf}         — PLZ map: chosen schedule-size per
                                              PLZ at operating point
  fig08_feature_scatter.{png,pdf}         — 3 features × scatter (PLZ as dots,
                                              colored by chosen schedule_size,
                                              size by weekly_parcels)
  fig09_feature_importance_per_cell.{png,pdf}
                                          — heatmap: Spearman(feature, chosen
                                              schedule_size) for each of the
                                              55 (P, share) cells

Operating point: P=0.5, share_willing=1.0 unless otherwise stated.

Inputs:
  results/overnight_2026_05_27/tab_chosen_schedules.csv
  results/overnight_2026_05_27/tab_ml_grid.csv
  results/checkpoints/01_demand.pkl, 04_optim_prep.pkl
  data/geodata/cluster_raumtyp.csv
  data/geodata/regionclusters.gpkg  (PLZ polygons)

Status B (Task 19): 74_-legacy's tab_chosen_schedules.csv has no unsuffixed
schedule_size/dd_cost_eur columns (v5/v6 always carries two plans);
schedule_size_balanced/dd_cost_balanced (the operator-polished/final plan)
are aliased back to schedule_size/dd_cost_eur once at load, so every
downstream function is untouched. tab_per_plz_costs_theta1.csv is theta=1
only, so this port stays on tab_chosen_schedules.csv (all theta) instead,
per the inventory's own note that either source works for this script.
"""
from __future__ import annotations
import argparse
import pickle
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
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paper_v6_common as V6  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OVERNIGHT = ROOT / "results" / "overnight_2026_05_27"

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

FREQ_COLOR = {
    1: "#9d2226", 2: "#1d3557", 3: "#2a9d8f",
    4: "#e9c46a", 5: "#f4a261", 6: "#e76f51",
}
RAUMTYP_3_COLOR = {"urban": "#1d3557", "suburban": "#2a9d8f", "rural": "#f4a261"}
OPERATING_P = 0.5
OPERATING_SHARE = 1.0
SRC_NOTE = "tab_chosen_schedules.csv (historical path)"


def load_plz_features():
    """Build a per-(provider, PLZ) feature dataframe (PLZ-level)."""
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    provider_data = chk["provider_data"]
    optim_data = chk4["optimization_data"]
    rows = []
    for prov, odata in optim_data.items():
        for plz in odata["plz_keys"]:
            pdata = odata["plz_data"][plz]
            row_pd = provider_data[prov]["plz_demand"]
            r = row_pd[row_pd.plz == plz]
            if r.empty:
                continue
            r = r.iloc[0]
            weekly = int(r.weekly_parcels)
            stops = float(pdata["total_points"])
            area = float(pdata["area_km2"])
            rows.append({
                "provider": prov, "plz": plz,
                "weekly_parcels": weekly,
                "n_stops": stops,
                "area_km2": area,
                "hub_dist_km": float(pdata["hub_dist_km"]),
                "b2c_share": float(r.b2c_weekly / max(1, weekly)),
                "parcels_per_stop": weekly / max(1, stops),
                "parcels_per_km2": weekly / max(0.01, area),
            })
    return pd.DataFrame(rows)


def load_raumtyp():
    """Return raumtyp mapping for every PLZ AND every cluster_id.

    We need to handle two cases:
      * the `plz` field in chosen_schedules is a PLZ that is itself a
        cluster representative (most cases),
      * the `plz` field is a non-rep PLZ that needs cluster lookup.

    We build a single long table  cluster_id → raumtyp_3 / raumtyp_8_name
    by combining ``plz_raumtyp.csv`` (full PLZ-level) and
    ``cluster_raumtyp.csv`` (cluster-level) so every code in our scope
    finds a category.
    """
    plz_rt = pd.read_csv(ROOT / "data/geodata/plz_raumtyp.csv",
                          dtype={"plz": str})
    cl_rt = pd.read_csv(ROOT / "data/geodata/cluster_raumtyp.csv",
                         dtype={"cluster_id": str})
    cl_rt = cl_rt.rename(columns={"cluster_id": "plz"})
    rt = pd.concat([plz_rt, cl_rt], ignore_index=True)
    rt["plz"] = rt["plz"].astype(str).str.zfill(5)
    rt = rt.drop_duplicates(subset=["plz"], keep="first")
    rt = rt.rename(columns={"plz": "cluster_id"})
    return rt


def fig07a_raumtyp_summary(chosen, rt_df):
    """Saving% and schedule-size mix per raumtyp_3 at operating point."""
    sub = chosen[(np.isclose(chosen.penalty, OPERATING_P)) &
                 (np.isclose(chosen.share_willing, OPERATING_SHARE))].copy()
    sub["plz"] = sub.plz.astype(str).str.zfill(5)
    rt = rt_df[["cluster_id", "raumtyp_3", "raumtyp_8_name"]].copy()
    rt["cluster_id"] = rt.cluster_id.astype(str).str.zfill(5)
    # Also build cluster-membership lookup so non-rep PLZs get their cluster's
    # raumtyp via the cluster representative.
    cl = pd.read_csv(ROOT / "data/geodata/plz_clusters.csv",
                     dtype={"cluster_id": str})
    cl["members"] = cl["member_plz_list"].str.split(",")
    cl_long = (cl.explode("members")
                 .assign(member=lambda d: d["members"].astype(str).str.zfill(5))
                 [["cluster_id", "member"]])
    cl_long["cluster_id"] = cl_long["cluster_id"].str.zfill(5)
    cl_long = cl_long.merge(rt[["cluster_id", "raumtyp_3", "raumtyp_8_name"]],
                              on="cluster_id", how="left")
    cl_long = cl_long.rename(columns={"member": "plz"})

    sub = sub.merge(rt, left_on="plz", right_on="cluster_id",
                     how="left", suffixes=("", "_rt"))
    # For PLZs not directly mapped, look them up as cluster members
    miss = sub["raumtyp_3"].isna()
    if miss.any():
        fill = cl_long[["plz", "raumtyp_3", "raumtyp_8_name"]] \
            .rename(columns={"raumtyp_3": "raumtyp_3_cl",
                              "raumtyp_8_name": "raumtyp_8_name_cl"})
        sub = sub.merge(fill, on="plz", how="left")
        sub["raumtyp_3"] = sub["raumtyp_3"].fillna(sub["raumtyp_3_cl"])
        sub["raumtyp_8_name"] = sub["raumtyp_8_name"].fillna(sub["raumtyp_8_name_cl"])
    sub["raumtyp_3"] = sub["raumtyp_3"].fillna("unknown")
    print(f"    raumtyp_3 distribution: "
          f"{sub.raumtyp_3.value_counts().to_dict()}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # Panel A: avg cost per PLZ within each raumtyp (cost/parcel)
    sub["cost_per_parcel"] = sub.dd_cost_eur / sub.weekly_parcels.clip(lower=1)
    g = sub.groupby("raumtyp_3").agg(
        n_plz=("plz", "count"),
        mean_cost_per_parcel=("cost_per_parcel", "mean"),
        median_size=("schedule_size", "median"),
    )
    ax = axes[0]
    rt_order = ["urban", "suburban", "rural"]
    rt_order = [rt for rt in rt_order if rt in g.index]
    colors = [RAUMTYP_3_COLOR.get(r, "grey") for r in rt_order]
    ax.bar(rt_order, [g.loc[r, "mean_cost_per_parcel"] for r in rt_order],
           color=colors, edgecolor="black")
    for i, r in enumerate(rt_order):
        n = int(g.loc[r, "n_plz"])
        ax.text(i, g.loc[r, "mean_cost_per_parcel"] + 0.05,
                f"n={n}\nmedian {int(g.loc[r,'median_size'])} d/wk",
                ha="center", fontsize=10)
    ax.set_ylabel("Mean cost per parcel [€]")
    ax.set_xlabel("Region type (BBSR Raumtyp_3)")
    ax.set_title("Per-parcel cost by region type @ $P=0.5$, share=100%")
    ax.grid(axis="y", alpha=0.3)

    # Panel B: schedule-size mix per raumtyp (stacked bars)
    pivot = (sub.groupby(["raumtyp_3", "schedule_size"]).size().unstack(fill_value=0))
    pivot = pivot.reindex(rt_order)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    ax = axes[1]
    bottom = np.zeros(len(rt_order))
    for sz in sorted(pivot_pct.columns):
        if sz < 2:
            continue
        h = pivot_pct[sz].values
        ax.bar(rt_order, h, bottom=bottom, color=FREQ_COLOR.get(sz, "grey"),
               label=f"{sz} day/wk", edgecolor="white")
        bottom += h
    ax.set_ylabel("Delivery-frequency mix [%]")
    ax.set_xlabel("Region type")
    ax.set_title("Chosen schedule-size mix per region type")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Spatial heterogeneity: cost and schedule choice by Raumtyp",
                  fontsize=13, y=1.04)
    fig.tight_layout()
    V6.add_provenance_footer(fig, plan="operator-polished (balanced)",
                             script="paper_plots_spatial_features.py",
                             source=SRC_NOTE)
    V6.savefig_pair(fig, OVERNIGHT / "fig07a_raumtyp_summary.png", OVERNIGHT / "fig07a_raumtyp_summary.pdf")
    plt.close(fig)
    print("  fig07a_raumtyp_summary")


def fig07b_plz_choropleth(chosen, rt_df):
    """PLZ choropleth: chosen schedule-size at operating point."""
    try:
        import geopandas as gpd
    except ImportError:
        print("  fig07b skipped: geopandas not available")
        return
    # Use gdf_plz from the demand checkpoint (has `plz` column + polygons)
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    plz_gdf = chk["gdf_plz"].copy()
    plz_gdf["plz"] = plz_gdf["plz"].astype(str).str.zfill(5)
    # Merge with cluster mapping so that merged clusters share their saving
    cl = pd.read_csv(ROOT / "data/geodata/plz_clusters.csv",
                     dtype={"cluster_id": str})
    # Expand member_plz_list into rows
    cl["members"] = cl["member_plz_list"].str.split(",")
    cl_long = cl.explode("members").rename(columns={"members": "plz"})
    cl_long["plz"] = cl_long["plz"].astype(str).str.zfill(5)
    cl_long["cluster_id"] = cl_long["cluster_id"].astype(str)
    plz_gdf = plz_gdf.merge(cl_long[["plz", "cluster_id"]], on="plz", how="left")
    # PLZ not in cluster map use plz as cluster_id
    plz_gdf["cluster_id"] = plz_gdf["cluster_id"].fillna(plz_gdf["plz"])

    # Aggregate chosen schedule_size per PLZ across providers (use mean)
    sub = chosen[(np.isclose(chosen.penalty, OPERATING_P)) &
                 (np.isclose(chosen.share_willing, OPERATING_SHARE))].copy()
    sub["plz"] = sub.plz.astype(str)
    plz_agg = sub.groupby("plz", as_index=False).agg(
        median_size=("schedule_size", "median"),
        max_size=("schedule_size", "max"),
        n_providers=("provider", "count"),
    )
    merged = plz_gdf.merge(plz_agg, left_on="cluster_id", right_on="plz", how="left")

    fig, ax = plt.subplots(figsize=(11, 9))
    boundaries = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    cmap = ListedColormap([FREQ_COLOR[s] for s in (2, 3, 4, 5, 6)])
    norm = BoundaryNorm(boundaries, cmap.N)
    merged.plot(column="median_size", cmap=cmap, norm=norm,
                edgecolor="white", linewidth=0.3, ax=ax,
                missing_kwds={"color": "lightgrey", "label": "no data"})
    # Manual legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=FREQ_COLOR[s])
               for s in (2, 3, 4, 5, 6)]
    labels = ["2 day/wk", "3 day/wk", "4 day/wk", "5 day/wk", "6 day/wk"]
    ax.legend(handles, labels, title="Median chosen schedule (across LSPs)",
              loc="upper right")
    ax.set_axis_off()
    ax.set_title(f"PLZ-level chosen delivery frequency @ $P={OPERATING_P}$, "
                  f"share={int(OPERATING_SHARE*100)}%",
                  fontsize=12, pad=15)
    fig.tight_layout()
    V6.add_provenance_footer(fig, plan="operator-polished (balanced)",
                             script="paper_plots_spatial_features.py",
                             source=SRC_NOTE)
    V6.savefig_pair(fig, OVERNIGHT / "fig07b_plz_choropleth.png", OVERNIGHT / "fig07b_plz_choropleth.pdf")
    plt.close(fig)
    print("  fig07b_plz_choropleth")


def fig08_feature_scatter(chosen, plz_feats):
    """Scatter: (feature1, feature2) of each (provider, PLZ), color by chosen schedule_size."""
    sub = chosen[(np.isclose(chosen.penalty, OPERATING_P)) &
                 (np.isclose(chosen.share_willing, OPERATING_SHARE))].copy()
    sub["plz"] = sub.plz.astype(str)
    # Drop chosen-side weekly_parcels to avoid merge collision
    sub = sub.drop(columns=["weekly_parcels"], errors="ignore")
    sub = sub.merge(plz_feats, on=["provider", "plz"], how="left")

    # 3 scatter panels showcasing the strongest feature interactions
    pairs = [
        ("hub_dist_km", "parcels_per_stop", "Distance from hub vs Parcels per drop-site"),
        ("hub_dist_km", "weekly_parcels", "Distance from hub vs Weekly volume"),
        ("area_km2", "parcels_per_km2", "Service area vs Parcel density"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (fx, fy, title) in zip(axes, pairs):
        size_arr = np.sqrt(sub.weekly_parcels.clip(lower=1)) * 0.7
        sc = ax.scatter(sub[fx], sub[fy], c=sub["schedule_size"],
                        cmap=ListedColormap([FREQ_COLOR[s] for s in (2,3,4,5,6)]),
                        vmin=2, vmax=6, s=size_arr, alpha=0.75,
                        edgecolor="white", linewidth=0.4)
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
        if fy == "weekly_parcels":
            ax.set_yscale("log")
        if fx == "area_km2":
            ax.set_xscale("log")
        if fy == "parcels_per_km2":
            ax.set_yscale("log")
    # Colorbar (discrete)
    cbar = fig.colorbar(sc, ax=axes, ticks=[2, 3, 4, 5, 6],
                         orientation="vertical", pad=0.02, fraction=0.025)
    cbar.set_label("Chosen schedule size [days/wk]")
    fig.suptitle(f"How input features map to the chosen delivery frequency "
                  f"(operating point $P={OPERATING_P}$, share={int(OPERATING_SHARE*100)}%)",
                  fontsize=13, y=1.02)
    V6.add_provenance_footer(fig, plan="operator-polished (balanced)",
                             script="paper_plots_spatial_features.py",
                             source=SRC_NOTE)
    V6.savefig_pair(fig, OVERNIGHT / "fig08_feature_scatter.png", OVERNIGHT / "fig08_feature_scatter.pdf")
    plt.close(fig)
    print("  fig08_feature_scatter")


def fig09_feature_importance_per_cell(chosen, plz_feats):
    """Heatmap: Spearman(feature, chosen schedule_size) for each (P, share) cell."""
    chosen = chosen.copy()
    chosen["plz"] = chosen.plz.astype(str)
    chosen = chosen.drop(columns=["weekly_parcels"], errors="ignore")
    merged = chosen.merge(plz_feats, on=["provider", "plz"], how="left")

    feats = ["weekly_parcels", "n_stops", "area_km2", "hub_dist_km",
              "b2c_share", "parcels_per_stop", "parcels_per_km2"]
    cell_groups = merged.groupby(["penalty", "share_willing"])
    pen_values = sorted(merged.penalty.unique())
    share_values = sorted(merged.share_willing.unique())

    # 5 sub-heatmaps, one per penalty: rows=features, cols=share_willing
    fig, axes = plt.subplots(1, len(pen_values), figsize=(4.4 * len(pen_values), 4.2),
                              sharey=True)
    vmin, vmax = -1, 1

    for ax, P in zip(axes, pen_values):
        M = np.full((len(feats), len(share_values)), np.nan)
        for j, s in enumerate(share_values):
            g = merged[(np.isclose(merged.penalty, P)) &
                       (np.isclose(merged.share_willing, s))]
            if g.schedule_size.nunique() <= 1:
                continue
            for i, f in enumerate(feats):
                rho, _ = spearmanr(g[f].values, g["schedule_size"].values)
                M[i, j] = rho
        im = ax.imshow(M, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(share_values)))
        ax.set_xticklabels([f"{int(s*100)}%" for s in share_values],
                            rotation=45, ha="right")
        ax.set_title(f"$P={P:g}$ €/parcel/day")
        ax.set_xlabel("Share willing")
        for i in range(len(feats)):
            for j in range(len(share_values)):
                v = M[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.5 else "black", fontsize=7)
    axes[0].set_yticks(range(len(feats)))
    axes[0].set_yticklabels(feats)
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", pad=0.02,
                         fraction=0.025, shrink=0.85)
    cbar.set_label("Spearman ρ (feature ↔ chosen schedule_size)")
    fig.suptitle("Feature importance per operating point — "
                  "ρ between input feature and chosen delivery frequency",
                  fontsize=13, y=1.02)
    V6.add_provenance_footer(fig, plan="operator-polished (balanced)",
                             script="paper_plots_spatial_features.py",
                             source=SRC_NOTE)
    V6.savefig_pair(fig, OVERNIGHT / "fig09_feature_importance_per_cell.png", OVERNIGHT / "fig09_feature_importance_per_cell.pdf")
    plt.close(fig)
    print("  fig09_feature_importance_per_cell")


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
    global SRC_NOTE
    SRC_NOTE = ("B: 74_-legacy tab_chosen_schedules.csv "
               "(schedule_size_balanced/dd_cost_balanced)" if v6_mode
               else "tab_chosen_schedules.csv (historical path)")

    print("Loading data ...")
    chosen = pd.read_csv(in_dir / "tab_chosen_schedules.csv")
    if "schedule_size" not in chosen.columns:
        chosen = chosen.rename(columns={"schedule_size_balanced": "schedule_size",
                                        "dd_cost_balanced": "dd_cost_eur"})
    rt_df = load_raumtyp()
    plz_feats = load_plz_features()
    print(f"  chosen rows: {len(chosen)}, raumtyp rows: {len(rt_df)}, "
          f"feature rows: {len(plz_feats)}")

    print("\nGenerating spatial + feature plots:")
    fig07a_raumtyp_summary(chosen, rt_df)
    fig07b_plz_choropleth(chosen, rt_df)
    fig08_feature_scatter(chosen, plz_feats)
    fig09_feature_importance_per_cell(chosen, plz_feats)

    print(f"\nDone. Outputs in {OVERNIGHT}")


if __name__ == "__main__":
    main()
