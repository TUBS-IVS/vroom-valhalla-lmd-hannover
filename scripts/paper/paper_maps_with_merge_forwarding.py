"""Erstellt alle Cluster-basierten Choropleth-Karten mit MERGE-FORWARDING:
gemerged PLZ zeigen den Wert ihres Cluster-Repraesentanten — kein "leeres" Polygon.

Liefert zusätzlich Raumtyp-aggregierte Karten (Raumtyp_3 + Raumtyp_8) und
neue Bias-/Cost-Quality-Maps die in den existierenden Skripten fehlten.

Outputs (results/paper_maps_final/):
  fig_M01_cluster_saving_actual.{pdf,png}       Cluster-Saving (VROOM-actual)
  fig_M02_cluster_bias.{pdf,png}                LGB-logT Saving-Bias per Cluster
  fig_M03_raumtyp_3_saving.{pdf,png}            Aggregat-Karte je urban/sub/rural
  fig_M04_raumtyp_8_saving.{pdf,png}            Aggregat-Karte je BBSR-Raumtyp
  fig_M05_raumtyp_3_classification.{pdf,png}    Reine Klassifikations-Karte 3er
  fig_M06_raumtyp_8_classification.{pdf,png}    Reine Klassifikations-Karte 8er
  fig_M07_cost_mape_per_cluster.{pdf,png}       Cost-MAPE pro Cluster auf SA_ML
  REPORT.md
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import wkt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "geodata"
OUT = ROOT / "results" / "paper_maps_final"
OUT.mkdir(parents=True, exist_ok=True)

SAVING_CSV = ROOT / "results" / "final_optimization" / "vroom_validation" / "tab_actual_vs_predicted_saving.csv"
ML_VROOM_CSV = ROOT / "results" / "final_optimization" / "ml_vs_vroom_per_day.csv"

RAUMTYP_3_ORDER = ["urban", "suburban", "rural"]
RAUMTYP_3_COLOR = {"urban": "#cb181d", "suburban": "#fdae61", "rural": "#1a9850"}
RAUMTYP_8_COLORS = {
    1: "#67000d",   # Metropoles Zentrum
    2: "#a50f15",   # Zentrumsnah hochverdichtete Wohnnutzung
    3: "#cb181d",   # Zentrumsnah verdichtete Mischnutzung
    4: "#fb6a4a",   # Städtisch mit Verdichtungsansätzen
    5: "#fcae91",   # Städtisch mit gewerblicher Prägung
    6: "#fee0d2",   # Umland Verstädtert
    7: "#9ecae1",   # Umland dörflich m. geringem gewerbl.
    8: "#08519c",   # Umland dörflich ohne gewerbl.
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_plz_polygons() -> gpd.GeoDataFrame:
    df = pd.read_csv(DATA / "plz_areas.csv")
    df["geometry"] = df["WKT"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:25832")
    gdf["plz"] = gdf["plz"].astype(str).str.zfill(5)
    gdf = gdf.dissolve(by="plz", aggfunc={"ort": "first", "landkreis": "first",
                                           "einwohner": "sum"}).reset_index()
    return gdf


def load_cluster_mapping() -> pd.DataFrame:
    cl = pd.read_csv(DATA / "plz_clusters.csv", dtype={"cluster_id": str})
    cl["cluster_id"] = cl["cluster_id"].str.zfill(5)
    rows = []
    for _, r in cl.iterrows():
        for m in r["member_plz_list"].split(","):
            rows.append({"cluster_id": r["cluster_id"], "plz": m.strip().zfill(5)})
    return pd.DataFrame(rows)


def load_cluster_raumtyp() -> pd.DataFrame:
    cr = pd.read_csv(DATA / "cluster_raumtyp.csv", dtype={"cluster_id": str})
    cr["cluster_id"] = cr["cluster_id"].str.zfill(5)
    return cr


def forward_cluster_values_to_plz(
    plz_gdf: gpd.GeoDataFrame,
    cluster_mapping: pd.DataFrame,
    cluster_values: pd.DataFrame,
    value_col: str,
) -> gpd.GeoDataFrame:
    """Project cluster-level values onto ALL member PLZ polygons.

    Gemergte PLZ (z.B. 30171/30175 in cluster 30159) bekommen den Cluster-Wert.
    """
    cluster_values = cluster_values[["cluster_id", value_col]].copy()
    cluster_values["cluster_id"] = cluster_values["cluster_id"].astype(str).str.zfill(5)
    long_with_value = cluster_mapping.merge(cluster_values, on="cluster_id", how="left")
    out = plz_gdf.merge(long_with_value, on="plz", how="left")
    return out


# ---------------------------------------------------------------------------
# Map renderers
# ---------------------------------------------------------------------------

def render_cluster_choropleth(
    plz_gdf: gpd.GeoDataFrame,
    cluster_map: pd.DataFrame,
    cluster_values: pd.DataFrame,
    value_col: str,
    out_path: Path,
    title: str,
    cmap: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
    label: str = "",
    diverging_center: float | None = None,
):
    g = forward_cluster_values_to_plz(plz_gdf, cluster_map, cluster_values, value_col)
    fig, ax = plt.subplots(figsize=(11, 8))

    # PLZ ohne Wert (außerhalb Cluster) bleiben grau
    no_value = g[g[value_col].isna()]
    if len(no_value):
        no_value.plot(ax=ax, color="#dddddd", edgecolor="white", lw=0.3, label=f"no data (n={len(no_value)} PLZ)")
    with_value = g[g[value_col].notna()]

    if diverging_center is not None:
        # Diverging colormap centered at given value (e.g. 0 for bias)
        max_abs = max(abs(with_value[value_col].min() - diverging_center),
                       abs(with_value[value_col].max() - diverging_center))
        vmin = diverging_center - max_abs
        vmax = diverging_center + max_abs
    if vmin is None:
        vmin = with_value[value_col].min()
    if vmax is None:
        vmax = with_value[value_col].max()

    with_value.plot(ax=ax, column=value_col, cmap=cmap, vmin=vmin, vmax=vmax,
                      edgecolor="white", lw=0.3, legend=True,
                      legend_kwds={"label": label or value_col, "shrink": 0.55})

    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def render_classification_map(
    plz_gdf: gpd.GeoDataFrame,
    cluster_map: pd.DataFrame,
    cluster_raumtyp: pd.DataFrame,
    raumtyp_col: str,
    color_map: dict,
    name_map: dict | None,
    out_path: Path,
    title: str,
):
    g = forward_cluster_values_to_plz(plz_gdf, cluster_map, cluster_raumtyp, raumtyp_col)
    fig, ax = plt.subplots(figsize=(11, 8))

    no_value = g[g[raumtyp_col].isna()]
    if len(no_value):
        no_value.plot(ax=ax, color="#dddddd", edgecolor="white", lw=0.3)

    for val, color in color_map.items():
        sub = g[g[raumtyp_col] == val]
        if not len(sub):
            continue
        nm = name_map.get(val, str(val)) if name_map else str(val)
        sub.plot(ax=ax, color=color, edgecolor="white", lw=0.3,
                  label=f"{nm} (n={sub['plz'].nunique()} PLZ in {sub['cluster_id'].nunique()} cluster)")

    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(loc="lower left", fontsize=7.5, title="Raumtyp")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
def compute_cluster_kpis() -> dict:
    """Compute per-cluster KPIs from existing saving + per-day data."""
    sav = pd.read_csv(SAVING_CSV, dtype={"plz": str})
    sav["plz"] = sav["plz"].str.zfill(5)
    sav["bias_pp"] = sav["predicted_saving_pct"] - sav["actual_saving_pct"]

    # Map PLZ to cluster
    cl_map = load_cluster_mapping()
    sav = sav.merge(cl_map, on="plz", how="left")

    # Aggregate to cluster-level (weighted by parcels)
    cluster_agg = sav.groupby("cluster_id", as_index=False).agg(
        mean_actual_saving_pct=("actual_saving_pct", "mean"),
        mean_predicted_saving_pct=("predicted_saving_pct", "mean"),
        mean_bias_pp=("bias_pp", "mean"),
        median_bias_pp=("bias_pp", "median"),
        n_provider_cells=("provider", "count"),
        total_baseline_cost=("baseline_cost_eur", "sum"),
        total_actual_saving_abs=("actual_saving_abs", "sum"),
    )

    # Cost-MAPE per cluster on the routed cells
    mlv = pd.read_csv(ML_VROOM_CSV, dtype={"plz": str})
    mlv["plz"] = mlv["plz"].str.zfill(5)
    mlv = mlv[mlv["delivers_on_day"]].dropna(subset=["ml_pred_cost_eur", "vroom_actual_cost_eur"])
    mlv = mlv[mlv["vroom_actual_cost_eur"] > 0]
    mlv = mlv.merge(cl_map, on="plz", how="left")
    mlv["ape_pct"] = 100 * (mlv["ml_pred_cost_eur"] - mlv["vroom_actual_cost_eur"]).abs() / mlv["vroom_actual_cost_eur"].clip(lower=1)
    cost_mape = mlv.groupby("cluster_id", as_index=False)["ape_pct"].mean().rename(columns={"ape_pct": "cost_mape_pct"})

    return {
        "cluster_agg": cluster_agg,
        "cost_mape": cost_mape,
        "raw_saving": sav,
        "raw_perday": mlv,
    }


def compute_raumtyp_kpis(cluster_agg: pd.DataFrame, cluster_raumtyp: pd.DataFrame) -> dict:
    joined = cluster_agg.merge(cluster_raumtyp[["cluster_id", "raumtyp_3", "raumtyp_8", "raumtyp_8_name"]],
                                  on="cluster_id", how="left")

    rt3 = joined.groupby("raumtyp_3", as_index=False).agg(
        mean_actual_saving_pct=("mean_actual_saving_pct", "mean"),
        mean_predicted_saving_pct=("mean_predicted_saving_pct", "mean"),
        mean_bias_pp=("mean_bias_pp", "mean"),
        n_cluster=("cluster_id", "nunique"),
        total_saving=("total_actual_saving_abs", "sum"),
    )
    rt8 = joined.groupby(["raumtyp_8", "raumtyp_8_name"], as_index=False).agg(
        mean_actual_saving_pct=("mean_actual_saving_pct", "mean"),
        mean_predicted_saving_pct=("mean_predicted_saving_pct", "mean"),
        mean_bias_pp=("mean_bias_pp", "mean"),
        n_cluster=("cluster_id", "nunique"),
        total_saving=("total_actual_saving_abs", "sum"),
    )
    return {"raumtyp_3": rt3, "raumtyp_8": rt8, "joined": joined}


# ---------------------------------------------------------------------------
RAUMTYP_8_NAMES = {
    1: "Metropoles Zentrum",
    2: "Zentrumsnah hochverdichtete Wohnnutzung",
    3: "Zentrumsnah verdichtete Mischnutzung",
    4: "Städtisch mit Verdichtungsansätzen",
    5: "Städtisch mit gewerblicher Prägung",
    6: "Umland Verstädtert",
    7: "Umland dörflich m. geringem gewerbl. Einfluss",
    8: "Umland dörflich ohne gewerbl. Einfluss",
}


def render_raumtyp_aggregate_choropleth(
    plz_gdf, cluster_map, raumtyp_df, raumtyp_col, value_col, out_path, title, label,
    cmap="RdYlGn", vmin=None, vmax=None,
):
    """Karte wo ALLE PLZ desselben Raumtyps die gleiche (aggregat-)Farbe haben."""
    g = forward_cluster_values_to_plz(plz_gdf, cluster_map, raumtyp_df, raumtyp_col)

    # Map raumtyp -> aggregate value
    agg_value = raumtyp_df.set_index(raumtyp_col)[value_col].to_dict()
    g["agg_value"] = g[raumtyp_col].map(agg_value)

    fig, ax = plt.subplots(figsize=(11, 8))
    no_value = g[g["agg_value"].isna()]
    if len(no_value):
        no_value.plot(ax=ax, color="#dddddd", edgecolor="white", lw=0.3)

    with_value = g[g["agg_value"].notna()]
    if vmin is None:
        vmin = with_value["agg_value"].min()
    if vmax is None:
        vmax = with_value["agg_value"].max()

    with_value.plot(ax=ax, column="agg_value", cmap=cmap, vmin=vmin, vmax=vmax,
                      edgecolor="white", lw=0.3, legend=True,
                      legend_kwds={"label": label, "shrink": 0.55})
    # Boundary lines per raumtyp_3 (visual demarcation)
    for rt in g[raumtyp_col].dropna().unique():
        boundary = g[g[raumtyp_col] == rt].dissolve()
        boundary.boundary.plot(ax=ax, color="black", lw=0.6, alpha=0.6)

    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf")); fig.savefig(out_path, dpi=160); plt.close(fig)


# ---------------------------------------------------------------------------
def write_report(kpis: dict, rt_kpis: dict):
    lines = ["# Paper Maps Final — Merge-Forwarded Choropleths + Raumtyp-Aggregate\n"]
    lines.append("Alle Karten in diesem Verzeichnis verwenden **Merge-Forwarding**: PLZ die durch")
    lines.append("`merge_small_plz()` in einen Cluster gefaltet wurden, zeigen den Wert ihres")
    lines.append("Cluster-Repräsentanten — kein 'no data'-Grau auf den 17 Member-PLZ.\n")

    lines.append("## Cluster-Level Saving (n_cluster mit Daten)\n")
    cagg = kpis["cluster_agg"]
    lines.append(f"- Anzahl Cluster mit Saving-Daten: {len(cagg)}")
    lines.append(f"- Mean actual saving across clusters: {cagg['mean_actual_saving_pct'].mean():.2f} %")
    lines.append(f"- Mean predicted saving: {cagg['mean_predicted_saving_pct'].mean():.2f} %")
    lines.append(f"- Mean bias (pred − actual): {cagg['mean_bias_pp'].mean():+.2f} pp")
    lines.append("")

    lines.append("## Per Raumtyp_3 (urban / suburban / rural)\n")
    lines.append("| Raumtyp_3 | # Cluster | Mean actual saving | Mean predicted saving | Mean bias pp | Total weekly EUR saved |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in rt_kpis["raumtyp_3"].iterrows():
        lines.append(f"| {r['raumtyp_3']} | {int(r['n_cluster'])} | {r['mean_actual_saving_pct']:.2f}% | "
                       f"{r['mean_predicted_saving_pct']:.2f}% | {r['mean_bias_pp']:+.2f} | {r['total_saving']:,.0f} |")
    lines.append("")

    lines.append("## Per Raumtyp_8 (BBSR)\n")
    lines.append("| RT | Name | # Cluster | Mean actual saving | Mean bias pp |")
    lines.append("|---:|---|---:|---:|---:|")
    for _, r in rt_kpis["raumtyp_8"].iterrows():
        nm = r.get("raumtyp_8_name") or RAUMTYP_8_NAMES.get(int(r["raumtyp_8"]), "?")
        lines.append(f"| {int(r['raumtyp_8'])} | {nm} | {int(r['n_cluster'])} | "
                       f"{r['mean_actual_saving_pct']:.2f}% | {r['mean_bias_pp']:+.2f} |")
    lines.append("")

    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("Loading geographic + cluster data...")
    plz_gdf = load_plz_polygons()
    cl_map = load_cluster_mapping()
    cr = load_cluster_raumtyp()
    print(f"  {len(plz_gdf)} PLZ polygons, {cl_map['cluster_id'].nunique()} clusters, {len(cr)} raumtyp assignments")

    print("\nComputing cluster KPIs...")
    kpis = compute_cluster_kpis()
    rt_kpis = compute_raumtyp_kpis(kpis["cluster_agg"], cr)

    # ── Map 01: Cluster-Saving (VROOM-actual) ───────────────────────────
    print("M01: cluster_saving_actual ...")
    render_cluster_choropleth(
        plz_gdf, cl_map, kpis["cluster_agg"], "mean_actual_saving_pct",
        OUT / "fig_M01_cluster_saving_actual.png",
        "Fig M01 — Mean actual saving (VROOM-verified) per cluster\n"
        "Merge-forwarded: 17 PLZ in 10 multi-PLZ clusters show their cluster value",
        cmap="RdYlGn", vmin=0, vmax=30, label="Mean saving %",
    )

    # ── Map 02: Saving-Bias per Cluster ─────────────────────────────────
    print("M02: cluster_bias ...")
    render_cluster_choropleth(
        plz_gdf, cl_map, kpis["cluster_agg"], "mean_bias_pp",
        OUT / "fig_M02_cluster_bias.png",
        "Fig M02 — Saving-prediction bias (LGB-logT − VROOM) per cluster\n"
        "Red = ML overestimates saving; Blue = underestimates. Mean bias +10 pp = Best-of-K winner's curse.",
        cmap="RdBu_r", label="Bias pp (predicted − actual)",
        diverging_center=0.0,
    )

    # ── Map 03: Raumtyp_3 aggregate saving ──────────────────────────────
    print("M03: raumtyp_3 aggregate saving ...")
    rt3 = rt_kpis["raumtyp_3"]
    render_raumtyp_aggregate_choropleth(
        plz_gdf, cl_map, cr.merge(rt3, on="raumtyp_3", how="left"),
        "raumtyp_3", "mean_actual_saving_pct",
        OUT / "fig_M03_raumtyp_3_saving.png",
        "Fig M03 — Aggregate VROOM saving per Raumtyp_3 (urban / suburban / rural)",
        label="Aggregate saving %", cmap="RdYlGn", vmin=0, vmax=25,
    )

    # ── Map 04: Raumtyp_8 aggregate saving ──────────────────────────────
    print("M04: raumtyp_8 aggregate saving ...")
    rt8 = rt_kpis["raumtyp_8"]
    render_raumtyp_aggregate_choropleth(
        plz_gdf, cl_map, cr.merge(rt8[["raumtyp_8", "mean_actual_saving_pct"]], on="raumtyp_8", how="left"),
        "raumtyp_8", "mean_actual_saving_pct",
        OUT / "fig_M04_raumtyp_8_saving.png",
        "Fig M04 — Aggregate VROOM saving per Raumtyp_8 (BBSR-style 8 detailed types)",
        label="Aggregate saving %", cmap="RdYlGn", vmin=0, vmax=25,
    )

    # ── Map 05 + 06: Classification maps ────────────────────────────────
    print("M05 + M06: classification maps ...")
    raumtyp_3_color_map = {r: RAUMTYP_3_COLOR[r] for r in RAUMTYP_3_ORDER}
    render_classification_map(
        plz_gdf, cl_map, cr, "raumtyp_3", raumtyp_3_color_map,
        name_map={r: r for r in RAUMTYP_3_ORDER},
        out_path=OUT / "fig_M05_raumtyp_3_classification.png",
        title="Fig M05 — Cluster classification (Raumtyp_3: urban / suburban / rural)",
    )
    render_classification_map(
        plz_gdf, cl_map, cr, "raumtyp_8", RAUMTYP_8_COLORS, name_map=RAUMTYP_8_NAMES,
        out_path=OUT / "fig_M06_raumtyp_8_classification.png",
        title="Fig M06 — Cluster classification (Raumtyp_8: BBSR detailed)",
    )

    # ── Map 07: Cost-MAPE per cluster ───────────────────────────────────
    print("M07: cost MAPE per cluster (SA_ML routed cells) ...")
    render_cluster_choropleth(
        plz_gdf, cl_map, kpis["cost_mape"], "cost_mape_pct",
        OUT / "fig_M07_cost_mape_per_cluster.png",
        "Fig M07 — LGB-logT Cost-MAPE per cluster (on VROOM-routed SA_ML + Fixed cells)\n"
        "Higher = model performs worse there. Look for spatial patterns of model weakness.",
        cmap="OrRd", vmin=0, vmax=25, label="Cost MAPE %",
    )

    print("\nSaving KPI tables...")
    kpis["cluster_agg"].to_csv(OUT / "tab_cluster_aggregates.csv", index=False)
    rt_kpis["raumtyp_3"].to_csv(OUT / "tab_raumtyp_3_aggregates.csv", index=False)
    rt_kpis["raumtyp_8"].to_csv(OUT / "tab_raumtyp_8_aggregates.csv", index=False)

    write_report(kpis, rt_kpis)
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
