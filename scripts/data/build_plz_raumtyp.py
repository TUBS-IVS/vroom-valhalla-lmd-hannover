"""Spatial-Join: PLZ-Polygon x Raumtyp-Polygon (8 detailliert + 3 aggregiert).

Quellen:
  - data/geodata/plz_areas.csv           (85 PLZ, WKT, EPSG:25832)
  - data/geodata/regionclusters.gpkg     (8 raumtyp polygons, EPSG:25832)

Methode:
  Area-weighted majority: fuer jede PLZ wird der Raumtyp mit dem groessten
  Flaechenanteil der Ueberlappung zugewiesen.

Output:
  data/geodata/plz_raumtyp.csv  (plz, raumtyp_8, raumtyp_8_name, raumtyp_3,
                                 main_area_share, coverage_ok)
  results/audits/plz_raumtyp_map.png   (Karte 8er + 3er Klassifikation)
  results/audits/plz_raumtyp_report.md
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from shapely import wkt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "geodata"
OUT_DATA = DATA / "plz_raumtyp.csv"
OUT_CLUSTER = DATA / "cluster_raumtyp.csv"
CLUSTERS_CSV = DATA / "plz_clusters.csv"
OUT_AUDIT = ROOT / "results" / "audits"
OUT_AUDIT.mkdir(parents=True, exist_ok=True)


RAUMTYP_3 = {
    1: "urban",
    2: "urban",
    3: "urban",
    4: "suburban",
    5: "suburban",
    6: "suburban",
    7: "rural",
    8: "rural",
}

RAUMTYP_8_COLORS = {
    1: "#67000d",  # Metropoles Zentrum (dark red)
    2: "#a50f15",
    3: "#cb181d",
    4: "#fb6a4a",
    5: "#fcae91",
    6: "#fee0d2",
    7: "#9ecae1",
    8: "#08519c",  # Umland doerflich (blue)
}

RAUMTYP_3_COLORS = {
    "urban": "#cb181d",
    "suburban": "#fdae61",
    "rural": "#1a9850",
}


def load_plz_polygons() -> gpd.GeoDataFrame:
    df = pd.read_csv(DATA / "plz_areas.csv")
    df["geometry"] = df["WKT"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:25832")
    gdf["plz"] = gdf["plz"].astype(str).str.zfill(5)
    # Dissolve duplicate PLZ rows (87 -> 85)
    gdf = gdf[["plz", "ort", "landkreis", "einwohner", "geometry"]]
    gdf = gdf.dissolve(by="plz", aggfunc={"ort": "first", "landkreis": "first", "einwohner": "sum"}).reset_index()
    return gdf


def load_raumtypen() -> gpd.GeoDataFrame:
    gpkg = DATA / "regionclusters.gpkg"
    if not gpkg.exists():
        raise FileNotFoundError(
            f"{gpkg} not found. Run the conversion from regionclusters.pkl first."
        )
    gdf = gpd.read_file(gpkg)
    if gdf.crs is None or gdf.crs.to_epsg() != 25832:
        gdf = gdf.to_crs(epsg=25832)
    gdf["raumtyp"] = gdf["raumtyp"].astype(int)
    return gdf[["raumtyp", "name", "geometry"]]


def assign_raumtyp(plz_gdf: gpd.GeoDataFrame, raum_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Area-weighted majority assignment."""
    overlay = gpd.overlay(plz_gdf, raum_gdf, how="intersection", keep_geom_type=False)
    overlay["intersect_area"] = overlay.geometry.area
    plz_total_area = plz_gdf.set_index("plz").geometry.area
    overlay["plz_total_area"] = overlay["plz"].map(plz_total_area)
    overlay["area_share"] = overlay["intersect_area"] / overlay["plz_total_area"]
    grp = overlay.groupby(["plz", "raumtyp", "name"], as_index=False)["area_share"].sum()
    idx = grp.groupby("plz")["area_share"].idxmax()
    chosen = grp.loc[idx].reset_index(drop=True)
    chosen = chosen.rename(
        columns={
            "raumtyp": "raumtyp_8",
            "name": "raumtyp_8_name",
            "area_share": "main_area_share",
        }
    )
    chosen["raumtyp_3"] = chosen["raumtyp_8"].map(RAUMTYP_3)
    chosen["coverage_ok"] = chosen["main_area_share"] >= 0.50
    return chosen[["plz", "raumtyp_8", "raumtyp_8_name", "raumtyp_3", "main_area_share", "coverage_ok"]]


def plot_map(plz_gdf: gpd.GeoDataFrame, raum_gdf: gpd.GeoDataFrame, df_map: pd.DataFrame, out_png: Path,
              cluster_df: pd.DataFrame | None = None):
    # Merge-forward: if cluster_df provided, forward cluster raumtyp values to member PLZ
    if cluster_df is not None and "member_plz_list" in cluster_df.columns:
        # build long table: cluster_id -> member PLZ
        long_rows = []
        for _, r in cluster_df.iterrows():
            for m in r["member_plz_list"].split(","):
                long_rows.append({"cluster_id": r["cluster_id"], "plz": m.strip().zfill(5)})
        long_df = pd.DataFrame(long_rows)
        # Map plz directly to its cluster's raumtyp
        cluster_raumtyp_df = df_map.copy()  # if df_map already cluster-level
        if "cluster_id" in cluster_raumtyp_df.columns:
            long_df = long_df.merge(
                cluster_raumtyp_df[["cluster_id", "raumtyp_8", "raumtyp_8_name", "raumtyp_3"]],
                on="cluster_id", how="left")
            joined = plz_gdf.merge(long_df[["plz", "raumtyp_8", "raumtyp_8_name", "raumtyp_3"]],
                                     on="plz", how="left")
        else:
            joined = plz_gdf.merge(df_map, on="plz", how="left")
    else:
        joined = plz_gdf.merge(df_map, on="plz", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # (a) 8 detailed
    ax = axes[0]
    raum_gdf.boundary.plot(ax=ax, color="black", lw=0.4, alpha=0.6)
    for rt in range(1, 9):
        sub = joined[joined["raumtyp_8"] == rt]
        if not len(sub):
            continue
        nm = sub["raumtyp_8_name"].iloc[0]
        sub.plot(
            ax=ax,
            color=RAUMTYP_8_COLORS[rt],
            edgecolor="white",
            lw=0.4,
            label=f"{rt}: {nm}  (n={len(sub)})",
        )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a) 8 detaillierte Raumtypen", loc="left", fontsize=10)
    ax.legend(loc="lower left", fontsize=6.5)

    # (b) 3 aggregated
    ax = axes[1]
    raum_gdf.boundary.plot(ax=ax, color="black", lw=0.4, alpha=0.6)
    for rt3 in ("urban", "suburban", "rural"):
        sub = joined[joined["raumtyp_3"] == rt3]
        if not len(sub):
            continue
        sub.plot(
            ax=ax,
            color=RAUMTYP_3_COLORS[rt3],
            edgecolor="white",
            lw=0.4,
            label=f"{rt3}  (n={len(sub)})",
        )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(b) 3 aggregierte Raumtypen", loc="left", fontsize=10)
    ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("PLZ -> Raumtyp Zuordnung (area-weighted majority)", x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def write_report(df: pd.DataFrame, md_out: Path):
    lines = ["# PLZ -> Raumtyp Mapping\n"]
    lines.append(f"Total PLZ mit Raumtyp-Zuordnung: **{len(df)}**\n")
    low_cov = df[~df["coverage_ok"]]
    if len(low_cov):
        lines.append(f"⚠️  {len(low_cov)} PLZ haben < 50% Flaechenanteil im Hauptraumtyp (Grenzlage):\n")
        lines.append("| PLZ | raumtyp_8 | raumtyp_8_name | main_area_share |")
        lines.append("|---|---:|---|---:|")
        for _, r in low_cov.iterrows():
            lines.append(
                f"| {r['plz']} | {int(r['raumtyp_8'])} | {r['raumtyp_8_name']} | {r['main_area_share']:.2f} |"
            )
        lines.append("")

    lines.append("## Verteilung 8 detailliert\n")
    lines.append("| Raumtyp | Name | # PLZ |")
    lines.append("|---:|---|---:|")
    for rt, grp in df.sort_values("raumtyp_8").groupby("raumtyp_8"):
        lines.append(f"| {int(rt)} | {grp['raumtyp_8_name'].iloc[0]} | {len(grp)} |")
    lines.append("")

    lines.append("## Verteilung 3 aggregiert\n")
    lines.append("| Raumtyp_3 | # PLZ |")
    lines.append("|---|---:|")
    for rt3, n in df["raumtyp_3"].value_counts().reindex(["urban", "suburban", "rural"]).items():
        lines.append(f"| {rt3} | {int(n) if not pd.isna(n) else 0} |")
    lines.append("")
    md_out.write_text("\n".join(lines), encoding="utf-8")


def build_cluster_raumtyp(plz_gdf: gpd.GeoDataFrame, raum_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Cluster-level raumtyp assignment: union member PLZ polygons, then area-weighted majority."""
    clusters = pd.read_csv(CLUSTERS_CSV, dtype={"cluster_id": str})
    clusters["cluster_id"] = clusters["cluster_id"].str.zfill(5)

    # Build cluster-level GeoDataFrame by dissolving member PLZ polygons.
    long_rows = []
    for _, row in clusters.iterrows():
        for m in row["member_plz_list"].split(","):
            long_rows.append({"cluster_id": row["cluster_id"], "plz": m.strip()})
    long_df = pd.DataFrame(long_rows)
    plz_gdf_lookup = plz_gdf.set_index("plz")

    cluster_geoms = []
    for cid, grp in long_df.groupby("cluster_id"):
        members = grp["plz"].tolist()
        present = [p for p in members if p in plz_gdf_lookup.index]
        if not present:
            continue
        member_geoms = plz_gdf_lookup.loc[present, "geometry"]
        unioned = gpd.GeoSeries(member_geoms, crs=plz_gdf.crs).union_all()
        einwohner = float(plz_gdf_lookup.loc[present, "einwohner"].sum())
        cluster_geoms.append(
            {
                "cluster_id": cid,
                "member_plz_list": ",".join(sorted(present)),
                "n_members": len(present),
                "einwohner": einwohner,
                "geometry": unioned,
            }
        )
    cluster_gdf = gpd.GeoDataFrame(cluster_geoms, geometry="geometry", crs=plz_gdf.crs)

    # Spatial-join cluster polygons x raumtyp polygons (area-weighted majority).
    overlay = gpd.overlay(cluster_gdf, raum_gdf, how="intersection", keep_geom_type=False)
    overlay["intersect_area"] = overlay.geometry.area
    cluster_total = cluster_gdf.set_index("cluster_id").geometry.area
    overlay["cluster_total_area"] = overlay["cluster_id"].map(cluster_total)
    overlay["area_share"] = overlay["intersect_area"] / overlay["cluster_total_area"]
    grp = overlay.groupby(["cluster_id", "raumtyp", "name"], as_index=False)["area_share"].sum()
    idx = grp.groupby("cluster_id")["area_share"].idxmax()
    chosen = grp.loc[idx].reset_index(drop=True)
    chosen = chosen.rename(
        columns={
            "raumtyp": "raumtyp_8",
            "name": "raumtyp_8_name",
            "area_share": "main_area_share",
        }
    )
    chosen["raumtyp_3"] = chosen["raumtyp_8"].map(RAUMTYP_3)
    chosen["coverage_ok"] = chosen["main_area_share"] >= 0.50
    # Tack member info back on.
    info = cluster_gdf[["cluster_id", "member_plz_list", "n_members", "einwohner"]]
    out = chosen.merge(info, on="cluster_id", how="left")
    return out[
        [
            "cluster_id",
            "member_plz_list",
            "n_members",
            "einwohner",
            "raumtyp_8",
            "raumtyp_8_name",
            "raumtyp_3",
            "main_area_share",
            "coverage_ok",
        ]
    ]


def main():
    print("Loading PLZ polygons...")
    plz_gdf = load_plz_polygons()
    print(f"  {len(plz_gdf)} unique PLZ polygons")

    print("Loading Raumtyp polygons...")
    raum_gdf = load_raumtypen()
    print(f"  {len(raum_gdf)} Raumtypen")

    print("Computing area-weighted spatial join (PLZ-level)...")
    df_map = assign_raumtyp(plz_gdf, raum_gdf)
    df_map.to_csv(OUT_DATA, index=False)
    print(f"Wrote {OUT_DATA}")

    if CLUSTERS_CSV.exists():
        print("Computing area-weighted spatial join (Cluster-level)...")
        df_cluster = build_cluster_raumtyp(plz_gdf, raum_gdf)
        df_cluster.to_csv(OUT_CLUSTER, index=False)
        print(f"Wrote {OUT_CLUSTER}")
    else:
        print(f"  skip cluster step: {CLUSTERS_CSV} not present (run build_plz_clusters.py first)")
        df_cluster = None

    print("Rendering map (with cluster merge-forwarding)...")
    out_png = OUT_AUDIT / "plz_raumtyp_map.png"
    # If cluster-level raumtyp file exists, use it for merge-forwarding
    if df_cluster is not None and CLUSTERS_CSV.exists():
        cluster_df = pd.read_csv(CLUSTERS_CSV, dtype={"cluster_id": str})
        cluster_df["cluster_id"] = cluster_df["cluster_id"].str.zfill(5)
        plot_map(plz_gdf, raum_gdf, df_cluster, out_png, cluster_df=cluster_df)
    else:
        plot_map(plz_gdf, raum_gdf, df_map, out_png)
    print(f"Wrote {out_png}")

    md_out = OUT_AUDIT / "plz_raumtyp_report.md"
    write_report(df_map, md_out)
    print(f"Wrote {md_out}")

    print("\n=== Distribution (PLZ-level, 8 detail) ===")
    print(df_map.groupby(["raumtyp_8", "raumtyp_8_name"]).size().to_string())
    print("\n=== Distribution (PLZ-level, 3 aggregated) ===")
    print(df_map["raumtyp_3"].value_counts().to_string())

    if df_cluster is not None:
        print("\n=== Distribution (Cluster-level, 8 detail) ===")
        print(df_cluster.groupby(["raumtyp_8", "raumtyp_8_name"]).size().to_string())
        print("\n=== Distribution (Cluster-level, 3 aggregated) ===")
        print(df_cluster["raumtyp_3"].value_counts().to_string())
        n_multi = int((df_cluster["n_members"] > 1).sum())
        print(f"\nClusters with multiple member PLZ: {n_multi}")


if __name__ == "__main__":
    main()
