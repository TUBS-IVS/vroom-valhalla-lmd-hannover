"""FW6.A Option 3: compute disjoint-cluster topology features.

Per (provider, cluster_plz) emits geometric features that capture
the "is this cluster a fused-but-internally-disjoint multi-PLZ region" signal.

Key insight: post-union geometry often fuses (n_components=1) because urban
PLZ are contiguous. The discriminating signal lives at the MEMBER level —
how the original PLZ polygons are arranged before being merged.

Features:
    n_merged_members         : number of source PLZ in the cluster
    n_components_post_union  : disjoint polygons after unary_union of members
    hull_overhead_pct        : 100 * (hull_area - union_area) / union_area
    max_member_centroid_km   : max pairwise distance between MEMBER centroids
    mean_member_centroid_km  : mean pairwise distance between MEMBER centroids
    bbox_overhead_pct        : 100 * (bbox_area - union_area) / union_area
    isoperimetric_q          : 4*pi*A/P^2 (1 = circle, <1 = elongated)

Reads:
    results/checkpoints/01_demand.pkl   (gdf_plz + per-provider merge_map)

Writes:
    results/disjoint_features/topo_per_provider_plz.csv
"""
from __future__ import annotations
import math
import pickle
from pathlib import Path

import pandas as pd
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
CHK = ROOT / "results" / "checkpoints" / "01_demand.pkl"
OUT_DIR = ROOT / "results" / "disjoint_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "topo_per_provider_plz.csv"


def main():
    data = pickle.load(open(CHK, "rb"))
    gdf_plz = data["gdf_plz"].copy()
    gdf_plz["plz"] = gdf_plz["plz"].astype(str).str.zfill(5)
    # EPSG:25832 metric — geometry already in meters
    geom_by_plz = gdf_plz.groupby("plz")["geometry"].apply(lambda gs: unary_union(list(gs)))

    rows = []
    for provider, pdata in data["provider_data"].items():
        merge_map = pdata.get("merge_map", {})
        # cluster_plz -> set of source plz (including itself)
        cluster_members: dict[str, set[str]] = {}
        for src, tgt in merge_map.items():
            cluster_members.setdefault(tgt, set()).add(src)
        # singletons: every plz the provider serves that is itself the cluster head
        served = set(pdata["df_assignments"]["plz"].astype(str).str.zfill(5))
        for plz in served:
            cluster_members.setdefault(plz, set()).add(plz)

        for cluster_plz, members in cluster_members.items():
            member_geoms = [geom_by_plz.get(m) for m in members if m in geom_by_plz.index]
            if not member_geoms:
                continue
            union_geom = unary_union(member_geoms)
            comps = list(union_geom.geoms) if union_geom.geom_type == "MultiPolygon" else [union_geom]
            union_area_m2 = union_geom.area
            if union_area_m2 == 0:
                continue
            # Post-union shape descriptors
            hull_area_m2 = union_geom.convex_hull.area
            bbox = union_geom.envelope
            bbox_area_m2 = bbox.area
            perimeter_m = union_geom.length
            isoperimetric_q = (4.0 * math.pi * union_area_m2) / (perimeter_m ** 2) if perimeter_m > 0 else 0.0

            # Member-level centroid distances (KEY: captures internal-disjoint structure
            # even when unary_union fuses into a single polygon)
            member_centroids = [g.centroid for g in member_geoms]
            pair_dists_m = []
            for i, ci in enumerate(member_centroids):
                for cj in member_centroids[i + 1:]:
                    pair_dists_m.append(ci.distance(cj))
            max_member_d_km = (max(pair_dists_m) / 1000.0) if pair_dists_m else 0.0
            mean_member_d_km = (sum(pair_dists_m) / len(pair_dists_m) / 1000.0) if pair_dists_m else 0.0

            rows.append({
                "provider": provider,
                "plz": cluster_plz,
                "n_merged_members": len(members),
                "n_components_post_union": len(comps),
                "hull_overhead_pct": 100.0 * (hull_area_m2 - union_area_m2) / union_area_m2,
                "bbox_overhead_pct": 100.0 * (bbox_area_m2 - union_area_m2) / union_area_m2,
                "max_member_centroid_km": max_member_d_km,
                "mean_member_centroid_km": mean_member_d_km,
                "isoperimetric_q": isoperimetric_q,
                "sum_area_km2": union_area_m2 / 1e6,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df)} rows")
    print("\n=== Top 15 by mean_member_centroid_km (most spread-out clusters) ===")
    cols = ["provider", "plz", "n_merged_members", "max_member_centroid_km",
            "mean_member_centroid_km", "hull_overhead_pct", "isoperimetric_q",
            "sum_area_km2"]
    print(df.nlargest(15, "mean_member_centroid_km")[cols].to_string(index=False))
    print("\n=== Stats ===")
    print(f"  total rows: {len(df)}")
    print(f"  multi-member (>=2): {(df.n_merged_members >= 2).sum()}")
    print(f"  max n_merged_members: {df.n_merged_members.max()}")
    print(f"  max max_member_centroid_km: {df.max_member_centroid_km.max():.2f} km")
    print(f"  max hull_overhead_pct: {df.hull_overhead_pct.max():.1f}%")
    print(f"  min isoperimetric_q: {df.isoperimetric_q.min():.3f}  (most elongated)")


if __name__ == "__main__":
    main()
