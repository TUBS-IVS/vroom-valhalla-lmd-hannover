"""Hub loading and PLZ-to-hub assignment with capacity enforcement."""

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from batch_delivery.config.constants import (
    DATA_DIR,
    DEPOT_DAILY_CAP_SMALL,
    LARGE_HUB_TYPES,
    MIN_PLZ_PER_ZSP,
    SMALL_HUB_DISTANCE_PENALTY,
)
from batch_delivery.utils import log


def load_hubs(
    hubs_csv: Path | None = None,
    provider: str = "DHL",
) -> gpd.GeoDataFrame:
    """Load hub locations from CSV and filter by provider.

    Returns a GeoDataFrame with EPSG:25832 geometry plus WGS84 lon/lat.
    """
    if hubs_csv is None:
        hubs_csv = DATA_DIR / "hubs" / "KEP-hubs_v3.csv"
    df_hubs = pd.read_csv(hubs_csv, sep=";")

    mask = df_hubs["Anbieter"].str.contains(provider, case=False, na=False)
    df_filtered = df_hubs[mask].copy()

    gdf_hubs = gpd.GeoDataFrame(
        df_filtered,
        geometry=gpd.points_from_xy(df_filtered["X"], df_filtered["Y"]),
        crs="EPSG:25832",
    )
    gdf_hubs_wgs = gdf_hubs.to_crs(epsg=4326)
    gdf_hubs["lon"] = gdf_hubs_wgs.geometry.x
    gdf_hubs["lat"] = gdf_hubs_wgs.geometry.y

    log.debug(f"{provider} hubs: {len(gdf_hubs)}")
    return gdf_hubs


def assign_plz_to_hubs(
    gdf_plz: gpd.GeoDataFrame,
    gdf_hubs: gpd.GeoDataFrame,
    plz_with_demand: set,
    merge_map: dict | None = None,
) -> pd.DataFrame:
    """Size-weighted nearest-hub assignment for each PLZ.

    Large hubs (ZB, PZ/ZB, BZ) are preferred via distance penalty on small hubs.

    If ``merge_map`` is given, polygons of merged PLZ are rewritten to their
    cluster representative before dissolving, so the centroid (and therefore the
    nearest-hub assignment + ``dist_km``) reflects the *cluster* geometry rather
    than only the representative-PLZ's single polygon. Critical fix discovered
    2026-05-25: cluster 30159's centroid shifts by several km when its members
    30171/30173/30175 are included.

    Returns DataFrame with columns: plz, hub_name, hub_typ, hub_lon, hub_lat,
    hub_x, hub_y, dist_km.
    """
    plz_centroids = gdf_plz[["plz", "geometry"]].copy()
    # Rewrite plz codes via merge_map so dissolve groups cluster polygons.
    if merge_map:
        plz_centroids["plz"] = plz_centroids["plz"].map(
            lambda p: merge_map.get(p, p)
        )
    plz_centroids["centroid"] = plz_centroids.geometry.centroid
    plz_centroids = plz_centroids[plz_centroids["plz"].isin(plz_with_demand)].copy()
    plz_centroids = plz_centroids.dissolve(by="plz").reset_index()
    plz_centroids["centroid"] = plz_centroids.geometry.centroid

    hub_coords = np.array([(g.x, g.y) for g in gdf_hubs.geometry])
    hub_types = list(gdf_hubs["Typ"])

    assignments = []
    for _, row in plz_centroids.iterrows():
        cx, cy = row["centroid"].x, row["centroid"].y
        real_dists = np.sqrt(
            (hub_coords[:, 0] - cx) ** 2 + (hub_coords[:, 1] - cy) ** 2
        )
        weighted_dists = real_dists.copy()
        for i, ht in enumerate(hub_types):
            if ht not in LARGE_HUB_TYPES:
                weighted_dists[i] *= SMALL_HUB_DISTANCE_PENALTY
        nearest_idx = int(np.argmin(weighted_dists))
        nearest = gdf_hubs.iloc[nearest_idx]
        assignments.append(
            {
                "plz": row["plz"],
                "hub_name": nearest["Bezeichnun"],
                "hub_typ": nearest["Typ"],
                "hub_lon": nearest["lon"],
                "hub_lat": nearest["lat"],
                "hub_x": nearest["X"],
                "hub_y": nearest["Y"],
                "dist_km": round(real_dists[nearest_idx] / 1000, 2),
            }
        )

    df_assignments = pd.DataFrame(assignments)
    log.debug(f"Assigned {len(df_assignments)} PLZ to {df_assignments['hub_name'].nunique()} hubs")
    return df_assignments


def enforce_zsp_min_plz(
    df_assignments: pd.DataFrame,
    gdf_hubs: gpd.GeoDataFrame,
    plz_centroids_gdf: gpd.GeoDataFrame,
    merge_map: dict | None = None,
) -> pd.DataFrame:
    """Ensure each ZSP (small hub) has at least MIN_PLZ_PER_ZSP areas.

    Pulls nearest PLZ from large hubs if needed.

    If ``merge_map`` is given, polygons are dissolved to cluster reps before
    iteration so each cluster_id gets the true cluster centroid (not the
    representative polygon's centroid) — discovered 2026-05-25.
    """
    if merge_map:
        plz_centroids_gdf = plz_centroids_gdf.copy()
        plz_centroids_gdf["plz"] = plz_centroids_gdf["plz"].map(
            lambda p: merge_map.get(p, p)
        )
        # Keep only cluster reps; dissolve their polygons
        cluster_ids = set(df_assignments["plz"].unique())
        plz_centroids_gdf = plz_centroids_gdf[plz_centroids_gdf["plz"].isin(cluster_ids)]
        plz_centroids_gdf = plz_centroids_gdf.dissolve(by="plz").reset_index()
    pulled = 0
    hub_coords = np.array([(g.x, g.y) for g in gdf_hubs.geometry])

    for hub_name in df_assignments["hub_name"].unique():
        hub_mask = df_assignments["hub_name"] == hub_name
        hub_plzs = df_assignments[hub_mask]
        hub_typ = hub_plzs.iloc[0]["hub_typ"]
        if hub_typ in LARGE_HUB_TYPES or len(hub_plzs) >= MIN_PLZ_PER_ZSP:
            continue

        hub_row = gdf_hubs[gdf_hubs["Bezeichnun"] == hub_name].iloc[0]
        hx, hy = hub_row.geometry.x, hub_row.geometry.y
        need = MIN_PLZ_PER_ZSP - len(hub_plzs)
        already = set(hub_plzs["plz"])

        candidates = []
        for _, pc_row in plz_centroids_gdf.iterrows():
            pcode = pc_row["plz"]
            if pcode in already:
                continue
            cur = df_assignments[df_assignments["plz"] == pcode]
            if len(cur) == 0:
                continue
            cur_typ = cur.iloc[0]["hub_typ"]
            if cur_typ not in LARGE_HUB_TYPES:
                continue
            donor_n = (df_assignments["hub_name"] == cur.iloc[0]["hub_name"]).sum()
            if donor_n <= 3:
                continue
            pcx = pc_row.geometry.centroid.x
            pcy = pc_row.geometry.centroid.y
            dist = math.sqrt((hx - pcx) ** 2 + (hy - pcy) ** 2)
            candidates.append((pcode, dist))

        candidates.sort(key=lambda x: x[1])
        for pcode, dist in candidates[:need]:
            idx_mask = df_assignments["plz"] == pcode
            df_assignments.loc[idx_mask, "hub_name"] = hub_name
            df_assignments.loc[idx_mask, "hub_typ"] = hub_typ
            df_assignments.loc[idx_mask, "hub_lon"] = hub_row["lon"]
            df_assignments.loc[idx_mask, "hub_lat"] = hub_row["lat"]
            df_assignments.loc[idx_mask, "hub_x"] = hub_row["X"]
            df_assignments.loc[idx_mask, "hub_y"] = hub_row["Y"]
            df_assignments.loc[idx_mask, "dist_km"] = round(dist / 1000, 2)
            pulled += 1

    if pulled:
        log.debug(f"{pulled} PLZ pulled to ZSPs (min {MIN_PLZ_PER_ZSP} PLZ each)")
    return df_assignments


def enforce_depot_capacity(
    df_assignments: pd.DataFrame,
    gdf_hubs: gpd.GeoDataFrame,
    plz_avg_daily: dict,
    plz_centroids_gdf: gpd.GeoDataFrame,
    merge_map: dict | None = None,
) -> pd.DataFrame:
    """Reassign PLZ from over-capacity ZSP depots to nearest viable hub.

    Parameters
    ----------
    plz_avg_daily : dict
        {plz: average_daily_demand}
    merge_map : dict | None
        If given, dissolves polygons to cluster reps so centroid lookups
        reflect true cluster geometry. Fix 2026-05-25.
    """
    if merge_map:
        plz_centroids_gdf = plz_centroids_gdf.copy()
        plz_centroids_gdf["plz"] = plz_centroids_gdf["plz"].map(
            lambda p: merge_map.get(p, p)
        )
        cluster_ids = set(df_assignments["plz"].unique())
        plz_centroids_gdf = plz_centroids_gdf[plz_centroids_gdf["plz"].isin(cluster_ids)]
        plz_centroids_gdf = plz_centroids_gdf.dissolve(by="plz").reset_index()
    hub_coords = np.array([(g.x, g.y) for g in gdf_hubs.geometry])
    hub_types = list(gdf_hubs["Typ"])
    reassigned = 0

    df_assignments["_avg_daily"] = df_assignments["plz"].map(plz_avg_daily).fillna(0)

    for hub_name in df_assignments["hub_name"].unique():
        hub_mask = df_assignments["hub_name"] == hub_name
        hub_plzs = df_assignments[hub_mask]
        hub_typ = hub_plzs.iloc[0]["hub_typ"]
        if hub_typ in LARGE_HUB_TYPES:
            continue

        hub_daily = hub_plzs["_avg_daily"].sum()
        if hub_daily <= DEPOT_DAILY_CAP_SMALL:
            continue

        log.warning(
            f'ZSP "{hub_name}" exceeds cap: {hub_daily:,.0f} > {DEPOT_DAILY_CAP_SMALL:,} p/day'
        )

        sorted_plzs = hub_plzs.sort_values("dist_km", ascending=False)
        remaining = hub_daily
        for _, plz_row in sorted_plzs.iterrows():
            if remaining <= DEPOT_DAILY_CAP_SMALL:
                break
            plz = plz_row["plz"]
            plz_demand_val = plz_row["_avg_daily"]

            plz_c = plz_centroids_gdf[plz_centroids_gdf["plz"] == plz]
            if len(plz_c) == 0:
                continue
            cx = plz_c.iloc[0].geometry.centroid.x
            cy = plz_c.iloc[0].geometry.centroid.y
            real_dists = np.sqrt(
                (hub_coords[:, 0] - cx) ** 2 + (hub_coords[:, 1] - cy) ** 2
            )
            weighted_dists = real_dists.copy()
            for i, ht in enumerate(hub_types):
                if ht not in LARGE_HUB_TYPES:
                    weighted_dists[i] *= SMALL_HUB_DISTANCE_PENALTY

            for alt_idx in np.argsort(weighted_dists):
                alt_hub = gdf_hubs.iloc[alt_idx]
                if alt_hub["Bezeichnun"] == hub_name:
                    continue
                idx_mask = df_assignments["plz"] == plz
                df_assignments.loc[idx_mask, "hub_name"] = alt_hub["Bezeichnun"]
                df_assignments.loc[idx_mask, "hub_typ"] = alt_hub["Typ"]
                df_assignments.loc[idx_mask, "hub_lon"] = alt_hub["lon"]
                df_assignments.loc[idx_mask, "hub_lat"] = alt_hub["lat"]
                df_assignments.loc[idx_mask, "hub_x"] = alt_hub["X"]
                df_assignments.loc[idx_mask, "hub_y"] = alt_hub["Y"]
                df_assignments.loc[idx_mask, "dist_km"] = round(
                    real_dists[alt_idx] / 1000, 2
                )
                remaining -= plz_demand_val
                reassigned += 1
                break

    df_assignments.drop(columns=["_avg_daily"], inplace=True)
    if reassigned:
        log.debug(
            f"{reassigned} PLZ reassigned due to ZSP capacity limits "
            f"(max {DEPOT_DAILY_CAP_SMALL:,} p/day)"
        )
    return df_assignments
