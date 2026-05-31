"""Demand loading: HAGRID shapefiles, PLZ merge, weekday profiles."""

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt

from batch_delivery.config.constants import (
    DATA_DIR,
    FAST_SHARE_B2B,
    FAST_SHARE_B2C,
    MIN_PLZ_JOBS_MERGE,
    N_DAYS,
    OUTLIER_THRESHOLD,
    WEEKDAYS,
    provider_to_demand_prefix,
)
from batch_delivery.utils import log

# Default day files (HAGRID weekly shapefiles)
DEFAULT_DAY_FILES = [
    (0, "Monday",    "hagrid_parcel_demand_2025-05-12_(Monday).shp"),
    (1, "Tuesday",   "hagrid_parcel_demand_2025-05-13_(Tuesday).shp"),
    (2, "Wednesday", "hagrid_parcel_demand_2025-05-14_(Wednesday).shp"),
    (3, "Thursday",  "hagrid_parcel_demand_2025-05-15_(Thursday).shp"),
    (4, "Friday",    "hagrid_parcel_demand_2025-05-16_(Friday).shp"),
    (5, "Saturday",  "hagrid_parcel_demand_2025-05-17_(Saturday).shp"),
]


def load_daily_demand(
    week_dir: Path | None = None,
    day_files: list | None = None,
    lsp_prefix: str = "dhl",
) -> tuple[dict, dict, dict]:
    """Load weekly HAGRID shapefiles and return per-day GeoDataFrames.

    Parameters
    ----------
    week_dir : Path
        Directory with the daily shapefiles.
    day_files : list
        List of (day_idx, day_name, filename) tuples.
    lsp_prefix : str
        Column prefix in shapefile (e.g. 'dhl' → 'dhl_b2c', 'dhl_b2b').

    Returns
    -------
    daily_gdfs : dict[int, GeoDataFrame]
        Per-day GDFs with columns: str_idx, plz, {prefix}_b2c, {prefix}_b2b,
        {prefix}_total, geometry.
    plz_day_b2c : dict[str, dict[int, int]]
        {plz: {day_idx: b2c_total}}
    plz_day_b2b : dict[str, dict[int, int]]
        {plz: {day_idx: b2b_total}}
    """
    if week_dir is None:
        week_dir = DATA_DIR / "demand" / "week"
    if day_files is None:
        day_files = DEFAULT_DAY_FILES

    lsp_prefix = provider_to_demand_prefix(lsp_prefix)
    col_b2c = f"{lsp_prefix}_b2c"
    col_b2b = f"{lsp_prefix}_b2b"
    col_total = f"{lsp_prefix}_total"

    daily_gdfs = {}
    plz_day_b2c = {}
    plz_day_b2b = {}

    for day_idx, day_name, filename in day_files:
        gdf = gpd.read_file(week_dir / filename)
        keep_cols = ["str_idx", "postal_cod", col_b2c, col_b2b, "geometry"]
        missing = [c for c in keep_cols if c not in gdf.columns]
        if missing:
            raise KeyError(
                f"Missing demand columns for prefix '{lsp_prefix}' in {filename}: {missing}"
            )
        gdf = gdf[keep_cols].copy()
        gdf[col_total] = gdf[col_b2c] + gdf[col_b2b]

        gdf = gdf[gdf[col_total] > 0].copy()
        gdf = gdf[gdf[col_total] <= OUTLIER_THRESHOLD].copy()
        gdf["plz"] = gdf["postal_cod"].astype(str).str.zfill(5)

        # Rename to standard columns for downstream use
        gdf = gdf.rename(columns={col_total: "dhl_total",
                                   col_b2c: "dhl_b2c",
                                   col_b2b: "dhl_b2b"})
        daily_gdfs[day_idx] = gdf

        for plz_code, grp in gdf.groupby("plz"):
            plz_day_b2c.setdefault(plz_code, {})[day_idx] = int(grp["dhl_b2c"].sum())
            plz_day_b2b.setdefault(plz_code, {})[day_idx] = int(grp["dhl_b2b"].sum())

        log.debug(
            f"{day_name}: {len(gdf):>6,} pts  "
            f"B2C={gdf['dhl_b2c'].sum():>6,}  B2B={gdf['dhl_b2b'].sum():>5,}  "
            f"Total={gdf['dhl_total'].sum():>6,}"
        )

    return daily_gdfs, plz_day_b2c, plz_day_b2b


def build_unified_gdf(daily_gdfs: dict) -> gpd.GeoDataFrame:
    """Build a unified GDF of all unique delivery points across the week.

    Deduplicates by str_idx, keeping average daily demand.
    """
    frames = []
    for gdf in daily_gdfs.values():
        frames.append(
            gdf[["str_idx", "postal_cod", "dhl_b2c", "dhl_b2b",
                 "dhl_total", "geometry"]].copy()
        )

    all_points = pd.concat(frames, ignore_index=True)
    gdf_dhl = all_points.groupby("str_idx").agg(
        postal_cod=("postal_cod", "first"),
        dhl_tag=("dhl_total", "mean"),
        geometry=("geometry", "first"),
    ).reset_index()
    gdf_dhl = gpd.GeoDataFrame(gdf_dhl, geometry="geometry", crs=next(iter(daily_gdfs.values())).crs)
    gdf_dhl["dhl_tag"] = gdf_dhl["dhl_tag"].round().astype(int)
    gdf_dhl["plz"] = gdf_dhl["postal_cod"].astype(str).str.zfill(5)

    log.debug(
        f"Unified gdf_dhl: {len(gdf_dhl):,} unique delivery locations, "
        f'{gdf_dhl["plz"].nunique()} PLZ'
    )
    return gdf_dhl


def merge_small_plz(
    gdf_dhl: gpd.GeoDataFrame,
    gdf_plz: gpd.GeoDataFrame,
    plz_day_b2c: dict,
    plz_day_b2b: dict,
    daily_gdfs: dict | None = None,
) -> tuple[gpd.GeoDataFrame, dict, dict, dict]:
    """Merge PLZ areas with fewer than MIN_PLZ_JOBS_MERGE points into neighbours.

    Returns updated (gdf_dhl, plz_day_b2c, plz_day_b2b, merge_map).

    If ``daily_gdfs`` is provided, also rewrites the per-day per-PLZ assignments
    in-place so that routing receives the full cluster demand instead of only
    the representative-PLZ-polygon points. Critical fix discovered 2026-05-25:
    without this update, VROOM only routes ~39% of cluster demand for merged
    clusters (e.g. 30159 cluster: HAGRID 9019 pkts/wk, VROOM 3562 pkts/wk).
    """
    plz_counts = gdf_dhl.groupby("plz").size()
    small_plzs = set(plz_counts[plz_counts < MIN_PLZ_JOBS_MERGE].index) - {"00000"}
    merge_map = {}

    if not small_plzs:
        log.debug(f"No small PLZ areas (< {MIN_PLZ_JOBS_MERGE} pts)")
        return gdf_dhl, plz_day_b2c, plz_day_b2b, merge_map

    plz_with_demand = set(gdf_dhl["plz"].unique()) - {"00000"}
    large_plzs = plz_with_demand - small_plzs

    plz_centroid_map = {}
    for _, row in gdf_plz.iterrows():
        if row["plz"] in plz_with_demand:
            plz_centroid_map[row["plz"]] = (
                row.geometry.centroid.x,
                row.geometry.centroid.y,
            )

    for plz_code in plz_with_demand:
        if plz_code not in plz_centroid_map:
            pts = gdf_dhl[gdf_dhl["plz"] == plz_code]
            plz_centroid_map[plz_code] = (
                pts.geometry.centroid.x.mean(),
                pts.geometry.centroid.y.mean(),
            )

    for s_plz in sorted(small_plzs):
        if s_plz not in plz_centroid_map:
            continue
        sx, sy = plz_centroid_map[s_plz]
        best_dist = float("inf")
        best_target = None
        for l_plz in large_plzs:
            if l_plz not in plz_centroid_map:
                continue
            lx, ly = plz_centroid_map[l_plz]
            d = math.sqrt((sx - lx) ** 2 + (sy - ly) ** 2)
            if d < best_dist:
                best_dist = d
                best_target = l_plz
        if best_target:
            merge_map[s_plz] = best_target

    n_merged_points = 0
    for s_plz, t_plz in merge_map.items():
        mask = gdf_dhl["plz"] == s_plz
        n_merged_points += mask.sum()
        gdf_dhl.loc[mask, "plz"] = t_plz

    log.debug(
        f"Merged {len(merge_map)} small PLZ (< {MIN_PLZ_JOBS_MERGE} pts): "
        f"{n_merged_points} points reassigned"
    )

    # Apply merge to per-day demand dicts
    for s_plz, t_plz in merge_map.items():
        if s_plz in plz_day_b2c:
            for d_idx, v in plz_day_b2c[s_plz].items():
                plz_day_b2c.setdefault(t_plz, {})[d_idx] = (
                    plz_day_b2c.get(t_plz, {}).get(d_idx, 0) + v
                )
            del plz_day_b2c[s_plz]
        if s_plz in plz_day_b2b:
            for d_idx, v in plz_day_b2b[s_plz].items():
                plz_day_b2b.setdefault(t_plz, {})[d_idx] = (
                    plz_day_b2b.get(t_plz, {}).get(d_idx, 0) + v
                )
            del plz_day_b2b[s_plz]

    # CRITICAL FIX 2026-05-25: also rewrite plz codes in daily_gdfs so that
    # routing (which uses daily_gdfs_wgs) sees the FULL cluster demand instead
    # of only the representative-PLZ-polygon points.
    if daily_gdfs is not None:
        for d_idx, gdf_d in daily_gdfs.items():
            if gdf_d is None or len(gdf_d) == 0:
                continue
            for s_plz, t_plz in merge_map.items():
                mask = gdf_d["plz"] == s_plz
                if mask.any():
                    gdf_d.loc[mask, "plz"] = t_plz

    return gdf_dhl, plz_day_b2c, plz_day_b2b, merge_map


def fill_missing_days(
    all_plz_set: set, plz_day_b2c: dict, plz_day_b2b: dict
) -> None:
    """Ensure every PLZ has entries for all 6 days (fill with 0)."""
    for plz_code in all_plz_set:
        for d_idx in range(N_DAYS):
            plz_day_b2c.setdefault(plz_code, {}).setdefault(d_idx, 0)
            plz_day_b2b.setdefault(plz_code, {}).setdefault(d_idx, 0)


def load_plz_areas(plz_csv: Path | None = None) -> gpd.GeoDataFrame:
    """Load PLZ polygon areas from CSV with WKT geometry."""
    if plz_csv is None:
        plz_csv = DATA_DIR / "geodata" / "plz_areas.csv"
    df_plz = pd.read_csv(plz_csv)
    df_plz["geometry"] = df_plz["WKT"].apply(wkt.loads)
    gdf_plz = gpd.GeoDataFrame(df_plz, geometry="geometry", crs="EPSG:25832")
    gdf_plz["plz"] = gdf_plz["plz"].astype(str).str.zfill(5)
    log.debug(f"PLZ areas: {len(gdf_plz)} zones")
    return gdf_plz


def compute_demand_stats(
    all_plz_set: set,
    plz_day_b2c: dict,
    plz_day_b2b: dict,
    df_assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, float, dict]:
    """Compute plz_demand DataFrame, blended FAST_SHARE, and weekday profiles.

    Returns
    -------
    plz_demand : DataFrame
    fast_share : float  (blended express share)
    weekday_shares : dict[str, float]
    """
    rows = []
    for plz_code in sorted(all_plz_set):
        b2c_by_day = plz_day_b2c.get(plz_code, {})
        b2b_by_day = plz_day_b2b.get(plz_code, {})
        b2c_week = sum(b2c_by_day.values())
        b2b_week = sum(b2b_by_day.values())
        total_week_plz = b2c_week + b2b_week

        row = {
            "plz": plz_code,
            "weekly_parcels": total_week_plz,
            "daily_parcels": int(round(total_week_plz / N_DAYS)),
            "b2c_weekly": b2c_week,
            "b2b_weekly": b2b_week,
            "b2c_daily": int(round(b2c_week / N_DAYS)),
            "b2b_daily": int(round(b2b_week / N_DAYS)),
        }
        for d_idx in range(N_DAYS):
            day = WEEKDAYS[d_idx]
            row[f"arrivals_b2c_{day}"] = b2c_by_day.get(d_idx, 0)
            row[f"arrivals_b2b_{day}"] = b2b_by_day.get(d_idx, 0)
            row[f"arrivals_{day}"] = row[f"arrivals_b2c_{day}"] + row[f"arrivals_b2b_{day}"]
        rows.append(row)

    plz_demand = pd.DataFrame(rows)
    plz_demand = plz_demand.merge(
        df_assignments[["plz", "hub_name", "hub_typ"]], on="plz", how="left"
    )

    total_weekly_all = plz_demand["weekly_parcels"].sum()
    total_b2c = plz_demand["b2c_weekly"].sum()
    total_b2b = plz_demand["b2b_weekly"].sum()

    b2c_share = total_b2c / total_weekly_all if total_weekly_all > 0 else 0.5
    b2b_share = total_b2b / total_weekly_all if total_weekly_all > 0 else 0.5
    fast_share = b2c_share * FAST_SHARE_B2C + b2b_share * FAST_SHARE_B2B

    weekday_shares = {}
    for day in WEEKDAYS:
        day_total = plz_demand[f"arrivals_{day}"].sum()
        weekday_shares[day] = day_total / total_weekly_all if total_weekly_all > 0 else 1 / N_DAYS

    # Express split columns
    plz_demand["fast_b2c_daily"] = (plz_demand["b2c_daily"] * FAST_SHARE_B2C).astype(int)
    plz_demand["fast_b2b_daily"] = (plz_demand["b2b_daily"] * FAST_SHARE_B2B).astype(int)
    plz_demand["fast_daily"] = plz_demand["fast_b2c_daily"] + plz_demand["fast_b2b_daily"]
    plz_demand["batch_weekly"] = (
        plz_demand["weekly_parcels"] - plz_demand["fast_daily"] * N_DAYS
    )

    log.debug(
        f"Demand stats: {len(plz_demand)} PLZ, "
        f'{plz_demand["daily_parcels"].sum():,} p/day, '
        f"express={fast_share:.1%}"
    )
    return plz_demand, fast_share, weekday_shares


# ── Demand shifting functions ────────────────────────────────────────────────

def get_source_days(delivery_day: int, delivery_days_sorted: list[int]) -> list[int]:
    """Which days' parcels accumulate to this delivery day?

    HAGRID logic: from the day after the previous delivery day up to and
    including this delivery day.  Cyclic wrapping for Sat→Mon.
    """
    idx = delivery_days_sorted.index(delivery_day)
    prev_dd = delivery_days_sorted[idx - 1]  # Python [-1] wraps
    source = []
    d = (prev_dd + 1) % N_DAYS
    while True:
        source.append(d)
        if d == delivery_day:
            break
        d = (d + 1) % N_DAYS
    return source


def compute_shifted_demand_plz(
    delivery_days_set: set,
    plz_b2c: dict,
    plz_b2b: dict,
    fast_share_b2c: float = FAST_SHARE_B2C,
    fast_share_b2b: float = FAST_SHARE_B2B,
) -> dict:
    """Per-delivery-day demand for one PLZ after accumulation and express split.

    Returns {day_idx: {total, b2c, b2b, source_days, express_only}}.
    Delivery days carry batch + express.
    Non-delivery days carry express only.
    Total weekly parcels are preserved.
    """
    del_sorted = sorted(delivery_days_set)
    result = {}

    # Delivery days: accumulated batch + today's express
    for dd in del_sorted:
        src = get_source_days(dd, del_sorted)
        tb2c = sum(plz_b2c.get(d, 0) for d in src)
        tb2b = sum(plz_b2b.get(d, 0) for d in src)

        # Subtract express from non-delivery source days (delivered separately)
        for d in src:
            if d not in delivery_days_set or d == dd:
                continue
            # This source day is actually a delivery day itself — irrelevant for
            # this schedule since src only contains consecutive non-DD days + dd.
            pass
        # Express from non-delivery source days is delivered separately on those days
        expr_sub_b2c = sum(
            round(plz_b2c.get(d, 0) * fast_share_b2c)
            for d in src if d != dd and d not in delivery_days_set
        )
        expr_sub_b2b = sum(
            round(plz_b2b.get(d, 0) * fast_share_b2b)
            for d in src if d != dd and d not in delivery_days_set
        )
        total = max(0, tb2c + tb2b - expr_sub_b2c - expr_sub_b2b)
        result[dd] = {
            "total": total,
            "b2c": tb2c,
            "b2b": tb2b,
            "source_days": src,
            "express_only": False,
        }

    # Non-delivery days: express only
    for d in range(N_DAYS):
        if d in delivery_days_set:
            continue
        expr_b2c = round(plz_b2c.get(d, 0) * fast_share_b2c)
        expr_b2b = round(plz_b2b.get(d, 0) * fast_share_b2b)
        expr_total = expr_b2c + expr_b2b
        if expr_total > 0:
            result[d] = {
                "total": expr_total,
                "b2c": expr_b2c,
                "b2b": expr_b2b,
                "source_days": [d],
                "express_only": True,
            }

    return result


def merge_source_points(
    source_days: list[int],
    plz_code: str,
    daily_gdfs_wgs: dict,
) -> pd.DataFrame:
    """Merge delivery points from multiple source days for one PLZ.

    Aggregates duplicate str_idx entries (sums demand).
    """
    frames = []
    for d_idx in source_days:
        gdf_w = daily_gdfs_wgs.get(d_idx)
        if gdf_w is None:
            continue
        pts = gdf_w[gdf_w["plz"] == plz_code]
        if len(pts) > 0:
            frames.append(
                pts[["str_idx", "plz", "dhl_total", "lon", "lat", "geometry"]].copy()
            )
    if not frames:
        return pd.DataFrame(
            columns=["str_idx", "plz", "dhl_total", "lon", "lat", "geometry"]
        )
    merged = pd.concat(frames, ignore_index=True)
    merged = (
        merged.groupby("str_idx")
        .agg(
            plz=("plz", "first"),
            dhl_total=("dhl_total", "sum"),
            lon=("lon", "first"),
            lat=("lat", "first"),
            geometry=("geometry", "first"),
        )
        .reset_index()
    )
    return merged


def convert_daily_gdfs_to_wgs(daily_gdfs: dict) -> dict:
    """Convert daily GDFs to WGS84 and add lon/lat columns."""
    daily_gdfs_wgs = {}
    for d_idx, gdf in daily_gdfs.items():
        gdf_w = gdf.to_crs(epsg=4326).copy()
        gdf_w["lon"] = gdf_w.geometry.x
        gdf_w["lat"] = gdf_w.geometry.y
        daily_gdfs_wgs[d_idx] = gdf_w
    return daily_gdfs_wgs


def compute_avg_waiting_days(
    delivery_days_set: set,
    plz_b2c: dict,
    plz_b2b: dict,
    fast_share_b2c: float = FAST_SHARE_B2C,
    fast_share_b2b: float = FAST_SHARE_B2B,
) -> float:
    """Compute demand-weighted average customer waiting time (days).

    Express parcels: 0 wait (same-day).
    Batch parcels on non-delivery days: wait until next delivery day.
    """
    total_parcels = 0.0
    total_wait = 0.0

    for d in range(N_DAYS):
        day_demand = plz_b2c.get(d, 0) + plz_b2b.get(d, 0)
        if day_demand == 0:
            continue

        if d in delivery_days_set:
            total_parcels += day_demand
            # All delivered same day → 0 wait
        else:
            expr = round(plz_b2c.get(d, 0) * fast_share_b2c) + round(
                plz_b2b.get(d, 0) * fast_share_b2b
            )
            batch = day_demand - expr
            total_parcels += day_demand

            # Wait = days until next delivery day (cyclic)
            wait = 0
            check = (d + 1) % N_DAYS
            while check != d:
                wait += 1
                if check in delivery_days_set:
                    break
                check = (check + 1) % N_DAYS
            total_wait += batch * wait

    return total_wait / max(1.0, total_parcels)


def prepare_plz_data(
    all_plz_set: set,
    df_assignments: pd.DataFrame,
    gdf_plz,
    gdf_dhl,
    daily_gdfs_wgs: dict,
    plz_day_b2c: dict,
    plz_day_b2b: dict,
    merge_map: dict | None = None,
) -> dict:
    """Build per-PLZ data dict needed by cost evaluators and optimisers.

    If ``merge_map`` is provided, ``area_km2`` is the *summed* polygon area of
    all cluster member PLZ (representative + merged-in members). Critical fix
    discovered 2026-05-25: without this, area_km2 stays at the representative
    PLZ's single polygon area (e.g. cluster 30159 = 1.96 km² instead of the
    correct ~9.1 km² sum). Cascades to parcels_per_km2 and inference quality.
    """
    # Build cluster -> member list mapping
    cluster_members: dict[str, list[str]] = {}
    if merge_map is not None:
        for s_plz, t_plz in merge_map.items():
            cluster_members.setdefault(t_plz, []).append(s_plz)

    data = {}
    for plz_code in sorted(all_plz_set):
        hub_row = df_assignments[df_assignments["plz"] == plz_code]
        if len(hub_row) == 0:
            continue
        hub_dist = hub_row.iloc[0]["dist_km"]
        # Summed area over all cluster members (representative + merged-in)
        members = [plz_code] + cluster_members.get(plz_code, [])
        plz_geom = gdf_plz[gdf_plz["plz"].isin(members)]
        area = plz_geom.geometry.area.sum() / 1e6 if len(plz_geom) > 0 else 10.0
        day_counts = []
        for d_idx in range(N_DAYS):
            gdf_w = daily_gdfs_wgs.get(d_idx)
            if gdf_w is not None:
                day_counts.append(len(gdf_w[gdf_w["plz"] == plz_code]))
        data[plz_code] = {
            "b2c": plz_day_b2c.get(plz_code, {}),
            "b2b": plz_day_b2b.get(plz_code, {}),
            "hub_dist_km": hub_dist,
            "area_km2": area,
            "n_stops_per_day": float(np.mean(day_counts)) if day_counts else 0.0,
            "total_points": len(gdf_dhl[gdf_dhl["plz"] == plz_code]),
        }
    return data
