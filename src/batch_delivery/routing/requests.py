"""VROOM payload builders (jobs, vehicles, scenario requests).

Translates the internal PLZ-level demand representation into VROOM's
JSON job/vehicle schema. The k-means helper splits oversized PLZs into
sub-clusters so VROOM never sees a > ``baseline_job_cap`` request.
"""
from __future__ import annotations

import math
import re
import threading

import numpy as np
import pandas as pd

from batch_delivery.config.constants import (
    AVAILABLE_WORK_S,
    BREAK_DURATION,
    BREAK_WINDOW,
    COST_PER_HOUR_CENTS,
    COST_PER_KM_CENTS,
    DELIVERY_WINDOW,
    FIXED_COST_CENTS,
    LARGE_HUB_TYPES,
    MAX_JOBS_PER_REQUEST,
    MAX_VEHICLES_PER_REQUEST,
    N_DAYS,
    PROFILE,
    SERVICE_TIME_CAP,
    SERVICE_TIME_PER_PARCEL,
    SMALL_HUB_DELAY,
    SPEED_FACTOR,
    VEH_START_LATEST,
    VEH_START_SEED,
    VEH_START_SPREAD_S,
    VEHICLE_CAPACITY,
    VEHICLE_TIME_WINDOW,
    WEEKDAYS,
)
from batch_delivery.utils import log

_print_lock = threading.Lock()

# Cluster key separator — used to split oversized PLZ requests into sub-areas
_CLUSTER_SEP = "__c"

# Module-level k-means clustering cache: (coord_hash, n_clusters) → labels
# Deterministic (fixed seed=42), so same coordinates always produce same split.
_kmeans_cache: dict[tuple[int, int], np.ndarray] = {}




# ─────────────────────────────────────────────────────────────────────────────
# Baseline job caps & spatial k-means splitting
# ─────────────────────────────────────────────────────────────────────────────

def compute_baseline_job_caps(
    baseline_solutions: dict[tuple[int, str], dict],
) -> dict[str, int]:
    """Compute per-PLZ maximum job count observed in the baseline.

    Parameters
    ----------
    baseline_solutions : dict
        ``{(day_idx, plz_code): vroom_solution}`` from a baseline solve.

    Returns
    -------
    dict  ``{plz_code: max_n_jobs}``
    """
    plz_jobs: dict[str, int] = {}
    for (_, req_key), sol in baseline_solutions.items():
        if req_key.startswith("_xpr_"):
            continue
        n_jobs = sum(1 for r in sol.get("routes", [])
                     for s in r.get("steps", []) if s["type"] == "job")
        n_jobs += len(sol.get("unassigned", []))
        plz_jobs[req_key] = max(plz_jobs.get(req_key, 0), n_jobs)
    return plz_jobs


def _split_points_kmeans(
    pts_df: pd.DataFrame,
    n_clusters: int,
    seed: int = 42,
) -> list[pd.DataFrame]:
    """Split delivery points into *n_clusters* spatially balanced groups.

    Uses scipy k-means on (lon, lat) coordinates with iterative
    assignment to produce roughly equal-sized clusters.  Results are
    cached by a coordinate hash for reuse across scenarios.
    """
    from scipy.cluster.vq import kmeans2

    coords = pts_df[["lon", "lat"]].values.astype(np.float64)
    n = len(coords)
    k = min(n_clusters, n)
    if k <= 1:
        return [pts_df]

    # Cache key: hash of coordinate bytes + cluster count
    coord_hash = hash(coords.tobytes())
    cache_key = (coord_hash, k)
    cached_labels = _kmeans_cache.get(cache_key)

    if cached_labels is not None and len(cached_labels) == n:
        balanced_labels = cached_labels
    else:
        # Run k-means for centroids
        rng = np.random.default_rng(seed)
        init_idx = rng.choice(n, size=k, replace=False)
        centroids, labels = kmeans2(coords, coords[init_idx], minit="matrix", iter=20)

        # Balanced reassignment: greedy nearest-centroid with size cap
        target = math.ceil(n / k)
        dists = np.linalg.norm(coords[:, None, :] - centroids[None, :, :], axis=2)
        order = np.argsort(dists.min(axis=1))  # assign closest points first
        balanced_labels = np.full(n, -1, dtype=int)
        counts = np.zeros(k, dtype=int)

        for idx in order:
            ranked = np.argsort(dists[idx])
            for ci in ranked:
                if counts[ci] < target:
                    balanced_labels[idx] = ci
                    counts[ci] += 1
                    break

        _kmeans_cache[cache_key] = balanced_labels
        log.debug("K-means cache MISS: n=%d k=%d → cached", n, k)

    # Build sub-DataFrames
    parts = []
    for ci in range(k):
        mask = balanced_labels == ci
        if mask.any():
            parts.append(pts_df.iloc[mask].copy())
    return parts if parts else [pts_df]


# ─────────────────────────────────────────────────────────────────────────────
# Request building
# ─────────────────────────────────────────────────────────────────────────────

def build_vroom_jobs(pts_df: pd.DataFrame) -> tuple[list[dict], int]:
    """Build VROOM jobs list from a points DataFrame.

    Only points with ``dhl_total >= 1`` produce jobs.  Callers must
    pre-filter to avoid phantom parcels from fractional demand.

    Parameters
    ----------
    pts_df : DataFrame
        Must have columns: ``dhl_total``, ``lon``, ``lat``.

    Returns
    -------
    (jobs, total_demand)
    """
    amounts = pts_df["dhl_total"].values.astype(int)
    lons = pts_df["lon"].values.astype(np.float64)
    lats = pts_df["lat"].values.astype(np.float64)

    valid = amounts > 0
    amounts = amounts[valid]
    lons = np.round(lons[valid], 6)
    lats = np.round(lats[valid], 6)

    total_demand = int(amounts.sum())
    jobs: list[dict] = []

    for i in range(len(amounts)):
        amt = int(amounts[i])
        loc = [float(lons[i]), float(lats[i])]
        if amt > VEHICLE_CAPACITY:
            rem = amt
            while rem > 0:
                chunk = min(rem, VEHICLE_CAPACITY)
                svc = min(chunk * SERVICE_TIME_PER_PARCEL, SERVICE_TIME_CAP)
                jobs.append({
                    "id": len(jobs) + 1, "location": loc,
                    "service": svc, "amount": [chunk],
                    "time_windows": [DELIVERY_WINDOW],
                })
                rem -= chunk
        else:
            svc = min(amt * SERVICE_TIME_PER_PARCEL, SERVICE_TIME_CAP)
            jobs.append({
                "id": len(jobs) + 1, "location": loc,
                "service": svc, "amount": [amt],
                "time_windows": [DELIVERY_WINDOW],
            })
    return jobs, total_demand


def build_vroom_vehicles(
    hub: dict | pd.Series,
    total_demand: int,
    day_idx: int,
    seed_key: str,
    speed_factor: float | None = None,
    n_jobs: int | None = None,
) -> tuple[list[dict], int]:
    """Build VROOM vehicles list with staggered start times.

    Vehicle count is the maximum of capacity-based and time-budget-based
    estimates.  Start time offsets are drawn from a half-normal distribution.

    Parameters
    ----------
    n_jobs : int, optional
        Number of VROOM jobs (stops).  When provided, used for the
        time-budget vehicle estimate.  Defaults to ``total_demand``.
    """
    if n_jobs is None:
        n_jobs = total_demand
    if speed_factor is None:
        speed_factor = SPEED_FACTOR

    # Capacity constraint: ceil(parcels / Q)
    n_veh_cap = max(1, math.ceil(total_demand / VEHICLE_CAPACITY))

    # Time-budget constraint: available driving seconds per vehicle
    # after subtracting break, divided by per-stop time (service +
    # estimated inter-stop driving adjusted for traffic via speed_factor).
    avail_s = AVAILABLE_WORK_S
    avg_svc_s = min(
        (total_demand / max(1, n_jobs)) * SERVICE_TIME_PER_PARCEL,
        SERVICE_TIME_CAP,
    )
    # Rough inter-stop drive: 2 min base, stretched by 1/speed_factor
    avg_drive_s = 120.0 / max(0.1, speed_factor)
    time_per_stop = avg_svc_s + avg_drive_s
    max_stops_per_veh = max(1, int(avail_s / max(1.0, time_per_stop)))
    n_veh_time = max(1, math.ceil(n_jobs / max_stops_per_veh))

    n_veh_needed = max(n_veh_cap, n_veh_time)
    n_veh = min(MAX_VEHICLES_PER_REQUEST, n_veh_needed)
    is_small = hub.get("hub_typ", hub.get("Typ", "")) not in LARGE_HUB_TYPES
    vtw = (
        [VEHICLE_TIME_WINDOW[0] + SMALL_HUB_DELAY, VEHICLE_TIME_WINDOW[1]]
        if is_small
        else list(VEHICLE_TIME_WINDOW)
    )

    rng = np.random.default_rng(
        VEH_START_SEED + day_idx * 1000 + hash(seed_key) % 10000
    )
    max_off = max(0, VEH_START_LATEST - vtw[0])
    offsets = np.abs(rng.normal(0, VEH_START_SPREAD_S, size=n_veh))
    offsets = np.clip(offsets, 0, max_off).astype(int)
    offsets.sort()

    hub_lon = round(float(hub.get("hub_lon", hub.get("lon", 0))), 6)
    hub_lat = round(float(hub.get("hub_lat", hub.get("lat", 0))), 6)

    vehicles = []
    for vid in range(1, n_veh + 1):
        costs = {"fixed": FIXED_COST_CENTS}
        if COST_PER_KM_CENTS > 0:
            costs["per_km"] = COST_PER_KM_CENTS
        if COST_PER_HOUR_CENTS > 0:
            costs["per_hour"] = COST_PER_HOUR_CENTS
        vtw_i = [vtw[0] + int(offsets[vid - 1]), vtw[1]]
        vehicles.append({
            "id": vid, "profile": PROFILE,
            "start": [hub_lon, hub_lat],
            "end": [hub_lon, hub_lat],
            "capacity": [VEHICLE_CAPACITY],
            "time_window": vtw_i,
            "speed_factor": speed_factor,
            "costs": costs,
            "breaks": [{
                "id": 1,
                "time_windows": [BREAK_WINDOW],
                "service": BREAK_DURATION,
                "description": "Lunch break",
            }],
        })
    return vehicles, n_veh


# ─────────────────────────────────────────────────────────────────────────────
# Single-PLZ solver with retry
# ─────────────────────────────────────────────────────────────────────────────

def _parse_unfound_loc(error_text: str) -> tuple[float, float] | None:
    m = re.search(
        r"Unfound route\(s\) (?:from|to) location \[([0-9.]+),([0-9.]+)\]",
        error_text,
    )
    if m:
        return (round(float(m.group(1)), 6), round(float(m.group(2)), 6))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Scenario-level request building
# ─────────────────────────────────────────────────────────────────────────────

def build_scenario_requests(
    schedules: dict[str, set[int]],
    df_assignments: pd.DataFrame,
    daily_gdfs_wgs: dict,
    fast_share: float,
    scenario_name: str,
    merge_source_fn=None,
    speed_factor: float | None = None,
    max_jobs_per_plz: dict[str, int] | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Build day x PLZ VROOM requests for a delivery scenario.

    Hub-bundled express: on non-delivery days, all express parcels from
    PLZ sharing the same hub are combined into one VROOM request.

    Parameters
    ----------
    schedules : dict
        {plz: set of delivery day indices}
    df_assignments : DataFrame
        PLZ → hub mapping.
    daily_gdfs_wgs : dict
        {day_idx: GeoDataFrame} with WGS84 delivery points.
    fast_share : float
        Blended express share (0 for batch-only scenarios).
    scenario_name : str
    merge_source_fn : callable, optional
        Function(source_days, plz_code, daily_gdfs_wgs) → DataFrame.
    speed_factor : float, optional
        Override speed factor (use weighted version after baseline).
    max_jobs_per_plz : dict, optional
        ``{plz: max_jobs}`` from baseline.  When accumulated jobs exceed
        this cap, points are split via spatial k-means into sub-requests.

    Returns
    -------
    (sc_requests, df_summary)
    """
    from batch_delivery.io.demand import get_source_days, merge_source_points

    if merge_source_fn is None:
        merge_source_fn = merge_source_points

    sc_requests: dict[int, dict] = {}
    summary = []

    for day_idx in range(N_DAYS):
        day_name = WEEKDAYS[day_idx]
        day_reqs: dict[str, dict] = {}
        hub_expr: dict[str, dict] = {}

        for plz_code in sorted(schedules.keys()):
            is_delivery_day = day_idx in schedules[plz_code]

            if not is_delivery_day:
                # Express-only: collect for hub bundling
                if fast_share <= 0:
                    continue
                pts = merge_source_fn([day_idx], plz_code, daily_gdfs_wgs)
                if len(pts) == 0:
                    continue
                pts = pts.copy()
                pts["dhl_total"] = (pts["dhl_total"] * fast_share).round().astype(int)
                pts = pts[pts["dhl_total"] > 0]
                if len(pts) == 0:
                    continue
                hub_row = df_assignments[df_assignments["plz"] == plz_code]
                if len(hub_row) == 0:
                    continue
                hn = hub_row.iloc[0]["hub_name"]
                if hn not in hub_expr:
                    hub_expr[hn] = {"hub": hub_row.iloc[0], "pts": [], "plzs": []}
                hub_expr[hn]["pts"].append(pts)
                hub_expr[hn]["plzs"].append(plz_code)
                continue

            # Delivery day: accumulated batch + today's express
            del_sorted = sorted(schedules[plz_code])
            src = get_source_days(day_idx, del_sorted)
            pts = merge_source_fn(src, plz_code, daily_gdfs_wgs)
            if len(pts) == 0:
                continue
            pts = pts.copy()

            # Demand conservation: subtract express already delivered on
            # non-delivery source days so that each parcel is counted
            # exactly once (matches demand.compute_shifted_demand_plz).
            total_raw = pts["dhl_total"].copy().astype(int)

            if fast_share > 0:
                already_sent = pd.Series(0, index=pts.index, dtype=int)
                non_del_src = [
                    d for d in src
                    if d != day_idx and d not in schedules[plz_code]
                ]
                for nd in non_del_src:
                    nd_pts = merge_source_fn([nd], plz_code, daily_gdfs_wgs)
                    if len(nd_pts) == 0:
                        continue
                    nd_map = nd_pts.set_index("str_idx")["dhl_total"].to_dict()
                    already_sent += (
                        fast_share * pts["str_idx"].map(nd_map).fillna(0)
                    ).round().astype(int)
                total_raw = (total_raw - already_sent).clip(lower=0)

            pts["dhl_total"] = total_raw
            pts = pts[pts["dhl_total"] > 0]
            if len(pts) == 0:
                continue

            hub_row = df_assignments[df_assignments["plz"] == plz_code]
            if len(hub_row) == 0:
                continue
            hub = hub_row.iloc[0]

            # ── Check whether spatial splitting is needed ────────────
            cap = (max_jobs_per_plz or {}).get(plz_code, MAX_JOBS_PER_REQUEST)
            if len(pts) > cap:
                n_clusters = math.ceil(len(pts) / cap)
                clusters = _split_points_kmeans(pts, n_clusters)
                for ci, c_pts in enumerate(clusters):
                    c_jobs, c_demand = build_vroom_jobs(c_pts)
                    if not c_jobs:
                        continue
                    c_vehs, c_nv = build_vroom_vehicles(
                        hub, c_demand, day_idx,
                        f"{plz_code}{_CLUSTER_SEP}{ci}",
                        speed_factor=speed_factor,
                        n_jobs=len(c_jobs),
                    )
                    c_key = f"{plz_code}{_CLUSTER_SEP}{ci}"
                    day_reqs[c_key] = {"vehicles": c_vehs, "jobs": c_jobs}
                    summary.append({
                        "day_idx": day_idx, "day": day_name,
                        "plz": plz_code, "hub": hub["hub_name"],
                        "hub_typ": hub["hub_typ"],
                        "is_express_only": False,
                        "n_jobs": len(c_jobs), "total_parcels": c_demand,
                        "n_vehicles": c_nv,
                    })
            else:
                jobs, total_demand = build_vroom_jobs(pts)
                if not jobs:
                    continue
                vehicles, n_veh = build_vroom_vehicles(
                    hub, total_demand, day_idx, plz_code,
                    speed_factor=speed_factor,
                    n_jobs=len(jobs),
                )
                day_reqs[plz_code] = {"vehicles": vehicles, "jobs": jobs}
                summary.append({
                    "day_idx": day_idx, "day": day_name,
                    "plz": plz_code, "hub": hub["hub_name"],
                    "hub_typ": hub["hub_typ"],
                    "is_express_only": False,
                    "n_jobs": len(jobs), "total_parcels": total_demand,
                    "n_vehicles": n_veh,
                })

        # Hub-bundled express requests (with k-means splitting for large hubs)
        for hn, hinfo in hub_expr.items():
            hub = hinfo["hub"]
            all_pts = pd.concat(hinfo["pts"], ignore_index=True)
            plz_label = "|".join(sorted(hinfo["plzs"]))

            if len(all_pts) > MAX_JOBS_PER_REQUEST:
                n_clusters = math.ceil(len(all_pts) / MAX_JOBS_PER_REQUEST)
                clusters = _split_points_kmeans(all_pts, n_clusters)
                for ci, c_pts in enumerate(clusters):
                    c_jobs, c_demand = build_vroom_jobs(c_pts)
                    if not c_jobs:
                        continue
                    c_vehs, c_nv = build_vroom_vehicles(
                        hub, c_demand, day_idx, hn, speed_factor=speed_factor,
                        n_jobs=len(c_jobs),
                    )
                    c_key = f"_xpr_{hn}{_CLUSTER_SEP}{ci}"
                    day_reqs[c_key] = {"vehicles": c_vehs, "jobs": c_jobs}
                    summary.append({
                        "day_idx": day_idx, "day": day_name,
                        "plz": plz_label,
                        "hub": hn, "hub_typ": hub["hub_typ"],
                        "is_express_only": True,
                        "n_jobs": len(c_jobs), "total_parcels": c_demand,
                        "n_vehicles": c_nv,
                    })
            else:
                xpr_jobs, xpr_demand = build_vroom_jobs(all_pts)
                if not xpr_jobs:
                    continue
                xpr_vehs, xpr_nv = build_vroom_vehicles(
                    hub, xpr_demand, day_idx, hn, speed_factor=speed_factor,
                    n_jobs=len(xpr_jobs),
                )
                day_reqs[f"_xpr_{hn}"] = {"vehicles": xpr_vehs, "jobs": xpr_jobs}
                summary.append({
                    "day_idx": day_idx, "day": day_name,
                    "plz": plz_label,
                    "hub": hn, "hub_typ": hub["hub_typ"],
                    "is_express_only": True,
                    "n_jobs": len(xpr_jobs), "total_parcels": xpr_demand,
                    "n_vehicles": xpr_nv,
                })

        sc_requests[day_idx] = day_reqs

    df_r = pd.DataFrame(summary)
    active_days = sum(1 for d in range(N_DAYS) if sc_requests.get(d))
    n_express = int(df_r["is_express_only"].sum()) if len(df_r) > 0 else 0
    n_split = sum(
        1 for d in sc_requests.values()
        for k in d if _CLUSTER_SEP in k
    )
    split_info = f", {n_split} cluster sub-requests" if n_split else ""
    log.debug(
        f"[{scenario_name}] {active_days} active days, "
        f"{len(df_r)} requests ({n_express} hub-bundled express{split_info}), "
        f'{df_r["n_jobs"].sum():,} jobs, '
        f'{df_r["total_parcels"].sum():,} parcels'
        if len(df_r) > 0 else f"[{scenario_name}] no requests"
    )
    return sc_requests, df_r
