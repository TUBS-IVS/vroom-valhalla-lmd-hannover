"""VROOM VRP routing: request building, solving, hash caching, route parsing."""

import copy
import hashlib
import json
import math
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests as http_requests
from tqdm.auto import tqdm

from batch_delivery.config.constants import (
    N_DAYS, WEEKDAYS,
    VEHICLE_CAPACITY, FIXED_COST_CENTS, COST_PER_KM_CENTS, COST_PER_HOUR_CENTS,
    COST_SCALE, SERVICE_TIME_PER_PARCEL, SERVICE_TIME_CAP, PROFILE,
    VROOM_API_URL, DELIVERY_WINDOW, VEHICLE_TIME_WINDOW,
    BREAK_DURATION, BREAK_WINDOW,
    VEH_START_SPREAD_S, VEH_START_LATEST, VEH_START_SEED,
    LARGE_HUB_TYPES, SMALL_HUB_DELAY,
    MAX_VEHICLES_PER_REQUEST, SOLVE_WORKERS, HTTP_TIMEOUT,
    HTTP_CONNECT_TIMEOUT, PER_JOB_BUDGET_S,
    MAX_RETRIES, CONNECTION_RETRIES, MAX_UNASSIGNED_RETRIES,
    SPEED_FACTOR, RESULTS_DIR,
    MAX_JOBS_PER_REQUEST, AVAILABLE_WORK_S,
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
# SHA-256 hash caching
# ─────────────────────────────────────────────────────────────────────────────

def _request_hash(request_body: dict) -> str:
    """Deterministic SHA-256 hash of a VROOM request."""
    canonical = json.dumps(request_body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _cache_path(tag: str) -> Path:
    d = RESULTS_DIR / "cache" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_cached_solution(tag: str, request_body: dict) -> dict | None:
    """Return cached VROOM solution or None."""
    h = _request_hash(request_body)
    p = _cache_path(tag) / f"{h}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def save_cached_solution(tag: str, request_body: dict, solution: dict) -> None:
    """Persist a VROOM solution to disk cache."""
    h = _request_hash(request_body)
    p = _cache_path(tag) / f"{h}.json"
    p.write_text(json.dumps(solution), encoding="utf-8")


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


def solve_single_plz(
    plz_code: str,
    request_body: dict,
    idx: int, total: int,
    day_name: str = "",
    cache_tag: str | None = None,
) -> dict:
    """Solve a single PLZ VRP with retry logic (thread-safe).

    If *cache_tag* is given, attempts to load a cached result first and
    stores the solution after a successful solve.
    """
    # Try cache first (skip if cached solution has unassigned jobs)
    if cache_tag:
        cached = load_cached_solution(cache_tag, request_body)
        if cached is not None and len(cached.get("unassigned", [])) == 0:
            n_routes = len(cached.get("routes", []))
            with _print_lock:
                log.debug(f"[{idx:3d}/{total}] PLZ {plz_code}: cache hit ({n_routes} routes)")
            start_load = 0
            for route in cached.get("routes", []):
                for s in route.get("steps", []):
                    if s["type"] == "start":
                        start_load += s.get("load", [0])[0]
            return {
                "plz": plz_code,
                "n_jobs_original": len(request_body.get("jobs", [])),
                "n_jobs": len(request_body.get("jobs", [])),
                "jobs_removed": 0,
                "parcels_removed": 0,
                "n_vehicles": len(request_body.get("vehicles", [])),
                "total_parcels": sum(j["amount"][0] for j in request_body.get("jobs", [])),
                "status": "CACHED",
                "attempts": 0,
                "solve_time_s": 0.0,
                "vroom_internal_ms": None,
                "n_routes": n_routes,
                "n_unassigned": 0,
                "cost": cached.get("summary", {}).get("cost"),
                "duration_s": cached.get("summary", {}).get("duration"),
                "distance_m": cached.get("summary", {}).get("distance"),
                "solution": cached,
            }

    rb = copy.deepcopy(request_body)
    session = http_requests.Session()

    n_jobs_original = len(rb["jobs"])
    n_vehicles = len(rb["vehicles"])
    total_parcels = sum(j["amount"][0] for j in rb["jobs"])

    attempt = 0
    conn_retries = 0
    jobs_removed = 0
    parcels_removed = 0  # parcels dropped via unfound-location retries (B6 fix)
    status = None
    solve_time_s = 0.0
    n_routes = n_unassigned = sol_cost = sol_duration = sol_distance = vroom_internal_ms = None
    solution = None
    job_t0 = time.perf_counter()

    while attempt <= MAX_RETRIES:
        attempt += 1
        if not rb["jobs"]:
            status = "NO_JOBS"
            break
        if time.perf_counter() - job_t0 > PER_JOB_BUDGET_S:
            with _print_lock:
                log.warning(
                    f"[{idx:3d}/{total}] PLZ {plz_code}: per-job budget "
                    f"({PER_JOB_BUDGET_S}s) exhausted -- giving up"
                )
            status = "BUDGET_EXCEEDED"
            break

        t0 = time.perf_counter()
        try:
            response = session.post(
                VROOM_API_URL, json=rb,
                headers={"Content-Type": "application/json"},
                timeout=(HTTP_CONNECT_TIMEOUT, HTTP_TIMEOUT),
            )
            solve_time_s = round(time.perf_counter() - t0, 4)

            if response.status_code == 200:
                solution = response.json()
                n_routes = len(solution.get("routes", []))
                n_unassigned = len(solution.get("unassigned", []))
                sol_cost = solution.get("summary", {}).get("cost")
                sol_duration = solution.get("summary", {}).get("duration")
                sol_distance = solution.get("summary", {}).get("distance")
                ct = solution.get("summary", {}).get("computing_times", {})
                vroom_internal_ms = (
                    ct.get("loading", 0) + ct.get("solving", 0) + ct.get("routing", 0)
                )

                # ── Auto-add vehicles when VROOM leaves jobs unassigned ──
                ua_retry = 0
                while n_unassigned > 0 and ua_retry < MAX_UNASSIGNED_RETRIES:
                    ua_retry += 1
                    ua_demand = sum(
                        j.get("amount", [1])[0]
                        for j in solution.get("unassigned", [])
                    )
                    extra = max(1, math.ceil(ua_demand / VEHICLE_CAPACITY))
                    max_id = max(v["id"] for v in rb["vehicles"])
                    tmpl = rb["vehicles"][-1]
                    for i in range(extra):
                        v = copy.deepcopy(tmpl)
                        v["id"] = max_id + i + 1
                        rb["vehicles"].append(v)
                    with _print_lock:
                        log.info(
                            f"[{idx:3d}/{total}] PLZ {plz_code}: "
                            f"{n_unassigned} unassigned ({ua_demand} parcels) "
                            f"→ +{extra} vehicles (retry {ua_retry})"
                        )
                    if time.perf_counter() - job_t0 > PER_JOB_BUDGET_S:
                        with _print_lock:
                            log.warning(
                                f"[{idx:3d}/{total}] PLZ {plz_code}: per-job budget "
                                f"({PER_JOB_BUDGET_S}s) exhausted during UA-retry -- accepting partial"
                            )
                        break
                    t0_ua = time.perf_counter()
                    try:
                        resp_ua = session.post(
                            VROOM_API_URL, json=rb,
                            headers={"Content-Type": "application/json"},
                            timeout=(HTTP_CONNECT_TIMEOUT, HTTP_TIMEOUT),
                        )
                        solve_time_s += round(time.perf_counter() - t0_ua, 4)
                        if resp_ua.status_code == 200:
                            solution = resp_ua.json()
                            n_routes = len(solution.get("routes", []))
                            n_unassigned = len(solution.get("unassigned", []))
                            sol_cost = solution.get("summary", {}).get("cost")
                            sol_duration = solution.get("summary", {}).get("duration")
                            sol_distance = solution.get("summary", {}).get("distance")
                            ct = solution.get("summary", {}).get("computing_times", {})
                            vroom_internal_ms = (
                                ct.get("loading", 0) + ct.get("solving", 0)
                                + ct.get("routing", 0)
                            )
                        else:
                            break
                    except http_requests.exceptions.RequestException:
                        break

                n_vehicles = len(rb["vehicles"])
                status = "OK" if (n_unassigned or 0) == 0 else "PARTIAL"
                # FIX 2026-05-25 (Audit D B5): do NOT cache solutions that
                # still have unassigned jobs (parcels would be silently lost
                # on cache hit). Cache hits in load_cached_solution skip such
                # entries — match that behavior on write.
                if cache_tag and (n_unassigned or 0) == 0 and solution is not None:
                    save_cached_solution(cache_tag, request_body, solution)
                elif (n_unassigned or 0) > 0:
                    with _print_lock:
                        log.warning(
                            f"[{idx:3d}/{total}] PLZ {plz_code}: solution has "
                            f"{n_unassigned} unassigned jobs — NOT caching"
                        )

                extra_info = f", +{ua_retry} veh retries" if ua_retry else ""
                retry_info = f" ({jobs_removed} removed)" if jobs_removed else ""
                prefix = f"{day_name} " if day_name else ""
                with _print_lock:
                    log.debug(
                        f"[{idx:3d}/{total}] {prefix}PLZ {plz_code}: "
                        f"{len(rb['jobs'])} jobs -> {n_routes} routes, "
                        f"{n_unassigned} ua | {solve_time_s:.1f}s"
                        f"{retry_info}{extra_info}"
                    )
                break
            else:
                error_text = response.text
                bad_loc = _parse_unfound_loc(error_text)
                if bad_loc and attempt <= MAX_RETRIES:
                    lon, lat = bad_loc
                    before = len(rb["jobs"])
                    # FIX 2026-05-25 (Audit D B6): track parcels dropped from
                    # unfound-location jobs so KPI/conservation checks can see
                    # what was silently removed. Previously only job counts.
                    dropped_jobs = [
                        j for j in rb["jobs"]
                        if (
                            round(j["location"][0], 6) == lon
                            and round(j["location"][1], 6) == lat
                        )
                    ]
                    parcels_dropped_now = sum(
                        int(j.get("amount", [0])[0]) for j in dropped_jobs
                    )
                    rb["jobs"] = [
                        j for j in rb["jobs"]
                        if not (
                            round(j["location"][0], 6) == lon
                            and round(j["location"][1], 6) == lat
                        )
                    ]
                    removed = before - len(rb["jobs"])
                    jobs_removed += removed
                    parcels_removed += parcels_dropped_now
                    if parcels_dropped_now > 0:
                        log.warning(
                            "Unfound location at (%.6f, %.6f): dropped %d jobs / %d parcels",
                            lon, lat, removed, parcels_dropped_now,
                        )
                    for new_id, job in enumerate(rb["jobs"], start=1):
                        job["id"] = new_id
                    continue

                # HTTP 500 without unfound location: retry once after brief pause
                if response.status_code == 500 and conn_retries == 0:
                    conn_retries += 1
                    time.sleep(5)
                    attempt -= 1
                    continue

                status = f"HTTP_{response.status_code}"
                with _print_lock:
                    log.debug(f"[{idx:3d}/{total}] PLZ {plz_code}: HTTP {response.status_code}")
                break

        except (
            http_requests.exceptions.ConnectionError,
            http_requests.exceptions.ReadTimeout,
        ):
            solve_time_s = round(time.perf_counter() - t0, 4)
            conn_retries += 1
            if conn_retries <= CONNECTION_RETRIES:
                time.sleep(10)
                attempt -= 1
                continue
            status = "CONN_ERROR"
            with _print_lock:
                log.debug(f"[{idx:3d}/{total}] PLZ {plz_code}: connection error")
            break

        except http_requests.exceptions.RequestException as e:
            solve_time_s = round(time.perf_counter() - t0, 4)
            status = "ERROR"
            with _print_lock:
                log.debug(f"[{idx:3d}/{total}] PLZ {plz_code}: {type(e).__name__}")
            break

    session.close()

    return {
        "plz": plz_code,
        "n_jobs_original": n_jobs_original,
        "n_jobs": len(rb["jobs"]),
        "jobs_removed": jobs_removed,
        "parcels_removed": parcels_removed,
        "n_vehicles": n_vehicles,
        "total_parcels": total_parcels,
        "status": status,
        "attempts": attempt,
        "solve_time_s": solve_time_s,
        "vroom_internal_ms": vroom_internal_ms,
        "n_routes": n_routes,
        "n_unassigned": n_unassigned,
        "cost": sol_cost,
        "duration_s": sol_duration,
        "distance_m": sol_distance,
        "solution": solution,
    }


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
    from batch_delivery.io.demand import merge_source_points, get_source_days

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


# ─────────────────────────────────────────────────────────────────────────────
# Scenario solver
# ─────────────────────────────────────────────────────────────────────────────

def _health_check() -> bool:
    """Quick VROOM health check with a trivial request."""
    try:
        r = http_requests.post(
            VROOM_API_URL,
            json={
                "vehicles": [{"id": 1, "profile": "auto",
                              "start": [9.73, 52.38], "end": [9.73, 52.38],
                              "capacity": [10]}],
                "jobs": [{"id": 1, "location": [9.74, 52.37],
                          "amount": [1], "service": 60}],
            },
            timeout=30,
        )
        return r.status_code == 200
    except Exception:
        return False


_MAX_RESTART_ATTEMPTS = 2
_VALHALLA_MEM_LIMIT_MB = 20_000  # Restart Valhalla when exceeding this (host: 64 GB)


def _get_container_mem_mb(container: str) -> float:
    """Query Docker for a container's current memory usage in MiB."""
    try:
        r = subprocess.run(
            ["docker", "stats", container, "--no-stream",
             "--format", "{{.MemUsage}}"],
            capture_output=True, text=True, timeout=10,
        )
        mem_str = r.stdout.strip().split("/")[0].strip()
        if "GiB" in mem_str:
            return float(mem_str.replace("GiB", "").strip()) * 1024
        elif "MiB" in mem_str:
            return float(mem_str.replace("MiB", "").strip())
        return 0.0
    except Exception:
        return 0.0


def _restart_container(name: str, timeout: int = 180) -> bool:
    """Restart a Docker container and wait until it is running."""
    log.warning(f"Restarting {name} container …")
    for cmd in (
        ["docker", "restart", name],
        ["docker-compose", "restart", name],
    ):
        try:
            subprocess.run(cmd, capture_output=True, timeout=60)
            break
        except Exception:
            continue
    else:
        log.error(f"docker restart {name} failed")
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(10)
        try:
            r = subprocess.run(
                ["docker", "inspect", name,
                 "--format", "{{.State.Running}}"],
                capture_output=True, text=True, timeout=10,
            )
            if r.stdout.strip().lower() == "true":
                log.warning(f"{name} container restarted successfully")
                return True
        except Exception:
            pass
    log.error(f"{name} did not recover within {timeout}s")
    return False


def _restart_vroom(timeout: int = 120) -> bool:
    """Restart the VROOM Docker container and wait until healthy."""
    if not _restart_container("vroom", timeout):
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(5)
        if _health_check():
            log.warning("VROOM recovered after restart")
            return True
    log.error("VROOM did not recover within timeout")
    return False


def _check_valhalla_memory() -> bool:
    """Check Valhalla memory; restart if exceeding limit. Returns True if OK."""
    mem = _get_container_mem_mb("valhalla")
    if mem <= 0:
        return True  # Can't read — assume OK
    if mem > _VALHALLA_MEM_LIMIT_MB:
        log.warning(
            f"Valhalla memory {mem:.0f} MiB exceeds "
            f"{_VALHALLA_MEM_LIMIT_MB} MiB limit — restarting"
        )
        if not _restart_container("valhalla", timeout=180):
            return False
        # Valhalla needs time to reload tiles
        time.sleep(30)
        # VROOM may also need restart after Valhalla restart
        if not _health_check():
            _restart_container("vroom", timeout=60)
            time.sleep(10)
        return _health_check()
    return True


def solve_scenario(
    scenario_requests: dict,
    scenario_name: str,
    cache_tag: str | None = None,
    save_intermediate: Path | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Solve all VRP requests for a scenario.

    Parameters
    ----------
    scenario_requests : dict
        {day_idx: {plz_or_key: {vehicles, jobs}}}
    scenario_name : str
    cache_tag : str, optional
        If given, use SHA-256 hash caching.
    save_intermediate : Path, optional
        If given, save per-day solutions to this directory.

    Returns
    -------
    (solutions, df_solve)
    """
    all_res = []
    solutions: dict[tuple[int, str], dict] = {}
    t_start = time.perf_counter()
    restarts_used = 0

    for day_idx in range(N_DAYS):
        day_reqs = scenario_requests.get(day_idx, {})
        if not day_reqs:
            continue
        day_name = WEEKDAYS[day_idx]

        # ── Pre-day health check with auto-recovery ──────────────
        if not _health_check():
            # Try Valhalla first — most likely culprit
            log.warning(f"[{scenario_name}] {day_name}: VROOM health check failed, checking Valhalla …")
            if not _check_valhalla_memory():
                log.error(f"[{scenario_name}] {day_name}: Valhalla unrecoverable")
            if not _health_check():
                if restarts_used < _MAX_RESTART_ATTEMPTS and _restart_vroom():
                    restarts_used += 1
                else:
                    log.error(
                        f"[{scenario_name}] {day_name}: VROOM unreachable — "
                        f"skipping remaining days"
                    )
                    break

        # ── Pre-day Valhalla memory check ────────────────────────
        _check_valhalla_memory()

        req_keys = sorted(
            day_reqs.keys(),
            key=lambda p: len(day_reqs[p]["jobs"]),
            reverse=True,
        )
        n_reqs = len(req_keys)
        day_t0 = time.perf_counter()
        log.debug(f"[{scenario_name}] === {day_name}: {n_reqs} requests ===")

        day_res = []
        total_jobs_day = sum(len(day_reqs[rk]["jobs"]) for rk in req_keys)
        pbar = tqdm(
            total=n_reqs, desc=f"{scenario_name} {day_name}",
            unit="plz", leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} PLZ [{elapsed}<{remaining}, {rate_fmt}]",
        )
        with ThreadPoolExecutor(max_workers=SOLVE_WORKERS) as exe:
            futs = {}
            for idx, rk in enumerate(req_keys, 1):
                f = exe.submit(
                    solve_single_plz, rk, day_reqs[rk],
                    idx, n_reqs,
                    f"{scenario_name} {day_name}",
                    cache_tag=cache_tag,
                )
                futs[f] = rk
            for f in as_completed(futs):
                r = f.result()
                r["day_idx"] = day_idx
                r["day"] = day_name
                if r["solution"] is not None:
                    solutions[(day_idx, futs[f])] = r["solution"]
                day_res.append(r)
                pbar.update(1)
        pbar.close()

        # ── Mid-day crash detection: >50 % failures → restart + retry ──
        n_fail = sum(1 for r in day_res if r["status"] not in ("OK", "CACHED"))
        if n_fail > n_reqs * 0.5 and n_reqs >= 4 and restarts_used < _MAX_RESTART_ATTEMPTS:
            failed_keys = [r["plz"] for r in day_res if r["status"] not in ("OK", "CACHED")]
            if _restart_vroom():
                restarts_used += 1
                log.warning(
                    f"[{scenario_name}] {day_name}: {n_fail}/{n_reqs} failed — "
                    f"retrying {len(failed_keys)} after restart"
                )
                retry_res = []
                pbar2 = tqdm(
                    total=len(failed_keys), desc=f"{scenario_name} {day_name} retry",
                    unit="plz", leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} PLZ [{elapsed}<{remaining}, {rate_fmt}]",
                )
                with ThreadPoolExecutor(max_workers=SOLVE_WORKERS) as exe:
                    futs2 = {}
                    for idx, rk in enumerate(failed_keys, 1):
                        f = exe.submit(
                            solve_single_plz, rk, day_reqs[rk],
                            idx, len(failed_keys),
                            f"{scenario_name} {day_name}",
                            cache_tag=cache_tag,
                        )
                        futs2[f] = rk
                    for f in as_completed(futs2):
                        r = f.result()
                        r["day_idx"] = day_idx
                        r["day"] = day_name
                        if r["solution"] is not None:
                            solutions[(day_idx, futs2[f])] = r["solution"]
                        retry_res.append(r)
                        pbar2.update(1)
                pbar2.close()
                # Replace failed results with retry results
                ok_from_first = [r for r in day_res if r["status"] in ("OK", "CACHED")]
                day_res = ok_from_first + retry_res

        # Save intermediate results
        if save_intermediate:
            save_intermediate.mkdir(parents=True, exist_ok=True)
            for r in day_res:
                if r.get("solution"):
                    safe_plz = re.sub(r'[\\/:*?"<>|]', "_", r["plz"])
                    safe_sc = re.sub(r'[\\/:*?"<>|]', "_", scenario_name)
                    p = save_intermediate / f"{safe_sc}_{day_name}_{safe_plz}.json"
                    p.write_text(json.dumps(r["solution"]), encoding="utf-8")

        for r in day_res:
            r.pop("solution", None)
        all_res.extend(day_res)

        day_el = round(time.perf_counter() - day_t0, 1)
        day_ok = sum(1 for r in day_res if r["status"] in ("OK", "CACHED"))
        day_fail = len(day_res) - day_ok
        day_rt = sum(r.get("n_routes", 0) or 0 for r in day_res
                     if r["status"] in ("OK", "CACHED"))
        val_mem = _get_container_mem_mb("valhalla")
        mem_info = f", valhalla={val_mem:.0f}MB" if val_mem > 0 else ""
        # Professional per-day summary (WARNING only when failures)
        if day_fail > 0:
            fail_statuses = {}
            for r in day_res:
                if r["status"] not in ("OK", "CACHED"):
                    fail_statuses[r["status"]] = fail_statuses.get(r["status"], 0) + 1
            status_str = ", ".join(f"{v}× {k}" for k, v in fail_statuses.items())
            log.warning(
                f"[{scenario_name}] {day_name}: {day_ok}/{n_reqs} OK, "
                f"{day_fail} failed ({status_str}), {day_el:.0f}s{mem_info}"
            )
        else:
            log.debug(
                f"[{scenario_name}] {day_name}: {day_ok}/{n_reqs} OK, "
                f"{day_rt} routes, {day_el:.0f}s{mem_info}"
            )
        if day_idx < N_DAYS - 1:
            time.sleep(0.5)  # brief pause between days for service stability

    all_res.sort(key=lambda x: (x.get("day_idx", 0), x.get("plz", "")))
    df_s = pd.DataFrame(all_res)
    el = round(time.perf_counter() - t_start, 1)
    log.debug(f"[{scenario_name}] Completed in {el / 60:.1f} min")
    return solutions, df_s


# ─────────────────────────────────────────────────────────────────────────────
# Route parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_routes(
    solutions: dict[tuple[int, str], dict],
    scenario_name: str,
    df_assignments: pd.DataFrame,
    provider_name: str | None = None,
) -> pd.DataFrame:
    """Parse VROOM solutions into a per-route DataFrame.

    Parameters
    ----------
    solutions : dict
        {(day_idx, req_key): vroom_solution_dict}
    scenario_name : str
    df_assignments : DataFrame
        PLZ → hub mapping.

    Returns
    -------
    DataFrame with one row per route.
    """
    rows = []
    for (day_idx, req_key), sol in solutions.items():
        day_name = WEEKDAYS[day_idx] if day_idx < len(WEEKDAYS) else f"Day{day_idx}"
        is_express = req_key.startswith("_xpr_")

        # Strip cluster suffix for PLZ lookup (e.g. "30855__c0" → "30855")
        base_key = req_key.split(_CLUSTER_SEP)[0] if _CLUSTER_SEP in req_key else req_key

        if is_express:
            hub_name = base_key[5:]  # strip "_xpr_" prefix (and cluster suffix already removed)
            hr = df_assignments[df_assignments["hub_name"] == hub_name]
            hub_typ = hr.iloc[0]["hub_typ"] if len(hr) > 0 else "Unknown"
            plz_code = req_key
        else:
            plz_code = base_key
            hr = df_assignments[df_assignments["plz"] == plz_code]
            hub_name = hr.iloc[0]["hub_name"] if len(hr) > 0 else "Unknown"
            hub_typ = hr.iloc[0]["hub_typ"] if len(hr) > 0 else "Unknown"

        for route in sol.get("routes", []):
            steps = route.get("steps", [])
            n_stops = sum(1 for s in steps if s["type"] == "job")
            start_load = 0
            for s in steps:
                if s["type"] == "start":
                    start_load = s.get("load", [0])[0]
                    break
            st = steps[0].get("arrival", 0) if steps else 0
            et = steps[-1].get("arrival", 0) if steps else 0
            rows.append({
                "provider": provider_name or "All",
                "scenario": scenario_name,
                "day_idx": day_idx, "day": day_name,
                "plz": plz_code, "hub": hub_name, "hub_typ": hub_typ,
                "is_express": is_express,
                "vehicle_id": route["vehicle"],
                "n_stops": n_stops, "parcels": start_load,
                "distance_km": round(route.get("distance", 0) / 1000, 2),
                "travel_h": round(route.get("duration", 0) / 3600, 3),
                "service_h": round(route.get("service", 0) / 3600, 3),
                "waiting_h": round(route.get("waiting_time", 0) / 3600, 3),
                "total_h": round(
                    (route.get("duration", 0) + route.get("service", 0)
                     + route.get("waiting_time", 0)) / 3600, 3
                ),
                "start_time": st, "end_time": et,
                "load_factor": round(start_load / VEHICLE_CAPACITY, 3),
                "cost": route.get("cost", 0),
            })
    return pd.DataFrame(rows)
