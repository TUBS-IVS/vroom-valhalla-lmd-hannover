"""High-level VROOM solve interface.

* :func:`solve_single_plz` — one PLZ, one scenario, with cache check
  and retry/restart logic.
* :func:`solve_scenario`   — fan-out over all (provider, PLZ) cells of
  a scenario.
* :func:`parse_routes`     — convert the VROOM solution into the
  per-route DataFrame used downstream.
"""
from __future__ import annotations

import copy
import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests as http_requests
from tqdm.auto import tqdm

from batch_delivery.config.constants import (
    CONNECTION_RETRIES,
    HTTP_CONNECT_TIMEOUT,
    HTTP_TIMEOUT,
    MAX_RETRIES,
    MAX_UNASSIGNED_RETRIES,
    N_DAYS,
    PER_JOB_BUDGET_S,
    SOLVE_WORKERS,
    VEHICLE_CAPACITY,
    VROOM_API_URL,
    WEEKDAYS,
)
from batch_delivery.utils import log

_print_lock = threading.Lock()

# Cluster key separator — used to split oversized PLZ requests into sub-areas
_CLUSTER_SEP = "__c"

# Module-level k-means clustering cache: (coord_hash, n_clusters) → labels
# Deterministic (fixed seed=42), so same coordinates always produce same split.
_kmeans_cache: dict[tuple[int, int], np.ndarray] = {}

from batch_delivery.routing.cache import (
    load_cached_solution,
    save_cached_solution,
)
from batch_delivery.routing.client import (
    _MAX_RESTART_ATTEMPTS,
    _check_valhalla_memory,
    _get_container_mem_mb,
    _health_check,
    _restart_vroom,
)
from batch_delivery.routing.requests import (
    _parse_unfound_loc,
)


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
