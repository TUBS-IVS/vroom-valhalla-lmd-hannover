"""64: VROOM bundle pool — the training labels for the bundle head.

Routes the tours the realistic-tour rule actually forms (Task 8's manifest,
selected down to a budgeted stratified core by ``64a_bundle_coverage.py``)
through the SAME VROOM configuration the surrogate's own training pool was
built with, and writes one labelled row per bundle.

LABEL CONSISTENCY — the non-negotiable part
-------------------------------------------
The bundle head shares the alpha-Daganzo backbone (alpha = 1.343) calibrated
on VROOM labels produced by ``batch_delivery.sweep.runner``. A bundle label
that came out of a different VROOM configuration would be a systematic offset
the head learns as fact. So the request is built by the SAME helpers, called
the same way:

    jobs, total_demand = build_vroom_jobs(pts)                    # sweep :224
    vehicles, n_veh    = build_vroom_vehicles(hub=..., total_demand=...,
                                              day_idx=..., seed_key=...,
                                              n_jobs=len(jobs))   # sweep :229
    result             = solve_single_plz(plz_code=..., request_body=...,
                                          idx=..., total=..., day_name=...,
                                          cache_tag=...)          # sweep :240

``seed_key`` goes to ``build_vroom_vehicles`` ONLY and ``cache_tag`` to
``solve_single_plz`` ONLY — the archived path2 script passed ``seed_key`` to
the solver and failed on every row. Both are asserted against the live
signatures at import (see ``_assert_helper_contract``), so the mistake cannot
come back silently. No constant is overridden anywhere in this script:
capacity, vehicle-count rule, costs, time windows, service times, retry and
unassigned-retry policy are all whatever ``config.constants`` says, exactly as
for the sweep.

INSTANCE CONSTRUCTION — mirrors ``20_validate_vroom_smoothed.py:140-200``
-------------------------------------------------------------------------
One VROOM request per bundle: the bundle is ONE tour group, so its members'
points are concatenated into a single instance (never split — a k-means split
would price a different object than the one the head is asked about).

  * ``delivery`` kind — the members' consolidated demand at THEIR schedules on
    that day. Per member: ``get_source_days(day, sched)`` frames; a HELD
    (non-delivery) source day contributes only the batched portion, the
    express being subtracted at the PLZ-day AGGREGATE level
    (``total - round(total * fs)``, separately for b2c/b2b) and re-spread over
    stops by ``_allocate_to_target`` — 20_'s exact treatment, which is what
    ``build_cost_matrices_ml``'s ``shifted_dd`` computes.
  * ``express`` kind — the members' day-``d`` points scaled TO the standard
    share (``round(total * fs)``, the complement of the above), which is
    exactly ``raw_express[z, d]``.

Each member's schedule comes from the deployed choice
``schedule_idx_system_smoothed`` at the bundle's ``first_seen`` (P, theta),
read from a COPY of the live ``_tab_chosen_v2.csv`` (never the live file — a
reader lock killed a grid writer overnight before 62_'s copy-first pattern).
The instance's summed demand is asserted equal to the manifest's ``parcels``
within rounding; a mismatch is logged and the row is kept and flagged.

OPERATIONS
----------
Throughput is measured before committing to a setting: 10 instances
single-threaded, then 2- and 3-concurrent batches on FRESH instances (re-using
the same ones would measure the solution cache). The fastest stable setting
wins and ``N_A = min(400, floor(8h / t_median * parallelism))``. The probe
solves are real Phase-A labels, not throwaway.

Resumable by ``bundle_id``; append-only CSV with the retry-backoff writer from
20_ (a morning AV/sync lock once cost hours of progress); parquet rebuilt from
the CSV periodically. Per-instance wall cap of 15 min marks ``TIMEOUT`` and
moves on — the straggler thread keeps running and its real result is written
later as a late completion, so no work is thrown away. ``PARTIAL`` rows are
kept and flagged, never dropped (Kompendium 38.2b). Rows in = rows out.

Run:
    .venv\\Scripts\\python.exe scripts/revision/64_solve_bundles_vroom.py
    .venv\\Scripts\\python.exe scripts/revision/64_solve_bundles_vroom.py --throughput-only
    .venv\\Scripts\\python.exe scripts/revision/64_solve_bundles_vroom.py --report-only

Output (results/revision_2026_08/bundles/):
    bundles_solved.parquet/.csv   one row per solved bundle
    bundles_throughput.json       the measurement N_A was derived from
    bundles_pool_report.md        pool quality report (feeds Task 10)
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect
import json
import os
import shutil
import sys
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _stage3_common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "src"))
from batch_delivery.config.constants import (  # noqa: E402
    AVAILABLE_WORK_S, BREAK_DURATION, BREAK_WINDOW, COST_PER_HOUR_CENTS,
    COST_PER_KM_CENTS, COST_SCALE, DELIVERY_WINDOW, FIXED_COST_CENTS,
    MAX_JOBS_PER_REQUEST, MAX_RETRIES, MAX_UNASSIGNED_RETRIES,
    MAX_VEHICLES_PER_REQUEST, PER_JOB_BUDGET_S, PROFILE, SERVICE_TIME_CAP,
    SERVICE_TIME_PER_PARCEL, SPEED_FACTOR, VEHICLE_CAPACITY,
    VEHICLE_TIME_WINDOW, VROOM_API_URL, WEEKDAYS as CDAYS,
    provider_to_demand_prefix,
)
from batch_delivery.io.demand import get_source_days  # noqa: E402
from batch_delivery.routing.client import (  # noqa: E402
    _check_valhalla_memory, _health_check, _restart_vroom,
)
from batch_delivery.routing.core import (  # noqa: E402
    build_vroom_jobs, build_vroom_vehicles, solve_single_plz,
)

import logging  # noqa: E402
logging.disable(logging.INFO)

# 64a is a sibling script whose name starts with a digit — importable only by
# path. Selection/binning/plan logic lives THERE and is never duplicated here.
_spec = importlib.util.spec_from_file_location(
    "bundle_coverage", Path(__file__).resolve().parent / "64a_bundle_coverage.py")
cov = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(cov)

# ─────────────────────────────────────────────────────────────────────────────
# Paths / knobs
# ─────────────────────────────────────────────────────────────────────────────

LIVE_DIR = C.ROOT / "results" / "revision_2026_08"
LIVE_CHOSEN = LIVE_DIR / "_tab_chosen_v2.csv"
DEFAULT_OUT = LIVE_DIR / "bundles"
DEFAULT_COPIES = LIVE_DIR / "_solvecopy"

SOLVED_CSV = "bundles_solved.csv"
SOLVED_PARQUET = "bundles_solved.parquet"
THROUGHPUT_JSON = "bundles_throughput.json"
REPORT_MD = "bundles_pool_report.md"

#: Per-instance wall cap (brief: 15 min -> TIMEOUT, continue).
INSTANCE_CAP_S = 900.0
#: How long to keep harvesting stragglers after the queue drains.
STRAGGLER_GRACE_S = 1800.0
#: Phase-A budget horizon in the N_A formula (v3 amendment).
BUDGET_HOURS = 8.0
N_A_CAP = 400
#: Probe sizes for the throughput measurement.
PROBE_SERIAL = 10
PROBE_BATCH = 6
#: Rebuild the parquet twin every N appended rows.
PARQUET_EVERY = 25
#: Re-check docker health every N completed solves.
HEALTH_EVERY = 50

MANIFEST_COLS = [
    "provider", "hub_idx", "day", "kind", "members", "member_idx",
    "parcels", "stops", "area_km2", "features", "demand_level_key",
    "n_members", "occurrences", "first_seen",
]
JSON_COLS = ("members", "member_idx", "features", "first_seen")

#: The solved table's column order. FIXED: appends go to one CSV with the
#: header written once, so a row carrying a different column set (the TIMEOUT
#: stub does) would silently shift every field one column to the left. Every
#: append is reindexed onto this list.
OUTPUT_COLS = MANIFEST_COLS + [
    "bundle_id", "bin", "selected_reason", "phase", "late_completion",
    "vroom_cost_eur", "vroom_n_routes", "vroom_distance_km",
    "vroom_duration_h", "vroom_n_parcels", "vroom_status",
    "instance_parcels", "instance_stops", "parcels_mismatch",
    "n_jobs", "n_jobs_over_cap", "n_vehicles_planned", "n_unassigned",
    "jobs_removed", "parcels_removed", "hub_name", "solve_time_s",
    "solved_at",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Gate: the helper contract, asserted against the live signatures
# ─────────────────────────────────────────────────────────────────────────────

def _assert_helper_contract() -> None:
    veh = inspect.signature(build_vroom_vehicles).parameters
    slv = inspect.signature(solve_single_plz).parameters
    assert "seed_key" in veh, "build_vroom_vehicles lost its seed_key parameter"
    assert "cache_tag" not in veh, "build_vroom_vehicles grew a cache_tag"
    assert "cache_tag" in slv, "solve_single_plz lost its cache_tag parameter"
    assert "seed_key" not in slv, (
        "solve_single_plz grew a seed_key — the archived path2 script passed "
        "seed_key to the SOLVER and failed on every row; keep them separate")
    assert "n_jobs" in veh and "day_idx" in veh and "total_demand" in veh
    jobs = inspect.signature(build_vroom_jobs).parameters
    assert "pts_df" in jobs, "build_vroom_jobs signature changed"


_assert_helper_contract()


# ─────────────────────────────────────────────────────────────────────────────
# Copy-first read of the live chosen table (mirrors 62_/63_)
# ─────────────────────────────────────────────────────────────────────────────

def _retry_copy(src: Path, dst: Path) -> None:
    last: Exception | None = None
    for attempt in range(60):
        try:
            shutil.copy2(src, dst)
            return
        except PermissionError as e:
            last = e
            if attempt == 0:
                log(f"  WARNING: {src.name} locked ({e}); retrying up to 5 min")
            time.sleep(5)
    raise last  # type: ignore[misc]


def load_chosen_map(copies_dir: Path) -> dict[tuple, int]:
    """``(round(P,4), round(theta,4), provider, plz) -> schedule_idx``."""
    assert LIVE_CHOSEN.exists(), f"no grid output at {LIVE_CHOSEN}"
    copies_dir.mkdir(parents=True, exist_ok=True)
    dst = copies_dir / LIVE_CHOSEN.name
    _retry_copy(LIVE_CHOSEN, dst)
    df = pd.read_csv(dst, dtype={"plz": str})
    return {
        (round(float(r.penalty), 4), round(float(r.share_willing), 4),
         r.provider, str(r.plz)): int(r.schedule_idx_system_smoothed)
        for r in df.itertuples()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Instance construction — verbatim mirror of 20_validate_vroom_smoothed.py
# ─────────────────────────────────────────────────────────────────────────────

def _allocate_to_target(vals: np.ndarray, target: int) -> np.ndarray:
    """Distribute integer *target* across *vals*, largest-remainder.

    Verbatim from ``20_validate_vroom_smoothed._allocate_to_target``: preserves
    the aggregate total exactly (matching ``build_cost_matrices_ml``'s PLZ-level
    express rounding) instead of zeroing low-count stops via per-stop rounding.
    """
    vals = np.asarray(vals, dtype=float)
    s = vals.sum()
    if target <= 0 or s <= 0:
        return np.zeros(len(vals), dtype=int)
    raw = vals / s * target
    out = np.floor(raw).astype(int)
    rem = int(target - out.sum())
    if rem > 0:
        order = np.argsort(-(raw - out))
        out[order[:rem]] += 1
    return out


def _day_frame(pdata: dict, prefix: str, plz: str, d: int) -> pd.DataFrame | None:
    """One PLZ's points on day *d* — 20_'s source-day frame, unchanged."""
    gdf_d = pdata["daily_gdfs_wgs"].get(d)
    if gdf_d is None:
        return None
    pts = gdf_d[gdf_d["plz"] == plz]
    if len(pts) == 0:
        return None
    return pd.DataFrame({
        "str_idx": pts["str_idx"].astype(str).values,
        "lon": pts["lon"].astype(np.float64).values,
        "lat": pts["lat"].astype(np.float64).values,
        "dhl_b2c": pts.get(f"{prefix}_b2c",
                           pts.get("dhl_b2c", 0)).astype(int).values,
        "dhl_b2b": pts.get(f"{prefix}_b2b",
                           pts.get("dhl_b2b", 0)).astype(int).values,
    })


def build_instance(row, provider_data: dict, chosen_map: dict,
                   sched_by_idx: dict) -> tuple[pd.DataFrame | None, str]:
    """The bundle's VROOM points. Returns ``(pts_agg, note)``.

    ``note`` is ``"OK"``, or the reason the instance is unusable
    (``NO_SCHEDULE`` / ``EMPTY``) — the caller keeps a flagged row either way.
    """
    prov = str(row.provider)
    pdata = provider_data[prov]
    prefix = provider_to_demand_prefix(prov)
    P, th = float(row.first_seen[0]), float(row.first_seen[1])
    fs_b2c_v, fs_b2b_v = C.fs_b2c(th), C.fs_b2b(th)
    d = int(row.day)
    frames: list[pd.DataFrame] = []

    for plz in [str(m) for m in row.members]:
        if row.kind == "express":
            # 20_'s non-delivery-day treatment, taken from the other side: the
            # express residual IS round(total * fs), spread over stops by
            # largest remainder. Equals raw_express[z, d] by construction.
            f = _day_frame(pdata, prefix, plz, d)
            if f is None:
                continue
            b2c = f["dhl_b2c"].values.astype(int)
            b2b = f["dhl_b2b"].values.astype(int)
            f["dhl_b2c"] = _allocate_to_target(
                b2c, int(round(b2c.sum() * fs_b2c_v)))
            f["dhl_b2b"] = _allocate_to_target(
                b2b, int(round(b2b.sum() * fs_b2b_v)))
            f["dhl_total"] = f["dhl_b2c"] + f["dhl_b2b"]
            frames.append(f[f["dhl_total"] > 0])
            continue

        # delivery: this member's consolidated demand at ITS schedule
        si = chosen_map.get((round(P, 4), round(th, 4), prov, plz))
        if si is None:
            return None, "NO_SCHEDULE"
        sched_days = sched_by_idx[si]
        if d not in sched_days:
            # The bundle was sampled from a hub-day where every member
            # delivers; a chosen vector that disagrees means the (P, theta)
            # block moved under us. Flag rather than silently mis-route.
            return None, "NOT_A_DELIVERY_DAY"
        sched_set = set(sched_days)
        for sdy in get_source_days(d, sched_days):
            f = _day_frame(pdata, prefix, plz, sdy)
            if f is None:
                continue
            if sdy not in sched_set:
                b2c = f["dhl_b2c"].values.astype(int)
                b2b = f["dhl_b2b"].values.astype(int)
                f["dhl_b2c"] = _allocate_to_target(
                    b2c, int(b2c.sum() - round(b2c.sum() * fs_b2c_v)))
                f["dhl_b2b"] = _allocate_to_target(
                    b2b, int(b2b.sum() - round(b2b.sum() * fs_b2b_v)))
            f["dhl_total"] = f["dhl_b2c"] + f["dhl_b2b"]
            frames.append(f[f["dhl_total"] > 0])

    if not frames:
        return None, "EMPTY"
    pts = pd.concat(frames, ignore_index=True)
    agg = pts.groupby("str_idx", as_index=False).agg(
        lon=("lon", "first"), lat=("lat", "first"),
        dhl_total=("dhl_total", "sum"))
    agg = agg[agg["dhl_total"] > 0]
    if agg.empty:
        return None, "EMPTY"
    return agg, "OK"


def hub_row_for(row, provider_data: dict):
    """The bundle's hub. All members share one hub by construction — asserted."""
    df_assign = provider_data[str(row.provider)]["df_assignments"]
    hubs = set()
    first = None
    for plz in [str(m) for m in row.members]:
        hr = df_assign[df_assign["plz"] == plz]
        if hr.empty:
            continue
        hubs.add(str(hr.iloc[0]["hub_name"]))
        if first is None:
            first = hr.iloc[0]
    return first, hubs


# ─────────────────────────────────────────────────────────────────────────────
# One bundle -> one labelled row
# ─────────────────────────────────────────────────────────────────────────────

def solve_bundle(row, provider_data: dict, chosen_map: dict,
                 sched_by_idx: dict, idx: int, total: int,
                 phase: str) -> dict:
    t0 = time.perf_counter()
    base = {c: (list(getattr(row, c)) if c in JSON_COLS else getattr(row, c))
            for c in MANIFEST_COLS}
    base.update({
        "bundle_id": row.bundle_id, "bin": getattr(row, "bin", ""),
        "selected_reason": getattr(row, "selected_reason", ""),
        "phase": phase, "late_completion": False,
        "vroom_cost_eur": np.nan, "vroom_n_routes": 0,
        "vroom_distance_km": np.nan, "vroom_duration_h": np.nan,
        "vroom_n_parcels": 0, "vroom_status": "ERROR",
        "instance_parcels": np.nan, "instance_stops": 0,
        "parcels_mismatch": np.nan, "n_jobs": 0, "n_jobs_over_cap": False,
        "n_vehicles_planned": 0, "n_unassigned": 0, "jobs_removed": 0,
        "parcels_removed": 0, "hub_name": "", "solve_time_s": 0.0,
        "solved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    hub, hub_names = hub_row_for(row, provider_data)
    if hub is None:
        base["vroom_status"] = "NO_HUB"
        base["solve_time_s"] = round(time.perf_counter() - t0, 3)
        return base
    base["hub_name"] = str(hub["hub_name"])
    if len(hub_names) > 1:
        log(f"  WARNING {row.bundle_id}: members span {len(hub_names)} hubs "
            f"({sorted(hub_names)}) — using {base['hub_name']}")

    pts, note = build_instance(row, provider_data, chosen_map, sched_by_idx)
    if pts is None:
        base["vroom_status"] = note
        base["solve_time_s"] = round(time.perf_counter() - t0, 3)
        return base

    inst_parcels = int(pts["dhl_total"].sum())
    base["instance_parcels"] = inst_parcels
    base["instance_stops"] = int(len(pts))
    base["parcels_mismatch"] = float(inst_parcels - float(row.parcels))
    if abs(base["parcels_mismatch"]) > 0.5:
        log(f"  MISMATCH {row.bundle_id} ({row.provider} {row.kind} "
            f"d{row.day} n={row.n_members}): instance {inst_parcels} vs "
            f"manifest {row.parcels:.0f} (delta {base['parcels_mismatch']:+.0f})")

    jobs, total_demand = build_vroom_jobs(pts)
    if not jobs:
        base["vroom_status"] = "NO_JOBS"
        base["solve_time_s"] = round(time.perf_counter() - t0, 3)
        return base
    base["n_jobs"] = len(jobs)
    if len(jobs) > MAX_JOBS_PER_REQUEST:
        # NOT split: the bundle is one tour group, and a k-means split would
        # label a different object than the head is asked to price. Flagged so
        # Task 10 can see which rows sit above the sweep's request size.
        base["n_jobs_over_cap"] = True

    vehicles, n_veh = build_vroom_vehicles(
        hub=hub, total_demand=total_demand, day_idx=int(row.day),
        seed_key=f"bundle_{row.bundle_id}", n_jobs=len(jobs))
    base["n_vehicles_planned"] = int(n_veh)

    label = "|".join(sorted(str(m) for m in row.members))
    result = solve_single_plz(
        plz_code=label, request_body={"jobs": jobs, "vehicles": vehicles},
        idx=idx, total=total,
        day_name=CDAYS[int(row.day)] if int(row.day) < len(CDAYS) else "",
        cache_tag=f"bundle_{row.provider}")

    base.update({
        "vroom_status": str(result["status"]),
        "vroom_cost_eur": (float(result["cost"]) / COST_SCALE
                           if result["cost"] is not None else np.nan),
        "vroom_n_routes": int(result["n_routes"] or 0),
        "vroom_distance_km": (float(result["distance_m"]) / 1000.0
                              if result["distance_m"] is not None else np.nan),
        "vroom_duration_h": (float(result["duration_s"]) / 3600.0
                             if result["duration_s"] is not None else np.nan),
        "vroom_n_parcels": inst_parcels,
        "n_unassigned": int(result["n_unassigned"] or 0),
        "jobs_removed": int(result["jobs_removed"] or 0),
        "parcels_removed": int(result["parcels_removed"] or 0),
        "n_vehicles_planned": int(result["n_vehicles"] or n_veh),
        "solve_time_s": round(time.perf_counter() - t0, 3),
    })
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Append-only writer (retry-backoff pattern from 20_validate_vroom_smoothed)
# ─────────────────────────────────────────────────────────────────────────────

class Writer:
    """Appends rows to the CSV, rebuilds the parquet twin periodically.

    The 2026-07-16 overnight run died with PermissionError [Errno 13] on a
    checkpoint append (morning backup/AV/sync briefly locking the CSV).
    Losing hours of resumable progress to a transient lock is unacceptable, so
    retry with backoff for ~5 minutes before giving up.
    """

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.csv = out_dir / SOLVED_CSV
        self.parquet = out_dir / SOLVED_PARQUET
        self.n_since_parquet = 0
        self.n_written = 0

    def append(self, rows: list[dict]) -> None:
        if not rows:
            return
        unknown = set(rows[0]) - set(OUTPUT_COLS)
        assert not unknown, f"row carries columns outside OUTPUT_COLS: {unknown}"
        df = pd.DataFrame(rows).reindex(columns=OUTPUT_COLS)
        for c in JSON_COLS:
            df[c] = df[c].apply(
                lambda v: json.dumps(list(v)) if isinstance(v, (list, tuple,
                                                                np.ndarray))
                else v)
        last: Exception | None = None
        for attempt in range(60):
            try:
                df.to_csv(self.csv, mode="a", header=not self.csv.exists(),
                          index=False)
                last = None
                break
            except PermissionError as e:
                last = e
                if attempt == 0:
                    log(f"  WARNING: {self.csv.name} locked ({e}); retrying "
                        "up to 5 min")
                time.sleep(5)
        if last is not None:
            raise last
        self.n_written += len(rows)
        self.n_since_parquet += len(rows)
        if self.n_since_parquet >= PARQUET_EVERY:
            self.flush_parquet()

    def flush_parquet(self) -> None:
        df = read_solved(self.out_dir)
        if df.empty:
            return
        try:
            df.to_parquet(self.parquet, index=False)
            self.n_since_parquet = 0
        except Exception as exc:                  # pragma: no cover
            log(f"  WARNING: parquet rebuild failed ({exc})")


def read_solved(out_dir: Path, dedupe: bool = True) -> pd.DataFrame:
    """The solved table, list columns decoded, one row per ``bundle_id``.

    A TIMEOUT row is superseded by the straggler's real result when it lands,
    so dedupe keeps the last NON-timeout row per bundle when one exists.
    """
    p = out_dir / SOLVED_CSV
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"bundle_id": str, "demand_level_key": str})
    if df.empty:
        return df
    for c in JSON_COLS:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda v: json.loads(v) if isinstance(v, str) else v)
    if dedupe:
        df["_is_to"] = (df["vroom_status"] == "TIMEOUT").astype(int)
        df = (df.sort_values(["bundle_id", "_is_to"])
                .drop_duplicates("bundle_id", keep="first")
                .drop(columns="_is_to").reset_index(drop=True))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Throughput measurement
# ─────────────────────────────────────────────────────────────────────────────

def _solve_batch(rows, workers: int, ctx: dict, phase: str) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futs = [exe.submit(solve_bundle, r, ctx["provider_data"],
                           ctx["chosen_map"], ctx["sched_by_idx"],
                           i + 1, len(rows), phase)
                for i, r in enumerate(rows)]
        for f in futs:
            out.append(f.result())
    return out, time.perf_counter() - t0


def _fresh_times(rows: list[dict]) -> list[float]:
    """Solve times of rows that actually hit VROOM (a CACHED row measures the
    disk cache, not the solver, and would flatter the throughput)."""
    return [float(r["solve_time_s"]) for r in rows
            if r["vroom_status"] not in ("CACHED",) and r["solve_time_s"] > 0]


def measure_throughput(probe_rows, ctx: dict, writer: Writer) -> dict:
    """10 serial, then 2- and 3-concurrent on FRESH instances. Fastest stable
    setting wins; every probe solve is kept as a real Phase-A label."""
    log("=" * 72)
    log("THROUGHPUT probe: 10 serial, then 2- and 3-concurrent (fresh rows)")
    log("=" * 72)
    need = PROBE_SERIAL + 2 * PROBE_BATCH
    assert len(probe_rows) >= 1, "no probe rows available"
    probe_rows = probe_rows[:need]
    cuts = (probe_rows[:PROBE_SERIAL],
            probe_rows[PROBE_SERIAL:PROBE_SERIAL + PROBE_BATCH],
            probe_rows[PROBE_SERIAL + PROBE_BATCH:need])

    res: dict = {"probe_n": len(probe_rows)}
    settings: list[dict] = []

    for par, rows in zip((1, 2, 3), cuts):
        if not rows:
            continue
        out, wall = _solve_batch(rows, par, ctx, "A")
        writer.append(out)
        ok = sum(1 for r in out if r["vroom_status"] in ("OK", "CACHED", "PARTIAL"))
        times = _fresh_times(out)
        med = float(np.median(times)) if times else float("nan")
        sph = 3600.0 * len(rows) / wall if wall > 0 else 0.0
        stable = ok >= max(1, int(np.ceil(0.8 * len(rows))))
        settings.append({"parallelism": par, "n": len(rows),
                         "wall_s": round(wall, 1),
                         "median_solve_s": round(med, 2) if times else None,
                         "solves_per_hour": round(sph, 1),
                         "n_ok": ok, "stable": stable})
        log(f"  p={par}: {len(rows)} instances in {wall:6.1f}s -> "
            f"{sph:7.1f} solves/h  (median solve {med:6.1f}s, {ok}/{len(rows)} "
            f"ok, {'stable' if stable else 'UNSTABLE'})")

    res["settings"] = settings
    stable = [s for s in settings if s["stable"]] or settings
    best = max(stable, key=lambda s: s["solves_per_hour"])
    res["chosen_parallelism"] = int(best["parallelism"])
    res["chosen_solves_per_hour"] = float(best["solves_per_hour"])
    serial = next((s for s in settings if s["parallelism"] == 1), None)
    t_med = (serial or best)["median_solve_s"] or 60.0
    res["t_solve_median_s"] = float(t_med)
    res["n_a"] = int(min(N_A_CAP,
                         np.floor(BUDGET_HOURS * 3600.0 / t_med
                                  * res["chosen_parallelism"])))
    log(f"  -> parallelism {res['chosen_parallelism']} "
        f"({res['chosen_solves_per_hour']:.0f} solves/h), "
        f"median serial solve {t_med:.1f}s")
    log(f"  -> N_A = min({N_A_CAP}, floor({BUDGET_HOURS}h / {t_med:.1f}s x "
        f"{res['chosen_parallelism']})) = {res['n_a']}")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Phase A driver — resumable, per-instance cap, straggler harvesting
# ─────────────────────────────────────────────────────────────────────────────

def run_phase_a(queue, ctx: dict, writer: Writer, parallelism: int,
                cap_s: float = INSTANCE_CAP_S) -> None:
    total = len(queue)
    if total == 0:
        log("[phase A] nothing left to solve")
        return
    log("=" * 72)
    log(f"PHASE A: {total} bundles, parallelism {parallelism}, "
        f"{cap_s / 60:.0f} min per-instance cap")
    log("=" * 72)

    pending: dict[Future, dict] = {}
    stragglers: dict[Future, dict] = {}
    finished: set[str] = set()        # bundle_ids counted once (TIMEOUT or not)
    done_n = 0
    t_start = time.time()
    exe = ThreadPoolExecutor(max_workers=parallelism)

    def _count(bid: str) -> bool:
        nonlocal done_n
        if bid in finished:
            return False
        finished.add(bid)
        done_n += 1
        return True

    def _sink(f: Future, info: dict, late: bool) -> None:
        try:
            row = f.result()
        except Exception as exc:                  # pragma: no cover
            row = dict(info["stub"])
            row["vroom_status"] = f"EXC_{type(exc).__name__}"
            log(f"  EXC {info['bid']}: {type(exc).__name__}: {exc}")
        row["late_completion"] = bool(late)
        writer.append([row])
        advanced = _count(info["bid"])
        el = time.time() - t_start
        rate = done_n / max(el, 1e-9)
        eta = (total - done_n) / rate if rate > 0 else float("nan")
        tag = " LATE" if late else ""
        log(f"  [{done_n}/{total}]{tag} {info['bid']} {row['provider']:<7s} "
            f"{row['kind']:<9s} d{int(row['day'])} n={int(row['n_members'])} "
            f"jobs={int(row['n_jobs']):4d} -> {row['vroom_status']:<8s} "
            f"{float(row['vroom_cost_eur'] or 0):9.2f} EUR "
            f"{float(row['solve_time_s']):6.1f}s | "
            f"{rate * 3600:5.0f}/h eta {eta / 3600:5.2f}h")
        if advanced and done_n % HEALTH_EVERY == 0:
            _health_gate()

    def _harvest(block: bool) -> None:
        progressed = False
        for f in list(pending):
            if f.done():
                info = pending.pop(f)
                _sink(f, info, late=False)
                progressed = True
        for f in list(stragglers):
            if f.done():
                info = stragglers.pop(f)
                _sink(f, info, late=True)
                progressed = True
        now = time.time()
        for f, info in list(pending.items()):
            if not f.done() and now - info["t0"] > cap_s:
                pending.pop(f)
                stragglers[f] = info
                stub = dict(info["stub"])
                stub["vroom_status"] = "TIMEOUT"
                stub["solve_time_s"] = round(now - info["t0"], 1)
                writer.append([stub])
                _count(info["bid"])
                log(f"  TIMEOUT {info['bid']} after "
                    f"{(now - info['t0']) / 60:.1f} min — row flagged, thread "
                    "left running; its result will land as a late completion")
                progressed = True
        if block and not progressed:
            time.sleep(0.5)

    for i, r in enumerate(queue, 1):
        while len(pending) + len(stragglers) >= parallelism:
            _harvest(block=True)
        stub = {c: (list(getattr(r, c)) if c in JSON_COLS else getattr(r, c))
                for c in MANIFEST_COLS}
        stub.update({"bundle_id": r.bundle_id, "bin": getattr(r, "bin", ""),
                     "selected_reason": getattr(r, "selected_reason", ""),
                     "phase": "A", "late_completion": False,
                     "vroom_status": "TIMEOUT", "n_jobs": 0,
                     "n_members": int(r.n_members),
                     "vroom_cost_eur": np.nan, "solve_time_s": 0.0,
                     "solved_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        f = exe.submit(solve_bundle, r, ctx["provider_data"], ctx["chosen_map"],
                       ctx["sched_by_idx"], i, total, "A")
        pending[f] = {"bid": r.bundle_id, "t0": time.time(), "stub": stub}

    while pending:
        _harvest(block=True)
    if stragglers:
        log(f"[phase A] queue drained; {len(stragglers)} straggler(s) still "
            f"running — harvesting for up to {STRAGGLER_GRACE_S / 60:.0f} min")
        deadline = time.time() + STRAGGLER_GRACE_S
        while stragglers and time.time() < deadline:
            _harvest(block=True)
    exe.shutdown(wait=True)
    _harvest(block=False)
    writer.flush_parquet()
    log(f"[phase A] done: {done_n}/{total} in "
        f"{(time.time() - t_start) / 3600:.2f} h")


def _health_gate() -> None:
    if _health_check():
        _check_valhalla_memory()
        return
    log("  WARNING: VROOM health check failed — checking Valhalla")
    if not _check_valhalla_memory():
        log("  ERROR: Valhalla unrecoverable")
    if not _health_check():
        if _restart_vroom():
            log("  VROOM restarted")
        else:
            log("  ERROR: VROOM unreachable and restart failed")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _request_param_table() -> list[str]:
    """The exact VROOM request parameters, side by side with the sweep's.

    Read from ``config.constants`` at runtime rather than transcribed, so the
    report cannot drift from what was actually sent.
    """
    rows = [
        ("vehicle capacity", VEHICLE_CAPACITY, "build_vroom_vehicles"),
        ("max vehicles / request", MAX_VEHICLES_PER_REQUEST, "build_vroom_vehicles"),
        ("fixed cost (cents)", FIXED_COST_CENTS, "build_vroom_vehicles"),
        ("cost per km (cents)", COST_PER_KM_CENTS, "build_vroom_vehicles"),
        ("cost per hour (cents)", COST_PER_HOUR_CENTS, "build_vroom_vehicles"),
        ("profile", PROFILE, "build_vroom_vehicles"),
        ("speed factor", SPEED_FACTOR, "build_vroom_vehicles"),
        ("vehicle time window", VEHICLE_TIME_WINDOW, "build_vroom_vehicles"),
        ("break window / duration", f"{BREAK_WINDOW} / {BREAK_DURATION}s",
         "build_vroom_vehicles"),
        ("available work (s)", AVAILABLE_WORK_S, "build_vroom_vehicles"),
        ("delivery time window", DELIVERY_WINDOW, "build_vroom_jobs"),
        ("service time / parcel", f"{SERVICE_TIME_PER_PARCEL}s", "build_vroom_jobs"),
        ("service time cap", f"{SERVICE_TIME_CAP}s", "build_vroom_jobs"),
        ("max retries", MAX_RETRIES, "solve_single_plz"),
        ("max unassigned retries", MAX_UNASSIGNED_RETRIES, "solve_single_plz"),
        ("per-job budget (s)", PER_JOB_BUDGET_S, "solve_single_plz"),
        ("VROOM url", VROOM_API_URL, "solve_single_plz"),
        ("exploration level (-x)", "not set (VROOM default)", "solve_single_plz"),
    ]
    out = ["| parameter | value | set by | sweep pool | bundle pool |",
           "|---|---|---|---|---|"]
    for name, val, who in rows:
        out.append(f"| {name} | `{val}` | `{who}` | same | same |")
    return out


def write_report(out_dir: Path, throughput: dict | None) -> None:
    sel_p = out_dir / cov.SELECTION_PARQUET
    solved = read_solved(out_dir)
    md: list[str] = ["# Bundle pool report\n",
                     f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n"]

    if solved.empty:
        md.append("No solved rows yet.\n")
        (out_dir / REPORT_MD).write_text("\n".join(md), encoding="utf-8")
        return

    n_a = int((solved["phase"] == "A").sum())
    n_b = int((solved["phase"] == "B").sum())
    md.append("## Counts\n")
    md.append(f"- Phase A rows: **{n_a}**")
    md.append(f"- Phase B rows: **{n_b}**")
    md.append(f"- Total labelled rows (unique bundle_id): **{len(solved)}**")
    if throughput:
        md.append(f"- Measured throughput: **"
                  f"{throughput.get('chosen_solves_per_hour', 0):.0f} solves/h** "
                  f"at parallelism **{throughput.get('chosen_parallelism')}** "
                  f"(median serial solve "
                  f"{throughput.get('t_solve_median_s', 0):.1f}s); "
                  f"N_A = **{throughput.get('n_a')}**")
    md.append("")

    md.append("## Status histogram\n")
    md.append("| status | rows |")
    md.append("|---|---:|")
    for s, n in solved["vroom_status"].value_counts().items():
        md.append(f"| {s} | {n} |")
    md.append("")
    n_partial = int((solved["vroom_status"] == "PARTIAL").sum())
    md.append(f"`PARTIAL` rows are KEPT and flagged ({n_partial}) — never "
              "dropped (Kompendium 38.2b).\n")

    ok = solved[solved["vroom_status"].isin(cov.LABEL_OK)]
    if len(ok):
        md.append("## Solve time (labelled rows)\n")
        q = ok["solve_time_s"].quantile([0.5, 0.9, 0.99]).round(1)
        md.append(f"- median **{q.loc[0.5]}s**, p90 {q.loc[0.9]}s, "
                  f"p99 {q.loc[0.99]}s, max {ok['solve_time_s'].max():.1f}s")
        md.append(f"- total VROOM wall time: "
                  f"{ok['solve_time_s'].sum() / 3600:.2f} h")
        md.append("")
        md.append("## Label sanity\n")
        mm = ok["parcels_mismatch"].abs()
        md.append(f"- manifest `parcels` vs instance demand: "
                  f"{int((mm <= 0.5).sum())}/{len(ok)} exact, "
                  f"max |delta| = {mm.max():.1f}")
        md.append(f"- cost/parcel: median "
                  f"{(ok['vroom_cost_eur'] / ok['vroom_n_parcels'].clip(lower=1)).median():.3f} EUR")
        md.append(f"- rows above the sweep's request size cap "
                  f"({MAX_JOBS_PER_REQUEST} jobs, not split on purpose): "
                  f"{int(ok['n_jobs_over_cap'].sum())}")
        md.append("")

    md.append("## VROOM request parameters (sweep vs bundle pool)\n")
    md.append("Both pools call `build_vroom_jobs` / `build_vroom_vehicles` / "
              "`solve_single_plz` unchanged, so every value below is shared by "
              "construction — read from `config.constants` at report time, not "
              "transcribed.\n")
    md.extend(_request_param_table())
    md.append("")
    md.append("Two deliberate, documented differences, neither a parameter "
              "change:\n")
    md.append("1. **Instance content.** The sweep routes one PLZ's perturbed "
              "demand; the bundle pool routes a whole tour group's real "
              "consolidated demand. That IS the object the head prices.")
    md.append("2. **`seed_key` value.** Sweep: `combo.cache_tag()`; bundles: "
              "`bundle_<bundle_id>`. It seeds vehicle start-time offsets only, "
              "drawn from the same half-normal in both pools. Note that "
              "`build_vroom_vehicles` mixes in Python's `hash(seed_key)`, which "
              "is per-process randomised unless `PYTHONHASHSEED` is set — true "
              "of the sweep as well, so this is a shared noise source, not an "
              "offset. Neither pool sets it.\n")

    if sel_p.exists():
        sel = pd.read_parquet(sel_p)
        labelled = set(ok["bundle_id"]) if len(ok) else set()
        tab = cov.coverage_table(sel, labelled)
        md.append("## Coverage\n")
        md.append(f"- non-empty bins: **{len(tab)}**")
        md.append(f"- bins with >= {cov.BIN_FLOOR} labelled rows: "
                  f"**{int((tab['n_labelled'] >= cov.BIN_FLOOR).sum())}**")
        md.append(f"- bins with >= {cov.SPARSE_FLOOR} labelled rows: "
                  f"**{int((tab['n_labelled'] >= cov.SPARSE_FLOOR).sum())}**")
        thin = tab[tab["n_labelled"] < cov.BIN_FLOOR]
        md.append(f"- **still-sparse bins (< {cov.BIN_FLOOR} labelled): "
                  f"{len(thin)}** — these are the head's uncovered "
                  "compositions: Task 10 reports OOF per bin, Task 11 may fall "
                  "back to Sigma-single pricing where support is missing. "
                  "Flagged here, not decided here.\n")
        if len(thin):
            md.append("| bin | realised | selected | labelled | occurrences |")
            md.append("|---|---:|---:|---:|---:|")
            for _, b in thin.sort_values(
                    "occurrences", ascending=False).head(60).iterrows():
                md.append(f"| {cov.md_cell(b['bin'])} | "
                          f"{int(b['n_realised'])} | "
                          f"{int(b['n_selected'])} | {int(b['n_labelled'])} | "
                          f"{int(b['occurrences'])} |")
            md.append("")
        unsolved = sel[~sel["bundle_id"].isin(set(solved["bundle_id"]))]
        md.append("## Deployment test set\n")
        md.append(f"{len(unsolved)} realised-but-unsolved bundles "
                  f"(of {len(sel)}). Per the v3 amendment these are the head's "
                  "deployment test set; Phase C is simply a larger `--n-a`.\n")
        md.append("### Labelled rows by (kind, n_members)\n")
        if len(ok):
            lab = sel[sel["bundle_id"].isin(labelled)]
            md.append("| kind | n_members | labelled | realised |")
            md.append("|---|---|---:|---:|")
            real = sel.groupby(["kind", "nm_bin"]).size()
            for (k, nm), n in lab.groupby(["kind", "nm_bin"]).size().items():
                md.append(f"| {k} | {nm} | {n} | {int(real.get((k, nm), 0))} |")
            md.append("")

    (out_dir / REPORT_MD).write_text("\n".join(md), encoding="utf-8")
    log(f"[write] {out_dir / REPORT_MD}")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, default=cov.MANIFEST_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--copies-dir", type=Path, default=DEFAULT_COPIES)
    ap.add_argument("--phase", choices=["A", "B"], default="A")
    ap.add_argument("--n-a", type=int, default=None,
                    help="override the throughput-derived Phase-A budget")
    ap.add_argument("--parallelism", type=int, default=None,
                    help="override the measured parallelism")
    ap.add_argument("--throughput-only", action="store_true")
    ap.add_argument("--skip-throughput", action="store_true",
                    help="reuse bundles_throughput.json (requires --n-a or a "
                         "previous measurement)")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--retry-timeouts", action="store_true",
                    help="re-queue bundles whose last row is TIMEOUT")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many new solves (smoke tests)")
    ap.add_argument("--instance-cap-s", type=float, default=INSTANCE_CAP_S,
                    help=f"per-instance wall cap before TIMEOUT "
                         f"(default {INSTANCE_CAP_S:.0f}s = 15 min)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tp_path = args.out_dir / THROUGHPUT_JSON
    throughput = (json.loads(tp_path.read_text(encoding="utf-8"))
                  if tp_path.exists() else None)

    if args.report_only:
        write_report(args.out_dir, throughput)
        return

    if args.phase == "B":
        raise SystemExit(
            "Phase B solving is not enabled in this revision. Run\n"
            "  64a_bundle_coverage.py --plan-only\n"
            "to (re)generate bundles_phaseB_plan.md and get the controller's "
            "approval of the plan size first (v2 brief: 'Print the plan BEFORE "
            "solving').")

    log("=" * 72)
    log("64: VROOM bundle pool — Phase A")
    log("=" * 72)

    # ── Gate 1: docker health ────────────────────────────────────────
    if not _health_check():
        _check_valhalla_memory()
        if not _health_check():
            raise SystemExit("VROOM is not healthy — start docker compose first")
    log("[gate] VROOM health OK; helper contract asserted at import "
        "(seed_key -> build_vroom_vehicles, cache_tag -> solve_single_plz)")

    # ── Inputs ───────────────────────────────────────────────────────
    log("[load] checkpoints ...")
    t0 = time.perf_counter()
    provider_data, optim_data = C.load_checkpoints()
    del optim_data                     # not needed: no cost matrices here
    slim = {p: {"daily_gdfs_wgs": d["daily_gdfs_wgs"],
                "df_assignments": d["df_assignments"]}
            for p, d in provider_data.items()}
    del provider_data
    gc.collect()
    schedules = C.enumerate_schedules()
    assert len(schedules) == 39, f"expected 39 schedules, got {len(schedules)}"
    sched_by_idx = {i: sorted(s) for i, s in enumerate(schedules)}
    chosen_map = load_chosen_map(args.copies_dir)
    log(f"[load] done in {time.perf_counter() - t0:.0f}s "
        f"({len(chosen_map):,} chosen rows, copy-first)")
    ctx = {"provider_data": slim, "chosen_map": chosen_map,
           "sched_by_idx": sched_by_idx}

    writer = Writer(args.out_dir)
    solved = read_solved(args.out_dir)
    done_ids = set(solved["bundle_id"]) if len(solved) else set()
    if args.retry_timeouts and len(solved):
        done_ids -= set(solved.loc[solved["vroom_status"] == "TIMEOUT",
                                   "bundle_id"])
    log(f"[resume] {len(done_ids)} bundle_id(s) already solved")

    # ── Throughput probe -> N_A ──────────────────────────────────────
    if args.skip_throughput or args.n_a is not None:
        n_a = args.n_a or (throughput or {}).get("n_a") or N_A_CAP
        parallelism = args.parallelism or (throughput or {}).get(
            "chosen_parallelism", 2)
        probe_ids: set[str] = set()
        log(f"[throughput] skipped — N_A={n_a}, parallelism={parallelism}")
    else:
        # Provisional ranking at the cap: selection(N) is a PREFIX of
        # selection(400) under one ranking, so probe rows drawn from the
        # provisional top-400 are legitimate members of any final selection
        # (and are carried explicitly via `extra` if N_A lands below them).
        prov_sel = cov.build_selection(args.manifest, args.out_dir, N_A_CAP,
                                       write=False)
        cand = prov_sel[prov_sel["selected"]
                        & ~prov_sel["bundle_id"].isin(done_ids)]
        need = PROBE_SERIAL + 2 * PROBE_BATCH
        if len(cand) > need:
            # spread over the whole selection, not just its heaviest head
            step = max(1, len(cand) // need)
            cand = cand.iloc[::step].head(need)
        probe_rows = list(cand.itertuples())
        if not probe_rows:
            raise SystemExit("nothing left to probe — everything is solved")
        throughput = measure_throughput(probe_rows, ctx, writer)
        tp_path.write_text(json.dumps(throughput, indent=2), encoding="utf-8")
        log(f"[write] {tp_path}")
        probe_ids = {r.bundle_id for r in probe_rows}
        n_a = args.n_a or throughput["n_a"]
        parallelism = args.parallelism or throughput["chosen_parallelism"]
        done_ids |= probe_ids
        if args.throughput_only:
            writer.flush_parquet()
            write_report(args.out_dir, throughput)
            log("[done] --throughput-only")
            return

    # ── Selection at the measured budget ─────────────────────────────
    sel = cov.build_selection(args.manifest, args.out_dir, n_a,
                              extra=probe_ids)
    cov.print_selection_summary(sel)

    queue = sel[sel["selected"] & ~sel["bundle_id"].isin(done_ids)]
    queue = queue.sort_values("select_rank")
    if args.limit:
        queue = queue.head(args.limit)
    run_phase_a(list(queue.itertuples()), ctx, writer, int(parallelism),
                cap_s=float(args.instance_cap_s))

    writer.flush_parquet()
    write_report(args.out_dir, throughput)
    log("[done] Phase A finished. Re-run 64a_bundle_coverage.py --plan-only "
        "for the Phase-B plan against actual labels.")


if __name__ == "__main__":
    main()
