"""67: VROOM validation v2 — both plans, both cost lenses, cache census.

Out-of-sample VROOM re-routing of the plans the revised grid (v5 schema)
produces. Where ``20_validate_vroom_smoothed.py`` re-routed ONE plan at
theta = 1 and compared ONE number (the per-(provider, PLZ) batched delivery
cost), this script re-routes **every instance the realistic-tour rule forms**
— per-cell delivery tours, pooled delivery groups, express singletons and
express bundles — for TWO plans per (P, theta) and reports the comparison in
BOTH cost lenses:

    routing lens   Sigma predicted  vs  Sigma vroom_cost
    operator lens  variable = cost - 189.15 * n_routes
                   peak_h   = max_d Sigma n_routes of hub h's instances
                   OpCost   = Sigma variable + 1134.90 * Sigma_h peak_h

WHAT IS VALIDATED (controller amendment 2026-08-27, priority order)
-------------------------------------------------------------------
  item 0  theta = 0, the DAILY BASELINE (Task 12b) — every cell pinned to the
          all-daily schedule (``61_grid_run_v2.py``'s G-6f-1), no express by
          construction. This is the VROOM baseline the paper's savings % are
          actually measured against; items 1-3 alone cannot state an ACTUAL
          saving because there was no VROOM-solved reference point. Built at
          ONE representative P (0.0) and asserted P-invariant against every
          other P in the grid — solving the identical daily tours once per P
          would waste VROOM budget on duplicate requests.
  item 1  theta = 1, OPERATOR plan (``schedule_idx_balanced``),
          P in {0, 0.25, 0.5, 0.75} — every instance.
  item 2  theta = 1, ROUTING plan (``schedule_idx_stage1``), P in {0, 0.25} —
          continuity with the submitted validation (22.8 % pred vs 23.7 %
          actual at P=0; 18.5 vs 19.8 at P=0.25).
  item 3  (P, theta) = (0.25, 0.5), OPERATOR plan — the only point with all
          instance kinds in volume. G6 stratified fallback available.

Hard budget: 8 VROOM-hours for items 1-3 together (item 0 is cheap: mostly
cache hits against the 2026-07 validation's daily solves, see below). The
CENSUS runs first and never solves anything; the controller approves the
solve from its numbers.

THE PREDICTED SIDE IS THE GRID'S OWN, NOT A RE-IMPLEMENTATION
--------------------------------------------------------------
Every predicted number is read from the SAME matrices the grid priced with
(``build_cost_matrices_ml`` via ``_stage3_common``), through the same
expressions ``costs.py`` uses:

  * per-cell delivery tour  -> ``cost_3d[z, s, d]``
  * pooled delivery group   -> ``head=None``: Sigma ``small_delivery_price``
                               over members (what ``_hub_smallday_pool_ml``
                               sums);  with a head: ``price_group(...)``
  * express group           -> ``head=None``: Sigma ``express_cost``
                               over members;  with a head: ``price_group(...)``

and the reconstruction is GATED, not trusted:

  G1 (routing identity)  Sigma predicted over a (P, theta, provider, plan)
     must equal ``tab_costs_v2.csv``'s ``cost_stage2_eur`` (operator plan) /
     ``cost_stage1_eur`` (routing plan) within 1e-6 relative. Both columns are
     bundled routing totals (dd + express + pool) at their own plan — see
     ``61_grid_run_v2.py`` (``init_cost = bal["initial_total_cost"]``, the
     stage-3 assert ``res["cost"] == dd + expr + pool``).
  G2 (fleet identity)  the per-instance PREDICTED vehicle counts, aggregated
     per (hub, day), must equal ``tab_fleet_per_hub_v2.csv`` — which is the
     grid's own ``_daily_fleet_per_hub`` output. Balanced plan only (the fleet
     table is written at the final plan).

A failed gate aborts: it means the instance set is not the one the grid
priced, and a validation of the wrong instance set is worse than none.

INSTANCE CONSTRUCTION IS 20_'s AND 64_'s, VERBATIM
---------------------------------------------------
The VROOM points come from ``64_solve_bundles_vroom.build_instance``, which is
itself the verbatim mirror of ``20_validate_vroom_smoothed``'s delivery-day
construction (source-day frames, ``_allocate_to_target`` willing reduction,
``groupby str_idx`` dedupe) — imported here by path, never copied, so a bundle
already solved for the head's training pool is the SAME request. ``jobs`` /
``vehicles`` are built by the same ``build_vroom_jobs`` / ``build_vroom_vehicles``
helpers with ``seed_key`` -> vehicles and ``cache_tag`` -> solver, and no
constant is overridden anywhere (``per_hour`` included).

THE CACHE CENSUS, AND THE hash() PROBLEM IT HAD TO SOLVE
---------------------------------------------------------
``build_vroom_vehicles`` seeds the vehicle start-time stagger with
``VEH_START_SEED + day*1000 + hash(seed_key) % 10000``. Python's ``hash`` of a
str is **per-process randomised**, so the same instance built in two processes
carries different ``time_window`` starts — a different request body, a
different SHA-256, and therefore a cache MISS against everything solved by an
earlier process. A naive census would report ~0 hits and a budget three times
too large.

The only process-dependent input is that one integer, and it has exactly
10 000 possible values. The census therefore enumerates all of them
(``_offsets_for_seed``), hashes the 10 000 candidate bodies (prefix-shared
SHA-256 over a string template — ~0.2 s per instance) and asks whether ANY of
them is on disk. Correctness is gated per instance: candidate ``k0 =
hash(seed_key) % 10000`` must reproduce the vehicles ``build_vroom_vehicles``
actually returned, byte for byte (``_assert_offset_replication``), and the
assembled canonical string must hash to ``routing.cache._request_hash`` of the
real body.

When solving, a hit is only usable if the request CARRIES those offsets, so
the matching candidate's offsets are pinned into the vehicles
(``--no-offset-pinning`` disables it and re-solves instead). This is a noise
draw, not a modelling choice: the offsets are an i.i.d. half-normal stagger,
the cached draw was made by an RNG seeded independently of any cost, and the
instance (jobs, fleet size, hub, time windows) is otherwise identical. The
column ``offsets_pinned`` records where it happened.

Cache tags follow the scripts whose solutions we want to inherit:
``smval_p{P}_s{theta:.1f}_{prov}`` for per-cell delivery tours (20_'s tag) and
``bundle_{prov}`` for groups (64_'s tag). The census additionally probes EVERY
tag directory and, when a body is found under another tag, solves with that
tag so the hit is real.

OPERATIONS
----------
Resumable (append-only CSV, fixed header, one row per instance keyed by
``instance_id``), ``InstanceLock`` over the output dir, retry-backoff writer,
PARTIAL rows kept and flagged (excluded from MAPE, included in the totals with
both numbers stated). Refuses to start from a dirty ``scripts/`` or ``src/``
tree (launch-only-from-committed-state) unless ``--allow-dirty``.

Run (census; safe, no VROOM solves):
    .venv\\Scripts\\python.exe scripts/revision/67_validate_vroom_v2.py --census-only

Run (solve, outside the harness):
    Start-Process -FilePath .venv\\Scripts\\python.exe -ArgumentList \\
      "scripts/revision/67_validate_vroom_v2.py","--rev-dir","results/revision_2026_08_v5",\\
      "--items","1,2,3","--parallelism","3" \\
      -RedirectStandardOutput results/revision_2026_08_v5/validation/67.log \\
      -RedirectStandardError  results/revision_2026_08_v5/validation/67.err

Output (results/<rev>/validation/):
    census.md / census.csv        per item x kind: instances, cache hits, hours
    instance_queue.csv            the full instance list (predicted side + cache)
    tab_vroom_v2.csv              one row per solved instance
    validation_report.md          MAPE/bias per kind, both lenses, per plan/point,
                                   and (once item 0 is solved) predicted-vs-actual
                                   saving % vs the theta=0 baseline, both lenses

A CENSUS for a SUBSET of items (e.g. ``--items 0`` to add the baseline onto an
already-censused/partly-solved output dir) MERGES onto the existing
instance_queue.csv: only the requested item(s)' rows are (re)computed, every
other item's rows — including their cache-hit flags and any G6 selection —
are carried through untouched. ``tab_vroom_v2.csv`` is never touched by a
census; it only grows via ``run_solve``'s append.
"""
from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _stage3_common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "src"))
from batch_delivery.config.constants import (  # noqa: E402
    COST_SCALE, LARGE_HUB_TYPES, RESULTS_DIR, SMALL_HUB_DELAY,
    VEH_START_LATEST, VEH_START_SEED, VEH_START_SPREAD_S,
    VEHICLE_TIME_WINDOW, WEEKDAYS as CDAYS,
)
from batch_delivery.optimization.balancing import (  # noqa: E402
    WEEK_FIXED_COST_EUR,
)
from batch_delivery.config.constants import FIXED_COST_EUR  # noqa: E402
from batch_delivery.optimization.core import (  # noqa: E402
    build_cost_matrices_ml,
)
from batch_delivery.optimization.costs import (  # noqa: E402
    _express_partition, _express_partition_vehicles,
    _delivery_partition_vehicles, _smallday_members, _smallday_partition,
)
from batch_delivery.routing.cache import _request_hash  # noqa: E402
from batch_delivery.routing.client import _health_check  # noqa: E402
from batch_delivery.routing.core import (  # noqa: E402
    build_vroom_jobs, build_vroom_vehicles, solve_single_plz,
)

import logging  # noqa: E402
logging.disable(logging.INFO)


def _load_sibling(filename: str, modname: str):
    """Import a sibling script whose name starts with a digit."""
    spec = importlib.util.spec_from_file_location(
        modname, Path(__file__).resolve().parent / filename)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


#: 64_ owns instance construction (points, hub, trunc convention). Imported,
#: never copied: a bundle already solved for the pool must be the SAME request.
B64 = _load_sibling("64_solve_bundles_vroom.py", "bundle_solver_64")


# ─────────────────────────────────────────────────────────────────────────────
# Paths / knobs
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_REV_DIR = C.ROOT / "results" / "revision_2026_08_v5"
CHOSEN_CSV = "_tab_chosen_v2.csv"
COSTS_CSV = "tab_costs_v2.csv"
FLEET_CSV = "tab_fleet_per_hub_v2.csv"

QUEUE_CSV = "instance_queue.csv"
SOLVED_CSV = "tab_vroom_v2.csv"
CENSUS_CSV = "census.csv"
CENSUS_MD = "census.md"
REPORT_MD = "validation_report.md"
G6_NOTE_MD = "G6_sampling_note.md"
LOCK_JSON = "67.lock"

#: Measured throughput source (64_'s Phase-A pool).
THROUGHPUT_SRC = (C.ROOT / "results" / "revision_2026_08" / "bundles"
                  / "bundles_solved.csv")
THROUGHPUT_JSON = (C.ROOT / "results" / "revision_2026_08" / "bundles"
                   / "bundles_throughput.json")

#: Brief's fallback when no measurement is available: 60 s singles, 150 s bundles.
FALLBACK_T_SINGLE_S = 60.0
FALLBACK_T_BUNDLE_S = 150.0

#: Hard budget for items 1-3 together (controller amendment).
BUDGET_HOURS = 8.0

#: Re-check docker health every N completed solves.
HEALTH_EVERY = 50
#: Rebuild the parquet twin / print progress every N appended rows.
PROGRESS_EVERY = 20

#: hash(seed_key) % 10000 — the entire process-dependent candidate space.
HASH_MOD = 10000

#: Statuses worth another attempt on a resumed run.
RETRYABLE = frozenset({"ERROR", "CONN_ERROR", "BUDGET_EXCEEDED", "TIMEOUT"})

#: Instance kinds (census taxonomy).
KIND_DELIVERY_SINGLE = "delivery_single"
KIND_DELIVERY_GROUP = "delivery_group"
KIND_EXPRESS_SINGLE = "express_single"
KIND_EXPRESS_GROUP = "express_group"
KINDS = (KIND_DELIVERY_SINGLE, KIND_DELIVERY_GROUP,
         KIND_EXPRESS_SINGLE, KIND_EXPRESS_GROUP)

#: What to validate. ``plan`` names the column in _tab_chosen_v2.csv and the
#: cost column in tab_costs_v2.csv the identity gate anchors on.
#:
#: item 0's ``penalties`` is deliberately a SINGLETON ([0.0]): theta=0 pins
#: every cell to the daily schedule independently of P (G-6f-1), so the same
#: VROOM tours would be re-requested once per P for nothing. The single P=0.0
#: point is asserted P-invariant against the grid's OTHER P values by
#: ``assert_baseline_invariant`` before it is trusted (see ``run_census``).
ITEMS: dict[int, dict] = {
    0: dict(theta=0.0, plan="stage1", penalties=[0.0],
            label="theta=0 daily baseline (no VROOM reference existed before "
                  "Task 12b)"),
    1: dict(theta=1.0, plan="balanced", penalties=[0.0, 0.25, 0.5, 0.75],
            label="theta=1 operator plan"),
    2: dict(theta=1.0, plan="stage1", penalties=[0.0, 0.25],
            label="theta=1 routing plan"),
    3: dict(theta=0.5, plan="balanced", penalties=[0.25],
            label="(P=0.25, theta=0.5) operator plan"),
}
PLAN_SCHED_COL = {"balanced": "schedule_idx_balanced",
                  "stage1": "schedule_idx_stage1"}
PLAN_COST_COL = {"balanced": "cost_stage2_eur", "stage1": "cost_stage1_eur"}

#: The solved table's column order. FIXED: appends go to one CSV whose header
#: is written once, so a row with a different column set would shift fields.
QUEUE_COLS = [
    "instance_id", "item", "penalty", "share_willing", "plan", "provider",
    "hub_idx", "hub_name", "instance_kind", "vroom_kind", "day", "n_members",
    "members", "member_idx", "predicted_cost_eur", "predicted_source",
    "predicted_bin", "predicted_n_routes", "predicted_parcels",
    "predicted_stops", "instance_parcels", "instance_stops", "n_jobs",
    "n_vehicles_planned", "cache_tag", "cache_hit", "cache_hit_tag",
    "cache_seed_k", "build_note", "est_solve_s", "g6_selected",
]
SOLVED_COLS = QUEUE_COLS + [
    "vroom_cost_eur", "vroom_n_routes", "vroom_distance_km",
    "vroom_duration_h", "vroom_n_parcels", "vroom_status", "n_unassigned",
    "jobs_removed", "parcels_removed", "n_vehicles_final", "offsets_pinned",
    "solve_time_s", "solved_at",
]


# The console this runs on is cp1252; the reports are UTF-8 by design. Without
# this, a single non-ASCII character in a progress line kills a finished census
# on the last print (observed 2026-08-27).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                               # pragma: no cover
        pass


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Gate: launch only from a committed tree
# ─────────────────────────────────────────────────────────────────────────────

def assert_clean_tree(allow_dirty: bool) -> None:
    """Refuse to start from a working tree with uncommitted scripts/ or src/.

    A long solve launched from an editable tree cannot be attributed to any
    commit afterwards — the launch-only-from-committed-state rule.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--short", "scripts/", "src/"],
            cwd=str(C.ROOT), capture_output=True, text=True, timeout=60)
    except Exception as e:                          # pragma: no cover
        raise SystemExit(f"cannot run git status: {e}")
    dirty = [ln for ln in out.stdout.splitlines() if ln.strip()]
    if not dirty:
        return
    msg = ("working tree is dirty under scripts/ or src/:\n  "
           + "\n  ".join(dirty)
           + "\n\nCommit (or stash) first — a run launched from an editable "
             "tree cannot be reproduced. Pass --allow-dirty to override.")
    if allow_dirty:
        log("WARNING: --allow-dirty — " + msg.replace("\n", " "))
        return
    raise SystemExit(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Gate: one instance per output dir (64_'s lock, same semantics)
# ─────────────────────────────────────────────────────────────────────────────

class InstanceLock:
    """Exclusive lock over one output directory (mirrors 64_'s)."""

    def __init__(self, out_dir: Path):
        self.path = out_dir / LOCK_JSON
        self.held = False

    def acquire(self) -> None:
        if self.path.exists():
            try:
                info = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                info = {}
            pid = int(info.get("pid", -1))
            if pid > 0 and B64._pid_alive(pid, info.get("create_time")):
                raise SystemExit(
                    f"another 67_ instance is already running on "
                    f"{self.path.parent} (PID {pid}, started "
                    f"{info.get('started', '?')}).\n"
                    "Two instances appending to tab_vroom_v2.csv can "
                    "interleave inside one row. Wait for it, or — if you are "
                    f"certain it is gone — delete\n  {self.path}")
            log(f"  [lock] stale lock from dead PID {pid} — taking over")
        rec = {"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
               "argv": sys.argv[1:], "out_dir": str(self.path.parent)}
        try:
            import psutil
            rec["create_time"] = psutil.Process(os.getpid()).create_time()
        except Exception:                           # pragma: no cover
            rec["create_time"] = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        self.held = True
        atexit.register(self.release)
        log(f"  [lock] acquired {self.path.name} (PID {os.getpid()})")

    def release(self) -> None:
        if not self.held:
            return
        self.held = False
        try:
            self.path.unlink(missing_ok=True)
        except Exception:                           # pragma: no cover
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Append-only writer (retry-backoff pattern from 20_/64_)
# ─────────────────────────────────────────────────────────────────────────────

def append_rows(path: Path, rows: list[dict], cols: list[str]) -> None:
    """Append rows onto *path*, reindexed onto the FIXED column order.

    Retries transient Windows file locks (AV/backup/sync) for ~5 minutes
    rather than losing hours of resumable progress.
    """
    if not rows:
        return
    df = pd.DataFrame(rows).reindex(columns=cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(60):
        try:
            df.to_csv(path, mode="a", header=not path.exists(), index=False)
            return
        except PermissionError as e:                # transient lock
            last = e
            if attempt == 0:
                log(f"  WARNING: {path.name} locked ({e}); retrying up to 5 min")
            time.sleep(5)
    raise last                                      # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# The predicted side: the grid's own pricing path
# ─────────────────────────────────────────────────────────────────────────────

_PRICE_GROUP_HAS_SOURCE: bool | None = None


def price_group_adapter(members, day: int, matrices: dict, kind: str, head,
                        parcels_by_cell=None, stops_by_cell=None,
                        freq: float = 1.0) -> tuple[float, str, str]:
    """``price_group`` behind a contract-tolerant shim.

    ``surrogate/bundle.py`` is being extended (certified support, Task 10b) and
    its return type is either a bare float or a ``GroupPrice(price, source,
    bin)``. This adapter reads the live signature once and normalises to
    ``(price, source, bin)`` so this script never has to be edited when the
    head contract settles. It NEVER re-implements the price.
    """
    global _PRICE_GROUP_HAS_SOURCE
    from batch_delivery.surrogate.bundle import price_group
    if _PRICE_GROUP_HAS_SOURCE is None:
        _PRICE_GROUP_HAS_SOURCE = (
            "with_source" in inspect.signature(price_group).parameters)
    kw = dict(kind=kind, parcels_by_cell=parcels_by_cell,
              stops_by_cell=stops_by_cell, freq=freq, head=head)
    if _PRICE_GROUP_HAS_SOURCE:
        r = price_group(members, day, matrices, with_source=True, **kw)
        if isinstance(r, tuple) and len(r) == 3:
            src = getattr(r[1], "value", r[1])
            return float(r[0]), str(src), str(r[2] or "")
        return float(r), "head", ""
    return float(price_group(members, day, matrices, **kw)), "head", ""


def price_delivery_group(idx: list[int], d: int, chosen: np.ndarray,
                         m: dict, head, parcels_arr, stops_arr
                         ) -> tuple[float, str, str]:
    """A pooled delivery group's predicted price — ``_hub_smallday_pool_ml``'s."""
    if head is None:
        # head=None regime: price_group prices the group as the Sigma of its
        # members' SINGLETON prices, which is exactly the precomputed table
        # _hub_smallday_pool_ml sums (costs.py §9c). Read it, do not re-derive.
        sdp = m["small_delivery_price"]
        val = float(sum(sdp[z, int(chosen[z]), d] for z in idx))
        return val, "sigma_small_delivery_price", ""
    return price_group_adapter(idx, d, m, "delivery", head,
                               parcels_by_cell=parcels_arr,
                               stops_by_cell=stops_arr, freq=1.0)


def price_express_group(idx: list[int], d: int, m: dict, head
                        ) -> tuple[float, str, str]:
    """An express group's predicted price — ``_hub_express_day_ml``'s."""
    if head is None:
        ec = m["express_cost"]
        return float(sum(ec[z, d] for z in idx)), "sigma_express_cost", ""
    return price_group_adapter(idx, d, m, "express", head, freq=1.0)


def enumerate_instances(item: int, P: float, th: float, plan: str, prov: str,
                        chosen: np.ndarray, od: dict, m: dict,
                        schedules: list, head) -> list[dict]:
    """Every VROOM instance the realistic-tour rule forms at this grid point.

    Mirrors ``63_bundle_sampler.extract_bundles``'s hub-day loops, but keeps
    1-member groups too (a lone small cell IS a tour, and a lone express cell
    IS a request) — 63_ drops them because the head is only trained on >= 2.

    The per-cell delivery tours are the cells the small-delivery rule did NOT
    pool: exactly the entries ``dd_cost_mx`` sums.
    """
    plz_keys = od["plz_keys"]
    hub_plz_list = od["hub_plz_list"]
    cost_3d = m["cost_3d"]
    veh_3d = m["veh_3d"]
    sched_active = m["sched_active"]
    sdm = m["small_delivery_mask"]
    cd = m["combined_demand"]
    cs = m["combined_stops"]
    raw_express = m["raw_express"]
    expr_stops = m["expr_stops"]
    hub_name_by_plz = od.get("_hub_name_by_plz", {})
    hub_of = {int(z): hi for hi, h in enumerate(hub_plz_list) for z in h}

    def _row(kind, vroom_kind, hi, d, idx, price, source, bin_name,
             n_routes, parcels, stops) -> dict:
        members = [str(plz_keys[z]).zfill(5) for z in idx]
        return dict(
            item=item, penalty=P, share_willing=th, plan=plan, provider=prov,
            hub_idx=int(hi), hub_name=str(hub_name_by_plz.get(members[0], "")),
            instance_kind=kind, vroom_kind=vroom_kind, day=int(d),
            n_members=len(idx), members=members, member_idx=[int(z) for z in idx],
            predicted_cost_eur=float(price), predicted_source=source,
            predicted_bin=bin_name, predicted_n_routes=float(n_routes),
            predicted_parcels=float(parcels), predicted_stops=float(stops),
        )

    out: list[dict] = []

    # ── per-cell delivery tours (unpooled, >= MIN_TOUR_PARCELS) ──────────
    for z in range(len(plz_keys)):
        si = int(chosen[z])
        for d in sorted(schedules[si]):
            if sdm[z, si, d] or cd[z, si, d] <= 0:
                continue
            out.append(_row(
                KIND_DELIVERY_SINGLE, "delivery", hub_of.get(z, -1), d, [z],
                float(cost_3d[z, si, d]), "cost_3d", "",
                float(veh_3d[z, si, d]), float(cd[z, si, d]),
                float(max(1.0, cs[z, si, d]))))

    # ── hub-day loops: pooled delivery groups + express groups ───────────
    for hi, h_ps in enumerate(hub_plz_list):
        if len(h_ps) == 0:
            continue
        for d in range(C.N_DAYS):
            small, _key = _smallday_members(hi, d, chosen, hub_plz_list, m)
            if small:
                parts, parcels_arr, stops_arr = _smallday_partition(
                    hi, d, chosen, small, m)
                for g in parts:
                    idx = sorted(int(z) for z in g)
                    price, source, bin_name = price_delivery_group(
                        idx, d, chosen, m, head, parcels_arr, stops_arr)
                    # spec §4.3 v3: a pooled group is ONE tour — members'
                    # parcels summed BEFORE the ceil, never rounded up each.
                    n_routes = _delivery_partition_vehicles((tuple(idx),),
                                                            parcels_arr)
                    parcels = float(np.trunc(
                        sum(np.trunc(parcels_arr[z]) for z in idx)))
                    stops = max(1.0, float(
                        sum(np.trunc(stops_arr[z]) for z in idx)))
                    out.append(_row(
                        KIND_DELIVERY_GROUP if len(idx) >= 2
                        else KIND_DELIVERY_SINGLE,
                        "delivery", hi, d, idx, price, source, bin_name,
                        n_routes, parcels, stops))

            contributing = _express_contributing(hi, d, chosen, hub_plz_list,
                                                 raw_express, m, schedules)
            if contributing:
                parts = _express_partition(contributing, d, raw_express,
                                           expr_stops, m)
                for g in parts:
                    idx = sorted(int(z) for z in g)
                    price, source, bin_name = price_express_group(idx, d, m, head)
                    n_routes = _express_partition_vehicles((tuple(idx),), d,
                                                           raw_express)
                    parcels = float(np.trunc(
                        sum(np.trunc(raw_express[z, d]) for z in idx)))
                    stops = max(1.0, float(
                        sum(np.trunc(expr_stops[z, d]) for z in idx)))
                    out.append(_row(
                        KIND_EXPRESS_GROUP if len(idx) >= 2
                        else KIND_EXPRESS_SINGLE,
                        "express", hi, d, idx, price, source, bin_name,
                        n_routes, parcels, stops))
    return out


def _express_contributing(hi: int, d: int, chosen: np.ndarray,
                          hub_plz_list: list, raw_express: np.ndarray,
                          matrices: dict, schedules: list) -> list[int]:
    """Non-delivering, express-carrying cells of hub-day (hi, d).

    Verbatim mirror of the mask inlined in ``_hub_express_day_ml`` — the same
    one ``63_bundle_sampler`` mirrors (costs.py has no standalone version).
    """
    h_ps = hub_plz_list[hi]
    if len(h_ps) == 0:
        return []
    sa = matrices.get("sched_active")
    if sa is not None:
        is_non_delivery = ~sa[chosen[h_ps], d]
    else:
        is_non_delivery = np.array(
            [d not in schedules[int(chosen[pi])] for pi in h_ps], dtype=bool)
    mask = is_non_delivery & (raw_express[h_ps, d] > 0)
    if not mask.any():
        return []
    return h_ps[mask].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Identity gates
# ─────────────────────────────────────────────────────────────────────────────

def identity_gate_routing(rows: list[dict], expected: float, tag: str,
                          rtol: float = 1e-6) -> float:
    """G1: Sigma predicted over the instance set == the grid's own total.

    ``expected`` is ``cost_stage2_eur`` (operator plan) or ``cost_stage1_eur``
    (routing plan) — both bundled routing totals (dd + express + pool) at
    their own plan. A mismatch means the reconstruction is not the instance
    set the grid priced; abort rather than validate the wrong object.
    """
    got = float(sum(float(r["predicted_cost_eur"]) for r in rows))
    tol = rtol * max(1.0, abs(float(expected)))
    assert abs(got - float(expected)) <= tol, (
        f"G1 routing identity FAILED for {tag}: Sigma predicted {got:.6f} != "
        f"grid {float(expected):.6f} (delta {got - float(expected):.3e}, "
        f"tol {tol:.3e}) over {len(rows)} instance(s)")
    return got


def identity_gate_fleet(rows: list[dict], fleet_df: pd.DataFrame, tag: str
                        ) -> None:
    """G2: per-instance predicted vehicles, per (hub, day), == the grid's fleet.

    ``tab_fleet_per_hub_v2.csv`` is written straight from the grid's
    ``_daily_fleet_per_hub``. Aggregating this script's per-instance counts
    onto (hub, day) must reproduce it exactly — the strongest available proof
    that the instance set IS the grid's tour set, not merely a set with the
    same total cost. Rows exist only for hub-days with >= 1 delivering cell,
    so the comparison is over the fleet table's own keys.
    """
    if fleet_df is None or fleet_df.empty:
        return
    got: dict[tuple, float] = {}
    for r in rows:
        got[(int(r["hub_idx"]), int(r["day"]))] = (
            got.get((int(r["hub_idx"]), int(r["day"])), 0.0)
            + float(r["predicted_n_routes"]))
    bad = []
    for rec in fleet_df.itertuples():
        key = (int(rec.hub_idx), int(rec.day))
        mine = got.get(key, 0.0)
        if abs(mine - float(rec.fleet)) > 1e-9:
            bad.append(f"hub_idx={key[0]} d={key[1]}: mine {mine} != "
                       f"grid {float(rec.fleet)}")
    assert not bad, (
        f"G2 fleet identity FAILED for {tag} ({len(bad)} hub-day(s)):\n  "
        + "\n  ".join(bad[:10]))


# ─────────────────────────────────────────────────────────────────────────────
# The cache census: exact keys under a per-process-randomised hash()
# ─────────────────────────────────────────────────────────────────────────────

def _vtw_for_hub(hub) -> tuple[int, int]:
    """``build_vroom_vehicles``'s vehicle time window for this hub.

    Duplicated from ``routing/requests.py`` on purpose — that module must not
    change — and gated per instance by ``_assert_offset_replication``, which
    compares the reconstruction against the vehicles the real builder returned.
    """
    is_small = hub.get("hub_typ", hub.get("Typ", "")) not in LARGE_HUB_TYPES
    if is_small:
        return VEHICLE_TIME_WINDOW[0] + SMALL_HUB_DELAY, VEHICLE_TIME_WINDOW[1]
    return VEHICLE_TIME_WINDOW[0], VEHICLE_TIME_WINDOW[1]


def _offsets_for_seed(k: int, day_idx: int, n_veh: int, max_off: int
                      ) -> np.ndarray:
    """The start-time offsets ``build_vroom_vehicles`` draws for ``hash%1e4 == k``."""
    rng = np.random.default_rng(VEH_START_SEED + day_idx * 1000 + k)
    offs = np.abs(rng.normal(0, VEH_START_SPREAD_S, size=n_veh))
    offs = np.clip(offs, 0, max_off).astype(int)
    offs.sort()
    return offs


#: ``(day, n_veh, max_off) -> (HASH_MOD, n_veh)`` offset table. Instances share
#: these three numbers heavily (six days, few distinct fleet sizes), so the
#: 10 000 RNG draws a probe needs are paid once per shape instead of per
#: instance — the difference between a ~40 min and a ~4 min census.
_OFFSET_TABLES: dict[tuple[int, int, int], np.ndarray] = {}
#: Bounded so a pathological spread of fleet sizes cannot eat the heap
#: (one table is 10 000 x n_veh int32).
_OFFSET_TABLE_CAP = 48


def _offset_table(day_idx: int, n_veh: int, max_off: int) -> np.ndarray:
    key = (int(day_idx), int(n_veh), int(max_off))
    hit = _OFFSET_TABLES.get(key)
    if hit is not None:
        return hit
    tab = np.empty((HASH_MOD, n_veh), dtype=np.int32)
    for k in range(HASH_MOD):
        tab[k] = _offsets_for_seed(k, day_idx, n_veh, max_off)
    if len(_OFFSET_TABLES) >= _OFFSET_TABLE_CAP:
        _OFFSET_TABLES.pop(next(iter(_OFFSET_TABLES)))
    _OFFSET_TABLES[key] = tab
    return tab


def _assert_offset_replication(vehicles: list[dict], seed_key: str,
                               day_idx: int, vtw: tuple[int, int]) -> int:
    """Prove the local reconstruction IS ``build_vroom_vehicles``'s draw.

    Returns ``k0 = hash(seed_key) % HASH_MOD`` — this process's candidate.
    """
    k0 = hash(seed_key) % HASH_MOD
    max_off = max(0, VEH_START_LATEST - vtw[0])
    offs = _offsets_for_seed(k0, day_idx, len(vehicles), max_off)
    real = [int(v["time_window"][0]) - vtw[0] for v in vehicles]
    assert real == [int(o) for o in offs], (
        "offset replication FAILED — routing/requests.build_vroom_vehicles no "
        f"longer matches _offsets_for_seed (seed_key={seed_key!r}, day={day_idx}, "
        f"n_veh={len(vehicles)}): real {real[:5]} vs reconstructed "
        f"{[int(o) for o in offs][:5]}")
    assert all(int(v["time_window"][1]) == vtw[1] for v in vehicles), (
        "vehicle time-window END differs from the reconstruction — "
        "_vtw_for_hub is out of date with routing/requests.py")
    return k0


def _canonical_parts(jobs: list[dict], vehicles: list[dict], vtw: tuple[int, int]
                     ) -> tuple[str, list[tuple[str, str]]]:
    """``(prefix, [(before, after), ...])`` for the canonical request string.

    ``routing/cache._request_hash`` hashes
    ``json.dumps(body, sort_keys=True, separators=(",", ":"))``. Top-level keys
    sort as ``jobs`` < ``vehicles``, so the string is
    ``{"jobs":<J>,"vehicles":[<V1>,...,<Vn>]}`` and only each ``Vi``'s
    ``time_window`` start moves between candidates. Each vehicle is serialised
    once with a sentinel start and split around it, so a candidate costs n
    string concatenations instead of a full re-serialisation.
    """
    sep = (",", ":")
    jobs_json = json.dumps(jobs, sort_keys=True, separators=sep)
    prefix = '{"jobs":' + jobs_json + ',"vehicles":'
    parts: list[tuple[str, str]] = []
    sentinel = -987654321
    for v in vehicles:
        tmp = dict(v)
        tmp["time_window"] = [sentinel, vtw[1]]
        s = json.dumps(tmp, sort_keys=True, separators=sep)
        token = str(sentinel)
        i = s.index(token)
        parts.append((s[:i], s[i + len(token):]))
    return prefix, parts


def _assemble(prefix: str, parts: list[tuple[str, str]], offsets, vtw0: int
              ) -> str:
    body = ",".join(a + str(vtw0 + int(o)) + b
                    for (a, b), o in zip(parts, offsets))
    return prefix + "[" + body + "]}"


def cache_probe(jobs: list[dict], vehicles: list[dict], hub, day_idx: int,
                seed_key: str, hash_index: dict[str, str],
                ) -> tuple[bool, str, int]:
    """Is ANY of the 10 000 possible bodies of this instance already solved?

    Returns ``(hit, tag, k)`` — ``tag`` is the cache directory the body was
    found under and ``k`` the candidate whose offsets produce it (so the solver
    can pin them and turn the hit into a real cache read).

    The search is exhaustive over the only process-dependent input
    (``hash(seed_key) % 10000``), and its correctness is gated twice: the real
    draw ``k0`` must be reproduced exactly, and the assembled canonical string
    for ``k0`` must hash to ``_request_hash`` of the real body.
    """
    vtw = _vtw_for_hub(hub)
    k0 = _assert_offset_replication(vehicles, seed_key, day_idx, vtw)
    prefix, parts = _canonical_parts(jobs, vehicles, vtw)
    max_off = max(0, VEH_START_LATEST - vtw[0])
    n_veh = len(vehicles)

    # Gate: the assembled string must BE the canonical form of the real body.
    real_offsets = [int(v["time_window"][0]) - vtw[0] for v in vehicles]
    assembled = _assemble(prefix, parts, real_offsets, vtw[0])
    want = _request_hash({"jobs": jobs, "vehicles": vehicles})
    got = hashlib.sha256(assembled.encode()).hexdigest()[:16]
    assert got == want, (
        "canonical-string assembly does not reproduce routing/cache."
        f"_request_hash ({got} != {want}) — the cache census would be blind")

    if not hash_index:
        return False, "", -1

    head_hash = hashlib.sha256(prefix.encode())
    table = _offset_table(day_idx, n_veh, max_off)
    seen: set[bytes] = set()
    order = [k0] + [k for k in range(HASH_MOD) if k != k0]
    for k in order:
        offs = table[k]
        key = offs.tobytes()
        if key in seen:                             # duplicate draw, same body
            continue
        seen.add(key)
        tail = ",".join(a + str(vtw[0] + int(o)) + b
                        for (a, b), o in zip(parts, offs))
        h = head_hash.copy()
        h.update(("[" + tail + "]}").encode())
        tag = hash_index.get(h.hexdigest()[:16])
        if tag is not None:
            return True, tag, k
    return False, "", -1


def build_hash_index(cache_root: Path) -> dict[str, str]:
    """``{request_hash: tag}`` over every cached solution on disk.

    One dict for the whole census: ~5 000 entries, and a candidate lookup is
    then a hash-table probe instead of a filesystem stat.
    """
    idx: dict[str, str] = {}
    if not cache_root.exists():
        return idx
    for d in sorted(cache_root.iterdir()):
        if not d.is_dir():
            continue
        for f in d.glob("*.json"):
            idx.setdefault(f.stem, d.name)
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Request construction (64_'s builder, never a copy)
# ─────────────────────────────────────────────────────────────────────────────

def default_cache_tag(inst: dict) -> str:
    """Where a solution of this instance kind is looked for and stored.

    Per-cell delivery tours inherit ``20_validate_vroom_smoothed``'s namespace
    (``smval_p{P}_s{theta:.1f}_{prov}``) and groups inherit 64_'s
    (``bundle_{prov}``), so previously-solved identical requests are hits.
    """
    if int(inst["n_members"]) == 1 and inst["vroom_kind"] == "delivery":
        return (f"smval_p{float(inst['penalty'])}"
                f"_s{float(inst['share_willing']):.1f}_{inst['provider']}")
    return f"bundle_{inst['provider']}"


def build_request(inst: dict, provider_data: dict, chosen_map: dict,
                  sched_by_idx: dict) -> dict:
    """``(pts, hub, jobs, vehicles, seed_key, note)`` for one instance.

    Points/hub come from ``64_``'s builder (which is ``20_``'s construction);
    jobs/vehicles from the shared routing helpers with no constant overridden.
    """
    members = (inst["members"] if isinstance(inst["members"], list)
               else json.loads(inst["members"]))
    row = SimpleNamespace(
        provider=str(inst["provider"]), members=[str(x).zfill(5) for x in members],
        kind=str(inst["vroom_kind"]), day=int(inst["day"]),
        first_seen=(float(inst["penalty"]), float(inst["share_willing"])),
        parcels=float(inst.get("predicted_parcels", 0.0)),
        n_members=int(inst["n_members"]),
        bundle_id=str(inst["instance_id"]),
    )
    hub, hub_names = B64.hub_row_for(row, provider_data)
    if hub is None:
        return dict(note="NO_HUB", pts=None, hub=None, jobs=None,
                    vehicles=None, seed_key="", hub_names=set())
    pts, note = B64.build_instance(row, provider_data, chosen_map, sched_by_idx)
    if pts is None:
        return dict(note=note, pts=None, hub=hub, jobs=None, vehicles=None,
                    seed_key="", hub_names=hub_names)
    jobs, total_demand = build_vroom_jobs(pts)
    if not jobs:
        return dict(note="NO_JOBS", pts=pts, hub=hub, jobs=None,
                    vehicles=None, seed_key="", hub_names=hub_names)
    seed_key = f"v2_{inst['instance_id']}"
    vehicles, n_veh = build_vroom_vehicles(
        hub=hub, total_demand=total_demand, day_idx=int(inst["day"]),
        seed_key=seed_key, n_jobs=len(jobs))
    return dict(note="OK", pts=pts, hub=hub, jobs=jobs, vehicles=vehicles,
                seed_key=seed_key, n_vehicles_planned=int(n_veh),
                hub_names=hub_names)


def instance_id(inst: dict) -> str:
    members = (inst["members"] if isinstance(inst["members"], list)
               else json.loads(inst["members"]))
    return (f"i{int(inst['item'])}|P{float(inst['penalty']):g}"
            f"|t{float(inst['share_willing']):g}|{inst['plan']}"
            f"|{inst['provider']}|{inst['vroom_kind']}|d{int(inst['day'])}"
            f"|{'+'.join(str(x).zfill(5) for x in members)}")


# ─────────────────────────────────────────────────────────────────────────────
# Budget arithmetic
# ─────────────────────────────────────────────────────────────────────────────

#: n_jobs bin edges of the empirical solve-time model.
TIME_BIN_EDGES = (50, 100, 200, 300, 400, 500)


def load_time_model(path: Path = THROUGHPUT_SRC) -> dict:
    """Mean solve seconds per ``n_jobs`` bin, measured on 64_'s Phase-A pool.

    Falls back to the brief's flat 60 s (single) / 150 s (bundle) estimate when
    no measurement is on disk — reported as such, never silently.
    """
    if not path.exists():
        return {"source": "brief fallback (60 s single / 150 s bundle)",
                "bins": None,
                "flat": {"single": FALLBACK_T_SINGLE_S,
                         "bundle": FALLBACK_T_BUNDLE_S}}
    df = pd.read_csv(path)
    df = df[df["solve_time_s"] > 0]
    edges = [0, *TIME_BIN_EDGES, 10 ** 9]
    bins: list[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = df[(df.n_jobs > lo) & (df.n_jobs <= hi)]
        if len(sub) == 0:
            continue
        bins.append({"lo": lo, "hi": hi, "n": int(len(sub)),
                     "mean_s": float(sub.solve_time_s.mean()),
                     "median_s": float(sub.solve_time_s.median()),
                     "p90_s": float(sub.solve_time_s.quantile(0.9))})
    assert bins, f"no usable solve-time rows in {path}"
    return {"source": f"{path.name} ({len(df)} solved instances, 64_ Phase A)",
            "bins": bins, "flat": None}


def est_solve_seconds(n_jobs: int, n_members: int, model: dict,
                      stat: str = "mean_s") -> float:
    """Projected wall seconds for ONE solve of this size."""
    if model.get("bins") is None:
        flat = model["flat"]
        return flat["bundle"] if int(n_members) > 1 else flat["single"]
    n_jobs = int(n_jobs)
    for b in model["bins"]:
        if b["lo"] < n_jobs <= b["hi"]:
            return float(b[stat])
    # Above every measured bin: scale the top bin's time linearly in n_jobs
    # from that bin's lower edge, rather than pretending it still holds.
    top = model["bins"][-1]
    return float(top[stat]) * max(1.0, n_jobs / max(1.0, float(top["lo"])))


def projected_hours(seconds: list[float], parallelism: int) -> float:
    """Wall hours for a queue of solves at *parallelism* concurrent requests.

    Concurrency is assumed perfect (VROOM is the bottleneck and 64_'s
    measurement at parallelism 3 was stable); the number is therefore a
    LOWER bound on wall time, stated as such in the census.
    """
    assert parallelism >= 1
    return float(sum(seconds)) / 3600.0 / float(parallelism)


# ─────────────────────────────────────────────────────────────────────────────
# G6 stratified fallback (item 3 only)
# ─────────────────────────────────────────────────────────────────────────────

def g6_select(rows: list[dict], demand_fraction: float = 0.5,
              n_full_providers: int = 3) -> set[str]:
    """Spec G6: full smallest-3 providers + >= 50 % of demand for the rest.

    Deterministic and stratified: within a sampled provider the instances are
    grouped by ``(instance_kind, n_jobs tercile)`` and drawn round-robin,
    largest-demand first inside each stratum, until the provider's selected
    parcels reach *demand_fraction* of its total. Round-robin across strata is
    what keeps small-instance strata represented — taking the largest
    instances outright would reach 50 % of demand with only the biggest tours
    and validate nothing about the rest.
    """
    by_prov: dict[str, list[dict]] = {}
    for r in rows:
        by_prov.setdefault(str(r["provider"]), []).append(r)
    totals = {p: sum(float(r["predicted_parcels"]) for r in rs)
              for p, rs in by_prov.items()}
    order = sorted(totals, key=lambda p: (totals[p], p))
    full = set(order[:n_full_providers])

    keep: set[str] = set()
    for prov, rs in by_prov.items():
        if prov in full:
            keep.update(str(r["instance_id"]) for r in rs)
            continue
        target = demand_fraction * totals[prov]
        jobs = sorted(float(r.get("n_jobs") or 0) for r in rs)
        if jobs:
            q1 = jobs[len(jobs) // 3]
            q2 = jobs[(2 * len(jobs)) // 3]
        else:                                       # pragma: no cover
            q1 = q2 = 0.0

        def _stratum(r: dict) -> tuple:
            nj = float(r.get("n_jobs") or 0)
            t = 0 if nj <= q1 else (1 if nj <= q2 else 2)
            return (str(r["instance_kind"]), t)

        strata: dict[tuple, list[dict]] = {}
        for r in rs:
            strata.setdefault(_stratum(r), []).append(r)
        for k in strata:
            strata[k].sort(key=lambda r: (-float(r["predicted_parcels"]),
                                          str(r["instance_id"])))
        got = 0.0
        keys = sorted(strata)
        while got < target and any(strata[k] for k in keys):
            for k in keys:
                if not strata[k]:
                    continue
                r = strata[k].pop(0)
                keep.add(str(r["instance_id"]))
                got += float(r["predicted_parcels"])
                if got >= target:
                    break
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# Operator lens on the ACTUAL side
# ─────────────────────────────────────────────────────────────────────────────

def operator_cost_actual(df: pd.DataFrame, fixed_cost_eur: float = FIXED_COST_EUR,
                         week_fixed_eur: float = WEEK_FIXED_COST_EUR) -> dict:
    """Rebuild the operator lens from solved instances.

    ``variable = Sigma (vroom_cost - 189.15 * n_routes)`` — every VROOM label
    carries one fixed charge per vehicle-DAY, while the operator's fixed bill
    is weekly and sized by each hub's PEAK day:

        peak_h  = max_d Sigma n_routes of hub h's instances on day d
        OpCost  = Sigma variable + 1134.90 * Sigma_h peak_h

    Expects columns ``hub_name``/``hub_idx``, ``day``, ``vroom_cost_eur``,
    ``vroom_n_routes``. Rows with a missing cost are dropped from ``variable``
    but their routes still count towards the peaks (an unpriced tour still
    needs its van) — both counts are returned so a report can state either.
    """
    if df is None or len(df) == 0:
        return {"variable_eur": 0.0, "sum_hub_peak": 0.0, "opcost_eur": 0.0,
                "vehicle_days": 0.0, "routing_eur": 0.0, "n": 0,
                "n_missing_cost": 0}
    d = df.copy()
    hub_key = "hub_name" if "hub_name" in d.columns else "hub_idx"
    d["vroom_n_routes"] = pd.to_numeric(d["vroom_n_routes"],
                                        errors="coerce").fillna(0.0)
    d["vroom_cost_eur"] = pd.to_numeric(d["vroom_cost_eur"], errors="coerce")
    priced = d[d["vroom_cost_eur"].notna()]
    variable = float((priced["vroom_cost_eur"]
                      - fixed_cost_eur * priced["vroom_n_routes"]).sum())
    per_hub_day = d.groupby([hub_key, "day"])["vroom_n_routes"].sum()
    sum_hub_peak = float(per_hub_day.groupby(level=0).max().sum())
    return {
        "variable_eur": variable,
        "sum_hub_peak": sum_hub_peak,
        "opcost_eur": variable + week_fixed_eur * sum_hub_peak,
        "vehicle_days": float(d["vroom_n_routes"].sum()),
        "routing_eur": float(priced["vroom_cost_eur"].sum()),
        "n": int(len(d)),
        "n_missing_cost": int(len(d) - len(priced)),
    }


def operator_cost_predicted(df: pd.DataFrame) -> dict:
    """The same lens on the PREDICTED side, from the same instance rows."""
    if df is None or len(df) == 0:
        return {"variable_eur": 0.0, "sum_hub_peak": 0.0, "opcost_eur": 0.0,
                "vehicle_days": 0.0, "routing_eur": 0.0, "n": 0}
    d = df.copy()
    hub_key = "hub_name" if "hub_name" in d.columns else "hub_idx"
    d["predicted_n_routes"] = pd.to_numeric(d["predicted_n_routes"],
                                            errors="coerce").fillna(0.0)
    d["predicted_cost_eur"] = pd.to_numeric(d["predicted_cost_eur"],
                                            errors="coerce").fillna(0.0)
    variable = float((d["predicted_cost_eur"]
                      - FIXED_COST_EUR * d["predicted_n_routes"]).sum())
    per_hub_day = d.groupby([hub_key, "day"])["predicted_n_routes"].sum()
    sum_hub_peak = float(per_hub_day.groupby(level=0).max().sum())
    return {
        "variable_eur": variable,
        "sum_hub_peak": sum_hub_peak,
        "opcost_eur": variable + WEEK_FIXED_COST_EUR * sum_hub_peak,
        "vehicle_days": float(d["predicted_n_routes"].sum()),
        "routing_eur": float(d["predicted_cost_eur"].sum()),
        "n": int(len(d)),
    }


def savings_vs_baseline(df: pd.DataFrame, base_item: int = 0) -> list[dict]:
    """Predicted vs ACTUAL saving %, both lenses, for every point vs item 0.

    Before item 0 existed there was no VROOM-solved daily baseline, so only a
    PREDICTED saving % could ever be stated (relative to ``cost_stage1_eur``
    at theta=0). This is the reconstruction that finally allows the actual
    number: for every OTHER (item, P, theta, plan) point present in *df*,

        pred saving %   = (base_pred  - point_pred)  / base_pred  * 100
        actual saving % = (base_act   - point_act)   / base_act   * 100

    in both the routing lens (``Sigma vroom_cost`` / ``Sigma predicted``) and
    the operator lens (``operator_cost_actual`` / ``operator_cost_predicted``,
    i.e. OpCost with the peak-fleet weekly-fixed term).

    PREDICTED totals never depend on solve status (they are read from the
    grid's own pricing path at census time, independently of whether VROOM
    ever ran), so they are always computed over every row of a group. ACTUAL
    totals are reported TWICE per the controller's "state both totals"
    instruction: "clean" pre-filters to OK/CACHED rows before computing
    anything (a PARTIAL tour is dropped entirely, including its vehicles from
    the peak); "all" passes every row straight to ``operator_cost_actual``,
    which keeps its nuanced treatment (a PARTIAL row's cost is excluded from
    ``variable`` but its ``vroom_n_routes`` still counts toward the peak — an
    unpriced tour still needs its van). The two can legitimately differ, most
    visibly in the operator lens's peak term; that difference IS the "with
    and without PARTIAL rows" the spec asks for, not a rounding artifact.

    Returns ``[]`` if item 0 has not been solved yet in *df* — an actual
    saving needs an actual baseline, which is the entire reason item 0 exists.
    """
    base_all = df[df["item"] == base_item]
    if base_all.empty:
        return []
    base_clean = base_all[base_all["vroom_status"].isin(["OK", "CACHED"])]
    bp = operator_cost_predicted(base_all)
    ba_all = operator_cost_actual(base_all)
    ba_clean = operator_cost_actual(base_clean)

    def _save_pct(base_val: float, point_val: float) -> float:
        return (base_val - point_val) / base_val * 100.0 if base_val else float("nan")

    out: list[dict] = []
    for keys, g in df[df["item"] != base_item].groupby(
            ["item", "penalty", "share_willing", "plan"], sort=True):
        g_clean = g[g["vroom_status"].isin(["OK", "CACHED"])]
        p = operator_cost_predicted(g)
        a_all = operator_cost_actual(g)
        a_clean = operator_cost_actual(g_clean)
        out.append(dict(
            item=int(keys[0]), penalty=float(keys[1]),
            share_willing=float(keys[2]), plan=str(keys[3]),
            n_all=int(len(g)), n_clean=int(len(g_clean)),
            base_n_all=int(len(base_all)), base_n_clean=int(len(base_clean)),
            pred_routing_save_pct=_save_pct(bp["routing_eur"], p["routing_eur"]),
            pred_opcost_save_pct=_save_pct(bp["opcost_eur"], p["opcost_eur"]),
            act_routing_save_pct_clean=_save_pct(ba_clean["routing_eur"],
                                                 a_clean["routing_eur"]),
            act_routing_save_pct_all=_save_pct(ba_all["routing_eur"],
                                               a_all["routing_eur"]),
            act_opcost_save_pct_clean=_save_pct(ba_clean["opcost_eur"],
                                                a_clean["opcost_eur"]),
            act_opcost_save_pct_all=_save_pct(ba_all["opcost_eur"],
                                              a_all["opcost_eur"]),
            base_routing_pred_eur=bp["routing_eur"],
            point_routing_pred_eur=p["routing_eur"],
            base_opcost_pred_eur=bp["opcost_eur"],
            point_opcost_pred_eur=p["opcost_eur"],
            base_routing_act_clean_eur=ba_clean["routing_eur"],
            point_routing_act_clean_eur=a_clean["routing_eur"],
            base_routing_act_all_eur=ba_all["routing_eur"],
            point_routing_act_all_eur=a_all["routing_eur"],
            base_opcost_act_clean_eur=ba_clean["opcost_eur"],
            point_opcost_act_clean_eur=a_clean["opcost_eur"],
            base_opcost_act_all_eur=ba_all["opcost_eur"],
            point_opcost_act_all_eur=a_all["opcost_eur"],
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Grid inputs
# ─────────────────────────────────────────────────────────────────────────────

def load_grid_tables(rev_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame,
                                             pd.DataFrame | None]:
    chosen_p = rev_dir / CHOSEN_CSV
    costs_p = rev_dir / COSTS_CSV
    assert chosen_p.exists(), f"no chosen table at {chosen_p}"
    assert costs_p.exists(), f"no cost table at {costs_p}"
    chosen = pd.read_csv(chosen_p, dtype={"plz": str})
    chosen["plz"] = chosen["plz"].astype(str).str.zfill(5)
    for col in PLAN_SCHED_COL.values():
        assert col in chosen.columns, (
            f"{chosen_p} lacks {col} — this is not a v5 two-plan table")
    costs = pd.read_csv(costs_p)
    for col in PLAN_COST_COL.values():
        assert col in costs.columns, f"{costs_p} lacks {col}"
    fleet_p = rev_dir / FLEET_CSV
    fleet = pd.read_csv(fleet_p) if fleet_p.exists() else None
    return chosen, costs, fleet


def chosen_array(chosen: pd.DataFrame, P: float, th: float, prov: str,
                 plan: str, plz_keys: list[str]) -> np.ndarray:
    col = PLAN_SCHED_COL[plan]
    sub = chosen[(np.isclose(chosen.penalty, P))
                 & (np.isclose(chosen.share_willing, th))
                 & (chosen.provider == prov)]
    assert len(sub), f"no chosen rows for P={P} th={th} {prov}"
    m = {str(r.plz).zfill(5): int(getattr(r, col)) for r in sub.itertuples()}
    missing = [pc for pc in plz_keys if str(pc).zfill(5) not in m]
    assert not missing, (
        f"chosen table misses {len(missing)} cell(s) for P={P} th={th} {prov} "
        f"(first: {missing[:5]}) — the grid block is incomplete")
    return np.array([m[str(pc).zfill(5)] for pc in plz_keys], dtype=np.int64)


def grid_cost(costs: pd.DataFrame, P: float, th: float, prov: str, plan: str
              ) -> float:
    sub = costs[(np.isclose(costs.penalty, P))
                & (np.isclose(costs.share_willing, th))
                & (costs.provider == prov)]
    assert len(sub) == 1, (
        f"expected exactly 1 cost row for P={P} th={th} {prov}, got {len(sub)}")
    return float(sub.iloc[0][PLAN_COST_COL[plan]])


def grid_express_cost(costs: pd.DataFrame, P: float, th: float, prov: str
                      ) -> float:
    """``express_cost_eur`` of the grid's own bundled cost split."""
    sub = costs[(np.isclose(costs.penalty, P))
                & (np.isclose(costs.share_willing, th))
                & (costs.provider == prov)]
    assert len(sub) == 1, (
        f"expected exactly 1 cost row for P={P} th={th} {prov}, got {len(sub)}")
    return float(sub.iloc[0]["express_cost_eur"])


def assert_baseline_invariant(chosen_df: pd.DataFrame, costs_df: pd.DataFrame,
                              prov: str, plz_keys: list[str]
                              ) -> tuple[np.ndarray, list[float]]:
    """Item 0's three baked-in invariants of the theta=0 grid block.

    theta=0 is a stage-2 NO-OP by construction (``61_grid_run_v2.py``'s
    G-6f-1): every cell is pinned to the daily schedule, independently of P
    (the wait penalty's ``local_willing`` term vanishes at theta=0, and stage
    1 hard-codes the daily index besides — see ``run_triple``'s
    ``if th == 0.0: chosen_s1 = np.full(...)``). Item 0 therefore builds VROOM
    instances at ONE representative P (0.0) instead of once per P. This
    function proves that shortcut is valid against the GRID'S OWN tables
    rather than assuming it, and aborts loudly if it is not:

      * ``schedule_idx_stage1 == schedule_idx_balanced`` at (P=0, theta=0)
        — the two plan columns item 0 could validate under are the same plan;
      * that array is identical for every OTHER P present in the grid at
        theta=0 — the P-invariance the singleton ``penalties=[0.0]`` relies on;
      * ``express_cost_eur == 0`` at (P=0, theta=0) — no cell has a
        non-delivery day to carry express demand on, so the baseline is
        delivery-only, exactly as ``enumerate_instances`` will independently
        find (no cell of an all-daily schedule is ever "non-delivering").

    Returns ``(chosen_stage1_array, [other P values checked])``.
    """
    s1 = chosen_array(chosen_df, 0.0, 0.0, prov, "stage1", plz_keys)
    bal = chosen_array(chosen_df, 0.0, 0.0, prov, "balanced", plz_keys)
    assert np.array_equal(s1, bal), (
        f"item 0 FAILED for {prov}: schedule_idx_stage1 != "
        f"schedule_idx_balanced at (P=0, theta=0) in "
        f"{int((s1 != bal).sum())} cell(s) — theta=0 is supposed to be a "
        "stage-2 no-op (G-6f-1); the baseline plan is not well-defined")

    sub = chosen_df[np.isclose(chosen_df.share_willing, 0.0)
                    & (chosen_df.provider == prov)]
    checked_p: list[float] = []
    for P in sorted(sub.penalty.unique()):
        if np.isclose(float(P), 0.0):
            continue
        arr = chosen_array(chosen_df, float(P), 0.0, prov, "stage1", plz_keys)
        assert np.array_equal(arr, s1), (
            f"item 0 FAILED for {prov}: theta=0 schedule at P={P:g} differs "
            f"from P=0 in {int((arr != s1).sum())} cell(s) — the baseline is "
            "supposed to be P-invariant at theta=0 (local_willing vanishes "
            "there for every P), so validating only P=0 would be wrong")
        checked_p.append(float(P))

    ecost = grid_express_cost(costs_df, 0.0, 0.0, prov)
    assert ecost == 0.0, (
        f"item 0 FAILED for {prov}: express_cost_eur={ecost:g} at "
        "(P=0, theta=0), expected exactly 0 — an all-daily schedule should "
        "leave no non-delivery day to carry express demand on")
    return s1, checked_p


def fleet_block(fleet: pd.DataFrame | None, P: float, th: float, prov: str,
                hub_names: list[str]) -> pd.DataFrame | None:
    """The grid's fleet rows for this triple, with ``hub_idx`` restored.

    ``tab_fleet_per_hub_v2.csv`` stores the hub NAME; G2 compares on the index,
    so the name is mapped back through this provider's hub order.
    """
    if fleet is None:
        return None
    sub = fleet[(np.isclose(fleet.penalty, P))
                & (np.isclose(fleet.share_willing, th))
                & (fleet.provider == prov)].copy()
    if sub.empty:
        return None
    if len(set(hub_names)) != len(hub_names):
        log(f"  WARNING: {prov} has duplicate hub names — G2 fleet identity "
            "cannot be keyed on the name; skipped for this triple")
        return None
    idx_of = {n: i for i, n in enumerate(hub_names)}
    sub["hub_idx"] = [idx_of.get(str(h), -1) for h in sub["hub"]]
    assert (sub["hub_idx"] >= 0).all(), (
        f"fleet table names a hub this provider does not have ({prov})")
    return sub


# ─────────────────────────────────────────────────────────────────────────────
# Census
# ─────────────────────────────────────────────────────────────────────────────

def merge_queue_rows(old: pd.DataFrame | None, fresh: pd.DataFrame,
                     items: list[int]) -> pd.DataFrame:
    """Union *fresh* (just (re)computed for *items*) onto *old*.

    Every OTHER item's rows in *old* — including their ``cache_hit`` /
    ``g6_selected`` flags — are carried through untouched; only the rows
    whose ``item`` is in *items* are replaced. A blind overwrite of
    ``instance_queue.csv`` on a partial census (e.g. ``--items 0`` to add the
    baseline onto a queue that already has items 1-3, some of them SOLVED in
    ``tab_vroom_v2.csv``) would desync the queue from the solved table:
    ``run_solve`` filters the queue by the requested items and would silently
    see "0 queued" for any item this run did not touch, believing it done.

    *old* is ``None`` the first time a census runs in a fresh output dir.
    """
    if old is None or old.empty:
        return fresh.reindex(columns=QUEUE_COLS)
    kept = old[~old["item"].isin(items)]
    merged = pd.concat([kept, fresh], ignore_index=True, sort=False)
    return merged.reindex(columns=QUEUE_COLS)


def run_census(args, rev_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Enumerate, price, gate, size and cache-probe every instance."""
    items = args.items
    chosen_df, costs_df, fleet_df = load_grid_tables(rev_dir)

    log("[load] checkpoints + model ...")
    t0 = time.perf_counter()
    provider_data, optim_data = C.load_checkpoints()
    model = C.load_model()
    ml_prep = C.build_ml_prep(provider_data)
    schedules = C.enumerate_schedules()
    assert len(schedules) == 39, f"expected 39 schedules, got {len(schedules)}"
    sched_by_idx = {i: sorted(s) for i, s in enumerate(schedules)}
    log(f"[load] done in {time.perf_counter() - t0:.0f}s")

    head = None
    if args.head:
        from batch_delivery.surrogate.bundle import BundleHead
        head = BundleHead.load(Path(args.head))
        log(f"[head] installed {args.head} "
            f"(restricted={getattr(head, 'restricted', '?')})")

    hash_index = build_hash_index(RESULTS_DIR / "cache")
    log(f"[cache] {len(hash_index)} cached solution(s) indexed over "
        f"{len(set(hash_index.values()))} tag(s)")

    tmodel = load_time_model()
    log(f"[time-model] {tmodel['source']}")

    providers = ([p.strip() for p in args.providers.split(",")]
                 if args.providers else list(C.PROVIDERS))

    # (theta, plan) blocks share a matrices build; P varies inside.
    blocks: dict[float, list[tuple[int, float, str]]] = {}
    for it in items:
        spec = ITEMS[it]
        for P in spec["penalties"]:
            blocks.setdefault(spec["theta"], []).append((it, P, spec["plan"]))

    rows: list[dict] = []
    for th in sorted(blocks):
        for prov in providers:
            od = optim_data.get(prov)
            prep = ml_prep.get(prov)
            if od is None or prep is None:
                log(f"  skip {prov}: no optimization data")
                continue
            od = dict(od)
            od["_hub_name_by_plz"] = prep["hub_name_by_plz"]
            hub_names = [
                prep["hub_name_by_plz"].get(od["plz_keys"][int(h[0])], f"hub_{hi}")
                if len(h) else f"hub_{hi}"
                for hi, h in enumerate(od["hub_plz_list"])
            ]
            t_m = time.perf_counter()
            m = build_cost_matrices_ml(
                od["plz_keys"], od["plz_data"], schedules, model, prov,
                prep["plz_day_coords"], prep["hub_coords_by_plz"],
                fast_share_b2c=C.fs_b2c(th), fast_share_b2b=C.fs_b2b(th))
            if head is not None:
                m["bundle_head"] = head
            log(f"[mtx] th={th:g} {prov:<7s} built in "
                f"{time.perf_counter() - t_m:.1f}s")

            for (it, P, plan) in blocks[th]:
                if it == 0:
                    _, checked_p = assert_baseline_invariant(
                        chosen_df, costs_df, prov, od["plz_keys"])
                    log(f"  [item 0] {prov}: daily baseline is P-invariant "
                        f"(checked P={checked_p} against P=0) and "
                        "stage1==balanced, express_cost_eur==0 at theta=0")
                chosen = chosen_array(chosen_df, P, th, prov, plan,
                                      od["plz_keys"])
                inst = enumerate_instances(it, P, th, plan, prov, chosen, od,
                                           m, schedules, head)
                if it == 0:
                    assert not any(r["vroom_kind"] == "express" for r in inst), (
                        f"item 0 FAILED for {prov}: the all-daily baseline "
                        "enumeration produced an express instance — no cell "
                        "should have a non-delivery day to carry one on")
                tag = f"item {it} P={P:g} th={th:g} {plan} {prov}"
                total = identity_gate_routing(
                    inst, grid_cost(costs_df, P, th, prov, plan), tag)
                if plan == "balanced" or it == 0:
                    # item 0's plan is labelled "stage1", but at theta=0
                    # stage1 == balanced == the fleet table's final plan
                    # (assert_baseline_invariant just proved the first
                    # equality; system_smooth_pass is off by default and a
                    # no-op here regardless, so the fleet table's chosen_s3
                    # coincides too) — G2 is exactly as meaningful here as for
                    # the "balanced" items.
                    identity_gate_fleet(
                        inst, fleet_block(fleet_df, P, th, prov, hub_names), tag)
                for r in inst:
                    r["hub_name"] = hub_names[int(r["hub_idx"])] \
                        if 0 <= int(r["hub_idx"]) < len(hub_names) else ""
                    r["instance_id"] = instance_id(r)
                log(f"  [gate OK] {tag}: {len(inst)} instance(s), "
                    f"Sigma predicted {total:,.2f} EUR")
                rows.extend(inst)
            del m
            gc.collect()

    # ── requests, sizes and the cache probe ──────────────────────────────
    log(f"[probe] building {len(rows)} request(s) and probing the cache ...")
    t_p = time.perf_counter()
    for i, r in enumerate(rows):
        req = build_request(r, provider_data, chosen_map_for(chosen_df, r),
                            sched_by_idx)
        r["build_note"] = req["note"]
        r["instance_parcels"] = (float(req["pts"]["dhl_total"].sum())
                                 if req["pts"] is not None else 0.0)
        r["instance_stops"] = int(len(req["pts"])) if req["pts"] is not None else 0
        r["n_jobs"] = len(req["jobs"]) if req["jobs"] else 0
        r["n_vehicles_planned"] = int(req.get("n_vehicles_planned") or 0)
        r["cache_tag"] = default_cache_tag(r)
        r["g6_selected"] = True
        if req["jobs"]:
            hit, htag, k = cache_probe(req["jobs"], req["vehicles"], req["hub"],
                                       int(r["day"]), req["seed_key"],
                                       hash_index)
            r["cache_hit"], r["cache_hit_tag"], r["cache_seed_k"] = hit, htag, k
            r["est_solve_s"] = 0.0 if hit else est_solve_seconds(
                r["n_jobs"], r["n_members"], tmodel)
        else:
            r["cache_hit"], r["cache_hit_tag"], r["cache_seed_k"] = False, "", -1
            r["est_solve_s"] = 0.0
        if (i + 1) % 200 == 0:
            el = time.perf_counter() - t_p
            log(f"  [probe] {i + 1}/{len(rows)}  el={el:.0f}s  "
                f"eta={el * (len(rows) - i - 1) / (i + 1):.0f}s")
    log(f"[probe] done in {time.perf_counter() - t_p:.0f}s")

    df = pd.DataFrame(rows)
    df["members"] = df["members"].apply(json.dumps)
    df["member_idx"] = df["member_idx"].apply(json.dumps)

    # ── G6 fallback for item 3, pre-computed so the controller can approve ──
    # (only meaningful when item 3 is actually part of THIS run — see the
    # merge note below for what happens to it otherwise.)
    item3 = [r for r in rows if int(r["item"]) == 3]
    g6_keep_fresh = g6_select(item3) if item3 else set()
    if item3:
        df.loc[df["item"] == 3, "g6_selected"] = df.loc[
            df["item"] == 3, "instance_id"].isin(g6_keep_fresh)
    if args.g6_fallback and item3:
        log(f"[G6] --g6-fallback: item 3 restricted to {len(g6_keep_fresh)} "
            f"of {len(item3)} instance(s)")
    df = df.reindex(columns=QUEUE_COLS)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── merge onto any existing queue — NEVER overwrite other items' rows ──
    # A partial census (e.g. `--items 0` to add the baseline onto a queue
    # that already has items 1-3, some of them SOLVED in tab_vroom_v2.csv)
    # must only replace the rows for the item(s) it just (re)computed. A
    # blind overwrite here would desync instance_queue.csv from
    # tab_vroom_v2.csv: a resumed `run_solve` filters the queue by
    # ``args.items`` and would silently see "0 queued" for every item this
    # run did not touch, believing it complete.
    queue_p = out_dir / QUEUE_CSV
    old = pd.read_csv(queue_p) if queue_p.exists() else None
    if old is not None:
        kept_preview = old[~old["item"].isin(items)]
        if len(kept_preview):
            log(f"[census] preserving {len(kept_preview)} existing row(s) "
                f"for item(s) "
                f"{sorted(int(i) for i in kept_preview['item'].unique())} "
                "not in this run")
    merged = merge_queue_rows(old, df, items)
    merged.to_csv(queue_p, index=False)
    log(f"[census] wrote {queue_p} ({len(merged)} row(s) total, "
        f"{len(df)} (re)computed this run)")

    # g6_keep for the REPORT must reflect the MERGED table's own column, not
    # just this run's fresh rows: a census that does not touch item 3 (e.g.
    # `--items 0`) must not blank out a previously-computed G6 selection.
    item3_merged = merged[merged["item"] == 3]
    g6_keep = (set(item3_merged.loc[item3_merged["g6_selected"].astype(bool),
                                    "instance_id"])
              if len(item3_merged) else set())

    write_census_report(merged, out_dir, args, tmodel, g6_keep)
    return merged


_CHOSEN_MAP_CACHE: dict[tuple, dict] = {}


def chosen_map_for(chosen_df: pd.DataFrame, r: dict) -> dict:
    """``(P, theta, provider, plz) -> schedule_idx`` for this row's PLAN.

    64_'s builder takes the map, not a column, so the plan under validation is
    injected here — the same builder therefore serves both plans without a
    second code path.
    """
    key = (round(float(r["penalty"]), 4), round(float(r["share_willing"]), 4),
           str(r["plan"]))
    hit = _CHOSEN_MAP_CACHE.get(key)
    if hit is not None:
        return hit
    col = PLAN_SCHED_COL[str(r["plan"])]
    sub = chosen_df[(np.isclose(chosen_df.penalty, float(r["penalty"])))
                    & (np.isclose(chosen_df.share_willing,
                                  float(r["share_willing"])))]
    out = {(key[0], key[1], str(x.provider), str(x.plz).zfill(5)):
           int(getattr(x, col)) for x in sub.itertuples()}
    _CHOSEN_MAP_CACHE[key] = out
    return out


def write_census_report(df: pd.DataFrame, out_dir: Path, args, tmodel: dict,
                        g6_keep: set) -> None:
    """census.csv (per item x kind) + census.md (the controller's decision page)."""
    par = int(args.parallelism)
    recs: list[dict] = []
    for (item, kind), g in df.groupby(["item", "instance_kind"], sort=True):
        new = g[~g["cache_hit"].astype(bool)]
        recs.append({
            "item": int(item), "instance_kind": kind,
            "n_instances": int(len(g)),
            "n_cache_hits": int(g["cache_hit"].astype(bool).sum()),
            "n_new": int(len(new)),
            "n_unbuildable": int((g["build_note"] != "OK").sum()),
            "median_n_jobs": float(g["n_jobs"].median()),
            "max_n_jobs": int(g["n_jobs"].max()) if len(g) else 0,
            "est_hours_new": projected_hours(
                new["est_solve_s"].tolist(), par),
        })
    cen = pd.DataFrame(recs)
    tot = cen.groupby("item", as_index=False).agg(
        n_instances=("n_instances", "sum"), n_cache_hits=("n_cache_hits", "sum"),
        n_new=("n_new", "sum"), est_hours_new=("est_hours_new", "sum"))
    cen.to_csv(out_dir / CENSUS_CSV, index=False)

    total_h = float(tot["est_hours_new"].sum())
    g6 = df[(df["item"] == 3) & (df["g6_selected"].astype(bool))
            & (~df["cache_hit"].astype(bool))]
    g6_h = projected_hours(g6["est_solve_s"].tolist(), par)
    item3_h = float(tot.loc[tot["item"] == 3, "est_hours_new"].sum()) \
        if (tot["item"] == 3).any() else 0.0

    reported_items = sorted(int(i) for i in df["item"].unique())
    md: list[str] = []
    md.append("# VROOM validation v2 — cache census\n")
    md.append(f"- grid: `{args.rev_dir}`")
    md.append(f"- items (this run): {', '.join(str(i) for i in args.items)}")
    if reported_items != sorted(args.items):
        md.append(f"- items (reported below, incl. preserved rows from "
                  f"earlier runs): {', '.join(str(i) for i in reported_items)}")
    md.append(f"- parallelism assumed: **{par}** concurrent VROOM requests")
    md.append(f"- solve-time model: {tmodel['source']}")
    if tmodel.get("bins"):
        md.append("  - mean seconds by `n_jobs` bin: "
                  + ", ".join(f"({b['lo']},{b['hi']}]→{b['mean_s']:.0f}s "
                              f"(n={b['n']})" for b in tmodel["bins"]))
    md.append(f"- hard budget: **{BUDGET_HOURS:.0f} VROOM-hours** for items "
              "1–3 together\n")
    md.append("Projected hours assume perfect concurrency "
              "(`Σ est_solve_s / 3600 / parallelism`) and are therefore a "
              "LOWER bound on wall time.\n")

    md.append("## Per item and instance kind\n")
    md.append("| item | kind | instances | cache hits | new | median n_jobs "
              "| max n_jobs | est. h (new) |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in cen.itertuples():
        md.append(f"| {r.item} | {r.instance_kind} | {r.n_instances} | "
                  f"{r.n_cache_hits} | {r.n_new} | {r.median_n_jobs:.0f} | "
                  f"{r.max_n_jobs} | {r.est_hours_new:.2f} |")
    md.append("")
    md.append("| item | description | instances | cache hits | new | est. h |")
    md.append("|---|---|---:|---:|---:|---:|")
    for r in tot.itertuples():
        md.append(f"| {r.item} | {ITEMS[int(r.item)]['label']} | "
                  f"{r.n_instances} | {r.n_cache_hits} | {r.n_new} | "
                  f"{r.est_hours_new:.2f} |")
    md.append(f"| **all** | | **{int(tot.n_instances.sum())}** | "
              f"**{int(tot.n_cache_hits.sum())}** | "
              f"**{int(tot.n_new.sum())}** | **{total_h:.2f}** |\n")

    if total_h > BUDGET_HOURS:
        md.append(f"> **OVER BUDGET**: {total_h:.2f} h > {BUDGET_HOURS:.0f} h. "
                  "Items 1–2 are never sampled; the G6 stratified fallback "
                  "below applies to item 3 only.\n")
    else:
        md.append(f"> Within budget ({total_h:.2f} h ≤ "
                  f"{BUDGET_HOURS:.0f} h) — no sampling needed.\n")

    md.append("## G6 stratified fallback for item 3 (pre-computed)\n")
    md.append("Full smallest-3 providers by demand + ≥ 50 % of demand for the "
              "rest, drawn round-robin over `(kind, n_jobs tercile)` strata.\n")
    md.append(f"- item 3 full: {int((df['item'] == 3).sum())} instances, "
              f"{item3_h:.2f} h of new solves")
    md.append(f"- item 3 under G6: {len(g6_keep)} instances selected, "
              f"{len(g6)} of them new → **{g6_h:.2f} h**")
    # This label is item-set-aware (not hardcoded "1+2"): `total_h` is a sum
    # over whatever items are actually being reported below (see
    # `reported_items` above), which need not be {1, 2, 3} — e.g. item 0
    # joins it on a merged census.
    non3_items = [i for i in reported_items if i != 3]
    non3_label = "+".join(str(i) for i in non3_items) if non3_items else "(none)"
    md.append(f"- items {non3_label} + G6 item 3: "
              f"**{total_h - item3_h + g6_h:.2f} h**\n")
    md.append("Apply with `--g6-fallback` (writes the sampling note into the "
              "output header); never applied to items 1–2.\n")

    md.append("## Unbuildable instances\n")
    bad = df[df["build_note"] != "OK"]
    if bad.empty:
        md.append("None — every instance produced VROOM jobs.\n")
    else:
        md.append("| item | provider | kind | note | n |")
        md.append("|---|---|---|---|---:|")
        for k, g in bad.groupby(["item", "provider", "instance_kind",
                                 "build_note"]):
            md.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {len(g)} |")
        md.append("")

    md.append("## Cache-hit provenance\n")
    hits = df[df["cache_hit"].astype(bool)]
    if hits.empty:
        md.append("No cached body matched any of this instance set.\n")
    else:
        md.append("| tag on disk | hits |")
        md.append("|---|---:|")
        for tag, g in hits.groupby("cache_hit_tag"):
            md.append(f"| `{tag}` | {len(g)} |")
        md.append("")
        md.append("A hit is only usable if the request carries the cached "
                  "vehicle start offsets (`hash(seed_key)` is per-process "
                  "randomised): the solver pins them, recording "
                  "`offsets_pinned`. `--no-offset-pinning` re-solves instead.\n")

    (out_dir / CENSUS_MD).write_text("\n".join(md), encoding="utf-8")
    log(f"[census] wrote {out_dir / CENSUS_MD}")
    log(f"[census] TOTAL {int(tot.n_instances.sum())} instances, "
        f"{int(tot.n_cache_hits.sum())} cache hits, "
        f"{int(tot.n_new.sum())} new → {total_h:.2f} h at parallelism {par}")


# ─────────────────────────────────────────────────────────────────────────────
# Solve phase
# ─────────────────────────────────────────────────────────────────────────────

def load_done(path: Path) -> set[str]:
    """``instance_id``s that need no further attempt (retryable ones excluded)."""
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "instance_id" not in df.columns:
        raise SystemExit(f"{path} has no instance_id column — refusing to "
                         "append onto an incompatible table")
    keep = df[~df["vroom_status"].astype(str).isin(RETRYABLE)]
    return set(keep["instance_id"].astype(str))


def solve_instance(inst: dict, provider_data: dict, chosen_df: pd.DataFrame,
                   sched_by_idx: dict, idx: int, total: int,
                   pin_offsets: bool) -> dict:
    """One instance -> one row. Never raises for a routing failure."""
    t0 = time.perf_counter()
    base = {c: inst.get(c) for c in QUEUE_COLS}
    base.update({
        "vroom_cost_eur": np.nan, "vroom_n_routes": 0,
        "vroom_distance_km": np.nan, "vroom_duration_h": np.nan,
        "vroom_n_parcels": 0, "vroom_status": "ERROR", "n_unassigned": 0,
        "jobs_removed": 0, "parcels_removed": 0, "n_vehicles_final": 0,
        "offsets_pinned": False, "solve_time_s": 0.0,
        "solved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    req = build_request(inst, provider_data, chosen_map_for(chosen_df, inst),
                        sched_by_idx)
    base["build_note"] = req["note"]
    if not req["jobs"]:
        base["vroom_status"] = req["note"]
        base["solve_time_s"] = round(time.perf_counter() - t0, 3)
        return base

    jobs, vehicles, hub = req["jobs"], req["vehicles"], req["hub"]
    base["instance_parcels"] = float(req["pts"]["dhl_total"].sum())
    base["instance_stops"] = int(len(req["pts"]))
    base["n_jobs"] = len(jobs)
    base["n_vehicles_planned"] = int(req.get("n_vehicles_planned") or 0)

    tag = str(inst.get("cache_hit_tag") or "") or default_cache_tag(inst)
    k = int(inst.get("cache_seed_k") or -1)
    if pin_offsets and bool(inst.get("cache_hit")) and k >= 0:
        # Turn the census's hit into a real cache read: the body must carry the
        # offsets the cached request had. Noise draw only — see the module
        # docstring — recorded per row.
        vtw = _vtw_for_hub(hub)
        offs = _offsets_for_seed(k, int(inst["day"]), len(vehicles),
                                 max(0, VEH_START_LATEST - vtw[0]))
        for v, o in zip(vehicles, offs):
            v["time_window"] = [vtw[0] + int(o), vtw[1]]
        base["offsets_pinned"] = True

    members = (json.loads(inst["members"]) if isinstance(inst["members"], str)
               else inst["members"])
    label = "|".join(sorted(str(x).zfill(5) for x in members))
    result = solve_single_plz(
        plz_code=label, request_body={"jobs": jobs, "vehicles": vehicles},
        idx=idx, total=total,
        day_name=CDAYS[int(inst["day"])] if int(inst["day"]) < len(CDAYS) else "",
        cache_tag=tag)
    base.update({
        "cache_tag": tag,
        "vroom_status": str(result["status"]),
        "vroom_cost_eur": (float(result["cost"]) / COST_SCALE
                           if result["cost"] is not None else np.nan),
        "vroom_n_routes": int(result["n_routes"] or 0),
        "vroom_distance_km": (float(result["distance_m"]) / 1000.0
                              if result["distance_m"] is not None else np.nan),
        "vroom_duration_h": (float(result["duration_s"]) / 3600.0
                             if result["duration_s"] is not None else np.nan),
        "vroom_n_parcels": int(base["instance_parcels"]),
        "n_unassigned": int(result["n_unassigned"] or 0),
        "jobs_removed": int(result["jobs_removed"] or 0),
        "parcels_removed": int(result["parcels_removed"] or 0),
        "n_vehicles_final": int(result["n_vehicles"] or 0),
        "solve_time_s": round(time.perf_counter() - t0, 3),
    })
    return base


def run_solve(args, out_dir: Path) -> None:
    queue_p = out_dir / QUEUE_CSV
    assert queue_p.exists(), (
        f"no instance queue at {queue_p} — run with --census-only first "
        "(the census is what sizes the budget the controller approves)")
    q = pd.read_csv(queue_p)
    q = q[q["item"].isin(args.items)]
    if args.providers:
        q = q[q["provider"].isin([p.strip() for p in args.providers.split(",")])]
    if args.g6_fallback:
        q = q[(q["item"] != 3) | (q["g6_selected"].astype(bool))]
        n3 = int((q["item"] == 3).sum())
        log(f"[G6] stratified fallback active for item 3 - {n3} instance(s) kept")
        # The sampling note travels with the OUTPUT, not just the log: a reader
        # of validation_report.md must not have to know how it was run.
        (out_dir / G6_NOTE_MD).write_text(
            "## Sampling note (spec G6)\n\n"
            "Item 3 was NOT validated exhaustively. The G6 stratified fallback "
            "was applied: every instance of the three smallest providers by "
            "demand, plus >= 50 % of each remaining provider's demand drawn "
            "round-robin over `(instance_kind, n_jobs tercile)` strata, "
            "largest instance first inside a stratum. Items 1 and 2 are "
            f"complete. Item 3 rows kept: {n3}.\n",
            encoding="utf-8")
    q = q[q["build_note"] == "OK"]

    solved_p = out_dir / SOLVED_CSV
    done = load_done(solved_p)
    pending = [r for r in q.to_dict("records")
               if str(r["instance_id"]) not in done]
    log(f"[solve] {len(q)} queued, {len(done)} already done, "
        f"{len(pending)} to solve")
    if not pending:
        log("[solve] nothing to do")
        return

    if not _health_check():
        raise SystemExit("VROOM is not healthy on localhost:3000 — start the "
                         "routing services (docker compose up -d) first")

    provider_data, _optim = C.load_checkpoints()
    chosen_df, _c, _f = load_grid_tables(Path(args.rev_dir))
    sched_by_idx = {i: sorted(s) for i, s in enumerate(C.enumerate_schedules())}

    t0 = time.perf_counter()
    n_done = 0
    buf: list[dict] = []
    with ThreadPoolExecutor(max_workers=int(args.parallelism)) as ex:
        futs = {
            ex.submit(solve_instance, r, provider_data, chosen_df,
                      sched_by_idx, i + 1, len(pending),
                      not args.no_offset_pinning): r
            for i, r in enumerate(pending)
        }
        for fut in as_completed(futs):
            try:
                row = fut.result()
            except Exception as e:                  # keep the queue moving
                r = futs[fut]
                row = {c: r.get(c) for c in QUEUE_COLS}
                row.update({"vroom_status": "ERROR", "solve_time_s": 0.0,
                            "solved_at": time.strftime("%Y-%m-%d %H:%M:%S")})
                log(f"  ERROR {r.get('instance_id')}: {type(e).__name__}: {e}")
            buf.append(row)
            n_done += 1
            if len(buf) >= 5 or n_done == len(pending):
                append_rows(solved_p, buf, SOLVED_COLS)
                buf = []
            if n_done % PROGRESS_EVERY == 0:
                el = time.perf_counter() - t0
                log(f"  [{n_done}/{len(pending)}] el={el / 60:.1f}min "
                    f"eta={el * (len(pending) - n_done) / n_done / 60:.1f}min")
            if n_done % HEALTH_EVERY == 0 and not _health_check():
                log("  WARNING: VROOM health check failed mid-run")
    if buf:
        append_rows(solved_p, buf, SOLVED_COLS)
    log(f"[solve] done in {(time.perf_counter() - t0) / 60:.1f}min")


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def diagnostics(pred: np.ndarray, act: np.ndarray) -> dict:
    err = pred - act
    denom = np.maximum(1e-6, np.abs(act))
    return {"n": int(len(act)),
            "MAPE_pct": float(np.mean(np.abs(err) / denom) * 100) if len(act) else np.nan,
            "bias_pct": float(np.mean(err / denom) * 100) if len(act) else np.nan,
            "sum_pred": float(pred.sum()), "sum_act": float(act.sum()),
            "sum_gap_pct": float((pred.sum() - act.sum())
                                 / max(1e-6, act.sum()) * 100) if len(act) else np.nan}


def write_report(out_dir: Path, args) -> None:
    solved_p = out_dir / SOLVED_CSV
    if not solved_p.exists():
        log("  no solved rows yet — nothing to aggregate")
        return
    df = pd.read_csv(solved_p)
    clean = df[df["vroom_status"].isin(["OK", "CACHED"])].copy()
    flagged = df[~df["vroom_status"].isin(["OK", "CACHED"])].copy()

    md = ["# VROOM validation v2 — report\n",
          f"- grid: `{args.rev_dir}`",
          f"- rows: {len(df)} ({len(clean)} clean, {len(flagged)} flagged: "
          + ", ".join(f"{k}={v}" for k, v in
                      flagged["vroom_status"].value_counts().items()) + ")",
          "\nFlagged rows (PARTIAL / unassigned / jobs removed) are EXCLUDED "
          "from MAPE and INCLUDED in the totals; both totals are stated.\n"]
    note_p = out_dir / G6_NOTE_MD
    if note_p.exists():
        md.append(note_p.read_text(encoding="utf-8"))

    md.append("## Routing lens — per plan/point and instance kind\n")
    md.append("| item | P | θ | plan | kind | n | MAPE % | bias % | "
              "Σ pred € | Σ actual € | gap % |")
    md.append("|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for keys, g in clean.groupby(["item", "penalty", "share_willing", "plan",
                                  "instance_kind"]):
        d = diagnostics(g["predicted_cost_eur"].values,
                        g["vroom_cost_eur"].values)
        md.append(f"| {keys[0]} | {keys[1]:g} | {keys[2]:g} | {keys[3]} | "
                  f"{keys[4]} | {d['n']} | {d['MAPE_pct']:.2f} | "
                  f"{d['bias_pct']:+.2f} | {d['sum_pred']:,.0f} | "
                  f"{d['sum_act']:,.0f} | {d['sum_gap_pct']:+.2f} |")
    md.append("")

    md.append("## Both lenses — per plan/point\n")
    md.append("| item | P | θ | plan | routing pred € | routing act € | "
              "variable pred € | variable act € | Σ peak pred | Σ peak act | "
              "OpCost pred € | OpCost act € | ΔOpCost % |")
    md.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for keys, g in df.groupby(["item", "penalty", "share_willing", "plan"]):
        p = operator_cost_predicted(g)
        a = operator_cost_actual(g)
        gap = (p["opcost_eur"] - a["opcost_eur"]) / max(1e-6, a["opcost_eur"]) * 100
        md.append(f"| {keys[0]} | {keys[1]:g} | {keys[2]:g} | {keys[3]} | "
                  f"{p['routing_eur']:,.0f} | {a['routing_eur']:,.0f} | "
                  f"{p['variable_eur']:,.0f} | {a['variable_eur']:,.0f} | "
                  f"{p['sum_hub_peak']:,.0f} | {a['sum_hub_peak']:,.0f} | "
                  f"{p['opcost_eur']:,.0f} | {a['opcost_eur']:,.0f} | "
                  f"{gap:+.2f} |")
    md.append("")
    md.append("`variable = Σ (vroom_cost − 189.15 · n_routes)`; "
              "`peak_h = max_d Σ n_routes of hub h`; "
              "`OpCost = Σ variable + 1134.90 · Σ_h peak_h`.\n")

    md.append("## Predicted vs actual saving vs the theta=0 baseline (item 0)\n")
    srows = savings_vs_baseline(df)
    if not srows:
        md.append("Item 0 (the theta=0 daily baseline) has not been solved "
                  "yet in this output dir — an ACTUAL saving % needs an "
                  "actual baseline, so only the predicted saving % (relative "
                  "to `cost_stage1_eur`, see the grid tables directly) can be "
                  "stated until it is.\n")
    else:
        b = srows[0]
        md.append(f"Baseline (item 0, n={b['base_n_all']}) totals — "
                  f"predicted: routing {b['base_routing_pred_eur']:,.0f} EUR, "
                  f"OpCost {b['base_opcost_pred_eur']:,.0f} EUR; actual "
                  f"clean (n={b['base_n_clean']}, OK/CACHED only): routing "
                  f"{b['base_routing_act_clean_eur']:,.0f} EUR, OpCost "
                  f"{b['base_opcost_act_clean_eur']:,.0f} EUR; actual incl. "
                  f"PARTIAL (n={b['base_n_all']}): routing "
                  f"{b['base_routing_act_all_eur']:,.0f} EUR, OpCost "
                  f"{b['base_opcost_act_all_eur']:,.0f} EUR.\n")
        md.append("| item | P | θ | plan | n (all/clean) | pred save % "
                  "routing | actual save % routing (clean) | actual save % "
                  "routing (all) | pred save % OpCost | actual save % OpCost "
                  "(clean) | actual save % OpCost (all) |")
        md.append("|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|")
        for r in srows:
            md.append(
                f"| {r['item']} | {r['penalty']:g} | {r['share_willing']:g} | "
                f"{r['plan']} | {r['n_all']}/{r['n_clean']} | "
                f"{r['pred_routing_save_pct']:+.2f} | "
                f"{r['act_routing_save_pct_clean']:+.2f} | "
                f"{r['act_routing_save_pct_all']:+.2f} | "
                f"{r['pred_opcost_save_pct']:+.2f} | "
                f"{r['act_opcost_save_pct_clean']:+.2f} | "
                f"{r['act_opcost_save_pct_all']:+.2f} |")
        md.append("")
        md.append("Saving % = `(baseline − point) / baseline * 100` in each "
                  "lens; predicted is read from the grid's pricing path "
                  "(status-independent); actual \"clean\" drops PARTIAL rows "
                  "entirely (incl. from the peak-fleet term), \"all\" keeps "
                  "`operator_cost_actual`'s treatment (PARTIAL excluded from "
                  "cost, its vehicles still counted in the peak) — the two "
                  "are the \"with and without PARTIAL rows\" totals.\n")
    (out_dir / REPORT_MD).write_text("\n".join(md), encoding="utf-8")
    log(f"[report] wrote {out_dir / REPORT_MD}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rev-dir", default=str(DEFAULT_REV_DIR),
                    help="grid directory holding _tab_chosen_v2.csv")
    ap.add_argument("--out-dir", default=None,
                    help="default: <rev-dir>/validation")
    ap.add_argument("--items", default="1,2,3",
                    help="comma-separated item id(s): 0=theta=0 daily "
                         "baseline (Task 12b), 1-3 per the controller "
                         "amendment; default 1,2,3")
    ap.add_argument("--providers", default=None,
                    help="comma-separated subset (smoke runs only)")
    ap.add_argument("--census-only", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--parallelism", type=int, default=3)
    ap.add_argument("--head", default=None,
                    help="bundle_head.pkl to price groups with (Task 11 grid)")
    ap.add_argument("--g6-fallback", action="store_true",
                    help="restrict item 3 to the G6 stratified subset")
    ap.add_argument("--no-offset-pinning", action="store_true",
                    help="do not reuse cached vehicle start offsets (re-solves)")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    args.items = [int(x) for x in str(args.items).split(",") if x.strip()]
    bad = [i for i in args.items if i not in ITEMS]
    if bad:
        raise SystemExit(f"unknown item(s) {bad}; known: {sorted(ITEMS)}")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    assert_clean_tree(args.allow_dirty)
    rev_dir = Path(args.rev_dir)
    if not rev_dir.is_absolute():
        rev_dir = C.ROOT / rev_dir
    out_dir = Path(args.out_dir) if args.out_dir else rev_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    log("=" * 72)
    log(f"67 VROOM validation v2 — items {args.items} on {rev_dir}")
    log("=" * 72)

    lock = InstanceLock(out_dir)
    lock.acquire()

    if args.report_only:
        write_report(out_dir, args)
        return
    if not (out_dir / QUEUE_CSV).exists() or args.census_only:
        run_census(args, rev_dir, out_dir)
    if args.census_only:
        log("[done] --census-only: nothing was solved")
        return
    run_solve(args, out_dir)
    write_report(out_dir, args)


if __name__ == "__main__":
    main()
