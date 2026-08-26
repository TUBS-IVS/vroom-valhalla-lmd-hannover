"""62: gate checks G1a / G1b / G3 / G4 for the v2 grid (read-only, live-safe).

The full base grid (``61_grid_run_v2.py``) runs for hours and appends to the
four CSVs in ``results/revision_2026_08/``. This script verifies that grid
WHILE it is running, without ever touching the live files directly:

  HARD RULE: never ``pd.read_csv`` the four live files in
  ``results/revision_2026_08/`` directly. A reader lock on one of them killed
  a writer overnight before this script existed. Every read in this script
  goes through :func:`refresh_copies`, which copies them (with the same
  retry/backoff the writer uses for its own appends) into
  ``results/revision_2026_08/_gatecopy/`` first. That copy step is reusable
  by any future script with the same coexistence requirement, not just this
  one.

  COMPLETION MARKER: the grid writes ``_tab_chosen_v2.csv`` LAST per triple
  (see ``61_grid_run_v2.py``'s own docstring), so a ``(penalty,
  share_willing, provider)`` triple is treated as complete iff it is present
  in the copy of that file, regardless of what partial rows the other three
  files may hold for the same triple.

Gates (brief: task-7-brief.md):

  G1a  theta=1 rows, cells whose EVERY instance is >= MIN_TOUR_PARCELS
       (230): ``schedule_idx_stage1`` must equal the per-cell argmin of
       ``dd_cost_mx + penalty*wait``. HARD-FAIL on any mismatch.

       The canonical run (``_stage3_common.RUN_DIR /
       _tab_chosen_with_system_smoothing.csv``) has no stage-1 column, so
       this recomputes it -- not from separate "old" matrices, but from the
       SAME theta=1 matrices the v2 grid itself built (this is therefore a
       SELF-consistency check against a value forced by an identity, not an
       independent cross-check against a differently-computed canonical
       source -- see the structural caveat printed alongside G1a in the
       report). That is valid, not a shortcut. The argument has two parts:

       (1) THE MATRIX IDENTITY. At theta=1, fast_share_b2c = fast_share_b2b
       = 0 (``_stage3_common.fs_b2c/fs_b2b``), so ``raw_express`` is
       identically zero for every cell and ``express_demand`` (which only
       ever lands on NON-delivery days) is zero too -- meaning
       ``combined_demand`` is zero on every non-delivery day, so
       ``active = combined_demand > 0`` never fires there either, and
       ``cost_3d`` only ever receives a prediction on a DELIVERY day. Since
       ``dd_cost_mx = (cost_3d * sched_active).sum(axis=2)`` already
       restricts to delivery days, masking a matrix that is already zero
       off them is a no-op: ``cost_3d_raw.sum(axis=2) == dd_cost_mx`` holds
       for EVERY cell at theta=1, not only eligible ones (``cost_3d_raw``
       is ``cost_3d`` copied before the small-delivery zeroing, so this
       also shows the zeroing changes nothing on cells it never touches).
       This is checked at runtime, not just asserted in prose: every
       theta=1 matrix build in this script runs
       ``assert np.allclose(m["cost_3d_raw"].sum(axis=2), m["dd_cost_mx"])``
       (see :func:`get_matrices`) -- if a future change breaks the
       identity, G1a's premise is invalidated loudly before it can produce
       a silent false result.

       (2) THE CD FIXED POINT. For a cell whose every instance is >=230
       parcels, ``small_delivery_mask`` never fires (by the >=230
       criterion), so it is never pool-coupled to a hub either -- combined
       with (1), such a cell has ZERO express/pool coupling to any other
       cell's schedule choice at theta=1. Reading
       ``coordinate_descent.py``'s inner loop (``optimize_cd_ml``) shows
       what this means for the STORED ``schedule_idx_stage1`` (which is
       CD's output, not the raw warm start, whenever theta > 0): for such a
       cell, every candidate move's ``delta`` reduces exactly to
       ``dd_cost_mx[pi, new_si] - dd_cost_mx[pi, old_si]`` (the express/pool
       correction terms are computed from ``affected``/``pool_affected``
       days, both empty here). CD's acceptance test is STRICT
       (``if delta < best_delta``, initial ``best_delta = 0.0``), and it
       scans ``new_si`` in ascending order over ALL 39 schedules on every
       visit to the cell (no restricted neighbourhood) -- so in a single
       visit it finds the new_si with the smallest ``dd_cost_mx[pi, new_si]``
       (ties broken by first-occurrence in ascending order, exactly
       ``np.argmin``'s own tie convention) and moves there if and only if it
       beats the cell's CURRENT value; if the current value already IS that
       minimum, no candidate can produce a strictly negative delta, so it
       is correctly kept. This is a ONE-VISIT fixed point independent of
       where the warm start (with its own 0.5%-near-tied /
       prefer-larger-schedule heuristic) put ``old_si`` -- so that
       heuristic's choice is provably irrelevant to the converged answer
       for these cells, it only affects unrelated (non-eligible or
       non-theta=1) cells' starting points. The ``theta == 0.0 -> force
       daily`` override in ``run_triple`` is separately unreachable here by
       construction: G1a only ever examines theta=1 rows, so that branch's
       condition is never true in this code path. Net effect: CD's stored
       result on a G1a-eligible theta=1 cell IS ``np.argmin(dd_cost_mx[pi,
       :] + penalty_mx[pi, :])``, not merely close to it -- which is why a
       mismatch is a hard fail rather than a tolerance check. The one
       remaining theoretical edge case is an EXACT bit-for-bit tie between
       the warm start's ``old_si`` and a DIFFERENT index sharing the same
       minimal value -- CD's strict ``<`` would then leave ``old_si`` in
       place even if it is not ``np.argmin``'s first-occurrence choice; this
       requires two schedules' ML-predicted costs to be bit-identical, not
       merely close, which a continuous surrogate does not produce in
       practice.

  G1b  same cells: diff ``schedule_idx_system_smoothed`` (v2, post
       balancing+smoothing) against the canonical run's OWN
       ``schedule_idx_system_smoothed`` column (a real column, no fallback
       needed). REPORT ONLY -- stage 2/3 are allowed to move even a
       decoupled cell for fleet-balancing reasons the old pipeline's
       non-express-aware balancer did not have, so a diff here is not a
       correctness bug. Diff rows carry the cell's hub as the swap's likely
       scope; attributing the exact triggering hub-day swap would require
       re-instrumenting the balancer's own log, which this script does not
       do -- flagged explicitly in the report rather than overclaimed.

  G3   for every (P, theta=1) point present in the grid's OWN copied
       ``_tab_chosen_v2.csv`` (all of them, not just a hardcoded four -- the
       brief says "don't hardcode", so this derives the theta=1 penalty
       list from what the grid has actually produced, which is also the
       more useful partial-grid-aware reading than the canonical table's
       full P list) plus (0.5, 0.5) and (0.5, 0.1) if present: rebuild the
       chosen vector from the copy and recompute the fleet exactly the way
       ``61_grid_run_v2.py``'s (post-Task-6c) output loop does:

           pool_cache, express_cache = {}, {}
           def _ev(hi, d, ch): return _hub_express_vehicles(hi, d, ch, ...)
           def _dv(hi, d, ch): return _hub_delivery_pool_vehicles(hi, d, ch, ...)
           def _pv(hi, d, ch): return _ev(hi, d, ch) + _dv(hi, d, ch)
           deliv = h_ps[m["sched_active"][chosen[h_ps], d]]      # MASKED
           dd_single = veh_3d[deliv, chosen[deliv], d].sum()
           dd_pool = _dv(hi, d, chosen); ex_veh = _ev(hi, d, chosen)
           fleet = dd_single + dd_pool + ex_veh
           fleet_arr = _daily_fleet_per_hub(chosen, plz_hub_arr, hub_plz_list,
               veh_3d, schedules, pool_veh_fn=_pv, sched_active=m["sched_active"])

       The per-cell delivery term is masked to DELIVERY days
       (``m["sched_active"][chosen[h_ps], d]``) before summing ``veh_3d`` --
       omitting that mask is exactly the double-count Task 6c fixed
       (commit ``8da1a86``): ``veh_3d`` still carries a cell's express
       residual on its non-delivery days, so an unmasked sum adds it a
       second time on top of ``express_veh``. Compare all FOUR fields
       (``dd_single_veh``, ``dd_pool_veh``, ``express_veh``, ``fleet``)
       against ``tab_fleet_per_hub_v2.csv`` EXACTLY -- that CSV's schema
       changed with the same fix (it no longer has a ``dd_veh`` column at
       all); a copy that still has ``dd_veh`` is pre-Task-6c data and G3
       reports "stale schema, re-run required" for it rather than comparing
       column-by-column against a schema that no longer matches what this
       script computes.

       Task 6b's report proved ``pandas.read_csv``'s C float parser is not
       correctly-rounded at the last digit and reported a false "exact" on
       this exact kind of comparison. This script reads the fleet CSV copy
       with the stdlib ``csv`` module (raw strings) and converts with the
       ``float()`` builtin, which -- unlike pandas' fast parser -- performs
       a correctly-rounded string-to-double conversion, i.e. it recovers
       the exact bit pattern ``DataFrame.to_csv`` wrote. Comparison is then
       plain ``==`` between two float64 values obtained by construction
       (the recomputed side) or by correctly-rounded parsing (the CSV
       side), which is exact by the above -- not a tolerance check.

  G4   baseline corridor: force every cell of every provider onto the daily
       schedule (fast_share=1.0, i.e. theta=0 semantics) and recompute
       dd + express + pool. The daily schedule has no non-delivery days, so
       the express twin is structurally zero (confirmed, not assumed -- see
       the report); small-delivery pooling can still fire per cell/day.
       Summed over all 7 providers, hard-fail only below the corridor's
       lower bound (1 895 580); >1% above the upper bound (1 909 748)
       requires a written explanation in the report, not a fail.

Partial-grid aware: every gate first asks the copied ``_tab_chosen_v2.csv``
which triples are actually done and only pays for a matrix build where
there is something to check. A triple counts as done only if its block in
the copy has EXACTLY ``len(optim_data[prov]["plz_keys"])`` rows -- the
runner's own resume convention since Task 6c: an interrupted run can leave
a short, partially-appended block for whichever triple was in flight, and
treating that as complete would silently under- or over-select the cell set
every downstream gate reconstructs from it. Re-running this script later,
as more triples land, picks up more coverage automatically -- there is no
persistent state outside the four source CSVs.

Run: ``.venv\\Scripts\\python.exe scripts/revision/62_gates_check.py``
Output: ``results/revision_2026_08/gates_report.md``
Exit code: 0 unless a HARD gate (G1a, G3, or G4-lower-bound) actually failed
(as opposed to "not yet available", which also exits 0 per the brief).
"""
from __future__ import annotations

import csv as csv_mod
import gc
import shutil
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # LGBM feature-name notices, one per predict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _stage3_common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "src"))
from batch_delivery.config.constants import MIN_TOUR_PARCELS  # noqa: E402
from batch_delivery.optimization.core import build_cost_matrices_ml  # noqa: E402
from batch_delivery.optimization.balancing import _daily_fleet_per_hub  # noqa: E402
from batch_delivery.optimization.costs import (  # noqa: E402
    _hub_delivery_pool_vehicles,
    _hub_express_day_ml,
    _hub_express_vehicles,
    _hub_smallday_pool_ml,
)
from batch_delivery.optimization.schedules import enumerate_valid_schedules  # noqa: E402

import logging  # noqa: E402
logging.disable(logging.INFO)  # silence the package's INFO/DEBUG chatter

assert MIN_TOUR_PARCELS == 230.0, (
    f"G1a's >=230 criterion is pinned to MIN_TOUR_PARCELS; got {MIN_TOUR_PARCELS}")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

LIVE_DIR = C.ROOT / "results" / "revision_2026_08"
COPY_DIR = LIVE_DIR / "_gatecopy"
LIVE_FILES = (
    "_tab_chosen_v2.csv", "tab_costs_v2.csv",
    "tab_fleet_per_hub_v2.csv", "tab_wait_v2.csv",
)
REPORT_PATH = LIVE_DIR / "gates_report.md"

# The brief names ``results/oracle_loop_extended_2026_05_22/...`` as the
# canonical table; that directory does not exist in the current tree (it
# predates the 2026-05-31 refactor). The actual canonical run every sibling
# script in this revision (``_stage3_common.py``, ``61_grid_run_v2.py``,
# ``10_recompute_stage3_outputs.py``) reads is ``_stage3_common.RUN_DIR``,
# which does exist with the expected filename -- used here instead, and
# called out explicitly in the report for auditability.
CANONICAL_CSV = C.RUN_DIR / "_tab_chosen_with_system_smoothing.csv"

# G4 corridor, from the brief (task-7-brief.md Step 1, G4).
G4_LOWER = 1_895_580.0
G4_UPPER = 1_909_748.0
G4_MARGIN = 0.01

# Dated, prunable notes about known issues in the SOURCE the grid was
# produced by (not in this script) that affect how a gate's PASS should be
# read. G3 recomputes fleet numbers with the same production functions
# (`_hub_express_vehicles`, `_hub_delivery_pool_vehicles`, `_daily_fleet_per_hub`)
# that wrote ``tab_fleet_per_hub_v2.csv`` in the first place -- if those
# functions have a bug, G3 reproduces it on both sides of the comparison and
# passes vacuously. Remove an entry once the underlying issue is fixed AND
# the grid has been re-run past the affected triples.
#
# HISTORY (empty as of 2026-08-26, both conditions now met -- see
# task-7-report.md for the full account): a fleet double-count bug (express
# vehicles counted twice -- once in the per-cell delivery fleet via
# unmasked `veh_3d`, once again in the hub express fleet) was fixed across
# two commits, `8da1a86` (mask `veh_3d` to delivery days) and `a128e1a`
# (pooled delivery tours count by partition vehicles, spec v3 Section 4.3,
# `_daily_fleet_per_hub`'s `express_veh_fn` renamed to `pool_veh_fn`,
# `tab_fleet_per_hub_v2.csv`'s `dd_veh` column split into `dd_single_veh` +
# `dd_pool_veh`). The grid that produced the FIRST 159 triples predated
# both commits; it was killed and relaunched from an empty CSV on the
# fixed, committed code, so every row currently in the live CSVs is
# post-fix and this list is empty. G3's own schema check (module docstring)
# additionally refuses to compare against a `dd_veh`-schema CSV outright, so
# a regression to the old schema cannot pass silently even without an entry
# here.
KNOWN_ISSUES: list[str] = []


# ─────────────────────────────────────────────────────────────────────────────
# Copy-first IO -- never read the four live files directly (see module doc)
# ─────────────────────────────────────────────────────────────────────────────

def _retry_copy(src: Path, dst: Path) -> None:
    """``shutil.copy2`` with the same retry/backoff the writer uses for its
    own appends (``61_grid_run_v2._retry_write``): up to 60 attempts, 5 s
    apart (~5 min), on a transient Windows ``PermissionError``.
    """
    last_err: Exception | None = None
    for attempt in range(60):
        try:
            shutil.copy2(src, dst)
            return
        except PermissionError as e:
            last_err = e
            if attempt == 0:
                print(f"  WARNING: {src.name} locked ({e}); retrying up to 5 min",
                      flush=True)
            time.sleep(5)
    raise last_err  # type: ignore[misc]


def refresh_copies() -> dict[str, Path]:
    """Copy the four live grid-output CSVs into ``_gatecopy/`` and return the
    paths that actually exist. Safe to call at any point in the grid's
    lifetime, including before it has produced anything.
    """
    COPY_DIR.mkdir(parents=True, exist_ok=True)
    copies: dict[str, Path] = {}
    for name in LIVE_FILES:
        src = LIVE_DIR / name
        if not src.exists():
            continue
        dst = COPY_DIR / name
        _retry_copy(src, dst)
        copies[name] = dst
    return copies


def _key(P: float, th: float, prov: str) -> tuple[float, float, str]:
    """Same rounding convention as ``61_grid_run_v2._key`` -- the triple
    identity the grid itself uses for its own resume bookkeeping."""
    return (round(float(P), 4), round(float(th), 4), str(prov))


def load_chosen_v2(path: Path | None) -> pd.DataFrame:
    cols = ["penalty", "share_willing", "provider", "plz",
            "schedule_idx_stage1", "schedule_idx_balanced",
            "schedule_idx_system_smoothed"]
    if path is None or not path.exists():
        return pd.DataFrame(columns=cols)
    # plz as str: PLZ codes are opaque identifiers, not integers to compute
    # with. Reading them as int64 and converting back with str(int(x)) round-
    # trips correctly for this dataset (no leading zeros in the Region
    # Hannover PLZ range), but it is an assumption; reading as str from the
    # start removes it entirely and matches optim_data[prov]["plz_keys"]'s
    # own str type (verified against the checkpoint) with no conversion.
    return pd.read_csv(path, dtype={"plz": str})


def load_done_triples(chosen_df: pd.DataFrame,
                      optim_data: dict) -> set[tuple[float, float, str]]:
    """A triple counts as done only if its block has EXACTLY
    ``len(optim_data[prov]["plz_keys"])`` rows -- 61_'s own resume
    convention since Task 6c. The four per-triple appends are not atomic; a
    kill between them (or, before Task 6c's own self-heal runs on the next
    resume, mid-append) can leave a SHORT block for the one triple that was
    in flight. Counting that as done would hand every downstream gate an
    incomplete cell set for that triple -- silently wrong, not just
    incomplete -- so short blocks are treated as not-yet-available (skipped)
    rather than done or failed.
    """
    if chosen_df.empty:
        return set()
    expected = {prov: len(od["plz_keys"]) for prov, od in optim_data.items()}
    counts = chosen_df.groupby(
        ["penalty", "share_willing", "provider"]).size()
    done = set()
    for (P, th, prov), n in counts.items():
        if n == expected.get(prov):
            done.add(_key(P, th, prov))
    return done


def filter_complete(df: pd.DataFrame, done_triples: set) -> pd.DataFrame:
    """Restrict *df* to rows whose (penalty, share_willing, provider) is in
    *done_triples* -- applied once, at the source, right after
    :func:`load_done_triples`, so every gate that reads ``chosen_df``
    downstream automatically only ever sees complete triples without having
    to re-derive completeness itself.
    """
    if df.empty or not done_triples:
        return df.iloc[0:0]
    keys = list(zip(df["penalty"].round(4), df["share_willing"].round(4),
                    df["provider"]))
    mask = [k in done_triples for k in keys]
    return df[mask]


def _daily_demand_only(od: dict) -> np.ndarray:
    """Per-cell, per-day raw demand (b2c + b2b), shape (n_plz, N_DAYS).

    A standalone copy of just the first few lines of
    ``build_cost_matrices_ml``'s ``daily_demand`` computation (costs.py
    Section 2) -- independent of theta/fast_share by construction (it reads
    only ``plz_data[pc]["b2c"/"b2b"]``, never ``fast_share_*``), so it lets
    the report state G1a's full cell-eligibility SCOPE (which cells could
    ever qualify, across all 7 providers) without paying for a full
    ``build_cost_matrices_ml`` call (surrogate predictions, tier-2 geometry,
    ...) for providers the grid has not reached theta=1 for yet.
    """
    plz_keys = od["plz_keys"]
    plz_data = od["plz_data"]
    daily_demand = np.zeros((len(plz_keys), C.N_DAYS), dtype=np.float64)
    for pi, pc in enumerate(plz_keys):
        pd_ = plz_data[pc]
        for d in range(C.N_DAYS):
            daily_demand[pi, d] = pd_["b2c"].get(d, 0) + pd_["b2b"].get(d, 0)
    return daily_demand


def load_fleet_v2_exact(path: Path | None) -> tuple[dict[tuple, dict], int, str]:
    """Read the fleet CSV copy as raw strings via the stdlib ``csv`` module
    (see module docstring for why -- pandas' C float parser is not
    correctly-rounded at the last digit). Returns
    ``(index, row_count, schema)`` where ``index`` is keyed by
    ``(penalty, share_willing, provider, hub, day)`` -> raw-string row dict,
    and ``schema`` is one of:

    * ``"new"``   -- has ``dd_single_veh``/``dd_pool_veh``/``express_veh``
                     (post-Task-6c, what this script's G3 compares against).
    * ``"stale"`` -- has ``dd_veh`` instead (pre-Task-6c). G3 must not
                     compare column-by-column against this: the columns do
                     not correspond 1:1 (the old ``fleet`` double-counted
                     express vehicles the new ``fleet`` does not).
    * ``"empty"`` -- the file does not exist yet.
    """
    index: dict[tuple, dict] = {}
    n = 0
    if path is None or not path.exists():
        return index, n, "empty"
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        fieldnames = set(reader.fieldnames or ())
        if "dd_veh" in fieldnames:
            schema = "stale"
        elif {"dd_single_veh", "dd_pool_veh", "express_veh"} <= fieldnames:
            schema = "new"
        else:
            schema = "unknown"
        for row in reader:
            n += 1
            if schema != "new":
                continue  # do not build a comparison index for a schema we won't compare
            key = (round(float(row["penalty"]), 4), round(float(row["share_willing"]), 4),
                   row["provider"], row["hub"], int(row["day"]))
            index[key] = row
    return index, n, schema


# ─────────────────────────────────────────────────────────────────────────────
# Matrices (cached per (theta, provider)) + the penalty-matrix formula
# ─────────────────────────────────────────────────────────────────────────────

def get_matrices(cache: dict, theta: float, prov: str, optim_data: dict,
                 ml_prep: dict, model, schedules: list) -> dict:
    key = (round(float(theta), 4), prov)
    if key in cache:
        return cache[key]
    od, prep = optim_data[prov], ml_prep[prov]
    fs_b2c_v, fs_b2b_v = C.fs_b2c(theta), C.fs_b2b(theta)
    t0 = time.perf_counter()
    m = build_cost_matrices_ml(
        od["plz_keys"], od["plz_data"], schedules, model, prov,
        prep["plz_day_coords"], prep["hub_coords_by_plz"],
        fast_share_b2c=fs_b2c_v, fast_share_b2b=fs_b2b_v)
    assert m.get("bundle_head") is None, (
        "base run must price with the Sigma fallback (head=None)")
    assert m.get("small_delivery_price") is not None, (
        "matrices lack 'small_delivery_price' -- would silently fall back "
        "to slow per-member partition pricing")
    if abs(float(theta) - 1.0) < 1e-9:
        # Runtime proof of G1a's fallback premise (module docstring, part
        # (1)): at theta=1 every cell's raw_express is zero, so cost_3d only
        # ever gets a prediction on a delivery day and the pre-zeroing copy
        # (cost_3d_raw) sums to exactly dd_cost_mx for EVERY cell. If this
        # ever breaks, G1a's argmin fallback is comparing against the wrong
        # thing -- fail loudly here rather than let G1a report a silently
        # meaningless PASS or FAIL.
        assert np.allclose(m["cost_3d_raw"].sum(axis=2), m["dd_cost_mx"]), (
            f"theta=1 matrix identity violated for {prov}: "
            "cost_3d_raw.sum(axis=2) != dd_cost_mx -- G1a's fallback "
            "premise (module docstring) no longer holds")
    print(f"    [mtx] th={theta:<4g} {prov:<7s} built in "
          f"{time.perf_counter() - t0:5.1f}s", flush=True)
    cache[key] = m
    return m


def _penalty_mx(m: dict, od: dict, sched_waits: np.ndarray,
                P: float, th: float) -> np.ndarray:
    """Verbatim copy of the penalty wiring in
    ``61_grid_run_v2.run_triple`` (lines ~193-206): weekly parcel counts per
    cell x per-cell local-willing fraction x P x per-schedule average wait.
    """
    plz_keys = od["plz_keys"]
    plz_data = od["plz_data"]
    n_plz = len(plz_keys)
    fs_b2c_v, fs_b2b_v = C.fs_b2c(th), C.fs_b2b(th)
    weekly_pkts = np.array([
        sum(plz_data[pc]["b2c"].values()) + sum(plz_data[pc]["b2b"].values())
        for pc in plz_keys
    ], dtype=np.float64)
    plz_b2c_share = m.get("plz_b2c_share")
    if plz_b2c_share is not None:
        local_willing = (plz_b2c_share * (1.0 - fs_b2c_v)
                         + (1.0 - plz_b2c_share) * (1.0 - fs_b2b_v))
    else:
        local_willing = np.full(n_plz, th)
    return P * local_willing[:, None] * weekly_pkts[:, None] * sched_waits[None, :]


# ─────────────────────────────────────────────────────────────────────────────
# G1a / G1b
# ─────────────────────────────────────────────────────────────────────────────

def run_g1a_g1b(chosen_df: pd.DataFrame, canonical_df: pd.DataFrame,
                done_triples: set, optim_data: dict, ml_prep: dict, model,
                schedules: list, sched_waits: np.ndarray, mtx_cache: dict) -> dict:
    out = dict(
        available=False, providers_available=[], P_values=[],
        g1a_status="SKIPPED", g1a_cells_checked=0, g1a_mismatches=[],
        g1b_cells_checked=0, g1b_diffs=[],
    )
    theta1 = chosen_df[np.isclose(chosen_df["share_willing"], 1.0)]
    if theta1.empty:
        return out

    providers_avail = sorted(theta1["provider"].unique())
    P_values = sorted(float(p) for p in theta1["penalty"].unique())
    out["available"] = True
    out["providers_available"] = providers_avail
    out["P_values"] = P_values

    canon_theta1 = canonical_df[np.isclose(canonical_df["share_willing"], 1.0)]
    has_weekdays_col = "weekdays_system_smoothed" in canonical_df.columns

    mismatches: list[dict] = []
    diffs: list[dict] = []
    n_g1a_checked = 0
    n_g1b_checked = 0

    for prov in providers_avail:
        od, prep = optim_data[prov], ml_prep[prov]
        m = get_matrices(mtx_cache, 1.0, prov, optim_data, ml_prep, model, schedules)
        plz_keys = od["plz_keys"]

        eligible_mask = m["daily_demand"].min(axis=1) >= MIN_TOUR_PARCELS
        eligible = {plz_keys[i]: i for i in np.where(eligible_mask)[0]}
        if not eligible:
            continue

        prov_new = theta1[theta1["provider"] == prov]
        prov_canon = canon_theta1[canon_theta1["provider"] == prov]
        canon_final: dict[str, tuple[int, str]] = {}
        for r in prov_canon.itertuples():
            pc = str(r.plz)
            wk = getattr(r, "weekdays_system_smoothed") if has_weekdays_col else ""
            canon_final[pc] = (int(r.schedule_idx_system_smoothed), wk)

        hub_name_by_plz = prep.get("hub_name_by_plz", {})

        for P in P_values:
            if _key(P, 1.0, prov) not in done_triples:
                continue  # defensive: shouldn't happen, chosen_df already filtered to done triples
            penalty_mx = _penalty_mx(m, od, sched_waits, P, th=1.0)
            obj = m["dd_cost_mx"] + penalty_mx
            canon_stage1 = obj.argmin(axis=1)

            rows = prov_new[np.isclose(prov_new["penalty"], P)]
            row_by_plz = {str(r.plz): r for r in rows.itertuples()}

            for pc, zi in eligible.items():
                r = row_by_plz.get(pc)
                if r is None:
                    continue
                n_g1a_checked += 1
                new_s1 = int(r.schedule_idx_stage1)
                exp_s1 = int(canon_stage1[zi])
                if new_s1 != exp_s1:
                    mismatches.append(dict(
                        provider=prov, plz=pc, P=P,
                        new_stage1=new_s1, canonical_stage1=exp_s1))

                canon_info = canon_final.get(pc)
                if canon_info is not None:
                    n_g1b_checked += 1
                    old_idx, old_wk = canon_info
                    new_idx = int(r.schedule_idx_system_smoothed)
                    if new_idx != old_idx:
                        new_wk = ",".join(C.WEEKDAYS[d] for d in sorted(schedules[new_idx]))
                        diffs.append(dict(
                            provider=prov, plz=pc,
                            hub=hub_name_by_plz.get(pc, "?"),
                            P=P, old_idx=old_idx, new_idx=new_idx,
                            old_weekdays=old_wk, new_weekdays=new_wk,
                        ))

    out["g1a_status"] = "FAIL" if mismatches else ("PASS" if n_g1a_checked else "SKIPPED")
    out["g1a_cells_checked"] = n_g1a_checked
    out["g1a_mismatches"] = mismatches
    out["g1b_cells_checked"] = n_g1b_checked
    out["g1b_diffs"] = diffs
    return out


# ─────────────────────────────────────────────────────────────────────────────
# G3
# ─────────────────────────────────────────────────────────────────────────────

def run_g3(chosen_df: pd.DataFrame, fleet_index: dict, fleet_schema: str,
          done_triples: set, optim_data: dict, ml_prep: dict, model,
          schedules: list, mtx_cache: dict) -> dict:
    theta1_Ps = sorted(float(p) for p in
                       chosen_df[np.isclose(chosen_df["share_willing"], 1.0)]
                       ["penalty"].unique())
    points = [(P, 1.0) for P in theta1_Ps] + [(0.5, 0.5), (0.5, 0.1)]

    if fleet_schema == "stale":
        # A dd_veh-schema CSV is pre-Task-6c data: its `fleet` column double-
        # counted express vehicles, and there is no dd_single_veh/dd_pool_veh
        # to compare against at all. Comparing anyway would either crash on
        # a KeyError or silently compare against the wrong column -- report
        # the staleness instead of attempting either.
        return dict(points=points, checked=[], skipped=[], mismatches=[],
                    status="STALE_SCHEMA", fleet_schema=fleet_schema)
    if fleet_schema in ("empty", "unknown"):
        return dict(points=points, checked=[], skipped=[], mismatches=[],
                    status="SKIPPED", fleet_schema=fleet_schema)

    checked: list[tuple] = []
    skipped: list[tuple] = []
    mismatches: list[dict] = []

    for P, th in points:
        for prov in C.PROVIDERS:
            key = _key(P, th, prov)
            if key not in done_triples:
                skipped.append((P, th, prov))
                continue

            od, prep = optim_data[prov], ml_prep[prov]
            m = get_matrices(mtx_cache, th, prov, optim_data, ml_prep, model, schedules)
            plz_keys = od["plz_keys"]
            hub_plz_list = od["hub_plz_list"]
            plz_hub_arr = od["plz_hub_arr"]
            raw_express = m["raw_express"]
            veh_3d = m["veh_3d"]
            sched_active = m["sched_active"]

            sub = chosen_df[
                np.isclose(chosen_df["penalty"], P)
                & np.isclose(chosen_df["share_willing"], th)
                & (chosen_df["provider"] == prov)
            ]
            by_plz = {str(r.plz): int(r.schedule_idx_system_smoothed)
                      for r in sub.itertuples()}
            if set(by_plz) != set(plz_keys):
                mismatches.append(dict(
                    P=P, th=th, provider=prov, hub="(all)", day=-1,
                    status="PLZ_SET_MISMATCH",
                    detail=f"csv has {len(by_plz)} plz, expected {len(plz_keys)}"))
                continue
            chosen = np.array([by_plz[pc] for pc in plz_keys], dtype=np.int64)

            hub_names = [
                prep["hub_name_by_plz"].get(plz_keys[int(h[0])], f"hub_{hi}")
                if len(h) else f"hub_{hi}"
                for hi, h in enumerate(hub_plz_list)
            ]

            # Mirrors 61_grid_run_v2.run_triple's (post-Task-6c) output loop
            # exactly: express and delivery-pool vehicles are two distinct
            # closures over two distinct caches, combined into one pooled
            # closure `_pv` for _daily_fleet_per_hub's `pool_veh_fn`.
            express_cache: dict = {}
            pool_cache: dict = {}

            def _ev(hi: int, d: int, ch: np.ndarray) -> float:
                return _hub_express_vehicles(
                    hi, d, ch, hub_plz_list, schedules, raw_express, m, express_cache)

            def _dv(hi: int, d: int, ch: np.ndarray) -> float:
                return _hub_delivery_pool_vehicles(
                    hi, d, ch, hub_plz_list, schedules, m, pool_cache)

            def _pv(hi: int, d: int, ch: np.ndarray) -> float:
                return _ev(hi, d, ch) + _dv(hi, d, ch)

            fleet_arr = _daily_fleet_per_hub(
                chosen, plz_hub_arr, hub_plz_list, veh_3d, schedules,
                pool_veh_fn=_pv, sched_active=sched_active)

            for hi, h_ps in enumerate(hub_plz_list):
                if len(h_ps) == 0:
                    continue
                hname = hub_names[hi]
                for d in range(C.N_DAYS):
                    # Only DELIVERING cells contribute a per-cell tour here --
                    # veh_3d is written for every ACTIVE instance, so on a
                    # non-delivery day it already holds the cell's express
                    # residual; summing it unmasked AND adding the pooled
                    # express term double-counts every express vehicle
                    # (exactly the Task 6c bug). This mask is the fix.
                    deliv = h_ps[sched_active[chosen[h_ps], d]]
                    dd_single = float(veh_3d[deliv, chosen[deliv], d].sum())
                    dd_pool = float(_dv(hi, d, chosen))         # cache hit
                    ex_veh = float(_ev(hi, d, chosen))          # cache hit
                    fleet_total = float(fleet_arr[hi, d])

                    csv_row = fleet_index.get(
                        (round(P, 4), round(th, 4), prov, hname, d))
                    if csv_row is None:
                        mismatches.append(dict(
                            P=P, th=th, provider=prov, hub=hname, day=d,
                            status="MISSING_IN_CSV",
                            dd_single_veh=dd_single, dd_pool_veh=dd_pool,
                            express_veh=ex_veh, fleet=fleet_total))
                        continue

                    csv_dds = float(csv_row["dd_single_veh"])
                    csv_ddp = float(csv_row["dd_pool_veh"])
                    csv_ex = float(csv_row["express_veh"])
                    csv_fl = float(csv_row["fleet"])
                    if not (dd_single == csv_dds and dd_pool == csv_ddp
                            and ex_veh == csv_ex and fleet_total == csv_fl):
                        mismatches.append(dict(
                            P=P, th=th, provider=prov, hub=hname, day=d,
                            status="MISMATCH",
                            dd_single_veh=dd_single, dd_pool_veh=dd_pool,
                            express_veh=ex_veh, fleet=fleet_total,
                            csv_dd_single_veh=csv_dds, csv_dd_pool_veh=csv_ddp,
                            csv_express_veh=csv_ex, csv_fleet=csv_fl))
            checked.append((P, th, prov))

    status = "FAIL" if mismatches else ("PASS" if checked else "SKIPPED")
    return dict(points=points, checked=checked, skipped=skipped,
                mismatches=mismatches, status=status, fleet_schema=fleet_schema)


# ─────────────────────────────────────────────────────────────────────────────
# G4
# ─────────────────────────────────────────────────────────────────────────────

def run_g4(optim_data: dict, ml_prep: dict, model, schedules: list,
          mtx_cache: dict) -> dict:
    daily_si = next(i for i, s in enumerate(schedules) if len(s) == C.N_DAYS)
    fs_b2c_v, fs_b2b_v = C.fs_b2c(0.0), C.fs_b2b(0.0)
    assert fs_b2c_v == 1.0 and fs_b2b_v == 1.0, (
        f"theta=0 fast-share assumption violated: b2c={fs_b2c_v}, b2b={fs_b2b_v}")

    per_provider: dict[str, dict] = {}
    total = 0.0
    for prov in C.PROVIDERS:
        od, prep = optim_data[prov], ml_prep[prov]
        # Routed through get_matrices (not a direct build_cost_matrices_ml
        # call) so G4 gets the same bundle_head-None / small_delivery_price
        # guards as every other gate's matrix build, and shares the cache
        # (theta=0 never collides with G1a/G3's theta=1/0.5/0.1 keys, so this
        # is purely a DRY simplification, not a semantic change).
        m = get_matrices(mtx_cache, 0.0, prov, optim_data, ml_prep, model, schedules)

        n_plz = len(od["plz_keys"])
        pidx = np.arange(n_plz)
        chosen = np.full(n_plz, daily_si, dtype=np.int64)

        dd_total = float(m["dd_cost_mx"][pidx, chosen].sum())

        raw_express = m["raw_express"]
        expr_stops = m["expr_stops"]
        hub_plz_list = od["hub_plz_list"]
        express_cache: dict = {}
        pool_cache: dict = {}
        expr_total = 0.0
        pool_total = 0.0
        for hi, h_ps in enumerate(hub_plz_list):
            for d in range(C.N_DAYS):
                expr_total += _hub_express_day_ml(
                    hi, d, chosen, hub_plz_list, schedules,
                    raw_express, expr_stops, m, express_cache, 1.0)
                pool_total += _hub_smallday_pool_ml(
                    hi, d, chosen, hub_plz_list, schedules, m, pool_cache)

        prov_total = dd_total + expr_total + pool_total
        per_provider[prov] = dict(
            dd=dd_total, express=expr_total, pool=pool_total, total=prov_total)
        total += prov_total
        # m is not del'd/collected here any more: get_matrices caches it in
        # mtx_cache (shared with G1a/G3), so a local del would not free it
        # and claiming otherwise was misleading.

    total_express = sum(r["express"] for r in per_provider.values())
    return dict(total=total, per_provider=per_provider,
                total_express=total_express,
                express_ok=(total_express == 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Report rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_report(*, n_done_triples: int, has_stage1_col: bool,
                  theta_inventory: list[dict], g1a_scope: dict,
                  g1: dict, g3: dict, g4: dict) -> str:
    lines: list[str] = []
    A = lines.append

    A("# Gate report -- G1a / G1b / G3 / G4 (v2 grid)")
    A("")
    A(f"Generated: {pd.Timestamp.now().isoformat(timespec='seconds')}")
    A(f"Live grid snapshot (via `_gatecopy/`): **{n_done_triples}** complete "
      f"`(penalty, share_willing, provider)` triple(s) in `_tab_chosen_v2.csv` "
      "at the time this report was generated -- \"complete\" means the "
      "triple's block has exactly `len(plz_keys)` rows for its provider "
      "(the truncated-block guard; a short block from an in-flight triple "
      "is not counted).")
    A("")
    if theta_inventory:
        A("**(P, theta) inventory** -- what the grid has actually produced, "
          "by theta (providers with >=1 complete triple at that theta / 7):")
        A("")
        A("| theta | complete triples | providers seen | P values seen |")
        A("|---:|---:|---|---|")
        for row in theta_inventory:
            A(f"| {row['theta']:g} | {row['n_triples']} | "
              f"{len(row['providers'])}/7 ({', '.join(row['providers'])}) | "
              f"{', '.join(f'{p:g}' for p in row['P_values'])} |")
        A("")
    else:
        A("**(P, theta) inventory:** no complete triples in the copy yet.")
        A("")
    A("This script never reads the four live CSVs in `results/revision_2026_08/` "
      "directly -- it copies them into `_gatecopy/` first (with retry/backoff "
      "on a transient lock) and reads only the copies. Re-run "
      "`scripts/revision/62_gates_check.py` at any time to pick up more "
      "completed triples; there is no other state to reset.")
    A("")
    A(f"Canonical comparison table: `{CANONICAL_CSV.relative_to(C.ROOT)}` "
      "(the brief names `results/oracle_loop_extended_2026_05_22/...`, which "
      "does not exist in the current tree; this is the canonical run every "
      "sibling script in this revision -- `_stage3_common.py`, "
      "`61_grid_run_v2.py` -- actually reads under that name).")
    A(f"Canonical table has a `schedule_idx_stage1` column: **{has_stage1_col}** "
      "-> " + ("no fallback needed." if has_stage1_col else
               "G1a's fallback recompute is ACTIVE (see module docstring)."))
    A("")
    A(f"**G1a scope inventory** (computed from `plz_data` alone, independent "
      "of theta/fast_share and of grid progress -- see `_daily_demand_only`; "
      "this is the full set of cells that COULD ever qualify, not what has "
      "been checked so far): "
      f"**{g1a_scope['eligible_cells']}** of **{g1a_scope['total_cells']}** "
      "(provider, plz) cells across all 7 providers have "
      f"`daily_demand.min(axis=1) >= {MIN_TOUR_PARCELS:g}` "
      f"(the brief's spec describes this mean-based; the actual gate "
      "criterion is a min over days per the brief's own G1a clarification, "
      "which is why the count here need not match a mean-based estimate). "
      f"**{g1a_scope['zero_demand_cells']}** cell(s) have a zero-demand day "
      "(none, if 0 -- every cell in the network gets at least some demand "
      "every day).")
    A("")

    g1a_ok = g1["g1a_status"] != "FAIL"
    g3_ok = g3["status"] != "FAIL"
    g4_ok = (g4["total"] >= G4_LOWER) and g4["express_ok"]
    overall = "PASS" if (g1a_ok and g3_ok and g4_ok) else "FAIL"

    A("## Summary")
    A("")
    A("| Gate | Status | Note |")
    A("|---|---|---|")
    A(f"| G1a (hard) | **{g1['g1a_status']}** | "
      f"{g1['g1a_cells_checked']} cell-P pair(s) checked, "
      f"{len(g1['g1a_mismatches'])} mismatch(es) |")
    g1b_note = ("PASS (0 diffs)" if not g1["g1b_diffs"]
                else f"{len(g1['g1b_diffs'])} diff(s)")
    A(f"| G1b (report-only) | {g1b_note} | "
      f"{g1['g1b_cells_checked']} cell-P pair(s) checked, never fails |")
    A(f"| G3 (hard) | **{g3['status']}** | "
      f"{len(g3['checked'])} triple(s) checked, "
      f"{len(g3['skipped'])} not yet available, "
      f"{len(g3['mismatches'])} mismatch(es) |")
    if not g4["express_ok"]:
        g4_flag = "FAIL (express != 0)"
    elif g4["total"] < G4_LOWER:
        g4_flag = "FAIL (below lower bound)"
    else:
        g4_flag = "ok"
    A(f"| G4 (hard) | **{g4_flag}** | "
      f"total = {g4['total']:,.2f} EUR/wk, express = {g4['total_express']:,.6f} EUR |")
    A("")
    A(f"**Overall: {overall}** (exit code {'0' if overall == 'PASS' else '1'}; "
      "\"not yet available\"/\"stale schema\" gates count as passing for "
      "this purpose, per the brief)")
    A("")

    # ── G1a ──────────────────────────────────────────────────────────────
    A("## G1a -- stage-1 schedule identity on large, decoupled cells (theta=1)")
    A("")
    A("**Structural caveat (always true, not specific to this run, mirrors "
      "G3's below):** the \"canonical\" value G1a compares against is "
      "recomputed from THIS grid run's own theta=1 matrices, not read from "
      "an independently-computed canonical source (the canonical table has "
      "no stage-1 column at all). Its validity as a stand-in for \"the "
      "canonical choice\" rests entirely on the theta=1 / >=230-parcels "
      "identity proved in the module docstring -- `cost_3d_raw.sum(axis=2) "
      "== dd_cost_mx` (checked at runtime via `assert np.allclose(...)` on "
      "every theta=1 matrix build, part (1)) plus zero express/pool "
      "coupling forcing CD's fixed point to equal the plain argmin (part "
      "(2)) -- not on an independent cross-check against a differently-"
      "computed value. G1a is therefore a self-consistency check of the "
      "current pipeline against a mathematically-forced value.")
    A("")
    if not g1["available"]:
        A("**G1a: 0 triples available yet.** No `share_willing == 1.0` rows "
          "are present in the copied `_tab_chosen_v2.csv`. The grid processes "
          "theta in ascending order, so theta=1 is the last block to complete; "
          "re-run this script once it lands.")
    else:
        A(f"Providers with at least one theta=1 triple done: "
          f"{', '.join(g1['providers_available'])}")
        A(f"Penalty (P) values checked (derived from the grid's own "
          f"`_tab_chosen_v2.csv` at share_willing=1.0, not hardcoded and not "
          f"from the canonical table -- see the (P, theta) inventory above "
          f"for what the grid has actually produced): "
          f"{', '.join(f'{p:g}' for p in g1['P_values'])}")
        A(f"Cell selection: `daily_demand.min(axis=1) >= {MIN_TOUR_PARCELS:g}` "
          "(MIN_TOUR_PARCELS) -- every instance of the cell, across all days, "
          "stays at or above the small-delivery pooling threshold, so holding "
          "can only grow it further. See the G1a scope inventory above for "
          "how many cells this selects network-wide.")
        A(f"Cell-P pairs checked: **{g1['g1a_cells_checked']}**. "
          f"Mismatches: **{len(g1['g1a_mismatches'])}**.")
        A("")
        if g1["g1a_mismatches"]:
            A("| Provider | PLZ | P | new schedule_idx_stage1 | canonical (recomputed) |")
            A("|---|---|---|---|---|")
            for r in g1["g1a_mismatches"]:
                A(f"| {r['provider']} | {r['plz']} | {r['P']:g} | "
                  f"{r['new_stage1']} | {r['canonical_stage1']} |")
        else:
            A("No mismatches -- the decoupled-cell invariant holds on every "
              "checked cell and P value.")
    A("")

    # ── G1b ──────────────────────────────────────────────────────────────
    A("## G1b -- final (system-smoothed) schedule diffs on the same cells")
    A("")
    A("Report only, never fails. Compares the v2 grid's "
      "`schedule_idx_system_smoothed` against the canonical run's OWN "
      "`schedule_idx_system_smoothed` column (a real column, no fallback), "
      "restricted to the same G1a cell set at theta=1. G1a already shows the "
      "*initial* per-cell choice is unchanged on these cells where it "
      "checked; a diff here means stage 2 (fleet balancing) or stage 3 "
      "(system smoothing) later reassigned the cell for fleet reasons -- "
      "expected, not a defect, since the v2 balancer is express-vehicle-aware "
      "and the canonical run's was not. The `hub` column is the swap's likely "
      "scope (the balancer only trades schedules within/across hubs); it is "
      "not a reconstruction of which specific hub-day swap triggered the "
      "change -- that would need the balancer's own swap log, which this "
      "script does not have access to.")
    if KNOWN_ISSUES:
        A("")
        A("**Known-issue caveat for this section:** `schedule_idx_system_smoothed` "
          "is stage 2/3 output, downstream of the balancer named in the "
          "known-issue note under G3 below -- a diff here may partly reflect "
          "that issue rather than purely legitimate express-aware "
          "rebalancing. See G3's known-issue note for detail.")
    A("")
    if not g1["available"]:
        A("(no theta=1 triples available yet -- see G1a above)")
    else:
        A(f"Cell-P pairs checked: **{g1['g1b_cells_checked']}**. "
          f"Diffs: **{len(g1['g1b_diffs'])}**.")
        A("")
        if g1["g1b_diffs"]:
            A("| Provider | PLZ | Hub | P | old idx (weekdays) | new idx (weekdays) |")
            A("|---|---|---|---|---|---|")
            for r in g1["g1b_diffs"]:
                A(f"| {r['provider']} | {r['plz']} | {r['hub']} | {r['P']:g} | "
                  f"{r['old_idx']} ({r['old_weekdays']}) | "
                  f"{r['new_idx']} ({r['new_weekdays']}) |")
        else:
            A("No diffs on the checked cells.")
    A("")

    # ── G3 ───────────────────────────────────────────────────────────────
    A("## G3 -- fleet-profile identity")
    A("")
    A("For every `(P, theta=1)` point present in the grid's own copied "
      "`_tab_chosen_v2.csv` (all of them, derived dynamically from what the "
      "grid has actually produced -- not hardcoded to a subset, and not "
      "read from the canonical table, which has a different, full P list "
      "that partial-grid runs may not have reached yet) plus `(0.5, 0.5)` "
      "and `(0.5, 0.1)` if present, per provider: rebuild the chosen vector "
      "from `_tab_chosen_v2.csv` and recompute `dd_single_veh` (per-cell "
      "`veh_3d`, MASKED to delivery days), `dd_pool_veh` (via "
      "`_hub_delivery_pool_vehicles`), `express_veh` (via "
      "`_hub_express_vehicles`) and the combined `fleet` (via "
      "`_daily_fleet_per_hub(..., pool_veh_fn=..., sched_active=...)`), "
      "mirroring `61_grid_run_v2.py`'s own (post-Task-6c) output loop "
      "field-for-field -- then compare all four against "
      "`tab_fleet_per_hub_v2.csv` exactly (see module docstring for why "
      "`==` on these values is exact and not a tolerance check).")
    A("")
    A("**Structural caveat (always true, not specific to this run):** G3 "
      "recomputes the fleet with the SAME production functions "
      "(`_hub_express_vehicles`, `_hub_delivery_pool_vehicles`, "
      "`_daily_fleet_per_hub`) the grid runner used to write "
      "`tab_fleet_per_hub_v2.csv` in the first place. This makes G3 a "
      "regression/consistency check on this script's own plumbing against "
      "that shared code -- it verifies the recompute MECHANICS are wired "
      "correctly, but it structurally cannot detect a bug that lives "
      "inside the shared functions themselves, since both sides of the "
      "comparison would inherit the same mistake.")
    if KNOWN_ISSUES:
        A("")
        A("**Known issue(s) as of this run:**")
        for note in KNOWN_ISSUES:
            A("")
            A(f"- {note}")
    A("")
    if g3.get("fleet_schema") == "stale":
        A("**G3: stale schema, re-run required.** The copied "
          "`tab_fleet_per_hub_v2.csv` still has a `dd_veh` column -- that is "
          "the pre-Task-6c schema, from before `dd_veh` was split into "
          "`dd_single_veh` + `dd_pool_veh` (spec v3 Section 4.3, commit "
          "`a128e1a`). Its `fleet` column double-counted express vehicles "
          "(the exact bug commit `8da1a86` and `a128e1a` fixed), so it is "
          "not just a differently-named column -- there is no "
          "column-for-column comparison this script can make against it "
          "that would mean what the brief asks G3 to check. Re-run this "
          "script once the grid has produced at least one triple under the "
          "current schema.")
        A("")
    elif g3.get("fleet_schema") in ("empty", "unknown"):
        A(f"**G3: no usable fleet CSV yet** (schema="
          f"`{g3.get('fleet_schema')}`). Re-run once "
          "`tab_fleet_per_hub_v2.csv` exists with header "
          "`..., dd_single_veh, dd_pool_veh, express_veh, fleet`.")
        A("")
    else:
        pts_fmt = ", ".join(f"({p:g}, {t:g})" for p, t in g3["points"])
        A(f"(P, theta) points considered: {pts_fmt}")
        A(f"Triples checked: **{len(g3['checked'])}**. "
          f"Not yet available (skipped): **{len(g3['skipped'])}**. "
          f"Mismatches: **{len(g3['mismatches'])}**.")
        A("")
        if g3["checked"]:
            A("Checked triples: " + ", ".join(
                f"(P={p:g}, th={t:g}, {pr})" for p, t, pr in g3["checked"]))
            A("")
        if g3["skipped"]:
            A("<details><summary>Not yet available (click to expand)</summary>")
            A("")
            A("(P, theta, provider): " + ", ".join(
                f"({p:g}, {t:g}, {pr})" for p, t, pr in g3["skipped"]))
            A("")
            A("</details>")
            A("")
        if g3["mismatches"]:
            A("| P | theta | Provider | Hub | Day | status | "
              "dd_single_veh (new/csv) | dd_pool_veh (new/csv) | "
              "express_veh (new/csv) | fleet (new/csv) |")
            A("|---|---|---|---|---|---|---|---|---|---|")
            for r in g3["mismatches"]:
                if r["status"] == "PLZ_SET_MISMATCH":
                    A(f"| {r['P']:g} | {r['th']:g} | {r['provider']} | - | - | "
                      f"{r['status']} | {r.get('detail', '')} | | | |")
                    continue
                dds_pair = f"{r['dd_single_veh']:.6f} / {r.get('csv_dd_single_veh', '(missing)')}"
                ddp_pair = f"{r['dd_pool_veh']:.6f} / {r.get('csv_dd_pool_veh', '(missing)')}"
                ex_pair = f"{r['express_veh']:.6f} / {r.get('csv_express_veh', '(missing)')}"
                fl_pair = f"{r['fleet']:.6f} / {r.get('csv_fleet', '(missing)')}"
                A(f"| {r['P']:g} | {r['th']:g} | {r['provider']} | {r['hub']} | "
                  f"{r['day']} | {r['status']} | {dds_pair} | {ddp_pair} | "
                  f"{ex_pair} | {fl_pair} |")
        elif g3["checked"]:
            A("No mismatches on any checked (hub, day) row.")
    A("")

    # ── G4 ───────────────────────────────────────────────────────────────
    A("## G4 -- baseline corridor")
    A("")
    A("All cells of all 7 providers forced onto the daily schedule "
      "(fast_share_b2c = fast_share_b2b = 1.0, i.e. theta=0 semantics), "
      "recomputing dd + express + pool from scratch (no optimization -- "
      "this is not a grid triple, it runs unconditionally regardless of grid "
      "progress).")
    A("")
    A(f"Corridor: [{G4_LOWER:,.0f}, {G4_UPPER:,.0f}] EUR/wk "
      f"(hard-fail only below the lower bound; >{G4_MARGIN:.0%} above the "
      "upper bound requires a written explanation, not a fail).")
    A("")
    A("| Provider | dd (EUR) | express (EUR) | pool (EUR) | total (EUR) |")
    A("|---|---:|---:|---:|---:|")
    for prov in C.PROVIDERS:
        r = g4["per_provider"][prov]
        A(f"| {prov} | {r['dd']:,.2f} | {r['express']:,.2f} | "
          f"{r['pool']:,.2f} | {r['total']:,.2f} |")
    A(f"| **Total** | | | | **{g4['total']:,.2f}** |")
    A("")

    A(f"Sanity check: total express cost across all providers = "
      f"{g4['total_express']:,.6f} EUR, asserted (not just noted) to be "
      "**exactly 0** -- the daily schedule has no non-delivery days, and "
      "`_hub_express_day_ml` returns 0 whenever no cell in a hub-day has a "
      "non-delivery-day express residual, which is every cell here. A "
      "violation is a HARD FAIL on its own, checked before the corridor "
      "verdict below -- there is no PASS with an unresolved anomaly "
      "hanging off it.")
    A("")

    if not g4["express_ok"]:
        A(f"**G4: FAIL.** total_express = {g4['total_express']:,.6f} EUR != 0. "
          "This means some cell/day combination has a non-delivery day under "
          "a schedule this script treated as \"daily\", which would indicate "
          "a schedule-enumeration or `daily_si` lookup bug in this gate "
          "script rather than a pipeline defect (the daily schedule by "
          "definition has `len(schedule) == N_DAYS`). The corridor check "
          "below did not run -- the total above cannot be trusted until "
          "this is fixed.")
    elif g4["total"] < G4_LOWER:
        A(f"**G4: FAIL.** Total {g4['total']:,.2f} EUR/wk is below the lower "
          f"bound {G4_LOWER:,.0f} EUR/wk.")
    elif g4["total"] <= G4_UPPER:
        A(f"**G4: PASS.** Total {g4['total']:,.2f} EUR/wk is within the "
          "corridor, and total_express is confirmed exactly 0.")
    elif g4["total"] <= G4_UPPER * (1.0 + G4_MARGIN):
        A(f"**G4: PASS** (above the upper bound but within {G4_MARGIN:.0%}). "
          f"Total {g4['total']:,.2f} EUR/wk vs upper bound {G4_UPPER:,.0f}; "
          "total_express confirmed exactly 0.")
    else:
        over_pct = (g4["total"] / G4_UPPER - 1.0) * 100
        pool_total = sum(r["pool"] for r in g4["per_provider"].values())
        A(f"**G4: needs explanation (not a fail).** Total {g4['total']:,.2f} "
          f"EUR/wk is {over_pct:.2f}% above the upper bound {G4_UPPER:,.0f} "
          "(total_express is confirmed exactly 0, so the overage is not an "
          "express-accounting bug).")
        A("")
        A("Written explanation (grounded in the per-provider breakdown above):")
        A("")
        A(f"- The overage is entirely in dd "
          f"({sum(r['dd'] for r in g4['per_provider'].values()):,.2f} EUR) "
          f"and pool ({pool_total:,.2f} EUR). The corridor's upper bound "
          f"({G4_UPPER:,.0f}) equals `_stage3_common.BASE_TOTAL` "
          "(1909747.75), the pre-rev1 baseline total; the rev1 "
          "small-delivery pooling rule should only ever REDUCE cost "
          "relative to that baseline (a sub-threshold instance now rides "
          "a shared hub tour instead of paying for its own), so a total "
          "above it suggests either a provider's baseline demand changed "
          "since BASE_TOTAL was computed, or the pooled price is pricing "
          "above the per-cell price it replaced for some instances -- "
          "worth comparing this table's per-provider `dd` figures "
          "against the pre-rev1 per-provider baseline to localize it.")
    A("")

    A("---")
    A("")
    A("Re-running: this script is idempotent and safe to run again at any "
      "point while the grid is still writing -- it always re-copies the "
      "live files first and recomputes from scratch; there is no cached or "
      "stale state between runs.")

    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.perf_counter()

    print("[62] copying live CSVs -> _gatecopy/ ...", flush=True)
    copies = refresh_copies()
    for name in LIVE_FILES:
        print(f"    {'OK ' if name in copies else 'MISSING'} {name}", flush=True)

    chosen_df_raw = load_chosen_v2(copies.get("_tab_chosen_v2.csv"))
    fleet_index, n_fleet_rows, fleet_schema = load_fleet_v2_exact(
        copies.get("tab_fleet_per_hub_v2.csv"))
    print(f"[62] {len(chosen_df_raw)} chosen row(s), {n_fleet_rows} fleet "
          f"row(s) (schema={fleet_schema}) in the copies", flush=True)

    if not CANONICAL_CSV.exists():
        raise SystemExit(f"canonical table not found: {CANONICAL_CSV}")
    canonical_df = pd.read_csv(CANONICAL_CSV, dtype={"plz": str})
    has_stage1_col = "schedule_idx_stage1" in canonical_df.columns
    print(f"[62] canonical columns: {list(canonical_df.columns)}", flush=True)
    print(f"[62] canonical has stage1 column: {has_stage1_col}", flush=True)

    print("[62] loading checkpoints + model ...", flush=True)
    provider_data, optim_data = C.load_checkpoints()
    model = C.load_model()
    ml_prep = C.build_ml_prep(provider_data)
    del provider_data
    gc.collect()
    schedules = C.enumerate_schedules()
    assert len(schedules) == 39, f"expected 39 schedules, got {len(schedules)}"
    assert schedules == enumerate_valid_schedules(), (
        "schedule ordering differs from batch_delivery.optimization.schedules "
        "-- stored schedule_idx_* columns would be meaningless")
    sched_waits = np.array([C.avg_wait_days(sorted(s)) for s in schedules])

    # Truncated-block guard (module docstring): needs optim_data's plz_keys
    # counts, so this is computed here, AFTER the checkpoint load, not
    # eagerly on the raw copy. chosen_df is then filtered to ONLY complete
    # triples at the source, so every gate downstream sees a consistent view
    # without re-deriving completeness itself.
    done_triples = load_done_triples(chosen_df_raw, optim_data)
    chosen_df = filter_complete(chosen_df_raw, done_triples)
    n_truncated_rows = len(chosen_df_raw) - len(chosen_df)
    print(f"[62] {len(done_triples)} complete triple(s) after the "
          f"truncated-block guard ({n_truncated_rows} row(s) from short "
          "blocks excluded)", flush=True)

    # G1a scope inventory: cheap (no surrogate calls), independent of grid
    # progress -- computed for all 7 providers regardless of which ones have
    # theta=1 data yet.
    scope_total = 0
    scope_eligible = 0
    scope_zero = 0
    for prov in C.PROVIDERS:
        dd = _daily_demand_only(optim_data[prov])
        mins = dd.min(axis=1)
        scope_total += dd.shape[0]
        scope_eligible += int((mins >= MIN_TOUR_PARCELS).sum())
        scope_zero += int((mins <= 0.0).sum())
    g1a_scope = dict(total_cells=scope_total, eligible_cells=scope_eligible,
                     zero_demand_cells=scope_zero)
    print(f"[62] G1a scope: {scope_eligible}/{scope_total} cells eligible "
          f"(>= {MIN_TOUR_PARCELS:g} on every day), {scope_zero} with a "
          "zero-demand day", flush=True)

    # (P, theta) inventory, from the guarded done_triples.
    by_theta: dict[float, dict] = {}
    for P, th, prov in done_triples:
        slot = by_theta.setdefault(th, {"providers": set(), "P_values": set()})
        slot["providers"].add(prov)
        slot["P_values"].add(P)
    theta_inventory = [
        dict(theta=th, n_triples=sum(
                1 for p, t, pr in done_triples if t == th),
            providers=sorted(v["providers"]), P_values=sorted(v["P_values"]))
        for th, v in sorted(by_theta.items())
    ]

    mtx_cache: dict = {}

    print("[62] G1a/G1b ...", flush=True)
    g1 = run_g1a_g1b(chosen_df, canonical_df, done_triples, optim_data,
                     ml_prep, model, schedules, sched_waits, mtx_cache)
    print(f"    G1a: {g1['g1a_status']} "
          f"({g1['g1a_cells_checked']} checked, "
          f"{len(g1['g1a_mismatches'])} mismatch(es))", flush=True)
    print(f"    G1b: {len(g1['g1b_diffs'])} diff(s) "
          f"of {g1['g1b_cells_checked']} checked (report-only)", flush=True)

    print("[62] G3 ...", flush=True)
    g3 = run_g3(chosen_df, fleet_index, fleet_schema, done_triples,
               optim_data, ml_prep, model, schedules, mtx_cache)
    print(f"    G3: {g3['status']} ({len(g3['checked'])} checked, "
          f"{len(g3['skipped'])} not yet available, "
          f"{len(g3['mismatches'])} mismatch(es))", flush=True)

    print("[62] G4 ...", flush=True)
    g4 = run_g4(optim_data, ml_prep, model, schedules, mtx_cache)
    print(f"    G4: total = {g4['total']:,.2f} EUR/wk "
          f"(corridor [{G4_LOWER:,.0f}, {G4_UPPER:,.0f}]), "
          f"express_ok={g4['express_ok']}", flush=True)

    report = render_report(
        n_done_triples=len(done_triples), has_stage1_col=has_stage1_col,
        theta_inventory=theta_inventory, g1a_scope=g1a_scope,
        g1=g1, g3=g3, g4=g4)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"[62] wrote {REPORT_PATH}", flush=True)

    hard_fail = (
        g1["g1a_status"] == "FAIL"
        or g3["status"] == "FAIL"
        or g4["total"] < G4_LOWER
        or not g4["express_ok"]
    )
    print(f"[62] done in {time.perf_counter() - t_start:.1f}s; "
          f"overall={'FAIL' if hard_fail else 'PASS'}", flush=True)
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
