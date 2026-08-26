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
       SAME theta=1 matrices the v2 grid itself built. That is valid, not a
       shortcut: at theta=1, fast_share_b2c = fast_share_b2b = 0 (see
       ``_stage3_common.fs_b2c/fs_b2b``), so every cell's ``raw_express`` is
       identically zero and ``cost_3d`` only gets predictions on delivery
       days -- i.e. ``cost_3d_raw.sum(axis=2) == dd_cost_mx`` for ALL cells,
       eligible or not. And for a cell whose every instance is >=230
       parcels, ``small_delivery_mask`` never fires, so it is never coupled
       into a hub pool either. Reading ``coordinate_descent.py``'s inner
       loop confirms the consequence directly: for such a cell every
       candidate move's delta reduces to
       ``dd_cost_mx[pi, new_si] - dd_cost_mx[pi, old_si]`` with a zero
       express/pool correction (express cache deltas are 0-0, pool_affected
       is empty), so CD's greedy per-cell step is mathematically the plain
       argmin -- "old = new machinery on those cells" is a provable
       identity here, not an approximation, which is why a mismatch is a
       hard fail rather than a tolerance check.

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

  G3   for every (P, theta=1) point present (all of them, not just a
       hardcoded four -- the brief says "don't hardcode", so this derives
       the theta=1 penalty list from the canonical table) plus (0.5, 0.5)
       and (0.5, 0.1) if present: rebuild the chosen vector from the copied
       ``_tab_chosen_v2.csv`` and recompute the fleet with
       ``_daily_fleet_per_hub(..., express_veh_fn=_hub_express_vehicles)``,
       plus ``dd_veh``/``express_veh`` the same way ``61_``'s output loop
       does. Compare all three fields against ``tab_fleet_per_hub_v2.csv``
       EXACTLY.

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
there is something to check. Re-running this script later, as more triples
land, picks up more coverage automatically -- there is no persistent state
outside the four source CSVs.

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
# (`_hub_express_vehicles`, `_daily_fleet_per_hub`) that wrote
# ``tab_fleet_per_hub_v2.csv`` in the first place -- if those functions have
# a bug, G3 reproduces it on both sides of the comparison and passes
# vacuously. Remove an entry once the underlying issue is fixed AND the grid
# has been re-run past the affected triples.
KNOWN_ISSUES = [
    "2026-08-26: the base grid was killed by the controller at 159/615 "
    "triples pending Task 6c, a fix for a fleet double-counting bug "
    "(express vehicles counted twice, in the per-cell delivery fleet AND "
    "the hub express fleet). Of the 159 triples already written: "
    "`schedule_idx_stage1` and the dd/express/pool cost columns are VALID "
    "(stage 1 and pricing do not go through the buggy vehicle-counting "
    "path); `schedule_idx_balanced`, `schedule_idx_system_smoothed` and "
    "every row of `tab_fleet_per_hub_v2.csv` were written by the PRE-FIX "
    "balancer and are STALE relative to the fix. Three outcomes are all "
    "'working as intended', not a bug in this script: (a) if "
    "`_hub_express_vehicles`/`_daily_fleet_per_hub` are still pre-fix when "
    "this runs, G3 reproduces the same double-count on both sides and "
    "PASSES VACUOUSLY -- read that as 'mechanics verified, semantics "
    "pending the fix landing and a grid re-run', not as confirmation the "
    "fleet numbers are correct; (b) if the fix has already landed (as "
    "observed on the 2026-08-26 run that produced 32 mismatches at the "
    "(0.5, 0.1) point, all 7 providers -- `dd_veh`/`express_veh` still "
    "matched the CSV individually, but the combined `fleet` this script "
    "recomputed was consistently LOWER than the CSV's, i.e. exactly the "
    "double-count coming out; commit `8da1a86`, 'fix(fleet): mask veh_3d "
    "to delivery days when pooled vehicles are counted (double-count)', "
    "documents the same symptom), G3 correctly detects the stale CSV no "
    "longer agrees with the current code -- that FAIL is accurate and "
    "expected, not a regression, and will clear once the grid is re-run "
    "past the affected triples with the fixed code; (c) if "
    "`_daily_fleet_per_hub`'s interface has moved further (observed the "
    "same day: its `express_veh_fn` keyword argument was renamed to "
    "`pool_veh_fn` with revised delivery-day masking, in an uncommitted "
    "change on top of 8da1a86), this script raises "
    "`TypeError: _daily_fleet_per_hub() got an unexpected keyword argument "
    "'express_veh_fn'` instead of writing a report -- this script "
    "deliberately calls the interface `61_grid_run_v2.py` used to actually "
    "produce the CSVs being audited (correct for auditing THAT data), and "
    "was not adapted to chase a still-uncommitted, in-flight rename in a "
    "file this task does not own; if you hit this, either wait for the "
    "rename to land and update this call site accordingly, or re-run "
    "against a commit where the two are in sync. G1b diffs on this grid "
    "may also partly reflect the same bug rather than purely legitimate "
    "express-aware rebalancing.",
]


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
    return pd.read_csv(path)


def load_done_triples(chosen_df: pd.DataFrame) -> set[tuple[float, float, str]]:
    if chosen_df.empty:
        return set()
    return {_key(r.penalty, r.share_willing, r.provider)
            for r in chosen_df.itertuples()}


def load_fleet_v2_exact(path: Path | None) -> tuple[dict[tuple, dict], int]:
    """Read the fleet CSV copy as raw strings via the stdlib ``csv`` module
    (see module docstring for why -- pandas' C float parser is not
    correctly-rounded at the last digit). Returns an index keyed by
    ``(penalty, share_willing, provider, hub, day)`` -> raw-string row dict,
    plus the total row count.
    """
    index: dict[tuple, dict] = {}
    n = 0
    if path is None or not path.exists():
        return index, n
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv_mod.DictReader(f):
            n += 1
            key = (round(float(row["penalty"]), 4), round(float(row["share_willing"]), 4),
                   row["provider"], row["hub"], int(row["day"]))
            index[key] = row
    return index, n


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
    print(f"    [mtx] th={theta:<4g} {prov:<7s} built in "
          f"{time.perf_counter() - t0:5.1f}s", flush=True)
    cache[key] = m
    return m


def _penalty_mx(m: dict, od: dict, schedules: list, sched_waits: np.ndarray,
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
            pc = str(int(r.plz))
            wk = getattr(r, "weekdays_system_smoothed") if has_weekdays_col else ""
            canon_final[pc] = (int(r.schedule_idx_system_smoothed), wk)

        hub_name_by_plz = prep.get("hub_name_by_plz", {})

        for P in P_values:
            if _key(P, 1.0, prov) not in done_triples:
                continue  # defensive: shouldn't happen, chosen_df already filtered to done triples
            penalty_mx = _penalty_mx(m, od, schedules, sched_waits, P, th=1.0)
            obj = m["dd_cost_mx"] + penalty_mx
            canon_stage1 = obj.argmin(axis=1)

            rows = prov_new[np.isclose(prov_new["penalty"], P)]
            row_by_plz = {str(int(r.plz)): r for r in rows.itertuples()}

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

def run_g3(chosen_df: pd.DataFrame, fleet_index: dict, done_triples: set,
          optim_data: dict, ml_prep: dict, model, schedules: list,
          mtx_cache: dict) -> dict:
    theta1_Ps = sorted(float(p) for p in
                       chosen_df[np.isclose(chosen_df["share_willing"], 1.0)]
                       ["penalty"].unique())
    points = [(P, 1.0) for P in theta1_Ps] + [(0.5, 0.5), (0.5, 0.1)]

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

            sub = chosen_df[
                np.isclose(chosen_df["penalty"], P)
                & np.isclose(chosen_df["share_willing"], th)
                & (chosen_df["provider"] == prov)
            ]
            by_plz = {str(int(r.plz)): int(r.schedule_idx_system_smoothed)
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

            express_cache: dict = {}

            def _ev(hi: int, d: int, ch: np.ndarray) -> float:
                return _hub_express_vehicles(
                    hi, d, ch, hub_plz_list, schedules, raw_express, m, express_cache)

            fleet_arr = _daily_fleet_per_hub(
                chosen, plz_hub_arr, hub_plz_list, veh_3d, schedules,
                express_veh_fn=_ev)

            for hi, h_ps in enumerate(hub_plz_list):
                if len(h_ps) == 0:
                    continue
                hname = hub_names[hi]
                for d in range(C.N_DAYS):
                    dd_veh = float(veh_3d[h_ps, chosen[h_ps], d].sum())
                    ex_veh = float(_ev(hi, d, chosen))          # cache hit
                    fleet_total = float(fleet_arr[hi, d])

                    csv_row = fleet_index.get(
                        (round(P, 4), round(th, 4), prov, hname, d))
                    if csv_row is None:
                        mismatches.append(dict(
                            P=P, th=th, provider=prov, hub=hname, day=d,
                            status="MISSING_IN_CSV",
                            dd_veh=dd_veh, express_veh=ex_veh, fleet=fleet_total))
                        continue

                    csv_dd = float(csv_row["dd_veh"])
                    csv_ex = float(csv_row["express_veh"])
                    csv_fl = float(csv_row["fleet"])
                    if not (dd_veh == csv_dd and ex_veh == csv_ex
                            and fleet_total == csv_fl):
                        mismatches.append(dict(
                            P=P, th=th, provider=prov, hub=hname, day=d,
                            status="MISMATCH",
                            dd_veh=dd_veh, express_veh=ex_veh, fleet=fleet_total,
                            csv_dd_veh=csv_dd, csv_express_veh=csv_ex, csv_fleet=csv_fl))
            checked.append((P, th, prov))

    status = "FAIL" if mismatches else ("PASS" if checked else "SKIPPED")
    return dict(points=points, checked=checked, skipped=skipped,
                mismatches=mismatches, status=status)


# ─────────────────────────────────────────────────────────────────────────────
# G4
# ─────────────────────────────────────────────────────────────────────────────

def run_g4(optim_data: dict, ml_prep: dict, model, schedules: list) -> dict:
    daily_si = next(i for i, s in enumerate(schedules) if len(s) == C.N_DAYS)
    fs_b2c_v, fs_b2b_v = C.fs_b2c(0.0), C.fs_b2b(0.0)
    assert fs_b2c_v == 1.0 and fs_b2b_v == 1.0, (
        f"theta=0 fast-share assumption violated: b2c={fs_b2c_v}, b2b={fs_b2b_v}")

    per_provider: dict[str, dict] = {}
    total = 0.0
    for prov in C.PROVIDERS:
        od, prep = optim_data[prov], ml_prep[prov]
        t0 = time.perf_counter()
        m = build_cost_matrices_ml(
            od["plz_keys"], od["plz_data"], schedules, model, prov,
            prep["plz_day_coords"], prep["hub_coords_by_plz"],
            fast_share_b2c=fs_b2c_v, fast_share_b2b=fs_b2b_v)
        print(f"    [mtx] th=0    {prov:<7s} built in "
              f"{time.perf_counter() - t0:5.1f}s (G4)", flush=True)

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
        del m
        gc.collect()

    return dict(total=total, per_provider=per_provider)


# ─────────────────────────────────────────────────────────────────────────────
# Report rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_report(*, n_done_triples: int, has_stage1_col: bool,
                  g1: dict, g3: dict, g4: dict) -> str:
    lines: list[str] = []
    A = lines.append

    A("# Gate report -- G1a / G1b / G3 / G4 (v2 grid)")
    A("")
    A(f"Generated: {pd.Timestamp.now().isoformat(timespec='seconds')}")
    A(f"Live grid snapshot (via `_gatecopy/`): **{n_done_triples}** "
      f"`(penalty, share_willing, provider)` triple(s) complete in "
      f"`_tab_chosen_v2.csv` at the time this report was generated.")
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

    g1a_ok = g1["g1a_status"] != "FAIL"
    g3_ok = g3["status"] != "FAIL"
    g4_lower_ok = g4["total"] >= G4_LOWER
    overall = "PASS" if (g1a_ok and g3_ok and g4_lower_ok) else "FAIL"

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
    lower_flag = "FAIL (below lower bound)" if not g4_lower_ok else "ok"
    A(f"| G4 (hard below lower bound) | **{lower_flag}** | "
      f"total = {g4['total']:,.2f} EUR/wk |")
    A("")
    A(f"**Overall: {overall}** (exit code {'0' if overall == 'PASS' else '1'}; "
      "\"not yet available\" gates count as passing for this purpose, per the brief)")
    A("")

    # ── G1a ──────────────────────────────────────────────────────────────
    A("## G1a -- stage-1 schedule identity on large, decoupled cells (theta=1)")
    A("")
    if not g1["available"]:
        A("**G1a: 0 triples available yet.** No `share_willing == 1.0` rows "
          "are present in the copied `_tab_chosen_v2.csv`. The grid processes "
          "theta in ascending order, so theta=1 is the last block to complete; "
          "re-run this script once it lands.")
    else:
        A(f"Providers with at least one theta=1 triple done: "
          f"{', '.join(g1['providers_available'])}")
        A(f"Penalty (P) values checked (derived from the canonical table's "
          f"unique `penalty` values at share_willing=1.0, not hardcoded): "
          f"{', '.join(f'{p:g}' for p in g1['P_values'])}")
        A(f"Cell selection: `daily_demand.min(axis=1) >= {MIN_TOUR_PARCELS:g}` "
          "(MIN_TOUR_PARCELS) -- every instance of the cell, across all days, "
          "stays at or above the small-delivery pooling threshold, so holding "
          "can only grow it further.")
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
    A("For every `(P, theta=1)` point present in the canonical table (all of "
      "them, derived dynamically -- not hardcoded to a subset) plus "
      "`(0.5, 0.5)` and `(0.5, 0.1)` if present, per provider: rebuild the "
      "chosen vector from `_tab_chosen_v2.csv` and recompute `dd_veh`, "
      "`express_veh` (via `_hub_express_vehicles`) and the combined `fleet` "
      "(via `_daily_fleet_per_hub(..., express_veh_fn=...)`), then compare "
      "against `tab_fleet_per_hub_v2.csv` exactly (see module docstring for "
      "why `==` on these values is exact and not a tolerance check).")
    A("")
    A("**Structural caveat (always true, not specific to this run):** G3 "
      "recomputes the fleet with the SAME production functions "
      "(`_hub_express_vehicles`, `_daily_fleet_per_hub`) the grid runner "
      "used to write `tab_fleet_per_hub_v2.csv` in the first place. This "
      "makes G3 a regression/consistency check on this script's own "
      "plumbing against that shared code -- it verifies the recompute "
      "MECHANICS are wired correctly, but it structurally cannot detect a "
      "bug that lives inside the shared functions themselves, since both "
      "sides of the comparison would inherit the same mistake.")
    if KNOWN_ISSUES:
        A("")
        A("**Known issue(s) as of this run:**")
        for note in KNOWN_ISSUES:
            A("")
            A(f"- {note}")
    A("")
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
        A("| P | theta | Provider | Hub | Day | status | dd_veh (new/csv) | "
          "express_veh (new/csv) | fleet (new/csv) |")
        A("|---|---|---|---|---|---|---|---|---|")
        for r in g3["mismatches"]:
            if r["status"] == "PLZ_SET_MISMATCH":
                A(f"| {r['P']:g} | {r['th']:g} | {r['provider']} | - | - | "
                  f"{r['status']} | {r.get('detail', '')} | | |")
                continue
            dd_pair = f"{r['dd_veh']:.6f} / {r.get('csv_dd_veh', '(missing)')}"
            ex_pair = f"{r['express_veh']:.6f} / {r.get('csv_express_veh', '(missing)')}"
            fl_pair = f"{r['fleet']:.6f} / {r.get('csv_fleet', '(missing)')}"
            A(f"| {r['P']:g} | {r['th']:g} | {r['provider']} | {r['hub']} | "
              f"{r['day']} | {r['status']} | {dd_pair} | {ex_pair} | {fl_pair} |")
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

    total_express = sum(r["express"] for r in g4["per_provider"].values())
    A(f"Sanity check: total express cost across all providers = "
      f"{total_express:,.6f} EUR. This is expected to be **exactly 0** -- "
      "the daily schedule has no non-delivery days, and "
      "`_hub_express_day_ml` returns 0 whenever no cell in a hub-day has a "
      "non-delivery-day express residual, which is every cell here. "
      f"{'Confirmed.' if total_express == 0.0 else '**NOT confirmed -- see explanation below.**'}")
    A("")

    if g4["total"] < G4_LOWER:
        A(f"**G4: FAIL.** Total {g4['total']:,.2f} EUR/wk is below the lower "
          f"bound {G4_LOWER:,.0f} EUR/wk.")
    elif g4["total"] <= G4_UPPER:
        A(f"**G4: PASS.** Total {g4['total']:,.2f} EUR/wk is within the corridor.")
    elif g4["total"] <= G4_UPPER * (1.0 + G4_MARGIN):
        A(f"**G4: PASS** (above the upper bound but within {G4_MARGIN:.0%}). "
          f"Total {g4['total']:,.2f} EUR/wk vs upper bound {G4_UPPER:,.0f}.")
    else:
        over_pct = (g4["total"] / G4_UPPER - 1.0) * 100
        A(f"**G4: needs explanation (not a fail).** Total {g4['total']:,.2f} "
          f"EUR/wk is {over_pct:.2f}% above the upper bound {G4_UPPER:,.0f}.")
        A("")
        A("Written explanation (grounded in the per-provider breakdown above):")
        A("")
        if total_express != 0.0:
            A(f"- Express cost is {total_express:,.2f} EUR instead of the "
              "expected 0 -- this means some cell/day combination has a "
              "non-delivery day under a schedule this script treated as "
              "\"daily\", which would indicate a schedule-enumeration or "
              "`daily_si` lookup bug in this gate script rather than a "
              "pipeline defect (the daily schedule by definition has "
              "`len(schedule) == N_DAYS`). Needs investigation before "
              "trusting the total above.")
        else:
            pool_total = sum(r["pool"] for r in g4["per_provider"].values())
            A(f"- Express cost is confirmed 0 as expected; the overage is "
              f"entirely in dd ({sum(r['dd'] for r in g4['per_provider'].values()):,.2f} "
              f"EUR) and pool ({pool_total:,.2f} EUR). The corridor's upper "
              f"bound ({G4_UPPER:,.0f}) equals `_stage3_common.BASE_TOTAL` "
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

    chosen_df = load_chosen_v2(copies.get("_tab_chosen_v2.csv"))
    done_triples = load_done_triples(chosen_df)
    fleet_index, n_fleet_rows = load_fleet_v2_exact(copies.get("tab_fleet_per_hub_v2.csv"))
    print(f"[62] {len(done_triples)} completed triple(s), "
          f"{len(chosen_df)} chosen row(s), {n_fleet_rows} fleet row(s) "
          "in the copies", flush=True)

    if not CANONICAL_CSV.exists():
        raise SystemExit(f"canonical table not found: {CANONICAL_CSV}")
    canonical_df = pd.read_csv(CANONICAL_CSV)
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
    g3 = run_g3(chosen_df, fleet_index, done_triples, optim_data, ml_prep,
               model, schedules, mtx_cache)
    print(f"    G3: {g3['status']} ({len(g3['checked'])} checked, "
          f"{len(g3['skipped'])} not yet available, "
          f"{len(g3['mismatches'])} mismatch(es))", flush=True)

    print("[62] G4 ...", flush=True)
    g4 = run_g4(optim_data, ml_prep, model, schedules)
    print(f"    G4: total = {g4['total']:,.2f} EUR/wk "
          f"(corridor [{G4_LOWER:,.0f}, {G4_UPPER:,.0f}])", flush=True)

    report = render_report(
        n_done_triples=len(done_triples), has_stage1_col=has_stage1_col,
        g1=g1, g3=g3, g4=g4)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"[62] wrote {REPORT_PATH}", flush=True)

    hard_fail = (
        g1["g1a_status"] == "FAIL"
        or g3["status"] == "FAIL"
        or g4["total"] < G4_LOWER
    )
    print(f"[62] done in {time.perf_counter() - t_start:.1f}s; "
          f"overall={'FAIL' if hard_fail else 'PASS'}", flush=True)
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
