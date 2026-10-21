"""72: per-cell plan costs for a v5/v6-schema grid (Task 13B).

The grid runner (``61_grid_run_v2.py``) records **schedules** per cell and
**totals** per (P, theta, provider) -- it never writes what one postal-code
area costs.  Fig. 6 (b)-(f) of the submitted paper is a per-PLZ euro saving,
so that column has to exist.  This script rebuilds it, from the same
matrices, under the same head, and proves the rebuild is exact by summing it
back to the grid's own totals.

What a cell pays
----------------
The routing cost of a plan is exactly three disjoint terms
(``61_grid_run_v2.run_triple``)::

    routing_total = dd_total + expr_total + pool_total
                  = sum_cells dd_cost_mx[cell, chosen]          # own tours
                  + sum_(hub, day) _hub_express_day_ml(...)     # express pool
                  + sum_(hub, day) _hub_smallday_pool_ml(...)   # delivery pool

The first term is already per cell.  The other two are prices of *tours*, and
a tour can carry several cells -- that is the whole point of the rev1
realistic-tour rule.  They are attributed **parcel-proportionally**: a member
carrying 30 % of a group's parcels is charged 30 % of the group's price.
That rule is used in BOTH head regimes on purpose.  At ``--head none`` the
group price happens to be the Sigma over its members' singleton prices, so an
exact per-member split exists; at ``--head installed`` it does not (one
surrogate call prices the whole tour and the bundling gain is a property of
the group, not of any member).  Using one rule for both keeps a v5 cell and a
v6 cell comparable; using the exact split at ``none`` and a proportional one
at ``installed`` would make every v5->v6 per-cell delta partly an artefact of
the attribution rule.

Three identity gates, per (P, theta, provider, plan)
----------------------------------------------------
1. ``sum_cells cell_cost_eur`` == ``cost_stage1_eur`` (stage-1 plan) /
   ``cost_stage2_eur`` (balanced plan) of ``tab_costs_v2.csv``;
2. ``sum_cells veh_days_share`` == ``vehicle_days_before`` / ``vehicle_days``;
3. ``sum_cells peak_veh_share`` == ``sum_hub_peak_before`` / ``sum_hub_peak``.

All three use the runner's own window (``61_grid_run_v2._tol``: a 1e-3 EUR
absolute floor, 1e-9 relative), and all three FAIL LOUD.
Gate 1 is the one the brief asks for; 2 and 3 come free with the vehicle
attribution and are what make the operator lens reconstructable per hub::

    operator_cost(hub) = sum_cells cell_cost_eur
                       - 189.15 * sum_cells veh_days_share
                       + 1134.90 * sum_cells peak_veh_share

**The operator lens is hub-attributable, not cell-attributable.**  The weekly
fixed bill is sized by the hub's PEAK day, so what a cell contributes to it
depends on which day the rest of the hub peaks on -- move one neighbour and
the same cell's "share" changes without the cell changing at all.  The peak
column here is therefore an *attribution*, exact in sum and meaningful at the
hub level; a per-cell operator saving is not a well-defined quantity and this
script does not write one.  ``70_figs_tables_v2.py`` shows the routing lens
per cell and the operator lens per hub, labelled as such.

Long here, wide in the runner
-----------------------------
Since commit 634433f the runner writes the same decomposition itself, into
``_tab_chosen_v2.csv``, in a WIDE layout: one row per cell with a
``_stage1`` / ``_stage2`` suffix per column.  This file stays LONG (one row
per cell AND plan, unsuffixed columns plus a ``plan`` key) because it also
carries the vehicle and hub-peak attribution and is grouped by plan by every
consumer.  The two layouts are the same numbers::

    72_ (long)                      61_ (wide, from v7 on)
    plan="stage1"   own_cost_eur    own_cost_eur_stage1
                    pool_share_eur  pool_share_eur_stage1
                    express_share_eur  express_share_eur_stage1
                    cell_cost_eur   cell_cost_eur_stage1
    plan="balanced" own_cost_eur    own_cost_eur_stage2
                    pool_share_eur  pool_share_eur_stage2
                    express_share_eur  express_share_eur_stage2
                    cell_cost_eur   cell_cost_eur_stage2
                    cell_parcels_week  cell_parcels_week

(the runner's ``stage2`` IS this file's ``balanced``: stage 3 is off.)
When those columns are present -- they are from v7 on, absent on v5/v6 --
:func:`crosscheck_wide` asserts the two agree cell by cell, so the post-hoc
reconstruction and the runner's in-line one can never drift apart.  When
they are absent the check logs that it skipped and why.

The head
--------
The head is loaded and installed by ``61_grid_run_v2``'s OWN helpers
(``load_head`` / ``install_head``), never a second implementation, and the
resulting ``head_id`` is asserted equal to the ``head_id`` on the grid's
rows.  Defaults come from the grid's ``head_manifest.json`` when it has one,
so the ordinary invocation needs no head flags at all.  A grid without that
file and without a ``head_id`` column is a pre-Task-11 (v5) grid and is only
priceable with ``--head none``.

Cost
----
One ``build_cost_matrices_ml`` per (theta, provider) -- 11 x 7 = 77 builds,
~20 s each -- plus one pricing pass per distinct plan in the block.  Plans are
memoised on the chosen vector, so the theta = 0 baseline (identical for every
P and for both plans) is priced once per provider rather than 16 times.
Resumable per (theta, provider): a block whose row count is short is dropped
and redone rather than trusted.

Usage
-----
    python scripts/revision/72_per_cell_costs_v2.py --rev-dir results/revision_2026_08_v6
    python scripts/revision/72_per_cell_costs_v2.py --rev-dir results/revision_2026_08_v5 --head none

Output: ``<rev>/tables/tab_per_cell_costs_v2.csv``, one row per
(P, theta, provider, plz, plan).
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _stage3_common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "src"))
from batch_delivery.config.constants import VEHICLE_CAPACITY  # noqa: E402
from batch_delivery.optimization.balancing import (  # noqa: E402
    _daily_fleet_per_hub,
)
from batch_delivery.optimization.costs import (  # noqa: E402
    _express_members,
    _express_partition,
    _hub_delivery_pool_vehicles,
    _hub_express_day_ml,
    _hub_express_vehicles,
    _hub_smallday_pool_ml,
    _smallday_members,
    _smallday_partition,
)
from batch_delivery.surrogate.bundle import price_group  # noqa: E402

ROOT = C.ROOT

#: The two plans every v5/v6 row carries.  ``schedule_idx_system_smoothed``
#: equals ``schedule_idx_balanced`` whenever stage 3 is off (it is, in
#: production), so it is deliberately not a third plan here.
PLAN_COL = {"stage1": "schedule_idx_stage1",
            "balanced": "schedule_idx_balanced"}
#: Which grid column each plan's routing total, vehicle-days and hub peak sum
#: must reproduce.  ``*_before`` is the STAGE-1 anchor in the v5/v6 schema.
PLAN_REF = {
    "stage1": dict(cost="cost_stage1_eur", vd="vehicle_days_before",
                   peak="sum_hub_peak_before"),
    "balanced": dict(cost="cost_stage2_eur", vd="vehicle_days",
                     peak="sum_hub_peak"),
}
OUT_NAME = "tab_per_cell_costs_v2.csv"

#: Absolute floor of the identity-gate window, in EUR (or vehicles), and
#: its relative term -- the SAME semantics as the runner's
#: ``61_grid_run_v2._tol``: a tenth of a cent is far above what two
#: summation orders of the same terms differ by, and 1e-9 relative keeps
#: the window scale-safe without ever tolerating a real regression (a
#: 1e-6 relative term would leave a 1.5 EUR blind spot at a routing
#: total of 1.5e6).
ABS_TOL = 1e-3
REL_TOL = 1e-9


def _tol(ref: float) -> float:
    """Comparison window at *ref* -- 61_grid_run_v2._tol, verbatim."""
    return max(ABS_TOL, REL_TOL * abs(ref))


def _load_runner():
    """``61_grid_run_v2.py`` as a module (its name starts with a digit).

    Imported for ``load_head`` / ``install_head`` ONLY: the head-loading
    contract (explicit certified-bins file, edge-drift check, cold-memo
    install, head_id) must be the runner's, not a copy of it.  Importing is
    side-effect free -- the module creates no directories until ``main()``.
    """
    path = Path(__file__).resolve().parent / "61_grid_run_v2.py"
    spec = importlib.util.spec_from_file_location("_grid_run_v2", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────
# The per-cell decomposition
# ─────────────────────────────────────────────────────────────────────────
def _shares(members: tuple[int, ...], weights: np.ndarray) -> np.ndarray:
    """Parcel-proportional weights of a tour's members, summing to 1.

    Fails loud on a zero-parcel group: both partitions are built from cells
    that carry demand that day (``_express_members`` requires
    ``raw_express > 0``, ``_smallday_members`` requires an ACTIVE instance,
    i.e. ``combined_demand > 0``), so a zero total would mean the group came
    from somewhere else and the attribution would be arbitrary.
    """
    w = np.array([float(weights[z]) for z in members], dtype=np.float64)
    tot = w.sum()
    assert tot > 0, (
        f"a realised tour with members {members} carries {tot} parcels -- "
        "there is no parcel-proportional share of its price")
    return w / tot


def decompose_plan(chosen: np.ndarray, plz_hub_arr: np.ndarray,
                   hub_plz_list: list, schedules: list, m: dict) -> dict:
    """Split one plan's routing cost, vehicle-days and hub peaks per cell.

    *chosen* is the schedule index per cell.  Returns a dict of ``(n_plz,)``
    arrays plus the block totals::

        own_cost, pool_share, express_share, cell_cost
        own_vd, pool_vd_share, express_vd_share, veh_days_share
        peak_veh_share, hub_idx, hub_peak_day
        dd_total, pool_total, express_total, routing_total,
        vehicle_days, sum_hub_peak, fleet (n_hubs, N_DAYS)

    Every pooled term is attributed parcel-proportionally (see the module
    docstring).  The tours are enumerated with the SAME helpers the cost path
    uses (``_express_members`` / ``_express_partition`` /
    ``_smallday_members`` / ``_smallday_partition`` / ``price_group``), and
    their per-group sums are asserted against the cost path's own hub-day
    totals -- so this is a decomposition of the runner's number, not a second
    opinion about it.
    """
    chosen = np.asarray(chosen, dtype=np.int64)
    n_plz = len(chosen)
    pidx = np.arange(n_plz)
    dd_cost_mx, veh_3d = m["dd_cost_mx"], m["veh_3d"]
    sched_active, raw_express = m["sched_active"], m["raw_express"]
    expr_stops = m["expr_stops"]
    head = m.get("bundle_head")

    own_cost = dd_cost_mx[pidx, chosen].astype(np.float64).copy()
    pool_share = np.zeros(n_plz)
    express_share = np.zeros(n_plz)
    # per (cell, day) vehicle attribution -- only needed to read off the
    # hub's peak day, so it is kept as three (n_plz, N_DAYS) planes rather
    # than a full per-day cost breakdown.
    own_vd_day = np.zeros((n_plz, C.N_DAYS))
    pool_vd_day = np.zeros((n_plz, C.N_DAYS))
    expr_vd_day = np.zeros((n_plz, C.N_DAYS))

    # The cost path's own hub-day totals, computed with fresh caches -- the
    # reference the per-group decomposition below must reproduce.
    express_cache: dict = {}
    pool_cache: dict = {}
    expr_total = 0.0
    pool_total = 0.0
    for hi in range(len(hub_plz_list)):
        for d in range(C.N_DAYS):
            expr_total += _hub_express_day_ml(
                hi, d, chosen, hub_plz_list, schedules, raw_express,
                expr_stops, m, express_cache, 1.0)
            pool_total += _hub_smallday_pool_ml(
                hi, d, chosen, hub_plz_list, schedules, m, pool_cache)

    expr_from_groups = 0.0
    pool_from_groups = 0.0
    for hi, h_ps in enumerate(hub_plz_list):
        for d in range(C.N_DAYS):
            # --- own tours: DELIVERING cells only (veh_3d holds the express
            # residual on a non-delivery day; the pooled term owns that).
            if len(h_ps):
                deliv = h_ps[sched_active[chosen[h_ps], d]]
                own_vd_day[deliv, d] = veh_3d[deliv, chosen[deliv], d]

            # --- express residual pool
            contributing, _ = _express_members(
                hi, d, chosen, hub_plz_list, schedules, raw_express, m)
            if contributing:
                for g in _express_partition(contributing, d, raw_express,
                                            expr_stops, m):
                    price = float(price_group(g, d, m, kind="express",
                                              head=head))
                    veh = float(np.ceil(
                        sum(np.trunc(raw_express[z, d]) for z in g)
                        / VEHICLE_CAPACITY))
                    w = _shares(g, raw_express[:, d])
                    for k, z in enumerate(g):
                        express_share[z] += price * w[k]
                        expr_vd_day[z, d] += veh * w[k]
                    expr_from_groups += price

            # --- pooled small-delivery groups
            small, _k = _smallday_members(hi, d, chosen, hub_plz_list, m)
            if small:
                parts, parcels, stops = _smallday_partition(
                    hi, d, chosen, small, m)
                for g in parts:
                    price = float(price_group(
                        g, d, m, kind="delivery", parcels_by_cell=parcels,
                        stops_by_cell=stops, freq=1.0, head=head))
                    veh = float(np.ceil(
                        sum(np.trunc(parcels[z]) for z in g)
                        / VEHICLE_CAPACITY))
                    w = _shares(g, parcels)
                    for k, z in enumerate(g):
                        pool_share[z] += price * w[k]
                        pool_vd_day[z, d] += veh * w[k]
                    pool_from_groups += price

    # The per-group decomposition IS the cost path's number, not a variant.
    for got, ref, name in ((expr_from_groups, expr_total, "express"),
                           (pool_from_groups, pool_total, "pooled delivery")):
        assert abs(got - ref) <= _tol(ref), (
            f"{name}: Sigma of per-group prices {got:.6f} != the cost path's "
            f"{ref:.6f} (delta {got - ref:.3e}) -- the attribution is not "
            "splitting the same tours the runner priced")

    dd_total = float(own_cost.sum())
    routing_total = dd_total + expr_total + pool_total
    own_vd = own_vd_day.sum(axis=1)
    pool_vd = pool_vd_day.sum(axis=1)
    expr_vd = expr_vd_day.sum(axis=1)

    # The fleet matrix, from the runner's own function, so the peak days and
    # the vehicle-day total are the ones the operator lens was priced on.
    def _pv(hi: int, d: int, ch: np.ndarray) -> float:
        return (_hub_express_vehicles(hi, d, ch, hub_plz_list, schedules,
                                      raw_express, m, express_cache)
                + _hub_delivery_pool_vehicles(hi, d, ch, hub_plz_list,
                                              schedules, m, pool_cache))

    fleet = _daily_fleet_per_hub(chosen, plz_hub_arr, hub_plz_list, veh_3d,
                                 schedules, pool_veh_fn=_pv,
                                 sched_active=sched_active)
    per_cell_day = own_vd_day + pool_vd_day + expr_vd_day
    hub_idx = np.asarray(plz_hub_arr, dtype=np.int64)
    # Attribution check: the per-cell planes must rebuild the fleet matrix.
    rebuilt = np.zeros_like(fleet)
    for hi in range(len(hub_plz_list)):
        sel = hub_idx == hi
        if sel.any():
            rebuilt[hi] = per_cell_day[sel].sum(axis=0)
    assert np.allclose(rebuilt, fleet, atol=1e-9), (
        "the per-cell vehicle attribution does not rebuild "
        "_daily_fleet_per_hub -- max |diff| "
        f"{float(np.abs(rebuilt - fleet).max()):.3e}")

    peak_day = fleet.argmax(axis=1)
    peak_share = np.zeros(n_plz)
    for hi in range(len(hub_plz_list)):
        sel = hub_idx == hi
        if sel.any():
            peak_share[sel] = per_cell_day[sel, int(peak_day[hi])]

    return dict(
        own_cost=own_cost, pool_share=pool_share,
        express_share=express_share,
        cell_cost=own_cost + pool_share + express_share,
        own_vd=own_vd, pool_vd_share=pool_vd, express_vd_share=expr_vd,
        veh_days_share=own_vd + pool_vd + expr_vd,
        peak_veh_share=peak_share,
        hub_idx=hub_idx, hub_peak_day=peak_day[hub_idx],
        dd_total=dd_total, express_total=expr_total, pool_total=pool_total,
        routing_total=routing_total,
        vehicle_days=float(fleet.sum()),
        sum_hub_peak=float(fleet.max(axis=1).sum()),
        fleet=fleet,
    )


def check_identity(dec: dict, ref_row: pd.Series, plan: str,
                   label: str) -> dict[str, float]:
    """The three identity gates against one grid row.  Raises on any drift."""
    ref = PLAN_REF[plan]
    checks = [
        ("cell_cost_eur", dec["routing_total"], float(ref_row[ref["cost"]]),
         ref["cost"]),
        ("veh_days_share", dec["vehicle_days"], float(ref_row[ref["vd"]]),
         ref["vd"]),
        ("peak_veh_share", dec["sum_hub_peak"], float(ref_row[ref["peak"]]),
         ref["peak"]),
    ]
    out = {}
    for what, got, want, col in checks:
        assert np.isfinite(want), (
            f"{label} plan={plan}: the grid's {col} is {want!r} -- there is "
            "nothing to gate the reconstruction against")
        delta = got - want
        assert abs(delta) <= _tol(want), (
            f"IDENTITY GATE FAILED {label} plan={plan}: "
            f"sum(cells) {what} = {got:.6f} != {col} = {want:.6f} "
            f"(delta {delta:.6e}, tol {_tol(want):.3e})")
        out[col] = delta
    return out


# ─────────────────────────────────────────────────────────────────────────
# Row assembly
# ─────────────────────────────────────────────────────────────────────────
def plan_rows(P: float, th: float, prov: str, plan: str, dec: dict,
              plz_keys: list, chosen: np.ndarray, hub_names: list,
              weekly_pkts: np.ndarray, sched_sizes: np.ndarray,
              sched_waits: np.ndarray, head_id: str) -> list[dict]:
    return [dict(
        penalty=P, share_willing=th, provider=prov, plz=str(pc), plan=plan,
        head_id=head_id,
        hub=hub_names[int(dec["hub_idx"][i])],
        schedule_idx=int(chosen[i]),
        own_cost_eur=float(dec["own_cost"][i]),
        pool_share_eur=float(dec["pool_share"][i]),
        express_share_eur=float(dec["express_share"][i]),
        cell_cost_eur=float(dec["cell_cost"][i]),
        cell_parcels_week=float(weekly_pkts[i]),
        mean_days=float(sched_sizes[int(chosen[i])]),
        wait_days=float(sched_waits[int(chosen[i])]),
        veh_days_share=float(dec["veh_days_share"][i]),
        peak_veh_share=float(dec["peak_veh_share"][i]),
        hub_peak_day=int(dec["hub_peak_day"][i]),
    ) for i, pc in enumerate(plz_keys)]


# ─────────────────────────────────────────────────────────────────────────
# Cross-check against the runner's own per-cell columns (v7 on)
# ─────────────────────────────────────────────────────────────────────────
#: long column -> the runner's wide name, per plan.  ``stage2`` is the
#: runner's name for what this file calls ``balanced``.
WIDE_SUFFIX = {"stage1": "_stage1", "balanced": "_stage2"}
WIDE_COLS = ("own_cost_eur", "pool_share_eur", "express_share_eur",
             "cell_cost_eur")


def crosscheck_wide(pc: pd.DataFrame, chosen: pd.DataFrame) -> str:
    """Assert this file equals the runner's in-line decomposition.

    From commit 634433f the runner writes the same four euro columns per
    plan into ``_tab_chosen_v2.csv``, suffixed ``_stage1`` / ``_stage2``.
    Two independent computations of the same quantity are only worth
    having if they are compared, so compare them -- cell by cell, plan by
    plan, at the runner's own tolerance.  Returns a one-line verdict; a
    grid whose runner predates those columns (v5, v6) is SKIPPED with the
    reason, never silently passed.
    """
    want = [f"{b}{s}" for b in WIDE_COLS for s in WIDE_SUFFIX.values()]
    missing = [c for c in want if c not in chosen.columns]
    if missing:
        return ("SKIPPED -- _tab_chosen_v2.csv predates the runner's own "
                f"per-cell columns (no {missing[0]!r}); this grid was "
                "written before 634433f, so there is nothing to compare "
                "against")
    key = ["penalty", "share_willing", "provider", "plz"]
    ch = chosen.copy()
    ch["plz"] = ch.plz.astype(str)
    worst = 0.0
    n = 0
    for plan, suf in WIDE_SUFFIX.items():
        mine = pc[pc.plan == plan].copy()
        mine["plz"] = mine.plz.astype(str)
        m = mine.merge(ch[key + [f"{b}{suf}" for b in WIDE_COLS]],
                       on=key, how="inner", validate="one_to_one")
        assert len(m) == len(mine), (
            f"plan={plan}: {len(mine)} long row(s) matched {len(m)} wide "
            "row(s) -- the two files disagree on the cell universe")
        for b in WIDE_COLS:
            d = (m[b] - m[f"{b}{suf}"]).abs()
            tol = np.maximum(ABS_TOL, REL_TOL * m[f"{b}{suf}"].abs())
            bad = m[d > tol]
            assert bad.empty, (
                f"WIDE CROSS-CHECK FAILED plan={plan} column={b}: "
                f"{len(bad)} cell(s) differ from the runner's "
                f"{b}{suf}, worst {d.max():.6e} EUR -- e.g. "
                f"{bad.iloc[0][key].to_dict()}")
            worst = max(worst, float(d.max()))
            n += len(m)
    return (f"PASSED -- {n:,d} (cell, plan, column) comparison(s) against "
            f"the runner's own decomposition, worst |delta| "
            f"{worst:.3e} EUR")


# ─────────────────────────────────────────────────────────────────────────
# Resume bookkeeping (per (theta, provider) block)
# ─────────────────────────────────────────────────────────────────────────
def _bkey(th: float, prov: str) -> tuple[float, str]:
    return (round(float(th), 4), str(prov))


def load_done(out_path: Path, expected: dict[tuple[float, str], int]) -> set:
    """Complete (theta, provider) blocks already in *out_path*.

    A block counts as done only when it carries exactly the number of rows
    it should (n_P x 2 plans x n_cells).  A short block -- a kill in the
    middle of its single append -- is DROPPED from the file so it is redone
    cleanly rather than half-trusted.
    """
    if not out_path.exists():
        return set()
    df = pd.read_csv(out_path, dtype={"plz": str})
    if len(df) == 0:
        return set()
    keys = [_bkey(t, v) for t, v in zip(df.share_willing, df.provider)]
    counts = Counter(keys)
    bad = {k for k, n in counts.items() if expected.get(k) not in (None, n)}
    if bad:
        detail = ", ".join(f"{k[1]} th={k[0]}: {counts[k]} rows "
                           f"(want {expected[k]})" for k in sorted(bad, key=str))
        print(f"[self-heal] {out_path.name}: dropping {len(bad)} incomplete "
              f"block(s) -- {detail}", flush=True)
        keep = np.array([k not in bad for k in keys])
        df[keep].to_csv(out_path, index=False)
    return {k for k in counts if k not in bad}


def append_rows(path: Path, rows: list[dict]) -> None:
    """Append, writing the header only on creation; refuse a schema change."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    if path.exists():
        have = list(pd.read_csv(path, nrows=0).columns)
        want = list(df.columns)
        if have != want:
            raise SystemExit(
                f"SCHEMA MISMATCH -- refusing to append to {path}.\n"
                f"  file has {len(have)} columns: {have}\n"
                f"  writing  {len(want)} columns: {want}\n"
                "Delete it (or point --rev-dir elsewhere); appending would "
                "misalign every column after the first difference.")
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


# ─────────────────────────────────────────────────────────────────────────
# Head resolution
# ─────────────────────────────────────────────────────────────────────────
def resolve_head_args(rev: Path, costs: pd.DataFrame, args) -> dict:
    """Head mode / pickle / edges, taken from the grid's own record.

    A grid written under a head carries ``head_manifest.json`` (the file the
    runner pins the directory with) and a ``head_id`` on every row.  Both are
    used: the manifest supplies the paths, the column supplies the identity
    the freshly loaded head must match.  A pre-Task-11 grid has neither and
    is only priceable head-free.
    """
    mpath = rev / "head_manifest.json"
    doc = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else None
    grid_ids = (sorted(set(costs["head_id"].astype(str)))
                if "head_id" in costs.columns else None)
    if grid_ids is not None:
        assert len(grid_ids) == 1, (
            f"{rev}/tab_costs_v2.csv mixes {len(grid_ids)} head_id(s) "
            f"{grid_ids} -- rows priced by different heads are not comparable")

    if doc is None:
        if grid_ids is not None and grid_ids != ["none"]:
            raise SystemExit(
                f"{rev} has no head_manifest.json but its rows carry "
                f"head_id={grid_ids[0]!r}. The manifest is the only record of "
                "WHICH pickle that is; pass --head installed --head-path ... "
                "--edges-path ... explicitly, and the head_id assert will "
                "confirm the choice.")
        mode = args.head or "none"
        if mode != "none":
            raise SystemExit(
                f"{rev} carries no head_manifest.json and no head_id column "
                "-- it is a pre-Task-11 (v5) grid, priced head-free. Run with "
                "--head none.")
        return dict(mode="none", path=None, edges=None,
                    expect_id=(grid_ids or ["none"])[0])

    mode = args.head or str(doc["mode"])
    if args.head and args.head != doc["mode"]:
        raise SystemExit(
            f"--head {args.head} contradicts {mpath.name}, which records "
            f"mode={doc['mode']!r}. The grid was priced under that head; "
            "pricing its plans under another one would not decompose its "
            "numbers.")
    return dict(
        mode=mode,
        path=Path(args.head_path) if args.head_path
        else (Path(doc["path"]) if doc.get("path") else None),
        edges=Path(args.edges_path) if args.edges_path
        else (Path(doc["edges_path"]) if doc.get("edges_path") else None),
        expect_id=str(doc["head_id"]) if grid_ids is None else grid_ids[0],
    )


# ─────────────────────────────────────────────────────────────────────────
def _report_crosscheck(out_path: Path, chosen_tab: pd.DataFrame) -> None:
    """Run and print the wide cross-check, if the file exists."""
    if not out_path.exists():
        return
    pc = pd.read_csv(out_path, dtype={"plz": str})
    print("[wide cross-check] " + crosscheck_wide(pc, chosen_tab),
          flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rev-dir", default="results/revision_2026_08_v6",
                    help="v5/v6-schema grid directory (default: %(default)s)")
    ap.add_argument("--head", choices=("installed", "none"), default=None,
                    help="override the head mode; the default is whatever "
                         "the grid's head_manifest.json records")
    ap.add_argument("--head-path", default=None,
                    help="bundle head pickle (default: the manifest's)")
    ap.add_argument("--edges-path", default=None,
                    help="bundles_bins.json the certified bin names came "
                         "from (default: the manifest's)")
    ap.add_argument("--only", default=None,
                    help="restrict the run, e.g. th=1,prov=DPD (same key "
                         "syntax as 61_'s --only)")
    ap.add_argument("--out", default=None,
                    help=f"output CSV (default: <rev>/tables/{OUT_NAME})")
    args = ap.parse_args(argv)

    rev = Path(args.rev_dir)
    if not rev.is_absolute():
        rev = (ROOT / rev).resolve()
    costs = pd.read_csv(rev / "tab_costs_v2.csv")
    chosen_tab = pd.read_csv(rev / "_tab_chosen_v2.csv", dtype={"plz": str})
    for col in ("cost_stage1_eur", "cost_stage2_eur", "vehicle_days",
                "vehicle_days_before", "sum_hub_peak", "sum_hub_peak_before"):
        assert col in costs.columns, (
            f"{rev}/tab_costs_v2.csv lacks {col!r} -- not a v5/v6-schema "
            "grid; point --rev-dir at a Task-6f (or later) run")
    out_path = (Path(args.out) if args.out
                else rev / "tables" / OUT_NAME)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hs = resolve_head_args(rev, costs, args)
    runner = _load_runner()

    print("=" * 74)
    print(f"per-cell plan costs   rev={rev}")
    print(f"                      out={out_path}")
    print("=" * 74)

    print("[load] checkpoints + model ...", flush=True)
    t0 = time.perf_counter()
    provider_data, optim_data = C.load_checkpoints()
    model = C.load_model()
    head, spec = runner.load_head(hs["mode"], hs["path"], model,
                                  edges_path=hs["edges"])
    assert spec.head_id == hs["expect_id"], (
        f"HEAD MISMATCH: the grid was written under head_id "
        f"{hs['expect_id']!r}, this run loaded {spec.head_id!r}. Pricing the "
        "grid's plans under a different head would not decompose its "
        "numbers -- it would produce different ones.")
    print(f"[head] mode={spec.mode} head_id={spec.head_id} "
          f"(matches the grid)", flush=True)
    ml_prep = C.build_ml_prep(provider_data)
    del provider_data
    gc.collect()
    schedules = C.enumerate_schedules()
    assert len(schedules) == 39, f"expected 39 schedules, got {len(schedules)}"
    sched_sizes = np.array([len(s) for s in schedules], dtype=np.int64)
    sched_waits = np.array([C.avg_wait_days(sorted(s)) for s in schedules])
    print(f"[load] done in {time.perf_counter() - t0:.0f}s", flush=True)

    only_P, only_th, only_prov = (runner.parse_only(args.only) if args.only
                                  else (None, None, None))
    pairs = sorted({(float(p), float(t)) for p, t in
                    zip(costs.penalty, costs.share_willing)})
    thetas = sorted({t for _, t in pairs})
    if only_th is not None:
        thetas = [t for t in thetas if np.isclose(t, only_th)]
    providers = [p for p in C.PROVIDERS
                 if p in set(costs.provider) and
                 (only_prov is None or p == only_prov)]
    if not thetas or not providers:
        raise SystemExit(f"--only {args.only!r} matched no grid point")

    n_cells = {p: len(optim_data[p]["plz_keys"]) for p in providers}
    expected = {}
    for th in thetas:
        Ps = [p for p, t in pairs if np.isclose(t, th)
              and (only_P is None or np.isclose(p, only_P))]
        for prov in providers:
            expected[_bkey(th, prov)] = len(Ps) * len(PLAN_COL) * n_cells[prov]
    done = load_done(out_path, expected)
    todo = [(th, prov) for th in thetas for prov in providers
            if _bkey(th, prov) not in done]
    print(f"[grid] {len(thetas)} theta x {len(providers)} provider(s); "
          f"{len(done)} block(s) already done; {len(todo)} to build",
          flush=True)
    if not todo:
        print("nothing to do", flush=True)
        _report_crosscheck(out_path, chosen_tab)
        return 0

    t_run = time.perf_counter()
    n_done = 0
    max_delta = 0.0
    for th in thetas:
        Ps = sorted({p for p, t in pairs if np.isclose(t, th)
                     and (only_P is None or np.isclose(p, only_P))})
        fs_b2c_v, fs_b2b_v = C.fs_b2c(th), C.fs_b2b(th)
        for prov in providers:
            if _bkey(th, prov) in done:
                continue
            od, prep = optim_data[prov], ml_prep[prov]
            plz_keys = od["plz_keys"]
            plz_data = od["plz_data"]
            hub_plz_list = od["hub_plz_list"]
            plz_hub_arr = od["plz_hub_arr"]
            hub_names = [
                prep["hub_name_by_plz"].get(plz_keys[int(h[0])], f"hub_{hi}")
                if len(h) else f"hub_{hi}"
                for hi, h in enumerate(hub_plz_list)]
            weekly_pkts = np.array(
                [sum(plz_data[pc]["b2c"].values())
                 + sum(plz_data[pc]["b2b"].values()) for pc in plz_keys],
                dtype=np.float64)

            t_b = time.perf_counter()
            m = runner.build_cost_matrices_ml(
                plz_keys, plz_data, schedules, model, prov,
                prep["plz_day_coords"], prep["hub_coords_by_plz"],
                fast_share_b2c=fs_b2c_v, fast_share_b2b=fs_b2b_v)
            assert m.get("small_delivery_price") is not None, (
                "matrices lack 'small_delivery_price' -- the pooled twin "
                "would fall back to per-member partition pricing")
            runner.install_head(m, head, spec)
            t_mtx = time.perf_counter() - t_b

            rows: list[dict] = []
            memo: dict[bytes, dict] = {}
            n_priced = 0
            for P in Ps:
                sel = chosen_tab[np.isclose(chosen_tab.penalty, P)
                                 & np.isclose(chosen_tab.share_willing, th)
                                 & (chosen_tab.provider == prov)]
                assert len(sel) == len(plz_keys), (
                    f"P={P} th={th} {prov}: _tab_chosen_v2.csv holds "
                    f"{len(sel)} rows for {len(plz_keys)} cells")
                sel = sel.set_index("plz").loc[[str(k) for k in plz_keys]]
                cref = costs[np.isclose(costs.penalty, P)
                             & np.isclose(costs.share_willing, th)
                             & (costs.provider == prov)]
                assert len(cref) == 1, (
                    f"P={P} th={th} {prov}: {len(cref)} cost row(s)")
                cref = cref.iloc[0]
                for plan, col in PLAN_COL.items():
                    ch = sel[col].to_numpy().astype(np.int64)
                    key = ch.tobytes()
                    dec = memo.get(key)
                    if dec is None:
                        dec = decompose_plan(ch, plz_hub_arr, hub_plz_list,
                                             schedules, m)
                        memo[key] = dec
                        n_priced += 1
                    d = check_identity(dec, cref, plan,
                                       f"P={P} th={th} {prov}")
                    max_delta = max(max_delta,
                                    max(abs(v) for v in d.values()))
                    rows += plan_rows(P, th, prov, plan, dec, plz_keys, ch,
                                      hub_names, weekly_pkts, sched_sizes,
                                      sched_waits, spec.head_id)
            append_rows(out_path, rows)
            done.add(_bkey(th, prov))
            n_done += 1
            el = time.perf_counter() - t_run
            eta = el * (len(todo) - n_done) / max(1, n_done) / 60.0
            print(f"[{n_done:3d}/{len(todo)}] th={th:<4g} {prov:<7s} "
                  f"mtx={t_mtx:5.1f}s  {n_priced} distinct plan(s) priced of "
                  f"{len(Ps) * len(PLAN_COL)}  {len(rows)} row(s)  "
                  f"gates OK (max |delta| {max_delta:.2e})  "
                  f"eta={eta:.1f}min", flush=True)
            del m, memo
            gc.collect()

    _report_crosscheck(out_path, chosen_tab)
    print(f"\n[done] {n_done} block(s) in "
          f"{(time.perf_counter() - t_run) / 60:.1f}min")
    print(f"  {out_path} ({len(pd.read_csv(out_path)):,d} rows)")
    print(f"  every identity gate passed; worst |delta| {max_delta:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
