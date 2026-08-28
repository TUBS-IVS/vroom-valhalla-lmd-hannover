"""Task 11c — the grid runner's NATIVE per-cell plan costs.

``61_grid_run_v2.py`` now writes what one postal-code area pays, at both
plans, straight into ``_tab_chosen_v2.csv``:
``own_cost_eur_{stage1,stage2}``, ``pool_share_eur_{stage1,stage2}``,
``express_share_eur_{stage1,stage2}``, ``cell_cost_eur_{stage1,stage2}`` and
``cell_parcels_week`` — so Fig. 6 (b)-(f)'s per-PLZ EUR saving never needs
Task 13B's post-hoc ``72_per_cell_costs_v2.py`` reconstruction for a v7+
grid. These tests pin the three new functions behind that
(``cell_plan_costs``, ``_parcel_shares``, ``_assert_cell_cost_identity``) on
a tiny synthetic 3-cell fixture — one cell above ``MIN_TOUR_PARCELS`` runs
its own tour, two below are pooled together, and ``fast_share < 1`` so a
non-delivery day also carries an express residual — and the schema guard
that must accept the new columns on a fresh output directory while refusing
to append them onto an existing pre-11c ``_tab_chosen_v2.csv``.

The fixture (``SPECS`` / ``FAST_SHARE`` below) is deliberately the SAME one
Task 13B's ``test_per_cell_costs_v2.py`` uses for its ``decompose_plan`` —
independent files, same numbers, so a result here and the analogous result
there are directly comparable by eye, not just by having similar shape.

No Docker, no model pickle: ``_stubs.StubPredictor`` is linear in
``n_parcels`` and ``hub_dist_km``. The runner is imported by path (its
module name starts with a digit), the same way ``test_head_enabled_grid.py``
does it.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _stubs import cell_matrices
from batch_delivery.optimization.costs import (
    _hub_express_day_ml,
    _hub_smallday_pool_ml,
)

_RUNNER = (Path(__file__).resolve().parents[2] / "scripts" / "revision"
           / "61_grid_run_v2.py")


@lru_cache(maxsize=1)
def runner():
    """Import ``scripts/revision/61_grid_run_v2.py`` by path.

    Its module body disables INFO logging and installs a blanket warnings
    filter (both wanted for a multi-hour grid run, neither wanted in a test
    session) — undone here so importing the runner has no session-wide side
    effects, same as ``test_head_enabled_grid.py``'s loader.
    """
    filters = warnings.filters[:]
    spec = importlib.util.spec_from_file_location(
        "grid_run_v2_cellcosts", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    logging.disable(logging.NOTSET)
    warnings.filters[:] = filters
    return mod


# ─────────────────────────────────────────────────────────────────────────
# Fixture: one own-tour cell, one pooled group of two, one express partition
# ─────────────────────────────────────────────────────────────────────────

#: one big cell (own tour) + two small co-located ones (pooled together) --
#: identical to Task 13B's 72_ fixture on purpose (see module docstring).
SPECS = [(400.0, 100.0), (30.0, 10.0), (25.0, 8.0)]
#: < 1 so a non-delivery day also carries an express residual.
FAST_SHARE = 0.6


@pytest.fixture(scope="module")
def fixture():
    plz_keys, m, sch, hub_plz_list, plz_hub_arr = cell_matrices(
        SPECS, fs=FAST_SHARE, spread=0.0)
    three_day = next(i for i, s in enumerate(sch) if len(s) == 3)
    chosen = np.array([three_day] * len(SPECS), dtype=np.int64)
    return dict(plz_keys=plz_keys, m=m, sch=sch, hub_plz_list=hub_plz_list,
                plz_hub_arr=plz_hub_arr, chosen=chosen, three_day=three_day)


@pytest.fixture(scope="module")
def dec(fixture):
    R = runner()
    return R.cell_plan_costs(fixture["chosen"], fixture["hub_plz_list"],
                             fixture["sch"], fixture["m"])


def _independent_routing_total(fixture, chosen: np.ndarray) -> float:
    """dd + express + pool, recomputed via the cost path's OWN hub-day sums.

    This is exactly what ``run_triple`` computes as ``routing_total`` before
    it ever calls ``cell_plan_costs`` — an independent reference the
    per-cell decomposition's identity gate must reproduce, built without
    going anywhere near ``cell_plan_costs`` or ``_parcel_shares``.
    """
    m, hpl, sch = fixture["m"], fixture["hub_plz_list"], fixture["sch"]
    express_cache: dict = {}
    pool_cache: dict = {}
    expr_total = pool_total = 0.0
    for hi in range(len(hpl)):
        for d in range(6):
            expr_total += _hub_express_day_ml(
                hi, d, chosen, hpl, sch, m["raw_express"], m["expr_stops"],
                m, express_cache, 1.0)
            pool_total += _hub_smallday_pool_ml(
                hi, d, chosen, hpl, sch, m, pool_cache)
    dd_total = float(m["dd_cost_mx"][np.arange(len(chosen)), chosen].sum())
    return dd_total + expr_total + pool_total


# ─────────────────────────────────────────────────────────────────────────
# cell_plan_costs: allocation
# ─────────────────────────────────────────────────────────────────────────

def test_the_fixture_exercises_all_three_cost_terms(dec):
    """A test that silently priced only own tours would prove nothing."""
    assert dec["dd_total"] > 0, "no own-tour cost -- fixture is degenerate"
    assert dec["pool_total"] > 0, "no pooled small-delivery cost"
    assert dec["express_total"] > 0, "no express residual cost"


def test_the_own_tour_cell_pays_its_tour_and_nothing_pooled(dec):
    assert dec["own_cost"][0] > 0
    assert dec["pool_share"][0] == 0.0


def test_the_pooled_cells_never_run_their_own_tour(dec):
    assert (dec["own_cost"][1:] == 0).all()
    assert (dec["pool_share"][1:] > 0).all()


def test_cell_cost_is_the_sum_of_its_three_components(dec):
    assert np.allclose(
        dec["cell_cost"],
        dec["own_cost"] + dec["pool_share"] + dec["express_share"])


def test_cell_costs_sum_to_the_three_totals(dec):
    assert dec["cell_cost"].sum() == pytest.approx(
        dec["dd_total"] + dec["pool_total"] + dec["express_total"], rel=1e-12)


def test_pool_share_is_split_parcel_proportionally(fixture):
    """The two pooled cells split their group's price like their demand.

    Checked on the DAILY plan, same as Task 13B's analogous test: with one
    source day behind every delivery day, the pooled demand is identical on
    all six days, so the week-aggregate ratio is exactly the per-tour ratio
    with no per-day gap-length bookkeeping needed. (The module fixture's
    3-day, non-uniform-gap schedule does NOT keep this ratio exactly
    constant across its delivery days -- cells 1 and 2 have slightly
    different b2c/b2b mixes, which combine differently over different held
    gap lengths -- so the daily plan, not ``dec``, is the clean case to
    check the allocation RULE against.)
    """
    R = runner()
    m, sch, hpl = fixture["m"], fixture["sch"], fixture["hub_plz_list"]
    daily = next(i for i, s in enumerate(sch) if len(s) == 6)
    chosen = np.array([daily] * len(SPECS), dtype=np.int64)
    d = R.cell_plan_costs(chosen, hpl, sch, m)
    assert d["pool_total"] > 0 and d["express_total"] == 0.0, (
        "the daily plan should pool the two small cells and have no "
        "express residual at all")
    cd = m["combined_demand"]
    per_day = np.array([cd[z, daily, 0] for z in (1, 2)])
    want = per_day[0] / per_day.sum()
    got = d["pool_share"][1] / (d["pool_share"][1] + d["pool_share"][2])
    assert got == pytest.approx(want, rel=1e-9)


def test_express_share_is_split_parcel_proportionally(fixture, dec):
    """Same rule on the express residual pool of the two small cells.

    ``raw_express[z, d]`` does not depend on the schedule and is constant
    across the week in this fixture, so the aggregate ratio is the per-tour
    ratio exactly.
    """
    m, ch = fixture["m"], fixture["chosen"]
    d_off = next(d for d in range(6) if not m["sched_active"][ch[0], d])
    rx = m["raw_express"][:, d_off]
    want = rx[1] / (rx[1] + rx[2])
    got = (dec["express_share"][1]
           / (dec["express_share"][1] + dec["express_share"][2]))
    assert got == pytest.approx(want, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────────
# _parcel_shares
# ─────────────────────────────────────────────────────────────────────────

def test_parcel_shares_sums_to_one_and_is_proportional():
    R = runner()
    w = np.array([0.0, 30.0, 10.0, 0.0])
    s = R._parcel_shares((1, 2), w)
    assert s.sum() == pytest.approx(1.0)
    assert s[0] / s[1] == pytest.approx(3.0)


def test_parcel_shares_refuses_a_zero_parcel_group():
    R = runner()
    with pytest.raises(AssertionError, match="parcel-proportional share"):
        R._parcel_shares((0, 1), np.zeros(3))


# ─────────────────────────────────────────────────────────────────────────
# the identity gate itself
# ─────────────────────────────────────────────────────────────────────────

def test_identity_holds_against_an_independently_recomputed_routing_total(
        fixture, dec):
    """The gate ``run_triple`` runs per (P, theta, provider, plan).

    The reference is built by ``_independent_routing_total``, which never
    calls ``cell_plan_costs`` or ``_parcel_shares`` -- it sums the cost
    path's own ``_hub_express_day_ml`` / ``_hub_smallday_pool_ml`` the way
    ``run_triple`` itself does. This is the real end-to-end check, not a
    tautology against ``cell_plan_costs``'s own totals.
    """
    R = runner()
    ref = _independent_routing_total(fixture, fixture["chosen"])
    R._assert_cell_cost_identity(dec["cell_cost"], ref, "routing_total_eur",
                                 "synthetic")


def test_assert_cell_cost_identity_fails_loud_on_a_wrong_total(dec):
    R = runner()
    bad_ref = float(dec["cell_cost"].sum()) + 1.0
    with pytest.raises(AssertionError, match="IDENTITY GATE FAILED"):
        R._assert_cell_cost_identity(dec["cell_cost"], bad_ref,
                                     "cost_stage1_eur", "synthetic")


def test_assert_cell_cost_identity_passes_on_a_tiny_relative_drift(dec):
    """1e-9 relative is float re-association, not a bookkeeping defect."""
    R = runner()
    ref = float(dec["cell_cost"].sum())
    R._assert_cell_cost_identity(dec["cell_cost"], ref * (1 + 1e-12),
                                 "cost_stage1_eur", "synthetic")


# ─────────────────────────────────────────────────────────────────────────
# two plans: independent decompositions, both correctly identity-checked
# ─────────────────────────────────────────────────────────────────────────

def test_two_different_plans_decompose_and_identity_check_independently(
        fixture):
    """Mirrors run_triple's stage-1 vs stage-2 wiring on a plan that MOVES.

    Cell 0 (the big one) switches from the 3-day pattern to daily between
    "stage 1" and "stage 2" here; cells 1/2 stay put. Both decompositions
    must independently satisfy the identity gate against their OWN
    independently recomputed routing total, and must actually differ where
    the plan differs (a test that silently reused stage 1's numbers for
    stage 2 would still pass a naive shape check).
    """
    R = runner()
    sch, hpl = fixture["sch"], fixture["hub_plz_list"]
    three_day = fixture["three_day"]
    daily = next(i for i, s in enumerate(sch) if len(s) == 6)

    chosen_s1 = np.array([three_day, three_day, three_day], dtype=np.int64)
    chosen_s2 = np.array([daily, three_day, three_day], dtype=np.int64)

    dec_s1 = R.cell_plan_costs(chosen_s1, hpl, sch, fixture["m"])
    dec_s2 = R.cell_plan_costs(chosen_s2, hpl, sch, fixture["m"])

    ref_s1 = _independent_routing_total(fixture, chosen_s1)
    ref_s2 = _independent_routing_total(fixture, chosen_s2)
    R._assert_cell_cost_identity(dec_s1["cell_cost"], ref_s1,
                                 "cost_stage1_eur", "synthetic stage1")
    R._assert_cell_cost_identity(dec_s2["cell_cost"], ref_s2,
                                 "cost_stage2_eur", "synthetic stage2")

    # the moved cell's own-tour cost actually changed between the two plans
    assert dec_s1["own_cost"][0] != pytest.approx(dec_s2["own_cost"][0])
    # the untouched pooled cells' shares are unaffected by cell 0's move
    assert dec_s1["pool_share"][1] == pytest.approx(dec_s2["pool_share"][1])
    assert dec_s1["pool_share"][2] == pytest.approx(dec_s2["pool_share"][2])


# ─────────────────────────────────────────────────────────────────────────
# schema guard: additive for a fresh dir, refused for a pre-11c one
# ─────────────────────────────────────────────────────────────────────────

_OLD_COLS = ["penalty", "share_willing", "provider", "head_id", "plz",
             "schedule_idx_stage1", "schedule_idx_balanced",
             "schedule_idx_system_smoothed"]
_NEW_ONLY_COLS = ["own_cost_eur_stage1", "pool_share_eur_stage1",
                  "express_share_eur_stage1", "cell_cost_eur_stage1",
                  "own_cost_eur_stage2", "pool_share_eur_stage2",
                  "express_share_eur_stage2", "cell_cost_eur_stage2",
                  "cell_parcels_week"]


def _row(**override) -> dict:
    row = {c: 0.0 for c in _OLD_COLS + _NEW_ONLY_COLS}
    row.update(penalty=0.0, share_willing=0.0, provider="DHL",
               head_id="none", plz="30159")
    row.update(override)
    return row


def test_append_rows_writes_the_new_columns_on_a_fresh_output_dir(tmp_path):
    """Additive-only requirement, half 1: a NEW directory just gets them."""
    R = runner()
    p = tmp_path / "_tab_chosen_v2.csv"
    R.append_rows(p, [_row()])
    df = pd.read_csv(p)
    for c in _NEW_ONLY_COLS:
        assert c in df.columns, f"{c!r} missing from a fresh-dir write"


def test_append_rows_refuses_a_pre_11c_chosen_file(tmp_path):
    """Additive-only requirement, half 2: an existing pre-11c dir refuses.

    A pre-Task-11c ``_tab_chosen_v2.csv`` carries the old 8 columns only;
    appending an 11c-schema row (17 columns) must raise, not silently
    misalign every column after ``schedule_idx_system_smoothed``.
    """
    R = runner()
    p = tmp_path / "_tab_chosen_v2.csv"
    pd.DataFrame([{c: 0 for c in _OLD_COLS}]).to_csv(p, index=False)
    with pytest.raises(SystemExit, match="SCHEMA MISMATCH"):
        R.append_rows(p, [_row()])


def test_append_rows_still_accepts_a_matching_11c_file(tmp_path):
    """A resumed 11c-native run (same 17 columns already on disk) appends."""
    R = runner()
    p = tmp_path / "_tab_chosen_v2.csv"
    R.append_rows(p, [_row(plz="30159")])
    R.append_rows(p, [_row(plz="30161")])   # must not raise
    assert len(pd.read_csv(p)) == 2
