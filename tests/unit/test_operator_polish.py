"""Stage 2 as an operator-cost minimisation (Task 6e, spec v3 §4.3).

The balancing objective is no longer "flatten the per-hub range inside a cost
budget" but the money the operator actually pays for a week of the provider's
solution:

    OpCost = variable + W * Sigma_h peak_h  (+ penalty)
    variable = routing_cost - 189.15 * vehicle_days
    W        = 6 * 189.15 EUR per peak vehicle per hub per week

Every VROOM label the surrogate learned from was priced with
``cost = 189.15 * n_vehicles + 0.3864 * km + 36 * route_hours`` (Task 6e brief,
verified on the training pool), so subtracting ``189.15 * vehicle_days`` from a
predicted cost leaves exactly the distance + time part. Below the hub's weekly
peak an extra vehicle-day costs only that variable part — the van is owned and
the driver employed either way — so the fixed bill is charged once per week per
peak vehicle.

Covered here:

* G-6e-2  the accept rule — no accepted move has ``dOpCost >= 0``, and the
  objective never rises across a run.
* G-6e-3  bookkeeping — the running OpCost after a sequence of accepted moves
  equals a from-scratch recomputation (rel 1e-9), with and without a penalty
  matrix.
* the decomposition identity ``routing == fixed + variable`` with
  ``vehicle_days`` taken from the masked, partition-aware fleet profile.

G-6e-1 (theta=1 stage-1 choices unchanged) is a property of the coordinate
descent, which Task 6e does not touch: ``optimization/coordinate_descent.py``
and ``optimization/costs.py`` carry no diff, and the runner-level identity is
gated empirically in the task report.
"""
import importlib.util
import logging
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from _stubs import StubPredictor, tiny_matrices

from batch_delivery.config.constants import FIXED_COST_EUR, N_DAYS
from batch_delivery.config.constants import FLEET_BALANCE_MAX_SWAPS
from batch_delivery.optimization.balancing import (
    BEST_OF_N_STARTS,
    RANGE_START_BUDGET_PCT,
    RANGE_START_MAX_SWAPS,
    WEEK_FIXED_COST_EUR,
    _daily_fleet_per_hub,
    balance_fleet_per_hub_ml,
    operator_cost_breakdown,
    operator_polish,
    operator_polish_best_of_n,
    operator_polish_best_of_two,
)
from batch_delivery.optimization.costs import (
    _hub_delivery_pool_vehicles,
    _hub_express_vehicles,
    build_cost_matrices_ml,
)
from batch_delivery.optimization.schedules import enumerate_valid_schedules


def _pool_fn(hpl, sch, m):
    """The production closure: express + pooled-delivery vehicles."""
    ec: dict = {}
    pc: dict = {}
    return lambda hi, d, ch: (
        _hub_express_vehicles(hi, d, ch, hpl, sch, m["raw_express"], m, ec)
        + _hub_delivery_pool_vehicles(hi, d, ch, hpl, sch, m, pc)
    )


def _fixture(fs: float, n_cells: int = 10, n_hubs: int = 1):
    """The 10-cell mixed-size fixture the balancing gates already use.

    Same construction as ``test_fleet_objective_v2._bal_fixture`` (a mix of
    sub- and super-threshold cells so both pooled partitions re-form under a
    move); ``n_hubs`` splits the cells round-robin so the per-hub peak term
    has more than one hub to charge.
    """
    rng = np.random.default_rng(0)
    plz_keys = [f"3{i:04d}" for i in range(n_cells)]
    plz_data, coords, hubs = {}, {}, {}
    for i, p in enumerate(plz_keys):
        base = 60 + 90 * i          # 60 .. 870: mix of small and large cells
        plz_data[p] = {
            "b2c": {d: float(base + rng.integers(-15, 15)) for d in range(6)},
            "b2b": {d: float(base * 0.15) for d in range(6)},
            "area_km2": 4.0 + 2.0 * (i % 5),
            "hub_dist_km": 3.0 + 1.5 * i,
            "n_stops_per_day": 30.0 + 20.0 * (i % 4),
            "total_points": 400.0 + 100.0 * i,
        }
        lon = 9.70 + 0.01 * i
        lat = 52.35 + 0.01 * (i % 3)
        coords[p] = {d: (np.array([lon, lon + 0.002]),
                         np.array([lat, lat + 0.002]),
                         np.array([3.0, 4.0])) for d in range(6)}
        hubs[p] = (9.73, 52.38)
    sch = enumerate_valid_schedules()
    m = build_cost_matrices_ml(
        plz_keys, plz_data, sch, StubPredictor(), "DHL",
        coords, hubs, fast_share_b2c=fs, fast_share_b2b=fs)
    pha = np.arange(n_cells) % n_hubs
    hpl = [np.flatnonzero(pha == h) for h in range(n_hubs)]
    return plz_keys, m, sch, hpl, pha


def _start(m, sch, n_cells, seed):
    rng = np.random.default_rng(1000 + seed)
    return rng.integers(0, len(sch), size=n_cells).astype(np.int64)


def _penalty(m, sch, n_cells, scale=1.0):
    """A wait penalty with the runner's shape: P * willing * pkts * wait."""
    waits = np.array([
        np.mean([min((d - di) % N_DAYS for d in s) for di in range(N_DAYS)])
        for s in sch
    ])
    pkts = m["daily_demand"].sum(axis=1)
    return scale * pkts[:, None] * waits[None, :]


# ─────────────────────────────────────────────────────────────────────────────
# The decomposition itself
# ─────────────────────────────────────────────────────────────────────────────

def test_decomposition_identity_routing_is_fixed_plus_variable():
    """routing == 189.15 * vehicle_days + variable, with vehicle_days read
    off the SAME masked partition-aware profile the balancing objective uses.

    Guards the wiring, not the algebra: ``vehicle_days`` must come from
    ``_daily_fleet_per_hub`` with the pooled closure supplied (delivery-day
    masking + one ceil per pooled tour), never from a raw ``veh_3d`` sum.
    """
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.5, n_cells=10, n_hubs=2)
    for seed in (0, 8, 59):
        chosen = _start(m, sch, 10, seed)
        b = operator_cost_breakdown(chosen, plz_keys, pha, hpl, m, sch)

        fleet = _daily_fleet_per_hub(
            chosen, pha, hpl, m["veh_3d"], sch,
            pool_veh_fn=_pool_fn(hpl, sch, m), sched_active=m["sched_active"])
        assert b["vehicle_days"] == pytest.approx(float(fleet.sum()))
        assert b["sum_hub_peak"] == pytest.approx(
            float(fleet.max(axis=1).sum()))
        # ... and it is NOT the unmasked per-cell count (the §0 double count)
        raw = float(m["veh_3d"][np.arange(10), chosen, :].sum())
        assert b["vehicle_days"] != pytest.approx(raw)

        assert b["fixed_cost"] == pytest.approx(
            FIXED_COST_EUR * b["vehicle_days"])
        assert b["routing_cost"] == pytest.approx(
            b["fixed_cost"] + b["variable_cost"], rel=1e-12)
        assert b["routing_cost"] == pytest.approx(
            b["dd_cost"] + b["express_cost"] + b["pool_cost"], rel=1e-12)
        assert b["operator_cost"] == pytest.approx(
            b["variable_cost"] + WEEK_FIXED_COST_EUR * b["sum_hub_peak"])
        assert b["opcost"] == pytest.approx(b["operator_cost"] + b["penalty"])
    assert WEEK_FIXED_COST_EUR == pytest.approx(FIXED_COST_EUR * N_DAYS)


# ─────────────────────────────────────────────────────────────────────────────
# G-6e-2 — the accept rule
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fs,seed", [(0.7, 8), (0.5, 59), (0.5, 0)])
def test_no_accepted_move_has_nonnegative_delta(fs, seed):
    """Every accepted move must strictly lower the operator objective."""
    plz_keys, m, sch, hpl, pha = _fixture(fs=fs, n_cells=10, n_hubs=2)
    chosen = _start(m, sch, 10, seed)
    res = operator_polish(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        max_swaps=40, seed=seed)

    assert res["swaps_made"] > 0                    # exercises the accept path
    assert len(res["accepted_deltas"]) == res["swaps_made"]
    assert all(dlt < 0.0 for dlt in res["accepted_deltas"])
    assert res["opcost_after"] <= res["opcost_before"] + 1e-9
    # the accepted deltas ARE the movement of the tracked objective
    assert res["opcost_after"] == pytest.approx(
        res["opcost_before"] + sum(res["accepted_deltas"]),
        rel=1e-9, abs=1e-6)


def test_direction_holds_with_a_penalty_matrix():
    """The service penalty is part of the objective, so the direction
    guarantee must hold for the PENALISED quantity too."""
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.5, n_cells=10, n_hubs=2)
    chosen = _start(m, sch, 10, 0)
    pen = _penalty(m, sch, 10, scale=0.5)
    res = operator_polish(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        max_swaps=40, seed=0, penalty_mx=pen)
    assert res["swaps_made"] > 0
    assert all(dlt < 0.0 for dlt in res["accepted_deltas"])
    assert res["opcost_after"] <= res["opcost_before"] + 1e-9


def test_theta_one_has_no_express_and_still_polishes_downhill():
    """At theta=1 nothing is express, so the whole objective rides on the
    delivery-day profile; the accept rule must still hold there."""
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.0, n_cells=10, n_hubs=2)
    assert not (m["raw_express"] > 0).any()
    chosen = _start(m, sch, 10, 3)
    res = operator_polish(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        max_swaps=40, seed=3)
    assert all(dlt < 0.0 for dlt in res["accepted_deltas"])
    assert res["opcost_after"] <= res["opcost_before"] + 1e-9


def test_max_swaps_zero_is_a_pure_measurement():
    """``max_swaps=0`` must report the input state and change nothing —
    the runner reads ``initial_total_cost`` off exactly this path."""
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.5, n_cells=10)
    chosen = _start(m, sch, 10, 0)
    res = operator_polish(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch, max_swaps=0)
    assert res["swaps_made"] == 0
    assert np.array_equal(res["chosen"], chosen)
    assert res["opcost_after"] == res["opcost_before"]
    assert res["cost"] == res["initial_total_cost"]
    b = operator_cost_breakdown(chosen, plz_keys, pha, hpl, m, sch)
    assert res["opcost_before"] == pytest.approx(b["opcost"], rel=1e-12)
    assert res["cost"] == pytest.approx(b["routing_cost"], rel=1e-12)


def test_inputs_are_not_mutated():
    """Stage 1's own state must survive stage 2 untouched: the polish works
    on a copy of ``chosen`` and never writes into the matrices it reads."""
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.5, n_cells=10, n_hubs=2)
    chosen = _start(m, sch, 10, 0)
    chosen_in = chosen.copy()
    dd_in = m["dd_cost_mx"].copy()
    veh_in = m["veh_3d"].copy()
    res = operator_polish(
        {"chosen": chosen_in}, plz_keys, pha, hpl, m, sch,
        max_swaps=40, seed=0)
    assert res["swaps_made"] > 0
    assert np.array_equal(chosen_in, chosen)          # input untouched
    assert np.array_equal(m["dd_cost_mx"], dd_in)
    assert np.array_equal(m["veh_3d"], veh_in)


def test_preserve_frequency_keeps_every_cell_s_delivery_count():
    """The production wiring pins the frequency (stage 1 owns service
    quality); then no move may change how MANY days a cell is served on."""
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.5, n_cells=10, n_hubs=2)
    chosen = _start(m, sch, 10, 0)
    res = operator_polish(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        max_swaps=40, seed=0, preserve_frequency=True)
    assert res["swaps_made"] > 0
    for pi in range(10):
        assert len(sch[int(res["chosen"][pi])]) == len(sch[int(chosen[pi])])


# ─────────────────────────────────────────────────────────────────────────────
# G-6e-3 — bookkeeping
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("use_pen", [False, True])
@pytest.mark.parametrize("fs,seed,n_hubs", [(0.7, 8, 1), (0.5, 0, 2),
                                            (0.5, 59, 3)])
def test_running_opcost_equals_a_from_scratch_recompute(use_pen, fs, seed,
                                                        n_hubs):
    """After a sequence of accepted moves the incrementally-tracked terms
    must equal an independent recomputation at the final ``chosen``.

    Mirrors the existing cost-bookkeeping gates: the caches
    (``ecache``/``pcache``/``pvcache``) and the hub fleet rows the polish
    carries forward are the only things that could drift, and every reported
    number is derived from them.
    """
    plz_keys, m, sch, hpl, pha = _fixture(fs=fs, n_cells=10, n_hubs=n_hubs)
    chosen = _start(m, sch, 10, seed)
    pen = _penalty(m, sch, 10, scale=0.25) if use_pen else None

    res = operator_polish(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        max_swaps=40, seed=seed, penalty_mx=pen)
    assert res["swaps_made"] > 0

    b0 = operator_cost_breakdown(chosen, plz_keys, pha, hpl, m, sch,
                                 penalty_mx=pen)
    b1 = operator_cost_breakdown(res["chosen"], plz_keys, pha, hpl, m, sch,
                                 penalty_mx=pen)
    for key, before, after in (
        ("opcost", "opcost_before", "opcost_after"),
        ("variable_cost", "variable_before", "variable_after"),
        ("sum_hub_peak", "sum_hub_peak_before", "sum_hub_peak_after"),
        ("vehicle_days", "vehicle_days_before", "vehicle_days_after"),
    ):
        assert res[before] == pytest.approx(b0[key], rel=1e-9, abs=1e-6), before
        assert res[after] == pytest.approx(b1[key], rel=1e-9, abs=1e-6), after
    assert res["initial_total_cost"] == pytest.approx(
        b0["routing_cost"], rel=1e-9, abs=1e-6)
    assert res["cost"] == pytest.approx(b1["routing_cost"], rel=1e-9, abs=1e-6)


def test_reported_fleet_terms_match_the_profile_at_the_final_choice():
    """``vehicle_days_after`` / ``sum_hub_peak_after`` must be the masked,
    partition-aware profile at the polish's own final ``chosen`` — the very
    numbers the runner turns into ``fixed_cost_eur`` and the weekly bill."""
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.5, n_cells=10, n_hubs=2)
    chosen = _start(m, sch, 10, 0)
    res = operator_polish(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        max_swaps=40, seed=0)
    fleet = _daily_fleet_per_hub(
        res["chosen"], pha, hpl, m["veh_3d"], sch,
        pool_veh_fn=_pool_fn(hpl, sch, m), sched_active=m["sched_active"])
    assert res["vehicle_days_after"] == pytest.approx(float(fleet.sum()))
    assert res["sum_hub_peak_after"] == pytest.approx(
        float(fleet.max(axis=1).sum()))
    assert res["variable_after"] == pytest.approx(
        res["cost"] - FIXED_COST_EUR * res["vehicle_days_after"])


# ─────────────────────────────────────────────────────────────────────────────
# The objective really is the operator's, not the old range heuristic
# ─────────────────────────────────────────────────────────────────────────────

def test_polish_buys_below_peak_vehicle_days_to_shave_the_peak():
    """The objective is the operator's bill, not the routing cost.

    On this fixture the polish ENDS with a higher routing cost and MORE
    vehicle-days than it started, because the extra days sit below the hub
    peak (variable cost only) while the peak itself — the weekly fixed bill —
    drops hard. A routing-cost or a vehicle-day objective would have rejected
    exactly those moves.
    """
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.5, n_cells=10, n_hubs=2)
    seen_costlier_routing = False
    seen_more_vehicle_days = False
    for seed in (0, 3, 8, 17, 59):
        chosen = _start(m, sch, 10, seed)
        res = operator_polish(
            {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
            max_swaps=40, seed=seed)
        assert res["swaps_made"] > 0
        assert res["opcost_after"] < res["opcost_before"]
        # Rule-derived, not a guessed direction: whatever the three terms did,
        # the objective's movement IS their priced sum. (The peak is allowed
        # to RISE when the variable saving outweighs W * Delta peak — see
        # test_a_cheap_peak_lets_the_variable_term_win.)
        assert (res["opcost_after"] - res["opcost_before"]) == pytest.approx(
            (res["variable_after"] - res["variable_before"])
            + WEEK_FIXED_COST_EUR * (res["sum_hub_peak_after"]
                                     - res["sum_hub_peak_before"])
            + (res["penalty_after"] - res["penalty_before"]),
            rel=1e-9, abs=1e-6)
        seen_costlier_routing |= res["cost"] > res["initial_total_cost"]
        seen_more_vehicle_days |= (
            res["vehicle_days_after"] > res["vehicle_days_before"])
    assert seen_costlier_routing, "no move traded routing cost for peak"
    assert seen_more_vehicle_days, "no move traded vehicle-days for peak"


def test_the_result_is_a_local_optimum_of_the_move_neighbourhood():
    """Sweeping until nothing improves means exactly that: a second polish
    from the result — with a different sweep order — accepts nothing."""
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.5, n_cells=10, n_hubs=2)
    for seed in (0, 8, 59):
        chosen = _start(m, sch, 10, seed)
        res = operator_polish(
            {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
            max_swaps=40, seed=seed)
        assert not res["max_swaps_binding"]          # converged, not truncated
        again = operator_polish(
            {"chosen": res["chosen"].copy()}, plz_keys, pha, hpl, m, sch,
            max_swaps=40, seed=seed + 777)
        assert again["swaps_made"] == 0
        assert again["opcost_after"] == pytest.approx(
            res["opcost_after"], rel=1e-9, abs=1e-6)


def test_a_cheaper_peak_beats_a_cheaper_route():
    """The economic content of the rule, on a hand-checked move.

    Take the polish's own best move at a cell and verify by explicit
    recomputation that its accepted delta is
    ``d(variable) + W * d(peak) + d(penalty)`` — i.e. that a vehicle-day
    below the peak is charged its variable part ONLY, and the weekly fixed
    bill follows the peak.
    """
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.5, n_cells=10, n_hubs=2)
    chosen = _start(m, sch, 10, 0)
    res = operator_polish(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        max_swaps=1, seed=0)
    assert res["swaps_made"] == 1

    b0 = operator_cost_breakdown(chosen, plz_keys, pha, hpl, m, sch)
    b1 = operator_cost_breakdown(res["chosen"], plz_keys, pha, hpl, m, sch)
    d_var = b1["variable_cost"] - b0["variable_cost"]
    d_peak = b1["sum_hub_peak"] - b0["sum_hub_peak"]
    assert res["accepted_deltas"][0] == pytest.approx(
        d_var + WEEK_FIXED_COST_EUR * d_peak, rel=1e-9, abs=1e-6)
    # the routing cost alone need not fall — only the operator's bill must
    assert b1["opcost"] < b0["opcost"]


def test_a_cheap_peak_lets_the_variable_term_win():
    """The symmetric branch of the rule: when the weekly fixed bill is cheap,
    a move that RAISES the hub peak is correctly accepted, provided the
    variable saving covers ``W * Delta peak``.

    Guards against a "peak must never rise" reading of the objective creeping
    into the accept gate. ``week_fixed=1.0`` (instead of 1 134.90) prices the
    peak, just barely.
    """
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.5, n_cells=10, n_hubs=2)
    chosen = _start(m, sch, 10, 2)
    res = operator_polish(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        max_swaps=40, seed=2, week_fixed=1.0)

    assert res["swaps_made"] > 0
    assert res["sum_hub_peak_after"] > res["sum_hub_peak_before"]   # rose
    assert all(dlt < 0.0 for dlt in res["accepted_deltas"])         # still legal
    assert res["opcost_after"] < res["opcost_before"]
    # and the bought peak really was paid for out of the variable term
    assert res["variable_after"] < res["variable_before"]
    b = operator_cost_breakdown(res["chosen"], plz_keys, pha, hpl, m, sch,
                                week_fixed=1.0)
    assert res["opcost_after"] == pytest.approx(b["opcost"], rel=1e-9, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Best-of-two start — the range state is a candidate, so the old heuristic
# can no longer win in the operator currency (review fix, 2026-08-26)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [0, 4, 9, 16])
def test_best_of_two_is_never_worse_than_the_range_heuristic(seed):
    """G-6e-4 by construction.

    ``W`` dwarfs the variable cost a single cell can shift, so the operator
    objective is flat in plateaus and a strict single-cell descent from
    stage 1 can stall in a basin the range balancer walks straight past. This
    fixture (fs=0.3, ONE hub — the shape of the 10 failing v3 triples)
    reproduces that: at these seeds the pre-6e range heuristic ends up
    CHEAPER in operator cost than a polish started from stage 1 alone.

    Best-of-two must dominate it, because the range state is itself one of
    the two starts and the polish never worsens the state it is given.
    """
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.3, n_cells=10, n_hubs=1)
    chosen = _start(m, sch, 10, seed)

    solo = operator_polish(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        max_swaps=200, seed=seed)
    rng_bal = balance_fleet_per_hub_ml(
        {"chosen": chosen.copy(), "best_cost": 0.0}, plz_keys, pha, hpl, m,
        sch, cost_budget_pct=RANGE_START_BUDGET_PCT, max_swaps=200, seed=seed)
    range_op = operator_cost_breakdown(
        rng_bal["chosen"], plz_keys, pha, hpl, m, sch)["opcost"]

    # the failure this fix is for: stage-1-only polish loses to the heuristic
    assert range_op < solo["opcost_after"] - 1e-9

    best = operator_polish_best_of_two(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        max_swaps=200, seed=seed)
    assert best["opcost_after"] <= range_op + 1e-9          # the guarantee
    assert best["opcost_after"] <= solo["opcost_after"] + 1e-9
    assert best["stage2_start_winner"] == "range"
    assert best["opcost_range_start"] == pytest.approx(range_op, rel=1e-9)


def test_best_of_two_reports_both_branches_and_anchors_before_at_stage_one():
    """Whichever start wins, ``*_before`` must still describe STAGE 1 — the
    grid gates ``initial_total_cost`` as the stage-1 routing anchor and the
    savings tables read ``before -> after`` as "stage 1 -> final"."""
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.3, n_cells=10, n_hubs=1)
    chosen = _start(m, sch, 10, 0)
    solo = operator_polish(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        max_swaps=200, seed=0)
    best = operator_polish_best_of_two(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        max_swaps=200, seed=0)

    assert best["stage2_start_winner"] in ("stage1", "range")
    assert best["opcost_after"] == pytest.approx(
        min(best["opcost_from_stage1"], best["opcost_from_range"]),
        rel=1e-12)
    assert best["opcost_from_stage1"] == pytest.approx(
        solo["opcost_after"], rel=1e-12)
    # every "before" is the stage-1 anchor, not the winning branch's start
    b0 = operator_cost_breakdown(chosen, plz_keys, pha, hpl, m, sch)
    for key, want in (("opcost_before", "opcost"),
                      ("variable_before", "variable_cost"),
                      ("sum_hub_peak_before", "sum_hub_peak"),
                      ("vehicle_days_before", "vehicle_days"),
                      ("initial_total_cost", "routing_cost")):
        assert best[key] == pytest.approx(b0[want], rel=1e-9, abs=1e-6), key
    # ... and the after fields describe the state actually returned
    b1 = operator_cost_breakdown(best["chosen"], plz_keys, pha, hpl, m, sch)
    assert best["opcost_after"] == pytest.approx(b1["opcost"], rel=1e-9,
                                                 abs=1e-6)
    assert best["cost"] == pytest.approx(b1["routing_cost"], rel=1e-9,
                                         abs=1e-6)


def test_best_of_two_keeps_the_stage_one_start_when_it_is_already_better():
    """The second start is insurance, not a preference: when the stage-1
    branch is at least as good it must be the one returned (ties included)."""
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.5, n_cells=10, n_hubs=2)
    seen_stage1 = False
    for seed in (0, 3, 8, 17, 59):
        chosen = _start(m, sch, 10, seed)
        best = operator_polish_best_of_two(
            {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
            max_swaps=200, seed=seed)
        assert best["opcost_after"] <= best["opcost_range_start"] + 1e-9
        if best["stage2_start_winner"] == "stage1":
            seen_stage1 = True
            assert best["opcost_from_stage1"] <= best["opcost_from_range"]
    assert seen_stage1, "fixture no longer exercises the stage-1 branch"


def test_best_of_two_respects_preserve_frequency_on_both_branches():
    """The range balancer runs with the same frequency pin as the polish, so
    no branch can smuggle a frequency change into the result — and the OpCost
    guarantee holds under the pin too, which is the wiring grid v4 shipped and
    the third candidate start of the Task-6f wrapper."""
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.3, n_cells=10, n_hubs=1)
    chosen = _start(m, sch, 10, 0)
    best = operator_polish_best_of_two(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        max_swaps=200, seed=0, preserve_frequency=True)
    for pi in range(10):
        assert len(sch[int(best["chosen"][pi])]) == len(sch[int(chosen[pi])])
    assert best["opcost_after"] <= best["opcost_range_start"] + 1e-9
    assert best["opcost_after"] == pytest.approx(
        min(best["opcost_from_stage1"], best["opcost_from_range"]), rel=1e-12)


@pytest.mark.parametrize("wrapper", [operator_polish_best_of_two,
                                     operator_polish_best_of_n])
def test_max_swaps_zero_is_a_pure_measurement_through_the_wrappers(wrapper):
    """``max_swaps=0`` is :func:`operator_polish`'s measurement contract and
    must survive the wrappers: no candidate start is BUILT (the range balancer
    carries its own swap budget and would return a different plan), nothing is
    compared, and the input plan is reported unchanged with no bound claim."""
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.5, n_cells=10, n_hubs=2)
    chosen = _start(m, sch, 10, 0)
    pen = _penalty(m, sch, 10, scale=0.25)
    res = wrapper({"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
                  max_swaps=0, seed=0, penalty_mx=pen)

    assert np.array_equal(res["chosen"], chosen)
    assert res["swaps_made"] == 0
    assert res["swaps_range_balancer"] == 0
    assert res["opcost_after"] == res["opcost_before"]
    assert res["cost"] == res["initial_total_cost"]
    assert not res["max_swaps_binding"]
    assert res["stage2_start_winner"] == "stage1"
    assert np.isnan(res["opcost_range_start"])
    b = operator_cost_breakdown(chosen, plz_keys, pha, hpl, m, sch,
                                penalty_mx=pen)
    assert res["opcost_after"] == pytest.approx(b["opcost"], rel=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Task 6f — the frequency-FREE polish and its best-of-THREE start set
#
# Stage 2 may now change a cell's delivery FREQUENCY (theta > 0), not merely
# re-time it. The third candidate start is the frequency-preserving
# best-of-two plan (what grid v4 shipped), which makes
# ``OpCost(v5) <= OpCost(v4)`` hold by construction.
# ─────────────────────────────────────────────────────────────────────────────

def _sizes(sch, chosen):
    return np.array([len(sch[int(si)]) for si in chosen])


@pytest.mark.parametrize("fs,n_hubs,seed", [(0.5, 2, 0), (0.3, 1, 4)])
def test_best_of_n_never_loses_to_the_frequency_preserving_best_of_two(
        fs, n_hubs, seed):
    """G-6f-2. The v4 plan is one of the three starts, and the polish never
    worsens the state it is given, so the free best-of-three can only ever be
    at least as cheap as the frequency-preserving best-of-two it contains."""
    plz_keys, m, sch, hpl, pha = _fixture(fs=fs, n_cells=10, n_hubs=n_hubs)
    chosen = _start(m, sch, 10, seed)
    pen = _penalty(m, sch, 10, scale=0.25)

    v4 = operator_polish_best_of_two(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        max_swaps=200, seed=seed, penalty_mx=pen, preserve_frequency=True)
    v5 = operator_polish_best_of_n(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        max_swaps=200, seed=seed, penalty_mx=pen, preserve_frequency=False)

    assert v5["opcost_after"] <= v4["opcost_after"] + 1e-9
    # the v4 state is recorded as such, on the same caches, so the guarantee
    # is auditable from the returned dict alone
    assert v5["opcost_freqpres_start"] == pytest.approx(
        v4["opcost_after"], rel=1e-9, abs=1e-6)
    # ... and against the range balancer too (the 6e guarantee still holds)
    assert v5["opcost_after"] <= v5["opcost_range_start"] + 1e-9


def test_best_of_n_returns_the_cheapest_of_its_three_candidates():
    """G-6f-2's other half: the returned state IS the minimum over the three
    polished candidates, and ``stage2_start_winner`` names the branch."""
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.5, n_cells=10, n_hubs=2)
    for seed in (0, 3, 8, 17, 59):
        chosen = _start(m, sch, 10, seed)
        res = operator_polish_best_of_n(
            {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
            max_swaps=200, seed=seed)
        ends = {"stage1": res["opcost_from_stage1"],
                "range": res["opcost_from_range"],
                "freqpres": res["opcost_from_freqpres"]}
        assert res["stage2_start_winner"] in BEST_OF_N_STARTS
        assert res["opcost_after"] == pytest.approx(min(ends.values()),
                                                    rel=1e-12)
        assert ends[res["stage2_start_winner"]] == pytest.approx(
            res["opcost_after"], rel=1e-12)
        # black-box restatement of the two guarantees (the wrapper asserts
        # them internally; check them from outside too, so removing the
        # internal assert cannot pass silently)
        assert res["opcost_after"] <= res["opcost_range_start"] + 1e-9
        assert res["opcost_after"] <= res["opcost_freqpres_start"] + 1e-9
        # the returned dict really describes the returned plan
        b1 = operator_cost_breakdown(res["chosen"], plz_keys, pha, hpl, m, sch)
        assert res["opcost_after"] == pytest.approx(b1["opcost"], rel=1e-9,
                                                    abs=1e-6)
        assert res["cost"] == pytest.approx(b1["routing_cost"], rel=1e-9,
                                            abs=1e-6)


def test_best_of_n_anchors_every_before_field_at_stage_one():
    """Whichever branch wins, ``*_before`` describes STAGE 1 — the grid gates
    ``initial_total_cost`` as the stage-1 routing anchor and the two-plan
    tables read ``before -> after`` as "stage 1 -> final"."""
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.5, n_cells=10, n_hubs=2)
    chosen = _start(m, sch, 10, 0)
    pen = _penalty(m, sch, 10, scale=0.25)
    res = operator_polish_best_of_n(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        max_swaps=200, seed=0, penalty_mx=pen)
    b0 = operator_cost_breakdown(chosen, plz_keys, pha, hpl, m, sch,
                                 penalty_mx=pen)
    for key, want in (("opcost_before", "opcost"),
                      ("variable_before", "variable_cost"),
                      ("sum_hub_peak_before", "sum_hub_peak"),
                      ("vehicle_days_before", "vehicle_days"),
                      ("penalty_before", "penalty"),
                      ("initial_total_cost", "routing_cost")):
        assert res[key] == pytest.approx(b0[want], rel=1e-9, abs=1e-6), key


def test_the_free_polish_actually_changes_delivery_frequency():
    """The leak Task 6f lifts: with ``preserve_frequency=False`` stage 2 may
    add or drop delivery days, not merely re-time them. Without this the
    one-cell hubs of the real grid have nothing to rotate."""
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.5, n_cells=10, n_hubs=2)
    changed_free = False
    beat_pinned = False
    for seed in (0, 3, 8, 17, 59):
        chosen = _start(m, sch, 10, seed)
        free = operator_polish(
            {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
            max_swaps=200, seed=seed, preserve_frequency=False)
        pinned = operator_polish(
            {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
            max_swaps=200, seed=seed, preserve_frequency=True)
        assert np.array_equal(_sizes(sch, pinned["chosen"]),
                              _sizes(sch, chosen))
        assert free["opcost_after"] <= free["opcost_before"] + 1e-9
        changed_free |= bool((_sizes(sch, free["chosen"])
                              != _sizes(sch, chosen)).any())
        beat_pinned |= free["opcost_after"] < pinned["opcost_after"] - 1e-9
    assert changed_free, "the free polish never changed a cell's frequency"
    # Not a domination theorem — both are greedy descents and the wider
    # candidate set makes the free one take a DIFFERENT path, so it is not
    # guaranteed to end lower on every instance. What must hold is that
    # freeing the frequency buys something somewhere; that is the whole
    # premise of Task 6f, and the best-of-three wrapper keeps the v4 plan as a
    # candidate start precisely so the per-instance ordering is guaranteed too.
    assert beat_pinned, "freeing the frequency never lowered OpCost"


@pytest.mark.parametrize("use_pen", [False, True])
@pytest.mark.parametrize("fs,seed,n_hubs", [(0.7, 8, 1), (0.5, 0, 2),
                                            (0.5, 59, 3)])
def test_bookkeeping_survives_a_frequency_changing_move(use_pen, fs, seed,
                                                        n_hubs):
    """G-6f-3. A move that changes schedule SIZE moves both pooled terms —
    the express partition on the days the cell stops/starts delivering, and
    the small-delivery pool on ``_pool_affected_days`` — so the caches the
    polish carries forward have strictly more to keep exact than under the
    frequency pin. Every tracked scalar must still equal an independent
    from-scratch rebuild at the final plan."""
    plz_keys, m, sch, hpl, pha = _fixture(fs=fs, n_cells=10, n_hubs=n_hubs)
    chosen = _start(m, sch, 10, seed)
    pen = _penalty(m, sch, 10, scale=0.25) if use_pen else None

    res = operator_polish(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        max_swaps=200, seed=seed, penalty_mx=pen, preserve_frequency=False)
    assert res["swaps_made"] > 0
    assert (_sizes(sch, res["chosen"]) != _sizes(sch, chosen)).any(), (
        "no accepted move changed a cell's schedule SIZE — this "
        "parametrisation no longer exercises the gate")

    b0 = operator_cost_breakdown(chosen, plz_keys, pha, hpl, m, sch,
                                 penalty_mx=pen)
    b1 = operator_cost_breakdown(res["chosen"], plz_keys, pha, hpl, m, sch,
                                 penalty_mx=pen)
    for key, before, after in (
        ("opcost", "opcost_before", "opcost_after"),
        ("variable_cost", "variable_before", "variable_after"),
        ("sum_hub_peak", "sum_hub_peak_before", "sum_hub_peak_after"),
        ("vehicle_days", "vehicle_days_before", "vehicle_days_after"),
        ("penalty", "penalty_before", "penalty_after"),
    ):
        assert res[before] == pytest.approx(b0[key], rel=1e-9, abs=1e-6), before
        assert res[after] == pytest.approx(b1[key], rel=1e-9, abs=1e-6), after
    assert res["initial_total_cost"] == pytest.approx(
        b0["routing_cost"], rel=1e-9, abs=1e-6)
    assert res["cost"] == pytest.approx(b1["routing_cost"], rel=1e-9, abs=1e-6)
    # and the three routing terms individually, so a compensating error in the
    # express/pool split cannot hide inside the total
    assert res["cost"] == pytest.approx(
        b1["dd_cost"] + b1["express_cost"] + b1["pool_cost"],
        rel=1e-9, abs=1e-6)


def test_best_of_n_with_preserve_frequency_pins_every_cell():
    """The wrapper honours the pin on every branch when it is asked for — the
    theta=0 fallback and the ``operator-freqpres`` ablation depend on it."""
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.3, n_cells=10, n_hubs=1)
    chosen = _start(m, sch, 10, 0)
    res = operator_polish_best_of_n(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        max_swaps=200, seed=0, preserve_frequency=True)
    assert np.array_equal(_sizes(sch, res["chosen"]), _sizes(sch, chosen))


@pytest.mark.parametrize("fs,n_hubs,seed", [(0.5, 2, 0), (0.5, 2, 59),
                                            (0.3, 1, 4)])
def test_capping_the_range_start_gives_an_IDENTICAL_plan(fs, n_hubs, seed):
    """The range start's swap budget is a wall-time knob, not a search knob.

    ``balance_fleet_per_hub_ml`` is a deterministic function of its seed, so a
    run with budget N executes exactly the first N iterations of the run with
    budget 5 000. As long as no swap is accepted after iteration N the two end
    states are IDENTICAL — bit-for-bit, not merely close — and so is everything
    downstream of them. That is what makes
    :data:`RANGE_START_MAX_SWAPS` = 250 safe: it is ~10x the largest swap count
    the frequency-free balancer has been observed to accept, while cutting the
    ~5 000 no-op iterations that dominate stage-2 wall time.

    Guarded here on the fixtures and, on the real grid, by the TEXT-identical
    OpCost of the four probes in task-6f-report.md.
    """
    plz_keys, m, sch, hpl, pha = _fixture(fs=fs, n_cells=10, n_hubs=n_hubs)
    chosen = _start(m, sch, 10, seed)
    pen = _penalty(m, sch, 10, scale=0.25)
    kw = dict(max_swaps=200, seed=seed, penalty_mx=pen)

    uncapped = operator_polish_best_of_n(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        range_max_swaps=FLEET_BALANCE_MAX_SWAPS, **kw)
    capped = operator_polish_best_of_n(
        {"chosen": chosen.copy()}, plz_keys, pha, hpl, m, sch,
        range_max_swaps=RANGE_START_MAX_SWAPS, **kw)

    assert capped["swaps_range_balancer"] == uncapped["swaps_range_balancer"]
    assert np.array_equal(capped["chosen"], uncapped["chosen"])
    assert capped["opcost_after"] == uncapped["opcost_after"]      # exact ==
    assert capped["opcost_from_range"] == uncapped["opcost_from_range"]
    assert capped["opcost_range_start"] == uncapped["opcost_range_start"]
    assert capped["stage2_start_winner"] == uncapped["stage2_start_winner"]
    # ... and the v4 branch is untouched by the cap by construction: its own
    # range balancer keeps the uncapped budget so it stays exactly grid v4's
    # plan, which is what the OpCost(v5) <= OpCost(v4) guarantee is against.
    assert capped["opcost_freqpres_start"] == uncapped["opcost_freqpres_start"]


def test_the_capped_range_start_is_the_default_and_the_ablations_keep_5000():
    """The cap applies to the best-of-N range start ONLY. The ablation
    wrappers must keep ``FLEET_BALANCE_MAX_SWAPS``, or grids v4 / v3 / run 2
    stop reproducing bit-for-bit."""
    import inspect
    assert RANGE_START_MAX_SWAPS == 250
    assert RANGE_START_MAX_SWAPS < FLEET_BALANCE_MAX_SWAPS
    sigs = {fn.__name__: inspect.signature(fn).parameters
            for fn in (operator_polish_best_of_n,
                       operator_polish_best_of_two,
                       balance_fleet_per_hub_ml)}
    assert sigs["operator_polish_best_of_n"]["range_max_swaps"].default == (
        RANGE_START_MAX_SWAPS)
    assert sigs["operator_polish_best_of_two"]["range_max_swaps"].default == (
        FLEET_BALANCE_MAX_SWAPS)
    assert sigs["balance_fleet_per_hub_ml"]["max_swaps"].default == (
        FLEET_BALANCE_MAX_SWAPS)


def test_best_of_n_does_not_mutate_its_inputs():
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.5, n_cells=10, n_hubs=2)
    chosen = _start(m, sch, 10, 0)
    chosen_in = chosen.copy()
    dd_in = m["dd_cost_mx"].copy()
    veh_in = m["veh_3d"].copy()
    operator_polish_best_of_n(
        {"chosen": chosen_in}, plz_keys, pha, hpl, m, sch,
        max_swaps=200, seed=0)
    assert np.array_equal(chosen_in, chosen)
    assert np.array_equal(m["dd_cost_mx"], dd_in)
    assert np.array_equal(m["veh_3d"], veh_in)


# ─────────────────────────────────────────────────────────────────────────────
# G-6f-1 — theta = 0 is a stage-2 NO-OP (the daily baseline the paper is
# measured against). Tested on the runner's own wiring function, since the
# rule is a property of that wiring, not of the polish.
# ─────────────────────────────────────────────────────────────────────────────

_RUNNER = (Path(__file__).resolve().parents[2] / "scripts" / "revision"
           / "61_grid_run_v2.py")


@lru_cache(maxsize=1)
def _runner():
    """Import ``scripts/revision/61_grid_run_v2.py`` by path.

    Its module body disables INFO logging and installs a blanket warnings
    filter (both wanted for a multi-hour grid run, neither wanted in a test
    session) — undone here so importing the runner has no session-wide side
    effects.
    """
    filters = warnings.filters[:]
    spec = importlib.util.spec_from_file_location("grid_run_v2", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    logging.disable(logging.NOTSET)
    warnings.filters[:] = filters
    return mod


def test_theta_zero_stage_two_is_a_no_op_on_the_daily_baseline():
    """G-6f-1. At theta=0 nobody is willing to wait, so ``penalty_mx`` is
    identically zero for every P and a frequency-free polish would face an
    UNPRICED service dimension — it would batch the daily baseline away. The
    runner therefore pins stage 2 to stage 1 there and only measures."""
    mod = _runner()
    plz_keys, m, sch, hpl, pha = _fixture(fs=1.0, n_cells=10, n_hubs=2)
    daily_si = next(i for i, s in enumerate(sch) if len(s) == N_DAYS)
    chosen_s1 = np.full(10, daily_si, dtype=np.int64)
    pen = np.zeros((10, len(sch)))          # theta=0: nobody waits

    bal, note = mod.stage2_plan(
        "operator", 0.0, chosen_s1, plz_keys, pha, hpl, m, sch, pen)

    assert np.array_equal(bal["chosen"], chosen_s1)          # bit-for-bit
    assert bal["swaps_made"] == 0
    assert bal["opcost_after"] == bal["opcost_before"]
    assert bal["cost"] == bal["initial_total_cost"]
    assert not bal["max_swaps_binding"]
    assert "NO-OP" in note
    # measurement only: the reported state is the stage-1 state
    b0 = operator_cost_breakdown(chosen_s1, plz_keys, pha, hpl, m, sch,
                                 penalty_mx=pen)
    assert bal["opcost_after"] == pytest.approx(b0["opcost"], rel=1e-12)
    # the row carries the full best-of-N schema, with the branches that were
    # never built reported as absent rather than fabricated
    assert bal["stage2_start_winner"] == mod.STAGE2_WINNER_NOOP
    assert np.isnan(bal["opcost_from_range"])
    assert np.isnan(bal["opcost_from_freqpres"])
    assert np.isnan(bal["opcost_range_start"])
    assert np.isnan(bal["opcost_freqpres_start"])
    assert bal["swaps_range_balancer"] == 0


def test_theta_positive_stage_two_runs_the_free_best_of_three():
    """The other side of G-6f-1: above theta=0 the wiring hands the triple to
    the frequency-FREE best-of-three, and the wait IS priced there."""
    mod = _runner()
    plz_keys, m, sch, hpl, pha = _fixture(fs=0.5, n_cells=10, n_hubs=2)
    chosen_s1 = _start(m, sch, 10, 0)
    pen = _penalty(m, sch, 10, scale=0.25)

    bal, note = mod.stage2_plan(
        "operator", 0.5, chosen_s1, plz_keys, pha, hpl, m, sch, pen)
    assert bal["stage2_start_winner"] in BEST_OF_N_STARTS
    assert np.isfinite(bal["opcost_from_freqpres"])
    assert bal["opcost_after"] <= bal["opcost_freqpres_start"] + 1e-9
    assert "start=" in note

    # ... and the frequency-preserving ablation is still reachable, unchanged
    fp, _ = mod.stage2_plan(
        "operator-freqpres", 0.5, chosen_s1, plz_keys, pha, hpl, m, sch, pen)
    assert np.array_equal(_sizes(sch, fp["chosen"]), _sizes(sch, chosen_s1))
    assert bal["opcost_after"] <= fp["opcost_after"] + 1e-9


def test_tiny_matrices_smoke():
    """The 2-cell fixture other balancing tests use must run through the new
    objective as well (hand-built matrices, single hub, both theta regimes)."""
    sch = enumerate_valid_schedules()
    hpl = [np.array([0, 1])]
    pha = np.array([0, 0])
    two = next(i for i, s in enumerate(sch) if len(s) == 2)
    for theta_one in (True, False):
        m = tiny_matrices(theta_one=theta_one)
        res = operator_polish(
            {"chosen": np.array([two, two])}, ["11111", "22222"],
            pha, hpl, m, sch, max_swaps=20)
        b = operator_cost_breakdown(res["chosen"], ["11111", "22222"],
                                    pha, hpl, m, sch)
        assert res["opcost_after"] == pytest.approx(b["opcost"], rel=1e-9,
                                                    abs=1e-6)
        assert res["opcost_after"] <= res["opcost_before"] + 1e-9
