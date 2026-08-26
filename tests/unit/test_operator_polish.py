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
import numpy as np
import pytest
from _stubs import StubPredictor, tiny_matrices

from batch_delivery.config.constants import FIXED_COST_EUR, N_DAYS
from batch_delivery.optimization.balancing import (
    WEEK_FIXED_COST_EUR,
    _daily_fleet_per_hub,
    operator_cost_breakdown,
    operator_polish,
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
        assert res["sum_hub_peak_after"] <= res["sum_hub_peak_before"]
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
