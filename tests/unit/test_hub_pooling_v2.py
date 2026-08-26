"""Partition-priced hub pooling: express twin, small-delivery twin, audit.

The last two tests are the spec-G5 optimizer-consistency audit: the
incremental delta bookkeeping inside ``optimize_cd_ml`` and
``balance_fleet_per_hub_ml`` must reproduce a from-scratch recomputation of
the objective at the schedule they report. A failure here is a real finding
about the optimiser, not a broken test.
"""
import numpy as np
import pytest

from _stubs import tiny_matrices
from batch_delivery.config.constants import N_DAYS, VEHICLE_CAPACITY
from batch_delivery.optimization.costs import (
    _hub_express_day_ml,
    _hub_express_vehicles,
    _hub_smallday_pool_ml,
)
from batch_delivery.optimization.schedules import enumerate_valid_schedules


def _setup(theta_one=False):
    m = tiny_matrices(theta_one=theta_one)
    schedules = enumerate_valid_schedules()
    hub_plz_list = [np.array([0, 1])]
    daily_idx = next(i for i, s in enumerate(schedules) if len(s) == 6)
    two_day = next(i for i, s in enumerate(schedules) if len(s) == 2)
    return m, schedules, hub_plz_list, daily_idx, two_day


def test_express_zero_at_theta_one():
    m, sch, hpl, daily, two = _setup(theta_one=True)
    chosen = np.array([two, two])
    cache = {}
    c = _hub_express_day_ml(0, 0, chosen, hpl, sch, m["raw_express"],
                            m["expr_stops"], m, cache, 1.0)
    assert c == 0.0
    assert _hub_express_vehicles(
        0, 0, chosen, hpl, sch, m["raw_express"], m, cache) == 0.0


def test_express_cost_and_vehicles_share_cache():
    m, sch, hpl, daily, two = _setup()
    chosen = np.array([two, two])
    cache = {}
    d = next(dd for dd in range(N_DAYS) if dd not in sch[two])
    c = _hub_express_day_ml(0, d, chosen, hpl, sch, m["raw_express"],
                            m["expr_stops"], m, cache, 1.0)
    v = _hub_express_vehicles(0, d, chosen, hpl, sch, m["raw_express"], m, cache)
    assert c > 0 and v >= 1.0
    assert len(cache) == 1                     # one entry, tuple-valued


def test_vehicles_are_per_partition_ceil():
    m, sch, hpl, daily, two = _setup()
    chosen = np.array([two, two])
    cache = {}
    d = next(dd for dd in range(N_DAYS) if dd not in sch[two])
    v = _hub_express_vehicles(0, d, chosen, hpl, sch, m["raw_express"], m, cache)
    rx = m["raw_express"]
    # cells 0 (small) and 1 (large) -> partition {(0,),(1,)} or {(0,1)}:
    lo = np.ceil((rx[0, d] + rx[1, d]) / VEHICLE_CAPACITY)
    hi = (np.ceil(rx[0, d] / VEHICLE_CAPACITY)
          + np.ceil(rx[1, d] / VEHICLE_CAPACITY))
    assert lo <= v <= hi


def test_express_scale_multiplies_cached_cost():
    m, sch, hpl, daily, two = _setup()
    chosen = np.array([two, two])
    d = next(dd for dd in range(N_DAYS) if dd not in sch[two])
    c1 = _hub_express_day_ml(0, d, chosen, hpl, sch, m["raw_express"],
                             m["expr_stops"], m, {}, 1.0)
    cache = {}
    c2 = _hub_express_day_ml(0, d, chosen, hpl, sch, m["raw_express"],
                             m["expr_stops"], m, cache, 0.5)
    # second call hits the cache and must still scale
    c3 = _hub_express_day_ml(0, d, chosen, hpl, sch, m["raw_express"],
                             m["expr_stops"], m, cache, 0.5)
    assert c2 == pytest.approx(0.5 * c1, rel=1e-12)
    assert c3 == pytest.approx(c2, rel=1e-12)


def test_small_delivery_mask_moves_cost_to_pool():
    m, sch, hpl, daily, two = _setup()
    mask = m["small_delivery_mask"]
    assert mask.shape == m["cost_3d"].shape
    assert np.all(m["cost_3d"][mask] == 0.0)   # pooled twin owns these
    assert np.any(mask)                        # fixture must exercise the rule
    chosen = np.array([daily, daily])
    pc = {}
    d = 0
    c = _hub_smallday_pool_ml(0, d, chosen, hpl, sch, m, pc)
    if mask[0, daily, d] or mask[1, daily, d]:
        assert c > 0.0
    assert len(pc) == 1
    # cache key carries (cell, schedule) pairs of the small delivering cells
    assert _hub_smallday_pool_ml(0, d, chosen, hpl, sch, m, pc) == c
    assert len(pc) == 1


def test_pool_is_zero_when_no_small_delivering_cell():
    m, sch, hpl, daily, two = _setup()
    mask = m["small_delivery_mask"]
    chosen = np.array([daily, daily])
    # cell 1 (350 parcels/day) is never below one vehicle load, so a hub that
    # holds only cell 1 has an empty pool on every day.
    assert not mask[1, daily, :].any()
    big_only = [np.array([1])]
    for d in range(N_DAYS):
        assert _hub_smallday_pool_ml(0, d, chosen, big_only, sch, m, {}) == 0.0


def test_mask_only_marks_delivery_days():
    m, sch, hpl, daily, two = _setup()
    sa = m["sched_active"]
    mask = m["small_delivery_mask"]
    # express (non-delivery) instances are pooled by _hub_express_day_ml,
    # never by the delivery-day twin.
    assert not np.any(mask & ~sa[None, :, :])


# ─────────────────────────────────────────────────────────────────────────────
# spec G5 — optimiser bookkeeping audit
# ─────────────────────────────────────────────────────────────────────────────

def _full_objective(chosen, m, sch, hpl):
    """From-scratch total: separable dd + express + small-delivery pools."""
    sa = m["sched_active"]
    c3 = m["cost_3d"]
    dd_cost = float(sum(
        (c3[z, int(chosen[z]), :] * sa[int(chosen[z])]).sum()
        for z in range(len(chosen))
    ))
    ec, pc = {}, {}
    tot = dd_cost
    for hi in range(len(hpl)):
        for d in range(N_DAYS):
            tot += _hub_express_day_ml(hi, d, chosen, hpl, sch,
                                       m["raw_express"], m["expr_stops"],
                                       m, ec, 1.0)
            tot += _hub_smallday_pool_ml(hi, d, chosen, hpl, sch, m, pc)
    return tot


def test_cd_bookkeeping_equals_full_recompute():
    from batch_delivery.optimization.coordinate_descent import optimize_cd_ml
    m, sch, hpl, daily, two = _setup()
    plz_hub_arr = np.array([0, 0])
    res = optimize_cd_ml(["11111", "22222"], plz_hub_arr, hpl, m, sch,
                         max_rounds=3, n_restarts=2, seed=42,
                         pair_polish=True, pair_polish_rounds=2)
    chosen = np.asarray(res["chosen"])
    reported = float(res["best_cost"])
    assert reported == pytest.approx(
        _full_objective(chosen, m, sch, hpl), rel=1e-9)


def test_balance_ml_bookkeeping_equals_full_recompute():
    from batch_delivery.optimization.balancing import balance_fleet_per_hub_ml
    m, sch, hpl, daily, two = _setup()
    plz_hub_arr = np.array([0, 0])
    start = np.array([two, two])
    sa_result = {
        "chosen": start,
        "best_cost": _full_objective(start, m, sch, hpl),
    }
    res = balance_fleet_per_hub_ml(
        sa_result, ["11111", "22222"], plz_hub_arr, hpl, m, sch,
        cost_budget_pct=50.0, max_swaps=25,
    )
    assert res["initial_total_cost"] == pytest.approx(
        _full_objective(start, m, sch, hpl), rel=1e-9)
    assert float(res["cost"]) == pytest.approx(
        _full_objective(np.asarray(res["chosen"]), m, sch, hpl), rel=1e-9)
