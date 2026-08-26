"""Fleet objective sees every vehicle (D2): express contributes to fleet.

``_daily_fleet_per_hub`` previously counted only the per-cell delivery-day
``veh_3d`` slice, so a hub's express partition vehicles (rev1 realistic
tours) were invisible to fleet balancing. An optional ``express_veh_fn``
adds them in, sourced from the same partition/cache path
``_hub_express_day_ml`` prices with (``_hub_express_vehicles``, Task 4).
"""
import numpy as np
import pytest
from _stubs import tiny_matrices

from batch_delivery.optimization.balancing import (
    _daily_fleet_per_hub,
    _fleet_imbalance,
    balance_fleet_per_hub_ml,
)
from batch_delivery.optimization.costs import _hub_express_vehicles
from batch_delivery.optimization.schedules import enumerate_valid_schedules


def test_theta1_profile_unchanged_with_express_fn():
    m = tiny_matrices(theta_one=True)
    sch = enumerate_valid_schedules()
    hpl = [np.array([0, 1])]
    two = next(i for i, s in enumerate(sch) if len(s) == 2)
    chosen = np.array([two, two])
    pha = np.array([0, 0])
    cache = {}
    fn = lambda hi, d, ch: _hub_express_vehicles(
        hi, d, ch, hpl, sch, m["raw_express"], m, cache)
    base = _daily_fleet_per_hub(chosen, pha, hpl, m["veh_3d"], sch)
    withx = _daily_fleet_per_hub(chosen, pha, hpl, m["veh_3d"], sch,
                                 express_veh_fn=fn)
    assert np.array_equal(base, withx)          # G1: no express at theta=1


def test_theta_lt1_profile_includes_express_vehicles():
    m = tiny_matrices(theta_one=False)
    sch = enumerate_valid_schedules()
    hpl = [np.array([0, 1])]
    two = next(i for i, s in enumerate(sch) if len(s) == 2)
    chosen = np.array([two, two])
    pha = np.array([0, 0])
    cache = {}
    fn = lambda hi, d, ch: _hub_express_vehicles(
        hi, d, ch, hpl, sch, m["raw_express"], m, cache)
    base = _daily_fleet_per_hub(chosen, pha, hpl, m["veh_3d"], sch)
    withx = _daily_fleet_per_hub(chosen, pha, hpl, m["veh_3d"], sch,
                                 express_veh_fn=fn)
    d_off = next(dd for dd in range(6) if dd not in sch[two])
    assert withx[0, d_off] > base[0, d_off]     # invisible fifth appears


def test_balance_ml_imbalance_before_sees_express_vehicles():
    """The swap-loop wiring, not just the leaf helper: ``imbalance_before``
    reported by ``balance_fleet_per_hub_ml`` must match an express-aware
    recompute of the SAME ``chosen`` state, and must differ from the old
    delivery-day-only accounting (the fleet no longer has an invisible
    partition of express vehicles).
    """
    m = tiny_matrices(theta_one=False)
    sch = enumerate_valid_schedules()
    hpl = [np.array([0, 1])]
    pha = np.array([0, 0])
    daily = next(i for i, s in enumerate(sch) if len(s) == 6)
    two = next(i for i, s in enumerate(sch) if len(s) == 2)
    chosen = np.array([two, daily])       # cell 0 has express off-days
    sa_result = {"chosen": chosen}

    res = balance_fleet_per_hub_ml(
        sa_result, ["11111", "22222"], pha, hpl, m, sch, max_swaps=0)
    assert res["swaps_made"] == 0         # isolates the initial-fleet build

    fresh_cache: dict = {}
    fn = lambda hi, d, ch: _hub_express_vehicles(
        hi, d, ch, hpl, sch, m["raw_express"], m, fresh_cache)
    expected_fleet = _daily_fleet_per_hub(
        chosen, pha, hpl, m["veh_3d"], sch, express_veh_fn=fn)
    legacy_fleet = _daily_fleet_per_hub(chosen, pha, hpl, m["veh_3d"], sch)

    assert res["imbalance_before"] == pytest.approx(
        _fleet_imbalance(expected_fleet))
    assert res["imbalance_before"] != pytest.approx(
        _fleet_imbalance(legacy_fleet))    # D2: express now visible
