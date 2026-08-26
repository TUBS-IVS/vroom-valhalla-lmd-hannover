"""Fleet objective sees every vehicle (D2): express contributes to fleet.

``_daily_fleet_per_hub`` previously counted only the per-cell delivery-day
``veh_3d`` slice, so a hub's express partition vehicles (rev1 realistic
tours) were invisible to fleet balancing. An optional ``express_veh_fn``
adds them in, sourced from the same partition/cache path
``_hub_express_day_ml`` prices with (``_hub_express_vehicles``, Task 4).
"""
import numpy as np
import pytest
from _stubs import StubPredictor, tiny_matrices

from batch_delivery.optimization.balancing import (
    _daily_fleet_per_hub,
    _fleet_imbalance,
    balance_fleet_per_hub_ml,
    system_smooth_pass,
)
from batch_delivery.optimization.costs import (
    _hub_express_vehicles,
    build_cost_matrices_ml,
)
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
    """The express partition's vehicles must be VISIBLE in the profile.

    Reference is the express-BLIND profile (same delivery-day masking, zero
    pooled term) — not the unmasked ``veh_3d`` sum, which since the §0
    double-count fix is no longer a meaningful baseline: it already contains
    a per-cell copy of exactly the express vehicles being added here.
    """
    m = tiny_matrices(theta_one=False)
    sch = enumerate_valid_schedules()
    hpl = [np.array([0, 1])]
    two = next(i for i, s in enumerate(sch) if len(s) == 2)
    chosen = np.array([two, two])
    pha = np.array([0, 0])
    cache = {}
    fn = lambda hi, d, ch: _hub_express_vehicles(
        hi, d, ch, hpl, sch, m["raw_express"], m, cache)
    blind = _daily_fleet_per_hub(chosen, pha, hpl, m["veh_3d"], sch,
                                 express_veh_fn=lambda hi, d, ch: 0.0)
    withx = _daily_fleet_per_hub(chosen, pha, hpl, m["veh_3d"], sch,
                                 express_veh_fn=fn)
    d_off = next(dd for dd in range(6) if dd not in sch[two])
    assert blind[0, d_off] == 0.0               # nobody delivers that day
    assert withx[0, d_off] > blind[0, d_off]    # invisible fifth appears


def test_express_vehicles_are_not_counted_twice():
    """Task 6c §0 regression — the double count that killed the base grid.

    ``veh_3d`` is written for EVERY ACTIVE instance, and on a NON-delivery day
    a cell's ``combined_demand`` is its express residual (> 0 whenever
    theta < 1). So the per-cell slice already carries >= 1 vehicle for every
    non-delivering express cell; adding the hub's POOLED express vehicles on
    top counted each express vehicle twice (measured on DPD 0.5/0.1: peak
    +58 %, hub spread 9 vs 47 true).

    With a pooled-vehicle closure supplied, the per-cell term must be masked
    to DELIVERY days -- the reporting fix of
    ``scripts/revision/50_recompute_fleet_wait_fixed.py:138-151``, now applied
    to the objective itself.
    """
    m = tiny_matrices(theta_one=False)
    sch = enumerate_valid_schedules()
    hpl = [np.array([0, 1])]
    pha = np.array([0, 0])
    daily = next(i for i, s in enumerate(sch) if len(s) == 6)
    two = next(i for i, s in enumerate(sch) if len(s) == 2)
    chosen = np.array([two, daily])          # cell 0 does not deliver every day
    veh = m["veh_3d"]
    cache: dict = {}
    fn = lambda hi, d, ch: _hub_express_vehicles(
        hi, d, ch, hpl, sch, m["raw_express"], m, cache)

    fleet = _daily_fleet_per_hub(chosen, pha, hpl, veh, sch,
                                 express_veh_fn=fn)
    sa = m["sched_active"]
    h_ps = hpl[0]
    for d in range(6):
        deliv = sa[chosen[h_ps], d]
        want = float(veh[h_ps[deliv], chosen[h_ps[deliv]], d].sum()) + fn(0, d, chosen)
        double = float(veh[h_ps, chosen[h_ps], d].sum()) + fn(0, d, chosen)
        assert fleet[0, d] == pytest.approx(want)
        if not deliv.all():                  # the day the bug showed up on
            assert fleet[0, d] < double


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
    # Express-BLIND reference: same delivery-day masking, zero pooled term
    # (see test_theta_lt1_profile_includes_express_vehicles on why the
    # unmasked veh_3d sum stopped being a usable baseline in §0).
    blind_fleet = _daily_fleet_per_hub(
        chosen, pha, hpl, m["veh_3d"], sch,
        express_veh_fn=lambda hi, d, ch: 0.0)

    assert res["imbalance_before"] == pytest.approx(
        _fleet_imbalance(expected_fleet))
    assert res["imbalance_before"] != pytest.approx(
        _fleet_imbalance(blind_fleet))     # D2: express now visible


# ─────────────────────────────────────────────────────────────────────────────
# I1/I2 (review round 2): the accept/reject GATE, not just the accepted
# state, must be express-aware -- otherwise a swap can be accepted that
# LOOKS improving under the veh_3d-only estimate but actually makes the true
# (express-aware) fleet profile worse. Fixture mirrors the reviewer's
# probe_gate.py exactly (10 cells, single hub, mix of big/small cells so the
# express partition re-forms), which is how these two counter-examples were
# found: fs=0.7/seed=8 for balance_fleet_per_hub_ml, fs=0.5/seed=59 for
# system_smooth_pass.
# ─────────────────────────────────────────────────────────────────────────────

def _bal_fixture(fs: float, n_cells: int = 10):
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
    hpl = [np.arange(n_cells)]
    pha = np.zeros(n_cells, dtype=int)
    return plz_keys, m, sch, hpl, pha


def test_balance_ml_never_increases_imbalance():
    """I2 direction regression. Pre-I1-fix counter-example on this exact
    fixture (fs=0.7, seed=8): the swap-acceptance gate ranked candidates by
    a veh_3d-only fleet estimate, blind to the express vehicles a swap
    silently moves (one-signed: a cell leaving delivery on day A JOINS
    day-A's express pool). Reported imbalance went 3.0 -> 4.0 even though
    swaps were only ever accepted because they looked improving.
    """
    n_cells = 10
    plz_keys, m, sch, hpl, pha = _bal_fixture(fs=0.7, n_cells=n_cells)
    rng = np.random.default_rng(1000 + 8)
    chosen = rng.integers(0, len(sch), size=n_cells)
    sa_result = {
        "chosen": chosen.copy(),
        "best_cost": float(m["dd_cost_mx"][np.arange(n_cells), chosen].sum()),
    }
    res = balance_fleet_per_hub_ml(
        sa_result, plz_keys, pha, hpl, m, sch,
        cost_budget_pct=100.0, max_swaps=40, seed=8)
    assert res["swaps_made"] > 0                 # exercises the swap path
    assert res["imbalance_after"] <= res["imbalance_before"] + 1e-9


def test_system_smooth_never_increases_spread():
    """I2 direction regression. Pre-I1-fix counter-example on this exact
    fixture (fs=0.5, seed=59): the cheap system-spread pre-check doubled as
    the final accept gate, blind to express-vehicle movement, so it accepted
    2 swaps that only LOOKED improving; reported spread went 6.0 -> 8.0.

    Post-fix, the true-profile-exact gate correctly finds that neither of
    those 2 candidates is actually improving and accepts nothing here
    (``swaps_made == 0``) -- that is the corrected behaviour, not a weaker
    test: see ``test_system_smooth_accepts_genuinely_improving_swaps`` below
    for a seed where the gate does still accept true-improving swaps.
    """
    n_cells = 10
    plz_keys, m, sch, hpl, pha = _bal_fixture(fs=0.5, n_cells=n_cells)
    rng = np.random.default_rng(1000 + 59)
    chosen = rng.integers(0, len(sch), size=n_cells)
    res = system_smooth_pass(
        chosen.copy(), plz_keys, pha, hpl, m, sch,
        cost_budget_pct=100.0, max_iterations=60, seed=59)
    assert res["system_spread_after"] <= res["system_spread_before"] + 1e-9


def test_system_smooth_accepts_genuinely_improving_swaps():
    """M1 coverage: system_smooth_pass's swap-ACCEPTANCE path (not just the
    reject path exercised above) must still run through committed tests.
    fs=0.5/seed=0 on the same fixture has true-improving swaps both before
    and after the I1 fix (3 swaps, spread 10.0 -> 2.0) -- the exact-gate fix
    does not make the pass unable to ever accept a swap, only unable to
    accept a WRONG one.
    """
    n_cells = 10
    plz_keys, m, sch, hpl, pha = _bal_fixture(fs=0.5, n_cells=n_cells)
    rng = np.random.default_rng(1000 + 0)
    chosen = rng.integers(0, len(sch), size=n_cells)
    res = system_smooth_pass(
        chosen.copy(), plz_keys, pha, hpl, m, sch,
        cost_budget_pct=100.0, max_iterations=60, seed=0)
    assert res["swaps_made"] > 0                  # M1: accept path covered
    assert res["system_spread_after"] <= res["system_spread_before"] + 1e-9
