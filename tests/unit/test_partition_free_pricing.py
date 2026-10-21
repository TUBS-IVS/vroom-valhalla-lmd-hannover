"""Task 6b — the partition-free fast paths must equal the partition path.

At ``head=None`` (the base run) ``price_group`` sums PER-MEMBER prices, so the
grouping ``build_partition`` produces cannot influence the price:

* express  -> ``sum(express_cost[z, d])``          (bundle.py:172-173)
* delivery -> ``sum(predict_single(bundle_features((z,), ...)))``  (bundle.py:174-179)

``_hub_express_day_ml`` and ``_hub_smallday_pool_ml`` exploit that and skip
``build_partition`` entirely; the express VEHICLE count still needs it, so it is
computed lazily by ``_hub_express_vehicles`` and folded back into the shared
cache entry.

These tests pin the equivalence against the partition path itself (built here
from ``_express_partition`` + ``price_group``, i.e. exactly the pre-Task-6b miss
path), so a future change to the fallback or to the partition cannot silently
drift the two apart.  Ordering note: the fast path sums the members in cell
order, the partition path group by group, so IEEE-754 associativity — not
semantics — sets the tolerance.
"""
import numpy as np
import pytest

from _stubs import StubPredictor, tiny_matrices
from batch_delivery.config.constants import (
    MIN_TOUR_PARCELS, N_DAYS, VEHICLE_CAPACITY,
)
from batch_delivery.optimization.costs import (
    _express_partition,
    _hub_express_day_ml,
    _hub_express_vehicles,
    _hub_smallday_pool_ml,
)
from batch_delivery.optimization.partition import build_partition
from batch_delivery.optimization.schedules import enumerate_valid_schedules
from batch_delivery.surrogate.bundle import price_group

REL = 1e-12          # summation-order slack, not a semantic tolerance


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _wide_matrices(n: int = 6):
    """A hub whose partition really splits: several MULTI-cell groups.

    ``tiny_matrices`` has two cells, so its partition is one group and the two
    summation orders coincide by accident. Here cells 0-3 stay below
    ``MIN_TOUR_PARCELS`` (they pack together) while 4-5 clear it (own tour),
    which is the case that actually discriminates Sigma-over-cells from
    Sigma-over-groups.
    """
    from batch_delivery.optimization.costs import build_cost_matrices_ml
    rng = np.random.default_rng(7)
    plz_keys = [f"3{i:04d}" for i in range(n)]
    plz_data, coords, hubs = {}, {}, {}
    for i, pc in enumerate(plz_keys):
        plz_data[pc] = {
            "b2c": {d: 60 + 90 * i + 5 * d for d in range(N_DAYS)},
            "b2b": {d: 10 + 2 * i for d in range(N_DAYS)},
            "area_km2": 3.0 + 2.0 * i,
            "hub_dist_km": 4.0 + 1.5 * i,
            "n_stops_per_day": 30.0 + 10.0 * i,
            "total_points": 900.0,
        }
        lon = 9.6 + 0.03 * i + rng.normal(0, 0.002, 5)
        lat = 52.3 + 0.02 * i + rng.normal(0, 0.002, 5)
        coords[pc] = {d: (lon, lat, np.full(5, 2.0 + 0.5 * d))
                      for d in range(N_DAYS)}
        hubs[pc] = (9.73, 52.38)
    m = build_cost_matrices_ml(
        plz_keys, plz_data, enumerate_valid_schedules(), StubPredictor(),
        "DHL", coords, hubs, fast_share_b2c=0.55, fast_share_b2b=0.45)
    return m, [np.arange(n)]


def _chosen_vectors(n_plz: int, schedules: list) -> list[np.ndarray]:
    """Assignments to sweep: daily, a 2-day pattern, and random 'swap' states."""
    daily = next(i for i, s in enumerate(schedules) if len(s) == N_DAYS)
    two = next(i for i, s in enumerate(schedules) if len(s) == 2)
    out = [np.full(n_plz, daily, dtype=np.int64),
           np.full(n_plz, two, dtype=np.int64)]
    rng = np.random.default_rng(11)
    base = np.full(n_plz, daily, dtype=np.int64)
    for _ in range(6):                       # simulated trial moves
        base = base.copy()
        base[int(rng.integers(n_plz))] = int(rng.integers(len(schedules)))
        out.append(base.copy())
    for _ in range(4):
        out.append(rng.integers(0, len(schedules), size=n_plz).astype(np.int64))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reference implementations — the pre-Task-6b miss paths, verbatim
# ─────────────────────────────────────────────────────────────────────────────

def _express_reference(hi, d, chosen, hpl, sch, m):
    """(cost, vehicles) via build_partition + price_group, as before Task 6b."""
    h_ps = hpl[hi]
    nd = ~m["sched_active"][chosen[h_ps], d]
    mask = nd & (m["raw_express"][h_ps, d] > 0)
    if not mask.any():
        return 0.0, 0.0, ()
    contributing = h_ps[mask].tolist()
    parts = _express_partition(contributing, d, m["raw_express"],
                               m["expr_stops"], m)
    cost, veh = 0.0, 0.0
    for g in parts:
        cost += price_group(g, d, m, kind="express", head=None)
        veh += float(np.ceil(sum(np.trunc(m["raw_express"][z, d]) for z in g)
                             / VEHICLE_CAPACITY))
    return cost, veh, parts


def _pool_reference(hi, d, chosen, hpl, sch, m):
    """Pooled small-delivery cost via build_partition + price_group."""
    h_ps = hpl[hi]
    sa = m["sched_active"]
    mask3 = m["small_delivery_mask"]
    small = [int(z) for z in h_ps
             if sa[int(chosen[z]), d] and mask3[z, int(chosen[z]), d]]
    if not small:
        return 0.0, ()
    cd, cs = m["combined_demand"], m["combined_stops"]
    parcels = np.zeros(cd.shape[0])
    stops = np.zeros(cd.shape[0])
    for z in small:
        parcels[z] = cd[z, int(chosen[z]), d]
        stops[z] = max(1.0, cs[z, int(chosen[z]), d])
    parts = build_partition(
        np.array(small), parcels, stops, m["area_arr"], m["hd_arr"],
        m["_cent_lon"], m["_cent_lat"],
        pts_lon={z: m["plz_day_lon"][z][d] for z in small},
        pts_lat={z: m["plz_day_lat"][z][d] for z in small},
    )
    cost = float(sum(
        price_group(g, d, m, kind="delivery", parcels_by_cell=parcels,
                    stops_by_cell=stops, freq=1.0, head=None)
        for g in parts
    ))
    return cost, parts


# ─────────────────────────────────────────────────────────────────────────────
# G-eq1 — cost identity, express twin
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("wide", [False, True])
def test_express_fast_path_equals_partition_path(wide):
    sch = enumerate_valid_schedules()
    if wide:
        m, hpl = _wide_matrices()
    else:
        m, hpl = tiny_matrices(theta_one=False), [np.array([0, 1])]
    n_plz = m["raw_express"].shape[0]
    split = False        # a partition with >1 group AND a multi-cell group
    for chosen in _chosen_vectors(n_plz, sch):
        for hi in range(len(hpl)):
            for d in range(N_DAYS):
                ref_cost, _, parts = _express_reference(hi, d, chosen, hpl, sch, m)
                got = _hub_express_day_ml(
                    hi, d, chosen, hpl, sch, m["raw_express"],
                    m["expr_stops"], m, {}, 1.0)
                assert got == pytest.approx(ref_cost, rel=REL, abs=1e-9)
                split |= len(parts) > 1 and any(len(g) > 1 for g in parts)
    if wide:      # the fixture must actually exercise a non-trivial grouping
        assert split


@pytest.mark.parametrize("wide", [False, True])
def test_pool_fast_path_equals_partition_path(wide):
    sch = enumerate_valid_schedules()
    if wide:
        m, hpl = _wide_matrices()
    else:
        m, hpl = tiny_matrices(theta_one=False), [np.array([0, 1])]
    n_plz = m["raw_express"].shape[0]
    n_priced = 0
    multi_member = False
    for chosen in _chosen_vectors(n_plz, sch):
        for hi in range(len(hpl)):
            for d in range(N_DAYS):
                ref, parts = _pool_reference(hi, d, chosen, hpl, sch, m)
                got = _hub_smallday_pool_ml(hi, d, chosen, hpl, sch, m, {})
                assert got == pytest.approx(ref, rel=REL, abs=1e-9)
                n_priced += ref > 0.0
                multi_member |= any(len(g) > 1 for g in parts)
    assert n_priced          # the sweep must hit non-empty pools
    if wide:
        assert multi_member  # ... and pool at least two cells into one tour


def test_small_delivery_price_matches_price_group_per_member():
    """The precomputed table IS ``price_group(head=None)`` per member."""
    m, _ = _wide_matrices()
    sdp = m["small_delivery_price"]
    mask = m["small_delivery_mask"]
    assert sdp.shape == m["cost_3d"].shape
    assert np.all(sdp[~mask] == 0.0)          # only masked cells are priced
    assert np.any(sdp[mask] > 0.0)
    cd, cs = m["combined_demand"], m["combined_stops"]
    zi, si, di = np.where(mask)
    rng = np.random.default_rng(3)
    for k in rng.choice(len(zi), size=min(12, len(zi)), replace=False):
        z, s, d = int(zi[k]), int(si[k]), int(di[k])
        parcels = np.zeros(cd.shape[0])
        stops = np.zeros(cd.shape[0])
        parcels[z] = cd[z, s, d]
        stops[z] = max(1.0, cs[z, s, d])
        want = price_group((z,), d, m, kind="delivery", parcels_by_cell=parcels,
                           stops_by_cell=stops, freq=1.0, head=None)
        assert sdp[z, s, d] == pytest.approx(want, rel=1e-15, abs=1e-12)


def test_masked_cells_are_below_one_vehicle_load():
    """Sanity on the fixture: the mask is the sub-vehicle-load delivery rule."""
    m, _ = _wide_matrices()
    mask = m["small_delivery_mask"]
    assert np.all(m["combined_demand"][mask] < MIN_TOUR_PARCELS)


# ─────────────────────────────────────────────────────────────────────────────
# G-eq2 — the lazily upgraded vehicle count equals the eager one
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("wide", [False, True])
def test_lazy_vehicles_equal_eager_partition_vehicles(wide):
    sch = enumerate_valid_schedules()
    if wide:
        m, hpl = _wide_matrices()
    else:
        m, hpl = tiny_matrices(theta_one=False), [np.array([0, 1])]
    n_plz = m["raw_express"].shape[0]
    for chosen in _chosen_vectors(n_plz, sch):
        for hi in range(len(hpl)):
            for d in range(N_DAYS):
                _, ref_veh, _ = _express_reference(hi, d, chosen, hpl, sch, m)
                cache: dict = {}
                # cost first (fast path -> vehicles slot deferred), then fleet
                _hub_express_day_ml(hi, d, chosen, hpl, sch, m["raw_express"],
                                    m["expr_stops"], m, cache, 1.0)
                got = _hub_express_vehicles(hi, d, chosen, hpl, sch,
                                            m["raw_express"], m, cache)
                assert got == ref_veh
                # and the fleet-first order must give the same answer
                got2 = _hub_express_vehicles(hi, d, chosen, hpl, sch,
                                             m["raw_express"], m, {})
                assert got2 == ref_veh


def test_vehicle_upgrade_keeps_one_cache_entry_and_its_cost():
    """The lazy upgrade rewrites the slot in place — no second entry, same cost."""
    m, hpl = _wide_matrices()
    sch = enumerate_valid_schedules()
    two = next(i for i, s in enumerate(sch) if len(s) == 2)
    chosen = np.full(m["raw_express"].shape[0], two, dtype=np.int64)
    d = next(dd for dd in range(N_DAYS) if dd not in sch[two])
    cache: dict = {}
    c = _hub_express_day_ml(0, d, chosen, hpl, sch, m["raw_express"],
                            m["expr_stops"], m, cache, 1.0)
    assert len(cache) == 1
    key = next(iter(cache))
    assert cache[key][1] is None                    # partition deferred
    v = _hub_express_vehicles(0, d, chosen, hpl, sch, m["raw_express"], m, cache)
    assert len(cache) == 1                          # upgraded in place
    assert cache[key] == (c, v) and v >= 1.0
    # cost is still served from the same entry, and still scales
    assert _hub_express_day_ml(0, d, chosen, hpl, sch, m["raw_express"],
                               m["expr_stops"], m, cache, 0.5) == 0.5 * c


def test_head_present_prices_through_the_partition():
    """With a head installed the fast paths must NOT fire (Gate U regime)."""
    class _Head:
        def predict_single(self, x25):        # marker value per group
            return 1000.0 + float(x25[0])

    m, hpl = _wide_matrices()
    sch = enumerate_valid_schedules()
    two = next(i for i, s in enumerate(sch) if len(s) == 2)
    chosen = np.full(m["raw_express"].shape[0], two, dtype=np.int64)
    d = next(dd for dd in range(N_DAYS) if dd not in sch[two])
    m_head = dict(m)
    m_head["bundle_head"] = _Head()
    parts = _express_partition(
        [int(z) for z in hpl[0] if m["raw_express"][z, d] > 0], d,
        m["raw_express"], m["expr_stops"], m_head)
    want = float(sum(price_group(g, d, m_head, kind="express",
                                 head=m_head["bundle_head"]) for g in parts))
    got = _hub_express_day_ml(0, d, chosen, hpl, sch, m["raw_express"],
                              m["expr_stops"], m_head, {}, 1.0)
    assert got == pytest.approx(want, rel=1e-12)
    # ... and the head path still fills the vehicle slot eagerly
    cache: dict = {}
    _hub_express_day_ml(0, d, chosen, hpl, sch, m["raw_express"],
                        m["expr_stops"], m_head, cache, 1.0)
    assert next(iter(cache.values()))[1] is not None
