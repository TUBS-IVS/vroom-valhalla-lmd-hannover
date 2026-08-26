"""Task 6d — the three memo layers are EXACT caches, never approximations.

The head regime (``matrices["bundle_head"]`` installed by Gate U) prices every
tour through ``build_partition`` + ``price_group``, which made the θ=0.5 DPD
triple cost 575 s against a ~60 s bar (6c report §G-6c-2). Three memo layers
make it affordable:

* **L1** ``price_group``      — group price by (members, day, kind, demand, freq, head)
* **L2** ``_hull_km2``        — hull km² by the point-set actually concatenated
* **L3** ``build_partition``  — the grouping, by (day, cell state), in the two
  ``costs.py`` callers

Every one of them is a cache of a deterministic function of its key, so the
memoised answer must be the SAME FLOAT — not approximately, bit for bit — as
the un-memoised one. These tests pin exactly that, plus the two ways a cache
can silently lie: serving a value across a changed head (L1) and surviving into
a different matrices dict (all three).
"""
import numpy as np
import pytest

from _stubs import StubPredictor
from batch_delivery.config.constants import N_DAYS
from batch_delivery.optimization.costs import (
    _express_partition,
    _hub_express_day_ml,
    _hub_smallday_pool_ml,
    _memo_stats,
    _smallday_members,
    _smallday_partition,
    MEMO_KEYS,
)
from batch_delivery.optimization.partition import _hull_km2, build_partition
from batch_delivery.optimization.schedules import enumerate_valid_schedules
from batch_delivery.surrogate.bundle import _daganzo_scalar, bundle_features, price_group


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

class Head:
    """A head whose price genuinely depends on the whole group (not a Sigma).

    Mirrors ``61_grid_run_v2.DummyHead``: the Daganzo backbone of the 25-row,
    which is super-additive in the group's members, so a memo that mixed two
    groups up would be caught by value, not only by counter.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)

    def predict_single(self, x25: np.ndarray) -> float:
        from batch_delivery.features import ALL_COLS
        i = {c: k for k, c in enumerate(ALL_COLS)}
        return self.alpha * _daganzo_scalar(
            n_parcels=x25[i["n_parcels"]], n_stops=x25[i["n_stops"]],
            area_km2=x25[i["area_km2"]], hub_dist_km=x25[i["hub_dist_km"]])


def _matrices(n: int = 6, fs: float = 0.55):
    """A hub whose partition really splits: several MULTI-cell groups."""
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
        "DHL", coords, hubs, fast_share_b2c=fs, fast_share_b2b=fs)
    return m, [np.arange(n)]


def _cold(m: dict) -> dict:
    """A view of *m* with every memo layer stripped — the pre-6d behaviour.

    A single call cannot hit a memo it just created, so a FRESH cold view per
    call is an exact un-memoised reference implementation.
    """
    return {k: v for k, v in m.items() if k not in MEMO_KEYS}


def _chosen_vectors(n_plz: int, schedules: list) -> list[np.ndarray]:
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


def _pool_state(m, hpl, chosen, d):
    """``(small, parcels, stops)`` for the hub-day pool, built from scratch."""
    small, _ = _smallday_members(0, d, chosen, hpl, m)
    if not small:
        return None
    cd, cs = m["combined_demand"], m["combined_stops"]
    parcels = np.zeros(cd.shape[0])
    stops = np.zeros(cd.shape[0])
    for z in small:
        parcels[z] = cd[z, int(chosen[z]), d]
        stops[z] = max(1.0, cs[z, int(chosen[z]), d])
    return small, parcels, stops


# ─────────────────────────────────────────────────────────────────────────────
# G-6d-3 (a) — L1 group-price memo
# ─────────────────────────────────────────────────────────────────────────────

def test_l1_express_memo_equals_unmemoised_bit_for_bit():
    m, hpl = _matrices()
    head = Head(1.3)
    sch = enumerate_valid_schedules()
    seen_multi = False
    for chosen in _chosen_vectors(m["raw_express"].shape[0], sch):
        for d in range(N_DAYS):
            contrib = [int(z) for z in hpl[0]
                       if not m["sched_active"][chosen[z], d]
                       and m["raw_express"][z, d] > 0]
            if len(contrib) < 2:
                continue
            for g in _express_partition(contrib, d, m["raw_express"],
                                        m["expr_stops"], m):
                if len(g) == 1:
                    # Pre-6d shortcut (bundle.py:169-170): a LONE express cell
                    # is served from the §9b table whatever the head — the memo
                    # sits behind that early return and must not disturb it.
                    assert price_group(g, d, m, kind="express", head=head) == \
                        float(m["express_cost"][g[0], d])
                    continue
                want = float(head.predict_single(bundle_features(
                    g, d, _cold(m), kind="express")))
                got = price_group(g, d, m, kind="express", head=head)
                assert got == want          # bit-for-bit, not approx
                # ... and again, now served from the memo
                assert price_group(g, d, m, kind="express", head=head) == want
                seen_multi = True
    assert seen_multi, "fixture never produced a multi-cell express tour"
    assert _memo_stats(m)["price_hit"] > 0


def test_l1_delivery_memo_equals_unmemoised_bit_for_bit():
    m, hpl = _matrices()
    head = Head(1.3)
    sch = enumerate_valid_schedules()
    n_priced = 0
    for chosen in _chosen_vectors(m["raw_express"].shape[0], sch):
        for d in range(N_DAYS):
            st = _pool_state(m, hpl, chosen, d)
            if st is None:
                continue
            small, parcels, stops = st
            parts, _, _ = _smallday_partition(0, d, chosen, small, m)
            for g in parts:
                want = float(head.predict_single(bundle_features(
                    g, d, _cold(m), kind="delivery", parcels_by_cell=parcels,
                    stops_by_cell=stops, freq=1.0)))
                got = price_group(g, d, m, kind="delivery",
                                  parcels_by_cell=parcels, stops_by_cell=stops,
                                  freq=1.0, head=head)
                assert got == want
                assert price_group(g, d, m, kind="delivery",
                                   parcels_by_cell=parcels,
                                   stops_by_cell=stops, freq=1.0,
                                   head=head) == want
                n_priced += 1
    assert n_priced, "fixture never pooled a small-delivery instance"


def test_l1_head_none_memo_equals_unmemoised():
    """The Sigma-fallback branch is memoised on the same key, same value."""
    m, hpl = _matrices()
    d = 0
    g = tuple(int(z) for z in hpl[0][:3])
    want = float(sum(m["express_cost"][z, d] for z in g))
    assert price_group(g, d, m, kind="express", head=None) == want
    assert price_group(g, d, m, kind="express", head=None) == want


def test_l1_misses_when_the_head_object_changes():
    """A swapped head must never be served a price the previous head made."""
    m, hpl = _matrices()
    a, b = Head(1.0), Head(2.0)        # same class, different prices
    d = 0
    g = tuple(int(z) for z in hpl[0][:3])
    va = price_group(g, d, m, kind="express", head=a)
    vb = price_group(g, d, m, kind="express", head=b)
    assert vb == pytest.approx(2.0 * va, rel=1e-12) and vb != va
    # both entries coexist; neither head can serve the other's price
    assert price_group(g, d, m, kind="express", head=a) == va
    assert price_group(g, d, m, kind="express", head=b) == vb
    # and head=None is a third, distinct key
    vn = price_group(g, d, m, kind="express", head=None)
    assert vn not in (va, vb)
    keys = [k for k in m["_group_price_memo"] if k[0] == g and k[1] == d]
    assert len(keys) == 3


def test_l1_head_identity_is_pinned_against_id_reuse():
    """The memo holds a strong ref per head, so CPython cannot recycle its id."""
    m, hpl = _matrices()
    g = tuple(int(z) for z in hpl[0][:3])
    ids = set()
    for alpha in (1.0, 2.0, 3.0, 4.0):
        h = Head(alpha)                # goes out of scope each iteration
        ids.add(id(h))
        assert price_group(g, 0, m, kind="express", head=h) == pytest.approx(
            alpha * price_group(g, 0, m, kind="express", head=Head(1.0)),
            rel=1e-12)
    assert len(ids) == 4, "a head id was recycled — the memo did not pin it"


def test_l1_memo_distinguishes_the_demand_signature():
    """Same members, same day, different parcels -> different price."""
    m, hpl = _matrices()
    head = Head(1.0)
    g = tuple(int(z) for z in hpl[0][:3])
    n = m["raw_express"].shape[0]
    p1, s1 = np.zeros(n), np.full(n, 20.0)
    p2, s2 = np.zeros(n), np.full(n, 20.0)
    for z in g:
        p1[z], p2[z] = 40.0, 90.0
    v1 = price_group(g, 0, m, kind="delivery", parcels_by_cell=p1,
                     stops_by_cell=s1, freq=1.0, head=head)
    v2 = price_group(g, 0, m, kind="delivery", parcels_by_cell=p2,
                     stops_by_cell=s2, freq=1.0, head=head)
    assert v1 != v2
    assert price_group(g, 0, m, kind="delivery", parcels_by_cell=p1,
                       stops_by_cell=s1, freq=1.0, head=head) == v1


def test_l1_memo_distinguishes_freq_and_kind():
    m, hpl = _matrices()
    head = Head(1.0)
    g = tuple(int(z) for z in hpl[0][:2])
    n = m["raw_express"].shape[0]
    p, s = np.zeros(n), np.full(n, 20.0)
    for z in g:
        p[z] = 60.0
    common = dict(parcels_by_cell=p, stops_by_cell=s, head=head)
    v_f1 = price_group(g, 0, m, kind="delivery", freq=1.0, **common)
    v_f3 = price_group(g, 0, m, kind="delivery", freq=3.0, **common)
    v_x1 = price_group(g, 0, m, kind="express", freq=1.0, **common)
    # freq and kind are both key components: a hit on one must not serve another
    assert price_group(g, 0, m, kind="delivery", freq=1.0, **common) == v_f1
    assert price_group(g, 0, m, kind="delivery", freq=3.0, **common) == v_f3
    assert price_group(g, 0, m, kind="express", freq=1.0, **common) == v_x1
    keys = {k for k in m["_group_price_memo"] if k[0] == g}
    assert len(keys) == 3


# ─────────────────────────────────────────────────────────────────────────────
# G-6d-3 (b) — L2 hull memo
# ─────────────────────────────────────────────────────────────────────────────

def test_l2_hull_cache_entries_equal_direct_hull_km2():
    m, hpl = _matrices()
    d = 2
    cells = [int(z) for z in hpl[0]]
    pts_lon = {z: m["plz_day_lon"][z][d] for z in cells}
    pts_lat = {z: m["plz_day_lat"][z][d] for z in cells}
    cache: dict = {}
    build_partition(
        np.array(cells), m["raw_express"][:, d], m["expr_stops"][:, d],
        m["area_arr"], m["hd_arr"], m["_cent_lon"], m["_cent_lat"],
        pts_lon=pts_lon, pts_lat=pts_lat, hull_cache=cache)
    assert cache, "the fixture never reached a hull check"
    for members, hull in cache.items():
        L = np.concatenate([pts_lon[c] for c in members])
        A = np.concatenate([pts_lat[c] for c in members])
        assert hull == _hull_km2(L, A)      # bit-for-bit


def test_l2_hull_cache_does_not_change_the_partition():
    m, hpl = _matrices()
    sch = enumerate_valid_schedules()
    cache: dict = {}
    for chosen in _chosen_vectors(m["raw_express"].shape[0], sch):
        for d in range(N_DAYS):
            cells = [int(z) for z in hpl[0]
                     if not m["sched_active"][chosen[z], d]
                     and m["raw_express"][z, d] > 0]
            if not cells:
                continue
            args = (np.array(cells), m["raw_express"][:, d],
                    m["expr_stops"][:, d], m["area_arr"], m["hd_arr"],
                    m["_cent_lon"], m["_cent_lat"])
            kw = dict(pts_lon={z: m["plz_day_lon"][z][d] for z in cells},
                      pts_lat={z: m["plz_day_lat"][z][d] for z in cells})
            assert build_partition(*args, **kw, hull_cache=cache) == \
                build_partition(*args, **kw)


def test_l2_hull_key_is_the_concatenation_order_not_the_set():
    """Keys are ORDERED: a permuted point set may not reuse another's hull.

    ``_hull_km2`` divides by ``cos(radians(mean(lat)))``, and a pairwise mean
    over a permuted array can differ in the last ULP — so reusing a hull across
    orderings would be an approximation, not a cache.
    """
    m, hpl = _matrices()
    d = 1
    cells = [int(z) for z in hpl[0]]
    pts_lon = {z: m["plz_day_lon"][z][d] for z in cells}
    pts_lat = {z: m["plz_day_lat"][z][d] for z in cells}
    cache: dict = {}
    build_partition(
        np.array(cells), m["raw_express"][:, d], m["expr_stops"][:, d],
        m["area_arr"], m["hd_arr"], m["_cent_lon"], m["_cent_lat"],
        pts_lon=pts_lon, pts_lat=pts_lat, hull_cache=cache)
    for members in cache:
        assert isinstance(members, tuple)          # ordered, not a frozenset


# ─────────────────────────────────────────────────────────────────────────────
# G-6d-3 (c) — L3 partition memo
# ─────────────────────────────────────────────────────────────────────────────

def test_l3_express_partition_memo_equals_direct_build_partition():
    m, hpl = _matrices()
    sch = enumerate_valid_schedules()
    rng = np.random.default_rng(5)
    n_checked = 0
    for chosen in _chosen_vectors(m["raw_express"].shape[0], sch):
        for d in range(N_DAYS):
            contrib = [int(z) for z in hpl[0]
                       if not m["sched_active"][chosen[z], d]
                       and m["raw_express"][z, d] > 0]
            if not contrib:
                continue
            want = build_partition(
                np.array(contrib), m["raw_express"][:, d],
                m["expr_stops"][:, d], m["area_arr"], m["hd_arr"],
                m["_cent_lon"], m["_cent_lat"],
                pts_lon={z: m["plz_day_lon"][z][d] for z in contrib},
                pts_lat={z: m["plz_day_lat"][z][d] for z in contrib})
            assert _express_partition(contrib, d, m["raw_express"],
                                      m["expr_stops"], m) == want
            # permuted-input replay: same cells, different order -> same answer
            perm = list(rng.permutation(contrib))
            assert _express_partition(perm, d, m["raw_express"],
                                      m["expr_stops"], m) == want
            n_checked += 1
    assert n_checked
    assert _memo_stats(m)["partition_hit"] > 0


def test_l3_smallday_partition_memo_equals_direct_build_partition():
    m, hpl = _matrices()
    sch = enumerate_valid_schedules()
    n_checked = 0
    for chosen in _chosen_vectors(m["raw_express"].shape[0], sch):
        for d in range(N_DAYS):
            st = _pool_state(m, hpl, chosen, d)
            if st is None:
                continue
            small, parcels, stops = st
            want = build_partition(
                np.array(small), parcels, stops, m["area_arr"], m["hd_arr"],
                m["_cent_lon"], m["_cent_lat"],
                pts_lon={z: m["plz_day_lon"][z][d] for z in small},
                pts_lat={z: m["plz_day_lat"][z][d] for z in small})
            got, p_got, s_got = _smallday_partition(0, d, chosen, small, m)
            assert got == want
            assert np.array_equal(p_got, parcels)
            assert np.array_equal(s_got, stops)
            # replay with the members in a different order
            got2, _, _ = _smallday_partition(
                0, d, chosen, list(reversed(small)), m)
            assert got2 == want
            n_checked += 1
    assert n_checked


def test_l3_returned_parcel_arrays_are_not_shared_between_calls():
    """The memo caches the grouping only — the parcel/stop vectors stay fresh."""
    m, hpl = _matrices()
    sch = enumerate_valid_schedules()
    for chosen in _chosen_vectors(m["raw_express"].shape[0], sch):
        for d in range(N_DAYS):
            st = _pool_state(m, hpl, chosen, d)
            if st is None:
                continue
            small = st[0]
            _, p1, _ = _smallday_partition(0, d, chosen, small, m)
            _, p2, _ = _smallday_partition(0, d, chosen, small, m)
            assert p1 is not p2 and np.array_equal(p1, p2)
            p1[small[0]] = -999.0                 # a caller mutating its copy
            _, p3, _ = _smallday_partition(0, d, chosen, small, m)
            assert p3[small[0]] != -999.0
            return
    pytest.fail("fixture never pooled a small-delivery instance")


# ─────────────────────────────────────────────────────────────────────────────
# G-6d-3 (d) — memos are per matrices dict
# ─────────────────────────────────────────────────────────────────────────────

def test_memos_start_empty_on_a_fresh_build():
    m1, _ = _matrices()
    m2, _ = _matrices()
    for key in ("_group_price_memo", "_partition_memo", "_hull_memo"):
        assert m1[key] == {} and m2[key] == {}
        assert m1[key] is not m2[key]
    assert _memo_stats(m1) == _memo_stats(m2)
    assert all(v == 0 for v in _memo_stats(m1).values())


def test_memos_do_not_leak_between_two_matrices_dicts():
    m1, hpl = _matrices()
    m2, _ = _matrices()
    head = Head(1.0)
    g = tuple(int(z) for z in hpl[0][:3])
    price_group(g, 0, m1, kind="express", head=head)
    _express_partition(list(g), 0, m1["raw_express"], m1["expr_stops"], m1)
    assert m1["_group_price_memo"] and m1["_partition_memo"]
    assert m2["_group_price_memo"] == {} and m2["_partition_memo"] == {}


def test_memo_layers_work_on_a_hand_built_matrices_dict():
    """Legacy/hand-built dicts have no memo keys — they must be created lazily."""
    m, hpl = _matrices()
    bare = _cold(m)
    head = Head(1.0)
    g = tuple(int(z) for z in hpl[0][:3])
    v = price_group(g, 0, bare, kind="express", head=head)
    assert price_group(g, 0, bare, kind="express", head=head) == v
    p = _express_partition(list(g), 0, bare["raw_express"],
                           bare["expr_stops"], bare)
    assert _express_partition(list(g), 0, bare["raw_express"],
                             bare["expr_stops"], bare) == p


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: the head regime's hub twins are unchanged by the memos
# ─────────────────────────────────────────────────────────────────────────────

def test_head_regime_hub_costs_are_identical_with_and_without_memos():
    """The strongest statement: a warm memo changes no hub-day cost at all."""
    m, hpl = _matrices()
    sch = enumerate_valid_schedules()
    head = Head(1.3)
    m["bundle_head"] = head
    n_nonzero = 0
    for chosen in _chosen_vectors(m["raw_express"].shape[0], sch):
        for d in range(N_DAYS):
            cold = _cold(m)                    # fresh, un-memoised each call
            want_x = _hub_express_day_ml(
                0, d, chosen, hpl, sch, cold["raw_express"],
                cold["expr_stops"], cold, {}, 1.0)
            got_x = _hub_express_day_ml(
                0, d, chosen, hpl, sch, m["raw_express"], m["expr_stops"],
                m, {}, 1.0)
            assert got_x == want_x
            want_p = _hub_smallday_pool_ml(0, d, chosen, hpl, sch, _cold(m), {})
            got_p = _hub_smallday_pool_ml(0, d, chosen, hpl, sch, m, {})
            assert got_p == want_p
            n_nonzero += (got_x > 0) + (got_p > 0)
    assert n_nonzero
    st = _memo_stats(m)
    assert st["price_hit"] and st["partition_hit"]
