import numpy as np
import pytest
from batch_delivery.optimization.partition import build_partition
from batch_delivery.config.constants import (
    MIN_TOUR_PARCELS, MAX_TOUR_STOPS, MAX_TOUR_AREA_KM2, MAX_HULL_RATIO,
)

def _mk(n, parcels, stops=None, areas=None, hd=None, lon=None, lat=None):
    cells = np.arange(n)
    return dict(
        cells=cells,
        parcels=np.asarray(parcels, float),
        stops=np.asarray(stops if stops is not None else [100.0] * n, float),
        areas=np.asarray(areas if areas is not None else [10.0] * n, float),
        hub_dist=np.asarray(hd if hd is not None else [5.0] * n, float),
        cent_lon=np.asarray(lon if lon is not None else np.linspace(9.7, 9.7 + 0.01 * n, n), float),
        cent_lat=np.asarray(lat if lat is not None else [52.4] * n, float),
    )

def test_large_cells_stay_singletons():
    p = build_partition(**_mk(3, [300, 400, 500]))
    assert p == ((0,), (1,), (2,))

def test_small_adjacent_cells_pool():
    p = build_partition(**_mk(3, [100, 100, 300]))
    groups = {g for g in p}
    assert (2,) in groups                       # large stays alone
    assert (0, 1) in groups                     # smalls pool

def test_stop_cap_splits():
    k = _mk(3, [100, 100, 100], stops=[300, 300, 300])
    p = build_partition(**k)                    # any pair = 600 > 556
    assert all(len(g) == 1 for g in p)

def test_area_cap_splits():
    k = _mk(2, [100, 100], areas=[100, 100])    # union 200 > 159
    assert build_partition(**k) == ((0,), (1,))

def test_hull_cap_rejects_dispersed_merge():
    # two tiny areas 50 km apart: hull >> summed area
    k = _mk(2, [100, 100], areas=[2, 2], lon=[9.0, 9.7], lat=[52.0, 52.0])
    pts_lon = {0: np.array([9.0, 9.001, 9.0]), 1: np.array([9.7, 9.701, 9.7])}
    pts_lat = {0: np.array([52.0, 52.0, 52.001]), 1: np.array([52.0, 52.0, 52.001])}
    p = build_partition(**k, pts_lon=pts_lon, pts_lat=pts_lat)
    assert p == ((0,), (1,))

def test_deterministic_under_input_permutation():
    k = _mk(5, [100, 120, 90, 110, 80])
    p1 = build_partition(**k)
    k2 = dict(k); k2["cells"] = k["cells"][::-1].copy()
    p2 = build_partition(**k2)
    assert p1 == p2

def test_canonical_form():
    p = build_partition(**_mk(4, [100, 100, 100, 100]))
    for g in p:
        assert list(g) == sorted(g)
    assert list(p) == sorted(p, key=lambda g: g[0])

def test_hull_check_skips_when_trial_group_has_no_point_geometry():
    # Both members of the candidate merge have EMPTY pts arrays (e.g. a PLZ
    # with no geometry recorded for this day). np.concatenate([]) must not
    # raise -- treated like the <3-points -> hull 0.0 case, so the merge
    # proceeds under the size caps (mirrors test_small_adjacent_cells_pool).
    k = _mk(2, [100, 100])
    pts_lon = {0: np.array([]), 1: np.array([])}
    pts_lat = {0: np.array([]), 1: np.array([])}
    p = build_partition(**k, pts_lon=pts_lon, pts_lat=pts_lat)
    assert p == ((0, 1),)

def test_hull_check_never_feeds_mismatched_lon_lat_arrays(monkeypatch):
    # M2 (review round 2): lon_parts was filtered on pts_lon.get(c) and
    # lat_parts on pts_lat.get(c) INDEPENDENTLY. A cell present in one dict
    # but empty/missing in the other (a data problem, not the "nobody has
    # geometry" case above) desyncs the two lists: np.concatenate still
    # succeeds (concatenate does not require matching per-array lengths) but
    # L and A end up different TOTAL lengths, which _hull_km2 silently
    # swallows via its caught ValueError (returns 0.0) instead of raising --
    # a silent wrong-answer, not a crash, so a plain "does it raise" test
    # cannot catch it. Spy on _hull_km2 and assert it is always called with
    # paired (same-length) arrays.
    import batch_delivery.optimization.partition as partition_mod
    calls: list[tuple[int, int]] = []
    orig_hull_km2 = partition_mod._hull_km2

    def spy(lon, lat):
        calls.append((len(lon), len(lat)))
        return orig_hull_km2(lon, lat)

    monkeypatch.setattr(partition_mod, "_hull_km2", spy)

    k = _mk(2, [100, 100], areas=[2.0, 2.0], lon=[9.000, 9.001], lat=[52.0, 52.0])
    pts_lon = {
        0: np.array([9.000, 9.0001, 9.0002]),   # paired: 3 lon / 3 lat
        1: np.array([9.0011, 9.0012]),          # MISMATCHED: 2 lon / 0 lat
    }
    pts_lat = {
        0: np.array([52.000, 52.0001, 52.0002]),
        1: np.array([]),
    }
    build_partition(**k, pts_lon=pts_lon, pts_lat=pts_lat)

    assert calls, "fixture must actually exercise the hull check"
    for n_lon, n_lat in calls:
        assert n_lon == n_lat, f"_hull_km2 called with mismatched shapes {(n_lon, n_lat)}"
