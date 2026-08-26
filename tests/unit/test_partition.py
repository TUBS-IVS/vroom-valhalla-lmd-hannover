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
