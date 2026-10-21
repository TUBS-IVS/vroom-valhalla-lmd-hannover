"""Shared test doubles for cost-matrix unit tests.

``StubPredictor`` is a linear-in-features stand-in for ``MLCostPredictor``
so tests can assert exact predicted costs without loading a pickle.
``tiny_matrices`` builds a minimal 2-cell ``build_cost_matrices_ml`` result
for reuse across test modules.

No ``__init__.py`` here on purpose: ``tests/unit`` has none either, so
pytest's default "prepend" import mode puts this directory on ``sys.path``
and a plain ``from _stubs import ...`` resolves it.
"""
import numpy as np
import pandas as pd

from batch_delivery.features import ALL_COLS


class StubPredictor:
    """Linear in n_parcels + hub_dist so tests can assert exact values."""
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return 2.0 * df["n_parcels"].to_numpy() + 10.0 * df["hub_dist_km"].to_numpy()
    def predict_single(self, x25: np.ndarray) -> float:
        i_p, i_h = ALL_COLS.index("n_parcels"), ALL_COLS.index("hub_dist_km")
        return float(2.0 * x25[i_p] + 10.0 * x25[i_h])

def cell_matrices(specs, fs: float, spread: float = 0.01,
                  provider: str = "DHL"):
    """Matrices for cells given as ``(b2c_per_day, b2b_per_day)`` pairs.

    Every cell shares one hub. ``spread=0.0`` co-locates them, which is what
    makes ``build_partition`` pack the sub-threshold ones into a single tour —
    the case the pooled-vehicle rule (spec §4.3 v3) is about.

    Returns ``(plz_keys, matrices, schedules, hub_plz_list, plz_hub_arr)``.
    """
    import numpy as _np

    from batch_delivery.config.constants import N_DAYS
    from batch_delivery.optimization.costs import build_cost_matrices_ml
    from batch_delivery.optimization.schedules import enumerate_valid_schedules

    plz_keys = [f"3{i:04d}" for i in range(len(specs))]
    plz_data, coords, hubs = {}, {}, {}
    for i, (b2c, b2b) in enumerate(specs):
        pc = plz_keys[i]
        plz_data[pc] = {
            "b2c": {d: float(b2c) for d in range(N_DAYS)},
            "b2b": {d: float(b2b) for d in range(N_DAYS)},
            "area_km2": 5.0, "hub_dist_km": 4.0 + i,
            "n_stops_per_day": 25.0, "total_points": 600.0,
        }
        lon, lat = 9.70 + spread * i, 52.35 + spread * i
        coords[pc] = {d: (_np.array([lon, lon + 0.001]),
                          _np.array([lat, lat + 0.001]),
                          _np.array([2.0, 3.0])) for d in range(N_DAYS)}
        hubs[pc] = (9.73, 52.38)
    sch = enumerate_valid_schedules()
    m = build_cost_matrices_ml(
        plz_keys, plz_data, sch, StubPredictor(), provider, coords, hubs,
        fast_share_b2c=fs, fast_share_b2b=fs)
    return (plz_keys, m, sch, [_np.arange(len(specs))],
            _np.zeros(len(specs), dtype=int))


def tiny_matrices(theta_one: bool):
    from batch_delivery.optimization.costs import build_cost_matrices_ml
    from batch_delivery.optimization.schedules import enumerate_valid_schedules
    # 2 cells, tiny coords; fast shares: theta=1 -> 0.0 (no express)
    plz_keys = ["11111", "22222"]
    plz_data = {
        "11111": {"b2c": {d: 100 for d in range(6)},
                   "b2b": {d: 20 for d in range(6)},
                   "area_km2": 12.0, "hub_dist_km": 7.5,
                   "n_stops_per_day": 80.0, "total_points": 1000.0},
        "22222": {"b2c": {d: 300 for d in range(6)},
                   "b2b": {d: 50 for d in range(6)},
                   "area_km2": 30.0, "hub_dist_km": 14.0,
                   "n_stops_per_day": 200.0, "total_points": 3000.0},
    }
    coords = {p: {d: (np.array([9.7, 9.71]), np.array([52.4, 52.41]),
                      np.array([3.0, 4.0])) for d in range(6)}
              for p in plz_keys}
    hubs = {p: (9.73, 52.38) for p in plz_keys}
    fs = 0.0 if theta_one else 0.6
    return build_cost_matrices_ml(
        plz_keys, plz_data, enumerate_valid_schedules(), StubPredictor(), "DHL",
        coords, hubs, fast_share_b2c=fs, fast_share_b2b=fs)
