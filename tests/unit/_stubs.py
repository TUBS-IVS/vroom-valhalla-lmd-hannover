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
