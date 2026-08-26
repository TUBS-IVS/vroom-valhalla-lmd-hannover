"""Verify that delivery_frequency feature also scales with willing fraction.

At share=0 (no batching): freq=1 everywhere (today's parcels only)
At share=1 (full batching): freq=n_source (matches training agg_k)
At share=0.5: freq=(1+n_source)/2 (interpolated)
"""
from __future__ import annotations
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from batch_delivery.config import N_DAYS
from batch_delivery.optimization.core import build_cost_matrices_ml


def _enumerate_schedules(max_hold: int = 3):
    out = []
    for k in range(1, N_DAYS + 1):
        for combo in combinations(range(N_DAYS), k):
            days = sorted(combo)
            ok = True
            for i in range(len(days)):
                gap = (days[(i + 1) % len(days)] - days[i]) % N_DAYS
                if gap == 0:
                    gap = N_DAYS
                if gap > max_hold:
                    ok = False
                    break
            if ok:
                out.append(frozenset(days))
    return out


def _plz_data():
    return {
        "30159": {
            "b2c": {d: 100 for d in range(N_DAYS)},
            "b2b": {d: 30 for d in range(N_DAYS)},
            "area_km2": 5.0,
            "hub_dist_km": 8.0,
            "n_stops_per_day": 50,
            "total_points": 1000,
        }
    }


def _coords():
    rng = np.random.default_rng(42)
    out = {}
    for d in range(N_DAYS):
        lons = rng.uniform(9.7, 9.8, 50)
        lats = rng.uniform(52.3, 52.4, 50)
        psd = np.full(50, 2.6)
        out[d] = (lons, lats, psd)
    return {"30159": out}


class _FreqSpy:
    combo_cols: list[str] = []
    kind = "spy"
    def predict(self, df_feats: pd.DataFrame) -> np.ndarray:
        return df_feats["delivery_frequency"].to_numpy(dtype=np.float64)


def test_freq_is_1_at_share_zero():
    """At share=0, no batching → freq=1 everywhere regardless of schedule."""
    schedules = _enumerate_schedules(max_hold=3)
    mon_thu_idx = next(i for i, s in enumerate(schedules)
                       if s == frozenset({0, 3}))

    out = build_cost_matrices_ml(
        plz_keys=["30159"], plz_data=_plz_data(), schedules=schedules,
        ml_predictor=_FreqSpy(), provider="DHL",
        plz_day_coords=_coords(),
        hub_coords_by_plz={"30159": (9.73, 52.38)},
        fast_share_b2c=1.0, fast_share_b2b=1.0,
    )
    # ``cost_3d_raw`` = unpooled per-cell prediction. ``cost_3d`` zeroes
    # delivery instances below MIN_TOUR_PARCELS (rev1 small-delivery rule);
    # at share=0 this PLZ delivers 130 parcels/day, below that threshold.
    cost_3d = out["cost_3d_raw"]
    for d in range(N_DAYS):
        assert cost_3d[0, mon_thu_idx, d] == pytest.approx(1.0, abs=1e-6), (
            f"At share=0, day {d}: expected freq=1, got {cost_3d[0, mon_thu_idx, d]}"
        )


def test_freq_is_n_source_at_share_one():
    """At share=1, full batching → freq=n_source on delivery days."""
    schedules = _enumerate_schedules(max_hold=3)
    mon_thu_idx = next(i for i, s in enumerate(schedules)
                       if s == frozenset({0, 3}))

    out = build_cost_matrices_ml(
        plz_keys=["30159"], plz_data=_plz_data(), schedules=schedules,
        ml_predictor=_FreqSpy(), provider="DHL",
        plz_day_coords=_coords(),
        hub_coords_by_plz={"30159": (9.73, 52.38)},
        fast_share_b2c=0.0, fast_share_b2b=0.0,
    )
    cost_3d = out["cost_3d"]
    # Mon, Thu delivery days have n_source=3
    assert cost_3d[0, mon_thu_idx, 0] == pytest.approx(3.0, abs=1e-6)
    assert cost_3d[0, mon_thu_idx, 3] == pytest.approx(3.0, abs=1e-6)


def test_freq_interpolates_at_half_share():
    """At share=0.5, freq = (1 + 3) / 2 = 2 on delivery days."""
    schedules = _enumerate_schedules(max_hold=3)
    mon_thu_idx = next(i for i, s in enumerate(schedules)
                       if s == frozenset({0, 3}))

    out = build_cost_matrices_ml(
        plz_keys=["30159"], plz_data=_plz_data(), schedules=schedules,
        ml_predictor=_FreqSpy(), provider="DHL",
        plz_day_coords=_coords(),
        hub_coords_by_plz={"30159": (9.73, 52.38)},
        fast_share_b2c=0.5, fast_share_b2b=0.5,
    )
    cost_3d = out["cost_3d"]
    assert cost_3d[0, mon_thu_idx, 0] == pytest.approx(2.0, abs=1e-6)
    assert cost_3d[0, mon_thu_idx, 3] == pytest.approx(2.0, abs=1e-6)
