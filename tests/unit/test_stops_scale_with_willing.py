"""Verify that combined_stops scales with the willing fraction at delivery days.

Old (buggy) formula:
    dd_stops = stops_per_day × n_source                  # share-independent

Fixed formula:
    dd_stops = stops_per_day × (1 + willing_blend × (n_source − 1))

At share=0 (nobody willing): batched delivery day stops = today's stops only
                              (matches "no batching" semantics).
At share=1 (everyone willing): batched delivery day stops = stops_per_day × n_source
                                (matches training agg_k=n_source semantics).
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
            "total_points": 1000,    # high cap so it doesn't clip
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


class _StopsSpy:
    combo_cols: list[str] = []
    kind = "spy"

    def predict(self, df_feats: pd.DataFrame) -> np.ndarray:
        return df_feats["n_stops"].to_numpy(dtype=np.float64)


def test_dd_stops_at_share_zero_equals_stops_per_day():
    """At share_willing=0 (fast_share=1), delivery day stops should NOT be
    inflated by n_source — no batching is happening."""
    schedules = _enumerate_schedules(max_hold=3)
    mon_thu_idx = next(i for i, s in enumerate(schedules)
                       if s == frozenset({0, 3}))

    spy = _StopsSpy()
    out = build_cost_matrices_ml(
        plz_keys=["30159"],
        plz_data=_plz_data(),
        schedules=schedules,
        ml_predictor=spy,
        provider="DHL",
        plz_day_coords=_coords(),
        hub_coords_by_plz={"30159": (9.73, 52.38)},
        fast_share_b2c=1.0,
        fast_share_b2b=1.0,
    )
    # ``cost_3d_raw`` = unpooled per-cell prediction. ``cost_3d`` zeroes
    # delivery instances below MIN_TOUR_PARCELS (rev1 small-delivery rule);
    # at share=0 this PLZ delivers 130 parcels/day, below that threshold.
    cost_3d = out["cost_3d_raw"]

    # At share=0, every day delivers today's parcels only.
    # Stops per tour = stops_per_day = 50 (not 150 like the old bug).
    for d in range(N_DAYS):
        assert cost_3d[0, mon_thu_idx, d] == pytest.approx(50.0, abs=1.0), (
            f"At share=0, day {d}: expected stops=50, "
            f"got {cost_3d[0, mon_thu_idx, d]}"
        )


def test_dd_stops_at_share_one_scales_with_n_source():
    """At share_willing=1 (fast_share=0), delivery day stops should scale with
    n_source (matches training agg_k semantics)."""
    schedules = _enumerate_schedules(max_hold=3)
    mon_thu_idx = next(i for i, s in enumerate(schedules)
                       if s == frozenset({0, 3}))

    spy = _StopsSpy()
    out = build_cost_matrices_ml(
        plz_keys=["30159"],
        plz_data=_plz_data(),
        schedules=schedules,
        ml_predictor=spy,
        provider="DHL",
        plz_day_coords=_coords(),
        hub_coords_by_plz={"30159": (9.73, 52.38)},
        fast_share_b2c=0.0,
        fast_share_b2b=0.0,
    )
    cost_3d = out["cost_3d"]

    # On delivery day Mon/Thu: n_source=3 → 3 × 50 = 150 stops
    assert cost_3d[0, mon_thu_idx, 0] == pytest.approx(150.0, abs=1.0)
    assert cost_3d[0, mon_thu_idx, 3] == pytest.approx(150.0, abs=1.0)


def test_dd_stops_at_share_half_interpolates():
    """At share_willing=0.5, dd_stops should interpolate linearly between
    today's stops and n_source × stops_per_day."""
    schedules = _enumerate_schedules(max_hold=3)
    mon_thu_idx = next(i for i, s in enumerate(schedules)
                       if s == frozenset({0, 3}))

    spy = _StopsSpy()
    out = build_cost_matrices_ml(
        plz_keys=["30159"],
        plz_data=_plz_data(),
        schedules=schedules,
        ml_predictor=spy,
        provider="DHL",
        plz_day_coords=_coords(),
        hub_coords_by_plz={"30159": (9.73, 52.38)},
        fast_share_b2c=0.5,
        fast_share_b2b=0.5,
    )
    cost_3d = out["cost_3d"]

    # At share=0.5: dd_stops = sd × (1 + 0.5 × (n_source−1))
    #             = 50 × (1 + 0.5 × 2)
    #             = 100 stops on Mon/Thu
    assert cost_3d[0, mon_thu_idx, 0] == pytest.approx(100.0, abs=1.0)
    assert cost_3d[0, mon_thu_idx, 3] == pytest.approx(100.0, abs=1.0)
