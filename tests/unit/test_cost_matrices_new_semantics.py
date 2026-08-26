"""Failing tests for the new build_cost_matrices_ml semantics.

The new model:
  * Delivery day demand  = today's ALL arrivals + willing fraction of prior
                            unscheduled days' arrivals
  * Non-delivery day     = non-willing fraction of today's arrivals
  * No separate express-tour cost layer — every (PLZ, day) tour predicted
                            individually by the ML surrogate.

This contrasts with the previous semantics where delivery days only carried
the willing fraction of source-day arrivals (today's non-willing went via
hub-bundled express tours).
"""
from __future__ import annotations
from itertools import combinations
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from batch_delivery.config import N_DAYS, WEEKDAYS
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


def _make_plz_data(b2c_per_day: int = 100, b2b_per_day: int = 30):
    """Single-PLZ fixture with uniform daily arrivals."""
    return {
        "30159": {
            "b2c": {d: b2c_per_day for d in range(N_DAYS)},
            "b2b": {d: b2b_per_day for d in range(N_DAYS)},
            "area_km2": 5.0,
            "hub_dist_km": 8.0,
            "n_stops_per_day": 50,
            "total_points": 300,
        }
    }


class _DemandSpy:
    """Predictor that returns the n_parcels feature it was given, so we can
    invert the test and check the demand that was actually fed in.
    """

    combo_cols: list[str] = []
    kind = "spy"

    def predict(self, df_feats: pd.DataFrame) -> np.ndarray:
        return df_feats["n_parcels"].to_numpy(dtype=np.float64)


def _coords_for(plz: str):
    """Synthetic per-day coords (lon, lat, per_stop_demand)."""
    rng = np.random.default_rng(42)
    out = {}
    for d in range(N_DAYS):
        lons = rng.uniform(9.7, 9.8, 50)
        lats = rng.uniform(52.3, 52.4, 50)
        psd = np.full(50, 2.6)
        out[d] = (lons, lats, psd)
    return out


# ---------------------------------------------------------------------------


def test_delivery_day_includes_todays_non_willing():
    """At a scheduled delivery day, demand must include the FULL today's
    arrivals (willing + non-willing), plus the willing fraction of prior
    non-scheduled days back to the previous delivery day.
    """
    plz_data = _make_plz_data(b2c_per_day=100, b2b_per_day=30)
    schedules = _enumerate_schedules(max_hold=3)
    mon_thu_idx = next(i for i, s in enumerate(schedules) if s == frozenset({0, 3}))

    spy = _DemandSpy()
    out = build_cost_matrices_ml(
        plz_keys=["30159"],
        plz_data=plz_data,
        schedules=schedules,
        ml_predictor=spy,
        provider="DHL",
        plz_day_coords={"30159": _coords_for("30159")},
        hub_coords_by_plz={"30159": (9.73, 52.38)},
        fast_share_b2c=0.5,    # 50% B2C unwilling
        fast_share_b2b=0.5,    # 50% B2B unwilling
    )
    cost_3d = out["cost_3d"]

    # Schedule {Mon=0, Thu=3} → source days for Mon = {Fri=4, Sat=5, Mon=0}
    # Daily arrivals: 100 b2c + 30 b2b = 130 per day
    # On Mon (delivery day):
    #   today_all          = 130          (Mon's full arrivals)
    #   willing_prior_fri  = 100·0.5 + 30·0.5 = 65   (50% willing, batched from Fri)
    #   willing_prior_sat  = 65
    #   expected demand    = 130 + 65 + 65 = 260
    expected_mon_demand = 130 + 65 + 65
    actual_mon_demand = cost_3d[0, mon_thu_idx, 0]
    assert actual_mon_demand == pytest.approx(expected_mon_demand, abs=1.0), (
        f"Mon delivery demand was {actual_mon_demand} but expected "
        f"{expected_mon_demand} (today's ALL + willing-prior-fri + willing-prior-sat)"
    )


def test_non_delivery_day_only_non_willing():
    """At a non-scheduled day, only the non-willing fraction of today's
    arrivals is on tour."""
    plz_data = _make_plz_data(b2c_per_day=100, b2b_per_day=30)
    schedules = _enumerate_schedules(max_hold=3)
    mon_thu_idx = next(i for i, s in enumerate(schedules) if s == frozenset({0, 3}))

    spy = _DemandSpy()
    out = build_cost_matrices_ml(
        plz_keys=["30159"],
        plz_data=plz_data,
        schedules=schedules,
        ml_predictor=spy,
        provider="DHL",
        plz_day_coords={"30159": _coords_for("30159")},
        hub_coords_by_plz={"30159": (9.73, 52.38)},
        fast_share_b2c=0.5,
        fast_share_b2b=0.5,
    )
    cost_3d = out["cost_3d"]

    # On Tue (non-delivery day in {Mon, Thu}):
    #   non_willing = 100·0.5 + 30·0.5 = 65
    expected_tue_demand = 65
    actual_tue_demand = cost_3d[0, mon_thu_idx, 1]
    assert actual_tue_demand == pytest.approx(expected_tue_demand, abs=1.0), (
        f"Tue non-delivery demand was {actual_tue_demand} but expected "
        f"{expected_tue_demand} (only non-willing today)"
    )


def test_share_zero_keeps_today_arrivals():
    """At share_willing=0 (fast_share=1.0), no parcels are batched.
    Every day delivers today's arrivals only — i.e. cost should be the same
    across all schedules for the same day.
    """
    plz_data = _make_plz_data(b2c_per_day=100, b2b_per_day=30)
    schedules = _enumerate_schedules(max_hold=3)
    daily_idx = next(i for i, s in enumerate(schedules)
                     if s == frozenset({0, 1, 2, 3, 4, 5}))
    mon_thu_idx = next(i for i, s in enumerate(schedules) if s == frozenset({0, 3}))

    spy = _DemandSpy()
    out = build_cost_matrices_ml(
        plz_keys=["30159"],
        plz_data=plz_data,
        schedules=schedules,
        ml_predictor=spy,
        provider="DHL",
        plz_day_coords={"30159": _coords_for("30159")},
        hub_coords_by_plz={"30159": (9.73, 52.38)},
        fast_share_b2c=1.0,   # 100% non-willing
        fast_share_b2b=1.0,
    )
    # ``cost_3d_raw`` is the UNPOOLED per-cell prediction. ``cost_3d`` zeroes
    # delivery instances below MIN_TOUR_PARCELS (the rev1 small-delivery rule
    # hands those to ``_hub_smallday_pool_ml``), and at 130 parcels/day every
    # instance here is below that threshold — so the spy must read the raw
    # matrix to see the demand that actually reached the featuriser.
    cost_3d = out["cost_3d_raw"]

    # Daily (size=6): every day is a delivery day with today's ALL arrivals
    # No willing-prior fraction since fast_share=1.0 → demand = 130 every day
    assert all(
        cost_3d[0, daily_idx, d] == pytest.approx(130, abs=1.0)
        for d in range(N_DAYS)
    )

    # {Mon, Thu}: Mon and Thu are delivery days (today's 130 + 0 willing-prior = 130).
    #             Tue/Wed/Fri/Sat are non-delivery days with non_willing today = 130.
    # Total weekly = 6 × 130 = 780 (same as daily — no batching possible).
    weekly_sum = cost_3d[0, mon_thu_idx, :].sum()
    daily_weekly_sum = cost_3d[0, daily_idx, :].sum()
    assert weekly_sum == pytest.approx(daily_weekly_sum, rel=0.05), (
        f"At share_willing=0, all schedules should cost the same; "
        f"{{Mon,Thu}}={weekly_sum:.0f} vs daily={daily_weekly_sum:.0f}"
    )


def test_share_one_full_batching_on_delivery_days_only():
    """At share_willing=1.0 (fast_share=0.0), all parcels are willing.
    Non-delivery days have 0 demand. Delivery days carry full source-day load.
    """
    plz_data = _make_plz_data(b2c_per_day=100, b2b_per_day=30)
    schedules = _enumerate_schedules(max_hold=3)
    mon_thu_idx = next(i for i, s in enumerate(schedules) if s == frozenset({0, 3}))

    spy = _DemandSpy()
    out = build_cost_matrices_ml(
        plz_keys=["30159"],
        plz_data=plz_data,
        schedules=schedules,
        ml_predictor=spy,
        provider="DHL",
        plz_day_coords={"30159": _coords_for("30159")},
        hub_coords_by_plz={"30159": (9.73, 52.38)},
        fast_share_b2c=0.0,
        fast_share_b2b=0.0,
    )
    cost_3d = out["cost_3d"]

    # Tue/Wed/Fri/Sat (non-delivery, fast_share=0) → 0 demand
    for d in [1, 2, 4, 5]:
        assert cost_3d[0, mon_thu_idx, d] == pytest.approx(0, abs=1.0), (
            f"Non-delivery day {d} should have 0 demand at share=1 "
            f"but got {cost_3d[0, mon_thu_idx, d]}"
        )
    # Mon delivery: today (130) + willing-prior-fri (130) + willing-prior-sat (130) = 390
    assert cost_3d[0, mon_thu_idx, 0] == pytest.approx(390, abs=1.0)
    # Thu delivery: today (130) + willing-prior-tue (130) + willing-prior-wed (130) = 390
    assert cost_3d[0, mon_thu_idx, 3] == pytest.approx(390, abs=1.0)
