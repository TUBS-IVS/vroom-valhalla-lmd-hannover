"""Verify the demand/stops/freq logic across all 11 share values (0, 10, ..., 100%).

For a test PLZ with known b2c/b2b daily arrivals, print the actual values used
by build_cost_matrices_ml for delivery & non-delivery days. Highlight any
discrepancies (e.g. blend assumption vs. volume-weighted truth).
"""
from __future__ import annotations
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from batch_delivery.config import N_DAYS
from batch_delivery.optimization.core import build_cost_matrices_ml


# Smooth Powerlaw model (same as orchestrators)
B2B_GLOBAL_SHARE = 0.2170
B2B_ADVANTAGE = 2.0
_B2C_SHARE = 1.0 - B2B_GLOBAL_SHARE
_EXP_LO = 1.0 / (1.0 + B2B_ADVANTAGE)
_EXP_HI = 1.0 + B2B_ADVANTAGE

def _solve_t(s):
    if s <= 0.0: return 0.0
    if s >= 1.0: return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        t = 0.5 * (lo + hi)
        agg = B2B_GLOBAL_SHARE * t**_EXP_LO + _B2C_SHARE * t**_EXP_HI
        if agg < s: lo = t
        else: hi = t
    return 0.5 * (lo + hi)

def willing_b2b(s): return _solve_t(s) ** _EXP_LO
def willing_b2c(s): return _solve_t(s) ** _EXP_HI
def fs_b2c(s): return 1.0 - willing_b2c(s)
def fs_b2b(s): return 1.0 - willing_b2b(s)


def enumerate_schedules(max_hold=3):
    out = []
    for k in range(1, N_DAYS + 1):
        for combo in combinations(range(N_DAYS), k):
            days = sorted(combo)
            ok = True
            for i in range(len(days)):
                gap = (days[(i + 1) % len(days)] - days[i]) % N_DAYS
                if gap == 0: gap = N_DAYS
                if gap > max_hold:
                    ok = False; break
            if ok: out.append(frozenset(days))
    return out


class _FeatureSpy:
    """Predictor that records what the ML sees, then returns dummy cost."""
    combo_cols = []
    kind = "spy"
    last_df = None
    def predict(self, df_feats):
        _FeatureSpy.last_df = df_feats.copy()
        return df_feats["n_parcels"].to_numpy(dtype=np.float64)


def main():
    # Two test PLZs — one B2C-heavy, one B2B-heavy
    plz_b2c_heavy = {
        "30159": {  # Hannover-Stadt-Cluster, B2C heavy
            "b2c": {d: 1000 for d in range(N_DAYS)},
            "b2b": {d: 100 for d in range(N_DAYS)},
            "area_km2": 5.0, "hub_dist_km": 8.0,
            "n_stops_per_day": 200, "total_points": 1500,
        }
    }
    plz_b2b_heavy = {
        "30900": {  # Rural, B2B heavy
            "b2c": {d: 100 for d in range(N_DAYS)},
            "b2b": {d: 200 for d in range(N_DAYS)},
            "area_km2": 50.0, "hub_dist_km": 25.0,
            "n_stops_per_day": 30, "total_points": 300,
        }
    }
    rng = np.random.default_rng(42)
    coords = {p: {d: (rng.uniform(9.7, 9.8, 30),
                       rng.uniform(52.3, 52.4, 30),
                       np.full(30, 2.6)) for d in range(N_DAYS)}
              for p in ["30159", "30900"]}

    schedules = enumerate_schedules(3)
    daily_idx = next(i for i, s in enumerate(schedules) if len(s) == 6)
    mon_thu_idx = next(i for i, s in enumerate(schedules) if s == frozenset({0, 3}))

    print("=" * 90)
    print("Scenario verification across 11 share values")
    print("=" * 90)
    print()

    # B2C-heavy PLZ
    print("PLZ 30159 — B2C-heavy (b2c=1000/day, b2b=100/day, B2B-share=9%)")
    print("-" * 90)
    print(f"{'share':>5s} | {'fs_b2c':>7s} | {'fs_b2b':>7s} | "
          f"{'willing_aggregate':>17s} | "
          f"{'Mon-deliv-demand-{Mon,Thu}':>26s} | {'Tue-nondeliv-demand':>20s} | {'Daily-demand':>13s}")
    print("-" * 90)
    for sh in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        fc = fs_b2c(sh); fb = fs_b2b(sh)
        plz_b2c_share = 1000 / 1100  # b2c / total for this PLZ
        # Volume-weighted non-willing fraction (the TRUE aggregate)
        non_will_eff = plz_b2c_share * fc + (1 - plz_b2c_share) * fb
        will_eff = 1.0 - non_will_eff

        out = build_cost_matrices_ml(
            plz_keys=["30159"], plz_data=plz_b2c_heavy,
            schedules=schedules, ml_predictor=_FeatureSpy(), provider="DHL",
            plz_day_coords=coords, hub_coords_by_plz={"30159": (9.73, 52.38)},
            fast_share_b2c=fc, fast_share_b2b=fb,
        )
        c = out["cost_3d"]
        mon_dem_2d = c[0, mon_thu_idx, 0]
        tue_dem_2d = c[0, mon_thu_idx, 1]
        daily_dem = c[0, daily_idx, 0]
        print(f"{sh:5.2f} | {fc:7.4f} | {fb:7.4f} | "
              f"{will_eff:17.4f} | "
              f"{mon_dem_2d:26.0f} | {tue_dem_2d:20.0f} | {daily_dem:13.0f}")

    print()
    print("PLZ 30900 — B2B-heavy (b2c=100/day, b2b=200/day, B2B-share=67%)")
    print("-" * 90)
    print(f"{'share':>5s} | {'fs_b2c':>7s} | {'fs_b2b':>7s} | "
          f"{'willing_aggregate':>17s} | "
          f"{'Mon-deliv-demand-{Mon,Thu}':>26s} | {'Tue-nondeliv-demand':>20s} | {'Daily-demand':>13s}")
    print("-" * 90)
    for sh in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        fc = fs_b2c(sh); fb = fs_b2b(sh)
        plz_b2c_share = 100 / 300
        non_will_eff = plz_b2c_share * fc + (1 - plz_b2c_share) * fb
        will_eff = 1.0 - non_will_eff

        out = build_cost_matrices_ml(
            plz_keys=["30900"], plz_data=plz_b2b_heavy,
            schedules=schedules, ml_predictor=_FeatureSpy(), provider="DHL",
            plz_day_coords=coords, hub_coords_by_plz={"30900": (9.73, 52.38)},
            fast_share_b2c=fc, fast_share_b2b=fb,
        )
        c = out["cost_3d"]
        mon_dem_2d = c[0, mon_thu_idx, 0]
        tue_dem_2d = c[0, mon_thu_idx, 1]
        daily_dem = c[0, daily_idx, 0]
        print(f"{sh:5.2f} | {fc:7.4f} | {fb:7.4f} | "
              f"{will_eff:17.4f} | "
              f"{mon_dem_2d:26.0f} | {tue_dem_2d:20.0f} | {daily_dem:13.0f}")

    # Now check stops + freq for the same PLZs
    print()
    print("=" * 90)
    print("STOPS feature check (PLZ 30159, schedule {Mon, Thu})")
    print("-" * 90)
    print(f"{'share':>5s} | {'Mon-deliv-stops':>16s} | {'Tue-nondeliv-stops':>20s} | {'Daily-stops':>12s}")
    print("-" * 90)
    class _StopsSpy:
        combo_cols = []
        kind = "spy"
        def predict(self, df): return df["n_stops"].to_numpy(dtype=np.float64)
    for sh in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        fc = fs_b2c(sh); fb = fs_b2b(sh)
        out = build_cost_matrices_ml(
            plz_keys=["30159"], plz_data=plz_b2c_heavy,
            schedules=schedules, ml_predictor=_StopsSpy(), provider="DHL",
            plz_day_coords=coords, hub_coords_by_plz={"30159": (9.73, 52.38)},
            fast_share_b2c=fc, fast_share_b2b=fb,
        )
        c = out["cost_3d"]
        mon_stops = c[0, mon_thu_idx, 0]
        tue_stops = c[0, mon_thu_idx, 1]
        daily_stops = c[0, daily_idx, 0]
        print(f"{sh:5.2f} | {mon_stops:16.1f} | {tue_stops:20.1f} | {daily_stops:12.1f}")

    print()
    print("=" * 90)
    print("FREQ feature check (PLZ 30159, schedule {Mon, Thu})")
    print("-" * 90)
    print(f"{'share':>5s} | {'Mon-deliv-freq':>15s} | {'Tue-nondeliv-freq':>19s} | {'Daily-freq':>11s}")
    print("-" * 90)
    class _FreqSpy:
        combo_cols = []
        kind = "spy"
        def predict(self, df): return df["delivery_frequency"].to_numpy(dtype=np.float64)
    for sh in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        fc = fs_b2c(sh); fb = fs_b2b(sh)
        out = build_cost_matrices_ml(
            plz_keys=["30159"], plz_data=plz_b2c_heavy,
            schedules=schedules, ml_predictor=_FreqSpy(), provider="DHL",
            plz_day_coords=coords, hub_coords_by_plz={"30159": (9.73, 52.38)},
            fast_share_b2c=fc, fast_share_b2b=fb,
        )
        c = out["cost_3d"]
        mon_freq = c[0, mon_thu_idx, 0]
        tue_freq = c[0, mon_thu_idx, 1]
        daily_freq = c[0, daily_idx, 0]
        print(f"{sh:5.2f} | {mon_freq:15.3f} | {tue_freq:19.3f} | {daily_freq:11.3f}")

    print()
    print("=" * 90)
    print("Notes:")
    print("  - At share=0.00: Mon-delivery should = today's full demand (1100); "
          "Daily = same; Tue-non-delivery = today's non-willing × (b2c×1 + b2b×1)")
    print("  - At share=1.00: Mon-delivery should = today + 2 prior days willing "
          "(= 3 × today's parcels = 3300); Tue-non-delivery = 0")
    print("  - At share=0.50: Mon-delivery should = today + ~half prior; "
          "Tue-non-delivery = ~half today")
    print()


if __name__ == "__main__":
    main()
