"""Unit tests for the new optimization neighborhood helpers and result writers.

Covers:
* ``_day_toggle_neighbors`` — neighbour count + symmetry property
* ``optimize_cd_ml(..., pair_polish=True)`` — runs without error and never
  worsens the cost vs. plain CD on the same input
* ``export_optimization_results`` — produces all expected CSVs + PNGs
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from batch_delivery.config import N_DAYS, WEEKDAYS
from batch_delivery.optimization.core import (
    _day_toggle_neighbors,
    enumerate_valid_schedules,
    optimize_cd_ml,
)
from batch_delivery.optimization.results import export_optimization_results


# ---------------------------------------------------------------------------
# Day-toggle neighbourhood
# ---------------------------------------------------------------------------

def test_day_toggle_neighbors_symmetry() -> None:
    schedules = enumerate_valid_schedules()
    nbrs = _day_toggle_neighbors(schedules)
    assert len(nbrs) == len(schedules)
    # Symmetry: si in nbrs[sj] iff sj in nbrs[si]
    for si in range(len(schedules)):
        for sj in nbrs[si]:
            assert si in nbrs[int(sj)], (
                f"asymmetric: {si} → {sj} but not back"
            )
        # Distance check: each neighbour really differs by exactly one day
        for sj in nbrs[si]:
            sym = schedules[si].symmetric_difference(schedules[int(sj)])
            assert len(sym) == 1


def test_day_toggle_neighbors_full_pattern() -> None:
    # The "all days" pattern only has neighbours that drop a single day
    schedules = enumerate_valid_schedules()
    nbrs = _day_toggle_neighbors(schedules)
    full_idx = next(
        i for i, s in enumerate(schedules) if len(s) == N_DAYS
    )
    # Every neighbour must have N_DAYS - 1 days
    for sj in nbrs[full_idx]:
        assert len(schedules[int(sj)]) == N_DAYS - 1


# ---------------------------------------------------------------------------
# Pair-polish in optimize_cd_ml (synthetic small instance)
# ---------------------------------------------------------------------------

def _synthetic_problem(
    n_plz: int = 8, n_hubs: int = 2, seed: int = 0,
) -> dict:
    """Tiny synthetic optimisation problem used for the polish smoke."""
    rng = np.random.default_rng(seed)
    schedules = enumerate_valid_schedules()
    n_sched = len(schedules)

    plz_keys = [f"301{i:02d}" for i in range(n_plz)]
    plz_hub_arr = np.array(
        [i % n_hubs for i in range(n_plz)], dtype=np.intp,
    )
    hub_plz_list = [
        np.where(plz_hub_arr == h)[0].astype(np.intp)
        for h in range(n_hubs)
    ]
    hub_names = [f"H{h}" for h in range(n_hubs)]

    dd_cost_mx = rng.uniform(50, 500, size=(n_plz, n_sched))
    veh_3d = rng.integers(1, 5, size=(n_plz, n_sched, N_DAYS)).astype(np.float64)
    raw_express = rng.uniform(0, 100, size=(n_hubs, N_DAYS))
    expr_stops = rng.uniform(0, 30, size=(n_hubs, N_DAYS))

    matrices = {
        "dd_cost_mx": dd_cost_mx,
        "veh_3d": veh_3d,
        "raw_express": raw_express,
        "expr_stops": expr_stops,
        # placeholders consumed by _hub_express_day_ml — keep tests focussed
        # on batch-only path which sidesteps the express predictor.
        "plz_n_parcels": rng.integers(20, 200, size=n_plz),
        "plz_n_stops": rng.integers(10, 60, size=n_plz),
    }
    return {
        "plz_keys": plz_keys,
        "plz_hub_arr": plz_hub_arr,
        "hub_plz_list": hub_plz_list,
        "hub_names": hub_names,
        "schedules": schedules,
        "matrices": matrices,
    }


def test_optimize_cd_ml_pair_polish_runs() -> None:
    """Pair-polish must not crash and must not raise the cost above plain CD.

    Uses the batch-only fast path so we don't depend on the ML express
    predictor for this synthetic test.
    """
    p = _synthetic_problem(n_plz=8, n_hubs=2)

    plain = optimize_cd_ml(
        plz_keys=p["plz_keys"],
        plz_hub_arr=p["plz_hub_arr"],
        hub_plz_list=p["hub_plz_list"],
        matrices=p["matrices"],
        schedules=p["schedules"],
        batch_only=True,
    )
    polished = optimize_cd_ml(
        plz_keys=p["plz_keys"],
        plz_hub_arr=p["plz_hub_arr"],
        hub_plz_list=p["hub_plz_list"],
        matrices=p["matrices"],
        schedules=p["schedules"],
        batch_only=True,
        pair_polish=True,        # no-op on batch-only, but must not error
        pair_polish_rounds=2,
        pair_polish_max_pairs=20,
    )
    # Batch-only is already globally optimal — pair polish must not worsen
    assert polished["best_cost"] == pytest.approx(plain["best_cost"])


# ---------------------------------------------------------------------------
# Result export
# ---------------------------------------------------------------------------

def test_export_optimization_results_writes_artifacts(tmp_path: Path) -> None:
    p = _synthetic_problem(n_plz=8, n_hubs=2)
    cd = optimize_cd_ml(
        plz_keys=p["plz_keys"],
        plz_hub_arr=p["plz_hub_arr"],
        hub_plz_list=p["hub_plz_list"],
        matrices=p["matrices"],
        schedules=p["schedules"],
        batch_only=True,
    )

    paths = export_optimization_results(
        plz_keys=p["plz_keys"],
        plz_hub_arr=p["plz_hub_arr"],
        hub_names=p["hub_names"],
        hub_plz_list=p["hub_plz_list"],
        schedules=p["schedules"],
        matrices=p["matrices"],
        cd_result=cd,
        out_dir=tmp_path,
        summary_extras={"dd_cost": cd["best_cost"], "express_cost": 0.0,
                        "total_cost": cd["best_cost"]},
    )

    # CSVs exist + parse
    sched = pd.read_csv(paths["schedule_csv"])
    assert len(sched) == 8
    assert {"plz", "hub", "delivery_days"}.issubset(sched.columns)
    daily = pd.read_csv(paths["daily_volume_csv"])
    assert len(daily) == N_DAYS
    assert daily["weekday"].tolist() == WEEKDAYS

    fleet = pd.read_csv(paths["hub_fleet_csv"])
    assert len(fleet) == 2 * N_DAYS
    assert "vehicles" in fleet.columns

    # PNGs exist + nonzero
    for k in ("heatmap_png", "daily_volume_png", "cost_breakdown_png"):
        assert paths[k].exists() and paths[k].stat().st_size > 0

    # Summary JSON exists
    summary_path = tmp_path / "optimization_summary.json"
    assert summary_path.exists()
