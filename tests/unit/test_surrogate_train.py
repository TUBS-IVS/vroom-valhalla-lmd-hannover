"""Unit tests for :mod:`batch_delivery.surrogate.train`.

These tests use very small synthetic data so they run quickly without
requiring a real sweep CSV.  They cover:

* :func:`load_training_data` — reads CSVs, dedupes, drops failed rows
* :func:`cross_validate` — group-aware k-fold returns sensible metrics
* :func:`append_iteration_row` — creates + appends history CSV
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from batch_delivery.features import ALL_COLS
from batch_delivery.surrogate.train import (
    append_iteration_row,
    cross_validate,
    load_training_data,
)


def _synthetic_row(
    plz: str, seed: int, *, baseline: bool = False, scale: float = 1.0,
) -> dict:
    rng = np.random.default_rng(hash((plz, seed)) % (2**32))
    feats = {c: float(rng.uniform(0.5, 5.0)) for c in ALL_COLS}
    cost = 50.0 + 0.4 * feats["n_parcels"] + 0.3 * feats["hub_dist_km"] * scale
    cost += rng.normal(0, 1.0)
    return {
        **feats,
        "provider": "DHL",
        "plz": plz,
        "base_day": 1,
        "agg_k": 1,
        "scale": scale,
        "p_keep": 1.0,
        "noise_sigma": 0.0,
        "seed": seed,
        "is_baseline": baseline,
        "actual_cost_eur": float(cost),
        "actual_distance_km": float(cost * 0.5),
        "actual_duration_h": float(cost / 30.0),
        "actual_n_routes": 2,
        "n_vehicles_planned": 3,
        "solve_time_s": 0.1,
        "vroom_status": 0,
    }


def _synthetic_df(n_plz: int = 12, n_seeds: int = 4) -> pd.DataFrame:
    # Inflate base features so n_parcels actually varies between PLZ
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n_plz):
        plz = f"301{i:02d}"
        plz_scale = 1.0 + i * 0.4  # vary cost between groups
        # one identity / baseline row per plz
        rows.append(_synthetic_row(plz, seed=42, baseline=True, scale=1.0))
        for s in range(n_seeds):
            rows.append(_synthetic_row(plz, seed=100 + s, scale=plz_scale))
    df = pd.DataFrame(rows)
    # Make n_parcels strongly group-dependent so models can learn something
    df["n_parcels"] = df.groupby("plz")["n_parcels"].transform(
        lambda s: float(rng.uniform(50, 500))
    )
    df["actual_cost_eur"] = (
        50 + 0.4 * df["n_parcels"] + rng.normal(0, 5, size=len(df))
    )
    return df


def test_load_training_data_dedupes_and_drops_failed(tmp_path: Path) -> None:
    df = _synthetic_df(n_plz=4, n_seeds=2)
    # add a duplicate row + a failed (NaN cost) row
    dup = df.iloc[[0]].copy()
    failed = df.iloc[[1]].copy()
    failed["actual_cost_eur"] = np.nan
    df_in = pd.concat([df, dup, failed], ignore_index=True)
    path = tmp_path / "matrix.csv"
    df_in.to_csv(path, index=False)

    td = load_training_data(path)
    # original len was len(df) + dup + failed; both must drop
    assert len(td.y) == len(df)
    assert td.X.shape[1] == len(ALL_COLS)
    assert td.is_baseline.sum() == 4  # one baseline per plz
    assert (td.y > 0).all()


def test_cross_validate_returns_per_fold_and_overall(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    df = _synthetic_df(n_plz=10, n_seeds=4)
    path = tmp_path / "matrix.csv"
    df.to_csv(path, index=False)
    td = load_training_data(path)

    # tiny arch + few seeds so CI stays fast
    metrics = cross_validate(
        td, k=3, arch=(8,), alpha=1e-3, seeds=[0, 1], group_aware=True,
    )
    assert metrics["k"] == 3
    assert "GroupKFold" in metrics["cv_kind"]
    assert len(metrics["fold_metrics"]) == 3
    o = metrics["overall"]
    assert np.isfinite(o["mae"]) and o["mae"] > 0
    assert np.isfinite(o["mape"])
    # baseline subset must have been measured
    assert metrics["n_baseline"] == 10
    assert np.isfinite(metrics["baseline_overall"]["mae"])


def test_append_iteration_row_creates_then_appends(tmp_path: Path) -> None:
    history = tmp_path / "history.csv"
    metrics = {
        "n_rows": 100, "n_groups": 10, "n_baseline": 10,
        "cv_kind": "GroupKFold(group=plz, k=5)", "k": 5,
        "overall": {"mae": 12.3, "mape": 4.5, "r2": 0.9},
        "baseline_overall": {"mae": 10.0, "mape": 3.5, "r2": 0.92},
    }
    append_iteration_row(history, iteration=1, metrics=metrics,
                         extra={"data_csv": "a.csv"})
    append_iteration_row(history, iteration=2, metrics=metrics,
                         extra={"data_csv": "b.csv"})

    df = pd.read_csv(history)
    assert len(df) == 2
    assert list(df["iteration"]) == [1, 2]
    assert df["mae_eur"].iloc[0] == pytest.approx(12.3)
    assert df["data_csv"].iloc[1] == "b.csv"
