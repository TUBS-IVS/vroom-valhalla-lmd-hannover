"""Training / cross-validation utilities for :class:`MLCostPredictor`.

This module wraps the sweep-CSV → trained-model pipeline with:

* **Group-aware k-fold CV** (default group=plz) so we measure how well the
  surrogate generalises to *unseen* delivery areas — not just unseen seeds.
* **Per-fold + ensemble metrics**: MAE, MAPE, R², plus a baseline-only
  subset metric (the unperturbed HAGRID rows) which serves as the
  "ground truth" generalisation gauge.
* **Iteration history**: append-only CSV of metrics over time so the user
  can watch the learning curve as more sweep data accumulates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from batch_delivery.features import ALL_COLS
from batch_delivery.surrogate.core import (
    ENSEMBLE_SEEDS,
    MLCostPredictor,
)
from batch_delivery.utils import log

TARGET_COL = "actual_cost_eur"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class TrainingData:
    X: pd.DataFrame                # 25 base feature columns
    y: np.ndarray                  # target cost [EUR]
    groups: np.ndarray             # group key per row (default = plz)
    is_baseline: np.ndarray        # boolean mask of unperturbed HAGRID rows
    df_full: pd.DataFrame          # full original frame (kept for diagnostics)


def load_training_data(
    csv_path: Path | str,
    *,
    group_col: str = "plz",
    extra_csvs: list[Path] | None = None,
    drop_failed: bool = True,
) -> TrainingData:
    """Read one (or several) sweep CSVs and return a :class:`TrainingData`.

    ``extra_csvs`` are concatenated before deduplication so iteration runs
    can pass a list of cumulative training-matrix slices.  Duplicate rows
    are dropped on the cache-tag-equivalent key
    ``(provider, plz, base_day, agg_k, scale, p_keep, noise_sigma, seed)``.
    """
    paths: list[Path] = [Path(csv_path)]
    if extra_csvs:
        paths.extend(Path(p) for p in extra_csvs)

    frames = [pd.read_csv(p) for p in paths if Path(p).exists()]
    if not frames:
        raise FileNotFoundError(f"No training CSVs found among: {paths}")
    df = pd.concat(frames, ignore_index=True)

    if drop_failed:
        before = len(df)
        df = df.dropna(subset=[TARGET_COL])
        df = df[df[TARGET_COL] > 0].reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            log.info(f"train: dropped {dropped} rows with missing/zero target")

    key = ["provider", "plz", "base_day", "agg_k",
           "scale", "p_keep", "noise_sigma", "seed"]
    df = df.drop_duplicates(subset=key, keep="last").reset_index(drop=True)

    missing = [c for c in ALL_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"training CSV missing feature columns: {missing}")
    if group_col not in df.columns:
        raise KeyError(f"group column '{group_col}' not in training CSV")

    X = df[ALL_COLS].copy()
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    groups = df[group_col].astype(str).to_numpy()
    is_baseline = (
        df["is_baseline"].astype(bool).to_numpy()
        if "is_baseline" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    return TrainingData(
        X=X, y=y, groups=groups, is_baseline=is_baseline, df_full=df,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE / MAPE / R² on a single prediction array."""
    if len(y_true) == 0:
        return {"mae": float("nan"), "mape": float("nan"), "r2": float("nan")}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / np.maximum(1.0, np.abs(y_true)))) * 100
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "mape": mape, "r2": r2}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(
    data: TrainingData,
    *,
    k: int = 5,
    arch: tuple[int, ...] = (256, 128, 64, 32),
    alpha: float = 1e-4,
    seeds: list[int] | None = None,
    group_aware: bool = True,
) -> dict[str, Any]:
    """Run k-fold CV and return aggregated + per-fold metrics.

    ``group_aware`` uses :class:`GroupKFold` so that all rows of a given
    group (default = same PLZ) end up in the same fold — measures
    generalisation to *unseen* areas instead of unseen seeds.
    """
    seeds = seeds or ENSEMBLE_SEEDS
    n = len(data.y)
    if n < k * 2:
        raise ValueError(
            f"need at least {k*2} rows for {k}-fold CV, got {n}"
        )

    if group_aware and len(np.unique(data.groups)) >= k:
        splitter = GroupKFold(n_splits=k)
        split_iter = splitter.split(data.X, data.y, groups=data.groups)
        cv_kind = f"GroupKFold(group=plz, k={k})"
    else:
        splitter = KFold(n_splits=k, shuffle=True, random_state=42)
        split_iter = splitter.split(data.X, data.y)
        cv_kind = f"KFold(k={k}, shuffle=True)"
    log.info(f"train: cross-validating with {cv_kind}, n={n} rows")

    fold_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    oof_pred = np.full(n, np.nan, dtype=np.float64)

    for i, (tr, te) in enumerate(split_iter):
        model = MLCostPredictor.train(
            data.X.iloc[tr],
            data.y[tr],
            alpha=alpha,
            arch=arch,
            seeds=seeds,
        )
        y_hat = model.predict(data.X.iloc[te])
        oof_pred[te] = y_hat

        m = _metrics(data.y[te], y_hat)
        m.update(fold=i, n_train=len(tr), n_test=len(te))
        fold_rows.append(m)
        log.info(
            f"train: fold {i+1}/{k} — n_train={len(tr)} n_test={len(te)} "
            f"MAE={m['mae']:.1f}€ MAPE={m['mape']:.2f}% R²={m['r2']:.3f}"
        )

        # Baseline-only diagnostics on this fold
        bl_mask = data.is_baseline[te]
        if bl_mask.any():
            bm = _metrics(data.y[te][bl_mask], y_hat[bl_mask])
            bm.update(fold=i, n=int(bl_mask.sum()))
            base_rows.append(bm)

    overall = _metrics(data.y, oof_pred)
    baseline_mask_all = data.is_baseline & ~np.isnan(oof_pred)
    bl_overall = (
        _metrics(data.y[baseline_mask_all], oof_pred[baseline_mask_all])
        if baseline_mask_all.any()
        else {"mae": float("nan"), "mape": float("nan"), "r2": float("nan")}
    )

    return {
        "cv_kind": cv_kind,
        "k": k,
        "n_rows": n,
        "n_groups": len(np.unique(data.groups)),
        "n_baseline": int(data.is_baseline.sum()),
        "fold_metrics": fold_rows,
        "overall": overall,
        "baseline_overall": bl_overall,
        "baseline_per_fold": base_rows,
        "oof_pred": oof_pred,
    }


# ---------------------------------------------------------------------------
# Train + persist
# ---------------------------------------------------------------------------

def train_full_model(
    data: TrainingData,
    *,
    arch: tuple[int, ...] = (256, 128, 64, 32),
    alpha: float = 1e-4,
    seeds: list[int] | None = None,
) -> MLCostPredictor:
    """Train a final model on *all* data (no held-out split)."""
    seeds = seeds or ENSEMBLE_SEEDS
    return MLCostPredictor.train(
        data.X, data.y, alpha=alpha, arch=arch, seeds=seeds,
    )


# ---------------------------------------------------------------------------
# Iteration-history bookkeeping
# ---------------------------------------------------------------------------

def append_iteration_row(
    history_csv: Path | str,
    *,
    iteration: int,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one row of metrics to a learning-curve CSV.

    Creates the file (with header) on first call.
    """
    history_csv = Path(history_csv)
    history_csv.parent.mkdir(parents=True, exist_ok=True)

    overall = metrics.get("overall", {})
    baseline = metrics.get("baseline_overall", {})
    row = {
        "iteration": int(iteration),
        "n_rows": int(metrics.get("n_rows", 0)),
        "n_groups": int(metrics.get("n_groups", 0)),
        "n_baseline": int(metrics.get("n_baseline", 0)),
        "cv_kind": str(metrics.get("cv_kind", "")),
        "k": int(metrics.get("k", 0)),
        "mae_eur": float(overall.get("mae", float("nan"))),
        "mape_pct": float(overall.get("mape", float("nan"))),
        "r2": float(overall.get("r2", float("nan"))),
        "mae_eur_baseline": float(baseline.get("mae", float("nan"))),
        "mape_pct_baseline": float(baseline.get("mape", float("nan"))),
        "r2_baseline": float(baseline.get("r2", float("nan"))),
    }
    if extra:
        row.update(extra)

    df_row = pd.DataFrame([row])
    if history_csv.exists():
        df_row.to_csv(history_csv, mode="a", header=False, index=False)
    else:
        df_row.to_csv(history_csv, mode="w", header=True, index=False)
