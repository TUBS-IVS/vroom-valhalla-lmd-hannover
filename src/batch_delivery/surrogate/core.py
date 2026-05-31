"""MLP Ensemble cost predictor for VRP schedule optimisation.

Wraps a 5-seed MLP Ensemble trained on 44 combo features (25 base +
8 interaction + 11 log-transform) to predict delivery cost from
Akkerman-style spatial & demand features.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from batch_delivery.features import ALL_COLS
from batch_delivery.utils import log


# ─────────────────────────────────────────────────────────────────────────────
# Combo feature recipe (must match calibration notebook §10k)
# ─────────────────────────────────────────────────────────────────────────────

INTERACTION_DEFS: list[tuple[str, str, str, str]] = [
    # (new_col, col_a, col_b, op)
    ("parcels_x_nn_dist",    "n_parcels",      "mean_nn_dist_km",  "mul"),
    ("vehicles_x_hub_dist",  "min_vehicles",    "hub_dist_km",      "mul"),
    ("stops_x_nn_dist",      "n_stops",         "mean_nn_dist_km",  "mul"),
    ("load_x_parcels",       "load_factor",     "n_parcels",        "mul"),
    ("parcels_x_hub_dist",   "n_parcels",       "hub_dist_km",      "mul"),
    ("stops_x_hub_dist",     "n_stops",         "hub_dist_km",      "mul"),
    ("aspect_x_area",        "aspect_ratio",    "ch_area_km2",      "mul"),
    ("demand_concentration", "max_stop_demand",  "n_stops",         "div"),
]

SKEWED_COLS: list[str] = [
    "n_parcels", "n_stops", "area_km2", "hub_dist_km",
    "ch_area_km2", "ch_perimeter_km", "max_stop_demand",
    "mean_nn_dist_km", "mean_inter_stop_dist_km",
    "max_hub_dist_km", "demand_std",
]

ENSEMBLE_SEEDS: list[int] = [42, 123, 456, 789, 2024]

# ─────────────────────────────────────────────────────────────────────────────
# Precomputed index maps for fast numpy-only combo feature construction
# (avoids DataFrame overhead in hot loops)
# ─────────────────────────────────────────────────────────────────────────────
_BASE_IDX: dict[str, int] = {c: i for i, c in enumerate(ALL_COLS)}

_INTERACTION_IDX: list[tuple[int, int, str]] = [
    (_BASE_IDX[col_a], _BASE_IDX[col_b], op)
    for _, col_a, col_b, op in INTERACTION_DEFS
]

_SKEWED_BASE_IDX: list[int] = [
    _BASE_IDX[c] for c in SKEWED_COLS if c in _BASE_IDX
]

N_BASE = len(ALL_COLS)                           # 25
N_INTERACTION = len(INTERACTION_DEFS)             # 8
N_LOG = len(_SKEWED_BASE_IDX)                     # 11
N_COMBO = N_BASE + N_INTERACTION + N_LOG          # 44


def _build_combo_array_1d(base25: np.ndarray) -> np.ndarray:
    """Build 44-element combo feature vector from 25 base features (no DataFrame)."""
    out = np.empty(N_COMBO, dtype=np.float64)
    out[:N_BASE] = base25
    off = N_BASE
    for ia, ib, op in _INTERACTION_IDX:
        if op == "mul":
            out[off] = base25[ia] * base25[ib]
        else:
            out[off] = base25[ia] / max(1.0, base25[ib])
        off += 1
    for idx in _SKEWED_BASE_IDX:
        out[off] = np.log1p(max(0.0, base25[idx]))
        off += 1
    return out


def build_combo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expand 25 base features → 44 combo features.

    Parameters
    ----------
    df : DataFrame with columns from ``ALL_COLS`` (25 cols).

    Returns
    -------
    DataFrame with 44 columns (base + interactions + log-transforms).
    """
    out = df.copy()

    # Interaction features
    for name, col_a, col_b, op in INTERACTION_DEFS:
        if op == "mul":
            out[name] = out[col_a] * out[col_b]
        else:  # div
            out[name] = out[col_a] / out[col_b].clip(lower=1)

    # Log-transforms
    for col in SKEWED_COLS:
        if col in out.columns:
            out[f"log_{col}"] = np.log1p(out[col].clip(lower=0))

    return out


class MLCostPredictor:
    """Ensemble of MLP Pipelines for VRP cost prediction.

    Each pipeline: StandardScaler → MLPRegressor.
    Prediction = mean over all seed models.
    """

    def __init__(
        self,
        pipelines: list[Pipeline],
        combo_cols: list[str],
        best_alpha: float,
        best_arch: tuple[int, ...],
    ) -> None:
        self.pipelines = pipelines
        self.combo_cols = combo_cols
        self.best_alpha = best_alpha
        self.best_arch = best_arch

    # ─── Prediction ──────────────────────────────────────────────────

    def predict(self, df_base: pd.DataFrame) -> np.ndarray:
        """Predict cost from a DataFrame of 25 base features.

        Parameters
        ----------
        df_base : DataFrame
            Must contain columns ``ALL_COLS`` (25 features).

        Returns
        -------
        ndarray, shape (n_samples,) — ensemble-averaged cost.
        """
        df_combo = build_combo_features(df_base[ALL_COLS])
        X = np.nan_to_num(
            df_combo[self.combo_cols].values.astype(np.float64),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        preds = np.stack([p.predict(X) for p in self.pipelines])
        return np.maximum(0, preds.mean(axis=0))

    def predict_array(self, X_combo: np.ndarray) -> np.ndarray:
        """Predict from pre-built combo feature array (44 cols)."""
        X = np.nan_to_num(X_combo.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        preds = np.stack([p.predict(X) for p in self.pipelines])
        return np.maximum(0, preds.mean(axis=0))

    def predict_with_variance(
        self, df_base: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) of the per-seed ensemble predictions.

        Used by active-sampling: high std indicates the ensemble disagrees
        and the surrogate is uncertain about that input region.
        """
        df_combo = build_combo_features(df_base[ALL_COLS])
        X = np.nan_to_num(
            df_combo[self.combo_cols].values.astype(np.float64),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        preds = np.stack([p.predict(X) for p in self.pipelines])  # (n_seeds, N)
        return np.maximum(0, preds.mean(axis=0)), preds.std(axis=0)

    def predict_single(self, base25: np.ndarray) -> float:
        """Predict cost from a single 25-element base feature vector (no DataFrame)."""
        combo = _build_combo_array_1d(base25)
        X = np.nan_to_num(combo.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
        preds = np.array([float(p.predict(X)[0]) for p in self.pipelines])
        return max(0.0, float(preds.mean()))

    # ─── Persistence ─────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialize to pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pipelines": self.pipelines,
            "combo_cols": self.combo_cols,
            "best_alpha": self.best_alpha,
            "best_arch": self.best_arch,
        }
        path.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        log.info("MLCostPredictor saved → %s (%.1f MB)",
                 path.name, path.stat().st_size / 1_048_576)

    @classmethod
    def load(cls, path: str | Path) -> "MLCostPredictor":
        """Load from pickle."""
        path = Path(path)
        data = pickle.loads(path.read_bytes())  # noqa: S301 — trusted local
        obj = cls(
            pipelines=data["pipelines"],
            combo_cols=data["combo_cols"],
            best_alpha=data["best_alpha"],
            best_arch=data["best_arch"],
        )
        log.info("MLCostPredictor loaded ← %s (%d pipelines)",
                 path.name, len(obj.pipelines))
        return obj

    @classmethod
    def train(
        cls,
        X_base: pd.DataFrame,
        y: np.ndarray,
        *,
        alpha: float = 0.0001,
        arch: tuple[int, ...] = (256, 128, 64, 32),
        seeds: list[int] | None = None,
    ) -> "MLCostPredictor":
        """Train a new ensemble on full data.

        Parameters
        ----------
        X_base : DataFrame with 25 base feature columns.
        y : ndarray, target cost values.
        alpha : L2 regularisation strength.
        arch : hidden layer sizes.
        seeds : random seeds for each ensemble member.
        """
        if seeds is None:
            seeds = ENSEMBLE_SEEDS

        df_combo = build_combo_features(X_base[ALL_COLS])
        combo_cols = list(df_combo.columns)
        X = np.nan_to_num(
            df_combo.values.astype(np.float64),
            nan=0.0, posinf=0.0, neginf=0.0,
        )

        pipelines = []
        for s in seeds:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=arch,
                    activation="relu",
                    max_iter=2000,
                    learning_rate="adaptive",
                    learning_rate_init=0.001,
                    early_stopping=True,
                    validation_fraction=0.15,
                    alpha=alpha,
                    random_state=s,
                )),
            ])
            pipe.fit(X, y)
            log.debug("Seed %d trained — loss %.4f", s, pipe[-1].loss_)
            pipelines.append(pipe)

        pred = np.stack([p.predict(X) for p in pipelines]).mean(axis=0)
        mape = float(np.mean(np.abs(pred - y) / np.maximum(1, np.abs(y)))) * 100
        log.info("Ensemble trained: %d seeds, MAPE=%.2f%% (in-sample)", len(seeds), mape)

        return cls(pipelines, combo_cols, alpha, arch)
