"""LightGBM-logT surrogate with the same predict()/predict_single() interface
as MLCostPredictor, so it can be dropped into stage 5 of the pipeline.

Use:
    from batch_delivery.surrogate.lgb_adapter import LGBLogTSurrogate
    pred = LGBLogTSurrogate.load('path/to/production_lgb_logT_v1.pkl')
    state.artefacts['ml_predictor'] = pred
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from batch_delivery.features import ALL_COLS
from batch_delivery.surrogate.core import (
    _build_combo_array_1d,
    build_combo_features,
)


class LGBLogTSurrogate:
    """Drop-in replacement for MLCostPredictor backed by a LightGBM-logT model.

    The underlying model is a TransformedTargetRegressor(LGBMRegressor) trained
    on 44 combo features (25 base + 8 interaction + 11 log-transform). Predictions
    are guaranteed non-negative.

    Attributes
    ----------
    model : TransformedTargetRegressor
        The fitted LightGBM-logT pipeline.
    combo_cols : list[str]
        Names of the 44 combo features in the order expected by the model.
    best_alpha : float
        Sentinel for the MLP API (not used by LGB, always 0.0).
    best_arch : tuple
        Sentinel for the MLP API (always empty tuple).
    """

    def __init__(self, model, combo_cols: list[str]) -> None:
        self.model = model
        self.combo_cols = combo_cols
        self.best_alpha = 0.0
        self.best_arch: tuple = ()
        self.pipelines = [model]  # MLCostPredictor-style len() probe

    # ── prediction (matches MLCostPredictor.predict) ────────────────────────
    def predict(self, df_base: pd.DataFrame) -> np.ndarray:
        df_combo = build_combo_features(df_base[ALL_COLS])
        X = np.nan_to_num(df_combo[self.combo_cols].values.astype(np.float64),
                            nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(0.0, self.model.predict(X))

    def predict_single(self, base25: np.ndarray) -> float:
        combo = _build_combo_array_1d(base25)
        X = np.nan_to_num(combo.reshape(1, -1), nan=0.0, posinf=0.0, neginf=0.0)
        return float(np.maximum(0.0, self.model.predict(X)[0]))

    def predict_with_variance(self, df_base: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """LGB is single-seed so std=0. Returned for API compatibility only."""
        mean = self.predict(df_base)
        return mean, np.zeros_like(mean)

    # ── persistence ────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "combo_cols": self.combo_cols,
                          "kind": "LGBLogTSurrogate"}, f)

    @classmethod
    def load(cls, path: str | Path) -> LGBLogTSurrogate:
        """Load from either the train_production_lgb.py pickle (dict format)
        or from a save()-produced pickle. Both are supported."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "model" in data and "feature_cols" in data:
            return cls(model=data["model"], combo_cols=data["feature_cols"])
        if isinstance(data, dict) and "model" in data and "combo_cols" in data:
            return cls(model=data["model"], combo_cols=data["combo_cols"])
        raise ValueError(f"Unrecognised pickle format at {path}")
