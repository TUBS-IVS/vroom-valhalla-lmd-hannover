"""Cost surrogate: 5-seed MLP ensemble.

Bulk implementation in :mod:`batch_delivery.surrogate.core`.
"""
from __future__ import annotations

from batch_delivery.surrogate.core import (
    ENSEMBLE_SEEDS,
    INTERACTION_DEFS,
    N_BASE,
    N_COMBO,
    N_INTERACTION,
    N_LOG,
    SKEWED_COLS,
    MLCostPredictor,
    build_combo_features,
)
from batch_delivery.surrogate.train import (
    TARGET_COL,
    TrainingData,
    append_iteration_row,
    cross_validate,
    load_training_data,
    train_full_model,
)
from batch_delivery.surrogate.tune import (  # noqa: F401
    DEFAULT_ALPHAS,
    DEFAULT_ARCHS,
    DEFAULT_LR_INITS,
    TrialConfig,
    build_search_space,
    tune_hyperparameters,
)
from batch_delivery.surrogate.validate import (  # noqa: F401
    append_to_training,
    classify_extremity,
    compute_feature_importance,
    split_extreme_holdout,
    summarize_training_data,
    validate_against_vroom,
)

__all__ = [
    "ENSEMBLE_SEEDS",
    "INTERACTION_DEFS",
    "N_BASE",
    "N_COMBO",
    "N_INTERACTION",
    "N_LOG",
    "SKEWED_COLS",
    "TARGET_COL",
    "MLCostPredictor",
    "TrainingData",
    "append_iteration_row",
    "build_combo_features",
    "cross_validate",
    "load_training_data",
    "train_full_model",
]
