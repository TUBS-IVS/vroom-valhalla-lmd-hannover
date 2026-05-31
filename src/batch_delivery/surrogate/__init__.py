"""Cost surrogate: 5-seed MLP ensemble.

Bulk implementation in :mod:`batch_delivery.surrogate.core`.
"""
from __future__ import annotations

from batch_delivery.surrogate.core import (  # noqa: F401
    ENSEMBLE_SEEDS,
    INTERACTION_DEFS,
    MLCostPredictor,
    N_BASE,
    N_COMBO,
    N_INTERACTION,
    N_LOG,
    SKEWED_COLS,
    build_combo_features,
)
from batch_delivery.surrogate.train import (  # noqa: F401
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
    "MLCostPredictor",
    "ENSEMBLE_SEEDS",
    "INTERACTION_DEFS",
    "SKEWED_COLS",
    "N_BASE",
    "N_INTERACTION",
    "N_LOG",
    "N_COMBO",
    "build_combo_features",
    "TrainingData",
    "TARGET_COL",
    "load_training_data",
    "cross_validate",
    "train_full_model",
    "append_iteration_row",
]
