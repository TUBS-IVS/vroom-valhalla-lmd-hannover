"""Feature engineering for the cost surrogate.

44 columns total: 25 base + 8 interactions + 11 log-transforms.
Bulk implementation in :mod:`batch_delivery.features.core`.
"""
from __future__ import annotations

from batch_delivery.features.core import (  # noqa: F401
    _PROVIDER_IDX,
    ALL_COLS,
    TIER1_COLS,
    TIER2_COLS,
    TIER3_COLS,
    build_feature_matrix,
    compute_all_features,
    compute_tier1_features,
    compute_tier2_features,
    compute_tier3_features,
    provider_index,
)

__all__ = [
    "ALL_COLS",
    "TIER1_COLS",
    "TIER2_COLS",
    "TIER3_COLS",
    "build_feature_matrix",
    "compute_all_features",
    "compute_tier1_features",
    "compute_tier2_features",
    "compute_tier3_features",
    "provider_index",
]
