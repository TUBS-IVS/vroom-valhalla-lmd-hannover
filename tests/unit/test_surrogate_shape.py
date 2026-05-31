"""Surrogate sanity tests — verify feature-count and ensemble constants."""
from __future__ import annotations

import pytest

from batch_delivery.features import ALL_COLS, TIER1_COLS, TIER2_COLS, TIER3_COLS
from batch_delivery.surrogate import (
    ENSEMBLE_SEEDS,
    INTERACTION_DEFS,
    N_BASE,
    N_COMBO,
    N_INTERACTION,
    N_LOG,
    SKEWED_COLS,
)

pytestmark = pytest.mark.unit


def test_feature_tier_sizes() -> None:
    assert len(TIER1_COLS) == 8
    assert len(TIER2_COLS) == 10
    assert len(TIER3_COLS) == 7
    assert len(ALL_COLS) == 25
    assert ALL_COLS == TIER1_COLS + TIER2_COLS + TIER3_COLS


def test_combo_feature_size() -> None:
    assert N_BASE == 25
    assert N_INTERACTION == 8
    assert N_LOG == 11
    assert N_COMBO == 44


def test_ensemble_seeds() -> None:
    assert ENSEMBLE_SEEDS == [42, 123, 456, 789, 2024]
    assert len(ENSEMBLE_SEEDS) == 5


def test_interaction_defs_shape() -> None:
    assert len(INTERACTION_DEFS) == N_INTERACTION
    for spec in INTERACTION_DEFS:
        assert len(spec) == 4   # (name, op, left, right) or similar tuple shape


def test_skewed_cols_count() -> None:
    assert len(SKEWED_COLS) == N_LOG == 11
