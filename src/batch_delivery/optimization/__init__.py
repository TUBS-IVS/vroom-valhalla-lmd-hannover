"""Optimisation sub-package.

Bulk implementation in :mod:`batch_delivery.optimization.core`.
"""
from __future__ import annotations

from batch_delivery.config.constants import EXPECTED_PATTERN_COUNT_K3
from batch_delivery.config.validation import assert_runtime_pattern_count
from batch_delivery.optimization.core import (  # noqa: F401
    balance_fleet_per_hub,
    balance_fleet_per_hub_ml,
    build_cost_matrices,
    build_cost_matrices_ml,
    build_fixed_schedules,
    build_hub_arrays,
    compute_scenario_corrections,
    enumerate_valid_schedules,
    optimize_cd_ml,
    sa_optimize,
    sa_optimize_ml,
)

_n_patterns = len(enumerate_valid_schedules())
assert_runtime_pattern_count(_n_patterns, EXPECTED_PATTERN_COUNT_K3)
del _n_patterns

__all__ = [
    "enumerate_valid_schedules",
    "optimize_cd_ml",
    "sa_optimize_ml",
    "sa_optimize",
    "balance_fleet_per_hub_ml",
    "balance_fleet_per_hub",
    "build_cost_matrices_ml",
    "build_cost_matrices",
    "build_fixed_schedules",
    "build_hub_arrays",
    "compute_scenario_corrections",
]
