"""``batch_delivery.optimization.core`` — backwards-compatible re-exports.

The implementation was split into focused submodules during the 2026-05-31
GitHub-ready refactor. This module re-exports every public and private
symbol so that existing imports such as::

    from batch_delivery.optimization.core import build_cost_matrices_ml

continue to work without modification. New code should import directly
from the focused submodule (``schedules``, ``costs``, ``simulated_annealing``,
``coordinate_descent``, ``balancing``).
"""
from __future__ import annotations

# ruff: noqa: F401  (this file is a re-export shim — every name is "unused" here)

from batch_delivery.optimization.balancing import (
    _daily_fleet_per_hub,
    _fleet_imbalance,
    balance_fleet_per_hub,
    balance_fleet_per_hub_ml,
    system_smooth_pass,
)
from batch_delivery.optimization.coordinate_descent import (
    _day_toggle_neighbors,
    _pair_polish_round,
    optimize_cd_ml,
)
from batch_delivery.optimization.costs import (
    _hub_express_day,
    _hub_express_day_ml,
    _hub_express_vehicles,
    _hub_smallday_pool_ml,
    _pool_affected_days,
    build_cost_matrices,
    build_cost_matrices_ml,
    compute_scenario_corrections,
)
from batch_delivery.optimization.schedules import (
    _compute_wait_mx,
    build_fixed_schedules,
    build_hub_arrays,
    enumerate_valid_schedules,
)
from batch_delivery.optimization.simulated_annealing import (
    _sa_optimize_ml_LEGACY,
    _sa_optimize_ml_single,
    sa_optimize,
    sa_optimize_ml,
)

__all__ = [
    "balance_fleet_per_hub",
    "balance_fleet_per_hub_ml",
    "build_cost_matrices",
    "build_cost_matrices_ml",
    "build_fixed_schedules",
    "build_hub_arrays",
    "compute_scenario_corrections",
    "enumerate_valid_schedules",
    "optimize_cd_ml",
    "sa_optimize",
    "sa_optimize_ml",
    "system_smooth_pass",
]
