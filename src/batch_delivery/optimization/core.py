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

from batch_delivery.optimization.schedules import (
    enumerate_valid_schedules,
    _compute_wait_mx,
    build_fixed_schedules,
    build_hub_arrays,
)

from batch_delivery.optimization.costs import (
    build_cost_matrices,
    _hub_express_day,
    compute_scenario_corrections,
    build_cost_matrices_ml,
    _hub_express_day_ml,
)

from batch_delivery.optimization.simulated_annealing import (
    sa_optimize,
    _sa_optimize_ml_single,
    sa_optimize_ml,
    _sa_optimize_ml_LEGACY,
)

from batch_delivery.optimization.coordinate_descent import (
    optimize_cd_ml,
    _day_toggle_neighbors,
    _pair_polish_round,
)

from batch_delivery.optimization.balancing import (
    _daily_fleet_per_hub,
    _fleet_imbalance,
    balance_fleet_per_hub,
    balance_fleet_per_hub_ml,
    system_smooth_pass,
)

__all__ = [
    "enumerate_valid_schedules",
    "build_fixed_schedules",
    "build_hub_arrays",
    "build_cost_matrices",
    "compute_scenario_corrections",
    "build_cost_matrices_ml",
    "sa_optimize",
    "sa_optimize_ml",
    "optimize_cd_ml",
    "balance_fleet_per_hub",
    "balance_fleet_per_hub_ml",
    "system_smooth_pass",
]
