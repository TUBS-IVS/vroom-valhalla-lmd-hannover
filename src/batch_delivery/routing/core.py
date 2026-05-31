"""``batch_delivery.routing.core`` — backwards-compatible re-exports.

The implementation was split into focused submodules during the
2026-05-31 GitHub-ready refactor. This shim re-exports every
symbol so existing imports such as::

    from batch_delivery.routing.core import solve_single_plz

keep working. New code should import directly from the focused
submodule (``cache``, ``client``, ``requests``, ``solver``).
"""
from __future__ import annotations

# ruff: noqa: F401  (this file is a re-export shim — every name is "unused" here)

from batch_delivery.routing.cache import (
    _cache_path,
    _request_hash,
    load_cached_solution,
    save_cached_solution,
)
from batch_delivery.routing.client import (
    _check_valhalla_memory,
    _get_container_mem_mb,
    _health_check,
    _restart_container,
    _restart_vroom,
)
from batch_delivery.routing.requests import (
    _parse_unfound_loc,
    _split_points_kmeans,
    build_scenario_requests,
    build_vroom_jobs,
    build_vroom_vehicles,
    compute_baseline_job_caps,
)
from batch_delivery.routing.solver import (
    parse_routes,
    solve_scenario,
    solve_single_plz,
)

__all__ = [
    "build_scenario_requests",
    "build_vroom_jobs",
    "build_vroom_vehicles",
    "compute_baseline_job_caps",
    "load_cached_solution",
    "parse_routes",
    "save_cached_solution",
    "solve_scenario",
    "solve_single_plz",
]
