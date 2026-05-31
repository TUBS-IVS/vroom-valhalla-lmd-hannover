"""``batch_delivery.routing.core`` — backwards-compatible re-exports.

The implementation was split into focused submodules during the
2026-05-31 GitHub-ready refactor. This shim re-exports every
symbol so existing imports such as::

    from batch_delivery.routing.core import solve_single_plz

keep working. New code should import directly from the focused
submodule (``cache``, ``client``, ``requests``, ``solver``).
"""
from __future__ import annotations

from batch_delivery.routing.cache import (
    _request_hash,
    _cache_path,
    load_cached_solution,
    save_cached_solution,
)

from batch_delivery.routing.client import (
    _health_check,
    _get_container_mem_mb,
    _restart_container,
    _restart_vroom,
    _check_valhalla_memory,
)

from batch_delivery.routing.requests import (
    compute_baseline_job_caps,
    _split_points_kmeans,
    build_vroom_jobs,
    build_vroom_vehicles,
    _parse_unfound_loc,
    build_scenario_requests,
)

from batch_delivery.routing.solver import (
    solve_single_plz,
    solve_scenario,
    parse_routes,
)

__all__ = [
    "load_cached_solution",
    "save_cached_solution",
    "compute_baseline_job_caps",
    "build_vroom_jobs",
    "build_vroom_vehicles",
    "build_scenario_requests",
    "solve_single_plz",
    "solve_scenario",
    "parse_routes",
]
