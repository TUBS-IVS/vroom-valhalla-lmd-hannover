"""Routing sub-package — VROOM/Valhalla integration.

Bulk implementation lives in :mod:`batch_delivery.routing.core` (ported from
the legacy ``routing.py``).
"""
from __future__ import annotations

from batch_delivery.routing.core import (  # noqa: F401
    build_scenario_requests,
    build_vroom_jobs,
    build_vroom_vehicles,
    parse_routes,
    solve_scenario,
    solve_single_plz,
)

__all__ = [
    "build_scenario_requests",
    "build_vroom_jobs",
    "build_vroom_vehicles",
    "parse_routes",
    "solve_scenario",
    "solve_single_plz",
]
