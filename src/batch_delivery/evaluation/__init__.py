"""KPI computation, scenario comparison, HTML reports.

Bulk implementation in :mod:`batch_delivery.evaluation.core`.
"""
from __future__ import annotations

from batch_delivery.evaluation.core import (  # noqa: F401
    compare_scenarios,
    compute_scenario_kpis,
    daily_fleet_profile,
    kpi_table_html,
    per_hub_kpis,
    plot_schedule_overview,
    save_kpis,
)

__all__ = [
    "compute_scenario_kpis",
    "compare_scenarios",
    "daily_fleet_profile",
    "per_hub_kpis",
    "kpi_table_html",
    "save_kpis",
    "plot_schedule_overview",
]
