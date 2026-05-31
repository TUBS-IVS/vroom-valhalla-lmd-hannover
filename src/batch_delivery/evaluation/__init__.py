"""KPI computation, scenario comparison, HTML reports.

Bulk implementation in :mod:`batch_delivery.evaluation.core`.
"""
from __future__ import annotations

from batch_delivery.evaluation.core import (
    compare_scenarios,
    compute_scenario_kpis,
    daily_fleet_profile,
    kpi_table_html,
    per_hub_kpis,
    plot_schedule_overview,
    save_kpis,
)

__all__ = [
    "compare_scenarios",
    "compute_scenario_kpis",
    "daily_fleet_profile",
    "kpi_table_html",
    "per_hub_kpis",
    "plot_schedule_overview",
    "save_kpis",
]
