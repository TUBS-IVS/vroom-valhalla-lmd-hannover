"""I/O sub-package: HAGRID demand, hub registry."""
from __future__ import annotations

from batch_delivery.io import demand, hubs
from batch_delivery.io.demand import (
    compute_shifted_demand_plz,
    get_source_days,
    load_daily_demand,
    merge_source_points,
    prepare_plz_data,
)
from batch_delivery.io.hubs import (
    assign_plz_to_hubs,
    enforce_depot_capacity,
    enforce_zsp_min_plz,
    load_hubs,
)

__all__ = [
    "assign_plz_to_hubs",
    "compute_shifted_demand_plz",
    "demand",
    "enforce_depot_capacity",
    "enforce_zsp_min_plz",
    "get_source_days",
    "hubs",
    "load_daily_demand",
    "load_hubs",
    "merge_source_points",
    "prepare_plz_data",
]
