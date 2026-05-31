"""Schedule enumeration + waiting-time helpers.

This module is the smallest, most stable layer of the optimisation
package. Anything that needs the canonical 39 weekly patterns or the
average waiting-time matrix imports from here.
"""

import itertools
import math

import numpy as np
import pandas as pd

from batch_delivery.config.constants import (
    CARRIER_FIXED_INDICES,
    MAX_HOLDING_DAYS,
    N_DAYS,
)
from batch_delivery.utils import log

# ─────────────────────────────────────────────────────────────────────────────
# Valid schedule enumeration
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_valid_schedules() -> list[frozenset[int]]:
    """Return all delivery-day subsets satisfying the holding-day constraint.

    A schedule is valid iff the maximum gap between consecutive delivery
    days (cyclic) does not exceed ``MAX_HOLDING_DAYS``.
    """
    min_freq = max(2, math.ceil(N_DAYS / MAX_HOLDING_DAYS))
    valid: list[frozenset[int]] = []
    for size in range(min_freq, N_DAYS + 1):
        for combo in itertools.combinations(range(N_DAYS), size):
            days_sorted = list(combo)
            ok = True
            for i in range(len(days_sorted)):
                gap = (
                    days_sorted[(i + 1) % len(days_sorted)] - days_sorted[i]
                ) % N_DAYS
                if gap == 0:
                    gap = N_DAYS
                if gap > MAX_HOLDING_DAYS:
                    ok = False
                    break
            if ok:
                valid.append(frozenset(combo))
    log.debug(f"Valid schedules: {len(valid)} (min freq {min_freq})")
    return valid




# ─────────────────────────────────────────────────────────────────────────────
# Vectorised waiting-time helper (shared by Daganzo & ML matrices)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_wait_mx(
    sched_active: np.ndarray,
    schedules: list[frozenset[int]],
    daily_demand: np.ndarray,
    daily_b2c: np.ndarray,
    daily_b2b: np.ndarray,
    fast_share_b2c: float,
    fast_share_b2b: float,
) -> np.ndarray:
    """Vectorised waiting-time matrix: shape (n_plz, n_sched).

    For each (PLZ, schedule) pair, computes the demand-weighted average
    number of waiting days for batch (non-express) parcels that arrive
    on non-delivery days.
    """
    n_sched = len(schedules)

    # Wait-days per (schedule, day): distance to next delivery day
    wait_per_day_mx = np.zeros((n_sched, N_DAYS), dtype=np.float64)
    for si, sched in enumerate(schedules):
        for d in range(N_DAYS):
            if d in sched:
                continue
            w = 0
            check = (d + 1) % N_DAYS
            while check != d:
                w += 1
                if check in sched:
                    break
                check = (check + 1) % N_DAYS
            wait_per_day_mx[si, d] = w

    # Express parcels per day (removed from batch waiting)
    expr_day = (
        np.round(daily_b2c * fast_share_b2c)
        + np.round(daily_b2b * fast_share_b2b)
    )  # (n_plz, N_DAYS)

    # Batch demand = total - express (on non-delivery days; delivery-day
    # contributions are zeroed by wait_per_day_mx == 0 on those days)
    batch_demand = daily_demand - expr_day  # (n_plz, N_DAYS)

    # Total parcels per PLZ (all days, schedule-independent)
    total_p = daily_demand.sum(axis=1)  # (n_plz,)

    # Weighted wait: (n_plz, 1, N_DAYS) * (1, n_sched, N_DAYS) → sum axis=2
    total_w = (batch_demand[:, None, :] * wait_per_day_mx[None, :, :]).sum(axis=2)

    return total_w / np.maximum(1.0, total_p[:, None])




# ─────────────────────────────────────────────────────────────────────────────
# Fixed-schedule helper
# ─────────────────────────────────────────────────────────────────────────────

def build_fixed_schedules(
    plz_keys: list[str],
    carrier: str = "DHL",
) -> dict[str, set[int]]:
    """Build fixed delivery schedules for all PLZ from CARRIER_DAYS."""
    days = CARRIER_FIXED_INDICES.get(carrier) or CARRIER_FIXED_INDICES["DHL"]
    return {pc: set(days) for pc in plz_keys}




# ─────────────────────────────────────────────────────────────────────────────
# Hub assignment arrays (utility for SA setup)
# ─────────────────────────────────────────────────────────────────────────────

def build_hub_arrays(
    plz_keys: list[str],
    df_assignments: pd.DataFrame,
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    """Build PLZ-to-hub index array and per-hub PLZ lists.

    Returns
    -------
    (plz_hub_arr, hub_plz_list, hub_names)
    """
    plz_to_hub: dict[str, str] = {}
    for pc in plz_keys:
        hr = df_assignments[df_assignments["plz"] == pc]
        if len(hr) > 0:
            plz_to_hub[pc] = hr.iloc[0]["hub_name"]
    hub_names = sorted(set(plz_to_hub.values()))
    hub_idx = {h: i for i, h in enumerate(hub_names)}
    n_hubs = len(hub_names)
    plz_hub_arr = np.array(
        [hub_idx.get(plz_to_hub.get(p, ""), 0) for p in plz_keys]
    )
    hub_plz_list = [np.where(plz_hub_arr == hi)[0] for hi in range(n_hubs)]
    return plz_hub_arr, hub_plz_list, hub_names
