"""Frozen literal constants — single source of truth.

This module **only** contains literal values and the trivial derived values
that never depend on a YAML.  No I/O, no logging, no heavy imports.

The bug-fix invariants live next to the constants so they fire at *import
time* — if anyone ever flips ``MAX_HOLDING_DAYS`` to ``2`` again, every
import of :mod:`batch_delivery` aborts loudly.

References to the underlying decision: see ``docs/REFACTOR_PLAN.md`` and the
paper section *Methodology — Scheduling task*.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]   # project root (../../../ from this file)
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_V2 = RESULTS_DIR / "v2"
CONF_DIR = ROOT / "conf"

# ─── Weekdays (Mon–Sat, no Sunday delivery in Germany) ──────────────────────
WEEKDAYS: list[str] = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
]
N_DAYS: int = len(WEEKDAYS)

# ─── Carriers / LSPs ────────────────────────────────────────────────────────
CARRIER_DAYS: dict[str, list[str]] = {
    "DHL":    ["Monday", "Wednesday", "Friday"],
    "Amazon": ["Tuesday", "Thursday", "Saturday"],
    "DPD":    ["Tuesday", "Friday"],
    "FedEx":  ["Monday", "Thursday"],
    "GLS":    ["Monday", "Thursday"],
    "Hermes": ["Tuesday", "Saturday"],
    "UPS":    ["Monday", "Thursday"],
}
CARRIER_FIXED_INDICES: dict[str, set[int]] = {
    k: {WEEKDAYS.index(d) for d in days}
    for k, days in CARRIER_DAYS.items()
}
PROVIDERS: list[str] = list(CARRIER_DAYS.keys())
PROVIDER_PREFIXES: dict[str, str] = {
    "Amazon": "ama", "DHL": "dhl", "DPD": "dpd", "FedEx": "fxt",
    "GLS": "gls", "Hermes": "her", "UPS": "ups",
}


def provider_to_demand_prefix(provider: str) -> str:
    """Map provider display name to HAGRID demand-column prefix."""
    return PROVIDER_PREFIXES.get(provider, provider.lower())


# ─── Scenario identifiers ───────────────────────────────────────────────────
SC_BASELINE = "Baseline"
SC_FIXED_EXPRESS = "Fixed + Express"
SC_SA_EXPRESS = "SA + Express"               # legacy SA path (kept for ablation)
SC_FIXED_BATCH = "Fixed Batch-Only"
SC_SA_BATCH = "SA Batch-Only"
SC_SA_ML_EXPRESS = "SA_ML + Express"          # paper's Scenario I
SC_SA_ML_BATCH = "SA_ML Batch-Only"           # paper's Scenario II

SCENARIO_NAMES: list[str] = [
    SC_BASELINE,
    SC_FIXED_EXPRESS, SC_SA_ML_EXPRESS,
    SC_FIXED_BATCH, SC_SA_ML_BATCH,
]
NON_BASELINE_SCENARIOS: list[str] = [
    SC_FIXED_EXPRESS, SC_SA_ML_EXPRESS, SC_FIXED_BATCH, SC_SA_ML_BATCH,
]
EXPRESS_SCENARIOS: set[str] = {SC_FIXED_EXPRESS, SC_SA_ML_EXPRESS}
BATCH_ONLY_SCENARIOS: set[str] = {SC_FIXED_BATCH, SC_SA_ML_BATCH}
SA_ML_SCENARIOS: set[str] = {SC_SA_ML_EXPRESS, SC_SA_ML_BATCH}

# ─── HAGRID express / fast share (sigmoid model) ────────────────────────────
def hagrid_fast_share(cost_level: float) -> tuple[float, float]:
    """B2C / B2B express shares from the HAGRID logistic preference model.

    Parameters
    ----------
    cost_level : float
        Operating point on the price axis ∈ [0, 1].
        ``0.00`` → basecase (daily); ``0.75`` → ``batchhigh``.
    """
    price = np.linspace(1, 10, 100)
    price_shifted = price - 1.25
    idx = min(int(len(price) * cost_level), len(price) - 1)
    p = price_shifted[idx]
    b2c = (10 + 90 / (1 + np.exp(1.5 * (p - 3)))) / 100
    b2b = (5 + 95 / (1 + np.exp(3.5 * (p - 1.75)))) / 100
    return float(b2c), float(b2b)


COST_LEVEL: float = 0.75   # ≙ HAGRID "batchhigh"
FAST_SHARE_B2C, FAST_SHARE_B2B = hagrid_fast_share(COST_LEVEL)

# ─── Batch delivery — the bug-fix invariant ─────────────────────────────────
#
# A parcel may be held at most three calendar days before delivery.
# This was previously documented as "two" in older paper drafts; the code
# value of ``3`` is authoritative (see docs/REFACTOR_PLAN.md §Risks).
#
MAX_HOLDING_DAYS: int = 3

# Number of feasible weekly delivery patterns under the holding constraint
# (computed once below; pinned as a constant so any drift fails loudly).
def _enumerate_pattern_count(n_days: int, max_gap: int) -> int:
    """Count cyclic delivery-day subsets with max gap ≤ max_gap. Pure helper."""
    import itertools
    min_freq = max(2, math.ceil(n_days / max_gap))
    n = 0
    for size in range(min_freq, n_days + 1):
        for combo in itertools.combinations(range(n_days), size):
            days_sorted = list(combo)
            ok = True
            for i in range(len(days_sorted)):
                gap = (days_sorted[(i + 1) % len(days_sorted)] - days_sorted[i]) % n_days
                if gap == 0:
                    gap = n_days
                if gap > max_gap:
                    ok = False
                    break
            if ok:
                n += 1
    return n


EXPECTED_PATTERN_COUNT_K3: int = _enumerate_pattern_count(N_DAYS, MAX_HOLDING_DAYS)
"""Number of feasible weekly delivery patterns under MAX_HOLDING_DAYS=3."""

# ─── Service quality / waiting time ─────────────────────────────────────────
MAX_AVG_WAITING_DAYS: float = 2.0
WAITING_PENALTY_EUR: float = 0.0   # Disabled in the surrogate-based objective

# ─── Monte Carlo demand simulation (legacy) ─────────────────────────────────
MC_ITERATIONS: int = 100
MC_SEED: int = 42

# ─── Fixed DHL delivery schedule (comparison scenario) ──────────────────────
FIXED_DHL_SCHEDULE: list[str] = ["Monday", "Wednesday", "Friday"]

# ─── VROOM VRP configuration ────────────────────────────────────────────────
VEHICLE_CAPACITY: int = 230            # parcels per vehicle (Van >2 t)
FIXED_COST_EUR: float = 189.15         # EUR/day per vehicle (incl. 8 h labor)
COST_PER_KM_EUR: float = 0.3864        # EUR/km operating cost
COST_PER_HOUR_EUR: float = 0.0         # EUR/h (labor included in fixed cost)
COST_SCALE: int = 100                  # VROOM integer costs → cents
FIXED_COST_CENTS: int = int(round(FIXED_COST_EUR * COST_SCALE))
COST_PER_KM_CENTS: int = int(round(COST_PER_KM_EUR * COST_SCALE))
COST_PER_HOUR_CENTS: int = int(round(COST_PER_HOUR_EUR * COST_SCALE))

# ─── Service time model ─────────────────────────────────────────────────────
SERVICE_TIME_PER_PARCEL: int = 120     # seconds/parcel (2 min)
SERVICE_TIME_CAP: int = 1200           # max 20 min per stop
PROFILE: str = "auto"

# ─── VROOM / Valhalla API ───────────────────────────────────────────────────
VROOM_API_URL: str = "http://localhost:3000"

# ─── Time windows ───────────────────────────────────────────────────────────
COMMUTE_BUFFER: int = 3600                          # 1 h hub→area buffer
DELIVERY_WINDOW: list[int] = [28800, 72000]         # 08:00–20:00
VEHICLE_TIME_WINDOW: list[int] = [
    DELIVERY_WINDOW[0] - COMMUTE_BUFFER,            # 07:00
    DELIVERY_WINDOW[1] + COMMUTE_BUFFER,            # 21:00
]

# ─── Break (§4 ArbZG: ≥30 min after 6 h) ───────────────────────────────────
BREAK_DURATION: int = 1800                          # 30 min
BREAK_WINDOW: list[int] = [39600, 50400]            # 11:00–14:00

# ─── Staggered start times ──────────────────────────────────────────────────
VEH_START_SPREAD_S: int = 1800                      # σ = 30 min half-normal
VEH_START_LATEST: int = 43200                       # 12:00 latest departure
VEH_START_SEED: int = 42

# ─── Hourly traffic factors ─────────────────────────────────────────────────
TRAFFIC_FACTORS: dict[int, float] = {
    0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00, 4: 1.00, 5: 1.00,
    6: 1.10, 7: 1.30, 8: 1.35, 9: 1.25, 10: 1.15, 11: 1.15,
    12: 1.20, 13: 1.20, 14: 1.20, 15: 1.25, 16: 1.35, 17: 1.35,
    18: 1.25, 19: 1.10, 20: 1.00, 21: 1.00, 22: 1.00, 23: 1.00,
}
_veh_start_h = VEHICLE_TIME_WINDOW[0] // 3600
_veh_end_h = min(VEHICLE_TIME_WINDOW[1] // 3600, 23)
_window_factors = [TRAFFIC_FACTORS[h] for h in range(_veh_start_h, _veh_end_h)]
AVG_TRAFFIC_FACTOR: float = round(sum(_window_factors) / len(_window_factors), 4)
SPEED_FACTOR: float = round(1.0 / AVG_TRAFFIC_FACTOR, 3)

# ─── Hub assignment ─────────────────────────────────────────────────────────
LARGE_HUB_TYPES: set[str] = {"ZB", "PZ/ZB", "BZ"}
SMALL_HUB_DISTANCE_PENALTY: float = 1.5
SMALL_HUB_DELAY: int = 3600                         # ZSP starts 1 h later
MIN_PLZ_PER_ZSP: int = 2

# ─── Depot capacity limits ──────────────────────────────────────────────────
DEPOT_DAILY_CAP_SMALL: int = 5000

# ─── Vehicle provisioning ───────────────────────────────────────────────────
MAX_VEHICLES_PER_REQUEST: int = 50

# ─── Solver settings ────────────────────────────────────────────────────────
PARALLEL_WORKERS: int = 8
SOLVE_WORKERS: int = 8
HTTP_CONNECT_TIMEOUT: int = 15                      # connect phase ceiling
HTTP_TIMEOUT: int = 600                             # 10 min read ceiling per VROOM request
PER_JOB_BUDGET_S: int = 1200                        # 20 min total per PLZ (all retries together)
MAX_JOBS_PER_REQUEST: int = 500
MAX_RETRIES: int = 5
CONNECTION_RETRIES: int = 3
MAX_UNASSIGNED_RETRIES: int = 3
OUTLIER_THRESHOLD: int = 450
MIN_PLZ_JOBS_MERGE: int = 75

# ─── Daganzo VRP proxy (legacy ablation) ───────────────────────────────────
BHH_CONSTANT: float = 0.7124
LINE_HAUL_SPEED_KMH: float = 50.0

# ─── Simulated annealing (legacy ablation) ─────────────────────────────────
SA_ITERATIONS: int = 300_000
SA_T_INIT: float = 5000.0
SA_T_MIN: float = 1.0
SA_ALPHA: float = 0.99998
SA_SEED: int = 42

# ─── Fleet balancing post-processing ────────────────────────────────────────
FLEET_BALANCE_MAX_SWAPS: int = 5000

# ─── Delivery frequency (baseline) ──────────────────────────────────────────
DELIVERY_FREQUENCY: int = N_DAYS
BATCH_FACTOR: float = 1.0

# ─── Available work time (derived) ──────────────────────────────────────────
AVAILABLE_WORK_S: float = float(
    VEHICLE_TIME_WINDOW[1] - VEHICLE_TIME_WINDOW[0] - BREAK_DURATION
)  # 48 600 s = 13.5 h

# ─── Checkpoint control ─────────────────────────────────────────────────────
FORCE_RECOMPUTE: bool = False   # was True; flipped to use stage 1-5 caches after worst-schedule bugfix

# ─── Eager invariant check ─────────────────────────────────────────────────
# Imported here so any import of the package fires the assertion.
from batch_delivery.config import validation as _validation  # noqa: E402,F401
_validation.assert_invariants(
    max_holding_days=MAX_HOLDING_DAYS,
    n_days=N_DAYS,
    expected_pattern_count=EXPECTED_PATTERN_COUNT_K3,
)
