"""Mass-routing sweep runner — generates training data for the surrogate.

Iterates over (provider, base-day, multi-day-aggregation, PLZ, demand-scale,
stop-dropout, per-stop-noise, seed), solves each combination via VROOM,
and writes a feature/cost matrix for the surrogate model.

Public API:
    SweepConfig           — pydantic config
    load_sweep_yaml       — read yaml + apply defaults
    run_sweep             — main driver (DataFrame in/out)
    perturb_demand        — single deterministic demand perturbation
    aggregate_days        — sum HAGRID demand across consecutive base days
"""
from __future__ import annotations

from batch_delivery.sweep.config import SweepConfig, load_sweep_yaml
from batch_delivery.sweep.perturb import (
    aggregate_days,
    enumerate_combinations,
    perturb_demand,
)
from batch_delivery.sweep.runner import run_sweep

__all__ = [
    "SweepConfig",
    "aggregate_days",
    "enumerate_combinations",
    "load_sweep_yaml",
    "perturb_demand",
    "run_sweep",
]
