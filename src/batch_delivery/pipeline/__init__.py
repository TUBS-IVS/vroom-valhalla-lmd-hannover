"""End-to-end pipeline orchestrator.

Ported from the legacy notebook (``archive/legacy_2026_05/``), the modern
ML-SA workflow used for the MobilTUM 2026 paper. The pipeline runs seven
stages in order:

    1. step_load_demand_and_hubs   HAGRID demand + hub assignment per provider
    2. step_solve_baseline          two-pass VROOM baseline (raw -> traffic)
    3. step_prepare_optimisation    per-provider data structures
    4. step_train_surrogate         5-seed MLP ensemble from baseline samples
    5. step_optimize                coordinate descent (Express + Batch-Only)
    6. step_solve_scenarios         VROOM resolve for every non-baseline scenario
    7. step_evaluate                KPIs, scenario comparison, CSV/HTML reports

Each stage takes a :class:`PipelineState`, mutates ``state.artefacts``,
and returns the same state. The :func:`run_all` entry point chains them.

The implementation was split into a package during the 2026-05-31
GitHub-ready refactor:

* :mod:`batch_delivery.pipeline.state`        — PipelineState container
* :mod:`batch_delivery.pipeline.stages`       — the seven step_* functions
* :mod:`batch_delivery.pipeline.orchestrator` — run_all

Existing ``from batch_delivery.pipeline import run_all, PipelineState`` calls
continue to work because every symbol is re-exported here.
"""
from __future__ import annotations

from batch_delivery.pipeline.orchestrator import run_all
from batch_delivery.pipeline.stages import (
    step_evaluate,
    step_load_demand_and_hubs,
    step_optimize,
    step_prepare_optimisation,
    step_solve_baseline,
    step_solve_scenarios,
    step_train_surrogate,
)
from batch_delivery.pipeline.state import PipelineState

__all__ = [
    "PipelineState",
    "run_all",
    "step_evaluate",
    "step_load_demand_and_hubs",
    "step_optimize",
    "step_prepare_optimisation",
    "step_solve_baseline",
    "step_solve_scenarios",
    "step_train_surrogate",
]
