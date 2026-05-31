"""Runtime helpers (caching, run context, logging, parallel execution).

This sub-package adds the minimal infrastructure that turns the linear
notebook-style pipeline into a reproducible experiment runner without pulling
in heavy dependencies (no Hydra, no Prefect).

Public API (kept small on purpose)::

    from batch_delivery.runtime import (
        RunContext, StageCache, configure_logging, ParallelMap,
    )
"""
from __future__ import annotations

from .cache import StageCache, cache_key
from .logging import configure_logging, get_run_logger
from .parallel import ParallelMap
from .run_context import RunContext

__all__ = [
    "ParallelMap",
    "RunContext",
    "StageCache",
    "cache_key",
    "configure_logging",
    "get_run_logger",
]
