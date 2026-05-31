"""Configuration sub-package.

Three responsibilities, each in its own file:

* :mod:`batch_delivery.config.constants`  — frozen literal constants (UPPER_SNAKE_CASE).
* :mod:`batch_delivery.config.schema`     — Pydantic v2 models describing every
  knob that may legitimately differ between scenario YAMLs.
* :mod:`batch_delivery.config.loader`     — ``load_config(path)`` reads a YAML
  file, deep-merges it onto the defaults, validates it, and returns a
  :class:`PipelineConfig`.
* :mod:`batch_delivery.config.validation` — module-level invariants that must
  hold *regardless* of the YAML (e.g. ``MAX_HOLDING_DAYS == 3`` and the
  derived feasible-pattern count). Imported eagerly by :mod:`constants` so the
  assertion fires the moment the package is imported.

Public re-exports: :func:`load_config`, :class:`PipelineConfig` and the most
frequently used constants.
"""
from __future__ import annotations

from batch_delivery.config.constants import (
    MAX_HOLDING_DAYS,
    N_DAYS,
    WEEKDAYS,
    PROVIDERS,
    SCENARIO_NAMES,
    EXPECTED_PATTERN_COUNT_K3,
)
from batch_delivery.config.loader import load_config
from batch_delivery.config.schema import (
    OptimizationConfig,
    PipelineConfig,
    RoutingConfig,
    ScenarioConfig,
    SurrogateConfig,
)

__all__ = [
    "MAX_HOLDING_DAYS",
    "N_DAYS",
    "WEEKDAYS",
    "PROVIDERS",
    "SCENARIO_NAMES",
    "EXPECTED_PATTERN_COUNT_K3",
    "load_config",
    "PipelineConfig",
    "OptimizationConfig",
    "RoutingConfig",
    "ScenarioConfig",
    "SurrogateConfig",
]
