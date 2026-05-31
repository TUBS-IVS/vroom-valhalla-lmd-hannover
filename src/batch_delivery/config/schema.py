"""Pydantic v2 schema for YAML-driven scenario configuration.

The YAML file is the public knob surface; everything else flows from it.
:mod:`batch_delivery.config.constants` provides the *defaults*; this schema
validates the merged result and rejects malformed scenario files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from batch_delivery.config import constants as C


# ─── Sub-schemas ────────────────────────────────────────────────────────────


class RoutingConfig(BaseModel):
    """VROOM/Valhalla connection + solver parameters."""

    vroom_url: str = C.VROOM_API_URL
    http_timeout: int = C.HTTP_TIMEOUT
    max_retries: int = C.MAX_RETRIES
    connection_retries: int = C.CONNECTION_RETRIES
    max_unassigned_retries: int = C.MAX_UNASSIGNED_RETRIES
    max_jobs_per_request: int = C.MAX_JOBS_PER_REQUEST
    max_vehicles_per_request: int = C.MAX_VEHICLES_PER_REQUEST
    parallel_workers: int = C.PARALLEL_WORKERS

    vehicle_capacity: int = C.VEHICLE_CAPACITY
    fixed_cost_eur: float = C.FIXED_COST_EUR
    cost_per_km_eur: float = C.COST_PER_KM_EUR

    delivery_window: list[int] = Field(default_factory=lambda: list(C.DELIVERY_WINDOW))
    vehicle_time_window: list[int] = Field(default_factory=lambda: list(C.VEHICLE_TIME_WINDOW))
    break_duration: int = C.BREAK_DURATION
    break_window: list[int] = Field(default_factory=lambda: list(C.BREAK_WINDOW))

    @field_validator("vehicle_capacity", "fixed_cost_eur", "cost_per_km_eur")
    @classmethod
    def _positive(cls, v):
        if v <= 0:
            raise ValueError(f"value must be > 0, got {v}")
        return v

    @model_validator(mode="after")
    def _windows_monotonic(self) -> "RoutingConfig":
        if self.delivery_window[0] >= self.delivery_window[1]:
            raise ValueError("delivery_window must be strictly increasing")
        if self.vehicle_time_window[0] >= self.vehicle_time_window[1]:
            raise ValueError("vehicle_time_window must be strictly increasing")
        return self


class SurrogateConfig(BaseModel):
    """5-seed MLP ensemble + benchmark settings."""

    seeds: list[int] = Field(default_factory=lambda: [42, 123, 456, 789, 2024])
    hidden_layers: list[int] = Field(default_factory=lambda: [512, 512, 256, 128])
    alpha: float = 0.001
    max_iter: int = 2000
    early_stopping: bool = True
    cv_folds: int = 5
    benchmark_models: list[Literal["lr", "rf", "gbm", "mlp_single", "mlp_ensemble", "daganzo"]] = (
        Field(default_factory=lambda: ["lr", "rf", "gbm", "mlp_single", "mlp_ensemble", "daganzo"])
    )


class OptimizationConfig(BaseModel):
    """Optimisation kernel parameters."""

    max_holding_days: int = C.MAX_HOLDING_DAYS
    optimizer: Literal["coordinate_descent", "simulated_annealing"] = "coordinate_descent"
    cd_max_passes: int = 50
    cd_random_restarts: int = 20
    sa_iterations: int = C.SA_ITERATIONS
    sa_t_init: float = C.SA_T_INIT
    sa_alpha: float = C.SA_ALPHA
    sa_seed: int = C.SA_SEED
    fleet_balance_enabled: bool = True
    fleet_balance_max_swaps: int = C.FLEET_BALANCE_MAX_SWAPS
    fleet_balance_cost_budget: float = 0.01   # ≤ 1 % cost increase

    # Pair-PLZ-Move polish on top of single-PLZ coordinate descent.
    # Default-on for SA_ML scenarios — never worsens cost (only accepts
    # negative-Δ moves) and typically yields a 0.5–2 % improvement at
    # negligible cost on express scenarios where PLZ are coupled at the hub.
    pair_polish: bool = True
    pair_polish_rounds: int = 2
    pair_polish_max_pairs: int = 200

    @field_validator("max_holding_days")
    @classmethod
    def _holding_must_be_three(cls, v: int) -> int:
        # The schema does not allow YAMLs to override the bug-fix invariant.
        if v != 3:
            raise ValueError(
                f"max_holding_days must be 3 (bug-fix invariant); got {v}. "
                f"If you genuinely want a different value, change it in "
                f"batch_delivery.config.constants AND update validation.py."
            )
        return v


class ScenarioConfig(BaseModel):
    """Definition of one scenario the pipeline should solve."""

    name: str
    kind: Literal["baseline", "fixed_express", "sa_ml_express", "fixed_batch", "sa_ml_batch"]
    description: str = ""
    enabled: bool = True


# ─── Top-level ──────────────────────────────────────────────────────────────


class PipelineConfig(BaseModel):
    """Top-level configuration loaded from YAML."""

    seed: int = 42
    cost_level: float = C.COST_LEVEL
    providers: list[str] = Field(default_factory=lambda: list(C.PROVIDERS))

    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    surrogate: SurrogateConfig = Field(default_factory=SurrogateConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)

    scenarios: list[ScenarioConfig] = Field(default_factory=list)

    data_dir: Path = C.DATA_DIR
    results_dir: Path = C.RESULTS_V2

    # ── Runtime knobs (added by the minimal-invasive refactor) ──────────
    # Where the content-addressed stage cache lives. Defaults to a sub-folder
    # of results_dir so a clean ``rm -rf results/`` wipes both runs and cache.
    cache_dir: Path | None = None
    # >=1 process workers for the few stages that parallelise (joblib.Parallel).
    # ``1`` keeps everything serial and is the safe default for unit tests.
    parallel_jobs: int = 1
    # Optional human-friendly name for this run; appended to the run-id.
    run_name: str | None = None

    # Optional PLZ subset (5-digit strings). If set, demand & hubs are filtered
    # to this list. Used by integration smoke tests to keep run-time minimal.
    subset_plz: list[str] | None = None

    @field_validator("cost_level")
    @classmethod
    def _cost_level_unit(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"cost_level must be in [0, 1]; got {v}")
        return v

    @field_validator("providers")
    @classmethod
    def _known_providers(cls, providers: list[str]) -> list[str]:
        unknown = set(providers) - set(C.PROVIDERS)
        if unknown:
            raise ValueError(f"unknown provider(s): {sorted(unknown)}")
        return providers
