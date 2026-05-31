"""Pydantic schema for the mass-routing sweep configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from batch_delivery.config import constants as C


class SweepConfig(BaseModel):
    """Configuration for the VROOM mass-routing training-data sweep."""

    # ── what to sweep over ──────────────────────────────────────────────
    providers: list[str] = Field(default_factory=lambda: ["DHL"])
    base_days: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    agg_ks: list[int] = Field(default_factory=lambda: [1, 2, 3])
    plzs: list[str] | None = None
    """Subset of PLZ codes; ``None`` means *all* PLZs that have demand."""

    scales: list[float] = Field(default_factory=lambda: [0.7, 1.0, 1.3])
    p_keeps: list[float] = Field(default_factory=lambda: [1.0, 0.9])
    noise_sigmas: list[float] = Field(default_factory=lambda: [0.0, 0.2])
    b2c_scales: list[float] = Field(default_factory=lambda: [1.0])
    """Independent multiplier for the B2C parcel segment (default identity).
    Combine with ``b2c_scales=[1.0, 1.5, 2.0]`` to simulate Christmas-style
    B2C surges while leaving B2B untouched."""
    b2b_scales: list[float] = Field(default_factory=lambda: [1.0])
    """Independent multiplier for the B2B parcel segment (default identity).
    Use to model business-week peaks or summer dips in B2B parcel flow."""
    seeds: list[int] = Field(default_factory=lambda: [42, 123, 456])

    # ── output ──────────────────────────────────────────────────────────
    out_dir: Path = Path("results/sweep")
    out_csv: str = "training_matrix.csv"
    out_parquet: str | None = "training_matrix.parquet"

    # ── execution ──────────────────────────────────────────────────────
    max_combinations: int | None = None
    """Hard cap on the number of (combination) rows to attempt — useful for
    smoke runs.  ``None`` means run the full grid."""

    shuffle_seed: int | None = None
    """If set, the enumerated combo list is randomly shuffled with this seed
    *before* ``max_combinations`` truncation. This yields a stratified random
    sample over the full grid (provider/day/PLZ/perturbation) instead of a
    head-of-list slice. Use it together with ``max_combinations`` for fast,
    balanced validation passes inside the oracle-loop."""

    priority_keys: list[tuple[str, int, str, int]] | None = None
    """Active-sampling hint: list of ``(provider, base_day, plz, agg_k)``
    tuples that should be routed first (placed before the random shuffle and
    therefore protected from ``max_combinations`` truncation). The oracle-loop
    uses this to prioritise (provider, day, PLZ) regions where the surrogate
    ensemble currently disagrees the most."""

    parallel_jobs: int = 1
    parallel_backend: str = "threading"
    """joblib backend.  Default ``threading`` because VROOM solving is HTTP-IO
    bound (GIL released during requests) and avoids large pickle overhead.
    Use ``loky`` for CPU-heavy custom feature work."""
    use_cache: bool = True
    progress: bool = True

    # ── filtering ──────────────────────────────────────────────────────
    min_parcels: int = 1
    """Skip combinations with total demand below this threshold."""
    min_stops: int = 2
    """Skip combinations with fewer than this many delivery points."""

    @field_validator("agg_ks")
    @classmethod
    def _agg_ks_positive(cls, v: list[int]) -> list[int]:
        if not v or any(k < 1 or k > C.MAX_HOLDING_DAYS + 1 for k in v):
            raise ValueError(
                f"agg_ks must be non-empty and within [1, {C.MAX_HOLDING_DAYS + 1}]"
            )
        return v

    @field_validator("scales")
    @classmethod
    def _scales_positive(cls, v: list[float]) -> list[float]:
        if not v or any(s <= 0 for s in v):
            raise ValueError("scales must be > 0")
        return v

    @field_validator("p_keeps")
    @classmethod
    def _p_keeps_unit(cls, v: list[float]) -> list[float]:
        if not v or any(not (0 < p <= 1) for p in v):
            raise ValueError("p_keeps must be in (0, 1]")
        return v

    @field_validator("noise_sigmas")
    @classmethod
    def _noise_nonneg(cls, v: list[float]) -> list[float]:
        if any(s < 0 for s in v):
            raise ValueError("noise_sigmas must be >= 0")
        return v

    @field_validator("b2c_scales", "b2b_scales")
    @classmethod
    def _segment_scales_nonneg(cls, v: list[float]) -> list[float]:
        if not v or any(s < 0 for s in v):
            raise ValueError("b2c_scales / b2b_scales must be non-empty and >= 0")
        return v

    @field_validator("base_days")
    @classmethod
    def _base_days_in_range(cls, v: list[int]) -> list[int]:
        if not v or any(d < 0 or d >= C.N_DAYS for d in v):
            raise ValueError(f"base_days entries must be in [0, {C.N_DAYS - 1}]")
        return v

    @field_validator("providers")
    @classmethod
    def _providers_known(cls, v: list[str]) -> list[str]:
        unknown = [p for p in v if p not in C.PROVIDERS]
        if unknown:
            raise ValueError(f"unknown providers: {unknown} (known: {C.PROVIDERS})")
        return v


def load_sweep_yaml(path: str | Path | None) -> SweepConfig:
    """Read a YAML config file into a :class:`SweepConfig`.

    ``None`` returns the schema defaults.
    """
    if path is None:
        return SweepConfig()
    p = Path(path)
    raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return SweepConfig(**raw)
