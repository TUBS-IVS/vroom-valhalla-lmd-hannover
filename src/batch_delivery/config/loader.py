"""YAML loader for :class:`PipelineConfig`.

Reads ``conf/default.yaml`` first, then deep-merges the requested scenario
file on top, then validates the result. Unknown keys are rejected by Pydantic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from batch_delivery.config import constants as C
from batch_delivery.config.schema import PipelineConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*; *override* wins on leaves."""
    out = dict(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping")
    return data


def load_config(
    path: str | Path | None = None,
    *,
    overlays: list[str | Path] | None = None,
) -> PipelineConfig:
    """Load and validate a pipeline configuration.

    Parameters
    ----------
    path
        Primary YAML file. Defaults to ``conf/default.yaml`` under the
        repository root.
    overlays
        Optional list of additional YAML files merged on top of *path* in
        order (each overrides keys from the previous one). Use this to layer
        scenario-specific or environment-specific overrides.

    Returns
    -------
    PipelineConfig
        Fully validated configuration object.
    """
    primary = Path(path) if path is not None else (C.CONF_DIR / "default.yaml")
    merged: dict[str, Any] = _read_yaml(primary)

    for ov in overlays or []:
        ov_path = Path(ov)
        merged = _deep_merge(merged, _read_yaml(ov_path))

    return PipelineConfig.model_validate(merged)
