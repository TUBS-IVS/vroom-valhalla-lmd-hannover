"""Configuration loader + schema tests."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from batch_delivery.config import load_config
from batch_delivery.config.loader import _deep_merge
from batch_delivery.config.schema import PipelineConfig

pytestmark = pytest.mark.unit


def test_default_config_loads() -> None:
    cfg = load_config()
    assert isinstance(cfg, PipelineConfig)
    assert cfg.optimization.max_holding_days == 3
    assert {s.name for s in cfg.scenarios} >= {"Baseline", "SA_ML + Express", "SA_ML Batch-Only"}


def test_overlay_merges(tmp_path: Path) -> None:
    overlay = tmp_path / "ovl.yaml"
    overlay.write_text("seed: 999\nrouting:\n  parallel_workers: 1\n", encoding="utf-8")
    cfg = load_config(overlays=[overlay])
    assert cfg.seed == 999
    assert cfg.routing.parallel_workers == 1
    # untouched values still come from defaults
    assert cfg.routing.vehicle_capacity == 230


def test_deep_merge_nested() -> None:
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    over = {"nested": {"y": 99, "z": 3}}
    out = _deep_merge(base, over)
    assert out == {"a": 1, "nested": {"x": 1, "y": 99, "z": 3}}


def test_schema_rejects_unknown_provider(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("providers: [Nobody]\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(overlays=[bad])


def test_schema_rejects_holding_days_other_than_three(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("optimization:\n  max_holding_days: 2\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(overlays=[bad])


def test_schema_rejects_non_monotonic_windows(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "routing:\n  delivery_window: [50000, 40000]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(overlays=[bad])


def test_cost_level_must_be_unit(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("cost_level: 1.5\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(overlays=[bad])
