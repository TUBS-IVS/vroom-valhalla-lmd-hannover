"""Unit tests for SweepConfig validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from batch_delivery.sweep.config import SweepConfig, load_sweep_yaml


def test_defaults_are_valid():
    cfg = SweepConfig()
    assert "DHL" in cfg.providers
    assert cfg.base_days == [0, 1, 2, 3, 4, 5]
    assert all(s > 0 for s in cfg.scales)


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        SweepConfig(providers=["NotARealProvider"])


def test_invalid_base_day_rejected():
    with pytest.raises(ValueError):
        SweepConfig(base_days=[6])  # week is Mon..Sat = 0..5
    with pytest.raises(ValueError):
        SweepConfig(base_days=[-1])


def test_invalid_scale_rejected():
    with pytest.raises(ValueError):
        SweepConfig(scales=[0.0])
    with pytest.raises(ValueError):
        SweepConfig(scales=[-1.0])


def test_invalid_p_keep_rejected():
    with pytest.raises(ValueError):
        SweepConfig(p_keeps=[0.0])
    with pytest.raises(ValueError):
        SweepConfig(p_keeps=[1.5])


def test_invalid_noise_sigma_rejected():
    with pytest.raises(ValueError):
        SweepConfig(noise_sigmas=[-0.1])


def test_load_sweep_yaml_returns_defaults_for_none():
    cfg = load_sweep_yaml(None)
    assert isinstance(cfg, SweepConfig)


def test_load_sweep_yaml_reads_file(tmp_path: Path):
    yml = tmp_path / "sweep.yaml"
    yml.write_text(
        """
providers: [DHL, Amazon]
base_days: [0, 1]
agg_ks: [1, 2]
scales: [1.0]
p_keeps: [1.0]
noise_sigmas: [0.1]
seeds: [42]
out_dir: results/sweep_test
max_combinations: 5
""",
        encoding="utf-8",
    )
    cfg = load_sweep_yaml(yml)
    assert cfg.providers == ["DHL", "Amazon"]
    assert cfg.max_combinations == 5
    assert cfg.out_dir == Path("results/sweep_test")


def test_parallel_defaults():
    cfg = SweepConfig()
    assert cfg.parallel_jobs == 1
    assert cfg.parallel_backend == "threading"


def test_parallel_overrides_via_yaml(tmp_path: Path):
    yml = tmp_path / "p.yaml"
    yml.write_text(
        "parallel_jobs: 4\nparallel_backend: loky\n", encoding="utf-8"
    )
    cfg = load_sweep_yaml(yml)
    assert cfg.parallel_jobs == 4
    assert cfg.parallel_backend == "loky"
