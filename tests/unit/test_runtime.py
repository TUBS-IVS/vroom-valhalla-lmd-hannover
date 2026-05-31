"""Tests for ``batch_delivery.runtime`` (cache, run-context, parallel)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from batch_delivery.runtime import (
    ParallelMap,
    RunContext,
    StageCache,
    cache_key,
)


# ─── StageCache ────────────────────────────────────────────────────────────


def test_cache_key_is_deterministic():
    a = cache_key("01_demand", {"providers": ["DHL"]}, {"k": 1})
    b = cache_key("01_demand", {"providers": ["DHL"]}, {"k": 1})
    assert a == b
    assert len(a) == 16


def test_cache_key_changes_when_inputs_change():
    a = cache_key("01_demand", {"providers": ["DHL"]})
    b = cache_key("01_demand", {"providers": ["UPS"]})
    assert a != b


def test_cache_key_handles_paths_and_sets():
    # Sets and Paths must serialise; otherwise call would crash.
    k = cache_key("x", {"path": Path("/tmp/foo"), "days": {0, 1, 2}})
    assert isinstance(k, str)


def test_stage_cache_roundtrip(tmp_path: Path):
    cache = StageCache(root=tmp_path)
    cache.put("stage_a", "k1", {"hello": "world"})
    assert cache.has("stage_a", "k1")
    assert cache.get("stage_a", "k1") == {"hello": "world"}


def test_stage_cache_miss_returns_none(tmp_path: Path):
    cache = StageCache(root=tmp_path)
    assert cache.get("stage_a", "missing") is None
    assert not cache.has("stage_a", "missing")


def test_stage_cache_disabled(tmp_path: Path):
    cache = StageCache(root=tmp_path, enabled=False)
    cache.put("stage_a", "k1", {"hello": "world"})
    assert cache.get("stage_a", "k1") is None  # disabled = always miss
    assert not cache.has("stage_a", "k1")


def test_stage_cache_clear(tmp_path: Path):
    cache = StageCache(root=tmp_path)
    cache.put("a", "k1", 1)
    cache.put("a", "k2", 2)
    cache.put("b", "k3", 3)
    assert cache.clear("a") == 2
    assert cache.get("a", "k1") is None
    assert cache.get("b", "k3") == 3
    assert cache.clear() == 1


# ─── ParallelMap ───────────────────────────────────────────────────────────


def test_parallel_map_serial_when_n_jobs_1():
    pm = ParallelMap(n_jobs=1)
    assert pm.map(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]


def test_parallel_map_empty():
    pm = ParallelMap(n_jobs=4)
    assert pm.map(lambda x: x, []) == []


def _square(x):  # picklable for joblib's loky backend
    return x * x


def test_parallel_map_parallel_threading():
    pm = ParallelMap(n_jobs=2, backend="threading")
    out = pm.map(_square, [1, 2, 3, 4])
    assert sorted(out) == [1, 4, 9, 16]


# ─── RunContext ────────────────────────────────────────────────────────────


class _FakeConfig:
    """Minimal stand-in for PipelineConfig that supports model_dump."""

    def model_dump(self, mode: str = "json") -> dict:
        return {"providers": ["DHL"], "seed": 42}


def test_run_context_create_writes_files(tmp_path: Path):
    cfg = _FakeConfig()
    ctx = RunContext.create(
        config=cfg,
        runs_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
        run_name="unit",
        use_cache=True,
        parallel_jobs=1,
    )
    assert ctx.run_dir.exists()
    assert ctx.manifest_path.exists()
    assert (ctx.run_dir / "config.yaml").exists()
    assert ctx.jsonl_path.exists()  # log_event during create
    manifest = json.loads(ctx.manifest_path.read_text(encoding="utf-8"))
    assert manifest["config_hash"] == ctx.config_hash
    assert manifest["cache_enabled"] is True


def test_run_context_log_event_appends_jsonl(tmp_path: Path):
    ctx = RunContext.create(
        config=_FakeConfig(),
        runs_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
        run_name="logtest",
    )
    ctx.log_event("hello", foo=1, bar="baz")
    lines = ctx.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    parsed = [json.loads(line) for line in lines]
    events = [p for p in parsed if p.get("event") == "hello"]
    assert len(events) == 1
    assert events[0]["foo"] == 1
    assert events[0]["bar"] == "baz"


def test_run_context_finalize_records_kpis(tmp_path: Path):
    ctx = RunContext.create(
        config=_FakeConfig(),
        runs_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
        run_name="fintest",
    )
    ctx.finalize(kpis={"baseline_cost_eur": 12345.6})
    manifest = json.loads(ctx.manifest_path.read_text(encoding="utf-8"))
    assert manifest["kpis"] == {"baseline_cost_eur": 12345.6}
    assert manifest["finished_at"] is not None


def test_run_context_run_id_is_unique(tmp_path: Path):
    ids = {
        RunContext.create(
            config=_FakeConfig(),
            runs_root=tmp_path / "runs",
            cache_root=tmp_path / "cache",
            run_name=f"r{i}",
        ).run_id
        for i in range(3)
    }
    assert len(ids) == 3
