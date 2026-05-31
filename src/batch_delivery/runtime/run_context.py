"""Per-run context: identity, paths, manifest, cache, parallel handle.

A :class:`RunContext` is created once at the start of ``pipeline.run_all`` and
passed (via :class:`PipelineState`) into every stage. It captures *what was
run* in a way that survives across processes and is easy to compare:

    results/runs/<run_id>/
        ├── manifest.json   — git_sha, started_at, finished_at, config_hash, kpis
        ├── config.yaml     — fully resolved configuration
        └── run.jsonl       — structured per-event log

The legacy ``results/`` outputs (``scenario_comparison_kpis.csv`` etc.) are
*also* written so existing notebooks/scripts keep working — the per-run dir is
additive.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .cache import StageCache, _canonical_json
from .logging import configure_logging
from .parallel import ParallelMap

log = logging.getLogger(__name__)


def _git_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _git_dirty(repo_root: Path) -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root, stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return None


def _make_run_id(name: str | None) -> str:
    ts = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    suffix = name.replace(" ", "_") if name else os.urandom(2).hex()
    return f"{ts}_{suffix}"


@dataclass
class RunContext:
    """Lightweight handle used by every pipeline stage.

    Construct via :meth:`create`. The stages should treat this as read-only
    apart from :meth:`log_event` and :meth:`record_kpi` (both append-only).
    """

    run_id: str
    run_dir: Path
    cache: StageCache
    parallel: ParallelMap
    git_sha: str | None = None
    git_dirty: bool | None = None
    started_at: str = ""
    config_dump: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    _events: list[dict[str, Any]] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------ ctor
    @classmethod
    def create(
        cls,
        config: Any,
        runs_root: Path,
        cache_root: Path,
        *,
        run_name: str | None = None,
        use_cache: bool = True,
        parallel_jobs: int = 1,
        repo_root: Path | None = None,
    ) -> RunContext:
        run_id = _make_run_id(run_name)
        run_dir = Path(runs_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        config_dump = (
            config.model_dump(mode="json") if hasattr(config, "model_dump")
            else dict(config)
        )
        config_hash = hashlib.sha256(
            _canonical_json(config_dump).encode()
        ).hexdigest()[:16]

        repo = repo_root or Path.cwd()
        cache = StageCache(root=Path(cache_root), enabled=use_cache)
        parallel = ParallelMap(n_jobs=parallel_jobs)

        ctx = cls(
            run_id=run_id,
            run_dir=run_dir,
            cache=cache,
            parallel=parallel,
            git_sha=_git_sha(repo),
            git_dirty=_git_dirty(repo),
            started_at=datetime.now(tz=UTC).isoformat(),
            config_dump=config_dump,
            config_hash=config_hash,
        )
        # Configure JSONL logging into the run dir.
        configure_logging(ctx.run_dir / "run.jsonl")
        # Persist config + initial manifest.
        ctx._write_config_yaml()
        ctx._write_manifest()
        ctx.log_event("run_started", run_id=run_id, config_hash=config_hash)
        return ctx

    # ------------------------------------------------------------------ paths
    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def jsonl_path(self) -> Path:
        return self.run_dir / "run.jsonl"

    # ------------------------------------------------------------------ I/O
    def _write_config_yaml(self) -> None:
        try:
            import yaml
            (self.run_dir / "config.yaml").write_text(
                yaml.safe_dump(self.config_dump, sort_keys=False),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Failed to write config.yaml: %s", exc)

    def _write_manifest(self, finished_at: str | None = None,
                        kpis: dict[str, Any] | None = None) -> None:
        manifest = {
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "config_hash": self.config_hash,
            "cache_enabled": self.cache.enabled,
            "parallel_jobs": self.parallel.n_jobs,
            "kpis": kpis or {},
            "extra": self.extra,
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )

    # ------------------------------------------------------------------ logging
    def log_event(self, event: str, **fields: Any) -> None:
        rec = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        self._events.append(rec)
        try:
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception as exc:  # never let logging crash the pipeline
            log.debug("log_event write failed: %s", exc)
        log.info("[%s] %s %s", self.run_id, event,
                 {k: v for k, v in fields.items() if k not in ("traceback",)})

    def finalize(self, kpis: dict[str, Any] | None = None) -> None:
        finished_at = datetime.now(tz=UTC).isoformat()
        self._write_manifest(finished_at=finished_at, kpis=kpis)
        self.log_event("run_finished", finished_at=finished_at)
