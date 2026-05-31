"""Content-addressed stage cache.

Each pipeline stage is identified by a *cache key* derived from:

* the stage name (``"01_demand"``, ``"02_baseline"``, ...);
* a small, JSON-serialisable dict of *config inputs* (the subset of the
  :class:`PipelineConfig` that actually influences this stage);
* optional *upstream input hashes* (so changing stage 1 invalidates stage 2).

The cache lives at ``<cache_dir>/<stage_name>/<key>.pkl``. ``cache_dir``
defaults to ``results/cache/stages/`` but is configurable so tests can use
a fresh tmp dir.

Replaces the legacy ``FORCE_RECOMPUTE`` flag: callers either ``put`` a fresh
artefact or ``get`` a cached one. There is no global "force" toggle — instead
disable the cache per run via :class:`RunContext` (``use_cache=False``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _canonical_json(obj: Any) -> str:
    """Stable JSON encoding that survives pydantic models, Paths, sets, etc."""
    def default(o: Any) -> Any:
        if hasattr(o, "model_dump"):  # pydantic v2
            return o.model_dump(mode="json")
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        if hasattr(o, "tolist"):  # numpy
            return o.tolist()
        raise TypeError(f"non-serialisable type: {type(o).__name__}")

    return json.dumps(obj, sort_keys=True, default=default, separators=(",", ":"))


def cache_key(stage_name: str, *parts: Any) -> str:
    """Deterministic short hash from stage name + variadic input parts.

    Example::

        cache_key("02_baseline", config.routing, ["DHL"], upstream_hash)
    """
    payload = _canonical_json({"stage": stage_name, "parts": list(parts)})
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class StageCache:
    """Simple file-system cache for stage artefacts.

    Layout::

        <root>/<stage_name>/<key>.pkl
    """

    root: Path
    enabled: bool = True

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _path(self, stage_name: str, key: str) -> Path:
        return self.root / stage_name / f"{key}.pkl"

    def has(self, stage_name: str, key: str) -> bool:
        return self.enabled and self._path(stage_name, key).exists()

    def get(self, stage_name: str, key: str) -> Any | None:
        if not self.enabled:
            return None
        p = self._path(stage_name, key)
        if not p.exists():
            return None
        try:
            data = pickle.loads(p.read_bytes())  # noqa: S301 — local trusted files
        except Exception as exc:  # corrupt cache → recompute
            log.warning("StageCache: failed to load %s (%s); will recompute", p, exc)
            return None
        log.debug("StageCache hit %s/%s", stage_name, key)
        return data

    def put(self, stage_name: str, key: str, data: Any) -> Path:
        if not self.enabled:
            return Path()  # no-op
        p = self._path(stage_name, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        log.debug("StageCache put %s/%s (%.1f MB)",
                  stage_name, key, p.stat().st_size / 1_048_576)
        return p

    def clear(self, stage_name: str | None = None) -> int:
        """Remove cached artefacts; returns count deleted."""
        if not self.enabled:
            return 0
        target = self.root / stage_name if stage_name else self.root
        if not target.exists():
            return 0
        n = 0
        for p in target.rglob("*.pkl"):
            p.unlink(missing_ok=True)
            n += 1
        return n
