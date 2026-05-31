"""Lightweight logging configuration helper.

We deliberately avoid an external dependency (structlog/loguru) here so the
package keeps a small footprint. The standard library is enough for what we
need: a console handler at INFO and an optional JSONL file handler.

If a caller wants per-event structured records (run-level events, KPIs), they
should use ``RunContext.log_event`` which writes to the same JSONL file.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

_JSONL_HANDLERS: dict[Path, logging.Handler] = {}


class JsonlFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(jsonl_path: Path | None = None,
                      console_level: int = logging.INFO) -> None:
    """Idempotent logging setup.

    Parameters
    ----------
    jsonl_path : Path, optional
        If given, a file handler is attached that writes one JSON object per
        log record. Multiple calls with the same path are no-ops.
    console_level : int
        Level for the *console* handler (root logger stays at DEBUG so the
        file captures everything).
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # ── Console handler (re-use if already configured) ──────────────────
    have_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )
    if not have_console:
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s",
                                          datefmt="%H:%M:%S"))
        root.addHandler(ch)

    # ── Optional JSONL file handler ─────────────────────────────────────
    if jsonl_path is None:
        return
    jsonl_path = Path(jsonl_path)
    if jsonl_path in _JSONL_HANDLERS:
        return  # already attached
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(jsonl_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JsonlFormatter())
    root.addHandler(fh)
    _JSONL_HANDLERS[jsonl_path] = fh


def get_run_logger(name: str) -> logging.Logger:
    """Convenience wrapper for module loggers within a run."""
    return logging.getLogger(name)
