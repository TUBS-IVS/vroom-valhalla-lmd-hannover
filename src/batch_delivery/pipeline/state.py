"""Pipeline state container."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from batch_delivery.config import PipelineConfig
from batch_delivery.runtime import RunContext
from batch_delivery.utils import (
    get_logger,
)

log = get_logger(__name__, level=logging.INFO)




# ─── State container ────────────────────────────────────────────────────────


@dataclass
class PipelineState:
    """Mutable bag of intermediate artefacts produced by the pipeline.

    ``ctx`` is optional so notebooks/tests can still drive individual stages
    without booting a full :class:`RunContext`. ``run_all`` always creates one.
    """

    config: PipelineConfig
    out_dir: Path
    artefacts: dict[str, Any] = field(default_factory=dict)
    ctx: RunContext | None = None
