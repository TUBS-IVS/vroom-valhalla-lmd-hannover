"""Pipeline state container."""
from __future__ import annotations

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from batch_delivery.config import PipelineConfig, load_config
from batch_delivery.config.constants import (
    FAST_SHARE_B2B,
    FAST_SHARE_B2C,
    N_DAYS,
    PROVIDERS,
    RESULTS_DIR,
    SC_BASELINE,
    SC_FIXED_BATCH,
    SC_FIXED_EXPRESS,
    SC_SA_ML_BATCH,
    SC_SA_ML_EXPRESS,
    SCENARIO_NAMES,
    NON_BASELINE_SCENARIOS,
    EXPRESS_SCENARIOS,
    WEEKDAYS,
    provider_to_demand_prefix,
)
from batch_delivery.runtime import RunContext
from batch_delivery.utils import (
    compute_weighted_speed_factor,
    get_logger,
    load_checkpoint,
    save_checkpoint,
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
