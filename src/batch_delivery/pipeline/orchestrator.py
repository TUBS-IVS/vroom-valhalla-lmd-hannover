"""Pipeline orchestrator: ``run_all`` chains the seven stages."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from batch_delivery.config import load_config
from batch_delivery.runtime import RunContext
from batch_delivery.utils import (
    get_logger,
)

log = get_logger(__name__, level=logging.INFO)

from batch_delivery.pipeline.state import (
    PipelineState,
)
from batch_delivery.pipeline.stages import (
    step_evaluate,
    step_load_demand_and_hubs,
    step_optimize,
    step_prepare_optimisation,
    step_solve_baseline,
    step_solve_scenarios,
    step_train_surrogate,
)

# Canonical pipeline stage order. The orchestrator iterates this list so
# that re-ordering or skipping stages is a single edit here. Defined in
# orchestrator.py rather than stages.py to keep the stages module a pure
# library of step functions with no run-order opinion.
PIPELINE_STAGES = [
    step_load_demand_and_hubs,
    step_solve_baseline,
    step_prepare_optimisation,
    step_train_surrogate,
    step_optimize,
    step_solve_scenarios,
    step_evaluate,
]


def run_all(
    config_path: str | Path | None = None,
    *,
    use_cache: bool = True,
    parallel_jobs: int | None = None,
    run_name: str | None = None,
) -> PipelineState:
    """Execute every stage in order.

    Parameters
    ----------
    config_path : str | Path | None
        YAML config (defaults to ``conf/default.yaml``).
    use_cache : bool
        Forwarded to :class:`RunContext`. ``False`` mimics the legacy
        ``FORCE_RECOMPUTE=True`` behaviour for the *new* stage cache; legacy
        pickle checkpoints (``results/checkpoints``) are still controlled
        by :data:`batch_delivery.config.constants.FORCE_RECOMPUTE`.
    parallel_jobs : int | None
        Override ``cfg.parallel_jobs``.
    run_name : str | None
        Override ``cfg.run_name``.

    Returns
    -------
    PipelineState
        Always populated; ``state.ctx`` holds the RunContext.
    """
    cfg = load_config(config_path)
    out_dir = Path(cfg.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build runtime context (run_id, manifest, JSONL, stage cache).
    runs_root = out_dir / "runs"
    cache_root = Path(cfg.cache_dir) if cfg.cache_dir else (out_dir / "cache" / "stages")
    ctx = RunContext.create(
        config=cfg,
        runs_root=runs_root,
        cache_root=cache_root,
        run_name=run_name or cfg.run_name,
        use_cache=use_cache,
        parallel_jobs=parallel_jobs if parallel_jobs is not None else cfg.parallel_jobs,
    )
    state = PipelineState(config=cfg, out_dir=out_dir, ctx=ctx)

    log.info("pipeline.run_all: run_id=%s out_dir=%s providers=%s",
             ctx.run_id, out_dir, cfg.providers)
    ctx.log_event("pipeline_start", out_dir=str(out_dir), providers=list(cfg.providers))

    t_total = time.perf_counter()
    try:
        for stage in PIPELINE_STAGES:
            t0 = time.perf_counter()
            ctx.log_event("stage_start", stage=stage.__name__)
            state = stage(state)
            dt = time.perf_counter() - t0
            log.info("  ✓ %s — %.1fs", stage.__name__, dt)
            ctx.log_event("stage_done", stage=stage.__name__, duration_s=round(dt, 2))
    except Exception as exc:  # capture failure in manifest before re-raising
        ctx.log_event("pipeline_failed", error=type(exc).__name__, message=str(exc))
        ctx.finalize(kpis={"_failed": True})
        raise

    total_s = time.perf_counter() - t_total
    log.info("pipeline.run_all: done in %.1fs", total_s)

    # Pull headline KPIs into the manifest if stage 7 produced them.
    kpis: dict[str, Any] = {"total_runtime_s": round(total_s, 2)}
    df_kpi = state.artefacts.get("df_kpi")
    if df_kpi is not None and "cost_eur" in df_kpi.columns:
        try:
            df_reset = df_kpi.reset_index() if df_kpi.index.name else df_kpi
            label_col = "scenario" if "scenario" in df_reset.columns else df_reset.columns[0]
            kpis["scenario_cost_eur"] = {
                str(row[label_col]): float(row["cost_eur"])
                for _, row in df_reset.iterrows()
            }
        except Exception:  # don't fail run on a manifest-format hiccup
            pass
    ctx.finalize(kpis=kpis)
    return state
