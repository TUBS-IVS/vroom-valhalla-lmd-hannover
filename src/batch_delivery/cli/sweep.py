"""``batch-delivery sweep`` — parameter sweep driver."""
from __future__ import annotations

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from batch_delivery import __version__
from batch_delivery.config import load_config

from batch_delivery.cli._app import app, config_app  # noqa: F401


@app.command()
def sweep(
    path: Path = typer.Option(None, "--config", "-c", help="Sweep YAML file (defaults to conf/sweep_default.yaml)"),
    max_combinations: int = typer.Option(None, "--max", help="Cap on number of (combo) rows for smoke runs."),
    out_dir: Path = typer.Option(None, "--out-dir", help="Override output directory."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the VROOM solution cache."),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable the tqdm progress bar."),
    jobs: int = typer.Option(None, "--jobs", "-j", help="Parallel worker count (-1 = all cores). Overrides config."),
    backend: str = typer.Option(None, "--backend", help="joblib backend: threading (default, IO-bound) or loky."),
    providers: str = typer.Option(None, "--providers", help="Comma-separated provider list, e.g. 'DHL,Amazon,UPS'. Overrides config."),
) -> None:
    """Generate a VROOM-routed training matrix for the surrogate model."""
    from batch_delivery.runtime import RunContext
    from batch_delivery.sweep import load_sweep_yaml, run_sweep

    if path is None:
        default = Path("conf/sweep_default.yaml")
        path = default if default.exists() else None

    cfg = load_sweep_yaml(path)
    if max_combinations is not None:
        cfg = cfg.model_copy(update={"max_combinations": max_combinations})
    if out_dir is not None:
        cfg = cfg.model_copy(update={"out_dir": out_dir})
    if no_cache:
        cfg = cfg.model_copy(update={"use_cache": False})
    if no_progress:
        cfg = cfg.model_copy(update={"progress": False})
    if jobs is not None:
        cfg = cfg.model_copy(update={"parallel_jobs": jobs})
    if backend is not None:
        cfg = cfg.model_copy(update={"parallel_backend": backend})
    if providers is not None:
        prov_list = [p.strip() for p in providers.split(",") if p.strip()]
        cfg = cfg.model_copy(update={"providers": prov_list})

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext.create(
        runs_root=cfg.out_dir / "runs",
        cache_root=cfg.out_dir / "cache",
        config=cfg,
        use_cache=cfg.use_cache,
        parallel_jobs=cfg.parallel_jobs,
        run_name="sweep",
    )
    try:
        df = run_sweep(cfg, ctx=ctx)
        ctx.finalize(kpis={"n_rows": int(len(df))})
    except Exception:
        ctx.finalize(kpis={"_failed": True})
        raise

    typer.echo(f"sweep finished. run_id={ctx.run_id}")
    typer.echo(f"  rows:     {len(df):,}")
    typer.echo(f"  csv:      {cfg.out_dir / cfg.out_csv}")
    typer.echo(f"  manifest: {ctx.manifest_path}")
