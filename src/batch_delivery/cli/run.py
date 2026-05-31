"""``batch-delivery run`` — execute the full pipeline."""
from __future__ import annotations

from pathlib import Path

import typer

from batch_delivery.cli._app import app, config_app  # noqa: F401


@app.command()
def run(
    path: Path = typer.Option(None, "--config", "-c", help="YAML file (defaults to conf/default.yaml)"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable the stage cache (forces full recompute)."),
    jobs: int = typer.Option(None, "--jobs", "-j", help="Override parallel_jobs from config."),
    run_name: str = typer.Option(None, "--run-name", help="Human-friendly suffix for the run-id."),
) -> None:
    """Execute the full pipeline."""
    from batch_delivery.pipeline import run_all

    state = run_all(
        path,
        use_cache=not no_cache,
        parallel_jobs=jobs,
        run_name=run_name,
    )
    if state.ctx is not None:
        typer.echo(f"pipeline finished. run_id={state.ctx.run_id}")
        typer.echo(f"  manifest: {state.ctx.manifest_path}")
        typer.echo(f"  log:      {state.ctx.jsonl_path}")
    typer.echo(f"  results:  {state.out_dir}")
