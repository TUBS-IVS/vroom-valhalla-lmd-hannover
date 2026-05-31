"""Basic info commands: version, schedules, config show/validate."""
from __future__ import annotations

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from batch_delivery import __version__
from batch_delivery.config import load_config

from batch_delivery.cli._app import app, config_app  # noqa: F401




@app.command()
def version() -> None:
    """Print the package version and exit."""
    typer.echo(__version__)


@config_app.command("show")
def config_show(
    path: Path = typer.Option(None, "--config", "-c", help="YAML file (defaults to conf/default.yaml)"),
) -> None:
    """Resolve and dump the configuration as YAML on stdout."""
    cfg = load_config(path)
    typer.echo(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))


@config_app.command("validate")
def config_validate(
    path: Path = typer.Option(None, "--config", "-c", help="YAML file to validate"),
) -> None:
    """Load + validate a configuration file. Exits non-zero on failure."""
    cfg = load_config(path)
    typer.echo(f"OK  scenarios={len(cfg.scenarios)}  providers={len(cfg.providers)}")


@app.command()
def schedules() -> None:
    """Enumerate the 39 feasible weekly delivery patterns (MAX_HOLDING_DAYS=3)."""
    from batch_delivery.config import EXPECTED_PATTERN_COUNT_K3, WEEKDAYS
    from batch_delivery.optimization import enumerate_valid_schedules

    patterns = enumerate_valid_schedules()
    typer.echo(f"Total: {len(patterns)} (expected {EXPECTED_PATTERN_COUNT_K3})")
    for i, p in enumerate(patterns, 1):
        days = sorted(p)
        typer.echo(f"  {i:2d}: {[WEEKDAYS[d][:3] for d in days]}")
