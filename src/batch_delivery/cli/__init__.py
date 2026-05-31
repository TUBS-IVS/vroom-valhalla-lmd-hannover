"""Typer CLI: ``batch-delivery <subcommand>``.

This package was split out of the original flat ``cli.py`` during the
2026-05-31 GitHub-ready refactor. The top-level ``app`` and ``config_app``
Typer instances live in :mod:`batch_delivery.cli._app` and the actual
commands are organised by topic:

* :mod:`batch_delivery.cli.info`      — version, schedules, config show/validate
* :mod:`batch_delivery.cli.run`       — run the full pipeline
* :mod:`batch_delivery.cli.paper`     — reproduce EWGT 2026 paper outputs
* :mod:`batch_delivery.cli.sweep`     — parameter sweep
* :mod:`batch_delivery.cli.surrogate` — train / tune / validate / learn-loop
* :mod:`batch_delivery.cli.export`    — export optimisation results, build holdout
* :mod:`batch_delivery.cli.oracle`    — variance-driven oracle loop

Importing this package causes each submodule to be loaded, which fires the
``@app.command(...)`` decorators and registers every command. The
``batch_delivery.cli:app`` console-script entry point in ``pyproject.toml``
continues to work because ``app`` is re-exported here.
"""
from __future__ import annotations

from batch_delivery.cli._app import app, config_app

# Trigger registration of every command via its module import. Order does
# not matter functionally — alphabetical for readability.
from batch_delivery.cli import (  # noqa: F401  (import-for-side-effect)
    export,
    info,
    oracle,
    paper,
    run,
    surrogate,
    sweep,
)

__all__ = ["app", "config_app"]
