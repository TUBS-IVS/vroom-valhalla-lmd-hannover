"""Typer ``app`` and ``config_app`` instances.

Lives in a tiny module so command submodules can import the app objects
without pulling in the rest of the cli package (and triggering circular
imports). Both objects are re-exported from ``batch_delivery.cli`` for
convenience and entry-point compatibility.
"""
from __future__ import annotations

import typer

app = typer.Typer(
    name="batch-delivery",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode=None,  # plain Click help -> cp1252-safe terminal output
    help="ML-surrogate optimisation framework for time-based parcel-delivery consolidation.",
)
config_app = typer.Typer(
    no_args_is_help=True,
    rich_markup_mode=None,
    help="Inspect or validate configuration files.",
)
app.add_typer(config_app, name="config")
