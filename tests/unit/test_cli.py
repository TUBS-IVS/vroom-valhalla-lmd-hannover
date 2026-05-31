"""CLI smoke tests using Typer's CliRunner."""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from batch_delivery import __version__
from batch_delivery.cli import app

pytestmark = pytest.mark.unit

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_config_validate_default() -> None:
    result = runner.invoke(app, ["config", "validate"])
    assert result.exit_code == 0, result.stdout
    assert "OK" in result.stdout


def test_schedules_lists_pinned_count() -> None:
    result = runner.invoke(app, ["schedules"])
    assert result.exit_code == 0, result.stdout
    from batch_delivery.config import EXPECTED_PATTERN_COUNT_K3

    assert f"Total: {EXPECTED_PATTERN_COUNT_K3}" in result.stdout
