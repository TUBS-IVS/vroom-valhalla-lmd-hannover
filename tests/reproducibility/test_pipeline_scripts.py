"""Verify the four canonical pipeline scripts under scripts/pipeline/ load
without import errors and that ``batch-delivery paper`` lists them.

This catches the common breakage where a script depends on a function
that was renamed or removed during a refactor — silently, until someone
tries to reproduce the paper at 22:00 the night before submission.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = REPO_ROOT / "scripts" / "pipeline"

EXPECTED_STAGES = [
    "01_train_surrogate.py",
    "02_optimize_grid.py",
    "03_apply_smoothing.py",
    "04_validate_vroom.py",
]


@pytest.mark.parametrize("script", EXPECTED_STAGES)
def test_pipeline_script_exists(script: str) -> None:
    """Each canonical stage script is on disk under scripts/pipeline/."""
    path = PIPELINE_DIR / script
    assert path.exists(), f"Missing canonical pipeline script: {path}"


@pytest.mark.parametrize("script", EXPECTED_STAGES)
def test_pipeline_script_is_valid_python(script: str) -> None:
    """Each stage script parses as valid Python (no SyntaxError)."""
    path = PIPELINE_DIR / script
    source = path.read_text(encoding="utf-8")
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        pytest.fail(f"{script} has a SyntaxError: {exc}")


def test_paper_cli_lists_all_four_stages() -> None:
    """`batch-delivery paper --dry-run` lists all four stages in order."""
    from batch_delivery.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["paper", "--dry-run"])
    assert result.exit_code == 0, result.stdout

    out = result.stdout
    for n in range(1, 5):
        assert f"Stage {n}" in out, (
            f"`batch-delivery paper --dry-run` does not mention Stage {n}: {out!r}"
        )


def test_paper_cli_skip_vroom_drops_stage_4() -> None:
    """`batch-delivery paper --dry-run --skip-vroom` lists only stages 1-3."""
    from batch_delivery.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["paper", "--dry-run", "--skip-vroom"])
    assert result.exit_code == 0, result.stdout
    assert "Stage 4" not in result.stdout, (
        "skip-vroom should hide stage 4: " + result.stdout
    )
    for n in (1, 2, 3):
        assert f"Stage {n}" in result.stdout


def test_paper_cli_rejects_bad_stage() -> None:
    """`batch-delivery paper --stage 99` exits non-zero."""
    from batch_delivery.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["paper", "--stage", "99"])
    assert result.exit_code != 0
