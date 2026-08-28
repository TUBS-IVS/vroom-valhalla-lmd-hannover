"""``batch-delivery paper`` — orchestrate the four **submitted-version**
pipeline stages under ``scripts/pipeline/``.

.. warning::

   **This is not the EWGT rev1 pipeline.** These four stages reproduce the
   numbers of the *submitted* manuscript (``paper/EWGT_2026/``). They
   predate the universal tour rule, the pool term, the two cost lenses and
   the operator-cost polish, so their output is **not** comparable with the
   revision's. Every one of them raises a ``DeprecationWarning`` on import,
   and this command prints the same notice before it runs anything.

   The canonical pipeline for the EWGT 2026 **revision** is the numbered
   script sequence under ``scripts/revision/`` — see ``docs/PIPELINE.md``
   and ``scripts/README.md``. There is deliberately no CLI wrapper for it:
   the stages are long-running, resumable and take a live run directory, so
   they are invoked directly.

The command is kept runnable because reproducing the submitted version is a
legitimate reason to want it. It is simply no longer the default meaning of
"the paper pipeline".

The four submitted-version stages::

    01_train_surrogate.py     Daganzo-LGB-Hybrid surrogate training
    02_optimize_grid.py       (P, theta) 88-cell coordinate-descent run
    03_apply_smoothing.py     system-level fleet smoothing post-process
    04_validate_vroom.py      VROOM out-of-sample re-routing

Each stage writes its outputs to a fixed location under ``results/`` —
see ``scripts/README.md`` for the input/output mapping. The orchestrator
shells out to ``python <stage>.py`` so the stages can also be run
manually in isolation, which is important during development.

Typical usage::

    batch-delivery paper --dry-run            # list stages, do nothing
    batch-delivery paper                       # run all four stages
    batch-delivery paper --stage 3             # run only stage 3
    batch-delivery paper --from 2              # resume from stage 2
    batch-delivery paper --skip-vroom          # run stages 1-3 only
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import typer

from batch_delivery.cli._app import app

PIPELINE_DIR = Path(__file__).resolve().parents[3] / "scripts" / "pipeline"

STAGES: list[tuple[int, str, str, str]] = [
    (1, "01_train_surrogate.py",  "Train Daganzo-LGB-Hybrid surrogate",
     "results/supplementary/sweep_v3_mergefix/training_matrix.csv"),
    (2, "02_optimize_grid.py",    "Optimize the (P, theta) 88-cell grid",
     "results/runs/path2_*/tab_balancing_summary.csv"),
    (3, "03_apply_smoothing.py",  "System-level fleet smoothing",
     "results/runs/path2_*/_system_spread_per_cell.csv"),
    (4, "04_validate_vroom.py",   "VROOM out-of-sample validation",
     "results/paper_results_*/07_validation/tab_vroom_path2.csv"),
]


#: Printed on **every** invocation, before anything runs, including
#: ``--dry-run``. The stale-path failure mode this closes is concrete: a
#: reviewer follows a doc pointer, runs ``batch-delivery paper``, and gets a
#: grid computed without the pool term, without the universal tour rule,
#: without the two lenses and without the operator polish — i.e. numbers that
#: are not the revision's. Silence here is the defect; the notice is the fix.
REVISION_NOTICE = """
  ============================================================================
  NOTICE: this is the SUBMITTED-VERSION pipeline, not the EWGT rev1 pipeline.
  ============================================================================
  The four scripts/pipeline/ stages reproduce paper/EWGT_2026/ (the submitted
  manuscript). They predate the universal tour rule, the pool term, the two
  cost lenses and the operator-cost polish, so their numbers are NOT
  comparable with the revision's and must never be quoted beside them.

  The canonical pipeline for the EWGT 2026 revision is scripts/revision/:

      61_grid_run_v2.py        the (P, theta, provider) grid, two plans
      62_gates_check.py        gates G1a / G1b / G3 / G4 (read-only)
      63_/64_/64a_/65_         bundle pool -> bundle head + Gate U
      67_validate_vroom_v2.py  out-of-sample VROOM re-routing, both lenses
      70_figs_tables_v2.py     figures and tables
      71_sync_paper_figs.py    md5-verified sync into paper/EWGT_2026_rev1/
      72_/73_/74_              per-cell costs, ops tables, legacy adapter
      75_ .. 78_               supplementary figures
      79_build_final_pack.py   the final results pack

  See docs/PIPELINE.md. Those stages take a live run directory and are
  invoked directly, so there is deliberately no CLI wrapper for them.

  Continuing with the submitted-version pipeline.
  ============================================================================
"""


def _print_plan(stages_to_run: list[tuple[int, str, str, str]], dry_run: bool) -> None:
    typer.echo("")
    typer.echo("  EWGT 2026 SUBMITTED-version paper pipeline")
    typer.echo("  " + "-" * 42)
    for n, script, desc, _output in stages_to_run:
        marker = "  [DRY RUN]" if dry_run else "         "
        typer.echo(f"  {marker} Stage {n}  {desc}")
        typer.echo(f"             -> python {(PIPELINE_DIR / script).relative_to(PIPELINE_DIR.parents[1])}")
    typer.echo("")


@app.command(name="paper")
def paper_cmd(
    stage: int = typer.Option(
        None, "--stage", "-s",
        help="Run only this stage (1-4). Default: run all stages from --from onwards.",
    ),
    from_stage: int = typer.Option(
        1, "--from", "-f",
        help="Resume the pipeline from this stage. Default: 1.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n",
        help="Print the plan and exit without running anything.",
    ),
    skip_vroom: bool = typer.Option(
        False, "--skip-vroom",
        help="Skip stage 4 (VROOM validation). Useful when the Docker stack is offline.",
    ),
) -> None:
    """Reproduce the SUBMITTED EWGT 2026 paper numbers from scratch.

    Runs the four ``scripts/pipeline/`` stages in order. Each stage writes
    its outputs to a fixed location under ``results/`` so subsequent
    stages can pick them up. Stages are idempotent: re-running a stage
    with existing outputs simply overwrites them.

    These stages are DEPRECATED for the EWGT rev1 revision: they predate the
    universal tour rule, the pool term, the two cost lenses and the
    operator-cost polish. The revision's canonical pipeline is the numbered
    sequence under ``scripts/revision/`` (61_ grid ... 79_ final pack); see
    ``docs/PIPELINE.md``. A notice to that effect is printed before anything
    runs, including under ``--dry-run``.
    """
    # Fail loud before doing anything: never run stale stages silently.
    typer.echo(REVISION_NOTICE, err=True)

    if stage is not None and not (1 <= stage <= 4):
        typer.echo(f"error: --stage must be 1..4, got {stage}", err=True)
        raise typer.Exit(code=2)
    if not (1 <= from_stage <= 4):
        typer.echo(f"error: --from must be 1..4, got {from_stage}", err=True)
        raise typer.Exit(code=2)

    if stage is not None:
        stages_to_run = [s for s in STAGES if s[0] == stage]
    else:
        stages_to_run = [s for s in STAGES if s[0] >= from_stage]
        if skip_vroom:
            stages_to_run = [s for s in stages_to_run if s[0] != 4]

    if not stages_to_run:
        typer.echo("Nothing to do.")
        return

    _print_plan(stages_to_run, dry_run)

    if dry_run:
        return

    for n, script, desc, _output in stages_to_run:
        path = PIPELINE_DIR / script
        if not path.exists():
            typer.echo(f"  [FAIL] stage {n}: script not found at {path}", err=True)
            raise typer.Exit(code=1)
        t0 = time.time()
        typer.echo(f"+-- Stage {n}: {desc}")
        typer.echo(f"|   python {path}")
        typer.echo("+" + "-" * 60)
        proc = subprocess.run([sys.executable, str(path)], check=False)
        dt = time.time() - t0
        if proc.returncode != 0:
            typer.echo(f"  [FAIL] stage {n} failed (exit {proc.returncode}) after {dt:.0f}s", err=True)
            raise typer.Exit(code=proc.returncode)
        typer.echo(f"  [OK]   stage {n} done in {dt:.0f}s")
        typer.echo("")

    typer.echo("  [DONE] paper pipeline complete")
