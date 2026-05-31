# Changelog

All notable changes to this project will be documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [SemVer](https://semver.org/) loosely (this is research
code; the major version tracks paper revisions).

## [1.0.0] — 2026-05-31 — EWGT 2026 paper submission

### Repository refactor (GitHub-ready)

The 2026-05-31 refactor restructured the project from a research-process
working tree into a publishable, reproducible research artefact. No
scientific results changed; everything in this release is paper-faithful.

#### Added

- `paper/EWGT_2026/` — top-level frozen submission folder with figures,
  tables, abstract, and a claim → file MANIFEST.
- `results/CANONICAL.md` — single source-of-truth classification of every
  results sub-folder into canonical / supplementary / archive / delete.
- `scripts/pipeline/` — four numbered canonical pipeline stages:
  - `01_train_surrogate.py` (was `train_daganzo_hybrid.py`)
  - `02_optimize_grid.py` (was `overnight_orchestrator_balanced.py`)
  - `03_apply_smoothing.py` (was `_apply_system_smoothing.py`)
  - `04_validate_vroom.py` (was `_vroom_validate_path2_v2.py`)
- `scripts/figures/` — 26 paper-figure scripts (renamed from
  `_fig_*_for_EWGT.py` to `fig_*.py`).
- `scripts/paper/` — 44 paper-output builders (assembly, break-even,
  sweet-spot, etc.).
- `scripts/data/`, `scripts/exploratory/`, `scripts/_archive/` — supporting
  data prep, research-process diagnostics, and superseded versions.
- `batch-delivery paper` Typer CLI command for end-to-end reproduction
  (`--dry-run`, `--stage N`, `--from N`, `--skip-vroom` options).
- `docs/PIPELINE.md`, `docs/REPRODUCING_PAPER.md`, `docs/CHANGELOG.md`.
- `CITATION.cff`, `LICENSE` (MIT).
- `scripts/_refactor/` — one-shot AST-based source-code split helpers
  (kept for audit trail; not part of the runtime).

#### Changed

- `src/batch_delivery/optimization/core.py` (2 751 lines) split into six
  focused modules: `schedules.py` (168 l), `costs.py` (1 019 l),
  `simulated_annealing.py` (742 l), `coordinate_descent.py` (428 l),
  `balancing.py` (595 l). `core.py` is now a 64-line backwards-compatible
  re-export shim.
- `src/batch_delivery/cli.py` (1 793 lines) converted into a `cli/`
  package with one submodule per command group: `_app.py`, `info.py`,
  `run.py`, `sweep.py`, `surrogate.py`, `export.py`, `oracle.py`,
  `paper.py` (new).
- `src/batch_delivery/routing/core.py` (1 181 lines) split into
  `cache.py` (80 l), `client.py` (162 l), `requests.py` (522 l),
  `solver.py` (593 l), with a re-export shim.
- `src/batch_delivery/pipeline.py` (869 lines) converted into a
  `pipeline/` package with `state.py` (60 l), `stages.py` (745 l),
  `orchestrator.py` (139 l). Existing
  `from batch_delivery.pipeline import run_all` calls continue to work.
- `README.md` rewritten as a showcase: paper citation, headline numbers,
  quickstart, architecture diagram, repository map, reproduction recipe.
- `.gitignore` tightened: `data/` and `results/_archive/` gitignored;
  canonical paper outputs explicitly un-ignored; all `*.pkl`,
  `*.parquet`, `*.npz`, `*.shp` bulk binaries blocked; agent session
  artefacts (`.claude/`, `memory/`, `docs/superpowers/`) blocked.
- `results/` reorganised:
  - canonical:    `paper_outputs_2026_05_30/`, `paper_results_2026_05_30/`,
                  `paper_ewgt_2026/`, `runs/path2_2026_05_29/`
  - supplementary: 12 folders under `supplementary/`
  - archive:      30 folders under `_archive/` (gitignored)
  - deleted:      `cache/`, `checkpoints/`, `baseline/`, smoke tests,
                  one-off diagnostics — 4.6 GB freed

#### Removed

- 18 redundant `results/` folders (smoke / diagnostic / one-off).
  4.6 GB reclaimed from the working tree.
- 4 211 unreachable git blobs from old MLP-pickle history.
  `.git/` shrank from 285 MB to 3.1 MB.
- Stray `scripts/_path2_run/` and `scripts/results/` directories created
  by misrouted ad-hoc runs.
- `vroom/access.log` from git tracking (runtime log, not source).

#### Backwards compatibility

Every public import path in `src/batch_delivery/` is preserved through
re-export shims. The 104 unit tests pass unchanged. Existing scripts
that imported from `batch_delivery.optimization.core`,
`batch_delivery.routing.core`, or `batch_delivery.pipeline` continue to
work without modification — they will simply log a friendly deprecation
note in a future release.

### Scientific content

No model retraining, no parameter changes, no re-optimization. The
canonical Path-2 run (`results/runs/path2_2026_05_29/`) is the same
88-cell grid that produced every headline number in the paper.

## [0.x] — 2026-05-21 — Pre-refactor working tree

Frozen snapshot of the working tree before the GitHub-ready refactor.
Captured on git branch `pre-refactor-2026-05-31` for safety; this branch
should be considered read-only.

See `archive/legacy_2026_05/` for the even earlier notebook-driven
codebase that the modern installable package replaced.
