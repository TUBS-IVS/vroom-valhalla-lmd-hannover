# CLAUDE.md

This file is the repo-local operating manual for Claude Code and other coding
agents. Read it before making changes.

## Project Identity

This repository contains `batch-delivery`, the companion code for Bienzeisler
et al., "Machine Learning Surrogate Optimization Framework for Time-ased Consolidation in Last Mile Parcel Delivery", EWGT 2026

The project is a scientific transportation-research pipeline for the Region
Hannover last-mile parcel-delivery setting. It studies time-based delivery
consolidation across logistics service providers (LSPs) using:

- Parcel-demand data from our sister project
  [HAGRID](https://github.com/TUBS-IVS/HAGRID) (same institute, TU
  Braunschweig — IVS) and PLZ-level geodata.
- Hub/depot assignment for DHL, Amazon, DPD, FedEx, GLS, Hermes, and UPS.
- VROOM with Valhalla routing for operational route costs.
- A 5-seed MLP surrogate model trained on routing-derived samples.
- Schedule optimization for weekly delivery patterns under service-quality
  constraints.
- KPI comparison across baseline, fixed schedules, and ML-optimized batching.

Treat this as research code that must remain reproducible enough for paper
review, not as a generic web app or CRUD service.

## Current State

The 2026-05-31 GitHub-ready refactor consolidated the project from a
research working tree into a publishable reproducibility package:

- `paper/EWGT_2026/` — frozen submitted figures + tables + abstract +
  claim manifest. Never overwrite.
- `src/batch_delivery/` — installable Python package. The four
  originally-monolithic files (`optimization/core.py` 2 751 lines,
  `cli.py` 1 793 lines, `routing/core.py` 1 181 lines, `pipeline.py`
  869 lines) are now split into focused submodules. The original module
  names remain as backwards-compatible re-export shims, so existing
  imports continue to work.
- `scripts/pipeline/` — four numbered canonical pipeline stages
  (`01_train_surrogate.py` ... `04_validate_vroom.py`).
- `scripts/figures/`, `scripts/paper/`, `scripts/data/`, `scripts/exploratory/`,
  `scripts/_archive/` — research scripts sorted by role.
- `results/paper_*` + `results/runs/` + `results/supplementary/` —
  canonical and supplementary outputs (tracked). `results/_archive/`
  is gitignored.
- 104/104 unit tests pass on the refactored layout.
- `docs/CHANGELOG.md` lists the full refactor scope.

`docs/FOLDER_STRUCTURE_BLUEPRINT.md` is the legacy aspirational design
that pre-dates this refactor; treat it as historical context only.
`docs/PIPELINE.md` is the current architecture document.

The frozen pre-refactor implementation is in `archive/legacy_2026_05/`.
Treat that archive as read-only unless the user explicitly asks otherwise.

## Research Framing

When editing code, docs, or paper-adjacent text, preserve the transportation
research meaning:

- The unit of analysis is parcel delivery in the Region Hannover PLZ network.
- LSP/provider terms usually refer to DHL, Amazon, DPD, FedEx, GLS, Hermes, UPS.
- "Baseline" means daily delivery without batching.
- "Fixed + Express" and "Fixed Batch-Only" use carrier fixed schedules.
- "SA_ML + Express" and "SA_ML Batch-Only" are the ML-surrogate optimized
  scenarios used for the paper comparison.
- The Daganzo model is legacy/ablation/calibration support, not the main
  optimization path.
- Do not invent numerical findings. Only report values that are produced by
  code, results files, tests, or docs in this repo.
- If a paper statement conflicts with code, flag it clearly instead of silently
  changing scientific meaning.

## Hard Invariants

Do not weaken these without explicit user approval:

- `MAX_HOLDING_DAYS = 3` is authoritative.
- The paper draft previously mentioned "at most two holding days"; that text is
  outdated. See `docs/HOLDING_DAYS_INVARIANT.md`.
- The operational delivery week has six days: Monday through Saturday.
- The number of feasible weekly delivery patterns for `MAX_HOLDING_DAYS = 3`
  is pinned as `EXPECTED_PATTERN_COUNT_K3 = 39`.
- Config validation, import-time assertions, and unit tests all guard the
  holding-days invariant.

## Architecture Map

Current package layout (post-2026-05-31 refactor):

- `src/batch_delivery/config/`: constants, Pydantic config schema, YAML loader,
  invariant validation.
- `src/batch_delivery/io/`: HAGRID demand loading, PLZ handling, hub assignment.
- `src/batch_delivery/routing/`: VROOM/Valhalla integration — split into
  `cache.py` (solution cache), `client.py` (Docker health + restart),
  `requests.py` (VROOM payload builders), `solver.py` (high-level solve
  interface). `core.py` is a backwards-compatible re-export shim.
- `src/batch_delivery/features/core.py`: Akkerman-style spatial/demand feature
  engineering; 25 base features.
- `src/batch_delivery/surrogate/core.py`: 44-column feature expansion and
  `MLCostPredictor` 5-seed MLP ensemble.
- `src/batch_delivery/optimization/`: schedule enum + optimisers + balancing —
  split into `schedules.py`, `costs.py` (Daganzo + ML cost matrices),
  `simulated_annealing.py` (legacy SA variants),
  `coordinate_descent.py` (production CD path), `balancing.py` (per-hub +
  system-level fleet smoothing). `core.py` is a re-export shim.
- `src/batch_delivery/evaluation/core.py`: KPI tables, scenario comparison,
  reports and plots.
- `src/batch_delivery/pipeline/`: seven-stage orchestrator — split into
  `state.py` (PipelineState), `stages.py` (the seven step_* functions),
  `orchestrator.py` (run_all). The package preserves
  `from batch_delivery.pipeline import run_all` for backwards compat.
- `src/batch_delivery/cli/`: Typer CLI — split into `_app.py` (app
  instances), `info.py`, `run.py`, `sweep.py`, `surrogate.py`,
  `export.py`, `oracle.py`, `paper.py`. Entry-point
  `batch_delivery.cli:app` is preserved.
- `src/batch_delivery/legacy/daganzo.py`: Daganzo cost proxy retained for
  ablation/calibration only.

The main orchestration flow in `pipeline.stages` is:

1. Load demand and hubs.
2. Solve traffic-adjusted baseline with VROOM/Valhalla.
3. Prepare optimization data structures.
4. Train the surrogate.
5. Optimize schedules.
6. Resolve non-baseline scenarios with VROOM.
7. Evaluate KPIs and write comparison outputs.

The canonical paper pipeline (`batch-delivery paper`) instead runs the
four-stage pipeline scripts under `scripts/pipeline/` (01_train_surrogate
... 04_validate_vroom). The full seven-stage in-process orchestrator is
still available via `batch-delivery run`.

## Development Commands

Use PowerShell on this machine.

Install editable package with development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Start routing services for integration/full pipeline work:

```powershell
docker compose up -d
```

CLI examples:

```powershell
batch-delivery version
batch-delivery config show --config conf/default.yaml
batch-delivery config validate --config conf/default.yaml
batch-delivery schedules
batch-delivery run --config conf/default.yaml
```

Fast verification:

```powershell
python -m pytest tests/unit -v
```

Integration verification, only when VROOM and Valhalla are running:

```powershell
python -m pytest tests/integration -m integration -v
```

Full or slow pipeline work can be expensive. Do not run it casually unless the
user asked for full reproduction or the change touches the pipeline behavior.

## External Services And Data

VROOM listens on port `3000`. Valhalla listens on port `8002`. `docker-compose.yml`
downloads/serves Niedersachsen OSM data for Valhalla and uses VROOM with the
Valhalla router.

Relevant data directories:

- `data/demand/`: HAGRID demand shapefiles, including weekday demand.
- `data/geodata/`: Region Hannover shapes and PLZ areas.
- `data/hubs/`: KEP hub CSV data.
- `data/vehicles/`: HAGRID vehicle type definitions.
- `results/`: generated outputs, caches, reports, model artifacts.

Many runtime artifacts are ignored by `.gitignore`. Avoid committing generated
route solutions, cache files, checkpoints, or large routing tiles.

## Coding Rules For This Repo

- Prefer the existing public exports in package `__init__.py` files.
- Keep changes scoped. These modules are already large; avoid broad refactors
  unless the user asks for one.
- Use typed config and `conf/default.yaml` for scenario knobs. Avoid new
  hardcoded paper parameters outside `config/constants.py` or the schema.
- Keep randomness explicit and seeded through config where possible.
- Do not change public scenario names casually; tests and paper tables rely on
  them.
- Do not import from `archive/legacy_2026_05/` in modern package code.
- Keep `legacy/daganzo.py` isolated from the main optimization path.
- Unit tests should not require Docker. Integration tests may touch
  VROOM/Valhalla and are marked accordingly.
- If modifying feature dimensions, update tests that pin 25 base features,
  8 interactions, 11 log transforms, and 44 surrogate combo columns.
- If modifying routing cost assumptions, document the effect on paper KPIs.

## Common Pitfalls

- The docs may describe future split modules like `features/spatial.py` or
  `routing/solver.py`; current implementation mostly lives in `core.py`.
- Some console output may show mojibake for UTF-8 box-drawing characters in
  older docs. Do not treat that as a scientific or code error.
- `archive/` is ignored and can be huge. Do not scan it recursively unless you
  are explicitly comparing legacy behavior.
- Integration tests skip when routing services are unavailable; a skipped
  integration test is not proof the full pipeline works.
- `MAX_HOLDING_DAYS = 3` is not a typo.

## Paper-Aware Review Checklist

Before claiming a change is complete, check the relevant subset:

- Config still validates.
- The 39-pattern holding-days invariant still passes.
- Scenario names and KPI semantics are unchanged unless intentionally edited.
- Unit tests pass for the touched layer.
- Integration tests or a smoke run were executed if routing/pipeline behavior
  changed and services are available.
- Documentation or paper-facing wording does not overclaim results.

