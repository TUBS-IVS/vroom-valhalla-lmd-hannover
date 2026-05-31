# Refactor Plan: Notebook → CLI-Driven Python Pipeline (`batch_delivery`)

> Generated via the `refactor-plan` skill.
> Author: refactor planner · Date: 2026-05-21

---

## Refactor Goal

Konsolidiere die aktuell in mehreren großen Jupyter-Notebooks
(`notebooks/batch_delivery_day_optimization.ipynb`, `notebooks/optimization_new.ipynb`,
`notebooks/dhl_vrp_per_plz.ipynb`, …) und in `scripts/*.py` verstreute Pipeline
in **ein einziges, gut strukturiertes Python-Programm** mit klarem CLI-Einstiegspunkt,
typisierten Konfigurationen, hartem Bug-Fix für `MAX_HOLDING_DAYS = 3`,
durchgängiger Validierung und vollständiger Test-Abdeckung.
Alte Dateien bleiben als Backup unter `archive/` erhalten.

---

## Current State

* **Hot path liegt in Notebooks**: 5 000+ Zeilen Logik in
  `notebooks/batch_delivery_day_optimization.ipynb`; Reproduzierbarkeit nur
  per „Run All".
* **Skript-Friedhof**: `scripts/` enthält 25+ Einmal-Patches
  (`fix_*`, `add_*`, `rebuild_*`), die historisch Notebookzellen reparieren —
  nicht Teil der Produktions-Pipeline.
* **`src/lmd/`** ist zwar modular (`config.py`, `demand.py`, `hubs.py`,
  `daganzo.py`, `optimization.py`, `routing.py`, `evaluation.py`, `features.py`,
  `ml_cost.py`), aber:
  * keine zentrale Orchestrierung außerhalb des Notebooks,
  * Daganzo-Pfad und ML-Surrogate-Pfad koexistieren ohne klare Trennung,
  * keine YAML/Pydantic-Config; alle Parameter hartcodiert,
  * `MAX_HOLDING_DAYS = 3` korrekt gesetzt, aber stale Kommentare („≤ 2 holding days")
    in `scripts/restructure_notebook.py:263` und in Notebook-Markdownzellen.
* **Tests**: 6 pytest-Module unter `tests/`, nur Unit-Level; keine
  Integration- oder Smoke-Pipeline-Tests.
* **Output**: ungeschnittene `results/`-Hierarchie mit doppelten Szenarionamen
  (`fixed_batch_only/` ↔ `fixed_batch-only/`).

## Target State

* **Ein installierbares Paket `batch_delivery`** unter `src/batch_delivery/`
  mit klar getrennten Sub-Packages (`config/`, `io/`, `routing/`,
  `features/`, `surrogate/`, `optimization/`, `evaluation/`, `utils/`).
* **Eine CLI** (`python -m batch_delivery` oder `batch-delivery <cmd>`)
  mit Sub-Commands `prepare-data`, `solve-baseline`, `train-surrogate`,
  `optimize`, `evaluate`, `run-all`.
* **YAML-Konfiguration** mit Pydantic-Schema, validiert beim Laden;
  `MAX_HOLDING_DAYS` ist dort gepinnt + Cross-Check, dass die enumerierte
  Pattern-Anzahl exakt der erwarteten Konstante entspricht.
* **Active-Learning-Loop** (Routing → Features → Train → Predict → Optimize
  → Re-Route) als deklarativer Pipeline-Graph in `pipeline.py`.
* **Tests**: pytest-Markers `unit`, `integration`, `slow`; CI-tauglicher
  schneller Layer + reproduzierbare Integration-Smoke (synth. Daten).
* **Backup**: alle Notebook-/Skript-Altlasten unter `archive/legacy_2026_05/`,
  unverändert, mit `README.md` der erklärt warum sie dort liegen.
* **Saubere `results/`-Struktur**: `results/<scenario>/<lsp>/<artifact>.{json,csv,parquet}`.

---

## Affected Files

| File / Path | Change Type | Dependencies |
|---|---|---|
| `src/lmd/` (whole tree) | move → `archive/legacy_2026_05/src_lmd/` | blocks new package import paths |
| `notebooks/*.ipynb` | move → `archive/legacy_2026_05/notebooks/` | blocked by export of remaining logic into modules |
| `scripts/fix_*.py`, `scripts/add_*.py`, `scripts/rebuild_*.py`, `scripts/restructure_notebook.py`, `scripts/_check_nb.py` | move → `archive/legacy_2026_05/scripts/` | none (one-shot tools) |
| `tests/test_*.py` (current) | move → `archive/legacy_2026_05/tests/` (kept as reference); rewrite under `tests/unit/` | blocks new test layout |
| `src/batch_delivery/__init__.py` | create | blocks all new modules |
| `src/batch_delivery/config/{constants,loader,validation}.py` | create | blocks every other new module |
| `src/batch_delivery/io/{demand,hubs,geodata,checkpoints}.py` | create (port from `lmd/demand.py`, `lmd/hubs.py`) | blocked by `config/` |
| `src/batch_delivery/routing/{vroom_client,valhalla_client,request_builder,solver,cache}.py` | create (port from `lmd/routing.py`) | blocked by `config/`, `io/` |
| `src/batch_delivery/features/{spatial,demand_features,interactions,feature_set}.py` | create (port from `lmd/features.py`) | blocked by `io/` |
| `src/batch_delivery/surrogate/{mlp_ensemble,training,cross_validation,benchmark,registry}.py` | create (port from `lmd/ml_cost.py` + benchmark notebook §10) | blocked by `features/` |
| `src/batch_delivery/optimization/{schedule_enum,coordinate_descent,simulated_annealing,fleet_balancing,scenarios}.py` | create (port from `lmd/optimization.py`) | blocked by `surrogate/` |
| `src/batch_delivery/evaluation/{kpis,comparison,reports}.py` | create (port from `lmd/evaluation.py`) | blocked by `routing/`, `optimization/` |
| `src/batch_delivery/pipeline.py` | create (replaces orchestration notebook) | blocked by all of the above |
| `src/batch_delivery/cli.py` | create (Typer entry) | blocked by `pipeline.py` |
| `conf/default.yaml`, `conf/scenarios/*.yaml`, `conf/carriers/*.yaml` | create | none |
| `pyproject.toml` | modify (add `[project.scripts]` entry, package name, dev-deps: typer, pydantic, pytest-xdist) | blocks installable CLI |
| `requirements.txt` | modify (sync to pyproject) | blocks env reproducibility |
| `tests/unit/test_*.py` | create (rewrite of legacy tests against new modules) | blocks Phase 4 verification |
| `tests/integration/test_pipeline_smoke.py` | create | blocks final acceptance |
| `tests/fixtures/` | create (synthetic mini-PLZ + mini-demand) | blocks integration tests |
| `README.md` | modify (replace notebook-centric quick-start) | last |
| `docs/REFACTOR_PLAN.md` | this file | none |
| `docs/FOLDER_STRUCTURE_BLUEPRINT.md` | created in parallel | none |

---

## Execution Plan

### Phase 0 — Safety Net & Backup (no logic change)

- [ ] **0.1** Create `archive/legacy_2026_05/` and move:
      `notebooks/`, `scripts/`, `src/lmd/`, `tests/` (as-is) into it.
- [ ] **0.2** Add `archive/legacy_2026_05/README.md` explaining the freeze.
- [ ] **0.3** Tag git commit `pre-refactor-2026-05-21` so the prior state
      can be restored with one `git checkout`.
- [ ] **Verify**: `git status` clean after move; legacy `pytest archive/legacy_2026_05/tests` (with old `sys.path`) still passes (smoke import only — actual run optional).

### Phase 1 — Project Scaffolding

- [ ] **1.1** Create new folder tree per `docs/FOLDER_STRUCTURE_BLUEPRINT.md`.
- [ ] **1.2** Update `pyproject.toml`:
        package name `batch-delivery`, `[project.scripts] batch-delivery = "batch_delivery.cli:app"`, deps (`typer`, `pydantic>=2`, `pyyaml`).
- [ ] **1.3** `pip install -e .` to make the new namespace resolvable.
- [ ] **Verify**: `python -c "import batch_delivery"` succeeds; `batch-delivery --help` prints CLI skeleton.

### Phase 2 — Config & Constants (Types First)

- [ ] **2.1** Port `lmd/config.py` → `batch_delivery/config/constants.py` (pure constants only, no compute).
- [ ] **2.2** Add `batch_delivery/config/schema.py` (Pydantic models: `RoutingConfig`, `OptimizationConfig`, `SurrogateConfig`, `ScenarioConfig`, `PipelineConfig`).
- [ ] **2.3** Add `batch_delivery/config/loader.py` (`load_config(path: Path) -> PipelineConfig`).
- [ ] **2.4** Add `batch_delivery/config/validation.py` with **hard invariants**:
        * `assert MAX_HOLDING_DAYS == 3` (single source of truth)
        * `assert len(enumerate_valid_schedules()) == EXPECTED_PATTERN_COUNT_K3` (computed once and pinned in `constants.py`)
        * vehicle-time-window monotonicity, traffic-factor coverage 0–23h, etc.
- [ ] **2.5** `conf/default.yaml` mirrors all constants; `conf/scenarios/{baseline,fixed_express,sa_ml_express,fixed_batch,sa_ml_batch}.yaml`.
- [ ] **Verify**: `pytest tests/unit/test_config.py -k "holding or schedule_count or invariants"` passes; loading `conf/default.yaml` round-trips identical values to `constants.py`.

### Phase 3 — IO Layer

- [ ] **3.1** Port `lmd/demand.py` → `batch_delivery/io/demand.py` (HAGRID loader, weekday profiles, shifted-demand computer).
- [ ] **3.2** Port `lmd/hubs.py` → `batch_delivery/io/hubs.py` (PLZ→hub assignment, capacity).
- [ ] **3.3** Add `batch_delivery/io/geodata.py` (CRS handling, EPSG:25832 conversion, polygon centroids).
- [ ] **3.4** Add `batch_delivery/io/checkpoints.py` (parquet/pickle round-trip with hash keys).
- [ ] **Verify**: `pytest tests/unit/test_io.py` passes; sample run `batch-delivery prepare-data --conf conf/default.yaml --dry-run` outputs expected schemas.

### Phase 4 — Routing Layer

- [ ] **4.1** Split `lmd/routing.py`: HTTP client (`vroom_client.py`, `valhalla_client.py`), request builder (`request_builder.py`), solver loop (`solver.py`), SHA-256 cache (`cache.py`).
- [ ] **4.2** Add `--vroom-url` and `--valhalla-url` overrides via CLI.
- [ ] **Verify**: `docker compose up -d`; `pytest tests/integration/test_routing_smoke.py -m integration` solves a 50-stop fixture in < 10 s.

### Phase 5 — Features Layer

- [ ] **5.1** Port `lmd/features.py` → `batch_delivery/features/{spatial,demand_features,interactions,feature_set}.py` (split Tier 1/2/3 + 44-combo builder).
- [ ] **5.2** Add hard schema check: output `DataFrame` columns equal `ALL_44_COLS` exactly.
- [ ] **Verify**: `pytest tests/unit/test_features.py` passes (incl. NaN-free output, deterministic ordering).

### Phase 6 — Surrogate Layer

- [ ] **6.1** Port `lmd/ml_cost.py` → `batch_delivery/surrogate/mlp_ensemble.py`.
- [ ] **6.2** Add `training.py` (5-fold CV, ensemble of 5 seeds, MAPE / RMSE reporter).
- [ ] **6.3** Add `benchmark.py` (LR / RF / GBM / MLP-single / MLP-ensemble side-by-side; produces Figure 1 of the paper).
- [ ] **6.4** Add `registry.py` (versioned model artefacts under `results/models/`).
- [ ] **Verify**: `batch-delivery train-surrogate --conf conf/default.yaml` writes `results/models/mlp_ensemble_<hash>.pkl` + metrics JSON; CV-MAPE < 5 % on full data, < 10 % on smoke fixture.

### Phase 7 — Optimization Layer

- [ ] **7.1** Port `lmd/optimization.py` and split:
       `schedule_enum.py` (`enumerate_valid_schedules()` + pattern-count assertion),
       `coordinate_descent.py` (CD over surrogate cost matrix, the actual paper method),
       `simulated_annealing.py` (legacy SA, kept for ablation),
       `fleet_balancing.py`,
       `scenarios.py` (Scenario I / II builder).
- [ ] **7.2** Replace any code path that uses Daganzo-only cost with the surrogate path; mark Daganzo as `legacy/` (kept for calibration figure only).
- [ ] **7.3** Hard validation in `coordinate_descent.optimize()`: every PLZ end-state assignment ∈ enumerated schedule set; every gap ≤ 3.
- [ ] **Verify**: `pytest tests/unit/test_optimization.py` passes; smoke run on fixture converges in < 30 s.

### Phase 8 — Evaluation Layer

- [ ] **8.1** Port `lmd/evaluation.py` → `batch_delivery/evaluation/{kpis,comparison,reports}.py`.
- [ ] **8.2** Standardise output paths: `results/<scenario>/<lsp>/{routes.json,kpis.csv,fleet.csv}` and aggregated `results/comparison/scenario_kpis.csv`.
- [ ] **Verify**: `batch-delivery evaluate --conf conf/default.yaml --scenarios baseline,sa_ml_express,sa_ml_batch` reproduces Table 1 numbers within ±1 %.

### Phase 9 — Orchestration & CLI

- [ ] **9.1** Implement `pipeline.py` as a 7-stage state machine with explicit checkpoint guards.
- [ ] **9.2** Implement `cli.py` (Typer): `prepare-data`, `solve-baseline`, `train-surrogate`, `optimize`, `evaluate`, `run-all`, `validate-config`.
- [ ] **9.3** Add structured logging (`utils/logging.py` → JSON-Lines under `results/logs/`).
- [ ] **Verify**: `batch-delivery run-all --conf conf/default.yaml --workers 8` produces all expected artefacts; rerun is idempotent (caches hit).

### Phase 10 — Tests

- [ ] **10.1** Rewrite legacy tests under `tests/unit/` against new namespaces.
- [ ] **10.2** Add `tests/integration/test_pipeline_smoke.py` running `pipeline.run_all` on the fixture.
- [ ] **10.3** Add `tests/unit/test_holding_days_invariant.py` — direct guard for the bug (parametrised over MAX_HOLDING_DAYS values 2/3/4 to make the constraint explicit).
- [ ] **10.4** Wire `pytest -m "not slow"` into CI; full suite in nightly.
- [ ] **Verify**: `pytest -m "not slow"` < 60 s; `pytest -m integration` passes against a running docker compose stack.

### Phase 11 — Cleanup & Docs

- [ ] **11.1** Update `README.md` with new quick-start, CLI examples, scenario table.
- [ ] **11.2** Add `docs/PIPELINE.md` with sequence diagram (mermaid) + Figure 1 + methodology figure prompt (already drafted).
- [ ] **11.3** Update `requirements.txt` from `pyproject.toml` (`pip-compile`).
- [ ] **11.4** Add `archive/legacy_2026_05/README.md` with one-paragraph context per moved subtree.
- [ ] **Verify**: `pip install -e .[dev]` from a clean venv; `batch-delivery --help`; `pytest`; `pre-commit run --all-files`.

---

## Rollback Plan

If any phase fails midway:

1. `git reset --hard pre-refactor-2026-05-21` restores the entire repo state.
2. Phases 0–2 are idempotent; subsequent phases are additive (legacy `archive/` is read-only after Phase 0).
3. Per-phase rollback: `git restore -s pre-refactor-2026-05-21 -- <phase-paths>`.
4. If Pydantic-schema validation breaks an existing config, ship a one-line migration helper `batch-delivery migrate-config` and `--allow-legacy` flag rather than reverting code.

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hidden Daganzo↔surrogate coupling in legacy notebook | medium | Phase 7.2 explicitly inventories all callers; keep Daganzo as `legacy/` module so calibration figure still reproduces |
| Pattern-count constant drift if `MAX_HOLDING_DAYS` ever changes | medium | Constant computed once, asserted at import time, parametrised test guards intent |
| Cache invalidation across new request schemas | medium | New cache namespace key `v2/`; old cache untouched in `archive/` |
| `results/` collision between legacy and new runs | low | New runs write under `results/v2/<scenario>/...` while legacy stays under `results/<scenario>/...` |
| CI integration test depends on running docker stack | low | mark `integration` tests; document `docker compose up -d` prerequisite; provide `pytest -m "not integration"` for fast local |
| Paper text says „≤ 2 holding days" but code uses 3 | low (textual) | Document divergence in `docs/CONCERNS.md`, surface to user as `[ASK USER]` for paper correction |

---

## `[ASK USER]` Items

1. Is the **paper text** „at most two holding days" outdated and should be corrected to **three**, or is the **constant** the bug? (Code path: `MAX_HOLDING_DAYS = 3`. Confirmed by user that 3 is correct → paper text needs update.)
2. Keep the legacy **Daganzo proxy** as a parallel cost backend (for ablation/Figure 1 reproduction) or fully retire it?
3. Naming: `batch_delivery` vs. `batch-delivery` vs. keeping `lmd` as the import name? (Recommendation: import name `batch_delivery`, dist name `batch-delivery`.)
4. CLI framework: **Typer** (recommendation, type-checked) or stick with stdlib `argparse`?
5. Config format: **YAML** (recommendation) or TOML?
6. Should `archive/legacy_2026_05/` remain in-tree forever or be moved to a separate `*-legacy` repo after one release cycle?

---

**Shall I proceed with Phase 0 (Backup & Safety Net)?**
