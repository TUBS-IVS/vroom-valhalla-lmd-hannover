# Project Folder Structure Blueprint

> Generated via the `folder-structure-blueprint-generator` skill.
> Project type: **Python** (auto-detected: `pyproject.toml` + `requirements.txt` + `src/`-layout).
> Monorepo: **false** · Microservices: **false** (containerised dependencies VROOM/Valhalla, not microservices of this project) · Frontend: **false**.
> Visualization style: **ASCII** · Depth: 4.

---

## 1. Structural Overview

The project is a **scientific Python pipeline** with a **layered architecture**
inspired by hexagonal/clean-architecture principles, adapted for batch
optimisation research:

* **`config/`** — pure constants and typed config schema (no I/O, no business logic).
* **`io/`** — adapters reading raw HAGRID/PLZ/hub data, writing checkpoints.
* **`routing/`** — adapters around external services (VROOM, Valhalla).
* **`features/`** — pure feature engineering (depends on `io/`, no I/O of its own).
* **`surrogate/`** — ML model lifecycle (train, persist, predict, benchmark).
* **`optimization/`** — algorithmic core (schedule enumeration, CD, SA, fleet balancing).
* **`evaluation/`** — KPI computation and reporting.
* **`pipeline.py`** + **`cli.py`** — composition root + entry points.

Outer layers depend on inner layers, never the other way around. `optimization/`
and `surrogate/` together form the **research kernel**; everything else is
adapter or orchestration code.

The repository keeps a parallel **`archive/legacy_2026_05/`** subtree where the
previous notebook-centric implementation is frozen for reference and
reproducibility of the original results.

---

## 2. Directory Visualization (ASCII tree, depth 4)

```
vroom-valhalla-lmd-hannover/
│
├── pyproject.toml                       # build config, deps, [project.scripts]
├── requirements.txt                     # pinned mirror of pyproject deps
├── docker-compose.yml                   # VROOM + Valhalla containers
├── README.md                            # high-level quick-start (post-refactor)
├── .pre-commit-config.yaml              # ruff + black + mypy + pytest-collect
│
├── conf/                                # YAML configurations (loaded by config/loader.py)
│   ├── default.yaml                     # canonical defaults
│   ├── scenarios/
│   │   ├── baseline.yaml
│   │   ├── fixed_express.yaml
│   │   ├── sa_ml_express.yaml
│   │   ├── fixed_batch.yaml
│   │   └── sa_ml_batch.yaml
│   └── carriers/
│       ├── dhl.yaml
│       ├── amazon.yaml
│       └── …                            # one file per LSP
│
├── data/                                # raw inputs (untracked: HAGRID, OSM tiles)
│   ├── demand/
│   ├── geodata/
│   ├── hubs/
│   └── vehicles/
│
├── results/                             # versioned outputs (v2 namespace)
│   └── v2/
│       ├── baseline/
│       ├── fixed_express/
│       ├── sa_ml_express/
│       ├── fixed_batch/
│       ├── sa_ml_batch/
│       ├── models/                      # serialised surrogate artefacts
│       ├── comparison/                  # cross-scenario KPI tables
│       ├── figures/                     # paper figures (PDF + PNG)
│       └── logs/                        # JSON-Lines run logs
│
├── valhalla/                            # routing tiles (untracked)
│
├── src/
│   └── batch_delivery/                  # the new installable package
│       ├── __init__.py
│       ├── __main__.py                  # `python -m batch_delivery` → cli:app
│       ├── cli.py                       # Typer CLI (sub-commands)
│       ├── pipeline.py                  # 7-stage orchestrator
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   ├── constants.py             # MAX_HOLDING_DAYS=3, EXPECTED_PATTERN_COUNT_K3, …
│       │   ├── schema.py                # Pydantic models
│       │   ├── loader.py                # load_config(path) → PipelineConfig
│       │   └── validation.py            # cross-field invariants + import-time asserts
│       │
│       ├── io/
│       │   ├── __init__.py
│       │   ├── demand.py                # HAGRID loader, weekday profiles
│       │   ├── hubs.py                  # PLZ→hub assignment, capacity
│       │   ├── geodata.py               # CRS, polygons, centroids
│       │   └── checkpoints.py           # parquet/pickle round-trip + hashing
│       │
│       ├── routing/
│       │   ├── __init__.py
│       │   ├── vroom_client.py
│       │   ├── valhalla_client.py
│       │   ├── request_builder.py
│       │   ├── solver.py                # robust loop, retries, parallel
│       │   └── cache.py                 # SHA-256 deterministic cache
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── spatial.py               # Tier 1+2 (convex hull, NN, density)
│       │   ├── demand_features.py       # Tier 3 (B2C share, demand stats)
│       │   ├── interactions.py          # 8 interaction terms
│       │   └── feature_set.py           # 44-col combo builder + ALL_COLS
│       │
│       ├── surrogate/
│       │   ├── __init__.py
│       │   ├── mlp_ensemble.py          # 5-seed MLP ensemble
│       │   ├── training.py              # fit + 5-fold CV
│       │   ├── cross_validation.py      # MAPE/RMSE reporting helpers
│       │   ├── benchmark.py             # LR/RF/GBM/MLP comparison (Figure 1)
│       │   └── registry.py              # versioned model artefacts
│       │
│       ├── optimization/
│       │   ├── __init__.py
│       │   ├── schedule_enum.py         # enumerate_valid_schedules() + count check
│       │   ├── coordinate_descent.py    # paper method (surrogate-based)
│       │   ├── simulated_annealing.py   # legacy SA (ablation)
│       │   ├── fleet_balancing.py       # swap-based hub-level smoothing
│       │   └── scenarios.py             # Scenario I & II builders
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── kpis.py                  # cost, km, fleet, wait
│       │   ├── comparison.py            # baseline ↔ scenarios
│       │   └── reports.py               # HTML/CSV/Markdown emitters
│       │
│       ├── legacy/                      # preserved (calibration only)
│       │   ├── __init__.py
│       │   └── daganzo.py               # ported from lmd/daganzo.py for Figure 1
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logging.py               # JSON-lines structured logs
│           ├── geometry.py              # CRS conversions, distances
│           ├── parallel.py              # joblib/threading helpers
│           └── timing.py                # `with stopwatch():` context
│
├── tests/
│   ├── conftest.py                      # shared fixtures, tmp_path setup
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_holding_days_invariant.py    # MAX_HOLDING_DAYS=3 guard
│   │   ├── test_io.py
│   │   ├── test_features.py
│   │   ├── test_surrogate.py
│   │   ├── test_optimization.py
│   │   └── test_evaluation.py
│   ├── integration/
│   │   ├── test_routing_smoke.py        # needs docker compose up
│   │   └── test_pipeline_smoke.py       # full mini-pipeline on fixture
│   └── fixtures/
│       ├── mini_demand.parquet
│       ├── mini_plz.geojson
│       └── mini_hubs.csv
│
├── docs/
│   ├── REFACTOR_PLAN.md
│   ├── FOLDER_STRUCTURE_BLUEPRINT.md    # ← this file
│   ├── PIPELINE.md                      # mermaid sequence + methodology figure
│   ├── CONFIG.md                        # YAML schema reference
│   └── CONCERNS.md                      # known issues & paper-text drift
│
└── archive/
    └── legacy_2026_05/                  # frozen pre-refactor state
        ├── README.md                    # explains why this exists
        ├── notebooks/                   # all *.ipynb verbatim
        ├── scripts/                     # one-shot patches (fix_*, add_*, …)
        ├── src_lmd/                     # previous package
        └── tests/                       # legacy pytest modules
```

---

## 3. Key Directory Analysis

### `src/batch_delivery/config/`

* **Purpose:** single source of truth for every numeric, string, and path constant.
* **Contents:**
  * `constants.py` — module-level immutable values (UPPER_SNAKE_CASE).
  * `schema.py` — Pydantic v2 BaseModels per concern (`RoutingConfig`,
    `OptimizationConfig`, `SurrogateConfig`, `ScenarioConfig`, `PipelineConfig`).
  * `loader.py` — `load_config(Path) -> PipelineConfig`; deep-merges defaults
    with scenario overrides; validates on construction.
  * `validation.py` — cross-field invariants:
    - `MAX_HOLDING_DAYS == 3`
    - `len(enumerate_valid_schedules()) == EXPECTED_PATTERN_COUNT_K3`
    - traffic-factor coverage, time-window monotonicity, vehicle capacity > 0, …
* **Patterns:** no I/O here; importing `config.constants` is a free, idempotent operation.

### `src/batch_delivery/io/`

* **Purpose:** adapter boundary toward the file system and external data formats.
* **Contents:** one module per data domain. Each exposes `load_*` and `save_*`.
* **Patterns:** functions return typed dataframes / dataclasses. **No** numpy / scientific transforms here.

### `src/batch_delivery/routing/`

* **Purpose:** adapter boundary toward VROOM and Valhalla HTTP services.
* **Contents:** thin clients (`*_client.py`), pure request builders (`request_builder.py`),
  resilient solver loop (`solver.py`), deterministic cache (`cache.py`).
* **Patterns:** all retries, timeouts, and parallelisation live here. Caller
  layers see only `solve(request) -> Solution`.

### `src/batch_delivery/features/`, `surrogate/`, `optimization/`, `evaluation/`

* **Purpose:** the **research kernel**. Pure computation; reproducible from inputs.
* **Patterns:** no `print`, no path constants, no `os.environ` lookups. All
  randomness flows through explicit `seed` parameters.

### `src/batch_delivery/legacy/`

* **Purpose:** preserves the Daganzo cost proxy used to produce the calibration
  comparison figure (Figure 1 in the paper). Not part of the production
  optimisation path.

### `tests/`

* **Layout:** `unit/` + `integration/` split. `fixtures/` ships tiny synthetic
  data so `unit/` and `pipeline_smoke` run without external services.
* **Markers:** `unit` (default), `integration` (needs docker), `slow` (full data).
* **Naming:** `test_<module>.py`; one test class per public class, snake_case
  test functions describing intent.

### `archive/legacy_2026_05/`

* **Purpose:** read-only freeze of the pre-refactor state; lets reviewers
  reproduce the original notebook results bit-for-bit.
* **Pattern:** never imported by the new package; never modified by the new
  test suite; one `README.md` documents what was moved and why.

---

## 4. File Placement Patterns

| Concern | Where |
|---|---|
| Numeric / string constants | `config/constants.py` |
| Typed configuration schema | `config/schema.py` |
| YAML defaults & overrides | `conf/default.yaml`, `conf/scenarios/*.yaml`, `conf/carriers/*.yaml` |
| Cross-field invariant checks | `config/validation.py` (called from `loader.py`) |
| Reading raw data | `io/<domain>.py` |
| Persisting intermediate state | `io/checkpoints.py` |
| External-service requests | `routing/*_client.py`, `routing/request_builder.py` |
| Pure feature math | `features/*.py` |
| Model definition / fit / predict | `surrogate/*.py` |
| Algorithmic optimisation core | `optimization/*.py` |
| KPI computation & reporting | `evaluation/*.py` |
| Composition / orchestration | `pipeline.py` |
| User-facing entry point | `cli.py` |
| Logging / parallel / timing helpers | `utils/*.py` |
| Unit tests | `tests/unit/test_<module>.py` |
| Integration / smoke tests | `tests/integration/` |
| Test fixtures | `tests/fixtures/` |
| Documentation | `docs/*.md` |
| Frozen previous version | `archive/legacy_2026_05/…` |

---

## 5. Naming and Organization Conventions

* **Modules:** `snake_case.py`, single concern, ≤ 400 LoC where possible.
* **Sub-packages:** `snake_case/` directories with `__init__.py` re-exporting
  the public API.
* **Classes:** `PascalCase` (`MLPEnsemble`, `PipelineConfig`).
* **Functions / variables:** `snake_case`.
* **Constants:** `UPPER_SNAKE_CASE`, only in `config/constants.py`.
* **Private helpers:** leading underscore (`_compute_wait_mx`).
* **YAML files:** `snake_case.yaml`, one scenario per file, no inheritance
  beyond the loader's deep-merge with `default.yaml`.
* **Tests:** `test_<module_under_test>.py`, function names start with `test_`
  and describe the asserted behaviour (`test_holding_constraint_respected`).
* **Result artefacts:** `results/v2/<scenario>/<lsp>/<artifact>.{json,csv,parquet}`.
* **Model artefacts:** `results/v2/models/<model_name>_<config_hash>.pkl`
  with sibling `<model_name>_<config_hash>.metrics.json`.
* **Log files:** `results/v2/logs/<run_id>.jsonl` (one JSON object per line).

### Module / Folder Mapping

| Folder | Namespace | Public API |
|---|---|---|
| `src/batch_delivery/config/` | `batch_delivery.config` | `load_config`, `PipelineConfig`, all `UPPER_SNAKE_CASE` constants |
| `src/batch_delivery/io/` | `batch_delivery.io` | `load_demand`, `load_hubs`, `load_geodata`, `Checkpoint` |
| `src/batch_delivery/routing/` | `batch_delivery.routing` | `solve`, `VroomClient`, `ValhallaClient` |
| `src/batch_delivery/features/` | `batch_delivery.features` | `build_combo_features`, `ALL_COLS` |
| `src/batch_delivery/surrogate/` | `batch_delivery.surrogate` | `MLPEnsemble`, `train`, `cross_validate`, `benchmark` |
| `src/batch_delivery/optimization/` | `batch_delivery.optimization` | `enumerate_valid_schedules`, `coordinate_descent`, `fleet_balance`, `build_scenario` |
| `src/batch_delivery/evaluation/` | `batch_delivery.evaluation` | `compute_kpis`, `compare`, `report` |
| `src/batch_delivery/utils/` | `batch_delivery.utils` | `get_logger`, `Stopwatch`, `parallel_map` |

---

## 6. Navigation and Development Workflow

### Entry Points

* **CLI:** `batch-delivery <subcommand>` (declared in `pyproject.toml`).
* **Module:** `python -m batch_delivery …` via `__main__.py`.
* **Library:** `from batch_delivery.pipeline import run_all`.

### Common Development Tasks

| Task | Where to start |
|---|---|
| Tune a scenario parameter | edit `conf/scenarios/<scenario>.yaml`, re-run `batch-delivery run-all --conf …` |
| Add a new LSP | add `conf/carriers/<lsp>.yaml`, register prefix in `config/constants.PROVIDER_PREFIXES` |
| Add a new feature | extend `features/<tier>.py`, append to `feature_set.ALL_COLS`, retrain surrogate |
| Add a new scenario | new file under `conf/scenarios/`, new builder in `optimization/scenarios.py` |
| Add a new KPI | implement in `evaluation/kpis.py`, surface in `evaluation/comparison.py` |
| Replace surrogate model | new module under `surrogate/`, register in `surrogate/registry.py` |
| Add a new test | mirror module path under `tests/unit/test_<module>.py` |

### Dependency Direction

```
cli.py ─► pipeline.py ─► evaluation/ ─► routing/ ─► io/ ─► config/
                       └► optimization/ ─► surrogate/ ─► features/ ─┘
```

Inner layers (`config`, `features`) never import outer layers
(`pipeline`, `cli`). Lint rule: `import-linter` contract enforces this.

---

## 7. Build and Output Organization

* **Build config:** `pyproject.toml` (PEP 621, `setuptools`-backend) declares
  package metadata, deps, dev-deps, console-script entry, and test settings.
* **Lock file:** `requirements.txt` regenerated via `pip-compile pyproject.toml`.
* **Outputs:** `results/v2/` is the canonical write location. The legacy
  `results/<scenario>/` namespace stays untouched for reproducibility of the
  pre-refactor numbers.
* **Environment-specific runs:** scenario YAMLs are the only knob; no separate
  `dev`/`prod` branches.

---

## 8. Python-Specific Patterns

* **`src/`-layout** — prevents accidental imports of un-installed code.
* **PEP 621** — single `pyproject.toml` for build + tooling.
* **Pydantic v2** — typed config validated at load time, no untyped `dict`s
  passed across module boundaries.
* **Typer** — CLI built from typed function signatures; auto-help, auto-completion.
* **`__init__.py` re-exports** — narrow, explicit public surface per sub-package.
* **No global state** — `pipeline.py` is a function, not a singleton.
* **Logging** — `logging.getLogger(__name__)`, JSON-lines handler in
  `utils/logging.py`; no `print` outside `cli.py`.
* **Determinism** — every random source (np, sklearn, torch if any) seeded
  via `PipelineConfig.seed`.

---

## 9. Extension and Evolution

* **Add a model:** drop a new file in `surrogate/`, expose via
  `surrogate/__init__.py`, register hyperparameters in `config/schema.py`.
* **Add an algorithm variant:** drop in `optimization/`, gate by a
  `ScenarioConfig.optimizer` enum.
* **Scale to more LSPs/regions:** introduce `Region` Pydantic model; current
  Hannover-specific constants stay in `conf/carriers/` and a new
  `conf/regions/` folder.
* **Refactoring discipline:** every move is reflected in `archive/` once,
  never twice; `archive/` is append-only.

---

## 10. Structure Templates

### New Feature Module Template (e.g. a new spatial metric)

```
src/batch_delivery/features/<new_metric>.py        # implementation
tests/unit/test_<new_metric>.py                    # unit tests
docs/CONFIG.md                                     # mention exposed config knob
```

* Add column name to `features/feature_set.ALL_COLS`.
* Add interaction (if any) to `features/interactions.INTERACTION_DEFS`.
* Update `tests/unit/test_features.py::test_combo_dimensionality`.

### New Scenario Template

```
conf/scenarios/<name>.yaml                         # YAML override
src/batch_delivery/optimization/scenarios.py       # builder branch
tests/integration/test_pipeline_smoke.py           # add parametrised case
```

### New Test Structure

```
tests/<unit|integration>/test_<target>.py
```

* Use `pytest.mark.unit` / `pytest.mark.integration` / `pytest.mark.slow`.
* Use shared fixtures from `tests/conftest.py`; do not read from `data/` in
  unit tests — only from `tests/fixtures/`.

---

## 11. Structure Enforcement

* **`ruff`** — lint + import sort, configured in `pyproject.toml`.
* **`black`** — formatting (line length 100).
* **`mypy --strict`** on `src/batch_delivery/`.
* **`import-linter`** — contract that
  `batch_delivery.config` and `batch_delivery.features` never import
  `batch_delivery.pipeline`, `routing`, or `cli`.
* **`pre-commit`** — runs ruff, black, mypy, and `pytest -m "unit"` on staged files.
* **CI:** GitHub Actions matrix (Python 3.11/3.12) runs `pytest -m "not slow"`;
  nightly run executes the full suite incl. `integration` against ephemeral
  docker compose.

---

## 12. Migration Path

1. **Phase 0** of the refactor plan moves the old tree under
   `archive/legacy_2026_05/`. Nothing is deleted.
2. **Phase 1+** scaffolds the new tree per this blueprint.
3. The old `results/<scenario>/` paths remain readable; the new pipeline writes
   under `results/v2/<scenario>/` to avoid collisions.
4. Once the new pipeline reproduces Table 1 within tolerance, the README
   quick-start swaps to the CLI-based instructions.
