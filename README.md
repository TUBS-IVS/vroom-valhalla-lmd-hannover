# `batch-delivery` — ML Surrogate Optimization for Time-Based Parcel Consolidation

Companion code and reproducibility package for:

> **Bienzeisler, L., Petre, F., Wage, O., and Friedrich, B.**, *"Machine-Learning
> Surrogate Optimization for Time-Based Consolidation in Last-Mile Parcel
> Delivery"*, submitted to Transportation Research Procedia (EWGT 2026),
> preprint 2026-05-31 (currently under review).

[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 127 passing](https://img.shields.io/badge/tests-127%20passing-brightgreen.svg)](#tests)
[![Paper: EWGT 2026 (submitted)](https://img.shields.io/badge/paper-EWGT%202026%20%E2%80%94%20submitted-orange.svg)](paper/EWGT_2026/ABSTRACT.md)

> **Status:** the paper is **submitted to EWGT 2026, not yet accepted.**
> This repo is published alongside the submission as the
> reproducibility package and will be updated for the camera-ready
> version once review feedback is in.

---

## Headline result

Time-based consolidation (TBC) lets parcel carriers delay eligible
shipments and bundle them into batched delivery days. In the Hanover
Region case study (seven LSPs, 1.26M weekly parcels) the framework
identifies a Pareto sweet spot in `P ∈ [0.25, 0.75] €/parcel·day`
where:

| Metric | Value |
|---|---|
| Cost saving at the cost-optimal extreme | **22.8 %** |
| Cost saving at the operating sweet spot (P = 0.5) | **13.5 %** |
| VROOM-validated saving at P = 0.5 (surrogate is conservative) | **15.6 %** |
| Mo–Sa fleet coefficient-of-variation reduction in the efficient range | **up to 60 %** |
| Peak fleet reduction at (P = 0.5, θ = 1) | **12.9 %** |
| Daganzo-LGB-Hybrid surrogate MAPE (GroupKFold over postal codes) | **2.95 %** |
| Surrogate conservatism band (vs VROOM re-routing) | **1.3 – 2.1 pp** |

Three carrier classes emerge with distinct optimal penalty levels:
service-bound (P\* = 0.25), hybrid (P\* = 0.5), cost-aggressive
(P\* = 0.75). TBC pays most in rural postal-code areas with long stems,
large service area, and few delivered parcels per stop.

The submitted paper outputs are frozen in [`paper/EWGT_2026/`](paper/EWGT_2026/).
Every number above traces to a file in [`paper/EWGT_2026/MANIFEST.md`](paper/EWGT_2026/MANIFEST.md).

---

## Repository at a glance

```
.
├── paper/EWGT_2026/      ← FROZEN SUBMISSION: figures, tables, abstract, manifest
├── src/batch_delivery/   ← installable Python package (pip install -e .)
│   ├── config/             constants + Pydantic schema + invariants
│   ├── io/                 HAGRID demand + hub assignment
│   ├── routing/            VROOM / Valhalla client + cache + request builders
│   ├── features/           Akkerman-style spatial / demand features
│   ├── surrogate/          Daganzo-LGB-Hybrid trainer + predictor
│   ├── optimization/       schedule enum + CD + SA + fleet balancing
│   ├── evaluation/         KPI tables + scenario comparison
│   ├── pipeline/           seven-stage orchestrator
│   ├── cli/                Typer CLI (batch-delivery <subcommand>)
│   └── legacy/             Daganzo continuum proxy (ablation only)
├── scripts/
│   ├── pipeline/           four-stage canonical pipeline (numbered 01..04)
│   ├── figures/            one script per paper figure
│   ├── paper/              paper-output builders (assembly, break-even, sweetspot)
│   ├── data/               input-data preparation
│   ├── exploratory/        research-process scripts (diagnostic, sensitivity)
│   └── _archive/           superseded versions (kept for paper-review trail)
├── tests/                  104 unit + 1 integration smoke (all passing)
├── conf/                   YAML configuration files
├── data/                   inputs (gitignored, fetched via download script)
├── results/                pipeline outputs (canonical + supplementary tracked,
│                            old runs in results/_archive/ gitignored)
└── docs/                   PIPELINE.md, REPRODUCING_PAPER.md, CHANGELOG.md
```

Detailed inventories:
[`results/CANONICAL.md`](results/CANONICAL.md) ·
[`scripts/README.md`](scripts/README.md) ·
[`docs/PIPELINE.md`](docs/PIPELINE.md)

---

## Quickstart

### 1. Install

```powershell
# Editable install with dev dependencies (Python ≥ 3.12)
python -m pip install -e ".[dev]"

# Routing stack (VROOM on :3000, Valhalla on :8002)
docker compose up -d
```

### 2. Verify

```powershell
# Should print 104 passing, ~10 s
python -m pytest tests/unit -q

# Should list 12 subcommands incl. paper, run, sweep, ...
batch-delivery --help
```

### 3. Reproduce the paper

```powershell
# Show the four-stage plan without running anything
batch-delivery paper --dry-run

# Run the full pipeline end-to-end (≈ 20 h wall-clock on a 4-core laptop)
batch-delivery paper

# Or run individual stages:
batch-delivery paper --stage 1     # surrogate training (~30 min)
batch-delivery paper --stage 2     # 88-cell optimization (~16 h)
batch-delivery paper --stage 3     # system-level smoothing (<5 min)
batch-delivery paper --stage 4     # VROOM out-of-sample validation (~3 h)

# Skip the VROOM stage when Docker is offline:
batch-delivery paper --skip-vroom
```

See [`docs/REPRODUCING_PAPER.md`](docs/REPRODUCING_PAPER.md) for the
step-by-step recipe from a fresh `git clone`.

---

## Architecture

### Pipeline (four stages)

```
                          +----------------+
        data/  ---------> | 01_train_surro |  ---> daganzo_hybrid_v3aug.pkl
   (HAGRID + geodata)     |   gate.py      |       (LightGBM residual,
                          +----------------+        alpha = 1.34, MAPE 2.95%)
                                                              |
                                                              v
                          +----------------+
                          | 02_optimize_   |  ---> results/runs/path2_*/
                          |    grid.py     |       tab_balancing_summary.csv,
                          +----------------+       tab_chosen_schedules.csv,
                                |                  tab_fleet_per_hub.csv
                                v
                          +----------------+
                          | 03_apply_      |  ---> _system_spread_per_cell.csv,
                          |   smoothing.py |       _tab_balancing_summary_with_smoothing.csv
                          +----------------+
                                |
                                v
                          +----------------+
   docker compose ------> | 04_validate_   |  ---> results/paper_results_*/
   (VROOM + Valhalla)     |   vroom.py     |       07_validation/tab_vroom_*.csv
                          +----------------+
```

Each stage is a standalone script under `scripts/pipeline/` so it can be
run, restarted, or inspected in isolation. The `batch-delivery paper`
CLI command chains them with stop-on-error and dry-run support.

### Source-code organisation

The Python package follows a strict "one responsibility per module"
layout. The four originally-monolithic files (`optimization/core.py`,
`cli.py`, `routing/core.py`, `pipeline.py`, together 6.6 k lines) were
split during the 2026-05-31 GitHub-ready refactor into focused
submodules; the original module names are preserved as re-export shims
for backwards compatibility. See [`docs/CHANGELOG.md`](docs/CHANGELOG.md)
for the migration log.

---

## Hard invariants

The codebase enforces three invariants at import time, in the Pydantic
config schema, and in the test suite. Do not weaken them without
explicit author approval.

| Invariant | Value | Enforced in |
|---|---|---|
| Maximum holding days per parcel | `MAX_HOLDING_DAYS = 3` | [`config/constants.py`](src/batch_delivery/config/constants.py), [`config/validation.py`](src/batch_delivery/config/validation.py), [`tests/unit/test_holding_days_invariant.py`](tests/unit/test_holding_days_invariant.py) |
| Operating week | Monday–Saturday (6 days) | same |
| Feasible weekly delivery patterns | `EXPECTED_PATTERN_COUNT_K3 = 39` | same |

The paper's 22.8 % / 13.5 % / 60 % / 2.95 % numbers all depend on these
constants. If you change them you change the paper.

---

## Tests <a id="tests"></a>

```powershell
# Fast unit tests — no Docker required
python -m pytest tests/unit -q                # 104 passing, ~10 s

# Integration smoke — requires VROOM + Valhalla up
python -m pytest tests/integration -m integration -v

# Full pipeline regression — slow, requires Docker
python -m pytest -m slow
```

The unit tests cover schedule enumeration, cost-matrix semantics,
willingness-to-wait blending, holding-day invariants, the surrogate
training loop, the CLI dispatch, and the sweep configuration. Each
source-code split during the refactor was verified by running these
tests; every commit on `refactor/github-ready` keeps all 104 green.

---

## Citation

The paper is currently under review. Until acceptance, please cite it
as a submitted manuscript:

```bibtex
@unpublished{bienzeisler2026tbc,
  title  = {Machine-Learning Surrogate Optimization for Time-Based
            Consolidation in Last-Mile Parcel Delivery},
  author = {Bienzeisler, Lasse and Petre, Felix and Wage, Oskar
            and Friedrich, Bernhard},
  year   = {2026},
  note   = {Manuscript submitted to Transportation Research Procedia
            (EWGT 2026); under review}
}
```

If you use the **code** in this repository directly, you can also cite
the software artefact via [`CITATION.cff`](CITATION.cff) (GitHub renders
a "Cite this repository" button for it).

The BibTeX entry will be replaced with the proper `@inproceedings` form
once the paper is accepted.

---

## License

This research code is released under the MIT License — see
[`LICENSE`](LICENSE) for the full text. HAGRID demand data and the
Region Hannover geodata are NOT included in this repository and have
their own licensing terms; see `data/README.md` (after running the
download script) for the source URLs.

---

## Related projects

* **HAGRID** ([TUBS-IVS/HAGRID](https://github.com/TUBS-IVS/HAGRID)) —
  the parcel-demand generation pipeline maintained at our institute
  (TU Braunschweig, Institute of Transportation and Urban Engineering).
  HAGRID synthesises weekly per-postal-code demand vectors that this
  repository consumes as input. The output of HAGRID is what's expected
  under `data/demand/` (see [`data/README.md`](data/README.md)).
* [VROOM](https://github.com/VROOM-Project/vroom) /
  [Valhalla](https://github.com/valhalla/valhalla) — the open-source
  routing stack used inside `scripts/pipeline/04_validate_vroom.py`.
  Tile data comes from OpenStreetMap.

## Acknowledgements

The Daganzo continuum approximation used as the leading term of the
surrogate follows Daganzo (1984). The Akkerman-style feature taxonomy
used as the LightGBM residual inputs follows Akkerman et al. — see
the bibliography in [`paper/EWGT_2026/FullPaperLMDPCLiteratur.bib`](paper/EWGT_2026/FullPaperLMDPCLiteratur.bib)
for full references.

---

## Contact

For paper-related questions: [lasse.bienzeisler@tu-braunschweig.de](mailto:lasse.bienzeisler@tu-braunschweig.de).
For code issues, please open a GitHub issue.
