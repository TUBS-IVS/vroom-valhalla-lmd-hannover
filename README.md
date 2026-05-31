# `batch-delivery` — ML Surrogate Optimization for Time-Based Parcel Consolidation

Companion code and reproducibility package for:

> **Bienzeisler, L., Petre, F., Wage, O., and Friedrich, B.**, *"Machine-Learning
> Surrogate Optimization for Time-Based Consolidation in Last-Mile Parcel
> Delivery"*, submitted to Transportation Research Procedia (EWGT 2026),
> preprint 2026-05-31 (currently under review).

[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 127 passing](https://img.shields.io/badge/tests-127%20passing-brightgreen.svg)](#tests)
[![Paper: EWGT 2026 (submitted)](https://img.shields.io/badge/paper-EWGT%202026%20%E2%80%94%20submitted-orange.svg)](paper/EWGT_2026/ABSTRACT.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Status:** the paper is **submitted to EWGT 2026, not yet accepted.**
> This repo is published alongside the submission as the reproducibility
> package and will be updated for the camera-ready version once review
> feedback is in.

---

## What this project is about

E-commerce parcel volume in European cities continues to grow at
double-digit rates while urban curb space and driver labour stay
fixed. The status-quo response — daily delivery to every household,
seven LSPs running parallel routes through the same neighbourhoods —
is increasingly difficult to scale. **Time-based consolidation (TBC)**
asks whether the system could move more parcels with fewer tours by
*delaying* eligible shipments by a few days and merging them into
fewer, larger delivery rounds.

The trade-off is intuitive but its magnitude is hard to quantify:

- Customers wait a bit longer for non-urgent parcels.
- Carriers save fuel, kilometres, drivers, and capital.
- The system gains a flatter weekly fleet profile, easier to staff
  and electrify.

The questions the paper answers — and the questions this repository
**lets a reader re-examine for their own city** — are:

1. **How big is the win?** For the Hanover Region (seven LSPs,
   1.26 M parcels/week), what is the achievable cost reduction at
   each (service-penalty, willingness-to-wait) operating point?
2. **Where does it pay most?** Which postal-code areas have the
   largest savings — and which structural features (hub distance,
   area size, parcels per stop) predict it?
3. **How does it differ by carrier type?** Do service-bound carriers
   (Amazon, DHL) behave like cost-aggressive ones (DPD, GLS), or do
   distinct operating regimes emerge?
4. **Is the surrogate trustworthy?** A surrogate is only useful if it
   matches the real solver. How wide is the gap between the ML cost
   prediction and out-of-sample VROOM re-routing?

The repository contains every piece of code, configuration, test, and
documentation needed to answer those questions from scratch on a fresh
machine.

## Headline numbers

| Metric | Value |
|---|---|
| Cost saving at the cost-optimal extreme (P = 0, θ = 1) | **22.8 %** |
| Cost saving at the operating sweet spot (P = 0.5, θ = 1) | **13.5 %** |
| VROOM-validated saving at P = 0.5 (surrogate is conservative) | **15.6 %** |
| Mo–Sa fleet coefficient-of-variation reduction in the efficient range | **up to 60 %** |
| Peak fleet reduction at (P = 0.5, θ = 1) | **12.9 %** |
| Daganzo-LightGBM hybrid surrogate MAPE (GroupKFold over postal codes) | **2.95 %** |
| Surrogate conservatism band (vs VROOM re-routing) | **1.3 – 2.1 pp** |

Three carrier classes emerge from the optimization with distinct
optimal penalty levels:

- **Service-bound** (Amazon, DHL) → P\* = 0.25 € / parcel·day
- **Hybrid** (FedEx, Hermes, UPS) → P\* = 0.5 € / parcel·day
- **Cost-aggressive** (DPD, GLS) → P\* = 0.75 € / parcel·day

TBC pays most in rural postal-code areas with long stems, large
service area, and few delivered parcels per stop.

Every number above traces to a specific file in
[`paper/EWGT_2026/MANIFEST.md`](paper/EWGT_2026/MANIFEST.md).

---

## Idea & method in one minute

The simplest version of the optimization is one number per
(LSP, postal code, week):

> *Pick the weekly delivery schedule (a subset of {Mon, …, Sat})
> that minimises operating cost plus a service penalty for the
> resulting customer waiting time, subject to a maximum holding time
> of three days per parcel.*

The naive way to do that is to try every schedule and call VROOM for
each one — **39 admissible weekly patterns × 312 (provider, postal
code) cells = 12 168 VROOM calls per cell of the (P, θ) sweep**, and
the paper sweeps an 88-cell grid. That's ~1 million VROOM calls per
experiment, weeks of compute.

The paper replaces VROOM in the inner loop with a much cheaper
**Daganzo-LightGBM hybrid surrogate**:

```
cost_predicted = alpha * daganzo_base(n, n_stops, area, hub_dist)
                 + LGB(features) * sigma_resid
```

* `daganzo_base` is the closed-form continuum approximation from
  Daganzo (1984) — fast, calibrated to the local routing setting via a
  scalar `alpha = 1.343`.
* `LGB(features)` is a gradient-boosted residual on 25 Akkerman-style
  spatial / demand features, trained on ~2 700 routed samples with
  postal-code-grouped 5-fold cross-validation (final MAPE: 2.95 %).
* The hybrid takes ~5 ms per query versus ~30 s for VROOM, so the
  inner loop is now ~10⁵ × faster.

Around that surrogate, the four-stage pipeline runs:

1. **Train** the surrogate on a fixed routed sample pool.
2. **Optimize** weekly schedules per (provider, postal code) via
   coordinate descent over the (P, θ) grid.
3. **Smooth** the resulting fleet schedules so daily vehicle counts
   are balanced both within each hub and across the system.
4. **Validate** out-of-sample by routing the chosen schedules through
   VROOM and measuring the surrogate's conservatism.

Full architecture: [`docs/PIPELINE.md`](docs/PIPELINE.md).
Mathematical formulation: §2 of the
[paper preprint](paper/EWGT_2026/EWGT26_Full_Paper_LB_preprint.pdf).

---

## What you need to run it

### Inputs

| Input | Source | Why it's needed |
|---|---|---|
| Weekly per-postal-code parcel demand per LSP | [HAGRID](https://github.com/TUBS-IVS/HAGRID) — our sister project at the same institute (TU Braunschweig, IVS) | Drives the cost function for every schedule. Without it the optimization has nothing to optimize. |
| Postal-code geometry for the study region | OpenStreetMap extract via Geofabrik | Defines per-PLZ service-area features (area, stem distance). |
| Hub / depot CSV (one row per (LSP, hub)) | Hand-curated from public LSP information + Google Maps geocoding | Anchors stem-distance computations and feeds the per-hub fleet-balancing pass. |
| Vehicle type specifications | HAGRID's vehicle catalog | Defines capacity, fixed cost, per-km cost. |
| OSM road geometry for routing | Geofabrik Niedersachsen-Bremen extract | Required by Valhalla for the VROOM validation stage. |

These inputs are NOT distributed in this repository. See
[`data/README.md`](data/README.md) for how to obtain them. A future
commit will replace the manual request with a download script that
fetches from a Zenodo DOI.

### Software

| Requirement | Tested with | Notes |
|---|---|---|
| Operating system | Windows 11, Linux | macOS untested |
| Python | 3.12 or 3.13 | 3.11 may work but is not part of CI |
| RAM | 16 GB minimum | Valhalla needs ~8 GB for the Niedersachsen tile set |
| Disk | 20 GB free | `data/` ≈ 400 MB, peak `results/` ≈ 6 GB |
| Docker Desktop / Engine | recent | only Stage 4 (VROOM validation) needs it |

### Outputs

After a full pipeline run, the repository contains:

```
results/
├── runs/path2_<date>/        ← optimization grid (88 cells × 7 providers)
├── paper_outputs_<date>/     ← 13-chapter assembly: figures + tables
├── paper_results_<date>/     ← chapter-organised paper cut + VROOM validation
└── supplementary/            ← model battery, sensitivity, region maps

paper/EWGT_2026/
├── EWGT26_Full_Paper_LB_preprint.pdf   ← compiled preprint
├── tbc_preprint_main.tex                ← TeX source
├── FullPaperLMDPCLiteratur.bib          ← bibliography
├── figures/                              ← the 6 numbered paper figures
├── tables/                               ← 6 paper-cited CSVs
├── MANIFEST.md                           ← claim → file → script map
└── ABSTRACT.md                           ← frozen submitted abstract
```

The submitted figures and tables are byte-identical to what's inside
the compiled PDF.

---

## Repository structure

```
.
├── paper/EWGT_2026/        ← FROZEN: tex source, figures, tables, manifest
├── src/batch_delivery/     ← installable Python package
│   ├── config/             constants + Pydantic schema + invariants
│   ├── io/                 HAGRID demand + hub assignment
│   ├── routing/            VROOM / Valhalla client + cache + request builders
│   ├── features/           Akkerman-style spatial / demand features
│   ├── surrogate/          Daganzo-LightGBM hybrid trainer + predictor
│   ├── optimization/       schedule enum + CD + SA + fleet balancing
│   ├── evaluation/         KPI tables + scenario comparison
│   ├── pipeline/           seven-stage orchestrator + RunContext
│   ├── cli/                Typer CLI (batch-delivery <subcommand>)
│   └── legacy/             Daganzo continuum proxy (ablation only)
├── scripts/
│   ├── pipeline/           four canonical pipeline stages (01..04)
│   ├── figures/            one script per paper figure
│   ├── paper/              paper-output builders (assembly, sweetspot, ...)
│   ├── data/               input-data preparation
│   ├── exploratory/        research-process scripts (diagnostic, sensitivity)
│   └── _archive/           superseded versions (kept for review trail)
├── tests/
│   ├── unit/               104 unit tests (fast, no Docker)
│   ├── reproducibility/     23 tests verifying paper-claim integrity
│   └── integration/         smoke tests that touch VROOM
├── conf/                   YAML configuration (default + sweep variants)
├── data/                   inputs (gitignored, see data/README.md)
├── results/                pipeline outputs (canonical + supplementary tracked)
├── docs/                   PIPELINE, REPRODUCING_PAPER, CHANGELOG
├── CITATION.cff            GitHub-native software citation metadata
├── LICENSE                 MIT
└── pyproject.toml          editable-install entry point
```

Detailed per-folder inventories:
[`results/CANONICAL.md`](results/CANONICAL.md) ·
[`scripts/README.md`](scripts/README.md) ·
[`docs/PIPELINE.md`](docs/PIPELINE.md).

---

## Quickstart

### 1. Install

```powershell
# Editable install with dev dependencies
python -m pip install -e ".[dev]"
```

This installs `batch-delivery` as a console script and pulls all
runtime + test + lint dependencies.

### 2. Bring up the routing stack (optional, needed for Stage 4)

```powershell
docker compose up -d
# VROOM:    http://localhost:3000
# Valhalla: http://localhost:8002
# Wait ~3 min after first start for Valhalla to unpack tiles.
```

### 3. Verify everything imports

```powershell
batch-delivery --help                # lists all subcommands
batch-delivery version
batch-delivery config validate       # validates conf/default.yaml
batch-delivery schedules             # lists the 39 weekly patterns
python -m pytest tests/unit tests/reproducibility -q
# Expected: 127 passing, ~10 s
```

### 4. Reproduce the paper

```powershell
# Show the four-stage plan without running anything:
batch-delivery paper --dry-run

# Run the full pipeline end-to-end (~20 h wall-clock):
batch-delivery paper

# Or run stages individually:
batch-delivery paper --stage 1     # surrogate training (~30 min)
batch-delivery paper --stage 2     # 88-cell optimization (~16 h)
batch-delivery paper --stage 3     # system-level smoothing (<5 min)
batch-delivery paper --stage 4     # VROOM out-of-sample validation (~3 h)

# Skip Docker entirely (stages 1-3 only):
batch-delivery paper --skip-vroom
```

See [`docs/REPRODUCING_PAPER.md`](docs/REPRODUCING_PAPER.md) for the
full step-by-step recipe, troubleshooting, and reduced-scope
reproduction paths (figures-only in 10 minutes, no-VROOM in ~17 h).

---

## Configuration

Scenario knobs live in [`conf/default.yaml`](conf/default.yaml) and
are validated against the Pydantic schema in
[`src/batch_delivery/config/schema.py`](src/batch_delivery/config/schema.py).
The schema rejects out-of-range values before any compute is started.

Key parameters:

| Section | Field | Default | Meaning |
|---|---|---|---|
| `providers` | list | seven Hanover Region LSPs | Carriers included in the optimization. |
| `pipeline.max_holding_days` | int | **3** | Hard upper bound on parcel waiting time. **Invariant** — do not change without re-running every paper number. |
| `pipeline.fast_share_b2c` | float | 0.30 | Fraction of B2C parcels that are express (skip batching). |
| `pipeline.fast_share_b2b` | float | 0.50 | Same for B2B. |
| `optimization.penalty_grid` | list | `[0, 0.25, 0.5, 0.75, 1, 2, 5, 10]` | Service penalties `P` swept by Stage 2. |
| `optimization.share_grid` | list | `np.arange(0, 1.01, 0.1)` | Willingness-to-wait shares `θ` swept by Stage 2. |
| `routing.vroom_url` | str | `http://localhost:3000` | VROOM endpoint for Stage 4. |
| `routing.valhalla_url` | str | `http://localhost:8002` | Valhalla endpoint. |

Sweep variants for ablation / sensitivity studies are in
[`conf/sweep_*.yaml`](conf/).

## Hard invariants

The codebase enforces three invariants at import time, in the
Pydantic config schema, AND in the test suite. Do not weaken them
without explicit author approval — every paper number depends on
them.

| Invariant | Value | Enforced in |
|---|---|---|
| Maximum holding days per parcel | `MAX_HOLDING_DAYS = 3` | [`config/constants.py`](src/batch_delivery/config/constants.py), [`config/validation.py`](src/batch_delivery/config/validation.py), [`tests/unit/test_holding_days_invariant.py`](tests/unit/test_holding_days_invariant.py) |
| Operating week | Monday–Saturday (6 days) | same |
| Feasible weekly delivery patterns | `EXPECTED_PATTERN_COUNT_K3 = 39` | same |

---

## Architecture overview

### Source-code layout

The Python package follows a strict "one responsibility per module"
layout. Four originally-monolithic files
(`optimization/core.py` 2 751 lines, `cli.py` 1 793, `routing/core.py`
1 181, `pipeline.py` 869) were split during the 2026-05-31 refactor
into focused submodules. Each split keeps the original module name as
a backwards-compatible re-export shim so existing imports continue to
work.

See [`docs/CHANGELOG.md`](docs/CHANGELOG.md) for the full refactor
log and [`docs/PIPELINE.md`](docs/PIPELINE.md) for per-stage
documentation.

### Pipeline (four canonical stages)

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

Each stage is a standalone script under
[`scripts/pipeline/`](scripts/pipeline/), so it can be run,
restarted, or debugged in isolation. The `batch-delivery paper`
command chains them with stop-on-error and dry-run support.

---

## Tests <a id="tests"></a>

```powershell
# Fast unit tests — no Docker required (~10 s, 104 passing)
python -m pytest tests/unit -q

# Reproducibility tests — verify the paper can still be regenerated
# (~1 s, 23 passing)
python -m pytest tests/reproducibility -q

# Integration smoke — touches VROOM + Valhalla, ~60 s
python -m pytest tests/integration -m integration -v

# Full pipeline regression — slow, requires Docker
python -m pytest -m slow
```

The reproducibility suite is the one to watch: it verifies that every
file referenced in [`paper/EWGT_2026/MANIFEST.md`](paper/EWGT_2026/MANIFEST.md)
exists, that every canonical results CSV has the expected schema, and
that the documented cost-saving headline numbers (22.8 %, 18.6 %,
13.5 %) are within ±1.5 pp of the in-repo values.

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
the software artefact via [`CITATION.cff`](CITATION.cff) (GitHub
renders a "Cite this repository" button for it).

The BibTeX entry will be replaced with the proper `@inproceedings`
form once the paper is accepted.

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
[`paper/EWGT_2026/FullPaperLMDPCLiteratur.bib`](paper/EWGT_2026/FullPaperLMDPCLiteratur.bib)
for full references.

## License

MIT — see [`LICENSE`](LICENSE) for the full text. The HAGRID demand
data and the Region Hannover geodata are NOT covered by this license;
see [`data/README.md`](data/README.md) for their respective terms.

## Contact

For paper-related questions:
[lasse.bienzeisler@tu-braunschweig.de](mailto:lasse.bienzeisler@tu-braunschweig.de).
For code issues, please open a GitHub issue.
