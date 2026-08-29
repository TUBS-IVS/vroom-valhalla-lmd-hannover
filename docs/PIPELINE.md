# Pipeline Architecture

**There are two paper pipelines in this repository, and their numbers are
not comparable.**

| | Canonical for | Location | Driver |
|---|---|---|---|
| **A. Revision pipeline** | **EWGT 2026 rev1 — every current number** | `scripts/revision/` | run the numbered scripts directly |
| B. Submitted-version pipeline | `paper/EWGT_2026/` (the frozen submission) | `scripts/pipeline/` | `batch-delivery paper` |

Pipeline B is **deprecated for the revision**. It predates the universal
tour rule, the pool term, the two cost lenses and the operator-cost polish,
so a grid it produces is missing all four. It is kept runnable because
reproducing the submitted manuscript is a legitimate need; stages 02–04
raise a `DeprecationWarning` on import and `batch-delivery paper` prints a
notice before it runs anything. Never quote a pipeline-B number beside a
revision number.

Stage 01 of pipeline B (`01_train_surrogate.py`) is the one part still in
force: the revision uses the same surrogate,
`daganzo_hybrid_v3aug_median.pkl` (α = 1.343), trained on
`results/supplementary/sweep_v3_mergefix/training_matrix.csv`.

---

# Part A — The revision pipeline (`scripts/revision/`) — CANONICAL

Long-running, resumable, addressed at a run directory rather than wrapped in
a CLI: `61_` writes to `$REV2_OUT_DIR` (default
`results/revision_2026_08_v6/`) and the downstream stages take `--live-dir`
or `--rev-dir`. The canonical run for rev1 is
`results/revision_2026_08_v6/`; `79_` assembles the shipped pack into
`results/revision_2026_08_final/`. `results/` is gitignored, so the
**scripts are the tracked artefact** — see `results/CANONICAL.md`.

```
 61_ grid ──▶ 62_ gates ──▶ 67_ validation ──▶ 70_ figures/tables ──▶ 71_ sync
      ▲                                              │
      │                                              ├──▶ 72_/73_/74_ derived tables
 63_/64_/64a_/65_ bundle head                        ├──▶ 75_–78_ supplementary figs
      (Gate U certification)                          └──▶ 79_ final pack
```

## Stage order

| # | Script | Role | Key outputs |
|---|---|---|---|
| grid | `61_grid_run_v2.py` | The `(P, θ, provider)` grid under the universal tour rule. Stage 1 is the coordinate-descent **routing-optimal plan**; stage 2 is the **operator-cost polish** (best of three starts). Head-enabled. | `tab_grid_full_v2.csv`, `_tab_chosen_v2.csv`, `tab_costs_v2.csv`, `tab_fleet_per_hub_v2.csv`, `tab_wait_v2.csv`, `tab_head_usage_v2.csv` |
| gates | `62_gates_check.py --live-dir …` | G1a (stage-1 identity), G1b (report-only), G3, G4 (cost corridor). Read-only and live-safe: it may be run while `61_` is still writing. | `gates_report.md` |
| head | `63_bundle_sampler.py` → `64_solve_bundles_vroom.py` → `64a_bundle_coverage.py` → `65_train_bundle_head.py` | The deployment distribution of pooled tours, their VROOM labels, coverage accounting, then the bundle head with out-of-fold **Gate U** certification per bin. | `head_manifest.json`, the head pickle |
| validation | `67_validate_vroom_v2.py --rev-dir …` | Out-of-sample VROOM re-routing of both plans under both lenses, with a cache census and a startup identity gate (Σ predicted == grid cost). | `validation/tab_vroom_v2.csv`, `validation/validation_report.md`, `validation/census.{csv,md}` |
| figures/tables | `70_figs_tables_v2.py --rev-dir …` (helpers in `_figs_tables_v2.py`) | Every revision figure and table, plus the legacy-schema bridge. Carries a CO₂ completeness gate against the recorded target. | `figures/*.{pdf,png}`, `tables/tab_*.{csv,tex}` |
| sync | `71_sync_paper_figs.py --rev-dir … --include-companions` | md5-verified copy of the figures into `paper/EWGT_2026_rev1/figures/` and `elsevier_source/`. **Refuses** a destination byte-identical to the frozen submission, and refuses an unmapped stem. | tracked PDFs under `paper/EWGT_2026_rev1/` |
| derived | `72_per_cell_costs_v2.py`, `73_tables_ops_v2.py`, `74_v2_to_legacy_tables.py` | Per-cell plan costs; ops / P⋆-knee / value-of-stage-2 tables; the adapter that lets the frozen submitted-layout builders read a v6 grid. | `tab_per_cell_*.csv`, `tab_pstar_knees_v2.csv`, `legacy/` |
| supplementary | `75_fig_fleet_week_classes.py`, `76_maps_v2.py`, `77_mechanism_v2.py`, `78_fleet_week_v2.py` | Mon–Sat fleet profile per carrier class (indexed) and per LSP (absolute — see the script's own disclosure note), spatial figures, the mechanism figure. | `figures/supp_*.{pdf,png}` |
| pack | `79_build_final_pack.py` | Copies (never moves, never edits) the v6 outputs into `results/revision_2026_08_final/` with a README listing every file, its producing script, grid and md5. | `results/revision_2026_08_final/` |

## Guard tripwires that must stay armed

Named here because a silent regression in any of them is the failure mode
each was built for:

* `scripts/paper/guard_tex.py` — 17 pages / 23 bibitems / no swallowed
  control sequence in `paper/EWGT_2026_rev1/tbc_preprint_main.tex`. Wired
  as `tests/unit/test_guard_tex.py`.
* `71_` — the identical-to-submission refusal and the unmapped-stem refusal.
* `70_` — the CO₂ completeness gate against the recorded target.
* `62_` — the G1a tolerance bounds (`G1A_TOLERANCE_FLAT_EUR = 20.0`,
  `G1A_TOLERANCE_REL = 0.005`).
* `src/batch_delivery/surrogate/bundle.py` — the tercile-edge drift check.

---

# Part B — The submitted-version pipeline (`scripts/pipeline/`) — DEPRECATED for rev1

Four stages, each a standalone script, chained by `batch-delivery paper`.
The sections below describe the inputs, outputs and invariants of every
stage so you can run, restart or debug them in isolation. They reproduce
`paper/EWGT_2026/`, not the revision.

## High-level diagram

```
+-----------+   +-----------+   +-----------+   +-----------+
| Stage 1   |   | Stage 2   |   | Stage 3   |   | Stage 4   |
|           |   |           |   |           |   |           |
|  Train    +-->| Optimize  +-->| Smooth    +-->| VROOM     |
| surrogate |   | (P, theta)|   | system    |   | validate  |
|           |   | grid (88) |   | fleet     |   | (3-4 P)   |
+-----------+   +-----------+   +-----------+   +-----------+
      |              |               |              |
      v              v               v              v
  Daganzo-LGB    runs/path2/    runs/path2/      paper_results/
  Hybrid pkl     tab_*.csv      _system_*.csv    07_validation/
                                                  tab_vroom_*.csv
       \           |               |              /
        +----------+--------+------+-------------+
                            v
                  +--------------------+
                  | scripts/figures/   |
                  | scripts/paper/     |
                  +--------------------+
                            v
                  paper/EWGT_2026/
                  (frozen submission)
```

## Stage 1 — Train surrogate (`pipeline/01_train_surrogate.py`)

**Purpose:** Fit the Daganzo-LightGBM hybrid surrogate that replaces VROOM
inside the optimization loop.

**Architecture:**

```
cost_predicted = alpha * daganzo_base(n, n_stops, area, hub_dist)
                 + LGB(features) * sigma_resid
```

* `alpha` is calibrated to median ratio (production: 1.343).
* `daganzo_base` is the closed-form continuum approximation from
  [Daganzo 1984](https://doi.org/10.1287/trsc.18.4.331) applied per cell.
* `LGB(features)` is a LightGBM regressor on the 25-column Akkerman-style
  feature vector trained with GroupKFold over postal codes.

**Inputs:**
* `results/supplementary/sweep_v3_mergefix/training_matrix.csv` —
  ~2700 routed (provider, PLZ, schedule) samples.
* `src/batch_delivery/features/` — the 25-column feature builder.

**Outputs:**
* `daganzo_hybrid_v3aug_median.pkl` — the trained predictor (gitignored
  bulk artefact; regenerable).

**Performance:** 2.95 % MAPE under postal-code-grouped 5-fold CV. R² = 0.99
on the optimized scenarios used by Stage 2.

**Runtime:** ~30 minutes on 4 cores.

---

## Stage 2 — Optimize (P, theta) grid (`pipeline/02_optimize_grid.py`)

**Purpose:** Run coordinate descent over the 88-cell `(P, theta)` grid for
every (provider, PLZ) cell, then apply per-hub fleet balancing.

**Grid:**

| Dimension | Values |
|---|---|
| Service penalty `P` (EUR/parcel·day) | {0, 0.25, 0.5, 0.75, 1, 2, 5, 10} |
| Willingness-to-wait share `theta` | {0, 0.1, 0.2, ..., 1.0} |
| Carriers | 7 LSPs (Amazon, DHL, DPD, FedEx, GLS, Hermes, UPS) |
| Postal-code areas | 39–47 per LSP, 312 total cells |

Each cell runs:
1. Coordinate descent on the (PLZ, schedule) joint cost surface using the
   surrogate from Stage 1.
2. Pair-swap polish at convergence.
3. Per-hub fleet balancing: swap schedules within each hub so daily vehicle
   counts equalize, subject to a cost-increase budget.

**Inputs:**
* Trained surrogate from Stage 1.
* `data/` (HAGRID demand + PLZ geodata + hub assignments).

**Outputs (`results/runs/path2_<date>/`):**
* `tab_balancing_summary.csv` — per-cell initial + balanced cost.
* `tab_chosen_schedules.csv` — per-(provider, PLZ) chosen schedules.
* `tab_fleet_per_hub.csv` — daily fleet counts before/after balancing.
* `orchestrator.log` — append-only timing log per cell.
* `state.json` — completed cell list for crash recovery.

**Runtime:** ~16 hours on 4 cores. The orchestrator checkpoints per cell
so you can interrupt and resume.

---

## Stage 3 — Apply system smoothing (`pipeline/03_apply_smoothing.py`)

**Purpose:** A second balancing pass at the SYSTEM level: when two hubs end
up with high cross-hub fleet spread, exchange one cell's schedule between
them to flatten the system-wide load.

**Algorithm:**

```
for each (P, theta) cell:
    repeat until convergence (or max_swaps reached):
        find the (hub_i, hub_j, day) triple with worst system spread
        if a (cell_a in hub_i, cell_b in hub_j) swap reduces spread:
            swap them, subject to cost-increase budget
```

This typically affects 0–20 cells per (P, theta), with cost impact
< 0.25 % and system-spread reduction up to 60 %.

**Inputs:** Stage 2 outputs.

**Outputs (`results/runs/path2_<date>/`):**
* `_system_spread_per_cell.csv` — per-cell spread before/after.
* `_tab_balancing_summary_with_smoothing.csv` — costs post-smoothing.
* `_tab_chosen_with_system_smoothing.csv` — schedule rotations applied.

**Runtime:** ~5 minutes (single-threaded, mostly NumPy).

---

## Stage 4 — VROOM validate (`pipeline/04_validate_vroom.py`)

**Purpose:** Route the optimized weekly schedules through VROOM + Valhalla
and compare the ML-predicted per-cell cost to the routed actual cost. This
defended the submitted paper's claim that the surrogate is **conservative** —
that it underestimates achievable savings by 1.3–2.1 percentage points.

> **Withdrawn in the revision.** That comparison predates the universal tour
> rule and never re-routed the daily baseline itself. `67_validate_vroom_v2.py`
> does, and finds the surrogate **over-prices** every instance class, the thin
> baseline tours most, so predicted savings are an *upper* bound and the
> realized figures are 1.3–2.5 pp (operator-polished plan) and 2.1–3.7 pp
> (routing-optimal plan) *lower*. Use Part A for any current validation claim.

**Procedure:**

```
for cell in (theta=1, P in {0, 0.25, 0.5, 0.75}):
    for each (provider, PLZ) cell in the grid:
        load the chosen schedule from Stage 2 output
        for each delivery day:
            build VROOM request for that day's parcels
            POST to VROOM, parse routes, sum cost
        sum to weekly cost
        compare to dd_cost_ml from Stage 2
```

**Inputs:**
* Stage 2 outputs.
* Live VROOM (port 3000) + Valhalla (port 8002) Docker services.

**Outputs (`results/paper_results_<date>/07_validation/`):**
* `tab_vroom_path2.csv` — per-cell ML vs VROOM cost, 312 rows per P.
* `tab_vroom_balanced.csv` — same on the balanced schedules.

**Runtime:** ~3 hours per P value (Docker stack must be healthy and the
laptop must have ~8 GB free RAM for Valhalla tiles).

---

## Configuration

All scenario parameters live in [`conf/default.yaml`](../conf/default.yaml).
Sweep configurations for stages 2 and 3 are in `conf/sweep_*.yaml`. See
[`src/batch_delivery/config/schema.py`](../src/batch_delivery/config/schema.py)
for the Pydantic validation surface — out-of-range values fail the load
before any work is started.

## Checkpoint cache

All stages use the same checkpoint convention: each stage writes a pickle
to `results/checkpoints/<stage_id>.pkl` (gitignored) that the next stage
can pick up. Re-running a stage with an existing checkpoint overwrites it.
Pass `--no-cache` to `batch-delivery run` (the per-stage flag is documented
inside each script) to force a full recompute.

## Tests

Each stage has corresponding unit tests under `tests/unit/`:

| Stage | Coverage in |
|---|---|
| Surrogate (Stage 1) | `test_surrogate_train.py`, `test_surrogate_shape.py` |
| Optimization (Stage 2) | `test_cost_matrices_new_semantics.py`, `test_optimization_polish_and_results.py`, `test_holding_days_invariant.py`, `test_stops_scale_with_willing.py`, `test_delivery_frequency_share_aware.py` |
| CLI dispatch | `test_cli.py`, `test_config.py` |
| Sweep runner | `test_sweep_*.py` |
| Runtime | `test_runtime.py`, `test_imports.py` |

The integration smoke test (`tests/integration/test_pipeline_smoke.py`)
runs all stages on a single synthetic mini-region in under 60 s. It only
runs when `pytest -m integration` is passed.
