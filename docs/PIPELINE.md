# Pipeline Architecture

The EWGT 2026 paper pipeline is split into **four canonical stages**, each
exposed as a standalone Python script under `scripts/pipeline/` and chained
by the `batch-delivery paper` CLI command. This document describes the
inputs, outputs, and invariants for every stage so you can run, restart, or
debug them in isolation.

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
defends the paper's claim that the surrogate is **conservative** — it
underestimates achievable savings by 1.3–2.1 percentage points.

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
