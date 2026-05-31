# Reproducing the EWGT 2026 Paper

This document is the step-by-step recipe for reproducing every numeric
result, figure, and table in the submitted paper from a fresh clone of
this repository.

## Prerequisites

| Requirement | Tested with | Notes |
|---|---|---|
| Operating system | Windows 11 / Linux | macOS untested |
| Python | 3.12 or 3.13 | 3.11 may work but is not part of CI |
| RAM | 16 GB minimum | Valhalla needs ~8 GB for the Niedersachsen tile set |
| Disk | 20 GB free | data/ ≈ 400 MB, results/ ≈ 6 GB peak, Valhalla tiles ≈ 4 GB |
| Docker Desktop / Engine | recent | only Stage 4 (VROOM validation) requires it |
| Wall-clock time | ≈ 20 hours | dominated by Stage 2 (16 h) and Stage 4 (3 h per P value) |

## Step 1 — Clone and install

```powershell
git clone https://github.com/<org>/vroom-valhalla-lmd-hannover.git
cd vroom-valhalla-lmd-hannover

# Editable install with dev dependencies
python -m pip install -e ".[dev]"
```

Verify the package imports and tests pass before continuing:

```powershell
python -m pytest tests/unit -q
# Expected: 104 passed, ~10 s
```

## Step 2 — Fetch input data

The HAGRID parcel-demand shapefiles + Region Hannover PLZ geodata + hub
CSVs (~400 MB) are NOT in this repo. Fetch them:

```powershell
# Until the download script is in place, request the data archive from
# the corresponding author (see README contact section) and unpack into
# the `data/` folder. The expected layout is:
#
#   data/
#     demand/        HAGRID *.shp + weekday subsets
#     geodata/       region_hannover.geojson + plz_*.shp + cluster_raumtyp.csv
#     hubs/          kep_hubs.csv
#     vehicles/      hagrid_vehicle_types.csv
```

A future commit will replace this with a `scripts/data/download_inputs.py`
that fetches from a Zenodo DOI.

## Step 3 — Start the routing stack (Stage 4 only)

If you intend to run Stage 4 (VROOM out-of-sample validation), bring up
the Docker stack. Stages 1–3 do NOT require Docker.

```powershell
docker compose up -d

# Wait for the Valhalla container to finish unpacking tiles (~3 min).
# Verify health:
curl http://localhost:3000/health     # VROOM
curl http://localhost:8002/status     # Valhalla
```

## Step 4 — Run the pipeline

The simplest invocation runs all four stages in order:

```powershell
batch-delivery paper
```

To run individual stages (recommended for first-time reproduction so you
can inspect intermediate outputs):

```powershell
# Stage 1: Train Daganzo-LGB-Hybrid surrogate (~30 min)
batch-delivery paper --stage 1
# Output: results/checkpoints/daganzo_hybrid_v3aug_median.pkl
# Verify: python -c "import pickle; m = pickle.load(open('results/checkpoints/daganzo_hybrid_v3aug_median.pkl', 'rb')); print(m)"

# Stage 2: 88-cell coordinate-descent optimization (~16 h)
batch-delivery paper --stage 2
# Output: results/runs/path2_<date>/{tab_balancing_summary.csv, tab_chosen_schedules.csv, tab_fleet_per_hub.csv}
# Verify: should produce 616 rows in tab_balancing_summary.csv (88 cells x 7 providers)

# Stage 3: System-level fleet smoothing (~5 min)
batch-delivery paper --stage 3
# Output: results/runs/path2_<date>/{_system_spread_per_cell.csv, _tab_balancing_summary_with_smoothing.csv}

# Stage 4: VROOM out-of-sample validation (~3 h per P value)
batch-delivery paper --stage 4
# Output: results/paper_results_<date>/07_validation/tab_vroom_path2.csv
```

Or skip Docker entirely:

```powershell
batch-delivery paper --skip-vroom
# Runs stages 1-3 only. Adequate for reproducing the headline cost saving
# numbers but skips the conservatism claim against VROOM.
```

## Step 5 — Generate the figures

Once Stages 1–3 are done, render the EWGT figures:

```powershell
# Render every paper figure individually (each writes to results/EWGT_Results/)
foreach ($f in Get-ChildItem scripts/figures/fig_*.py) {
    python $f.FullName
}

# Or run the paper-assembly script that bundles everything into a dated
# results/paper_outputs_<date>/ folder:
python scripts/paper/paper_final_assembly.py
```

## Step 6 — Verify against the submitted numbers

The frozen submission outputs live in [`paper/EWGT_2026/`](../paper/EWGT_2026/).
Spot-check a few against your regenerated figures:

```powershell
# Compare a CSV byte-for-byte:
fc paper\EWGT_2026\tables\tab_op_kpi_weekly.csv results\paper_outputs_<date>\08_interpretation\tab_op_kpi_weekly.csv

# Visually compare a figure:
start paper\EWGT_2026\figures\fig_grid_heatmap_6.png
start results\EWGT_Results\fig_grid_heatmap_6.png
```

The headline numbers (22.8 %, 13.5 %, 60 %, 2.95 % MAPE, etc.) should
reproduce exactly. See
[`paper/EWGT_2026/MANIFEST.md`](../paper/EWGT_2026/MANIFEST.md) for the
full claim → file map.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'X' from 'batch_delivery.optimization.core'` | Stale `__pycache__` after the 2026-05-31 refactor | `Remove-Item -Recurse __pycache__` then re-run |
| Stage 2 stuck on a single cell for > 30 min | LightGBM hyperparameters mismatched with the training pool | Re-run Stage 1 with the canonical pool |
| Stage 4 returns `503` from VROOM | Valhalla still loading tiles | Wait ~3 min after `docker compose up -d` |
| Tests pass but `batch-delivery paper` errors with `script not found` | scripts/pipeline/ was not installed (editable install needed) | `python -m pip install -e ".[dev]"` |
| `results/` exceeds 10 GB | Cache folder repopulating | Periodically clear `results/cache/` (gitignored) |

## Reduced-scope reproduction

For reviewers who do not have 20 hours:

* **30-minute path:** Stage 1 only + load the supplied
  `results/runs/path2_2026_05_29/` outputs to reproduce only Stages 2–4
  artefacts. This confirms the surrogate trains as documented.
* **Figures-only path:** Skip Stages 1–4 entirely and run the figure
  scripts directly against the included `results/runs/path2_2026_05_29/`
  and `results/paper_results_2026_05_30/` folders. Reproduces every
  paper figure in under 10 minutes.

The included `results/runs/path2_2026_05_29/` matches the
2026-05-30 14:12 orchestrator completion timestamp — see
`orchestrator.log` for the per-cell timing audit trail.
