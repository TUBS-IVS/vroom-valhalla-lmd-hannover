# `scripts/` — Pipeline, Figures, and Research Tooling

Scripts are sorted into six folders by role. Anything that backs the paper
is in `pipeline/`, `figures/`, `paper/`, or `data/`. The rest is the
documented research process.

## Directory map

| Folder | Files | Role | Tracked? |
|---|---|---|---|
| `pipeline/` | 4 | The four canonical pipeline stages that produce the paper outputs end-to-end. Numbered so the run order is unambiguous. | yes |
| `figures/` | 26 | One script per figure (or figure group) in the EWGT 2026 paper. | yes |
| `paper/` | 44 | Paper-output builders: assembly, tables, break-even analyses, sensitivity, sweetspot identification. Glue between the optimization run and the figures. | yes |
| `data/` | 12 | Input-data preparation: PLZ clustering, raumtyp assignment, training-pool builds, coverage audits. | yes |
| `exploratory/` | 24 | Diagnostic / sensitivity / sanity-check scripts used during research. Tracked so the research process stays auditable. | yes |
| `_archive/` | 77 | Superseded versions (v1, v2 where v3+ exists). Kept for paper-review provenance. | yes |

## Canonical Pipeline (`pipeline/`)

Run these four scripts in order to reproduce every paper number from scratch
(assuming `data/` has been fetched via `scripts/data/download_inputs.py` and
the Docker stack is up).

| Stage | Script | Inputs | Outputs |
|---|---|---|---|
| 01 | `01_train_surrogate.py` | `results/supplementary/sweep_v3_mergefix/training_matrix.csv` | `daganzo_hybrid_v3aug_median.pkl` (surrogate) |
| 02 | `02_optimize_grid.py` | trained surrogate + `data/` | `results/runs/path2_2026_05_29/tab_*.csv` (88-cell grid) |
| 03 | `03_apply_smoothing.py` | path2 run | `results/runs/path2_2026_05_29/_tab_*_with_smoothing.csv` |
| 04 | `04_validate_vroom.py` | path2 run + Docker stack | `results/paper_results_2026_05_30/07_validation/tab_vroom_*.csv` |

After the four stages, run figures and assembly:

```powershell
# Render all EWGT figures
python -m scripts.figures.fig_combined_heatmap
python -m scripts.figures.fig_structural_grid
# ... (or use the Makefile target: make figures)

# Assemble paper outputs into a single dated folder
python scripts/paper/paper_final_assembly.py
```

The Typer CLI command `batch-delivery paper` orchestrates the full sequence;
see `docs/PIPELINE.md` for the architecture diagram.

## Figures (`figures/`)

One script per figure. Naming convention: `fig_<NAME>.py`. Each script reads
from `results/runs/path2_2026_05_29/` and writes to
`results/paper_ewgt_2026/` (frozen) or to a dated work folder
under `results/paper_outputs_*/` (active development).

Notable scripts:

| Figure (paper) | Script |
|---|---|
| Combined 6-panel heatmap (cost, wait, fleet) | `fig_combined_heatmap.py` |
| Structural grid (Pareto + carrier classes + raumtyp + quartiles) | `fig_structural_grid.py` |
| 8-P × theta mix shares | `fig_SM_mix_8P.py` |
| Sweet-spot λ sweep (dominance regions) | `fig_sweetspot_lambda_sweep.py` |
| PLZ structural correlation (3-panel) | `fig_plz_structural_correlation.py` |
| VROOM-vs-ML diagnostics (parity + per-day MAPE) | `fig_vroom_diagnostics.py` |
| Operational KPI dashboard | `fig_operational_kpi.py` |
| Value of optimization vs naive | `fig_value_of_path2_vs_naive.py` |

## Paper output builders (`paper/`)

These scripts take pipeline outputs and produce paper-ready tables,
break-even analyses, sweet-spot identification, and the multi-chapter
paper-output assembly. `paper_final_assembly.py` is the master orchestrator;
the rest are individual builders called from it.

## Archive policy

`_archive/` contains superseded versions kept ONLY for paper-review
nachvollziehbarkeit:

- v1/v2 of scripts where v3+ is canonical (`run_final_optimization*` →
  superseded by `pipeline/02_optimize_grid.py`)
- Older non-balanced orchestrators
- Pre-path2 paper-output assemblies (the older `paper_final_*` copies
  before they were re-cut for the canonical path2 run)
- One-off diagnostics that produced no paper number (bias correction,
  distribution shift, model debiasing)

If a future you needs an archived script, copy it back out — do not run
it in place.
