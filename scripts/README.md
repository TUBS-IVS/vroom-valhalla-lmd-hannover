# `scripts/` — Pipeline, Figures, and Research Tooling

Scripts are sorted into seven folders by role. **The canonical pipeline for
the EWGT 2026 revision (rev1) is `revision/`.** `pipeline/` is the
*submitted*-version pipeline, kept for reproducibility of
`paper/EWGT_2026/` and deprecated for the revision. The rest is the
documented research process.

## Directory map

| Folder | Files | Role | Tracked? |
|---|---|---|---|
| `revision/` | 32 | **Canonical for EWGT rev1.** The numbered `61_` … `79_` sequence that produces every current number; six superseded `10_` … `50_` stages remain and warn on import. | yes |
| `pipeline/` | 4 | The four **submitted-version** stages. Numbered so the run order is unambiguous. Deprecated for the revision (see below). | yes |
| `figures/` | 26 | One script per figure (or figure group) in the **submitted** EWGT 2026 paper. The revision's figures come from `revision/70_`, `75_`–`78_`. | yes |
| `paper/` | 45 | Paper-output builders: assembly, tables, break-even analyses, sensitivity, sweetspot identification. 19 of them warn on import (submitted-version builders). Also holds `guard_tex.py`, the manuscript structure guard, which is **current**. | yes |
| `data/` | 12 | Input-data preparation: PLZ clustering, raumtyp assignment, training-pool builds, coverage audits. | yes |
| `exploratory/` | 24 | Diagnostic / sensitivity / sanity-check scripts used during research. Tracked so the research process stays auditable. | yes |
| `_archive/` | 77 | Superseded versions (v1, v2 where v3+ exists). Kept for paper-review provenance. | yes |

## Canonical Pipeline for EWGT rev1 (`revision/`)

There is deliberately **no CLI wrapper**: these stages run for hours, are
resumable, and are addressed at a run directory. `61_` writes to
`$REV2_OUT_DIR` (default `results/revision_2026_08_v6/`); the downstream
stages take `--live-dir` or `--rev-dir`. `results/` is gitignored, so these
scripts are the tracked artefact.

| Stage | Script | Inputs | Outputs |
|---|---|---|---|
| grid | `61_grid_run_v2.py` | surrogate pickle + bundle head + `data/` | `tab_grid_full_v2.csv`, `_tab_chosen_v2.csv`, `tab_costs_v2.csv`, `tab_fleet_per_hub_v2.csv`, `tab_wait_v2.csv`, `tab_head_usage_v2.csv` |
| gates | `62_gates_check.py --live-dir <run>` | a live or finished run dir | `gates_report.md` (G1a / G1b / G3 / G4) |
| head | `63_bundle_sampler.py` → `64_solve_bundles_vroom.py` → `64a_bundle_coverage.py` → `65_train_bundle_head.py` | grid manifest + Docker stack | bundle pool, head pickle, `head_manifest.json`, Gate U report |
| validation | `67_validate_vroom_v2.py --rev-dir <run>` | run dir + Docker stack | `validation/tab_vroom_v2.csv`, `validation/validation_report.md`, `validation/census.{csv,md}` |
| figures/tables | `70_figs_tables_v2.py --rev-dir <run>` | run dir (+ `72_` per-cell tables) | `figures/*.{pdf,png}`, `tables/tab_*.{csv,tex}` |
| sync | `71_sync_paper_figs.py --rev-dir <run> --include-companions` | run dir figures | md5-verified PDFs in `paper/EWGT_2026_rev1/figures/` and `elsevier_source/` |
| derived | `72_per_cell_costs_v2.py`, `73_tables_ops_v2.py`, `74_v2_to_legacy_tables.py` | run dir | per-cell plan costs, ops/knee/value-of-stage-2 tables, legacy-schema bridge |
| supplementary | `75_fig_fleet_week_classes.py`, `76_maps_v2.py`, `77_mechanism_v2.py`, `78_fleet_week_v2.py` | run dir | `figures/supp_*.{pdf,png}` |
| pack | `79_build_final_pack.py` | run dir | `results/revision_2026_08_final/` with per-file provenance README |

`71_` refuses to write a destination byte-identical to the frozen
submission, and refuses an unmapped stem. Never `git add`
`paper/EWGT_2026_rev1/elsevier_source/` — `71_` rewrites it on every sync
and it is gitignored.

See `docs/PIPELINE.md` for the stage-by-stage detail and the guard
tripwires, and `results/CANONICAL.md` for which run directory is canonical.

## Submitted-version pipeline (`pipeline/`) — DEPRECATED for rev1

These four scripts reproduce `paper/EWGT_2026/`, the submitted manuscript.
They predate the universal tour rule, the pool term, the two cost lenses and
the operator-cost polish, so **their numbers are not comparable with the
revision's and must never be quoted beside them.** Stages 02–04 raise a
`DeprecationWarning` on import; `batch-delivery paper` prints the same
notice before it runs anything but stays runnable, because reproducing the
submitted version is a legitimate need.

Stage 01 is the exception and is **not** deprecated: the revision uses the
same surrogate it trains.

| Stage | Script | Inputs | Outputs |
|---|---|---|---|
| 01 | `01_train_surrogate.py` | `results/supplementary/sweep_v3_mergefix/training_matrix.csv` | `daganzo_hybrid_v3aug_median.pkl` (surrogate) |
| 02 | `02_optimize_grid.py` | trained surrogate + `data/` | `results/runs/path2_2026_05_29/tab_*.csv` (88-cell grid) |
| 03 | `03_apply_smoothing.py` | path2 run | `results/runs/path2_2026_05_29/_tab_*_with_smoothing.csv` |
| 04 | `04_validate_vroom.py` | path2 run + Docker stack | `results/paper_results_2026_05_30/07_validation/tab_vroom_*.csv` |

After the four stages, the submitted-version figures and assembly:

```powershell
# Render the submitted EWGT figures
python -m scripts.figures.fig_combined_heatmap
python -m scripts.figures.fig_structural_grid
# ... (or use the Makefile target: make figures)

# Assemble paper outputs into a single dated folder
python scripts/paper/paper_final_assembly.py
```

The Typer CLI command `batch-delivery paper` orchestrates that sequence.

## Figures (`figures/`) — submitted version

One script per figure of the **submitted** paper. The revision's figures are
built by `revision/70_figs_tables_v2.py` and `revision/75_`–`78_`, and reach
`paper/EWGT_2026_rev1/` only through `revision/71_sync_paper_figs.py`.

Naming convention: `fig_<NAME>.py`. Each script reads
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
