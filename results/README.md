# `results/` — Outputs of the EWGT 2026 Paper Pipeline

This directory holds every output that backs a number, figure, or table in
the paper *Bienzeisler et al., "Machine-Learning Surrogate Optimization for
Time-Based Consolidation in Last-Mile Parcel Delivery", EWGT 2026*.

Layout follows a discoverability-first convention: anything starting with
`paper_*` is directly cited in the paper, everything else supports it.

## Directory map

```
results/
├── paper_ewgt_2026/         ← FROZEN: figures + tables as submitted
├── paper_outputs_2026_05_30/  ← Full 13-chapter paper output assembly
├── paper_results_2026_05_30/  ← Post-VROOM-validation paper cut (10 chapters)
├── runs/
│   └── path2_2026_05_29/    ← Canonical optimization run (88-cell grid)
├── supplementary/           ← Supporting analyses (model battery, sensitivity, region maps)
└── _archive/                ← Superseded outputs (gitignored, kept locally)
```

## What backs what

| Paper claim | Folder | Key files |
|---|---|---|
| Headline cost savings (22.8% at cost-optimal, 13.5% at P=0.5) | `runs/path2_2026_05_29/` | `tab_balancing_summary.csv`, `tab_chosen_schedules.csv` |
| Operating sweet spot (P ∈ [0.25, 0.75]) | `paper_outputs_2026_05_30/05_optimization/` | `fig_PF1_pareto.png`, `tab_optimization_full_grid.csv` |
| Mo–Sa fleet CV reduction (up to 60% in efficient range) | `runs/path2_2026_05_29/` + `paper_outputs_2026_05_30/06_balancing/` | `tab_fleet_per_hub.csv`, `_system_spread_per_cell.csv` |
| VROOM validation (surrogate conservative by 1.3–2.1 pp) | `paper_results_2026_05_30/07_validation/` | `tab_vroom_balanced.csv`, `tab_vroom_path2.csv` |
| Daganzo-LGB-Hybrid 2.95% MAPE | `supplementary/model_battery_v3/` | `tab_model_comparison.csv` |
| Rural pays most (raumtyp breakdown) | `supplementary/region_type_breakdown_v2/` | `tab_saving_by_raumtyp_3.csv` |
| Per-cluster ML accuracy | `supplementary/ml_accuracy_per_cluster_v2/` | `tab_mape_per_cluster.csv` |
| Region maps (M01–M07) | `supplementary/paper_maps_final_v2/` | `fig_M0{1..7}_*.png` |
| Penalty sweep (1-D precursor) | `supplementary/penalty_sweep/` | `sched_cost_cache.npz` (gitignored), `tab_pareto.csv` |
| 2-D batch×penalty sensitivity | `supplementary/sensitivity_2d/` | `fig_sensitivity_2d_heatmap.png` |
| Break-even analysis | `supplementary/sensitivity_break_even/` | `tab_break_even.csv`, `fig_break_even.png` |
| Schedule pattern analysis | `supplementary/schedule_paper/` | `tab_chosen_schedules.csv`, `fig_pattern_distribution.png` |
| Training pool (for retraining) | `supplementary/sweep_v3_mergefix/` | `tab_features.csv` (parquet gitignored) |

## Submitted figures & tables (frozen)

Everything in [`paper_ewgt_2026/`](paper_ewgt_2026/) is **byte-identical** to the
files in the submitted PDF. Do not regenerate these — they are the immutable
record of what was reviewed.

For active development of the camera-ready version, use the figure scripts in
[`scripts/figures/`](../scripts/figures/) and the outputs land in the dated
`paper_outputs_*/` folder (a new dated copy is created per regen pass, never
overwriting).

## How to regenerate everything

The whole tree under `results/` is reproducible from `data/` via the pipeline.
See [`docs/REPRODUCING_PAPER.md`](../docs/REPRODUCING_PAPER.md) for the full
recipe, or use the orchestrated entry point:

```powershell
batch-delivery paper --config conf/default.yaml
```

This executes the six pipeline stages in order and re-creates everything in
`results/runs/`, `results/paper_outputs_*/`, and (with VROOM) `results/paper_results_*/07_validation/`.

## Archive policy

`_archive/` is local-only (gitignored). It contains superseded versions of
optimization runs, older model batteries, and deprecated willingness analyses.
See [`CANONICAL.md`](CANONICAL.md) for the complete list and supersession
mapping. Keep it locally for paper-review traceability; do not push.
