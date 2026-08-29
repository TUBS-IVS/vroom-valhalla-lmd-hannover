# `results/` Canonical Classification

This file classifies every sub-folder of `results/` for the EWGT 2026 paper
repository. Used as the source-of-truth for the Phase 2 reorganization:
canonical folders move to `results/paper/` and `results/runs/`, everything
else moves to `results/_archive/` (gitignored) or gets deleted outright.

Last reviewed: 2026-05-31 (submission). Revision addendum: 2026-08-28.

## 2026-08-28 addendum — the revision's canonical folder

The classification below is the **submission's** and is unchanged. The EWGT
2026 revision adds one canonical folder, which takes precedence for every
revision number:

| Folder | Status | Role |
|---|---|---|
| `revision_2026_08_final/` | **canonical (revision)** | The one folder for the revision: figures, tables, VROOM validation, controller analyses, dashboard sources, paper-side provenance and deck pointers, all from grid v6. Built by `scripts/revision/79_build_final_pack.py`; its `README.md` lists every file with its producing script, grid and md5. Gitignored like the rest of `results/`, so the scripts are the tracked artefact. |
| `revision_2026_08_v6/` | working | The live grid the pack is copied from (`61_` … `67_`, `70_` … `79_`). |
| `revision_2026_08_v5/` | **superseded** | v6 is v5 plus the bundle head. Referenced only by the pack's v6-vs-v5 delta table. |
| `revision_2026_07/` | **superseded** | The Stage-3 grid of the first revision round. Kept for history; its numbers must never be quoted next to v6 numbers without saying which grid they come from. |

Unchanged by all of this: `paper/EWGT_2026/` stays frozen at the submission,
and the submission-era folders below keep their status.

| Status | Meaning |
|---|---|
| **canonical** | Backs a paper figure, table, or numeric claim. Tracked in git. |
| **supplementary** | Useful for review (model battery, sensitivity tests). Tracked in git (small files only — `*.pkl`/`*.parquet` excluded by .gitignore). |
| **archive** | Superseded by a newer version. Kept locally in `results/_archive/` but **gitignored**. |
| **delete** | Smoke tests, one-off diagnostics, runtime caches. Removed entirely. |

---

## Canonical (tracked, backs the paper)

| Folder | Size | Status | Role |
|---|---|---|---|
| `overnight_2026_05_29_path2/` | 7 MB | canonical | Path-2 optimization grid (8 P × 11 θ × 7 LSPs = 616 cells). Backs every cost/saving/fleet number in the Results section. |
| `paper_final_2026_05_30/` | 109 MB | canonical | Final paper output assembly (figures + tables, all 13 chapters). |
| `paper_results_final/07_validation/` | 18 MB | canonical | VROOM out-of-sample validation for P ∈ {0, 0.25, 0.5}. Backs the "surrogate is conservative by 1.3–2.1 pp" claim. |
| `EWGT_Results/` | 7.5 MB | canonical | Frozen EWGT submission figures + tables. The actual files in the paper. |

**Sub-total canonical: ~140 MB tracked.**

---

## Supplementary (tracked, small files only)

| Folder | Size | Status | Role |
|---|---|---|---|
| `model_battery_v3/` | 64 KB | supplementary | Final model comparison (LGB-logT, MLP ensemble, Daganzo-Hybrid). MAPE table cited in paper. |
| `ml_accuracy_per_cluster_v2/` | 1.1 MB | supplementary | Per-cluster MAPE breakdown. Robustness evidence. |
| `region_type_breakdown_v2/` | 1.4 MB | supplementary | Raumtyp (urban/suburban/rural) cost-saving breakdown. Backs the rural-pays-most claim. |
| `paper_maps_final_v2/` | 2.7 MB | supplementary | Region-type maps (Figs M01-M07). |
| `production_quality_on_routed/` | 794 KB | supplementary | Production-model quality on routed scenarios. |
| `ml_vs_vroom_optimized/` | 2.0 MB | supplementary | ML vs VROOM scatter on optimized cells. |
| `penalty_sweep/` | 688 KB | supplementary | 1-D P sweep (precursor to 2-D grid). |
| `sensitivity_2d/` | 512 KB | supplementary | Earlier 2-D batch × penalty heatmap. |
| `sensitivity_break_even/` | 2.3 MB | supplementary | Break-even penalty analysis. |
| `schedule_paper/` | 2.4 MB | supplementary | Schedule pattern analysis. |
| `sweep_v3_mergefix/` | 20 MB | supplementary | Training sample pool that fed the Daganzo-LGB-Hybrid. Kept for retrainability. |
| `final_optimization_v3_mergefix/` | 108 MB | supplementary | Final optimization run with bundled-cost accounting. Tier-2 validation. |

**Sub-total supplementary tracked (after gitignore filters): ~10 MB.**

---

## Archive (kept locally, gitignored)

These are superseded versions kept for paper-review nachvollziehbarkeit but not
shipped to GitHub.

| Folder | Size | Superseded by |
|---|---|---|
| `paper_final_2026_05_28/` | 67 MB | `paper_final_2026_05_30/` |
| `overnight_2026_05_27/` | 28 MB | `overnight_2026_05_29_path2/` |
| `overnight_2026_05_27_balanced/` | 11 MB | `overnight_2026_05_29_path2/` |
| `final_optimization/` | 149 MB | `final_optimization_v3_mergefix/` |
| `final_optimization_v2/` | 104 MB | `final_optimization_v3_mergefix/` |
| `final_optimization_clean/` | 28 MB | `final_optimization_v3_mergefix/` |
| `oracle_loop_overnight_2026_05_21/` | 90 MB | `overnight_2026_05_29_path2/` |
| `oracle_loop_extended_2026_05_22/` | 244 MB | `overnight_2026_05_29_path2/` |
| `paper_figures/` | 5.5 MB | `EWGT_Results/` + `paper_final_2026_05_30/` |
| `paper_maps_final/` (v1) | 2.8 MB | `paper_maps_final_v2/` |
| `region_type_breakdown/` (v1) | 1.4 MB | `region_type_breakdown_v2/` |
| `ml_accuracy_per_cluster/` (v1) | 1.1 MB | `ml_accuracy_per_cluster_v2/` |
| `willingness_p050/` | 1.2 MB | linear-blended, deprecated; covered by `sensitivity_2d/` |
| `willingness_3d/` | 2.2 MB | linear-blended, deprecated |
| `willingness_to_wait_v2preview/` | 1.3 MB | superseded by Path-2 grid |
| `willingness_to_wait_2d_v2/` | 1.6 MB | superseded by Path-2 grid |
| `willingness_hub_bundled_v2/` | 1.9 MB | superseded by Path-2 grid |
| `willingness_hub_bundled_daganzo/` | 1.4 MB | superseded by Path-2 grid |
| `willingness_penalty_v2/` | 720 KB | superseded by Path-2 grid |
| `service_p050_final/` | 1.7 MB | subsumed by `paper_final_2026_05_30/` |
| `schedule_analysis/` | 2.4 MB | subsumed by `paper_final_2026_05_30/` |
| `model_battery_v2test/` | 432 KB | `model_battery_v3/` |
| `model_battery_inpool/` | 4 KB | `model_battery_v3/` |
| `v9_ensemble_test/` | 6.3 MB | superseded by Daganzo-LGB-Hybrid |
| `v6_v7_v8_test/` | 181 KB | superseded by Daganzo-LGB-Hybrid |
| `v5_honest_test/` | 3.6 MB | superseded by Daganzo-LGB-Hybrid |
| `v2/` | 1.0 MB | superseded by sweep_v3_mergefix |
| `v1_vs_v2_comparison/` | 132 KB | superseded |
| `sweep_v4_density_buffer/` | 349 KB | superseded by sweep_v3_mergefix |
| `sweep_augment_2026_05_25/` | 613 KB | merged into sweep_v3_mergefix |

**Sub-total archive: ~760 MB locally, 0 MB in git.**

---

## Delete (smoke tests, runtime caches, one-off diagnostics)

These have no review value and are pure runtime artefacts.

| Folder | Size | Reason |
|---|---|---|
| `cache/` | 2.9 GB | VROOM solution cache, regenerable |
| `checkpoints/` | 1.6 GB | Pipeline checkpoints, regenerable |
| `baseline/` | 181 MB | Baseline VROOM outputs, regenerable |
| `oracle_loop_smoke/` | 27 MB | Smoke test |
| `oracle_loop_smoke_v2/` | 21 MB | Smoke test |
| `surrogate_smoke/` | 614 KB | Smoke test |
| `sweep_oracle_smoke/` | 982 KB | Smoke test |
| `sweep_smoke/`, `sweep_smoke_par/`, `sweep_smoke_multi/`, `sweep_smoke_baseline/` | ~280 KB | Smoke tests |
| `audits/` | 1.2 MB | Audit logs, no paper relevance |
| `bias_correction_diagnostic/` | 809 KB | One-off diagnostic |
| `distribution_shift_diagnosis/` | 637 KB | One-off diagnostic |
| `diagnose_daganzo_hybrid/` | 232 KB | One-off diagnostic |
| `compare_express_handling/` | 256 KB | One-off comparison |
| `express_aware/` | 800 KB | Earlier variant, no paper use |
| `lgb_quality_improvement/` | 152 KB | One-off |
| `disjoint_features/` | 32 KB | Diagnostic |
| `model_level_debiasing/` | 12 KB | Diagnostic |

**Sub-total delete: ~4.7 GB freed.**

---

## Summary

| Bucket | Folder count | Size | Git status |
|---|---|---|---|
| canonical | 4 | ~140 MB | tracked |
| supplementary | 12 | ~10 MB (after .gitignore filter) | tracked |
| archive | 30 | ~760 MB | gitignored, kept locally |
| delete | 18+ | ~4.7 GB | removed |

**Total `results/` before: 5.7 GB. After Phase 2: ~150 MB tracked + ~760 MB local archive.**

---

## Phase 2 Reorganization Mapping

Phase 2 of the refactor will perform these `git mv` operations:

```
results/
├── paper/                  ← rename of paper_final_2026_05_30/
├── runs/
│   └── path2_2026_05_29/   ← rename of overnight_2026_05_29_path2/
├── validation/
│   └── vroom_p2/           ← extract from paper_results_final/07_validation/
├── ewgt_2026/              ← rename of EWGT_Results/  (or move to paper/ewgt_2026/)
├── supplementary/
│   ├── model_battery/
│   ├── ml_accuracy/
│   ├── region_type/
│   ├── paper_maps/
│   ├── penalty_sweep/
│   ├── sensitivity_2d/
│   ├── sensitivity_break_even/
│   ├── schedule_paper/
│   ├── production_quality/
│   ├── ml_vs_vroom/
│   ├── sweep_v3_mergefix/        ← training pool
│   └── final_optimization_v3/    ← bundled-cost validation
└── _archive/               ← gitignored
    └── (everything classified above as "archive")
```

The `delete` category is handled by `Remove-Item -Recurse -Force` outside git.
