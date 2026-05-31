## Full v2 pipeline orchestrator (Windows PowerShell)
## Run AFTER the augmentation sweep has completed.
## Sequence: merge -> train -> optimize -> eval -> plots -> compendium

$ErrorActionPreference = "Stop"
$start = Get-Date
Write-Host "[v2-pipeline] start at $start"

## --- 1. Merge augmented training pool ---
Write-Host "`n[1/8] merging training pools..."
python scripts/merge_augmented_training_pool.py
if (-not (Test-Path "results/oracle_loop_extended_2026_05_22/training_matrix_v2.csv")) {
    throw "merged training_matrix_v2.csv missing"
}

## --- 2. Retrain LGB-logT v2 ---
Write-Host "`n[2/8] retraining LGB-logT v2..."
python scripts/train_production_lgb_v2.py
if (-not (Test-Path "results/oracle_loop_extended_2026_05_22/production_lgb_logT_v2.pkl")) {
    throw "production_lgb_logT_v2.pkl missing"
}

## --- 3. Re-run schedule optimization (Stage 5) + VROOM resolve (Stage 6) ---
Write-Host "`n[3/8] running final optimization v2 (this includes Stage 6 VROOM)..."
python scripts/run_final_optimization_v2.py

## --- 4. ML accuracy per cluster v2 ---
Write-Host "`n[4/8] ML accuracy per cluster v2..."
python scripts/ml_accuracy_per_cluster_and_raumtyp_v2.py

## --- 5. Region type breakdown v2 ---
Write-Host "`n[5/8] region type breakdown v2..."
python scripts/region_type_breakdown_v2.py

## --- 6. Paper maps v2 ---
Write-Host "`n[6/8] paper maps v2..."
python scripts/paper_maps_with_merge_forwarding_v2.py

## --- 7. v1 vs v2 comparison ---
Write-Host "`n[7/8] v1 vs v2 comparison report..."
python scripts/compare_v1_vs_v2_all_metrics.py

## --- 8. Update Compendium ---
Write-Host "`n[8/8] update compendium with Section 34..."
python scripts/update_compendium_v2.py

$end = Get-Date
$elapsed = $end - $start
Write-Host "`n[v2-pipeline] DONE in $($elapsed.TotalMinutes.ToString('F1')) min"
