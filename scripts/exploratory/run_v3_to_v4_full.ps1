## Full v3 → v4 orchestration AFTER v3 final_optimization completes
## Runs sequentially:
##   1. v3 basic evals (saving table, ML accuracy, schedule analysis, v2-vs-v3)
##   2. v4 density-buffer sweep (~2h VROOM)
##   3. Merge v3+v4 → training_matrix_merged
##   4. Retrain LGB-logT v4 + aux distance + aux routes + daganzo-hybrid v4
##   5. Model battery on merged pool
##   6. Willingness analyses (1D, 2D, hub-bundled) on best v4 model
##   7. Compendium update

$ErrorActionPreference = "Stop"
$start = Get-Date

Write-Host "[v3->v4 orchestrator] start at $start"

## --- 1. v3 evals ---
Write-Host "`n[1/7] v3 basic evals..."
python scripts/build_saving_table_v2.py  # adapt for v3 path
python scripts/ml_accuracy_per_cluster_and_raumtyp_v2.py  # adapt
python scripts/schedule_pattern_analysis.py
python scripts/compare_v1_vs_v2_all_metrics.py  # extend to v3

## --- 2. v4 sweep ---
Write-Host "`n[2/7] v4 density-buffer sweep (~2h)..."
batch-delivery sweep --config conf/sweep_v4_density_buffer.yaml

## --- 3. Merge pools ---
Write-Host "`n[3/7] Merge v3 + v4 pools..."
python scripts/merge_v3_v4_training_pools.py

## --- 4. Retrain ---
Write-Host "`n[4/7] Retrain models on merged v4 pool..."
$pool = "results/sweep_v4_density_buffer/training_matrix_merged.csv"
python scripts/train_production_lgb_v3.py --pool $pool --version v4
python scripts/train_aux_models_v2.py  # adapt to v4 pool
python scripts/train_daganzo_hybrid.py --pool $pool --version v4

## --- 5. Model battery ---
Write-Host "`n[5/7] Model battery on merged v4 pool..."
python scripts/model_battery_v4.py --pool $pool --out results/model_battery_v4

## --- 6. Willingness on best model ---
Write-Host "`n[6/7] Willingness analyses on best v4 model..."
# Read best model from battery report and run flexible willingness
$best = (Get-Content results/model_battery_v4/REPORT.md | Select-String "Winner:\s*\*\*([\w-]+)\*\*").Matches[0].Groups[1].Value
Write-Host "  Best model: $best"
python scripts/willingness_flexible.py --model-path "results/sweep_v3_mergefix/daganzo_hybrid_v4.pkl"

## --- 7. Compendium ---
Write-Host "`n[7/7] Update Compendium..."
# (compendium update script to be written)

$end = Get-Date
$elapsed = $end - $start
Write-Host "`n[orchestrator] DONE in $($elapsed.TotalHours.ToString('F2')) hours"
