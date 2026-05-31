## Full v3 pipeline orchestrator — runs AFTER the v3_mergefix sweep completes.
## Sequence: train v3 model -> run full optimization -> generate eval + schedule analyses

$ErrorActionPreference = "Stop"
$start = Get-Date
Write-Host "[v3-pipeline] start at $start"

## --- 1. Train v3 model on merge-fix sweep data ---
Write-Host "`n[1/7] Training production_lgb_logT_v3..."
python scripts/train_production_lgb_v3.py
if (-not (Test-Path "results/sweep_v3_mergefix/production_lgb_logT_v3.pkl")) {
    throw "v3 model missing"
}

## --- 2. Archive ALL checkpoints so the fix propagates everywhere ---
Write-Host "`n[2/7] Archiving v2 checkpoints..."
$ck = "results/checkpoints"
$arch = "$ck/archive/pre_v3_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Force -Path $arch | Out-Null
Get-ChildItem -Path $ck -Filter "0?_*.pkl" -ErrorAction SilentlyContinue | ForEach-Object {
    Move-Item -Path $_.FullName -Destination $arch -Force
}

## --- 3. Run final optimization v3 (full pipeline with all merge fixes) ---
Write-Host "`n[3/7] Running final optimization v3..."
python scripts/run_final_optimization_v3.py

## --- 4. Build saving table v3 ---
Write-Host "`n[4/7] Build saving table v3..."
# Re-use build_saving_table_v2 but pointed at v3 outputs
$content = Get-Content scripts/build_saving_table_v2.py -Raw
$content -replace 'final_optimization_v2', 'final_optimization_v3_mergefix' | Set-Content scripts/build_saving_table_v3.py
python scripts/build_saving_table_v3.py

## --- 5. ML accuracy per cluster v3 ---
Write-Host "`n[5/7] ML accuracy per cluster v3..."
$content = Get-Content scripts/ml_accuracy_per_cluster_and_raumtyp_v2.py -Raw
$content -replace 'ml_accuracy_per_cluster_v2', 'ml_accuracy_per_cluster_v3' `
         -replace 'final_optimization_v2', 'final_optimization_v3_mergefix' | Set-Content scripts/ml_accuracy_per_cluster_and_raumtyp_v3.py
python scripts/ml_accuracy_per_cluster_and_raumtyp_v3.py

## --- 6. Schedule analysis on v3 ---
Write-Host "`n[6/7] Schedule analysis on v3..."
python scripts/schedule_pattern_analysis.py
python scripts/schedule_paper_figures.py

## --- 7. v2 vs v3 comparison ---
Write-Host "`n[7/7] v2 vs v3 comparison..."
python -c @"
import pandas as pd
v2_kpi = pd.read_csv('results/final_optimization_v2/scenario_comparison_kpis.csv')
v3_kpi = pd.read_csv('results/final_optimization_v3_mergefix/scenario_comparison_kpis.csv')
print('=== v2 vs v3 KPIs ===')
print('v2:'); print(v2_kpi)
print('v3:'); print(v3_kpi)
"@

$end = Get-Date
$elapsed = $end - $start
Write-Host "`n[v3-pipeline] DONE in $($elapsed.TotalMinutes.ToString('F1')) min"
