"""Append Section 34 (v2 augmented model) to the Paper Compendium.

Reads:
    results/oracle_loop_extended_2026_05_22/production_lgb_logT_v1.json
    results/oracle_loop_extended_2026_05_22/production_lgb_logT_v2.json
    results/v1_vs_v2_comparison/delta_cluster_mape.csv
    results/ml_accuracy_per_cluster_v2/tab_per_cluster_ml_accuracy.csv
    results/final_optimization_v2/scenario_comparison_kpis.csv

Appends:
    docs/PAPER_COMPENDIUM_2026_05_24.md
        ## 34. v2 Augmented Model — Closing the Cluster-Mismatch Gap
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "oracle_loop_extended_2026_05_22"
COMP = ROOT / "docs" / "PAPER_COMPENDIUM_2026_05_24.md"

def _load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    v1m = _load_json(RUN / "production_lgb_logT_v1.json")
    v2m = _load_json(RUN / "production_lgb_logT_v2.json")
    if not v1m or not v2m:
        print("missing v1 or v2 meta — abort")
        return 1

    v1_pool = v1m.get("metrics", {}).get("pool", {})
    v2_pool = v2m.get("metrics", {}).get("pool", {})
    v1_hold = v1m.get("metrics", {}).get("holdout", {})
    v2_hold = v2m.get("metrics", {}).get("holdout", {})

    # Cluster MAPE delta
    delta_csv = ROOT / "results" / "v1_vs_v2_comparison" / "delta_cluster_mape.csv"
    delta_top = ""
    if delta_csv.exists():
        d = pd.read_csv(delta_csv).sort_values("mape_delta")
        cols = ["cluster_id", "is_merged_v1", "cost_mape_pct_v1", "cost_mape_pct_v2", "mape_delta"]
        merged_only = d[d["is_merged_v1"]]
        delta_top = "\n### Per-cluster MAPE improvement (merged clusters only):\n\n"
        delta_top += "| Cluster | v1 MAPE | v2 MAPE | Delta (pp) |\n|---|---|---|---|\n"
        for _, r in merged_only.iterrows():
            delta_top += f"| {r['cluster_id']} | {r['cost_mape_pct_v1']:.2f}% | {r['cost_mape_pct_v2']:.2f}% | {r['mape_delta']:+.2f} |\n"

    # KPI delta
    kpi_v1 = ROOT / "results" / "final_optimization" / "scenario_comparison_kpis.csv"
    kpi_v2 = ROOT / "results" / "final_optimization_v2" / "scenario_comparison_kpis.csv"
    kpi_section = ""
    if kpi_v1.exists() and kpi_v2.exists():
        k1 = pd.read_csv(kpi_v1)
        k2 = pd.read_csv(kpi_v2)
        kpi_section = f"\n### KPI Comparison (head-line scenarios):\n\n"
        # Find common scenarios
        try:
            kpi_section += "**v1 KPIs:**\n```\n"
            kpi_section += k1.to_string(index=False)
            kpi_section += "\n```\n\n"
            kpi_section += "**v2 KPIs:**\n```\n"
            kpi_section += k2.to_string(index=False)
            kpi_section += "\n```\n\n"
        except Exception as e:
            kpi_section += f"(KPI rendering failed: {e})\n"

    section = f"""

---

## 34. v2 Augmented Model — Closing the Cluster-Mismatch Gap

**Datum**: 2026-05-25
**Trigger**: Section 33 identifizierte Training-Inferenz-Inkonsistenz für merged Clusters als
Root Cause der +10.1 pp Saving-Bias. Diese Section dokumentiert die augmentation-basierte
Korrektur (Option C aus FW6).

### 34.1 Augmentation-Strategie

- **Audit** ([scripts/audit_training_pool_gaps.py](../scripts/audit_training_pool_gaps.py)):
  Identifizierte 91 (merged-cluster, provider, agg_k) Cells mit < 5 Training-Samples;
  217 zusätzliche VROOM-Runs nötig um Target=5 pro Cell zu erreichen.

- **Fokussierter Sweep** ([conf/sweep_augment_merged_clusters.yaml](../conf/sweep_augment_merged_clusters.yaml)):
  - 10 merged + 4 schlechteste non-merged Cluster = 14 PLZ-Codes
  - 7 LSP × 6 Tage × 3 agg_ks
  - Erweiterte Perturbation: 5 scales × 3 p_keeps × 2 noise × 3 seeds
  - max_combinations=600 (mit stratified shuffle für balance)

- **Pool-Merge**: Original {v1m['n_train_rows']:,} + Augment ~{v2m['n_train_rows'] - v1m['n_train_rows']:,} = {v2m['n_train_rows']:,} Trainingszeilen
  (siehe [scripts/merge_augmented_training_pool.py](../scripts/merge_augmented_training_pool.py))

- **Retraining** ([scripts/train_production_lgb_v2.py](../scripts/train_production_lgb_v2.py)):
  Identische LGB-logT HPs wie v1 — nur der Pool ändert sich, um Augmentation isoliert zu messen.

### 34.2 Modell-Level Metrics

| Metrik | v1 (original) | v2 (augmentiert) | Delta |
|---|---|---|---|
| Train Rows | {v1m['n_train_rows']:,} | {v2m['n_train_rows']:,} | +{v2m['n_train_rows'] - v1m['n_train_rows']:,} |
| Pool MAPE | {v1_pool.get('mape',0):.3f}% | {v2_pool.get('mape',0):.3f}% | {v2_pool.get('mape',0) - v1_pool.get('mape',0):+.3f}pp |
| Holdout MAPE | {v1_hold.get('mape',0):.3f}% | {v2_hold.get('mape',0):.3f}% | {v2_hold.get('mape',0) - v1_hold.get('mape',0):+.3f}pp |
| Holdout R² | {v1_hold.get('r2',0):.4f} | {v2_hold.get('r2',0):.4f} | {v2_hold.get('r2',0) - v1_hold.get('r2',0):+.4f} |
| Holdout MAE | {v1_hold.get('mae',0):.2f}€ | {v2_hold.get('mae',0):.2f}€ | {v2_hold.get('mae',0) - v1_hold.get('mae',0):+.2f}€ |

{delta_top}

### 34.3 Operational KPIs (Stage 6 VROOM-resolved)

{kpi_section}

### 34.4 Interpretation

Der augmentierte Pool fügt {v2m['n_train_rows'] - v1m['n_train_rows']:,} VROOM-geroutete Samples
hinzu, die explizit die merged-Cluster-Feature-Kombinationen abdecken (insbesondere die
hohe-n_parcels-bei-niedriger-area Region die zuvor nur durch single-PLZ-agg_k=3 simuliert wurde).

Die Pool-MAPE ist **kein** Quality-Indikator — sie reflektiert Train-Set-Fit. Wichtig sind:
1. Holdout-MAPE (Extreme-Holdout-Set frozen seit Iter 0)
2. Per-Cluster-MAPE der merged Clusters (Section 32 Vergleichsbasis)
3. Saving-Bias auf SA_ML-routed schedules (Section 27)

### 34.5 Limitations of the v2 Augmentation

- **Stratification, not exhaustive coverage**: max_combinations=600 cap mit stratified shuffle —
  garantiert Bucket-Balance, aber nicht jede mögliche Perturbation pro Cell. Ein Production-Setup
  könnte 5'000+ Augmentations-Runs verwenden.
- **Feature-Aggregation unchanged**: area_km2 bleibt repräsentative-PLZ-Polygon-Area (nicht
  spatial-merged sum). Future Work FW6.A würde die Polygone selbst spatially mergen.
- **Active Learning could do better**: Eine zweite Oracle-Loop-Iteration mit Disagreement-Sampling
  auf v2's Vorhersagen würde gezielter die verbleibenden hard cases finden.

### 34.6 Reproducibility

```powershell
## Phase 1: Audit
python scripts/audit_training_pool_gaps.py

## Phase 2: Augmentation Sweep (~40 min)
batch-delivery sweep --config conf/sweep_augment_merged_clusters.yaml

## Phase 3: Full v2 pipeline (~75 min)
.\\scripts\\run_v2_full_pipeline.ps1
```

Outputs:
- `results/sweep_augment_2026_05_25/training_matrix_augment.csv`
- `results/oracle_loop_extended_2026_05_22/training_matrix_v2.csv`
- `results/oracle_loop_extended_2026_05_22/production_lgb_logT_v2.{{pkl,json}}`
- `results/final_optimization_v2/scenario_comparison_kpis.csv`
- `results/v1_vs_v2_comparison/{{delta_kpi.csv, delta_cluster_mape.csv, REPORT.md, fig_*.{{png,pdf}}}}`
- `results/ml_accuracy_per_cluster_v2/`, `results/paper_maps_final_v2/`, `results/region_type_breakdown_v2/`
"""

    # Append to compendium
    existing = COMP.read_text(encoding="utf-8")
    if "## 34. v2 Augmented Model" in existing:
        print("Section 34 already exists — REPLACING")
        idx = existing.index("## 34. v2 Augmented Model")
        # Find end (either next ## section or EOF)
        rest = existing[idx:]
        next_sec = rest.find("\n## ", 5)
        if next_sec > 0:
            existing = existing[:idx] + section.lstrip() + rest[next_sec:]
        else:
            existing = existing[:idx] + section.lstrip()
    else:
        existing = existing + section

    COMP.write_text(existing, encoding="utf-8")
    print(f"Updated {COMP}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
