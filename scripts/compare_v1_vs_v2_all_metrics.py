"""Compare v1 vs v2 production-LGB models on ALL paper KPIs.

Reads:
    results/final_optimization/scenario_comparison_kpis.csv         (v1)
    results/final_optimization_v2/scenario_comparison_kpis.csv      (v2)
    results/oracle_loop_extended_2026_05_22/production_lgb_logT_v1.json
    results/oracle_loop_extended_2026_05_22/production_lgb_logT_v2.json
    results/ml_accuracy_per_cluster/tab_per_cluster_ml_accuracy.csv  (v1)
    results/ml_accuracy_per_cluster_v2/tab_per_cluster_ml_accuracy.csv  (v2)

Outputs:
    results/v1_vs_v2_comparison/
        delta_kpi.csv
        delta_cluster_mape.csv
        fig_mape_v1_vs_v2.png/pdf
        fig_saving_v1_vs_v2.png/pdf
        REPORT.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "oracle_loop_extended_2026_05_22"
OUT = ROOT / "results" / "v1_vs_v2_comparison"
OUT.mkdir(parents=True, exist_ok=True)


def load_kpi(version: str) -> pd.DataFrame | None:
    p = ROOT / "results" / (f"final_optimization{'_v2' if version == 'v2' else ''}") / "scenario_comparison_kpis.csv"
    if not p.exists():
        print(f"[{version}] KPI csv missing: {p}")
        return None
    return pd.read_csv(p)


def load_meta(version: str) -> dict | None:
    p = RUN / f"production_lgb_logT_{version}.json"
    if not p.exists():
        print(f"[{version}] meta missing: {p}")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    print("=" * 60)
    print("v1 vs v2 — Side-by-side Comparison")
    print("=" * 60)

    # Model-level metrics
    v1m = load_meta("v1")
    v2m = load_meta("v2")
    if v1m and v2m:
        # v1 has a different schema: training_metrics / holdout_metrics flat
        v1_holdout = v1m.get("metrics", {}).get("holdout", v1m.get("holdout_metrics", {}))
        v2_holdout = v2m.get("metrics", {}).get("holdout", v2m.get("holdout_metrics", {}))
        v1_pool = v1m.get("metrics", {}).get("pool", v1m.get("training_metrics", {}))
        v2_pool = v2m.get("metrics", {}).get("pool", v2m.get("training_metrics", {}))
        # Normalise n_train_rows from either schema
        n_train_v1 = v1m.get("n_train_rows", v1m.get("training_metrics", {}).get("n"))
        n_train_v2 = v2m.get("n_train_rows", v2m.get("training_metrics", {}).get("n"))
        print(f"\n--- Model metrics ---")
        print(f"  Train rows : v1={n_train_v1:,}  v2={n_train_v2:,}  delta={n_train_v2 - n_train_v1:,}")
        print(f"  Pool MAPE  : v1={float(v1_pool.get('mape',0)):.3f}  v2={float(v2_pool.get('mape',0)):.3f}")
        print(f"  Holdout MAPE: v1={float(v1_holdout.get('mape',0)):.3f}  v2={float(v2_holdout.get('mape',0)):.3f}")
        print(f"  Holdout R2 : v1={float(v1_holdout.get('r2',0)):.4f}  v2={float(v2_holdout.get('r2',0)):.4f}")

    # KPI table
    v1_kpi = load_kpi("v1")
    v2_kpi = load_kpi("v2")
    if v1_kpi is not None and v2_kpi is not None:
        print(f"\n--- KPI table shape: v1={v1_kpi.shape}, v2={v2_kpi.shape} ---")
        # Pick the headline columns
        # NOTE: column names depend on the eval pipeline; print what we have
        print(f"v1 cols: {list(v1_kpi.columns)}")
        print(f"v2 cols: {list(v2_kpi.columns)}")

        # KPI table has no join keys (one row per scenario as rows)
        v1_kpi.columns = [f"{c}_v1" for c in v1_kpi.columns]
        v2_kpi.columns = [f"{c}_v2" for c in v2_kpi.columns]
        delta = pd.concat([v1_kpi.reset_index(drop=True), v2_kpi.reset_index(drop=True)], axis=1)
        delta.to_csv(OUT / "delta_kpi.csv", index=False)
        print(f"Wrote {OUT / 'delta_kpi.csv'}")

    # Cluster-level MAPE comparison
    v1_cl = ROOT / "results" / "ml_accuracy_per_cluster" / "tab_per_cluster_ml_accuracy.csv"
    v2_cl = ROOT / "results" / "ml_accuracy_per_cluster_v2" / "tab_per_cluster_ml_accuracy.csv"
    if v1_cl.exists() and v2_cl.exists():
        d1 = pd.read_csv(v1_cl)
        d2 = pd.read_csv(v2_cl)
        merged = d1.merge(d2, on="cluster_id", suffixes=("_v1", "_v2"))
        merged["mape_delta"] = merged["cost_mape_pct_v2"] - merged["cost_mape_pct_v1"]
        merged["bias_delta"] = merged["cost_bias_pct_v2"] - merged["cost_bias_pct_v1"]
        merged_sorted = merged.sort_values("mape_delta")
        merged_sorted.to_csv(OUT / "delta_cluster_mape.csv", index=False)
        print(f"Wrote {OUT / 'delta_cluster_mape.csv'}")

        # Figure: MAPE v1 vs v2 per cluster
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].scatter(merged["cost_mape_pct_v1"], merged["cost_mape_pct_v2"],
                      c=["#cb181d" if m else "#1a9850" for m in merged["is_merged_v1"]],
                      s=80, alpha=0.7, edgecolor="k")
        lim = max(merged[["cost_mape_pct_v1", "cost_mape_pct_v2"]].max().max(), 60)
        ax[0].plot([0, lim], [0, lim], "k--", alpha=0.4)
        ax[0].set_xlabel("MAPE v1 (%)")
        ax[0].set_ylabel("MAPE v2 (%)")
        ax[0].set_title("Per-cluster MAPE v1 vs v2 (red = merged cluster)")
        ax[0].grid(alpha=0.3)

        ax[1].bar(range(len(merged_sorted)), merged_sorted["mape_delta"],
                  color=["#cb181d" if m else "#888" for m in merged_sorted["is_merged_v1"]])
        ax[1].axhline(0, c="k")
        ax[1].set_xlabel("Cluster (sorted by Delta-MAPE)")
        ax[1].set_ylabel("MAPE v2 - MAPE v1 (%-points)")
        ax[1].set_title("Per-cluster MAPE improvement")
        ax[1].grid(alpha=0.3)
        plt.tight_layout()
        for ext in [".png", ".pdf"]:
            plt.savefig(OUT / f"fig_mape_v1_vs_v2{ext}", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Wrote {OUT / 'fig_mape_v1_vs_v2.png'}")

    # REPORT.md
    lines = ["# v1 vs v2 Comparison\n"]
    if v1m and v2m:
        lines.append(f"## Model metrics\n")
        lines.append(f"| Metric | v1 | v2 |\n|---|---|---|")
        lines.append(f"| Train rows | {v1m['n_train_rows']:,} | {v2m['n_train_rows']:,} |")
        lines.append(f"| Pool MAPE  | {v1_pool.get('mape',0):.3f} | {v2_pool.get('mape',0):.3f} |")
        lines.append(f"| Holdout MAPE | {v1_holdout.get('mape',0):.3f} | {v2_holdout.get('mape',0):.3f} |")
        lines.append(f"| Holdout R² | {v1_holdout.get('r2',0):.4f} | {v2_holdout.get('r2',0):.4f} |")
        lines.append("")
    if v1_cl.exists() and v2_cl.exists():
        lines.append("## Per-cluster MAPE delta (top improvements)\n")
        lines.append(merged_sorted.head(10)[["cluster_id", "is_merged_v1", "cost_mape_pct_v1", "cost_mape_pct_v2", "mape_delta"]].to_string(index=False))
        lines.append("")
        lines.append("## Per-cluster MAPE delta (worst regressions)\n")
        lines.append(merged_sorted.tail(10)[["cluster_id", "is_merged_v1", "cost_mape_pct_v1", "cost_mape_pct_v2", "mape_delta"]].to_string(index=False))
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    sys.exit(main() or 0)
