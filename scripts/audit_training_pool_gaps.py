"""Audit training pool: identify uncovered (cluster, provider, day, agg_k) cells.

For each merged cluster (is_merged=True) and even non-merged ones, check whether
the training_matrix already contains samples that resemble inference-time feature
combinations. Output a gap-report CSV listing (provider, plz_cluster, base_day,
agg_k, perturbation) combos to augment.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ORACLE_DIR = ROOT / "results" / "oracle_loop_extended_2026_05_22"
OUT_DIR = ROOT / "results" / "audits" / "training_gaps_2026_05_25"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROVIDERS = ["Amazon", "DHL", "DPD", "FedEx", "GLS", "Hermes", "UPS"]
N_DAYS = 6
AGG_KS = [1, 2, 3]

def main():
    train = pd.read_csv(ORACLE_DIR / "training_matrix.csv")
    print(f"[train] {len(train):,} rows, {train['plz'].nunique()} PLZ, providers={train['provider'].unique().tolist()}")
    
    # Per (provider, plz, base_day, agg_k) count samples
    cell_counts = train.groupby(["provider", "plz", "base_day", "agg_k"]).size().rename("n_samples").reset_index()
    print(f"[cells] {len(cell_counts):,} distinct cells")
    print("Cell-coverage distribution:")
    print(cell_counts["n_samples"].describe())
    
    # Identify merged clusters from the ml_accuracy table
    ml_acc = pd.read_csv(ROOT / "results" / "ml_accuracy_per_cluster" / "tab_per_cluster_ml_accuracy.csv")
    merged_clusters = ml_acc.loc[ml_acc["is_merged"], "cluster_id"].astype(int).tolist()
    all_clusters = ml_acc["cluster_id"].astype(int).tolist()
    print(f"[clusters] {len(merged_clusters)} merged, {len(all_clusters)} total")
    
    # For each cluster, count samples per (provider, agg_k) across days
    rows = []
    for cid in all_clusters:
        is_merged = cid in merged_clusters
        for prov in PROVIDERS:
            for k in AGG_KS:
                cell_n = cell_counts[
                    (cell_counts["plz"] == cid) &
                    (cell_counts["provider"] == prov) &
                    (cell_counts["agg_k"] == k)
                ]["n_samples"].sum()
                rows.append({
                    "cluster_id": cid,
                    "provider": prov,
                    "agg_k": k,
                    "is_merged": is_merged,
                    "n_train_samples": int(cell_n),
                })
    cov = pd.DataFrame(rows)
    cov.to_csv(OUT_DIR / "coverage_per_cluster_provider_aggk.csv", index=False)
    
    # Summary: undercovered cells
    print("\n--- Coverage per merged-status ---")
    print(cov.groupby(["is_merged", "agg_k"])["n_train_samples"].describe())
    
    # Worst-coverage merged-cluster cells (< 3 samples)
    under = cov[(cov["is_merged"]) & (cov["n_train_samples"] < 3)]
    print(f"\n[under] {len(under):,} (merged, provider, agg_k) cells with <3 samples")
    print(under.head(20).to_string())
    
    # Identify total gap to fill: target 5 samples per (merged_cluster, provider, agg_k)
    TARGET_PER_CELL = 5
    gap_plan = []
    for _, r in cov.iterrows():
        if not r["is_merged"]:
            continue
        if r["n_train_samples"] < TARGET_PER_CELL:
            n_missing = TARGET_PER_CELL - int(r["n_train_samples"])
            gap_plan.append({
                "cluster_id": int(r["cluster_id"]),
                "provider": r["provider"],
                "agg_k": int(r["agg_k"]),
                "n_to_add": n_missing,
            })
    gap_df = pd.DataFrame(gap_plan)
    gap_df.to_csv(OUT_DIR / "augmentation_plan.csv", index=False)
    
    total_runs = int(gap_df["n_to_add"].sum())
    print(f"\n[plan] {len(gap_df):,} cells under target={TARGET_PER_CELL}; total new VROOM runs needed: {total_runs}")
    
    # Cluster total summary
    by_cluster = gap_df.groupby("cluster_id")["n_to_add"].sum().sort_values(ascending=False)
    print("\nTop-undercovered merged clusters (samples to add):")
    print(by_cluster.head(15).to_string())
    
    return total_runs

if __name__ == "__main__":
    sys.exit(0 if main() >= 0 else 1)
