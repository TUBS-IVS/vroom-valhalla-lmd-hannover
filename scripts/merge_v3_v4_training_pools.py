"""Merge v3-mergefix + v4-density-buffer sweep outputs into one training pool.

Inputs:
    results/sweep_v3_mergefix/training_matrix.csv    (cluster-aware base pool)
    results/sweep_v4_density_buffer/training_matrix.csv  (low/high-density edge samples)

Output:
    results/sweep_v4_density_buffer/training_matrix_merged.csv

Dedupes on (provider, plz, base_day, agg_k, scale, p_keep, noise_sigma, seed).
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "results" / "sweep_v3_mergefix" / "training_matrix.csv"
V4 = ROOT / "results" / "sweep_v4_density_buffer" / "training_matrix.csv"
OUT = ROOT / "results" / "sweep_v4_density_buffer" / "training_matrix_merged.csv"

KEY = ["provider", "plz", "base_day", "agg_k", "scale", "p_keep",
        "noise_sigma", "seed"]


def main():
    if not V3.exists() or not V4.exists():
        print(f"Missing: V3={V3.exists()} V4={V4.exists()}")
        return 1
    v3 = pd.read_csv(V3)
    v4 = pd.read_csv(V4)
    print(f"v3 pool: {len(v3):,} rows")
    print(f"v4 buffer: {len(v4):,} rows")

    common = list(set(v3.columns) & set(v4.columns))
    merged = pd.concat([v3[common], v4[common]], ignore_index=True)
    n_before = len(merged)
    merged = merged.drop_duplicates(subset=KEY, keep="last")
    print(f"merged: {len(merged):,} rows ({n_before - len(merged)} dup dropped)")

    merged.to_csv(OUT, index=False)
    print(f"Saved: {OUT.relative_to(ROOT)}")

    # Coverage summary
    print(f"\nFeature ranges after merge:")
    for col in ["n_parcels", "area_km2", "parcels_per_km2"]:
        p = merged[col]
        print(f"  {col:18s}: p1={p.quantile(0.01):>6.1f} median={p.median():>6.1f} p99={p.quantile(0.99):>6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
