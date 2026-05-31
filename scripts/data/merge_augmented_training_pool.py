"""Merge the augmentation sweep into the production training pool.

Inputs:
    results/oracle_loop_extended_2026_05_22/training_matrix.csv   (original 11,523 rows)
    results/sweep_augment_2026_05_25/training_matrix_augment.csv  (~600 new rows)

Output:
    results/oracle_loop_extended_2026_05_22/training_matrix_v2.csv

We keep the original as-is (don't overwrite) so we can compare v1 vs v2 models.
Deduplicates on (provider, plz, base_day, agg_k, scale, p_keep, noise_sigma, seed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "results" / "oracle_loop_extended_2026_05_22" / "training_matrix.csv"
AUG = ROOT / "results" / "sweep_augment_2026_05_25" / "training_matrix_augment.csv"
OUT = ROOT / "results" / "oracle_loop_extended_2026_05_22" / "training_matrix_v2.csv"

KEY = ["provider", "plz", "base_day", "agg_k", "scale", "p_keep", "noise_sigma", "seed"]


def main():
    if not ORIG.exists():
        print(f"ERROR: original missing: {ORIG}")
        return 1
    if not AUG.exists():
        print(f"ERROR: augmentation missing: {AUG}")
        return 1

    orig = pd.read_csv(ORIG)
    aug = pd.read_csv(AUG)
    print(f"original: {len(orig):,} rows")
    print(f"augment : {len(aug):,} rows")

    # Align schemas (use union of columns)
    common = list(set(orig.columns) & set(aug.columns))
    missing_in_aug = set(orig.columns) - set(aug.columns)
    missing_in_orig = set(aug.columns) - set(orig.columns)
    print(f"common cols: {len(common)}, aug-missing-in-orig: {missing_in_orig}, orig-missing-in-aug: {missing_in_aug}")

    # Subset both to common cols if needed
    if missing_in_aug or missing_in_orig:
        orig = orig[common].copy()
        aug = aug[common].copy()

    # Concatenate and dedupe on key
    merged = pd.concat([orig, aug], ignore_index=True)
    n_before = len(merged)
    merged = merged.drop_duplicates(subset=KEY, keep="last")
    n_after = len(merged)
    n_dup = n_before - n_after
    n_new = len(merged) - len(orig)
    print(f"merged  : {n_after:,} rows ({n_dup:,} duplicates removed, {n_new:,} net new)")

    # Per-cluster coverage after merge
    merged_clusters = [30159, 30163, 30167, 30449, 30519, 30559, 30625, 30827, 30853, 30855]
    print("\nPer-merged-cluster row counts (v1 -> v2):")
    for cid in merged_clusters:
        n1 = (orig["plz"] == cid).sum()
        n2 = (merged["plz"] == cid).sum()
        print(f"  cluster {cid}: {n1:>3} -> {n2:>3} (+{n2 - n1})")

    merged.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
