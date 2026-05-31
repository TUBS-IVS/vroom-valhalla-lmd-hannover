"""Join the per-(provider,plz) disjoint topology features into a training pool.

Idempotent: writes to a sibling file `<pool>_topo.csv` so the original is preserved.

Usage:
    python scripts/add_disjoint_to_pool.py \
        --pool results/sweep_v3_mergefix/training_matrix.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOPO = ROOT / "results" / "disjoint_features" / "topo_per_provider_plz.csv"

NEW_COLS = [
    "n_merged_members", "n_components_post_union",
    "hull_overhead_pct", "bbox_overhead_pct",
    "max_member_centroid_km", "mean_member_centroid_km",
    "isoperimetric_q",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not TOPO.exists():
        raise FileNotFoundError(f"Run compute_disjoint_features.py first ({TOPO})")
    topo = pd.read_csv(TOPO, dtype={"plz": str})
    topo["plz"] = topo["plz"].astype(str).str.zfill(5)
    topo = topo[["provider", "plz"] + NEW_COLS]

    pool = Path(args.pool)
    df = pd.read_csv(pool, dtype={"plz": str})
    df["plz"] = df["plz"].astype(str).str.zfill(5)
    print(f"pool rows: {len(df):,}  cols: {len(df.columns)}")

    drop_existing = [c for c in NEW_COLS if c in df.columns]
    if drop_existing:
        print(f"dropping existing: {drop_existing}")
        df = df.drop(columns=drop_existing)

    merged = df.merge(topo, on=["provider", "plz"], how="left")
    missing = merged[NEW_COLS[0]].isna().sum()
    if missing:
        print(f"WARNING: {missing} rows did not match topo features — filling with safe defaults")
        merged["n_merged_members"] = merged["n_merged_members"].fillna(1)
        merged["n_components_post_union"] = merged["n_components_post_union"].fillna(1)
        merged["max_member_centroid_km"] = merged["max_member_centroid_km"].fillna(0)
        merged["mean_member_centroid_km"] = merged["mean_member_centroid_km"].fillna(0)
        merged["hull_overhead_pct"] = merged["hull_overhead_pct"].fillna(merged["hull_overhead_pct"].median())
        merged["bbox_overhead_pct"] = merged["bbox_overhead_pct"].fillna(merged["bbox_overhead_pct"].median())
        merged["isoperimetric_q"] = merged["isoperimetric_q"].fillna(merged["isoperimetric_q"].median())

    out = Path(args.out) if args.out else pool.with_name(pool.stem + "_topo.csv")
    merged.to_csv(out, index=False)
    print(f"wrote {out}: {len(merged):,} rows  cols: {len(merged.columns)}")
    print(f"new feature stats:")
    for c in NEW_COLS:
        print(f"  {c:30s} min={merged[c].min():>7.3f}  median={merged[c].median():>7.3f}  max={merged[c].max():>7.3f}  nonzero={(merged[c]>0).sum():,}")


if __name__ == "__main__":
    main()
