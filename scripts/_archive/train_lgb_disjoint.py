"""Train an LGB-logT model with 25 base + 7 disjoint topology features.

FW6.A Option 3: extends the production LGB with member-level topology
features so the model can distinguish single-PLZ from merged-cluster
geometries with the same area/density.

Pool requirements:
    columns include ALL_COLS (25) + the 7 new topo cols:
        n_merged_members, n_components_post_union, hull_overhead_pct,
        bbox_overhead_pct, max_member_centroid_km, mean_member_centroid_km,
        isoperimetric_q

Use scripts/add_disjoint_to_pool.py to build the _topo pool first.

Usage:
    python scripts/train_lgb_disjoint.py \\
        --pool results/sweep_v3_mergefix/training_matrix_topo.csv \\
        --out  results/sweep_v3_mergefix/lgb_disjoint_v3.pkl
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_helpers import safe_metrics  # noqa: E402
from batch_delivery.features import ALL_COLS  # noqa: E402
from batch_delivery.surrogate import build_combo_features  # noqa: E402

TOPO_COLS = [
    "n_merged_members", "n_components_post_union",
    "hull_overhead_pct", "bbox_overhead_pct",
    "max_member_centroid_km", "mean_member_centroid_km",
    "isoperimetric_q",
]

LGB_HPS = dict(
    n_estimators=1000, learning_rate=0.05, num_leaves=31, max_depth=-1,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=0.5, min_child_samples=10,
    n_jobs=4, random_state=42, verbosity=-1,
)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """44 base combo features + 7 topo passthrough."""
    base = build_combo_features(df[ALL_COLS])
    topo = df[TOPO_COLS].copy()
    return pd.concat([base.reset_index(drop=True), topo.reset_index(drop=True)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout-frac", type=float, default=1 / 7)
    ap.add_argument("--seed", type=int, default=20260526)
    args = ap.parse_args()

    pool_path = Path(args.pool)
    pool = pd.read_csv(pool_path)
    pool["plz"] = pool["plz"].astype(str).str.zfill(5)
    missing = [c for c in (ALL_COLS + TOPO_COLS) if c not in pool.columns]
    if missing:
        raise SystemExit(f"pool missing required cols: {missing}")
    print(f"pool: {len(pool):,} rows, {pool['plz'].nunique()} cluster_ids, {pool['provider'].nunique()} LSPs")

    rng = np.random.default_rng(args.seed)
    all_plz = sorted(pool["plz"].unique())
    n_hold = max(1, int(round(len(all_plz) * args.holdout_frac)))
    holdout_plz = rng.choice(all_plz, size=n_hold, replace=False).tolist()
    print(f"holdout cluster_ids ({len(holdout_plz)}): {holdout_plz[:8]}...")
    train = pool[~pool["plz"].isin(holdout_plz)]
    hold = pool[pool["plz"].isin(holdout_plz)]
    print(f"train: {len(train):,}  holdout: {len(hold):,}")

    X_tr = make_features(train)
    X_te = make_features(hold)

    print(f"\nFeatures: {len(X_tr.columns)} cols (44 base combos + {len(TOPO_COLS)} topo)")

    t0 = time.time()
    model = TransformedTargetRegressor(
        regressor=lgb.LGBMRegressor(**LGB_HPS),
        func=np.log1p, inverse_func=np.expm1,
    )
    model.fit(X_tr.values, train["actual_cost_eur"].values)
    fit_sec = time.time() - t0
    print(f"fit done in {fit_sec:.1f}s")

    yp_tr = model.predict(X_tr.values)
    yp_te = model.predict(X_te.values)
    m_tr = safe_metrics(train["actual_cost_eur"].values, yp_tr)
    m_te = safe_metrics(hold["actual_cost_eur"].values, yp_te)
    print(f"\nMetrics:")
    print(f"  train  : MAPE={m_tr['mape']:.3f}%  MAE={m_tr['mae']:.2f}EUR  R2={m_tr['r2']:.4f}")
    print(f"  holdout: MAPE={m_te['mape']:.3f}%  MAE={m_te['mae']:.2f}EUR  R2={m_te['r2']:.4f}")

    # FW6.A subset performance: rows where mean_member_centroid_km > 0
    mask_disjoint = train["mean_member_centroid_km"] > 0
    if mask_disjoint.any():
        m_dj_tr = safe_metrics(
            train.loc[mask_disjoint, "actual_cost_eur"].values,
            yp_tr[mask_disjoint.values],
        )
        print(f"  train@disjoint (mean_member_centroid_km>0, n={mask_disjoint.sum()}):"
              f"  MAPE={m_dj_tr['mape']:.3f}%")
    mask_disjoint_h = hold["mean_member_centroid_km"] > 0
    if mask_disjoint_h.any():
        m_dj_te = safe_metrics(
            hold.loc[mask_disjoint_h, "actual_cost_eur"].values,
            yp_te[mask_disjoint_h.values],
        )
        print(f"  hold@disjoint  (n={mask_disjoint_h.sum()}):  MAPE={m_dj_te['mape']:.3f}%")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({
            "kind": "LGBLogTDisjoint",
            "model": model,
            "feature_cols": list(X_tr.columns),
            "topo_cols": TOPO_COLS,
            "all_cols": ALL_COLS,
            "hps": LGB_HPS,
            "train_size": len(train),
            "holdout_plz": holdout_plz,
        }, f)
    meta = {
        "version": "lgb_disjoint",
        "fit_seconds": fit_sec,
        "n_train_rows": int(len(train)),
        "n_holdout_rows": int(len(hold)),
        "holdout_cluster_ids": holdout_plz,
        "metrics": {"pool": {k: float(v) for k, v in m_tr.items()},
                    "holdout": {k: float(v) for k, v in m_te.items()}},
        "hps": LGB_HPS,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
