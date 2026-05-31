"""Train auxiliary LGB models for distance_km and n_routes on v2 training pool.

These complement production_lgb_logT_v2 (cost target) so the willingness
analysis can predict the full triplet (cost, distance, routes) — enabling
paper-style Pareto / Fleet plots without VROOM re-runs.

Outputs:
    results/oracle_loop_extended_2026_05_22/aux_lgb_distance_v2.pkl
    results/oracle_loop_extended_2026_05_22/aux_lgb_routes_v2.pkl
    *_v2.json (metadata + holdout metrics)

Uses log1p transform on both targets (similar to cost) since both have
right-skewed distributions across the training pool.
"""
from __future__ import annotations
import json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_helpers import safe_metrics  # noqa: E402
from batch_delivery.features import ALL_COLS  # noqa: E402
from batch_delivery.surrogate import build_combo_features  # noqa: E402

from sklearn.compose import TransformedTargetRegressor
import lightgbm as lgb

RUN = ROOT / "results" / "oracle_loop_extended_2026_05_22"
LGB_HPS = dict(
    n_estimators=1000, learning_rate=0.05, num_leaves=31, max_depth=-1,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=0.5, min_child_samples=10,
    n_jobs=4, random_state=42, verbosity=-1,
)


def train(target_col: str, suffix: str):
    print(f"\n=== Training {suffix} (target={target_col}) ===")
    pool = pd.read_csv(RUN / "training_matrix.csv")
    hold = pd.read_csv(RUN / "holdout_extreme.csv")
    pool_combo = build_combo_features(pool[ALL_COLS])
    hold_combo = build_combo_features(hold[ALL_COLS])

    # Drop rows with missing or non-positive target
    y_train = pool[target_col].values.astype(np.float64)
    mask = np.isfinite(y_train) & (y_train > 0)
    if not mask.all():
        print(f"  dropping {(~mask).sum()} rows with bad {target_col}")
    X_train = pool_combo.values[mask]
    y_train = y_train[mask]

    y_hold = hold[target_col].values.astype(np.float64)
    h_mask = np.isfinite(y_hold) & (y_hold > 0)
    X_hold = hold_combo.values[h_mask]
    y_hold = y_hold[h_mask]

    print(f"  pool {len(X_train):,} rows | holdout {len(X_hold):,} rows")
    print(f"  target range: [{y_train.min():.2f}, {y_train.max():.2f}]")

    t0 = time.time()
    model = TransformedTargetRegressor(
        regressor=lgb.LGBMRegressor(**LGB_HPS),
        func=np.log1p, inverse_func=np.expm1,
    )
    model.fit(X_train, y_train)
    fit_sec = time.time() - t0

    yp_tr = model.predict(X_train)
    yp_te = model.predict(X_hold)
    m_tr = safe_metrics(y_train, yp_tr)
    m_te = safe_metrics(y_hold, yp_te)
    print(f"  fit {fit_sec:.1f}s")
    print(f"  train  : MAPE={m_tr['mape']:.3f}%  MAE={m_tr['mae']:.3f}  R2={m_tr['r2']:.4f}")
    print(f"  holdout: MAPE={m_te['mape']:.3f}%  MAE={m_te['mae']:.3f}  R2={m_te['r2']:.4f}")

    out_pkl = RUN / f"aux_lgb_{suffix}_v2.pkl"
    out_json = RUN / f"aux_lgb_{suffix}_v2.json"
    with open(out_pkl, "wb") as f:
        pickle.dump({"model": model, "feature_cols": list(pool_combo.columns),
                       "target": target_col, "kind": f"aux_{suffix}_log1p"}, f)
    (out_json).write_text(json.dumps({
        "target": target_col, "suffix": suffix,
        "fit_seconds": fit_sec,
        "n_train_rows": int(len(y_train)),
        "n_holdout_rows": int(len(y_hold)),
        "metrics_train": {k: float(v) for k, v in m_tr.items()},
        "metrics_holdout": {k: float(v) for k, v in m_te.items()},
        "hps": LGB_HPS,
    }, indent=2), encoding="utf-8")
    print(f"  saved: {out_pkl.relative_to(ROOT)}")


if __name__ == "__main__":
    train("actual_distance_km", "distance")
    train("actual_n_routes", "routes")
