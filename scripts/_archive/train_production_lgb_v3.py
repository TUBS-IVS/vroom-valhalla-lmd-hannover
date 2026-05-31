"""Train production LightGBM-logT v3 on the merge-fix-aware training pool.

Reads:  results/sweep_v3_mergefix/training_matrix.csv  (fresh, all fixes applied)
Writes: results/sweep_v3_mergefix/production_lgb_logT_v3.{pkl,json}

Uses identical hyperparameters as v1/v2 so we isolate the data-correctness effect.
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
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

RUN = ROOT / "results" / "sweep_v3_mergefix"
LGB_HPS = dict(
    n_estimators=1000, learning_rate=0.05, num_leaves=31, max_depth=-1,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=0.5, min_child_samples=10,
    n_jobs=4, random_state=42, verbosity=-1,
)

print("Loading merge-fix-aware training data ...")
pool = pd.read_csv(RUN / "training_matrix.csv")
print(f"  pool: {len(pool):,} rows, {pool['plz'].nunique()} cluster_ids, {pool['provider'].nunique()} LSPs")

# Hold out 15% of cluster_ids for a *true* generalization test
rng = np.random.default_rng(20260525)
all_plz = sorted(pool["plz"].unique())
n_hold = max(1, len(all_plz) // 7)
holdout_plz = rng.choice(all_plz, size=n_hold, replace=False).tolist()
print(f"  holdout cluster_ids ({len(holdout_plz)}): {holdout_plz[:10]}...")
pool_train = pool[~pool["plz"].isin(holdout_plz)]
pool_hold = pool[pool["plz"].isin(holdout_plz)]
print(f"  train: {len(pool_train):,}  holdout: {len(pool_hold):,}")

pool_combo = build_combo_features(pool_train[ALL_COLS])
hold_combo = build_combo_features(pool_hold[ALL_COLS])

print("\nFitting LGB-logT v3 ...")
t0 = time.time()
model = TransformedTargetRegressor(
    regressor=lgb.LGBMRegressor(**LGB_HPS),
    func=np.log1p, inverse_func=np.expm1,
)
model.fit(pool_combo.values, pool_train["actual_cost_eur"].values)
fit_sec = time.time() - t0
print(f"  fit done in {fit_sec:.1f}s")

yp_tr = model.predict(pool_combo.values)
yp_te = model.predict(hold_combo.values)
m_tr = safe_metrics(pool_train["actual_cost_eur"].values, yp_tr)
m_te = safe_metrics(pool_hold["actual_cost_eur"].values, yp_te)
print(f"\nMetrics:")
print(f"  train  : MAPE={m_tr['mape']:.3f}%  MAE={m_tr['mae']:.2f}EUR  R2={m_tr['r2']:.4f}")
print(f"  holdout: MAPE={m_te['mape']:.3f}%  MAE={m_te['mae']:.2f}EUR  R2={m_te['r2']:.4f}")

with open(RUN / "production_lgb_logT_v3.pkl", "wb") as f:
    pickle.dump({"model": model, "feature_cols": list(pool_combo.columns),
                  "hps": LGB_HPS, "train_size": len(pool_train),
                  "holdout_plz": holdout_plz, "augmented": True, "mergefix": True}, f)
meta = {"version": "v3_mergefix",
         "fit_seconds": fit_sec,
         "n_train_rows": int(len(pool_train)),
         "n_holdout_rows": int(len(pool_hold)),
         "holdout_cluster_ids": holdout_plz,
         "metrics": {"pool": {k: float(v) for k, v in m_tr.items()},
                       "holdout": {k: float(v) for k, v in m_te.items()}},
         "hps": LGB_HPS}
(RUN / "production_lgb_logT_v3.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"\nSaved: {RUN}/production_lgb_logT_v3.{{pkl,json}}")
