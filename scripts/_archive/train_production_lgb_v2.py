"""Train the production LightGBM-logT surrogate on the AUGMENTED pool (v2).

Reads:    results/oracle_loop_extended_2026_05_22/training_matrix_v2.csv
Outputs:  results/oracle_loop_extended_2026_05_22/production_lgb_logT_v2.pkl
          results/oracle_loop_extended_2026_05_22/production_lgb_logT_v2.json

Uses identical hyperparameters as v1 so we can isolate the augmentation effect.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=0.5,
    min_child_samples=10,
    n_jobs=4,
    random_state=42,
    verbosity=-1,
)

print("Loading augmented training data ...")
pool = pd.read_csv(RUN / "training_matrix_v2.csv")
hold = pd.read_csv(RUN / "holdout_extreme.csv")
pool_combo = build_combo_features(pool[ALL_COLS])
hold_combo = build_combo_features(hold[ALL_COLS])

print(f"  pool    : {len(pool):,} rows   ({pool['plz'].nunique()} PLZs, {pool['provider'].nunique()} LSPs)")
print(f"  holdout : {len(hold):,} rows   ({hold['plz'].nunique()} PLZs, {hold['provider'].nunique()} LSPs)")
print(f"  features: {len(pool_combo.columns)}")

print("\nFitting LightGBM-logT (production HPs):")
for k, v in LGB_HPS.items():
    print(f"  {k:20s} = {v}")

t0 = time.time()
model = TransformedTargetRegressor(
    regressor=lgb.LGBMRegressor(**LGB_HPS),
    func=np.log1p, inverse_func=np.expm1,
)
model.fit(pool_combo.values, pool["actual_cost_eur"].values)
fit_sec = time.time() - t0

print(f"\nFit complete in {fit_sec:.1f}s")

# Evaluate
yp_tr = model.predict(pool_combo.values)
yp_te = model.predict(hold_combo.values)
m_tr = safe_metrics(pool["actual_cost_eur"].values, yp_tr)
m_te = safe_metrics(hold["actual_cost_eur"].values, yp_te)

print(f"\nMetrics:")
print(f"  training pool  : MAPE={m_tr['mape']:.3f}%  MAE={m_tr['mae']:.2f}EUR  R2={m_tr['r2']:.4f}")
print(f"  frozen holdout : MAPE={m_te['mape']:.3f}%  MAE={m_te['mae']:.2f}EUR  R2={m_te['r2']:.4f}")

# Compare to v1 metrics if available
v1_json = RUN / "production_lgb_logT_v1.json"
if v1_json.exists():
    v1m = json.loads(v1_json.read_text(encoding="utf-8"))
    print(f"\nDelta vs v1 (positive = v2 better on R2 / lower MAPE):")
    print(f"  pool MAPE     : v1={v1m.get('metrics',{}).get('pool',{}).get('mape','N/A')} -> v2={m_tr['mape']:.3f}")
    print(f"  holdout MAPE  : v1={v1m.get('metrics',{}).get('holdout',{}).get('mape','N/A')} -> v2={m_te['mape']:.3f}")
    print(f"  holdout R2    : v1={v1m.get('metrics',{}).get('holdout',{}).get('r2','N/A')} -> v2={m_te['r2']:.4f}")

# Save
model_path = RUN / "production_lgb_logT_v2.pkl"
meta_path = RUN / "production_lgb_logT_v2.json"

with open(model_path, "wb") as f:
    pickle.dump({
        "model": model,
        "feature_cols": list(pool_combo.columns),
        "hps": LGB_HPS,
        "train_size": len(pool),
        "augmented": True,
    }, f)

meta = {
    "version": "v2_augmented",
    "fit_seconds": fit_sec,
    "n_train_rows": int(len(pool)),
    "n_holdout_rows": int(len(hold)),
    "metrics": {
        "pool": {k: float(v) for k, v in m_tr.items()},
        "holdout": {k: float(v) for k, v in m_te.items()},
    },
    "hps": LGB_HPS,
}
meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"\nSaved: {model_path}")
print(f"Saved: {meta_path}")
