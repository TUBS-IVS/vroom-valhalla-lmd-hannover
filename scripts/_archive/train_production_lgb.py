"""Train the production LightGBM-logT surrogate on the full cumulative pool.

Uses the hyperparameters chosen by Nested 5x3 GroupKFold(PLZ) HP search
(validation_v4_clean.py):
    n_estimators = 1000
    learning_rate = 0.05
    num_leaves = 31           (reduced from prior 63 to curb light overfit)
    max_depth = -1
    subsample = 0.85
    colsample_bytree = 0.85
    reg_lambda = 0.5
    min_child_samples = 10

Saves the trained model + metadata JSON to
    results/oracle_loop_extended_2026_05_22/production_lgb_logT_v1.pkl
    results/oracle_loop_extended_2026_05_22/production_lgb_logT_v1.json

A drop-in predictor class lives in batch_delivery.surrogate.lgb_surrogate.
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

# Tuned HPs from validation_v4_clean.py Nested-CV V5
LGB_HPS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=0.5,
    min_child_samples=10,
    n_jobs=2,                # leave headroom for the running oracle loop
    random_state=42,
    verbosity=-1,
)

print("Loading training data ...")
pool = pd.read_csv(RUN / "training_matrix.csv")
hold = pd.read_csv(RUN / "holdout_extreme.csv")
pool_combo = build_combo_features(pool[ALL_COLS])
hold_combo = build_combo_features(hold[ALL_COLS])

print(f"  pool    : {len(pool):,} rows   ({pool['plz'].nunique()} PLZs, {pool['provider'].nunique()} LSPs)")
print(f"  holdout : {len(hold):,} rows   ({hold['plz'].nunique()} PLZs, {hold['provider'].nunique()} LSPs)")
print(f"  features: {len(pool_combo.columns)} (25 base + 8 interaction + 11 log)")

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

# Evaluate on holdout
yp_tr = model.predict(pool_combo.values)
yp_te = model.predict(hold_combo.values)
m_tr = safe_metrics(pool["actual_cost_eur"].values, yp_tr)
m_te = safe_metrics(hold["actual_cost_eur"].values, yp_te)

print(f"\nMetrics:")
print(f"  training pool  : MAPE={m_tr['mape']:.3f}%  MAE={m_tr['mae']:.2f}€  R2={m_tr['r2']:.4f}")
print(f"  frozen holdout : MAPE={m_te['mape']:.3f}%  MAE={m_te['mae']:.2f}€  R2={m_te['r2']:.4f}")

# Save model + metadata
model_path = RUN / "production_lgb_logT_v1.pkl"
meta_path  = RUN / "production_lgb_logT_v1.json"

with open(model_path, "wb") as f:
    pickle.dump({
        "model": model,
        "feature_cols": pool_combo.columns.tolist(),
        "all_cols": ALL_COLS,
        "hps": LGB_HPS,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_run": str(RUN),
    }, f)

metadata = {
    "model_type":         "LightGBM-logT (TransformedTargetRegressor + LGBMRegressor)",
    "model_path":         str(model_path),
    "training_run":       str(RUN),
    "n_train_rows":       int(len(pool)),
    "n_holdout_rows":     int(len(hold)),
    "n_plz_train":        int(pool["plz"].nunique()),
    "n_plz_holdout":      int(hold["plz"].nunique()),
    "n_features":         len(pool_combo.columns),
    "feature_set":        "25 base + 8 interaction + 11 log-transform = 44 combo",
    "hyperparameters":    LGB_HPS,
    "training_metrics":   m_tr,
    "holdout_metrics":    m_te,
    "fit_time_sec":       fit_sec,
    "hp_source":          "Nested 5x3 GroupKFold(PLZ) HP search in scripts/validation_v4_clean.py",
    "notes":              ("num_leaves reduced from 63 to 31 based on Nested-CV "
                            "evidence of mild PLZ-memorisation overfit. Use this model "
                            "as drop-in replacement for the MLP-ensemble in scenario "
                            "evaluation."),
}
meta_path.write_text(json.dumps(metadata, indent=2, default=str, ensure_ascii=False))

print(f"\nWrote {model_path.name}  ({model_path.stat().st_size/1024:.1f} kB)")
print(f"Wrote {meta_path.name}")
print("\nLoad with:")
print("  import pickle; m = pickle.load(open('production_lgb_logT_v1.pkl','rb'))")
print("  pred = m['model'].predict(build_combo_features(df).values)")
