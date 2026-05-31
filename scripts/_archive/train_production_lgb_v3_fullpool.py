"""Production LGB-logT v3 trained on FULL v3 pool — NO holdout.

The original train_production_lgb_v3.py holds out 7 PLZ for evaluation,
then saves that model as production. This deprives production of training
data for those clusters (including FW6.A: 30159, 30449, etc).

Correct workflow:
    train_production_lgb_v3.py        — dev/eval model (with holdout)
    train_production_lgb_v3_fullpool.py — production deployment model

Both use the same hyperparameters; only train_size differs.
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

RUN = ROOT / "results" / "sweep_v3_mergefix"
LGB_HPS = dict(
    n_estimators=1000, learning_rate=0.05, num_leaves=31, max_depth=-1,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=0.5, min_child_samples=10,
    n_jobs=4, random_state=42, verbosity=-1,
)

print("Loading FULL v3 training pool ...")
pool = pd.read_csv(RUN / "training_matrix.csv")
print(f"  pool: {len(pool):,} rows, {pool['plz'].nunique()} cluster_ids, {pool['provider'].nunique()} LSPs")
pool_combo = build_combo_features(pool[ALL_COLS])

print("\nFitting LGB-logT v3 on FULL pool (no holdout) ...")
t0 = time.time()
model = TransformedTargetRegressor(
    regressor=lgb.LGBMRegressor(**LGB_HPS),
    func=np.log1p, inverse_func=np.expm1,
)
model.fit(pool_combo.values, pool["actual_cost_eur"].values)
fit_sec = time.time() - t0
print(f"  fit done in {fit_sec:.1f}s")

yp = model.predict(pool_combo.values)
m_all = safe_metrics(pool["actual_cost_eur"].values, yp)
print(f"\n  pool MAPE = {m_all['mape']:.3f}%  R2 = {m_all['r2']:.4f}")

# Per-cluster training fit
fw_mask = pool["plz"].astype(str).str.zfill(5).isin(["30159","30167","30449"])
if fw_mask.any():
    m_fw = safe_metrics(pool.loc[fw_mask,"actual_cost_eur"].values, yp[fw_mask.values])
    print(f"  FW6.A pool fit (in-training): MAPE={m_fw['mape']:.3f}%  abs-median={np.median(np.abs(pool.loc[fw_mask,'actual_cost_eur'].values - yp[fw_mask.values])/pool.loc[fw_mask,'actual_cost_eur'].values)*100:.2f}%")

out_pkl = RUN / "production_lgb_logT_v3_fullpool.pkl"
out_json = RUN / "production_lgb_logT_v3_fullpool.json"
with open(out_pkl, "wb") as f:
    pickle.dump({"model": model, "feature_cols": list(pool_combo.columns),
                 "hps": LGB_HPS, "train_size": len(pool),
                 "holdout_plz": [], "augmented": True, "mergefix": True,
                 "fullpool": True}, f)
out_json.write_text(json.dumps({
    "version": "v3_mergefix_fullpool",
    "fit_seconds": fit_sec,
    "n_train_rows": int(len(pool)),
    "n_holdout_rows": 0,
    "metrics": {"pool": {k: float(v) for k, v in m_all.items()}},
    "hps": LGB_HPS,
}, indent=2), encoding="utf-8")
print(f"\nSaved: {out_pkl.name}")
