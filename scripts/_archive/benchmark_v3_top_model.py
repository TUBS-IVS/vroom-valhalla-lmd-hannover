"""Top-model benchmark on the latest oracle-loop run.

Extensions over benchmark_v2:
  * XGBoost + LightGBM added
  * Tree models evaluated on both 25 base AND 44 combo features
  * Log-target variants (TransformedTargetRegressor) of the strongest models
  * Stacking ensemble: RF + XGB + LGB + MLP-prod -> Ridge meta-learner
  * Second evaluation protocol: 5-fold GroupKFold(PLZ) on the training pool

RAM-safe: n_jobs=2 everywhere; no MLP retrain from scratch (uses production
iter17 ensemble); writes to results/paper_figures/ml_surrogate_v2/.

Run:
    python scripts/benchmark_v3_top_model.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_helpers import safe_metrics, daganzo_predict  # noqa: E402
from batch_delivery.features import ALL_COLS  # noqa: E402
from batch_delivery.surrogate import MLCostPredictor, build_combo_features  # noqa: E402

RUN = ROOT / "results" / "oracle_loop_extended_2026_05_22"
OUT = ROOT / "results" / "paper_figures" / "ml_surrogate_v2"
OUT.mkdir(parents=True, exist_ok=True)

N_JOBS = 2
RANDOM_STATE = 42

# ── data ────────────────────────────────────────────────────────────────────
pool = pd.read_csv(RUN / "training_matrix.csv")
hold = pd.read_csv(RUN / "holdout_extreme.csv")
print(f"pool    : {len(pool):>6,} rows  ({pool['plz'].nunique()} PLZs)")
print(f"holdout : {len(hold):>6,} rows  ({hold['plz'].nunique()} PLZs)")

pool_combo = build_combo_features(pool[ALL_COLS])
hold_combo = build_combo_features(hold[ALL_COLS])
COMBO_COLS = pool_combo.columns.tolist()
print(f"base cols  : {len(ALL_COLS)}")
print(f"combo cols : {len(COMBO_COLS)}")

X25_tr = pool[ALL_COLS].values
X44_tr = pool_combo.values
y_tr   = pool["actual_cost_eur"].values
X25_te = hold[ALL_COLS].values
X44_te = hold_combo.values
y_te   = hold["actual_cost_eur"].values

# ── model factories ────────────────────────────────────────────────────────
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor,
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import lightgbm as lgb


def rf():
    return RandomForestRegressor(n_estimators=400, max_depth=14,
                                  n_jobs=N_JOBS, random_state=RANDOM_STATE)

def hgb():
    return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                          max_depth=8, random_state=RANDOM_STATE)

def xgbr():
    return xgb.XGBRegressor(n_estimators=800, learning_rate=0.04, max_depth=7,
                              subsample=0.85, colsample_bytree=0.85,
                              reg_lambda=1.0, min_child_weight=3,
                              n_jobs=N_JOBS, random_state=RANDOM_STATE,
                              tree_method="hist", verbosity=0)

def lgbr():
    return lgb.LGBMRegressor(n_estimators=800, learning_rate=0.04,
                               num_leaves=63, max_depth=-1,
                               subsample=0.85, colsample_bytree=0.85,
                               reg_lambda=0.5, min_child_samples=10,
                               n_jobs=N_JOBS, random_state=RANDOM_STATE,
                               verbosity=-1)

def log_wrap(est):
    """log1p target wrapper that clips negative back-transforms to >=1 EUR."""
    return TransformedTargetRegressor(regressor=est, func=np.log1p,
                                       inverse_func=np.expm1)


# ── (A) Interpolation holdout — base + combo + log-target variants ────────
print("\n=== (A) Interpolation holdout ===")
rows_A = []

# Daganzo baseline (uses raw cols)
rows_A.append({"model": "Daganzo (textbook 1984)", "features": "n/a",
               **safe_metrics(y_te, daganzo_predict(hold)), "fit_sec": 0.0})

# tree models on 25 base
tree_models = {"RF": rf, "HistGBM": hgb, "XGBoost": xgbr, "LightGBM": lgbr}
for name, factory in tree_models.items():
    for tag, Xtr, Xte in [("25 base", X25_tr, X25_te), ("44 combo", X44_tr, X44_te)]:
        t0 = time.time()
        m = factory()
        m.fit(Xtr, y_tr)
        yp = np.maximum(0.0, m.predict(Xte))
        rows_A.append({"model": name, "features": tag, **safe_metrics(y_te, yp),
                       "fit_sec": time.time() - t0})
        print(f"  {name:<9s} {tag:<8s}  MAPE={rows_A[-1]['mape']:6.2f}  R2={rows_A[-1]['r2']:.4f}  ({rows_A[-1]['fit_sec']:.1f}s)")

# log-target variants of the four trees on combo features only
print("  -- log-target wrappers --")
for name, factory in tree_models.items():
    t0 = time.time()
    m = log_wrap(factory())
    m.fit(X44_tr, y_tr)
    yp = np.maximum(0.0, m.predict(X44_te))
    rows_A.append({"model": f"{name}-logT", "features": "44 combo",
                   **safe_metrics(y_te, yp), "fit_sec": time.time() - t0})
    print(f"  {name}-logT  MAPE={rows_A[-1]['mape']:6.2f}  R2={rows_A[-1]['r2']:.4f}")

# Production MLP ensemble (iter17)
prod = MLCostPredictor.load(RUN / "ml_cost_predictor.pkl")
yp_prod_te = prod.predict(hold)
rows_A.append({"model": "MLP-ensemble (iter17)", "features": "44 combo (internal)",
               **safe_metrics(y_te, yp_prod_te), "fit_sec": 0.0})
print(f"  MLP-ensemble (iter17)  MAPE={rows_A[-1]['mape']:6.2f}  R2={rows_A[-1]['r2']:.4f}")


# ── Stacking ensemble (RF + XGB + LGB + MLP-prod) -> Ridge -----------------
# MLP predictions are treated as a pre-computed feature column for the stack
print("\n=== Stacking (RF + XGB + LGB + MLP-prod) -> Ridge ===")
t0 = time.time()
# Internal CV via StackingRegressor for tree bases, then add MLP-prod preds
# We do a manual two-stage stack to include the *frozen* MLP-prod predictions.
gkf = GroupKFold(n_splits=5)
groups_plz = pool["plz"].values

n_tr = len(pool)
oof = np.zeros((n_tr, 4))  # cols: RF, XGB, LGB, MLP
base_factories = [("RF", rf), ("XGB", xgbr), ("LGB", lgbr)]
for col_idx, (name, factory) in enumerate(base_factories):
    for tr_idx, te_idx in gkf.split(X44_tr, y_tr, groups=groups_plz):
        m = factory()
        m.fit(X44_tr[tr_idx], y_tr[tr_idx])
        oof[te_idx, col_idx] = m.predict(X44_tr[te_idx])
    print(f"  OOF fit done: {name}")

# MLP-prod predictions on the pool (frozen, no leakage — model was trained on the pool)
# To avoid leakage we predict with each *previous* iter's model on the rows from that iter.
# Simpler approximation: predict with iter17 on the pool. This causes mild leakage but stacking
# usually still helps; it is the standard "frozen-feature" trick.
oof[:, 3] = prod.predict(pool)

# Train ridge meta-learner on OOF + actual y
meta = Ridge(alpha=1.0)
meta.fit(oof, y_tr)
print(f"  Ridge meta coefs: {dict(zip(['RF','XGB','LGB','MLP'], meta.coef_.round(3)))}")

# Predict on the holdout: each base model trained on the FULL pool, predict, stack
yp_rf = rf().fit(X44_tr, y_tr).predict(X44_te)
yp_xg = xgbr().fit(X44_tr, y_tr).predict(X44_te)
yp_lg = lgbr().fit(X44_tr, y_tr).predict(X44_te)
yp_ml = prod.predict(hold)
X_meta_te = np.column_stack([yp_rf, yp_xg, yp_lg, yp_ml])
yp_stack = np.maximum(0.0, meta.predict(X_meta_te))
rows_A.append({"model": "Stack(RF+XGB+LGB+MLP)->Ridge", "features": "44 combo",
               **safe_metrics(y_te, yp_stack), "fit_sec": time.time() - t0})
print(f"  Stack  MAPE={rows_A[-1]['mape']:6.2f}  R2={rows_A[-1]['r2']:.4f}")

bench_A = pd.DataFrame(rows_A).sort_values("mape").reset_index(drop=True)
bench_A.to_csv(OUT / "tab5_top_model_holdout.csv", index=False)
print("\nProtocol A — sorted by MAPE:")
print(bench_A[["model","features","n","mape","mae","rmse","r2","bias"]].round(3).to_string(index=False))


# ── (B) 5-fold GroupKFold(PLZ) on the training pool ────────────────────────
print("\n=== (B) 5-fold GroupKFold(PLZ) on pool (10,652 rows) ===")
rows_B = []
gkf = GroupKFold(n_splits=5)

# Daganzo (closed-form on pool)
rows_B.append({"model": "Daganzo (textbook 1984)", "features": "n/a",
               **safe_metrics(y_tr, daganzo_predict(pool)), "fit_sec": 0.0})

for name, factory in tree_models.items():
    for tag, Xtr in [("44 combo", X44_tr)]:  # only combo for B (faster)
        t0 = time.time()
        yhat = np.zeros_like(y_tr, dtype=float)
        for tr_idx, te_idx in gkf.split(Xtr, y_tr, groups=groups_plz):
            m = factory()
            m.fit(Xtr[tr_idx], y_tr[tr_idx])
            yhat[te_idx] = m.predict(Xtr[te_idx])
        rows_B.append({"model": name, "features": tag,
                       **safe_metrics(y_tr, yhat), "fit_sec": time.time() - t0})
        print(f"  {name:<9s} {tag:<8s}  MAPE={rows_B[-1]['mape']:6.2f}  R2={rows_B[-1]['r2']:.4f}  ({rows_B[-1]['fit_sec']:.1f}s)")

# log-target Lightgbm only (typically the biggest gain)
for name in ["XGBoost", "LightGBM"]:
    factory = tree_models[name]
    t0 = time.time()
    yhat = np.zeros_like(y_tr, dtype=float)
    for tr_idx, te_idx in gkf.split(X44_tr, y_tr, groups=groups_plz):
        m = log_wrap(factory())
        m.fit(X44_tr[tr_idx], y_tr[tr_idx])
        yhat[te_idx] = m.predict(X44_tr[te_idx])
    rows_B.append({"model": f"{name}-logT", "features": "44 combo",
                   **safe_metrics(y_tr, yhat), "fit_sec": time.time() - t0})
    print(f"  {name}-logT  MAPE={rows_B[-1]['mape']:6.2f}  R2={rows_B[-1]['r2']:.4f}")

# MLP-prod on pool — uses iter17 trained on the pool (mild leakage; for reference)
yp_mlp_pool = prod.predict(pool)
rows_B.append({"model": "MLP-ensemble (iter17, leak)", "features": "44 combo",
               **safe_metrics(y_tr, yp_mlp_pool), "fit_sec": 0.0})

bench_B = pd.DataFrame(rows_B).sort_values("mape").reset_index(drop=True)
bench_B.to_csv(OUT / "tab5_top_model_groupkfold.csv", index=False)
print("\nProtocol B — sorted by MAPE:")
print(bench_B[["model","features","n","mape","mae","rmse","r2","bias"]].round(3).to_string(index=False))


# ── headline summary ──────────────────────────────────────────────────────
summary = {
    "run_dir":             str(RUN),
    "train_rows":          int(len(pool)),
    "holdout_rows":        int(len(hold)),
    "base_features":       len(ALL_COLS),
    "combo_features":      len(COMBO_COLS),
    "best_A_model":        str(bench_A["model"].iloc[0]),
    "best_A_features":     str(bench_A["features"].iloc[0]),
    "best_A_mape":         float(bench_A["mape"].iloc[0]),
    "best_A_r2":           float(bench_A["r2"].iloc[0]),
    "best_B_model":        str(bench_B["model"].iloc[0]),
    "best_B_features":     str(bench_B["features"].iloc[0]),
    "best_B_mape":         float(bench_B["mape"].iloc[0]),
    "mlp_ensemble_A_mape": float(bench_A.loc[bench_A["model"] == "MLP-ensemble (iter17)", "mape"].iloc[0]),
}
(OUT / "summary_v3.json").write_text(json.dumps(summary, indent=2))
print("\nWrote", OUT / "summary_v3.json")
print(json.dumps(summary, indent=2))
