"""In-pool model battery — KFold on ROWS (not PLZ groups).

Every PLZ is in BOTH train and test for every fold. This answers the question
"which model fits our existing data best" without the noise of unseen-PLZ
extrapolation that GroupKFold introduces. Production deployment uses the full
pool with no holdout, so in-pool fit is the operationally relevant metric.

Usage:
    python scripts/model_battery_inpool.py --pool results/oracle_loop_extended_2026_05_22/training_matrix_v2.csv
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_helpers import safe_metrics  # noqa: E402
from batch_delivery.features import ALL_COLS  # noqa: E402
from batch_delivery.surrogate import build_combo_features  # noqa: E402
from batch_delivery.legacy.daganzo import daganzo_vrp_cost_v0  # noqa: E402

CATBOOST_AVAILABLE = False
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    pass

LGB_BASE = dict(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                max_depth=-1, subsample=0.85, colsample_bytree=0.85,
                reg_lambda=0.5, min_child_samples=10, n_jobs=4,
                random_state=42, verbosity=-1)


def daganzo_vec(np_a, ns_a, area_a, hd_a):
    return np.array([
        daganzo_vrp_cost_v0(int(np_a[i]), int(max(1, ns_a[i])),
                            float(area_a[i]), float(hd_a[i]))
        for i in range(len(np_a))
    ], dtype=np.float64)


def make_model(name: str):
    if name == "LGB-logT":
        return TransformedTargetRegressor(
            regressor=lgb.LGBMRegressor(**LGB_BASE),
            func=np.log1p, inverse_func=np.expm1)
    if name == "LGB-raw":
        return lgb.LGBMRegressor(**LGB_BASE)
    if name == "LGB-quantile50":
        hp = {**LGB_BASE, "objective": "quantile", "alpha": 0.50}
        return TransformedTargetRegressor(
            regressor=lgb.LGBMRegressor(**hp),
            func=np.log1p, inverse_func=np.expm1)
    if name == "LGB-huber":
        hp = {**LGB_BASE, "objective": "huber", "alpha": 0.9}
        return TransformedTargetRegressor(
            regressor=lgb.LGBMRegressor(**hp),
            func=np.log1p, inverse_func=np.expm1)
    if name == "LGB-tweedie":
        hp = {**LGB_BASE, "objective": "tweedie", "tweedie_variance_power": 1.5}
        return lgb.LGBMRegressor(**hp)
    if name == "XGBoost":
        return xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=0.5,
            n_jobs=4, random_state=42, verbosity=0, objective="reg:squarederror")
    if name == "CatBoost":
        return CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6,
                                 random_seed=42, verbose=False, thread_count=4)
    if name == "RF":
        return RandomForestRegressor(n_estimators=300, max_depth=None,
                                     min_samples_leaf=3, n_jobs=4, random_state=42)
    if name == "MLP-5seed":
        return None
    raise ValueError(f"Unknown model: {name}")


class MLPEnsemble:
    def __init__(self, seeds=(42, 123, 456, 789, 2026)):
        self.seeds = seeds; self.scaler = None; self.models = []
    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        for s in self.seeds:
            m = MLPRegressor(hidden_layer_sizes=(128, 64), activation="tanh",
                             solver="adam", learning_rate_init=1e-3,
                             max_iter=400, early_stopping=True,
                             validation_fraction=0.1, random_state=s, verbose=False)
            m.fit(Xs, np.log1p(y)); self.models.append(m)
        return self
    def predict(self, X):
        Xs = self.scaler.transform(X)
        preds = np.array([m.predict(Xs) for m in self.models])
        return np.expm1(preds.mean(axis=0))


def evaluate_inpool_kfold(model_name, pool_combo, pool, n_splits=5):
    """KFold on ROWS — every PLZ in both train and test."""
    X = pool_combo.values
    y = pool["actual_cost_eur"].values
    p75 = np.percentile(y, 75)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    all_y, all_pred = [], []
    for fold, (tr, te) in enumerate(kf.split(X)):
        if model_name == "MLP-5seed":
            m = MLPEnsemble().fit(X[tr], y[tr]); pred = m.predict(X[te])
        elif model_name == "Daganzo-Hybrid":
            np_a = pool["n_parcels"].values[tr]
            ns_a = pool["n_stops"].values[tr]
            area_a = pool["area_km2"].values[tr]
            hd_a = pool["hub_dist_km"].values[tr]
            daganzo_tr = daganzo_vec(np_a, ns_a, area_a, hd_a)
            residual_tr = y[tr] - daganzo_tr
            lgb_res = lgb.LGBMRegressor(**LGB_BASE)
            lgb_res.fit(X[tr], residual_tr)
            np_a_te = pool["n_parcels"].values[te]
            ns_a_te = pool["n_stops"].values[te]
            area_a_te = pool["area_km2"].values[te]
            hd_a_te = pool["hub_dist_km"].values[te]
            daganzo_te = daganzo_vec(np_a_te, ns_a_te, area_a_te, hd_a_te)
            pred = daganzo_te + lgb_res.predict(X[te])
        else:
            m = make_model(model_name)
            if m is None: continue
            m.fit(X[tr], y[tr]); pred = m.predict(X[te])
        all_y.append(y[te]); all_pred.append(pred)
        fold_metrics.append(safe_metrics(y[te], pred))

    y_all = np.concatenate(all_y)
    p_all = np.concatenate(all_pred)

    topend_mask = y_all > p75
    topend_mape = float(np.mean(np.abs(y_all[topend_mask] - p_all[topend_mask])
                                / np.maximum(y_all[topend_mask], 1)) * 100) if topend_mask.any() else np.nan
    topend_bias = float(np.median(100 * (p_all[topend_mask] - y_all[topend_mask])
                                  / np.maximum(y_all[topend_mask], 1))) if topend_mask.any() else np.nan
    overall_bias = float(np.median(100 * (p_all - y_all) / np.maximum(y_all, 1)))

    return {
        "mape_mean": float(np.mean([m["mape"] for m in fold_metrics])),
        "mape_std":  float(np.std([m["mape"] for m in fold_metrics])),
        "r2_mean":   float(np.mean([m["r2"]   for m in fold_metrics])),
        "overall_median_bias_pct": overall_bias,
        "topend_mape": topend_mape,
        "topend_bias_pct": topend_bias,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", default="results/model_battery_inpool")
    args = ap.parse_args()

    pool_path = Path(args.pool).resolve()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(pool_path)
    pool["plz"] = pool["plz"].astype(str).str.zfill(5)
    print(f"[pool] {pool_path}")
    print(f"  {len(pool):,} rows, {pool['plz'].nunique()} PLZ, "
          f"{pool['provider'].nunique()} providers")
    pool_combo = build_combo_features(pool[ALL_COLS])

    models = [
        "LGB-logT", "LGB-raw", "LGB-quantile50", "LGB-huber", "LGB-tweedie",
        "XGBoost", "RF", "MLP-5seed", "Daganzo-Hybrid",
    ]
    if CATBOOST_AVAILABLE:
        models.insert(5, "CatBoost")

    print(f"\n[battery] {len(models)} models, 5-fold KFold on ROWS (in-pool)\n")
    print(f"{'model':18s} {'MAPE':>7s} {'±std':>6s} {'R²':>7s} {'med_bias':>9s} {'topMAPE':>8s} {'topBias':>9s} {'sec':>5s}")
    print("-" * 85)
    rows = []
    for mname in models:
        t0 = time.time()
        try:
            res = evaluate_inpool_kfold(mname, pool_combo, pool)
            sec = time.time() - t0
            rows.append({"model": mname, **res, "fit_time_s": sec})
            print(f"{mname:18s} {res['mape_mean']:6.2f}% {res['mape_std']:5.2f}% "
                  f"{res['r2_mean']:6.4f} {res['overall_median_bias_pct']:+8.2f}% "
                  f"{res['topend_mape']:7.2f}% {res['topend_bias_pct']:+8.2f}% {sec:4.0f}s")
        except Exception as e:
            print(f"{mname:18s} FAILED: {e}")
            rows.append({"model": mname, "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(out / "tab_model_inpool.csv", index=False)
    print(f"\nSaved: {out/'tab_model_inpool.csv'}")
    df_valid = df.dropna(subset=["mape_mean"]).sort_values("mape_mean")
    print(f"\nRanked by in-pool MAPE (lower=better):")
    print(df_valid[["model","mape_mean","mape_std","topend_mape","topend_bias_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
