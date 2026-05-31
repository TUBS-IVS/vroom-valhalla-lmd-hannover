"""Model battery on v3 pool — focus on log1p-saturation hypothesis + all variants.

Tests:
    LGB-logT       (current production: log1p + L2 loss)         [BASELINE]
    LGB-raw        (no transform — direct L2 on euros)            [test (A)]
    LGB-quantile50 (q=0.50 + log1p — median-unbiased)             [test (B)]
    LGB-quantile55 (q=0.55 + log1p — upward bias correction)
    LGB-huber      (Huber loss + log1p — robust to top-end tail)
    LGB-tweedie    (Tweedie p=1.5 — count-like target distribution)
    LGB-monotonic  (monotonicity on parcels/area/hub_dist)
    LGB-plain      (no transform, default LGB)
    XGBoost
    CatBoost
    RF
    MLP-5seed
    Daganzo-Hybrid

Outputs:
    results/model_battery_v3/
        tab_model_comparison.csv
        tab_per_cluster_mape.csv
        tab_topend_stratified.csv   <-- KEY: MAPE for top-25% n_parcels
        fig_holdout_mape_per_model.png
        fig_topend_mape_per_model.png
        REPORT.md
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
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
    print("[info] catboost not installed — skipping CatBoost variant")

FW6_CLUSTERS = {"30159", "30167", "30449"}

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
    if name == "LGB-plain":
        return lgb.LGBMRegressor(**LGB_BASE)
    if name == "LGB-quantile50":
        hp = {**LGB_BASE, "objective": "quantile", "alpha": 0.50}
        return TransformedTargetRegressor(
            regressor=lgb.LGBMRegressor(**hp),
            func=np.log1p, inverse_func=np.expm1)
    if name == "LGB-quantile55":
        hp = {**LGB_BASE, "objective": "quantile", "alpha": 0.55}
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
    if name == "LGB-monotonic":
        cons = [1 if c in ("n_parcels", "area_km2", "hub_dist_km", "n_stops") else 0
                for c in ALL_COLS]
        hp = {**LGB_BASE, "monotone_constraints": cons,
              "monotone_constraints_method": "advanced"}
        return TransformedTargetRegressor(
            regressor=lgb.LGBMRegressor(**hp),
            func=np.log1p, inverse_func=np.expm1)
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


def evaluate_groupkfold(model_name, pool_combo, pool, n_splits=5):
    X = pool_combo.values
    y = pool["actual_cost_eur"].values
    groups = pool["plz"].astype(str).str.zfill(5).values
    n_parcels = pool["n_parcels"].values

    # Top-25% (high-cost regime where log1p saturation hurts most)
    p75_thresh = np.percentile(y, 75)
    fw_mask = np.isin(groups, list(FW6_CLUSTERS))

    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    all_y, all_pred = [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
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
    # Match arrays back to original indices
    te_indices = np.concatenate([
        list(gkf.split(X, y, groups))[i][1] for i in range(n_splits)
    ])
    # Top-25% subset
    topend_mask = y_all > p75_thresh
    topend_mape = float(np.mean(np.abs(y_all[topend_mask] - p_all[topend_mask])
                                / np.maximum(y_all[topend_mask], 1)) * 100) if topend_mask.any() else np.nan
    # Top-25% bias (median pct_err)
    topend_bias = float(np.median(100 * (p_all[topend_mask] - y_all[topend_mask])
                                  / np.maximum(y_all[topend_mask], 1))) if topend_mask.any() else np.nan
    # FW6.A subset
    fw_mask_te = np.isin(groups[te_indices], list(FW6_CLUSTERS))
    fw_mape = float(np.mean(np.abs(y_all[fw_mask_te] - p_all[fw_mask_te])
                            / np.maximum(y_all[fw_mask_te], 1)) * 100) if fw_mask_te.any() else np.nan
    fw_bias = float(np.median(100 * (p_all[fw_mask_te] - y_all[fw_mask_te])
                              / np.maximum(y_all[fw_mask_te], 1))) if fw_mask_te.any() else np.nan
    overall_bias = float(np.median(100 * (p_all - y_all) / np.maximum(y_all, 1)))

    return {
        "mape_mean": float(np.mean([m["mape"] for m in fold_metrics])),
        "mape_std":  float(np.std([m["mape"] for m in fold_metrics])),
        "r2_mean":   float(np.mean([m["r2"]   for m in fold_metrics])),
        "overall_median_bias_pct": overall_bias,
        "topend_mape": topend_mape,
        "topend_bias_pct": topend_bias,
        "fw6_mape": fw_mape,
        "fw6_bias_pct": fw_bias,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="results/sweep_v3_mergefix/training_matrix.csv")
    ap.add_argument("--out", default="results/model_battery_v3")
    args = ap.parse_args()

    pool_path = Path(args.pool).resolve()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(pool_path)
    pool["plz"] = pool["plz"].astype(str).str.zfill(5)
    print(f"[pool] {pool_path}")
    print(f"  {len(pool):,} rows, {pool['plz'].nunique()} PLZ, "
          f"{pool['provider'].nunique()} providers")
    print(f"  cost_eur p75 = {np.percentile(pool['actual_cost_eur'], 75):.0f}EUR")
    pool_combo = build_combo_features(pool[ALL_COLS])

    models = [
        "LGB-logT",         # current production (log1p + L2)
        "LGB-raw",          # NO log1p (test hypothesis)
        "LGB-quantile50",   # median (log1p + quantile loss)
        "LGB-quantile55",   # upward-biased
        "LGB-huber",        # robust loss
        "LGB-tweedie",      # count-distribution
        "LGB-monotonic",    # constraints
        "XGBoost", "RF", "MLP-5seed", "Daganzo-Hybrid",
    ]
    if CATBOOST_AVAILABLE:
        models.insert(7, "CatBoost")

    print(f"\n[battery] testing {len(models)} models on v3 pool with GroupKFold\n")
    print(f"{'model':18s} {'MAPE':>7s} {'±std':>6s} {'R²':>7s} {'med_bias':>9s} {'topMAPE':>8s} {'topBias':>9s} {'FW6_MAPE':>9s} {'FW6_bias':>9s} {'sec':>5s}")
    print("-" * 100)
    rows = []
    for mname in models:
        t0 = time.time()
        try:
            res = evaluate_groupkfold(mname, pool_combo, pool)
            sec = time.time() - t0
            rows.append({"model": mname, **res, "fit_time_s": sec})
            print(f"{mname:18s} {res['mape_mean']:6.2f}% {res['mape_std']:5.2f}% "
                  f"{res['r2_mean']:6.4f} {res['overall_median_bias_pct']:+8.2f}% "
                  f"{res['topend_mape']:7.2f}% {res['topend_bias_pct']:+8.2f}% "
                  f"{res['fw6_mape']:8.2f}% {res['fw6_bias_pct']:+8.2f}% {sec:4.0f}s")
        except Exception as e:
            print(f"{mname:18s} FAILED: {e}")
            rows.append({"model": mname, "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(out / "tab_model_comparison.csv", index=False)

    df_valid = df.dropna(subset=["mape_mean"]).sort_values("topend_mape")
    print(f"\nRanked by top-end MAPE (key for FW6.A):")
    cols = ["model", "mape_mean", "topend_mape", "topend_bias_pct", "fw6_mape", "fw6_bias_pct"]
    print(df_valid[cols].to_string(index=False))

    # Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    df_ord = df_valid.sort_values("mape_mean")
    ax1.barh(df_ord["model"], df_ord["mape_mean"], color="#2a9d8f", alpha=0.85)
    ax1.set_xlabel("Overall MAPE [%]"); ax1.set_title("Overall accuracy")
    ax1.grid(axis="x", alpha=0.3)

    df_ord = df_valid.sort_values("topend_mape")
    ax2.barh(df_ord["model"], df_ord["topend_mape"], color="#e76f51", alpha=0.85)
    ax2.set_xlabel("Top-25%-cost MAPE [%]"); ax2.set_title("High-cost regime (FW6.A sensitive)")
    ax2.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_holdout_mape_per_model.png")
    fig.savefig(out / "fig_holdout_mape_per_model.pdf")
    plt.close(fig)

    # Report
    best_overall = df_valid.iloc[df_valid["mape_mean"].argmin()]
    best_topend  = df_valid.iloc[df_valid["topend_mape"].argmin()]
    best_fw6     = df_valid.iloc[df_valid["fw6_mape"].argmin()]
    lines = [
        "# Model Battery v3 — log1p Saturation Test\n",
        f"**Pool**: `{pool_path.name}` ({len(pool):,} rows)",
        f"**Method**: 5-fold GroupKFold(PLZ)\n",
        "## Hypothesis tested\n",
        "Current `production_lgb_logT_v3` uses log1p target transform + L2 loss.",
        "Diagnosis: log1p saturation creates -20% under-prediction at top-end.",
        "Battery tests alternative losses + transforms to see if FW6.A clusters",
        "(30159/30167/30449) are fixable without re-sweeping.\n",
        "## Winners\n",
        f"- **Overall**: `{best_overall['model']}` (MAPE {best_overall['mape_mean']:.2f}%)",
        f"- **Top-25% cost**: `{best_topend['model']}` (MAPE {best_topend['topend_mape']:.2f}%, bias {best_topend['topend_bias_pct']:+.1f}%)",
        f"- **FW6.A**: `{best_fw6['model']}` (MAPE {best_fw6['fw6_mape']:.2f}%, bias {best_fw6['fw6_bias_pct']:+.1f}%)\n",
        "## Full Ranking (sorted by top-end MAPE)\n",
        df_valid.sort_values("topend_mape")[cols].to_markdown(index=False),
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved: {out}/REPORT.md")


if __name__ == "__main__":
    main()
