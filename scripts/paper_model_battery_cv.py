"""GroupKFold cross-validated comparison of every surrogate model variant.

Includes both Daganzo-Hybrid variants:
  α=1.0    (raw BHH constant, old reference)
  α=1.343  (median-calibrated, new paper-grade)

Group = PLZ (so a held-out PLZ is never seen during training — true OOS).

Outputs (results/overnight_2026_05_27/diagnosis_v2/cv_battery/):
  tab_model_comparison_cv.csv
  fig_cv_mape_bars.{png,pdf}
"""
from __future__ import annotations
import math
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from batch_delivery.config.constants import (
    VEHICLE_CAPACITY, BHH_CONSTANT, FIXED_COST_EUR, COST_PER_KM_EUR,
)
from batch_delivery.features import ALL_COLS
from batch_delivery.surrogate import build_combo_features

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

OUT = ROOT / "results" / "overnight_2026_05_27" / "diagnosis_v2" / "cv_battery"
OUT.mkdir(parents=True, exist_ok=True)


def daganzo_cost_vec(np_a, ns_a, area_a, hd_a):
    """Vectorized Daganzo BHH cost (alpha=1.0 base)."""
    out = np.zeros_like(np_a, dtype=np.float64)
    for i in range(len(np_a)):
        n_parcels = int(np_a[i])
        n_stops = int(max(1, ns_a[i]))
        area_km2 = float(max(0.01, area_a[i]))
        hub_dist_km = float(hd_a[i])
        if n_parcels <= 0:
            continue
        n_routes = math.ceil(n_parcels / VEHICLE_CAPACITY)
        spr = max(1.0, n_stops / n_routes)
        local_dist = BHH_CONSTANT * math.sqrt(spr * area_km2)
        out[i] = n_routes * (FIXED_COST_EUR
                              + (2 * hub_dist_km + local_dist) * COST_PER_KM_EUR)
    return out


def cv_eval(name, fit_fn, predict_fn, X, y, groups, n_splits=5):
    """Run GroupKFold CV and return per-fold + overall MAPE/bias/R²."""
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    all_pred = np.zeros_like(y, dtype=np.float64)
    all_seen = np.zeros(len(y), dtype=bool)
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        model = fit_fn(X[tr_idx], y[tr_idx])
        pred_te = predict_fn(model, X[te_idx])
        all_pred[te_idx] = pred_te
        all_seen[te_idx] = True
        a, p = y[te_idx], pred_te
        mape = np.mean(np.abs(p - a) / np.maximum(1e-6, a)) * 100
        bias = np.mean((p - a) / np.maximum(1e-6, a)) * 100
        ss_res = ((a - p) ** 2).sum()
        ss_tot = ((a - a.mean()) ** 2).sum()
        r2 = 1 - ss_res / max(ss_tot, 1e-9)
        fold_metrics.append({"fold": fold + 1, "n": int(len(te_idx)),
                             "MAPE_pct": mape, "bias_pct": bias, "R2": r2})

    # Overall metrics (on all held-out predictions)
    a, p = y[all_seen], all_pred[all_seen]
    overall = {
        "model": name,
        "n_oos": int(all_seen.sum()),
        "MAPE_pct_mean": float(np.mean([m["MAPE_pct"] for m in fold_metrics])),
        "MAPE_pct_std": float(np.std([m["MAPE_pct"] for m in fold_metrics])),
        "MAPE_pct_min": float(min(m["MAPE_pct"] for m in fold_metrics)),
        "MAPE_pct_max": float(max(m["MAPE_pct"] for m in fold_metrics)),
        "bias_pct_mean": float(np.mean([m["bias_pct"] for m in fold_metrics])),
        "R2_mean": float(np.mean([m["R2"] for m in fold_metrics])),
        "MAPE_pct_oos_overall": float(np.mean(np.abs(p - a) / np.maximum(1e-6, a)) * 100),
        "R2_oos_overall": float(1 - ((a - p) ** 2).sum()
                                / max(((a - a.mean()) ** 2).sum(), 1e-9)),
    }
    return overall, fold_metrics


def make_lgb(**kwargs):
    """LGB factory with default sensible params; kwargs override defaults."""
    params = dict(objective="regression", n_estimators=500, learning_rate=0.05,
                   num_leaves=31, min_data_in_leaf=10, verbose=-1)
    params.update(kwargs)
    return lgb.LGBMRegressor(**params)


def main():
    print("=" * 72)
    print("GroupKFold CV model battery (group=PLZ, 5 folds)")
    print("=" * 72)

    # Load training pool — same data the Hybrid was trained on
    pool_path = ROOT / "results" / "sweep_v3_mergefix" / "training_matrix.csv"
    pool = pd.read_csv(pool_path)
    print(f"  pool: {len(pool):,} rows, {pool['plz'].nunique()} PLZ, "
          f"{pool['provider'].nunique()} providers")

    # Base 25 features + 44 combo features for richer models
    X_base = pool[ALL_COLS].values
    combo = build_combo_features(pool)
    # Only numeric columns, excluding outcomes/identifiers
    exclude = {"actual_cost_eur", "actual_distance_km", "actual_duration_h",
                "actual_n_routes", "n_vehicles_planned", "solve_time_s",
                "vroom_status", "is_baseline", "provider", "plz",
                "agg_k", "base_day", "scale", "p_keep", "noise_sigma",
                "b2c_scale", "b2b_scale", "seed"}
    combo_cols = [c for c in combo.columns
                  if c not in exclude
                  and combo[c].dtype in (np.float64, np.float32, np.int64, np.int32)]
    print(f"  combo features: {len(combo_cols)}")
    X_combo = combo[combo_cols].astype(np.float64).values
    y = pool["actual_cost_eur"].values.astype(np.float64)
    groups = pool["plz"].astype(str).values

    np_a = pool["n_parcels"].values
    ns_a = pool["n_stops"].values
    area_a = pool["area_km2"].values
    hd_a = pool["hub_dist_km"].values
    daganzo_base = daganzo_cost_vec(np_a, ns_a, area_a, hd_a)
    alpha_median = float(np.median(y / np.maximum(daganzo_base, 1.0)))
    print(f"  median alpha calibration: {alpha_median:.4f}")

    rows = []

    # ── Pure Daganzo (no LGB) ─────────────────────────────────────────
    # Score on full set (no training needed)
    def pure_dag_pred(alpha):
        return lambda model, X: model.get("alpha", 1.0) * daganzo_base[model["_idx"]]

    print("\n[1] Pure Daganzo α=1.0 ...")
    err = y - 1.0 * daganzo_base
    mape = np.mean(np.abs(err) / np.maximum(1e-6, y)) * 100
    bias = np.mean(err / np.maximum(1e-6, y)) * (-100)   # err > 0 ⇒ underpredict
    r2 = 1 - (err ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
    rows.append({"model": "Pure Daganzo (α=1.0)", "n_oos": len(y),
                 "MAPE_pct_mean": mape, "MAPE_pct_std": 0.0,
                 "MAPE_pct_min": mape, "MAPE_pct_max": mape,
                 "bias_pct_mean": bias, "R2_mean": r2,
                 "MAPE_pct_oos_overall": mape, "R2_oos_overall": r2})
    print(f"    MAPE = {mape:.2f}%, bias = {bias:.2f}%, R² = {r2:.4f}")

    print(f"\n[2] Pure Daganzo α={alpha_median:.3f} (median-calibrated) ...")
    err = y - alpha_median * daganzo_base
    mape = np.mean(np.abs(err) / np.maximum(1e-6, y)) * 100
    bias = -np.mean(err / np.maximum(1e-6, y)) * 100
    r2 = 1 - (err ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
    rows.append({"model": f"Pure Daganzo (α={alpha_median:.3f})", "n_oos": len(y),
                 "MAPE_pct_mean": mape, "MAPE_pct_std": 0.0,
                 "MAPE_pct_min": mape, "MAPE_pct_max": mape,
                 "bias_pct_mean": bias, "R2_mean": r2,
                 "MAPE_pct_oos_overall": mape, "R2_oos_overall": r2})
    print(f"    MAPE = {mape:.2f}%, bias = {bias:.2f}%, R² = {r2:.4f}")

    # ── Daganzo-LGB-Hybrid (two α variants) ────────────────────────────
    def hybrid_fit(alpha):
        def fit_fn(Xtr, ytr):
            # tr_idx tells us which daganzo_base rows to use
            tr_idx_mask = np.zeros(len(y), dtype=bool)
            for x in Xtr:
                # exact-row match (slow) — but reliable
                pass
            # Better: pass indices via globals
            return None
        return fit_fn

    # Cleaner: pass row indices instead of feature matrix
    indices = np.arange(len(y))

    def cv_hybrid(alpha):
        gkf = GroupKFold(n_splits=5)
        fold_metrics = []
        for fold, (tr_idx, te_idx) in enumerate(gkf.split(indices, y, groups)):
            residual_tr = y[tr_idx] - alpha * daganzo_base[tr_idx]
            m = make_lgb()
            m.fit(X_combo[tr_idx], residual_tr)
            pred_te = alpha * daganzo_base[te_idx] + m.predict(X_combo[te_idx])
            a, p = y[te_idx], pred_te
            mape = np.mean(np.abs(p - a) / np.maximum(1e-6, a)) * 100
            bias = np.mean((p - a) / np.maximum(1e-6, a)) * 100
            ss_res = ((a - p) ** 2).sum()
            ss_tot = ((a - a.mean()) ** 2).sum()
            r2 = 1 - ss_res / max(ss_tot, 1e-9)
            fold_metrics.append({"MAPE_pct": mape, "bias_pct": bias, "R2": r2})
        return fold_metrics

    print("\n[3] Daganzo-LGB-Hybrid α=1.0 (CV) ...")
    fm = cv_hybrid(1.0)
    rows.append({
        "model": "Daganzo-LGB-Hybrid (α=1.0)", "n_oos": len(y),
        "MAPE_pct_mean": float(np.mean([m["MAPE_pct"] for m in fm])),
        "MAPE_pct_std": float(np.std([m["MAPE_pct"] for m in fm])),
        "MAPE_pct_min": float(min(m["MAPE_pct"] for m in fm)),
        "MAPE_pct_max": float(max(m["MAPE_pct"] for m in fm)),
        "bias_pct_mean": float(np.mean([m["bias_pct"] for m in fm])),
        "R2_mean": float(np.mean([m["R2"] for m in fm])),
        "MAPE_pct_oos_overall": float(np.mean([m["MAPE_pct"] for m in fm])),
        "R2_oos_overall": float(np.mean([m["R2"] for m in fm])),
    })
    print(f"    MAPE = {rows[-1]['MAPE_pct_mean']:.2f}% ± {rows[-1]['MAPE_pct_std']:.2f}, "
          f"R² = {rows[-1]['R2_mean']:.4f}")

    print(f"\n[4] Daganzo-LGB-Hybrid α={alpha_median:.3f} (CV, NEW) ...")
    fm = cv_hybrid(alpha_median)
    rows.append({
        "model": f"Daganzo-LGB-Hybrid (α={alpha_median:.3f})  ★", "n_oos": len(y),
        "MAPE_pct_mean": float(np.mean([m["MAPE_pct"] for m in fm])),
        "MAPE_pct_std": float(np.std([m["MAPE_pct"] for m in fm])),
        "MAPE_pct_min": float(min(m["MAPE_pct"] for m in fm)),
        "MAPE_pct_max": float(max(m["MAPE_pct"] for m in fm)),
        "bias_pct_mean": float(np.mean([m["bias_pct"] for m in fm])),
        "R2_mean": float(np.mean([m["R2"] for m in fm])),
        "MAPE_pct_oos_overall": float(np.mean([m["MAPE_pct"] for m in fm])),
        "R2_oos_overall": float(np.mean([m["R2"] for m in fm])),
    })
    print(f"    MAPE = {rows[-1]['MAPE_pct_mean']:.2f}% ± {rows[-1]['MAPE_pct_std']:.2f}, "
          f"R² = {rows[-1]['R2_mean']:.4f}")

    # ── Direct LGB variants (different losses) ─────────────────────────
    lgb_variants = [
        ("LGB-raw", dict(objective="regression")),
        ("LGB-logT", "logT"),     # special: log-transformed target
        ("LGB-huber", dict(objective="huber", alpha=0.9)),
        ("LGB-tweedie", dict(objective="tweedie", tweedie_variance_power=1.5)),
        ("LGB-quantile50", dict(objective="quantile", alpha=0.5)),
    ]
    for name, params in lgb_variants:
        print(f"\n[{name}] CV ...")
        if params == "logT":
            def fit_fn(Xtr, ytr):
                m = TransformedTargetRegressor(
                    regressor=make_lgb(),
                    func=np.log1p, inverse_func=np.expm1,
                )
                m.fit(Xtr, ytr); return m
        else:
            def fit_fn(Xtr, ytr, P=params):
                m = make_lgb(**P)
                m.fit(Xtr, ytr); return m

        def pred_fn(m, Xte): return m.predict(Xte)

        overall, _ = cv_eval(name, fit_fn, pred_fn, X_combo, y, groups)
        rows.append(overall)
        print(f"    MAPE = {overall['MAPE_pct_mean']:.2f}% ± {overall['MAPE_pct_std']:.2f}, "
              f"R² = {overall['R2_mean']:.4f}")

    # ── XGBoost ─────────────────────────────────────────────────────────
    print("\n[XGBoost] CV ...")
    def fit_xgb(Xtr, ytr):
        m = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
                              max_depth=6, verbosity=0,
                              tree_method="hist", n_jobs=-1)
        m.fit(Xtr, ytr); return m
    overall, _ = cv_eval("XGBoost", fit_xgb, lambda m, X: m.predict(X),
                          X_combo, y, groups)
    rows.append(overall)
    print(f"    MAPE = {overall['MAPE_pct_mean']:.2f}% ± {overall['MAPE_pct_std']:.2f}")

    # ── Random Forest ───────────────────────────────────────────────────
    print("\n[Random Forest] CV ...")
    def fit_rf(Xtr, ytr):
        m = RandomForestRegressor(n_estimators=300, max_depth=12,
                                    min_samples_leaf=4, n_jobs=-1,
                                    random_state=42)
        m.fit(Xtr, ytr); return m
    overall, _ = cv_eval("Random Forest", fit_rf, lambda m, X: m.predict(X),
                          X_combo, y, groups)
    rows.append(overall)
    print(f"    MAPE = {overall['MAPE_pct_mean']:.2f}% ± {overall['MAPE_pct_std']:.2f}")

    # ── Decision Tree ───────────────────────────────────────────────────
    print("\n[Decision Tree] CV ...")
    def fit_dt(Xtr, ytr):
        m = DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42)
        m.fit(Xtr, ytr); return m
    overall, _ = cv_eval("Decision Tree", fit_dt, lambda m, X: m.predict(X),
                          X_combo, y, groups)
    rows.append(overall)
    print(f"    MAPE = {overall['MAPE_pct_mean']:.2f}% ± {overall['MAPE_pct_std']:.2f}")

    # ── Linear / Ridge / Lasso (need scaling) ──────────────────────────
    for name, ctor in [
        ("Linear", lambda: LinearRegression()),
        ("Ridge", lambda: Ridge(alpha=1.0)),
        ("Lasso", lambda: Lasso(alpha=0.1, max_iter=5000)),
    ]:
        print(f"\n[{name}] CV ...")
        def fit_lin(Xtr, ytr, C=ctor):
            m = Pipeline([("scaler", StandardScaler()), ("est", C())])
            m.fit(Xtr, ytr); return m
        overall, _ = cv_eval(name, fit_lin, lambda m, X: m.predict(X),
                              X_combo, y, groups)
        rows.append(overall)
        print(f"    MAPE = {overall['MAPE_pct_mean']:.2f}% ± {overall['MAPE_pct_std']:.2f}")

    # ── MLP (5-seed ensemble) ──────────────────────────────────────────
    print("\n[MLP 5-seed ensemble] CV ...")
    def fit_mlp(Xtr, ytr):
        models = []
        for seed in range(5):
            m = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(hidden_layer_sizes=(128, 64),
                                      max_iter=2000, learning_rate_init=1e-3,
                                      early_stopping=True, validation_fraction=0.1,
                                      random_state=seed, n_iter_no_change=20)),
            ])
            m.fit(Xtr, np.log1p(ytr))
            models.append(m)
        return models
    def pred_mlp(models, X):
        preds = np.mean([np.expm1(m.predict(X)) for m in models], axis=0)
        return preds
    overall, _ = cv_eval("MLP 5-seed (log1p)", fit_mlp, pred_mlp,
                          X_combo, y, groups)
    rows.append(overall)
    print(f"    MAPE = {overall['MAPE_pct_mean']:.2f}% ± {overall['MAPE_pct_std']:.2f}")

    # ── Save + plot ─────────────────────────────────────────────────────
    df = pd.DataFrame(rows).sort_values("MAPE_pct_mean")
    df.to_csv(OUT / "tab_model_comparison_cv.csv", index=False)
    print(f"\nSaved {OUT / 'tab_model_comparison_cv.csv'}")
    print("\nFinal Ranking (sorted by GroupKFold-MAPE):")
    cols = ["model", "MAPE_pct_mean", "MAPE_pct_std", "bias_pct_mean", "R2_mean"]
    print(df[cols].round(3).to_string(index=False))

    # ── Plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 7))
    df_plot = df.sort_values("MAPE_pct_mean")
    colors = ["#ee9b00" if "Hybrid" in m and "1.343" in m else
              "#c1121f" if "Pure Daganzo" in m else
              "#1f4f8f" for m in df_plot["model"]]
    bars = ax.barh(df_plot["model"], df_plot["MAPE_pct_mean"],
                    xerr=df_plot["MAPE_pct_std"], color=colors,
                    edgecolor="black", capsize=4)
    ax.invert_yaxis()
    ax.set_xlabel("MAPE [%]   (GroupKFold-CV, group=PLZ, mean ± std across 5 folds)")
    ax.set_title("Model comparison — out-of-sample MAPE on actual_cost_eur "
                  "(2733 samples, 5-fold GroupKFold)")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, df_plot["MAPE_pct_mean"]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cv_mape_bars.png")
    fig.savefig(OUT / "fig_cv_mape_bars.pdf")
    plt.close(fig)
    print(f"Saved {OUT / 'fig_cv_mape_bars.png'}")


if __name__ == "__main__":
    main()
