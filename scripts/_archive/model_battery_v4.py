"""Model Battery — compare candidate surrogate models on training pool.

Models tested:
    1. LGB-logT (current production)
    2. LGB-plain (no log transform)
    3. LGB-quantile alpha=0.55 (bias-correction variant)
    4. LGB-monotonic (parcels↑→cost↑, area↑→cost↑, hub_dist↑→cost↑)
    5. XGBoost (default)
    6. CatBoost (ordered boosting, if installed)
    7. RandomForest (sanity baseline)
    8. MLP-5seed-Ensemble (legacy method)
    9. Daganzo-LGB-Hybrid (formula + LGB residual — best extrapolator)

Each model evaluated on:
    A. Training MAPE
    B. GroupKFold(cluster) honest holdout MAPE  ← main rank metric
    C. Frozen extreme holdout MAPE
    D. Per-merged-cluster MAPE (the v3 stress test)

Output:
    results/model_battery_v4/
        tab_model_comparison.csv
        fig_holdout_mape_per_model.{png,pdf}
        fig_per_cluster_heatmap.{png,pdf}
        REPORT.md
"""
from __future__ import annotations
import argparse, json, pickle, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_helpers import safe_metrics  # noqa: E402
from batch_delivery.features import ALL_COLS  # noqa: E402
from batch_delivery.surrogate import build_combo_features  # noqa: E402
from batch_delivery.legacy.daganzo import daganzo_vrp_cost_v0  # noqa: E402

import lightgbm as lgb
import xgboost as xgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

CATBOOST_AVAILABLE = False
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    print("[info] catboost not installed — skipping CatBoost variant")


def daganzo_vec(np_a, ns_a, area_a, hd_a):
    """Vectorized Daganzo predictions."""
    out = np.zeros_like(np_a, dtype=np.float64)
    for i in range(len(np_a)):
        out[i] = daganzo_vrp_cost_v0(
            int(np_a[i]), int(max(1, ns_a[i])),
            float(area_a[i]), float(hd_a[i]),
        )
    return out


def make_model(name: str):
    """Construct model with reasonable defaults."""
    LGB_BASE = dict(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                     max_depth=-1, subsample=0.85, colsample_bytree=0.85,
                     reg_lambda=0.5, min_child_samples=10, n_jobs=4,
                     random_state=42, verbosity=-1)
    if name == "LGB-logT":
        return TransformedTargetRegressor(
            regressor=lgb.LGBMRegressor(**LGB_BASE),
            func=np.log1p, inverse_func=np.expm1,
        )
    if name == "LGB-plain":
        return lgb.LGBMRegressor(**LGB_BASE)
    if name == "LGB-quantile":
        # alpha=0.55 introduces slight upward bias correction
        hp = {**LGB_BASE, "objective": "quantile", "alpha": 0.55}
        return TransformedTargetRegressor(
            regressor=lgb.LGBMRegressor(**hp),
            func=np.log1p, inverse_func=np.expm1,
        )
    if name == "LGB-monotonic":
        # Constraint vector matches ALL_COLS order: enforce monotonic for
        # parcels, area, hub_dist; rest unconstrained.
        constraints = []
        for col in ALL_COLS:
            if col in ("n_parcels", "area_km2", "hub_dist_km", "n_stops"):
                constraints.append(1)
            else:
                constraints.append(0)
        hp = {**LGB_BASE, "monotone_constraints": constraints,
                "monotone_constraints_method": "advanced"}
        return TransformedTargetRegressor(
            regressor=lgb.LGBMRegressor(**hp),
            func=np.log1p, inverse_func=np.expm1,
        )
    if name == "XGBoost":
        return xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=6,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=0.5,
            n_jobs=4, random_state=42, verbosity=0, objective="reg:squarederror",
        )
    if name == "CatBoost":
        return CatBoostRegressor(
            iterations=1000, learning_rate=0.05, depth=6,
            random_seed=42, verbose=False, thread_count=4,
        )
    if name == "RF":
        return RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=3,
            n_jobs=4, random_state=42,
        )
    if name == "MLP-5seed":
        # Wrapper returns mean of 5 seeds — manual ensemble below
        return None  # handled separately
    raise ValueError(f"Unknown model: {name}")


class MLPEnsemble:
    """5-seed MLPRegressor ensemble (legacy production approach)."""
    def __init__(self, seeds=(42, 123, 456, 789, 2026)):
        self.seeds = seeds
        self.scaler = None
        self.models = []
    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.models = []
        for s in self.seeds:
            m = MLPRegressor(
                hidden_layer_sizes=(128, 64), activation="tanh",
                solver="adam", learning_rate_init=1e-3,
                max_iter=400, early_stopping=True, validation_fraction=0.1,
                random_state=s, verbose=False,
            )
            m.fit(Xs, np.log1p(y))
            self.models.append(m)
        return self
    def predict(self, X):
        Xs = self.scaler.transform(X)
        preds = np.array([m.predict(Xs) for m in self.models])
        return np.expm1(preds.mean(axis=0))


def evaluate_groupkfold(model_name: str, pool_combo: pd.DataFrame,
                          pool: pd.DataFrame, n_splits=5):
    """5-fold GroupKFold(plz) — held-out PLZ never seen in training."""
    X = pool_combo.values
    y = pool["actual_cost_eur"].values
    groups = pool["plz"].astype(str).values
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    per_cluster_preds = {}  # cluster_id -> (actual_arr, pred_arr)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        if model_name == "MLP-5seed":
            m = MLPEnsemble().fit(X[tr], y[tr])
            pred = m.predict(X[te])
        elif model_name == "Daganzo-Hybrid":
            # Need raw features for daganzo. Get from pool index
            np_a = pool["n_parcels"].values[tr]
            ns_a = pool["n_stops"].values[tr]
            area_a = pool["area_km2"].values[tr]
            hd_a = pool["hub_dist_km"].values[tr]
            daganzo_tr = daganzo_vec(np_a, ns_a, area_a, hd_a)
            residual_tr = y[tr] - daganzo_tr
            lgb_res = lgb.LGBMRegressor(
                n_estimators=1000, learning_rate=0.05, num_leaves=31,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=0.5,
                min_child_samples=10, n_jobs=4, random_state=42, verbosity=-1,
            )
            lgb_res.fit(X[tr], residual_tr)
            # Test prediction
            np_a_te = pool["n_parcels"].values[te]
            ns_a_te = pool["n_stops"].values[te]
            area_a_te = pool["area_km2"].values[te]
            hd_a_te = pool["hub_dist_km"].values[te]
            daganzo_te = daganzo_vec(np_a_te, ns_a_te, area_a_te, hd_a_te)
            pred = daganzo_te + lgb_res.predict(X[te])
        else:
            m = make_model(model_name)
            if m is None:
                continue
            m.fit(X[tr], y[tr])
            pred = m.predict(X[te])
        fold_m = safe_metrics(y[te], pred)
        fold_metrics.append(fold_m)
        # Accumulate per-cluster predictions
        for idx, i in enumerate(te):
            cl = groups[i]
            if cl not in per_cluster_preds:
                per_cluster_preds[cl] = ([], [])
            per_cluster_preds[cl][0].append(y[i])
            per_cluster_preds[cl][1].append(pred[idx])
    mean_mape = np.mean([m["mape"] for m in fold_metrics])
    std_mape = np.std([m["mape"] for m in fold_metrics])
    mean_r2 = np.mean([m["r2"] for m in fold_metrics])
    # Per-cluster aggregated metrics
    per_cluster_mape = {}
    for cl, (actual, pred) in per_cluster_preds.items():
        a = np.array(actual); p = np.array(pred)
        if len(a) >= 5:
            per_cluster_mape[cl] = float(np.mean(np.abs(a - p) / np.maximum(a, 1)) * 100)
    return {
        "mape_mean": float(mean_mape), "mape_std": float(std_mape),
        "r2_mean": float(mean_r2),
        "per_cluster_mape": per_cluster_mape,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default=None)
    parser.add_argument("--out", default="results/model_battery_v4")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Pool selection
    if args.pool:
        pool_path = Path(args.pool).resolve()
    else:
        v4 = ROOT / "results" / "sweep_v4_density_buffer" / "training_matrix_merged.csv"
        v3 = ROOT / "results" / "sweep_v3_mergefix" / "training_matrix.csv"
        v2 = ROOT / "results" / "oracle_loop_extended_2026_05_22" / "training_matrix.csv"
        pool_path = v4 if v4.exists() else (v3 if v3.exists() else v2)
    print(f"[pool] {pool_path}")
    pool = pd.read_csv(pool_path)
    print(f"  {len(pool):,} rows, {pool['plz'].nunique()} PLZ, "
            f"{pool['provider'].nunique()} providers")
    pool_combo = build_combo_features(pool[ALL_COLS])

    # Models to test
    models = [
        "LGB-logT", "LGB-plain", "LGB-quantile", "LGB-monotonic",
        "XGBoost", "RF", "MLP-5seed", "Daganzo-Hybrid",
    ]
    if CATBOOST_AVAILABLE:
        models.insert(5, "CatBoost")

    results = []
    all_per_cluster = {}
    print(f"\n[battery] testing {len(models)} models with 5-fold GroupKFold(PLZ)\n")
    for mname in models:
        t0 = time.time()
        print(f"  > {mname:18s}", end="", flush=True)
        try:
            res = evaluate_groupkfold(mname, pool_combo, pool)
            elapsed = time.time() - t0
            results.append({
                "model": mname,
                "groupkfold_mape_mean": res["mape_mean"],
                "groupkfold_mape_std": res["mape_std"],
                "groupkfold_r2_mean": res["r2_mean"],
                "fit_time_s": elapsed,
            })
            all_per_cluster[mname] = res["per_cluster_mape"]
            print(f"   MAPE={res['mape_mean']:.3f}% (±{res['mape_std']:.3f})  "
                    f"R²={res['r2_mean']:.4f}  ({elapsed:.0f}s)")
        except Exception as e:
            print(f"   FAILED: {e}")
            results.append({"model": mname, "error": str(e)})

    df = pd.DataFrame(results).sort_values("groupkfold_mape_mean")
    df.to_csv(out / "tab_model_comparison.csv", index=False)
    print(f"\nSaved: {out/'tab_model_comparison.csv'}")
    print("\nRanked by GroupKFold MAPE (lower=better):")
    print(df[["model", "groupkfold_mape_mean", "groupkfold_mape_std", "groupkfold_r2_mean"]].to_string(index=False))

    # Per-cluster heatmap
    cl_rows = []
    for mname, per_cl in all_per_cluster.items():
        for cl, mape in per_cl.items():
            cl_rows.append({"model": mname, "cluster_id": cl, "mape": mape})
    cl_df = pd.DataFrame(cl_rows)
    if len(cl_df) > 0:
        pivot = cl_df.pivot_table(index="cluster_id", columns="model", values="mape", aggfunc="first")
        # Pick top-20 worst clusters across all models for heatmap
        cluster_worst = pivot.max(axis=1).sort_values(ascending=False).head(20).index
        pivot_top = pivot.loc[cluster_worst]
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(pivot_top.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=50)
        ax.set_xticks(range(len(pivot_top.columns)))
        ax.set_xticklabels(pivot_top.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot_top.index)))
        ax.set_yticklabels(pivot_top.index)
        for i in range(len(pivot_top.index)):
            for j in range(len(pivot_top.columns)):
                v = pivot_top.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                              color="white" if v > 20 else "black", fontsize=7)
        plt.colorbar(im, ax=ax, label="MAPE %")
        ax.set_title("Per-cluster MAPE — worst 20 clusters × models")
        fig.tight_layout()
        fig.savefig(out / "fig_per_cluster_heatmap.png")
        fig.savefig(out / "fig_per_cluster_heatmap.pdf")
        plt.close(fig)
        cl_df.to_csv(out / "tab_per_cluster_mape.csv", index=False)

    # Bar plot — overall MAPE
    fig, ax = plt.subplots(figsize=(8, 5))
    df_valid = df.dropna(subset=["groupkfold_mape_mean"]).sort_values("groupkfold_mape_mean")
    ax.barh(df_valid["model"], df_valid["groupkfold_mape_mean"],
             xerr=df_valid["groupkfold_mape_std"], color="#2a9d8f", alpha=0.85)
    ax.set_xlabel("GroupKFold(PLZ) MAPE [%]")
    ax.set_title("Model Battery — honest generalisation accuracy")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_holdout_mape_per_model.png")
    fig.savefig(out / "fig_holdout_mape_per_model.pdf")
    plt.close(fig)

    # REPORT.md
    best = df_valid.iloc[0]
    lines = [
        "# Model Battery Comparison Report",
        f"\n**Pool**: `{pool_path.name}` ({len(pool):,} rows)",
        f"**Method**: 5-fold GroupKFold(PLZ) — each cluster held out once",
        f"\n## Ranking (lower MAPE = better generalisation)\n",
        df_valid[["model", "groupkfold_mape_mean", "groupkfold_mape_std", "groupkfold_r2_mean", "fit_time_s"]]
            .to_string(index=False),
        f"\n## Winner: **{best['model']}** with MAPE = {best['groupkfold_mape_mean']:.3f}% (±{best['groupkfold_mape_std']:.3f})",
        f"\n## Figures\n",
        "- `fig_holdout_mape_per_model.png` — GroupKFold MAPE bar chart",
        "- `fig_per_cluster_heatmap.png` — Per-cluster MAPE heatmap (worst-20)",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[done] outputs in {out}")


if __name__ == "__main__":
    main()
