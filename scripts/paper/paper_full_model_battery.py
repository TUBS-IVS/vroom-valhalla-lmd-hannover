"""Train every reasonable surrogate variant on the v2-aug pool, predict the
312 VROOM-validated cells, and compare.  Comprehensive sweep so the paper
has an unambiguous "why Daganzo-Hybrid" answer.

Models compared
---------------
  Pure Daganzo          (physics-only formula)
  Daganzo-LGB-Hybrid    (production, already pickled)
  LGB-logT v2 / v3      (pickled)
  MLP-5seed             (existing pickled ensemble)
  LGB-raw               (no log transform)
  LGB-huber             (Huber loss)
  LGB-quantile50        (median, log1p)
  LGB-tweedie           (Tweedie p=1.5)
  LGB-plain             (default LightGBM)
  XGBoost
  Random Forest
  Decision Tree (max_depth=8)
  Linear Regression
  Ridge
  Lasso

Inputs
------
  results/oracle_loop_extended_2026_05_22/training_matrix.csv      (12 119 rows)
  results/oracle_loop_extended_2026_05_22/daganzo_hybrid_v2aug.pkl
  ...

Outputs (results/overnight_2026_05_27/diagnosis_v2/full_battery/)
"""
from __future__ import annotations
import math
import pickle
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from batch_delivery.config.constants import (  # noqa: E402
    VEHICLE_CAPACITY, BHH_CONSTANT, FIXED_COST_EUR, COST_PER_KM_EUR,
)
from batch_delivery.features import ALL_COLS  # noqa: E402
from batch_delivery.surrogate import build_combo_features  # noqa: E402
from batch_delivery.optimization.core import build_cost_matrices_ml  # noqa: E402

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

BASE = ROOT / "results" / "overnight_2026_05_27"
OUT = BASE / "diagnosis_v2" / "full_battery"
OUT.mkdir(parents=True, exist_ok=True)
N_DAYS = 6
MAX_HOLD = 3
PROVIDERS = ["DHL", "Amazon", "DPD", "FedEx", "GLS", "Hermes", "UPS"]


def enumerate_schedules():
    out = []
    for k in range(1, N_DAYS + 1):
        for combo in combinations(range(N_DAYS), k):
            days = sorted(combo)
            ok = True
            for i in range(len(days)):
                gap = (days[(i + 1) % len(days)] - days[i]) % N_DAYS
                if gap == 0:
                    gap = N_DAYS
                if gap > MAX_HOLD:
                    ok = False
                    break
            if ok:
                out.append(frozenset(days))
    return out


def daganzo_cost(n_parcels, n_stops, area_km2, hub_dist_km):
    if n_parcels <= 0 or n_stops <= 0:
        return 0.0
    n_routes = math.ceil(n_parcels / VEHICLE_CAPACITY)
    spr = max(1.0, n_stops / n_routes)
    local_dist = BHH_CONSTANT * math.sqrt(spr * max(0.01, area_km2))
    return n_routes * (FIXED_COST_EUR + (2 * hub_dist_km + local_dist) * COST_PER_KM_EUR)


class PureDaganzo:
    def __init__(self):
        self.combo_cols = []
        self.kind = "PureDaganzo"

    def predict(self, df):
        return np.array([
            daganzo_cost(
                float(df["n_parcels"].iloc[i]), float(df["n_stops"].iloc[i]),
                float(df["area_km2"].iloc[i]), float(df["hub_dist_km"].iloc[i]),
            ) for i in range(len(df))
        ])

    def predict_single(self, base25):
        d = pd.DataFrame(base25.reshape(1, -1), columns=ALL_COLS)
        return float(self.predict(d)[0])


class SklearnWrapper:
    """Wrap any scikit-style predictor so it matches our build_cost_matrices_ml
    interface (predict on the 25 ALL_COLS or 44 combo features as needed).
    """
    def __init__(self, model, combo_cols, use_combo=True, scale=None):
        self.model = model
        self.combo_cols = combo_cols
        self.use_combo = use_combo
        self.scale = scale
        self.kind = "Sklearn"

    def predict(self, df_feats):
        if self.use_combo:
            X = build_combo_features(df_feats)[self.combo_cols].values
        else:
            X = df_feats[ALL_COLS].values
        if self.scale is not None:
            X = self.scale.transform(X)
        return np.asarray(self.model.predict(X), dtype=np.float64)

    def predict_single(self, base25):
        d = pd.DataFrame(base25.reshape(1, -1), columns=ALL_COLS)
        return float(self.predict(d)[0])


def load_pickled():
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_daganzo_hybrid import DaganzoLGBHybrid, _LGBIdentityWrap  # noqa
    import __main__
    __main__._LGBIdentityWrap = _LGBIdentityWrap

    models = {}
    with open(ROOT / "results/oracle_loop_extended_2026_05_22/daganzo_hybrid_v2aug.pkl", "rb") as f:
        d = pickle.load(f)
    hybrid = DaganzoLGBHybrid(model=d["model"], combo_cols=d["combo_cols"], alpha=d["alpha"])
    models["Daganzo-LGB-Hybrid"] = hybrid
    models["Pure Daganzo"] = PureDaganzo()

    from batch_delivery.surrogate.lgb_adapter import LGBLogTSurrogate
    p = ROOT / "results/oracle_loop_extended_2026_05_22/production_lgb_logT_v2.pkl"
    if p.exists():
        try:
            models["LGB-logT v2"] = LGBLogTSurrogate.load(p)
        except Exception as e:
            print(f"  WARN LGB v2: {e}")
    p = ROOT / "results/sweep_v3_mergefix/production_lgb_logT_v3.pkl"
    if p.exists():
        try:
            models["LGB-logT v3"] = LGBLogTSurrogate.load(p)
        except Exception as e:
            print(f"  WARN LGB v3: {e}")
    return models, hybrid


def train_extras(pool, combo_cols):
    """Train all the additional model variants on the same pool."""
    extras = {}
    X = build_combo_features(pool)[combo_cols].values
    y = pool["actual_cost_eur"].values
    print(f"  Training pool: {X.shape[0]} rows × {X.shape[1]} combo features")

    def reg_log(y): return np.log1p(y)
    def reg_inv(y): return np.expm1(y)

    # ── LightGBM variants
    print("  Training LGB-raw ...")
    t0 = time.time()
    lgb_raw = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=-1, num_leaves=63,
        min_child_samples=15, reg_lambda=0.5, verbose=-1, random_state=42)
    lgb_raw.fit(X, y)
    extras["LGB-raw"] = SklearnWrapper(lgb_raw, combo_cols)
    print(f"    {time.time()-t0:.1f}s")

    print("  Training LGB-huber ...")
    t0 = time.time()
    lgb_huber = TransformedTargetRegressor(
        regressor=lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=-1, num_leaves=63,
            min_child_samples=15, reg_lambda=0.5, objective="huber",
            alpha=0.9, verbose=-1, random_state=42),
        func=reg_log, inverse_func=reg_inv)
    lgb_huber.fit(X, y)
    extras["LGB-huber"] = SklearnWrapper(lgb_huber, combo_cols)
    print(f"    {time.time()-t0:.1f}s")

    print("  Training LGB-quantile50 ...")
    t0 = time.time()
    lgb_q50 = TransformedTargetRegressor(
        regressor=lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=-1, num_leaves=63,
            min_child_samples=15, reg_lambda=0.5, objective="quantile",
            alpha=0.5, verbose=-1, random_state=42),
        func=reg_log, inverse_func=reg_inv)
    lgb_q50.fit(X, y)
    extras["LGB-quantile50"] = SklearnWrapper(lgb_q50, combo_cols)
    print(f"    {time.time()-t0:.1f}s")

    print("  Training LGB-tweedie ...")
    t0 = time.time()
    try:
        lgb_tw = lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=-1, num_leaves=63,
            min_child_samples=15, reg_lambda=0.5,
            objective="tweedie", tweedie_variance_power=1.5,
            verbose=-1, random_state=42)
        lgb_tw.fit(X, y)
        extras["LGB-tweedie"] = SklearnWrapper(lgb_tw, combo_cols)
        print(f"    {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"    WARN tweedie failed: {e}")

    print("  Training XGBoost ...")
    t0 = time.time()
    xgb_r = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        reg_lambda=0.5, tree_method="hist", random_state=42)
    xgb_r.fit(X, y)
    extras["XGBoost"] = SklearnWrapper(xgb_r, combo_cols)
    print(f"    {time.time()-t0:.1f}s")

    print("  Training Random Forest ...")
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=400, max_depth=20,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    extras["Random Forest"] = SklearnWrapper(rf, combo_cols)
    print(f"    {time.time()-t0:.1f}s")

    print("  Training Decision Tree ...")
    t0 = time.time()
    dt = DecisionTreeRegressor(max_depth=10, min_samples_leaf=8, random_state=42)
    dt.fit(X, y)
    extras["Decision Tree"] = SklearnWrapper(dt, combo_cols)
    print(f"    {time.time()-t0:.1f}s")

    # ── Linear family
    print("  Training Linear / Ridge / Lasso ...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    for name, mdl in [("Linear", LinearRegression()),
                       ("Ridge", Ridge(alpha=10.0)),
                       ("Lasso", Lasso(alpha=10.0, max_iter=5000))]:
        t0 = time.time()
        mdl.fit(Xs, y)
        extras[name] = SklearnWrapper(mdl, combo_cols, scale=scaler)
        print(f"    {name}: {time.time()-t0:.1f}s")

    # ── MLP-5seed ensemble
    print("  Training MLP-5seed ensemble ...")
    t0 = time.time()
    mlps = []
    for seed in range(5):
        m = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(96, 64, 32), activation="relu",
                solver="adam", alpha=1e-3, max_iter=300,
                early_stopping=True, validation_fraction=0.1,
                random_state=seed + 10)),
        ])
        m.fit(X, np.log1p(y))
        mlps.append(m)
    class MLPEnsemble:
        def __init__(self, models, combo_cols):
            self.models = models
            self.combo_cols = combo_cols
            self.kind = "MLPEnsemble"
        def predict(self, df_feats):
            X = build_combo_features(df_feats)[self.combo_cols].values
            preds = np.mean([m.predict(X) for m in self.models], axis=0)
            return np.expm1(preds)
        def predict_single(self, base25):
            d = pd.DataFrame(base25.reshape(1, -1), columns=ALL_COLS)
            return float(self.predict(d)[0])
    extras["MLP-5seed (new)"] = MLPEnsemble(mlps, combo_cols)
    print(f"    {time.time()-t0:.1f}s")

    return extras


def build_ml_prep(provider_data):
    from batch_delivery.config.constants import provider_to_demand_prefix
    ml_prep = {}
    for prov in PROVIDERS:
        pdata = provider_data.get(prov)
        if pdata is None:
            continue
        df_assign = pdata["df_assignments"]
        hub_coords_by_plz = {row["plz"]: (row["hub_lon"], row["hub_lat"])
                              for _, row in df_assign.iterrows()}
        prefix = provider_to_demand_prefix(prov)
        col_total = f"{prefix}_total"
        plz_day_coords = {}
        for pc in pdata["all_plz_set"]:
            plz_day_coords[pc] = {}
            for d in range(N_DAYS):
                gdf_d = pdata["daily_gdfs_wgs"].get(d)
                if gdf_d is None:
                    continue
                pts = gdf_d[gdf_d["plz"] == pc]
                if len(pts) == 0:
                    continue
                lons = pts["lon"].values.astype(np.float64)
                lats = pts["lat"].values.astype(np.float64)
                psd = (pts[col_total].values.astype(np.float64)
                       if col_total in pts.columns else np.ones(len(pts)))
                plz_day_coords[pc][d] = (lons, lats, psd)
        ml_prep[prov] = {"plz_day_coords": plz_day_coords,
                          "hub_coords_by_plz": hub_coords_by_plz}
    return ml_prep


def diag(actual, predicted):
    err = predicted - actual
    mask = ~np.isnan(predicted)
    if mask.sum() == 0:
        return {"n": 0, "MAPE_pct": np.nan, "bias_pct": np.nan, "R2": np.nan}
    a, p = actual[mask], predicted[mask]
    e = p - a
    return {
        "n": int(mask.sum()),
        "MAPE_pct": float(np.mean(np.abs(e) / np.maximum(1e-6, a)) * 100),
        "bias_pct": float(np.mean(e / np.maximum(1e-6, a)) * 100),
        "R2": float(1 - (e ** 2).sum() / max(1, ((a - a.mean()) ** 2).sum())),
    }


def main():
    print("=" * 72)
    print("Full model battery on 312 optimized VROOM-validated cells")
    print("=" * 72)

    # ── Load training pool + fit "extras"
    print("\nLoading v2-aug training pool ...")
    pool_pkl = ROOT / "results/oracle_loop_extended_2026_05_22/v2_aug_pool.csv"
    if pool_pkl.exists():
        pool = pd.read_csv(pool_pkl)
    else:
        # Build from training_matrix
        pool = pd.read_csv(ROOT / "results/oracle_loop_extended_2026_05_22/training_matrix.csv")
        pool = pool[pool.vroom_status == "OK"].copy()
    pool = pool.dropna(subset=["actual_cost_eur"])
    # Fill any missing ALL_COLS
    for c in ALL_COLS:
        if c not in pool.columns:
            pool[c] = 0.0
    print(f"  pool rows: {len(pool)}")

    print("\nLoading already-pickled models ...")
    pickled, hybrid_template = load_pickled()
    combo_cols = hybrid_template.combo_cols
    print(f"  pickled: {list(pickled.keys())}")

    print(f"\nTraining additional model variants on the v2-aug pool ...")
    t_train = time.time()
    extras = train_extras(pool, combo_cols)
    print(f"  total training time: {time.time()-t_train:.0f}s")

    all_models = {**pickled, **extras}
    print(f"\nTotal models in comparison: {len(all_models)}")

    # ── Load VROOM truth
    chosen = pd.read_csv(BASE / "tab_chosen_schedules.csv")
    chosen = chosen[(np.isclose(chosen.penalty, 0.5)) &
                    (np.isclose(chosen.share_willing, 1.0))].copy()
    chosen["plz"] = chosen.plz.astype(str)
    chosen_idx_per_pp = {(r.provider, r.plz): int(r.schedule_idx)
                         for _, r in chosen.iterrows()}
    vroom = pd.read_csv(BASE / "tab_vroom_validation.csv")
    vroom = vroom[vroom.vroom_cost_eur > 0].copy()
    vroom["plz"] = vroom.plz.astype(str)
    v_agg = (vroom.groupby(["provider", "plz"], as_index=False).agg(
        vroom_cost=("vroom_cost_eur", "sum")))
    print(f"  VROOM cells: {len(v_agg)}")

    # ── Predict per model via build_cost_matrices_ml
    print("\nPredicting via build_cost_matrices_ml ...")
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    ml_prep = build_ml_prep(chk["provider_data"])
    schedules = enumerate_schedules()

    pred_dict = {name: {} for name in all_models}
    for name, m in all_models.items():
        t0 = time.time()
        for prov in PROVIDERS:
            if prov not in chk4["optimization_data"]:
                continue
            odata = chk4["optimization_data"][prov]
            prep = ml_prep[prov]
            plz_keys = odata["plz_keys"]
            try:
                mat = build_cost_matrices_ml(
                    plz_keys, odata["plz_data"], schedules, m, prov,
                    prep["plz_day_coords"], prep["hub_coords_by_plz"],
                    fast_share_b2c=0.0, fast_share_b2b=0.0,
                )
            except Exception as e:
                print(f"  WARN {name}/{prov}: {e}")
                continue
            cost_3d = mat["cost_3d"]
            sched_active = mat["sched_active"]
            dd_cost = (cost_3d * sched_active[None, :, :]).sum(axis=2)
            for pi, pc in enumerate(plz_keys):
                key = (prov, str(pc))
                if key not in chosen_idx_per_pp:
                    continue
                si = chosen_idx_per_pp[key]
                pred_dict[name][key] = float(dd_cost[pi, si])
        print(f"  {name:25s} {len(pred_dict[name])} cells  t={time.time()-t0:.0f}s")

    # Build comparison df
    df = v_agg.copy()
    for name in all_models:
        df[name] = df.apply(
            lambda r: pred_dict[name].get((r.provider, r.plz), np.nan), axis=1
        )

    diag_rows = []
    for name in all_models:
        if df[name].notna().sum() < 10:
            continue
        d = diag(df.vroom_cost.values, df[name].values)
        d["model"] = name
        diag_rows.append(d)
    diag_df = pd.DataFrame(diag_rows)[["model", "n", "MAPE_pct", "bias_pct", "R2"]] \
              .sort_values("MAPE_pct")
    diag_df.to_csv(OUT / "tab_full_model_comparison.csv", index=False)
    print("\nFinal comparison:")
    print(diag_df.round(2).to_string(index=False))

    # ── Plot: MAPE bars
    print("\nPlotting ...")
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = ["#e76f51" if m == "Pure Daganzo"
               else "#1f4f8f" if m == "Daganzo-LGB-Hybrid"
               else "#2a9d8f" if "LGB" in m
               else "#9d4edd" if "MLP" in m
               else "#f4a261"
               for m in diag_df.model]
    ax.barh(diag_df.model, diag_df.MAPE_pct, color=palette, edgecolor="black")
    for i, (m, mape, bias) in enumerate(zip(diag_df.model, diag_df.MAPE_pct,
                                              diag_df.bias_pct)):
        ax.text(mape + 0.3, i, f"{mape:.2f}%  (bias {bias:+.1f}%)",
                va="center", fontsize=9)
    ax.set_xlabel("MAPE on 312 VROOM-validated cells [%]")
    ax.set_title("Full model battery: comparison on optimized schedules\n"
                  "(sorted by MAPE; Daganzo physics blue, LGB family teal, MLP purple, classical orange)")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "fig_full_battery_mape.png")
    fig.savefig(OUT / "fig_full_battery_mape.pdf")
    plt.close(fig)

    # MAPE vs R² scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, r in diag_df.iterrows():
        marker = "*" if r.model == "Daganzo-LGB-Hybrid" else "o"
        size = 200 if r.model == "Daganzo-LGB-Hybrid" else 80
        color = ("#e76f51" if r.model == "Pure Daganzo"
                 else "#1f4f8f" if r.model == "Daganzo-LGB-Hybrid"
                 else "#2a9d8f" if "LGB" in r.model
                 else "#9d4edd" if "MLP" in r.model
                 else "#f4a261")
        ax.scatter(r.MAPE_pct, r.R2, s=size, color=color, edgecolor="black",
                    marker=marker)
        ax.annotate(r.model, (r.MAPE_pct, r.R2),
                     xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("MAPE [%]")
    ax.set_ylabel("R$^2$")
    ax.set_title("Model trade-off: accuracy vs ranking quality")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mape_vs_r2.png")
    fig.savefig(OUT / "fig_mape_vs_r2.pdf")
    plt.close(fig)

    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
