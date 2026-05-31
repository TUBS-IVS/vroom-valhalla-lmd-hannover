"""Comprehensive model-evaluation suite for 04_model, all on the training pool
with GroupKFold (group=PLZ) — VROOM ground-truth, no extra routing needed.

Generates:
  M2  predicted-vs-actual scatter (4 top models)
  M3  MAPE vs R2 model landscape
  M4  per-fold CV box plots (stability)
  M6  residual distribution per model
  M7  per-provider MAPE (Hybrid)
  M8  per-cost-quintile MAPE (Hybrid calibration)
  M9  Daganzo physics decomposition (alpha=1.343)
  M10 learning curve (MAPE vs training fraction)
"""
from __future__ import annotations
import math, pickle, sys, warnings
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
from sklearn.model_selection import GroupKFold
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from batch_delivery.config.constants import (
    VEHICLE_CAPACITY, BHH_CONSTANT, FIXED_COST_EUR, COST_PER_KM_EUR)
from batch_delivery.features import ALL_COLS
from batch_delivery.surrogate import build_combo_features

OUT = ROOT / "results" / "paper_final_2026_05_28" / "04_model"
rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
PROV_COLOR = {"Amazon": "#003049", "DHL": "#d62828", "DPD": "#f77f00",
              "FedEx": "#5a189a", "GLS": "#2a9d8f", "Hermes": "#9d4edd", "UPS": "#7d5a50"}
ALPHA = 1.343


def daganzo_vec(np_a, ns_a, area_a, hd_a):
    out = np.zeros(len(np_a), dtype=np.float64)
    for i in range(len(np_a)):
        n = int(np_a[i])
        if n <= 0:
            continue
        nr = math.ceil(n / VEHICLE_CAPACITY)
        spr = max(1.0, int(max(1, ns_a[i])) / nr)
        local = BHH_CONSTANT * math.sqrt(spr * max(0.01, area_a[i]))
        out[i] = nr * (FIXED_COST_EUR + (2 * hd_a[i] + local) * COST_PER_KM_EUR)
    return out


def make_lgb(**kw):
    p = dict(objective="regression", n_estimators=500, learning_rate=0.05,
             num_leaves=31, min_data_in_leaf=10, verbose=-1)
    p.update(kw)
    return lgb.LGBMRegressor(**p)


def main():
    pool = pd.read_csv(ROOT / "results/sweep_v3_mergefix/training_matrix.csv")
    y = pool.actual_cost_eur.values.astype(float)
    groups = pool.plz.astype(str).values
    combo = build_combo_features(pool)
    exclude = {"actual_cost_eur","actual_distance_km","actual_duration_h","actual_n_routes",
               "n_vehicles_planned","solve_time_s","vroom_status","is_baseline","provider",
               "plz","agg_k","base_day","scale","p_keep","noise_sigma","b2c_scale","b2b_scale","seed"}
    combo_cols = [c for c in combo.columns if c not in exclude
                  and combo[c].dtype in (np.float64, np.float32, np.int64, np.int32)]
    Xc = combo[combo_cols].astype(float).values
    dag = daganzo_vec(pool.n_parcels.values, pool.n_stops.values,
                      pool.area_km2.values, pool.hub_dist_km.values)
    print(f"  pool {len(pool)} rows, {pool.plz.nunique()} PLZ groups")

    # ── Collect OOF predictions per model (GroupKFold)
    gkf = GroupKFold(n_splits=5)
    models = {}

    def cv_predict(name, fit_fn, with_dag=False):
        oof = np.zeros(len(y)); fold_mape = []
        for fold, (tr, te) in enumerate(gkf.split(Xc, y, groups)):
            if with_dag:
                resid = y[tr] - ALPHA * dag[tr]
                m = make_lgb(); m.fit(Xc[tr], resid)
                pred = ALPHA * dag[te] + m.predict(Xc[te])
            else:
                m = fit_fn(Xc[tr], y[tr])
                pred = m.predict(Xc[te])
            oof[te] = pred
            fold_mape.append(np.mean(np.abs(pred - y[te]) / np.maximum(1e-6, y[te])) * 100)
        models[name] = {"oof": oof, "fold_mape": fold_mape}
        print(f"    {name}: MAPE {np.mean(fold_mape):.2f}% (±{np.std(fold_mape):.2f})")

    print("  Running GroupKFold for top models...")
    cv_predict("Daganzo-LGB-Hybrid", None, with_dag=True)
    cv_predict("XGBoost", lambda X, yy: xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6, verbosity=0,
        tree_method="hist", n_jobs=-1).fit(X, yy))
    cv_predict("LGB-logT", lambda X, yy: TransformedTargetRegressor(
        regressor=make_lgb(), func=np.log1p, inverse_func=np.expm1).fit(X, yy))
    cv_predict("Random Forest", lambda X, yy: RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=4, n_jobs=-1,
        random_state=42).fit(X, yy))
    cv_predict("Ridge", lambda X, yy: Ridge(alpha=1.0).fit(X, yy))
    # Pure Daganzo (no CV needed, deterministic)
    models["Pure Daganzo (α=1.343)"] = {"oof": ALPHA * dag,
        "fold_mape": [np.mean(np.abs(ALPHA*dag - y)/np.maximum(1e-6,y))*100]}

    prov = pool.provider.values

    # ── M2: predicted-vs-actual scatter (4 models)
    top4 = ["Daganzo-LGB-Hybrid", "Pure Daganzo (α=1.343)", "XGBoost", "LGB-logT"]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, name in zip(axes, top4):
        pred = models[name]["oof"]
        for p in PROV_COLOR:
            mask = prov == p
            ax.scatter(y[mask], pred[mask], s=5, alpha=0.4, color=PROV_COLOR[p], edgecolor="none")
        lim = [y.min(), y.max()]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        mape = np.mean(np.abs(pred - y)/np.maximum(1e-6,y))*100
        bias = np.mean((pred - y)/np.maximum(1e-6,y))*100
        r2 = 1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum()
        ax.set_title(f"{name}\nMAPE {mape:.2f}% · bias {bias:+.1f}% · R² {r2:.3f}", fontsize=10)
        ax.set_xlabel("VROOM actual [EUR]")
        if name == top4[0]:
            ax.set_ylabel("OOF predicted [EUR]")
    fig.suptitle("Out-of-fold predicted vs VROOM actual (GroupKFold, group=PLZ)", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "fig_M2_pred_vs_actual.png"); fig.savefig(OUT / "fig_M2_pred_vs_actual.pdf")
    plt.close(fig); print("  ✓ M2: pred_vs_actual")

    # ── M3: MAPE vs R2 landscape
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, m in models.items():
        pred = m["oof"]
        mape = np.mean(np.abs(pred-y)/np.maximum(1e-6,y))*100
        r2 = 1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum()
        is_hybrid = "Hybrid" in name
        ax.scatter(mape, r2, s=180 if is_hybrid else 90,
                    c="#ee9b00" if is_hybrid else ("#c1121f" if "Daganzo" in name else "#1f4f8f"),
                    edgecolor="black", zorder=5)
        ax.annotate(name, (mape, r2), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("MAPE [%] (lower better)"); ax.set_ylabel("R² (higher better)")
    ax.set_title("Model landscape — MAPE vs R² (GroupKFold OOF)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_M3_mape_vs_r2.png"); fig.savefig(OUT / "fig_M3_mape_vs_r2.pdf")
    plt.close(fig); print("  ✓ M3: mape_vs_r2")

    # ── M4: per-fold CV boxplot
    fig, ax = plt.subplots(figsize=(11, 6))
    cv_models = [n for n in models if len(models[n]["fold_mape"]) > 1]
    cv_models = sorted(cv_models, key=lambda n: np.mean(models[n]["fold_mape"]))
    data = [models[n]["fold_mape"] for n in cv_models]
    bp = ax.boxplot(data, labels=cv_models, patch_artist=True, showmeans=True)
    for patch, n in zip(bp["boxes"], cv_models):
        patch.set_facecolor("#ee9b00" if "Hybrid" in n else "#1f4f8f"); patch.set_alpha(0.7)
    ax.set_ylabel("MAPE [%] per fold")
    ax.set_title("Cross-validation stability — MAPE per fold (5-fold GroupKFold)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_M4_cv_stability.png"); fig.savefig(OUT / "fig_M4_cv_stability.pdf")
    plt.close(fig); print("  ✓ M4: cv_stability")

    # ── M6: residual distribution per model
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in top4:
        pred = models[name]["oof"]
        relerr = 100 * (pred - y) / np.maximum(1e-6, y)
        ax.hist(relerr, bins=60, histtype="step", linewidth=1.8, label=name,
                range=(-40, 40))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Relative error [%]  (pred − actual)/actual")
    ax.set_ylabel("Samples")
    ax.set_title("Residual distributions (OOF)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_M6_residual_dist.png"); fig.savefig(OUT / "fig_M6_residual_dist.pdf")
    plt.close(fig); print("  ✓ M6: residual_dist")

    # ── M7: per-provider MAPE (Hybrid)
    hyb = models["Daganzo-LGB-Hybrid"]["oof"]
    rows = []
    for p in PROV_COLOR:
        mask = prov == p
        if mask.sum() == 0: continue
        mape = np.mean(np.abs(hyb[mask]-y[mask])/np.maximum(1e-6,y[mask]))*100
        rows.append({"provider": p, "MAPE_pct": mape, "n": int(mask.sum())})
    pdf = pd.DataFrame(rows).sort_values("MAPE_pct")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(pdf.provider, pdf.MAPE_pct, color=[PROV_COLOR[p] for p in pdf.provider], edgecolor="black")
    for i, r in pdf.reset_index().iterrows():
        ax.text(i, r.MAPE_pct+0.05, f"{r.MAPE_pct:.2f}%", ha="center", fontsize=9)
    ax.set_ylabel("MAPE [%]"); ax.set_title("Daganzo-LGB-Hybrid MAPE per LSP (OOF)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_M7_per_provider_mape.png"); fig.savefig(OUT / "fig_M7_per_provider_mape.pdf")
    plt.close(fig); print("  ✓ M7: per_provider_mape")

    # ── M8: per-cost-quintile MAPE (calibration)
    pool2 = pool.copy(); pool2["pred"] = hyb
    pool2["q"] = pd.qcut(y, 5, labels=["Q1 (low)","Q2","Q3","Q4","Q5 (high)"])
    rows = []
    for q, g in pool2.groupby("q"):
        mape = np.mean(np.abs(g.pred-g.actual_cost_eur)/np.maximum(1e-6,g.actual_cost_eur))*100
        bias = np.mean((g.pred-g.actual_cost_eur)/np.maximum(1e-6,g.actual_cost_eur))*100
        rows.append({"quintile": q, "MAPE": mape, "bias": bias})
    qdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(qdf)); w = 0.38
    ax.bar(x-w/2, qdf.MAPE, w, label="MAPE", color="#1f4f8f", edgecolor="black")
    ax.bar(x+w/2, qdf.bias, w, label="bias", color="#e76f51", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(qdf.quintile)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("%"); ax.set_title("Hybrid calibration across cost quintiles (OOF)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_M8_cost_quintile.png"); fig.savefig(OUT / "fig_M8_cost_quintile.pdf")
    plt.close(fig); print("  ✓ M8: cost_quintile")

    # ── M9: Daganzo physics decomposition (route/km/cost ratio)
    n_routes_pred = np.ceil(pool.n_parcels.values / VEHICLE_CAPACITY)
    fig, ax = plt.subplots(figsize=(9, 6))
    ratio = (ALPHA * dag) / np.maximum(1.0, y)
    for p in PROV_COLOR:
        mask = prov == p
        ax.scatter(pool.n_parcels.values[mask], ratio[mask], s=8, alpha=0.4,
                   color=PROV_COLOR[p], label=p, edgecolor="none")
    ax.axhline(1.0, color="black", ls="--", lw=0.9)
    ax.set_xscale("log"); ax.set_xlabel("Parcels per delivery day")
    ax.set_ylabel("Pure Daganzo (α=1.343) / VROOM cost")
    ax.set_title(f"Daganzo physics calibration — cost ratio (median {np.median(ratio):.3f})")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3, which="both"); ax.set_ylim(0.5, 1.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_M9_daganzo_calibration.png"); fig.savefig(OUT / "fig_M9_daganzo_calibration.pdf")
    plt.close(fig); print("  ✓ M9: daganzo_calibration")

    # ── M10: learning curve (MAPE vs training fraction)
    fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
    rng = np.random.default_rng(42)
    lc = []
    for fr in fracs:
        fold_m = []
        for tr, te in gkf.split(Xc, y, groups):
            n_use = int(len(tr) * fr)
            tr_sub = rng.choice(tr, n_use, replace=False)
            resid = y[tr_sub] - ALPHA * dag[tr_sub]
            m = make_lgb(); m.fit(Xc[tr_sub], resid)
            pred = ALPHA * dag[te] + m.predict(Xc[te])
            fold_m.append(np.mean(np.abs(pred-y[te])/np.maximum(1e-6,y[te]))*100)
        lc.append({"frac": fr, "mape": np.mean(fold_m), "std": np.std(fold_m)})
    lcdf = pd.DataFrame(lc)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(lcdf["frac"]*100, lcdf["mape"], yerr=lcdf["std"], marker="o", capsize=4,
                color="#1f4f8f", linewidth=2)
    ax.set_xlabel("Training data used [%]"); ax.set_ylabel("Hybrid MAPE [%] (OOF)")
    ax.set_title("Learning curve — Hybrid MAPE vs training size")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_M10_learning_curve.png"); fig.savefig(OUT / "fig_M10_learning_curve.pdf")
    plt.close(fig); print("  ✓ M10: learning_curve")

    # Save consolidated metrics
    rows = []
    for name, m in models.items():
        pred = m["oof"]
        rows.append({"model": name,
            "MAPE_pct": np.mean(np.abs(pred-y)/np.maximum(1e-6,y))*100,
            "bias_pct": np.mean((pred-y)/np.maximum(1e-6,y))*100,
            "R2": 1-((y-pred)**2).sum()/((y-y.mean())**2).sum(),
            "fold_std": np.std(m["fold_mape"])})
    pd.DataFrame(rows).sort_values("MAPE_pct").to_csv(OUT / "tab_model_eval_oof.csv", index=False)
    print(f"\n  ✓ tab_model_eval_oof.csv")
    print(f"\nDone. 04_model now has comprehensive evaluation.")


if __name__ == "__main__":
    main()
