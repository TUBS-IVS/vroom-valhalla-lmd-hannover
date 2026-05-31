"""Deep diagnostic analysis: why is the MLP-ensemble 7x worse than LightGBM-logT?

Three diagnostic threads:
  1.  2x2 design  --  (raw target / log target) x (MLP-ensemble / LightGBM)
      Isolates "log-target effect" vs "model-family effect".
  2.  Residual decomposition  --  per cost-bin / LSP / extremity on the
      holdout; head-to-head between MLP-prod and LightGBM-logT.
  3.  Feature importance  --  permutation importance on the holdout for
      every top model + native gain importance for LightGBM/XGBoost/RF.

RAM-safe: n_jobs=2, single MLP-ensemble retrain (5 seeds, 256-128-64-32),
no MLPs per CV-fold.  Output to results/paper_figures/ml_surrogate_v2/deep/.
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

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_helpers import safe_metrics, apply_style, PALETTE, PROVIDERS  # noqa: E402
from batch_delivery.features import ALL_COLS  # noqa: E402
from batch_delivery.surrogate import MLCostPredictor, build_combo_features  # noqa: E402

apply_style()

RUN = ROOT / "results" / "oracle_loop_extended_2026_05_22"
OUT = ROOT / "results" / "paper_figures" / "ml_surrogate_v2" / "deep"
OUT.mkdir(parents=True, exist_ok=True)

N_JOBS = 2
SEEDS = [42, 123, 456, 789, 2024]   # production-matching seeds


# ── data ────────────────────────────────────────────────────────────────────
pool = pd.read_csv(RUN / "training_matrix.csv")
hold = pd.read_csv(RUN / "holdout_extreme.csv")
print(f"pool    : {len(pool):>6,} rows")
print(f"holdout : {len(hold):>6,} rows")

pool_combo = build_combo_features(pool[ALL_COLS])
hold_combo = build_combo_features(hold[ALL_COLS])
COMBO_COLS = pool_combo.columns.tolist()

X25_tr, X44_tr = pool[ALL_COLS].values, pool_combo.values
y_tr = pool["actual_cost_eur"].values
X25_te, X44_te = hold[ALL_COLS].values, hold_combo.values
y_te = hold["actual_cost_eur"].values


# ── (1) 2x2 design: (raw / logT) x (MLP-ensemble / LGB) ─────────────────────
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


def mlp_pipe(seed: int) -> Pipeline:
    return Pipeline([
        ("sc", StandardScaler()),
        ("m", MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32),
                            alpha=1e-4, max_iter=400, early_stopping=True,
                            random_state=seed)),
    ])

def train_mlp_ensemble(Xtr, ytr, log_target: bool):
    pipes = []
    for s in SEEDS:
        est = mlp_pipe(s)
        if log_target:
            est = TransformedTargetRegressor(regressor=est,
                                              func=np.log1p, inverse_func=np.expm1)
        est.fit(Xtr, ytr)
        pipes.append(est)
    return pipes

def predict_ensemble(pipes, X):
    return np.mean([p.predict(X) for p in pipes], axis=0)


print("\n=== (1) 2x2 design ===")
t0 = time.time()

# Load production MLP-ensemble (already trained, raw target)
prod = MLCostPredictor.load(RUN / "ml_cost_predictor.pkl")
yp_mlp_raw_te = prod.predict(hold)

# Train fresh MLP-ensemble with log-target on combo-44
print("  training MLP-ensemble (logT, 5 seeds) ...", flush=True)
mlp_logT_pipes = train_mlp_ensemble(X44_tr, y_tr, log_target=True)
yp_mlp_logT_te = predict_ensemble(mlp_logT_pipes, X44_te)
print(f"  done in {time.time()-t0:.1f}s")

# LightGBM raw + logT on combo-44 (same hyperparams as benchmark v3)
LGB_KWARGS = dict(n_estimators=800, learning_rate=0.04, num_leaves=63,
                   max_depth=-1, subsample=0.85, colsample_bytree=0.85,
                   reg_lambda=0.5, min_child_samples=10,
                   n_jobs=N_JOBS, random_state=42, verbosity=-1)

lgb_raw = lgb.LGBMRegressor(**LGB_KWARGS).fit(X44_tr, y_tr)
yp_lgb_raw_te = np.maximum(0.0, lgb_raw.predict(X44_te))

lgb_logT = TransformedTargetRegressor(regressor=lgb.LGBMRegressor(**LGB_KWARGS),
                                        func=np.log1p, inverse_func=np.expm1)
lgb_logT.fit(X44_tr, y_tr)
yp_lgb_logT_te = np.maximum(0.0, lgb_logT.predict(X44_te))

cell_rows = []
for label, yp in [("MLP-ensemble raw  (production iter17)", yp_mlp_raw_te),
                   ("MLP-ensemble logT (5 seeds, retrained)", yp_mlp_logT_te),
                   ("LightGBM raw",                          yp_lgb_raw_te),
                   ("LightGBM logT",                         yp_lgb_logT_te)]:
    m = safe_metrics(y_te, yp)
    cell_rows.append({"model": label, **m})
    print(f"  {label:<42s}  MAPE={m['mape']:6.2f}  MAE={m['mae']:6.1f}  R2={m['r2']:.4f}  bias={m['bias']:+6.2f}")

cell_df = pd.DataFrame(cell_rows)
cell_df.to_csv(OUT / "tab_2x2_logtarget_vs_modelfamily.csv", index=False)


# ── (2) Residual decomposition: head-to-head MLP-prod vs LGB-logT ───────────
print("\n=== (2) Residual decomposition (MLP-prod vs LightGBM-logT) ===")
hold_d = hold.copy()
hold_d["pred_mlp"]   = yp_mlp_raw_te
hold_d["pred_lgb"]   = yp_lgb_logT_te
hold_d["resid_mlp_pct"] = (yp_mlp_raw_te - y_te) / y_te * 100
hold_d["resid_lgb_pct"] = (yp_lgb_logT_te - y_te) / y_te * 100
hold_d["abs_err_mlp"] = np.abs(yp_mlp_raw_te - y_te)
hold_d["abs_err_lgb"] = np.abs(yp_lgb_logT_te - y_te)

# cost-bin breakdown
qcuts = pd.qcut(y_te, q=5, labels=["Q1 (cheapest)", "Q2", "Q3", "Q4", "Q5 (most expensive)"])
hold_d["cost_bin"] = qcuts.astype(str)

def per_group(df, col):
    rows = []
    for k, g in df.groupby(col):
        m_mlp = safe_metrics(g["actual_cost_eur"], g["pred_mlp"])
        m_lgb = safe_metrics(g["actual_cost_eur"], g["pred_lgb"])
        rows.append({"bucket": k, "n": int(len(g)),
                      "mape_mlp": m_mlp["mape"], "mape_lgb": m_lgb["mape"],
                      "delta_pp":  m_mlp["mape"] - m_lgb["mape"],
                      "bias_mlp": m_mlp["bias"], "bias_lgb": m_lgb["bias"]})
    return pd.DataFrame(rows)

by_bin = per_group(hold_d, "cost_bin")
by_lsp = per_group(hold_d, "provider")

def classify_extremity(row):
    if row.get("is_baseline", False): return "baseline"
    if ((row.get("b2c_scale", 1.0) >= 1.2) or
        (row.get("b2b_scale", 1.0) <= 0.93 or row.get("b2b_scale", 1.0) >= 1.075) or
        (row.get("noise_sigma", 0.0) >= 0.3) or
        (row.get("p_keep", 1.0) <= 0.6)):
        return "extreme"
    return "mild"

hold_d["extremity"] = hold_d.apply(classify_extremity, axis=1)
by_ext = per_group(hold_d, "extremity")

by_bin.to_csv(OUT / "tab_residual_by_costbin.csv", index=False)
by_lsp.to_csv(OUT / "tab_residual_by_lsp.csv", index=False)
by_ext.to_csv(OUT / "tab_residual_by_extremity.csv", index=False)

print("\n  per cost-quintile (MAPE for MLP vs LGB):")
print(by_bin.round(2).to_string(index=False))
print("\n  per LSP:")
print(by_lsp.round(2).to_string(index=False))
print("\n  per extremity:")
print(by_ext.round(2).to_string(index=False))


# ── (3) Feature importance ─────────────────────────────────────────────────
print("\n=== (3) Feature importance ===")
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Native gain importance: LGB, XGB, RF
rf = RandomForestRegressor(n_estimators=400, max_depth=14, n_jobs=N_JOBS, random_state=42)
rf.fit(X44_tr, y_tr)

xgbr_logT = TransformedTargetRegressor(
    regressor=xgb.XGBRegressor(n_estimators=800, learning_rate=0.04, max_depth=7,
                                subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                                min_child_weight=3, n_jobs=N_JOBS, random_state=42,
                                tree_method="hist", verbosity=0),
    func=np.log1p, inverse_func=np.expm1)
xgbr_logT.fit(X44_tr, y_tr)
yp_xgb_logT_te = xgbr_logT.predict(X44_te)

native_imp = pd.DataFrame({
    "feature":   COMBO_COLS,
    "RF":        rf.feature_importances_,
    "LightGBM":  lgb_logT.regressor_.feature_importances_ / lgb_logT.regressor_.feature_importances_.sum(),
    "XGBoost":   xgbr_logT.regressor_.feature_importances_ / xgbr_logT.regressor_.feature_importances_.sum(),
})
native_imp.to_csv(OUT / "tab_native_importance.csv", index=False)
print("\n  Top-10 native (gain-based) importance:")
print(native_imp.sort_values("LightGBM", ascending=False).head(10).round(4).to_string(index=False))

# Permutation importance (model-agnostic, comparable across all four models)
print("\n  computing permutation importance on holdout (R=10 reps) ...")
R = 10
rng = np.random.default_rng(2026)

def perm_importance(predict_fn, df_template, base_mape):
    deltas = {}
    for feat in COMBO_COLS:
        d_vals = []
        for _ in range(R):
            d2 = df_template.copy()
            d2[feat] = rng.permutation(d2[feat].values)
            yp = predict_fn(d2)
            d_vals.append(safe_metrics(y_te, yp)["mape"] - base_mape)
        deltas[feat] = (float(np.mean(d_vals)), float(np.std(d_vals)))
    return deltas

# MLP-prod uses 25 base features internally; we still permute on combo so it sees noisier base feature
def predict_mlp_prod(df):
    return prod.predict(df)
def predict_lgb_logT(df):
    return lgb_logT.predict(build_combo_features(df[ALL_COLS]).values)
def predict_xgb_logT(df):
    return xgbr_logT.predict(build_combo_features(df[ALL_COLS]).values)
def predict_rf(df):
    return rf.predict(build_combo_features(df[ALL_COLS]).values)

base = {
    "MLP-prod":  safe_metrics(y_te, predict_mlp_prod(hold))["mape"],
    "LGB-logT":  safe_metrics(y_te, predict_lgb_logT(hold))["mape"],
    "XGB-logT":  safe_metrics(y_te, predict_xgb_logT(hold))["mape"],
    "RF":        safe_metrics(y_te, predict_rf(hold))["mape"],
}
# permute base ALL_COLS (the 25 base features) to keep models comparable
perm_records = []
for feat in ALL_COLS:
    rec = {"feature": feat}
    for model_name, predict_fn in [("MLP-prod", predict_mlp_prod),
                                     ("LGB-logT", predict_lgb_logT),
                                     ("XGB-logT", predict_xgb_logT),
                                     ("RF", predict_rf)]:
        deltas = []
        for _ in range(R):
            d2 = hold.copy()
            d2[feat] = rng.permutation(d2[feat].values)
            yp = predict_fn(d2)
            deltas.append(safe_metrics(y_te, yp)["mape"] - base[model_name])
        rec[f"delta_{model_name}"] = float(np.mean(deltas))
        rec[f"std_{model_name}"]   = float(np.std(deltas))
    perm_records.append(rec)

perm_df = pd.DataFrame(perm_records)
perm_df.to_csv(OUT / "tab_permutation_importance.csv", index=False)
print("\n  Top-10 permutation importance for LightGBM-logT:")
print(perm_df.sort_values("delta_LGB-logT", ascending=False).head(10)[
    ["feature", "delta_LGB-logT", "delta_MLP-prod", "delta_XGB-logT", "delta_RF"]
].round(2).to_string(index=False))


# ── (4) Plots ──────────────────────────────────────────────────────────────
print("\n=== (4) Plotting ===")

# (a) Calibration scatter side by side
fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.9))
for ax, (name, yp) in zip(axes, [
    ("MLP-ensemble raw",  yp_mlp_raw_te),
    ("MLP-ensemble logT", yp_mlp_logT_te),
    ("LightGBM raw",      yp_lgb_raw_te),
    ("LightGBM logT",     yp_lgb_logT_te),
]):
    mask = (y_te > 0) & np.isfinite(yp) & (yp > 0)
    yt, ypp = y_te[mask], yp[mask]
    ax.scatter(yt, ypp, s=4, alpha=0.4, color="#0072B2", edgecolors="none", rasterized=True)
    lo = max(1.0, float(min(yt.min(), ypp.min())) * 0.9)
    hi = float(max(yt.max(), ypp.max())) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    m = safe_metrics(y_te, yp)
    ax.set_title(f"{name}\nMAPE={m['mape']:.2f}%  bias={m['bias']:+.2f}%",
                 loc="left", fontsize=8.5)
    ax.set_xlabel("VROOM cost  [€]")
    ax.set_ylabel("predicted cost  [€]" if ax is axes[0] else "")
fig.suptitle("Fig D1 — 2x2 calibration: (raw / logT) x (MLP / LGB)", x=0.005, ha="left")
fig.tight_layout()
fig.savefig(OUT / "figD1_2x2_calibration.png", dpi=140)
fig.savefig(OUT / "figD1_2x2_calibration.pdf")
plt.close(fig)
print("  wrote figD1")

# (b) Residual histograms head-to-head
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
for ax, (col, label, color) in zip(axes, [
    ("resid_mlp_pct", "MLP-ensemble (production iter17)", "#D55E00"),
    ("resid_lgb_pct", "LightGBM-logT", "#009E73"),
]):
    r = hold_d[col].clip(-50, 50)
    ax.hist(r, bins=80, color=color, alpha=0.75, edgecolor="white", lw=0.3)
    ax.axvline(0, color="k", lw=0.6)
    ax.axvline(r.median(), color="#0072B2", lw=0.7, ls="--",
                label=f"median = {r.median():+.2f}%")
    ax.set_xlabel("rel. residual  [%]"); ax.set_ylabel("count")
    ax.set_title(f"{label}\nMAPE={hold_d[col].abs().mean():.2f}%", loc="left", fontsize=8.5)
    ax.legend()
fig.suptitle("Fig D2 — Residual distribution on holdout (clipped to ±50%)",
             x=0.005, ha="left")
fig.tight_layout()
fig.savefig(OUT / "figD2_residual_hist.png", dpi=140)
fig.savefig(OUT / "figD2_residual_hist.pdf")
plt.close(fig)
print("  wrote figD2")

# (c) Permutation importance side-by-side (top-12 features)
top_feats = (perm_df.sort_values("delta_LGB-logT", ascending=False)
             .head(12)["feature"].tolist())
plot_df = perm_df.set_index("feature").loc[top_feats]

fig, ax = plt.subplots(figsize=(7.16, 4.0))
y_pos = np.arange(len(top_feats))
w = 0.2
colors = {"LGB-logT": "#009E73", "XGB-logT": "#E69F00",
          "MLP-prod": "#D55E00", "RF": "#CC79A7"}
for i, model in enumerate(["LGB-logT", "XGB-logT", "MLP-prod", "RF"]):
    ax.barh(y_pos + (i - 1.5) * w, plot_df[f"delta_{model}"], w,
             color=colors[model], label=model, edgecolor="k", lw=0.3)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_feats)
ax.invert_yaxis()
ax.set_xlabel(r"$\Delta$MAPE on holdout when feature permuted  [pp]")
ax.legend(loc="lower right")
ax.set_title("Fig D3 — Permutation feature importance (top-12, R=10)", loc="left")
fig.tight_layout()
fig.savefig(OUT / "figD3_perm_importance.png", dpi=140)
fig.savefig(OUT / "figD3_perm_importance.pdf")
plt.close(fig)
print("  wrote figD3")

# (d) Per-cost-bin MAPE bars (MLP vs LGB)
fig, ax = plt.subplots(figsize=(7.16, 2.8))
xs = np.arange(len(by_bin))
ax.bar(xs - 0.2, by_bin["mape_mlp"], 0.4, color="#D55E00", label="MLP-ensemble", edgecolor="k", lw=0.3)
ax.bar(xs + 0.2, by_bin["mape_lgb"], 0.4, color="#009E73", label="LightGBM-logT", edgecolor="k", lw=0.3)
ax.set_xticks(xs); ax.set_xticklabels(by_bin["bucket"], fontsize=8)
ax.set_ylabel("MAPE  [%]")
ax.set_yscale("log")
ax.set_title("Fig D4 — MAPE per cost-quintile (where the gap lives)", loc="left")
ax.legend()
for i, (mm, ll) in enumerate(zip(by_bin["mape_mlp"], by_bin["mape_lgb"])):
    ax.text(i - 0.2, mm, f"{mm:.1f}", ha="center", va="bottom", fontsize=7)
    ax.text(i + 0.2, ll, f"{ll:.1f}", ha="center", va="bottom", fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "figD4_per_costbin.png", dpi=140)
fig.savefig(OUT / "figD4_per_costbin.pdf")
plt.close(fig)
print("  wrote figD4")

# (e) Per-LSP MAPE bars (MLP vs LGB)
fig, ax = plt.subplots(figsize=(7.16, 2.8))
by_lsp_sorted = by_lsp.sort_values("mape_mlp", ascending=False)
xs = np.arange(len(by_lsp_sorted))
ax.bar(xs - 0.2, by_lsp_sorted["mape_mlp"], 0.4, color="#D55E00",
        label="MLP-ensemble", edgecolor="k", lw=0.3)
ax.bar(xs + 0.2, by_lsp_sorted["mape_lgb"], 0.4, color="#009E73",
        label="LightGBM-logT", edgecolor="k", lw=0.3)
ax.set_xticks(xs); ax.set_xticklabels(by_lsp_sorted["bucket"], rotation=20, fontsize=8)
ax.set_ylabel("MAPE  [%]")
ax.set_yscale("log")
ax.set_title("Fig D5 — MAPE per LSP (head-to-head)", loc="left")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "figD5_per_lsp.png", dpi=140)
fig.savefig(OUT / "figD5_per_lsp.pdf")
plt.close(fig)
print("  wrote figD5")


# ── summary JSON ───────────────────────────────────────────────────────────
diag = {
    "run_dir":       str(RUN),
    "train_rows":    int(len(pool)),
    "holdout_rows":  int(len(hold)),
    "twobytwo": {
        "MLP_raw_mape":  float(cell_df.set_index("model").loc["MLP-ensemble raw  (production iter17)", "mape"]),
        "MLP_logT_mape": float(cell_df.set_index("model").loc["MLP-ensemble logT (5 seeds, retrained)", "mape"]),
        "LGB_raw_mape":  float(cell_df.set_index("model").loc["LightGBM raw", "mape"]),
        "LGB_logT_mape": float(cell_df.set_index("model").loc["LightGBM logT", "mape"]),
    },
    "biggest_gap_costbin": str(by_bin.loc[by_bin["delta_pp"].idxmax(), "bucket"]),
    "biggest_gap_lsp":     str(by_lsp.loc[by_lsp["delta_pp"].idxmax(), "bucket"]),
    "top3_perm_importance_LGB": (
        perm_df.sort_values("delta_LGB-logT", ascending=False)
        .head(3)[["feature", "delta_LGB-logT"]]
        .to_dict(orient="records")
    ),
}
(OUT / "diagnosis.json").write_text(json.dumps(diag, indent=2, ensure_ascii=False))
print("\nWrote", OUT / "diagnosis.json")
print(json.dumps(diag, indent=2, ensure_ascii=False))
print("\nAll outputs in:", OUT)
