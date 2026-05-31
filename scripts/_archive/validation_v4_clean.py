"""Methodologically clean validation of the proposed surrogate (LightGBM-logT).

Implements the validation protocols that are standard in ML-surrogate
papers for VRP / last-mile logistics (e.g. Akkerman et al. 2023,
Wouda et al. 2024, Vidal et al. 2023):

  V1  Bootstrap 95% CI of holdout MAPE for every top model
  V2  Repeated 5x5 GroupKFold(PLZ) for LightGBM-logT (spatial CV)
  V3  Leave-one-LSP-out (LOGO) -- 7 folds, one carrier per fold
  V4  Learning curve: MAPE vs training-pool size (25 / 50 / 75 / 100 %)
  V5  Nested 5x3 CV with hyperparameter tuning (honest unbiased MAPE)
  V6  Calibration diagnostic + permutation null test
  V7  Paired Wilcoxon test: is LGB-logT significantly better than MLP-prod?

RAM-safe: n_jobs=2, no per-fold MLP training.  Output to
  results/paper_figures/ml_surrogate_v2/validation/
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_helpers import safe_metrics, apply_style  # noqa: E402
from batch_delivery.features import ALL_COLS  # noqa: E402
from batch_delivery.surrogate import MLCostPredictor, build_combo_features  # noqa: E402

apply_style()

RUN = ROOT / "results" / "oracle_loop_extended_2026_05_22"
OUT = ROOT / "results" / "paper_figures" / "ml_surrogate_v2" / "validation"
OUT.mkdir(parents=True, exist_ok=True)

N_JOBS = 2
RNG = np.random.default_rng(2026)

# ── data ────────────────────────────────────────────────────────────────────
pool = pd.read_csv(RUN / "training_matrix.csv")
hold = pd.read_csv(RUN / "holdout_extreme.csv")
print(f"pool    : {len(pool):>6,} rows  ({pool['plz'].nunique()} PLZs, {pool['provider'].nunique()} LSPs)")
print(f"holdout : {len(hold):>6,} rows  ({hold['plz'].nunique()} PLZs, {hold['provider'].nunique()} LSPs)")

pool_combo = build_combo_features(pool[ALL_COLS])
hold_combo = build_combo_features(hold[ALL_COLS])
COMBO_COLS = pool_combo.columns.tolist()

X_tr = pool_combo.values
y_tr = pool["actual_cost_eur"].values
X_te = hold_combo.values
y_te = hold["actual_cost_eur"].values

groups_plz = pool["plz"].values
groups_lsp = pool["provider"].values

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold, ParameterGrid
from scipy import stats
import lightgbm as lgb


def lgb_logT(seed: int = 42, **overrides):
    params = dict(n_estimators=800, learning_rate=0.04, num_leaves=63,
                   max_depth=-1, subsample=0.85, colsample_bytree=0.85,
                   reg_lambda=0.5, min_child_samples=10,
                   n_jobs=N_JOBS, random_state=seed, verbosity=-1)
    params.update(overrides)
    return TransformedTargetRegressor(regressor=lgb.LGBMRegressor(**params),
                                       func=np.log1p, inverse_func=np.expm1)


# ── V1: Bootstrap 95% CI of holdout MAPE for top models ────────────────────
print("\n=== V1: Bootstrap CI on holdout MAPE ===")
# First, get holdout predictions for each top model (train on full pool)
prod = MLCostPredictor.load(RUN / "ml_cost_predictor.pkl")
yp_mlp = prod.predict(hold)

lgb_full = lgb_logT().fit(X_tr, y_tr)
yp_lgb = lgb_full.predict(X_te)

import xgboost as xgb
xgb_full = TransformedTargetRegressor(
    regressor=xgb.XGBRegressor(n_estimators=800, learning_rate=0.04, max_depth=7,
                                subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                                min_child_weight=3, n_jobs=N_JOBS, random_state=42,
                                tree_method="hist", verbosity=0),
    func=np.log1p, inverse_func=np.expm1)
xgb_full.fit(X_tr, y_tr)
yp_xgb = xgb_full.predict(X_te)


def bootstrap_mape(y_true, y_pred, n_boot=2000, alpha=0.05):
    n = len(y_true)
    mapes = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        mapes[b] = safe_metrics(y_true[idx], y_pred[idx])["mape"]
    return float(np.mean(mapes)), float(np.percentile(mapes, 100 * alpha / 2)), float(np.percentile(mapes, 100 * (1 - alpha / 2)))


v1_rows = []
for name, yp in [("LightGBM-logT", yp_lgb), ("XGBoost-logT", yp_xgb),
                  ("MLP-ensemble (iter17, prod)", yp_mlp)]:
    mean, lo, hi = bootstrap_mape(y_te, yp, n_boot=2000)
    pt = safe_metrics(y_te, yp)["mape"]
    v1_rows.append({"model": name, "mape_point": pt,
                    "mape_boot_mean": mean, "ci95_lo": lo, "ci95_hi": hi})
    print(f"  {name:<32s}  MAPE={pt:.3f}  boot-mean={mean:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")

v1_df = pd.DataFrame(v1_rows)
v1_df.to_csv(OUT / "V1_bootstrap_holdout_ci.csv", index=False)


# ── V2: Repeated 5x5 GroupKFold(PLZ) for LightGBM-logT ─────────────────────
print("\n=== V2: Repeated 5x5 GroupKFold(PLZ) for LGB-logT ===")
N_REPS = 5
N_FOLDS = 5

v2_fold_mapes = []
for rep in range(N_REPS):
    # GroupKFold has no shuffle; we shuffle PLZ→fold mapping manually
    plz_unique = pool["plz"].unique()
    perm = RNG.permutation(plz_unique)
    fold_of = {plz: i % N_FOLDS for i, plz in enumerate(perm)}
    fold_assign = np.array([fold_of[p] for p in pool["plz"]])
    for k in range(N_FOLDS):
        tr_idx = np.where(fold_assign != k)[0]
        te_idx = np.where(fold_assign == k)[0]
        if len(te_idx) == 0: continue
        m = lgb_logT(seed=rep * 100 + k).fit(X_tr[tr_idx], y_tr[tr_idx])
        yp = m.predict(X_tr[te_idx])
        mape = safe_metrics(y_tr[te_idx], yp)["mape"]
        v2_fold_mapes.append({"rep": rep, "fold": k, "n_test": len(te_idx), "mape": mape})
    print(f"  rep {rep+1}/{N_REPS} done")

v2_df = pd.DataFrame(v2_fold_mapes)
v2_df.to_csv(OUT / "V2_repeated_groupkfold_per_fold.csv", index=False)
mapes = v2_df["mape"].values
print(f"  Repeated 5x5 GroupKFold(PLZ) MAPE:")
print(f"    mean ± std    = {mapes.mean():.3f} ± {mapes.std(ddof=1):.3f} %")
print(f"    95% CI (norm) = [{mapes.mean() - 1.96*mapes.std(ddof=1)/np.sqrt(len(mapes)):.3f}, "
      f"{mapes.mean() + 1.96*mapes.std(ddof=1)/np.sqrt(len(mapes)):.3f}]")
print(f"    range         = [{mapes.min():.3f}, {mapes.max():.3f}]")


# ── V3: Leave-one-LSP-out (LOGO) ───────────────────────────────────────────
print("\n=== V3: Leave-one-LSP-out (7 folds) ===")
lsp_unique = sorted(pool["provider"].unique())
v3_rows = []
for left_out in lsp_unique:
    tr_mask = pool["provider"] != left_out
    te_mask = pool["provider"] == left_out
    m = lgb_logT().fit(X_tr[tr_mask], y_tr[tr_mask])
    yp = m.predict(X_tr[te_mask])
    met = safe_metrics(y_tr[te_mask], yp)
    v3_rows.append({"left_out_lsp": left_out, **met})
    print(f"  left-out {left_out:<8s}  n={met['n']:>4d}  MAPE={met['mape']:6.2f}  R2={met['r2']:.4f}")

v3_df = pd.DataFrame(v3_rows)
v3_df.to_csv(OUT / "V3_leave_one_lsp_out.csv", index=False)
print(f"  mean LSP-out MAPE = {v3_df['mape'].mean():.3f}  std = {v3_df['mape'].std():.3f}")


# ── V4: Learning curve ────────────────────────────────────────────────────
print("\n=== V4: Learning curve (LGB-logT) — MAPE vs training-pool size ===")
fracs = [0.25, 0.50, 0.75, 1.00]
v4_rows = []
for frac in fracs:
    # Random subsample of PLZs first, then take their rows
    fold_assign_full = RNG.permutation(len(pool))[:int(len(pool) * frac)]
    Xtr_sub = X_tr[fold_assign_full]
    ytr_sub = y_tr[fold_assign_full]
    groups_sub = groups_plz[fold_assign_full]
    # 5-fold GroupKFold on the subset
    gkf = GroupKFold(n_splits=5)
    yhat = np.zeros_like(ytr_sub, dtype=float)
    for tr_idx, te_idx in gkf.split(Xtr_sub, ytr_sub, groups=groups_sub):
        m = lgb_logT().fit(Xtr_sub[tr_idx], ytr_sub[tr_idx])
        yhat[te_idx] = m.predict(Xtr_sub[te_idx])
    m_cv = safe_metrics(ytr_sub, yhat)
    # Also evaluate on the held-out extreme set
    m_full = lgb_logT().fit(Xtr_sub, ytr_sub)
    yp_h = m_full.predict(X_te)
    m_hold = safe_metrics(y_te, yp_h)
    v4_rows.append({"frac": frac, "n_used": len(Xtr_sub),
                    "cv_mape": m_cv["mape"], "cv_r2": m_cv["r2"],
                    "hold_mape": m_hold["mape"], "hold_r2": m_hold["r2"]})
    print(f"  frac={frac:.2f} n={len(Xtr_sub):>5,}  CV-MAPE={m_cv['mape']:.3f}  Hold-MAPE={m_hold['mape']:.3f}")

v4_df = pd.DataFrame(v4_rows)
v4_df.to_csv(OUT / "V4_learning_curve.csv", index=False)


# ── V5: Nested 5x3 CV with hyperparameter tuning ─────────────────────────
print("\n=== V5: Nested 5x3 CV with HP tuning (LGB-logT) ===")
# Small but realistic search grid
PARAM_GRID = list(ParameterGrid({
    "n_estimators":  [600, 1000],
    "learning_rate": [0.03, 0.05],
    "num_leaves":    [31, 127],
}))  # 8 combos x 3 inner x 5 outer = 120 fits

outer = GroupKFold(n_splits=5)
v5_outer_rows = []
v5_chosen_hps = []

for fold_idx, (tr_idx, te_idx) in enumerate(outer.split(X_tr, y_tr, groups=groups_plz)):
    Xo_tr, yo_tr = X_tr[tr_idx], y_tr[tr_idx]
    Xo_te, yo_te = X_tr[te_idx], y_tr[te_idx]
    go_tr = groups_plz[tr_idx]
    inner = GroupKFold(n_splits=3)

    # Inner: pick best HP combo by inner-CV MAPE
    best = {"hps": None, "mape": np.inf}
    for hps in PARAM_GRID:
        inner_mapes = []
        for i_tr, i_te in inner.split(Xo_tr, yo_tr, groups=go_tr):
            m = lgb_logT(**hps).fit(Xo_tr[i_tr], yo_tr[i_tr])
            inner_mapes.append(safe_metrics(yo_tr[i_te], m.predict(Xo_tr[i_te]))["mape"])
        mean_inner = float(np.mean(inner_mapes))
        if mean_inner < best["mape"]:
            best = {"hps": hps, "mape": mean_inner}

    # Retrain on full outer-train with best HPs, evaluate on outer-test
    m = lgb_logT(**best["hps"]).fit(Xo_tr, yo_tr)
    outer_mape = safe_metrics(yo_te, m.predict(Xo_te))["mape"]
    v5_outer_rows.append({"outer_fold": fold_idx, "n_test": len(te_idx),
                           "best_inner_mape": best["mape"],
                           "outer_mape": outer_mape, **best["hps"]})
    v5_chosen_hps.append(best["hps"])
    print(f"  outer fold {fold_idx+1}/5  HPs={best['hps']}  outer-MAPE={outer_mape:.3f}")

v5_df = pd.DataFrame(v5_outer_rows)
v5_df.to_csv(OUT / "V5_nested_cv.csv", index=False)
print(f"  Nested-CV unbiased MAPE = {v5_df['outer_mape'].mean():.3f} ± {v5_df['outer_mape'].std(ddof=1):.3f}")


# ── V6: Calibration diagnostic + permutation null test ─────────────────────
print("\n=== V6: Calibration + permutation null test ===")
# Calibration on holdout: predict-quantile vs actual-quantile
yp_lgb_te = lgb_full.predict(X_te)
df_cal = pd.DataFrame({"y": y_te, "yp": yp_lgb_te})
df_cal["q_pred"] = pd.qcut(df_cal["yp"], q=10, labels=False, duplicates="drop")
cal = df_cal.groupby("q_pred").agg(
    n=("y", "size"),
    mean_pred=("yp", "mean"),
    mean_actual=("y", "mean"),
).reset_index()
cal["ratio"] = cal["mean_pred"] / cal["mean_actual"]
cal.to_csv(OUT / "V6_calibration_deciles.csv", index=False)
print("  Calibration per decile (pred/actual ratio should be ~1.0):")
print(cal.round(3).to_string(index=False))

# Permutation null: shuffle y_tr, refit, look at holdout MAPE
print("  Permutation null test (LGB-logT on shuffled target, 3 reps) ...")
null_mapes = []
for r in range(3):
    y_perm = RNG.permutation(y_tr)
    m_null = lgb_logT(seed=r).fit(X_tr, y_perm)
    null_mapes.append(safe_metrics(y_te, m_null.predict(X_te))["mape"])
real_mape = safe_metrics(y_te, yp_lgb_te)["mape"]
print(f"    real model   MAPE = {real_mape:.3f}")
print(f"    null (perm)  MAPE = {np.mean(null_mapes):.3f} ± {np.std(null_mapes, ddof=1):.3f}")
print(f"    ratio        = {np.mean(null_mapes) / real_mape:.1f}x worse")


# ── V7: Paired Wilcoxon test between LGB-logT and MLP-ensemble ─────────────
print("\n=== V7: Paired Wilcoxon test (LGB-logT vs MLP-prod on holdout) ===")
abs_err_lgb = np.abs(yp_lgb_te - y_te) / y_te * 100
abs_err_mlp = np.abs(yp_mlp   - y_te) / y_te * 100
wstat, pval = stats.wilcoxon(abs_err_lgb, abs_err_mlp, alternative="less")
print(f"  median |APE| LGB-logT = {np.median(abs_err_lgb):.3f} %")
print(f"  median |APE| MLP-prod = {np.median(abs_err_mlp):.3f} %")
print(f"  Wilcoxon signed-rank (one-sided, LGB < MLP):  W = {wstat:.1f}  p = {pval:.2e}")


# ── Plots ─────────────────────────────────────────────────────────────────
# V1: forest plot
fig, ax = plt.subplots(figsize=(6.5, 2.2))
ys = np.arange(len(v1_df))
ax.errorbar(v1_df["mape_boot_mean"], ys,
             xerr=[v1_df["mape_boot_mean"] - v1_df["ci95_lo"],
                   v1_df["ci95_hi"] - v1_df["mape_boot_mean"]],
             fmt="o", capsize=4, color="#0072B2")
for i, r in v1_df.iterrows():
    ax.text(r["ci95_hi"] + 0.1, i,
             f"{r['mape_point']:.2f}  [{r['ci95_lo']:.2f}, {r['ci95_hi']:.2f}]",
             va="center", fontsize=8)
ax.set_yticks(ys); ax.set_yticklabels(v1_df["model"], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("MAPE on frozen holdout  [%]  (95% bootstrap CI)")
ax.set_xscale("log")
ax.set_title("Fig V1 — Bootstrap 95% CI of holdout MAPE", loc="left")
fig.tight_layout()
fig.savefig(OUT / "figV1_bootstrap_ci.png", dpi=140)
plt.close(fig)

# V2: box of repeated CV
fig, ax = plt.subplots(figsize=(5.0, 2.4))
ax.boxplot(mapes, vert=False, widths=0.5, patch_artist=True,
            boxprops=dict(facecolor="#0072B2", alpha=0.3))
ax.scatter(mapes, [1]*len(mapes), color="#0072B2", s=10, alpha=0.6)
ax.set_xlabel("MAPE per (rep, fold)  [%]")
ax.set_yticks([1]); ax.set_yticklabels(["LGB-logT"])
ax.set_title(f"Fig V2 — Repeated 5x5 GroupKFold(PLZ)  (n={len(mapes)} folds)", loc="left")
fig.tight_layout()
fig.savefig(OUT / "figV2_repeated_cv.png", dpi=140)
plt.close(fig)

# V3: LOGO bars
fig, ax = plt.subplots(figsize=(6.0, 2.4))
ds = v3_df.sort_values("mape", ascending=False)
ax.bar(ds["left_out_lsp"], ds["mape"], color="#D55E00", edgecolor="k", lw=0.4)
for i, r in enumerate(ds.itertuples()):
    ax.text(i, r.mape, f"{r.mape:.1f}", ha="center", va="bottom", fontsize=8)
ax.axhline(v3_df["mape"].mean(), color="k", lw=0.5, ls="--",
            label=f"mean = {v3_df['mape'].mean():.2f}%")
ax.set_ylabel("MAPE on left-out LSP  [%]"); ax.legend()
ax.set_title("Fig V3 — Leave-one-LSP-out", loc="left")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(OUT / "figV3_logo_lsp.png", dpi=140)
plt.close(fig)

# V4: Learning curve
fig, ax = plt.subplots(figsize=(5.5, 2.6))
ax.plot(v4_df["n_used"], v4_df["cv_mape"], "o-", color="#009E73",
         label="GroupKFold-CV MAPE")
ax.plot(v4_df["n_used"], v4_df["hold_mape"], "s-", color="#D55E00",
         label="Frozen-holdout MAPE")
ax.set_xlabel("training pool size  [rows]"); ax.set_ylabel("MAPE  [%]")
ax.set_title("Fig V4 — Learning curve (LGB-logT)", loc="left")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "figV4_learning_curve.png", dpi=140)
plt.close(fig)

# V6: Calibration plot
fig, ax = plt.subplots(figsize=(4.5, 2.4))
ax.plot(cal["mean_pred"], cal["mean_actual"], "o-", color="#0072B2")
lims = [min(cal["mean_pred"].min(), cal["mean_actual"].min()) * 0.9,
        max(cal["mean_pred"].max(), cal["mean_actual"].max()) * 1.1]
ax.plot(lims, lims, "k--", lw=0.5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
ax.set_xlabel("mean predicted cost / decile  [€]")
ax.set_ylabel("mean actual cost / decile  [€]")
ax.set_title("Fig V6 — Calibration (LGB-logT)", loc="left")
fig.tight_layout()
fig.savefig(OUT / "figV6_calibration.png", dpi=140)
plt.close(fig)


# ── Final summary ─────────────────────────────────────────────────────────
summary = {
    "run_dir":            str(RUN),
    "train_rows":         int(len(pool)),
    "holdout_rows":       int(len(hold)),
    "V1_holdout": {r["model"]: {"point": r["mape_point"], "ci95": [r["ci95_lo"], r["ci95_hi"]]}
                    for r in v1_rows},
    "V2_repeated_5x5_groupkfold_PLZ_mape": {
        "n_folds_total": int(len(mapes)),
        "mean": float(mapes.mean()),
        "std":  float(mapes.std(ddof=1)),
        "ci95_norm": [float(mapes.mean() - 1.96 * mapes.std(ddof=1) / np.sqrt(len(mapes))),
                       float(mapes.mean() + 1.96 * mapes.std(ddof=1) / np.sqrt(len(mapes)))],
        "min": float(mapes.min()), "max": float(mapes.max()),
    },
    "V3_LOGO_LSP_mape":  {"mean": float(v3_df["mape"].mean()),
                            "std":  float(v3_df["mape"].std(ddof=1)),
                            "worst_LSP": str(v3_df.loc[v3_df["mape"].idxmax(), "left_out_lsp"]),
                            "best_LSP":  str(v3_df.loc[v3_df["mape"].idxmin(), "left_out_lsp"])},
    "V4_learning_curve": v4_df.to_dict(orient="records"),
    "V5_nested_unbiased_MAPE": {"mean": float(v5_df["outer_mape"].mean()),
                                  "std":  float(v5_df["outer_mape"].std(ddof=1)),
                                  "chosen_HPs": v5_chosen_hps},
    "V6_calibration_deciles": cal.to_dict(orient="records"),
    "V6_permutation_null_mape": {"real": float(real_mape),
                                   "null_mean": float(np.mean(null_mapes)),
                                   "null_std":  float(np.std(null_mapes, ddof=1)),
                                   "ratio_null_over_real": float(np.mean(null_mapes) / real_mape)},
    "V7_paired_wilcoxon": {"median_APE_LGB": float(np.median(abs_err_lgb)),
                             "median_APE_MLP": float(np.median(abs_err_mlp)),
                             "W": float(wstat), "p_value": float(pval)},
}
(OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=str, ensure_ascii=False))
print("\nWrote", OUT / "validation_summary.json")
print("\n=== Headline ===")
print(f"V1 LGB-logT holdout MAPE = {v1_rows[0]['mape_point']:.3f} % "
      f"(95% CI [{v1_rows[0]['ci95_lo']:.3f}, {v1_rows[0]['ci95_hi']:.3f}])")
print(f"V2 spatial-CV MAPE       = {mapes.mean():.3f} ± {mapes.std(ddof=1):.3f} %  (n={len(mapes)} folds)")
print(f"V3 LSP-out MAPE          = {v3_df['mape'].mean():.3f} ± {v3_df['mape'].std(ddof=1):.3f} %")
print(f"V5 nested-CV (unbiased)  = {v5_df['outer_mape'].mean():.3f} ± {v5_df['outer_mape'].std(ddof=1):.3f} %")
print(f"V6 null ratio            = {np.mean(null_mapes) / real_mape:.1f}x worse than real model")
print(f"V7 Wilcoxon p-value      = {pval:.2e}  (LGB-logT < MLP-prod)")
