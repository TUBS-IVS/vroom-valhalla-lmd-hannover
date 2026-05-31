"""Ablation: does pool composition matter?  Are baselines redundant?

Four training configurations:
  A   Full pool (current)              11 241 rows (~56% baseline + 44% perturbed)
  B   Perturbed-only                    4 962 rows (drop all is_baseline=True)
  C   Baseline-only                     6 279 rows (drop all is_baseline=False)
  D   Dedup baselines                  ~7 000 rows (1 row per baseline group + all perturbed)

For each: train LightGBM-logT (production HPs) and evaluate on
  * the frozen extreme-holdout (1 927 rows, 100% perturbed)
  * 5-fold GroupKFold(PLZ) on the *full* pool, but using only the configured subset for training

Answers:
  Q1  Is the "baseline" half of the training pool actually informative for predicting extreme perturbations?
  Q2  Are the quasi-duplicate baselines (same geometry, different VROOM seed) just dead weight?
  Q3  How does each config compare on per-extremity MAPE?
"""
from __future__ import annotations

import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_helpers import safe_metrics
from batch_delivery.features import ALL_COLS
from batch_delivery.surrogate import build_combo_features

from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

RUN = ROOT / "results" / "oracle_loop_extended_2026_05_22"
OUT = ROOT / "results" / "paper_figures" / "ml_surrogate_v2" / "ablation"
OUT.mkdir(parents=True, exist_ok=True)

N_JOBS = 2
LGB_HPS = dict(n_estimators=1000, learning_rate=0.05, num_leaves=31, max_depth=-1,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=0.5,
                min_child_samples=10, n_jobs=N_JOBS, random_state=42, verbosity=-1)

def make_lgb():
    return TransformedTargetRegressor(regressor=lgb.LGBMRegressor(**LGB_HPS),
                                       func=np.log1p, inverse_func=np.expm1)

# ── data ───────────────────────────────────────────────────────────────────
pool = pd.read_csv(RUN / "training_matrix.csv")
hold = pd.read_csv(RUN / "holdout_extreme.csv")

# Build feature matrices and target
def build_X(df):
    return build_combo_features(df[ALL_COLS]).values

X_te = build_X(hold);  y_te = hold["actual_cost_eur"].values

# Extremity classification on holdout (for per-tier evaluation)
def classify_extremity(row):
    if row.get("is_baseline", False): return "baseline"
    if ((row.get("b2c_scale", 1.0) >= 1.2) or
        (row.get("b2b_scale", 1.0) <= 0.93 or row.get("b2b_scale", 1.0) >= 1.075) or
        (row.get("noise_sigma", 0.0) >= 0.3) or
        (row.get("p_keep", 1.0) <= 0.6)):
        return "extreme"
    return "mild"
hold = hold.copy(); hold["extremity"] = hold.apply(classify_extremity, axis=1)

print(f"pool    : {len(pool):,} rows")
print(f"holdout : {len(hold):,} rows")
print(f"  extremity: {hold['extremity'].value_counts().to_dict()}")
print(f"\nLGB HPs: {LGB_HPS}")

# ── define the four configs ────────────────────────────────────────────────
def dedup_baselines(df):
    """Keep exactly 1 row per (provider, plz, base_day, agg_k) baseline group + all perturbed."""
    gcols = ["provider", "plz", "base_day", "agg_k"]
    bl = df[df["is_baseline"]].sort_values("seed").drop_duplicates(subset=gcols, keep="first")
    pt = df[~df["is_baseline"]]
    return pd.concat([bl, pt]).reset_index(drop=True)

configs = {
    "A_full_pool":        pool.copy(),
    "B_perturbed_only":   pool[~pool["is_baseline"]].copy(),
    "C_baseline_only":    pool[pool["is_baseline"]].copy(),
    "D_dedup_baselines":  dedup_baselines(pool),
}

print("\n=== Config sizes ===")
for k, v in configs.items():
    n_bl = v["is_baseline"].sum()
    n_pt = (~v["is_baseline"]).sum()
    print(f"  {k:25s}  rows={len(v):>5,}  baseline={n_bl:>5,}  perturbed={n_pt:>5,}")

# ── run ablation ───────────────────────────────────────────────────────────
print("\n=== Training + evaluation on frozen holdout (100% perturbed) ===")
results = []
for name, df in configs.items():
    t0 = time.time()
    X_tr = build_X(df); y_tr = df["actual_cost_eur"].values
    m = make_lgb().fit(X_tr, y_tr)
    yp = m.predict(X_te)
    overall = safe_metrics(y_te, yp); overall["bucket"] = "all"
    # per-extremity
    per_ext = []
    for ext in ["extreme", "mild"]:
        mask = hold["extremity"].values == ext
        if mask.sum() > 0:
            mm = safe_metrics(y_te[mask], yp[mask]); mm["bucket"] = ext
            per_ext.append(mm)
    fit_sec = time.time() - t0
    for ent in [overall] + per_ext:
        ent["config"]  = name
        ent["n_train"] = len(df)
        ent["fit_sec"] = fit_sec
        results.append(ent)
    print(f"  {name:25s}  n_train={len(df):>5,}  fit={fit_sec:5.1f}s  "
          f"all-MAPE={overall['mape']:6.3f}%  R2={overall['r2']:.4f}")

res = pd.DataFrame(results)
res.to_csv(OUT / "ablation_holdout.csv", index=False)

print("\n=== Per-extremity head-to-head on holdout (MAPE %) ===")
pivot = res.pivot(index="config", columns="bucket", values="mape").round(3)
print(pivot.to_string())
pivot.to_csv(OUT / "ablation_holdout_pivot.csv")

# ── 5-fold GroupKFold(PLZ) — evaluate each config's training subset on full-pool PLZ structure ──
print("\n=== 5-fold GroupKFold(PLZ) per config (out-of-fold MAPE on each config's training subset) ===")
gkf = GroupKFold(n_splits=5)
cv_rows = []
for name, df in configs.items():
    X = build_X(df); y = df["actual_cost_eur"].values
    groups = df["plz"].values
    yhat = np.zeros_like(y, dtype=float)
    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        m = make_lgb().fit(X[tr_idx], y[tr_idx])
        yhat[te_idx] = m.predict(X[te_idx])
    met = safe_metrics(y, yhat); met["config"] = name; met["n_rows"] = len(df)
    cv_rows.append(met)
    print(f"  {name:25s}  CV-MAPE={met['mape']:6.3f}%")
cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(OUT / "ablation_groupkfold.csv", index=False)

# ── headline summary ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

m_full     = res[(res["config"] == "A_full_pool")        & (res["bucket"] == "all")]["mape"].iloc[0]
m_perturb  = res[(res["config"] == "B_perturbed_only")   & (res["bucket"] == "all")]["mape"].iloc[0]
m_baseline = res[(res["config"] == "C_baseline_only")    & (res["bucket"] == "all")]["mape"].iloc[0]
m_dedup    = res[(res["config"] == "D_dedup_baselines")  & (res["bucket"] == "all")]["mape"].iloc[0]

print(f"Holdout MAPE  full pool      = {m_full:6.3f}%  (n={len(configs['A_full_pool']):,})")
print(f"Holdout MAPE  perturbed-only = {m_perturb:6.3f}%  (n={len(configs['B_perturbed_only']):,})  delta vs full = {m_perturb - m_full:+.3f} pp")
print(f"Holdout MAPE  baseline-only  = {m_baseline:6.3f}%  (n={len(configs['C_baseline_only']):,})  delta vs full = {m_baseline - m_full:+.3f} pp")
print(f"Holdout MAPE  dedup          = {m_dedup:6.3f}%  (n={len(configs['D_dedup_baselines']):,})  delta vs full = {m_dedup - m_full:+.3f} pp")

print("\nInterpretation:")
if m_perturb - m_full < 0.1:
    print("  - Removing baselines does NOT hurt the holdout much.")
    print("    => baselines are nearly redundant for extreme-holdout prediction.")
else:
    print("  - Baselines provide measurable anchor information for predicting extreme perturbations.")
    print("    => keeping them is justified.")

if abs(m_dedup - m_full) < 0.1:
    print("  - Deduplicating quasi-duplicate baselines does NOT change holdout performance.")
    print("    => the duplicates are dead weight in terms of holdout-MAPE, but harmless.")
else:
    print("  - Deduplication changes holdout MAPE materially.")

if m_baseline - m_full > 1.0:
    print("  - Training on baselines ONLY is much worse on the extreme holdout.")
    print("    => the perturbation curriculum is essential for the active-learning story.")
