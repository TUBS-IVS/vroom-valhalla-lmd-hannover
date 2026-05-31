"""Lightweight model benchmark on the latest oracle-loop run.

Mirrors §6 of paper_figures_ml.ipynb but:
  * points to results/oracle_loop_extended_2026_05_22 (10,652 train / 1,854 holdout)
  * uses n_jobs=2 (the oracle-loop GUI is still using ~10 GB RAM)
  * does NOT retrain an MLP ensemble (it would be slow + RAM-heavy);
    instead reuses the production ml_cost_predictor.pkl

Run:
    python scripts/benchmark_v2_latest.py
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
from batch_delivery.surrogate import MLCostPredictor  # noqa: E402

RUN = ROOT / "results" / "oracle_loop_extended_2026_05_22"
OUT = ROOT / "results" / "paper_figures" / "ml_surrogate_v2"
OUT.mkdir(parents=True, exist_ok=True)

N_JOBS = 2  # leave headroom for the running oracle-loop generator

# ── data ────────────────────────────────────────────────────────────────────
pool = pd.read_csv(RUN / "training_matrix.csv")
hold = pd.read_csv(RUN / "holdout_extreme.csv")
hist = pd.read_csv(RUN / "oracle_loop_history.csv")
print(f"pool    : {len(pool):>6,} rows  ({pool['plz'].nunique()} PLZs, {pool['provider'].nunique()} LSPs)")
print(f"holdout : {len(hold):>6,} rows  ({hold['plz'].nunique()} PLZs, {hold['provider'].nunique()} LSPs)")
print(f"history : {len(hist)} iterations recorded")

# Shared train/test arrays
X_tr = pool[ALL_COLS].values
y_tr = pool["actual_cost_eur"].values
X_te = hold[ALL_COLS].values
y_te = hold["actual_cost_eur"].values

# ── A) Interpolation holdout — same PLZ overlap as iter17 oracle data ────
print("\n=== (A) Interpolation holdout benchmark ===")
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def sc(est):
    return Pipeline([("sc", StandardScaler()), ("m", est)])

models = {
    "LR":          sc(LinearRegression()),
    "Ridge":       sc(Ridge(alpha=1.0, random_state=42)),
    "RF":          RandomForestRegressor(n_estimators=200, max_depth=12,
                                          n_jobs=N_JOBS, random_state=42),
    "HistGBM":     HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                  max_depth=8, random_state=42),
    "MLP-single":  sc(MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32),
                                    alpha=1e-4, max_iter=400, early_stopping=True,
                                    random_state=42)),
}

rows = [{"model": "Daganzo (textbook 1984)", **safe_metrics(y_te, daganzo_predict(hold)),
         "fit_sec": 0.0}]
for name, est in models.items():
    t0 = time.time()
    est.fit(X_tr, y_tr)
    yp = est.predict(X_te)
    rows.append({"model": name, **safe_metrics(y_te, yp), "fit_sec": time.time() - t0})
    print(f"  {name:<14s} fit {rows[-1]['fit_sec']:6.1f}s  "
          f"MAPE={rows[-1]['mape']:6.2f}  R2={rows[-1]['r2']:.4f}")

# Production ensemble (already trained — no extra training cost)
prod = MLCostPredictor.load(RUN / "ml_cost_predictor.pkl")
yp_prod = prod.predict(hold)
rows.append({"model": "MLP-ensemble (oracle iter17)",
             **safe_metrics(y_te, yp_prod), "fit_sec": 0.0})
print(f"  MLP-ensemble    MAPE={rows[-1]['mape']:6.2f}  R2={rows[-1]['r2']:.4f}  (loaded, not retrained)")

bench = pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)
bench.to_csv(OUT / "tab2_benchmark_holdout.csv", index=False)
print("\nFinal ranking (sorted by MAPE):")
print(bench[["model", "n", "mape", "mae", "rmse", "r2", "bias"]].round(3).to_string(index=False))

# ── B) Per-iteration holdout trajectory of the saved MLP ensembles ───────
print("\n=== (B) Per-iteration MLP-ensemble on extended holdout ===")
iter_rows = []
for k in range(1, 18):  # iter01..iter17
    p = RUN / f"ml_cost_predictor_iter{k:02d}.pkl"
    if not p.exists():
        continue
    m = MLCostPredictor.load(p)
    yp = m.predict(hold)
    iter_rows.append({"iter": k, **safe_metrics(y_te, yp)})
    print(f"  iter{k:02d}  MAPE={iter_rows[-1]['mape']:6.2f}  R2={iter_rows[-1]['r2']:.4f}")

iter_df = pd.DataFrame(iter_rows)
iter_df.to_csv(OUT / "tab3_iter_trajectory.csv", index=False)
best_iter = iter_df.loc[iter_df["mape"].idxmin()]
print(f"\nBest iter on extended holdout: iter{int(best_iter['iter']):02d}  "
      f"MAPE={best_iter['mape']:.3f}  R2={best_iter['r2']:.4f}")

# ── C) Per-LSP / per-extremity breakdown for the production ensemble ────
def classify_extremity(row):
    if row.get("is_baseline", False):
        return "baseline"
    extreme = (
        (row.get("b2c_scale", 1.0) >= 1.2) or
        (row.get("b2b_scale", 1.0) <= 0.93 or row.get("b2b_scale", 1.0) >= 1.075) or
        (row.get("noise_sigma", 0.0) >= 0.3) or
        (row.get("p_keep", 1.0) <= 0.6) or
        (row.get("scale", 1.0) <= 0.7 or row.get("scale", 1.0) >= 1.5)
    )
    return "extreme" if extreme else "mild"

hold = hold.copy()
hold["pred"] = yp_prod
hold["extremity"] = hold.apply(classify_extremity, axis=1)
hold["resid_pct"] = (hold["pred"] - hold["actual_cost_eur"]) / hold["actual_cost_eur"] * 100

per_lsp = (hold.groupby("provider")
           .apply(lambda g: pd.Series(safe_metrics(g["actual_cost_eur"], g["pred"])))
           .sort_values("mape"))
per_ext = (hold.groupby("extremity")
           .apply(lambda g: pd.Series(safe_metrics(g["actual_cost_eur"], g["pred"])))
           .sort_values("mape"))

per_lsp.to_csv(OUT / "tab4_per_lsp.csv")
per_ext.to_csv(OUT / "tab4_per_extremity.csv")
print("\nPer-LSP MAPE (MLP-ensemble):")
print(per_lsp[["n", "mape", "r2", "bias"]].round(3))
print("\nPer-extremity MAPE (MLP-ensemble):")
print(per_ext[["n", "mape", "r2", "bias"]].round(3))

# ── headline summary ────────────────────────────────────────────────────
summary = {
    "run_dir":                str(RUN),
    "iterations_done":        int(len(hist)),
    "final_training_rows":    int(len(pool)),
    "extended_holdout_rows":  int(len(hold)),
    "best_holdout_iter":      int(best_iter["iter"]),
    "best_holdout_mape":      float(best_iter["mape"]),
    "best_holdout_r2":        float(best_iter["r2"]),
    "final_iter17_mape":      float(iter_df.set_index("iter").loc[max(iter_df["iter"]), "mape"]),
    "benchmark_winner":       str(bench["model"].iloc[0]),
    "benchmark_winner_mape":  float(bench["mape"].iloc[0]),
    "mlp_ensemble_mape":      float(bench.set_index("model").loc["MLP-ensemble (oracle iter17)", "mape"]),
    "daganzo_mape":           float(bench.set_index("model").loc["Daganzo (textbook 1984)", "mape"]),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print("\nWrote", OUT / "summary.json")
print(json.dumps(summary, indent=2))
