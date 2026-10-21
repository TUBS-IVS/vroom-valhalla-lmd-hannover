"""Model-Level Debiasing: Target-Transformation gegen die Saving-Differenz-Amplifikation.

Frage: Kann die Saving-Bias-Amplifikation (+10.1 pp) durch Target-Transformation
schon waehrend des Trainings korrigiert werden, statt durch Post-Hoc Calibration?

Hypothese: Log-Target-Training (model lernt log1p(cost)) gleicht relative Fehler
ueber den ganzen Cost-Range aus. Das reduziert die Amplifikation in
Saving = (C_base - C_batched) / C_base, weil beide Cost-Vorhersagen denselben
proportionalen Fehler haben.

Setup:
  - 4 Modell-Varianten auf training_matrix.csv (11'523 rows, 25 features)
    A) LGB-raw    : target = actual_cost_eur
    B) LGB-log    : target = log1p(actual_cost_eur), invert via expm1
    C) MLP-raw    : target = actual_cost_eur, with StandardScaler
    D) MLP-log    : target = log1p(actual_cost_eur), with StandardScaler

  - 5-fold GroupKFold ueber PLZ (gleiche Splits fuer alle 4 Varianten)
  - Per Fold: Cost-MAPE + naturlich-gepaarte Saving-Bias (304 Pairs in training_matrix)

Output:
  results/model_level_debiasing/ (default; override with --out-dir)
    tab_variant_comparison.csv       Cost-MAPE + Saving-Bias pro Variante
    tab_per_fold.csv                 Per-Fold-Details
    fig_DB1_variant_comparison.{pdf,png}
    REPORT.md

CLI (Task 19 W1c): --train-csv/--out-dir let a re-run point at the current
canonical pool and at a non-source output directory without editing the
script. The historical default input (results/oracle_loop_extended_2026_05_22/
training_matrix.csv, 11'523 rows) now lives only under gitignored
results/_archive/; the new default below points at the current canonical
pool instead (results/supplementary/sweep_v3_mergefix/training_matrix.csv,
2'733 rows) per the Task 19 inventory's recommendation.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[2]  # scripts/exploratory/<file> -> repo root (was parents[1] -> scripts/, off by one; Task 19 W1c fix)
DEFAULT_TRAIN_CSV = ROOT / "results" / "supplementary" / "sweep_v3_mergefix" / "training_matrix.csv"
DEFAULT_OUT = ROOT / "results" / "model_level_debiasing"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV,
                    help=f"training matrix CSV (default: {DEFAULT_TRAIN_CSV})")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help=f"output directory (default: {DEFAULT_OUT})")
    return p.parse_args()

BASE_FEATURES = [
    "n_parcels", "n_stops", "area_km2", "hub_dist_km", "parcels_per_stop",
    "load_factor", "min_vehicles", "parcels_per_km2",
    "ch_area_km2", "ch_perimeter_km", "mean_nn_dist_km", "mean_inter_stop_dist_km",
    "stop_density_ch", "centroid_hub_dist_km", "max_hub_dist_km",
    "coord_std_x", "coord_std_y", "aspect_ratio",
    "b2c_share", "demand_std", "max_stop_demand", "demand_cap_ratio",
    "provider_idx", "day_idx", "delivery_frequency",
]

PAIR_KEYS = ["provider", "plz", "base_day", "scale", "p_keep", "noise_sigma",
             "b2c_scale", "b2b_scale", "seed"]


def build_pair_index(df: pd.DataFrame) -> pd.DataFrame:
    """For each group of (PAIR_KEYS) that has multiple agg_k values,
    enumerate all (baseline_row_idx, batched_row_idx) pairs."""
    rows = []
    for keys, grp in df.groupby(PAIR_KEYS):
        if grp["agg_k"].nunique() < 2:
            continue
        baseline_rows = grp[grp["agg_k"] == grp["agg_k"].min()]
        for _, b_row in baseline_rows.iterrows():
            for _, ba_row in grp[grp["agg_k"] > grp["agg_k"].min()].iterrows():
                rows.append({
                    "baseline_idx": b_row.name,
                    "batched_idx": ba_row.name,
                    "baseline_agg_k": int(b_row["agg_k"]),
                    "batched_agg_k": int(ba_row["agg_k"]),
                    "plz": b_row["plz"],
                    "provider": b_row["provider"],
                })
    return pd.DataFrame(rows)


def make_lgb(log_target: bool):
    base = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                          n_jobs=2, random_state=42, verbosity=-1)
    return base, log_target


def make_mlp(log_target: bool, n_seeds: int = 3):
    """Smaller ensemble (3 seeds) for speed in CV — paper-grade would use 5."""
    # We return a list of pipelines for the ensemble
    pipes = []
    for s in [42, 123, 456][:n_seeds]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                                  alpha=0.01, max_iter=600, early_stopping=True,
                                  random_state=s)),
        ])
        pipes.append(pipe)
    return pipes, log_target


def fit_predict_ensemble(pipes, X_tr, y_tr, X_te):
    preds = []
    for p in pipes:
        p.fit(X_tr, y_tr)
        preds.append(p.predict(X_te))
    return np.mean(np.column_stack(preds), axis=1)


def evaluate_variant(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                       df_full: pd.DataFrame, pair_df: pd.DataFrame,
                       model_factory, log_target: bool, n_splits: int = 5) -> dict:
    """Run k-fold CV and compute cost-MAPE + saving-bias on naturally-paired samples."""
    t0 = time.time()
    gkf = GroupKFold(n_splits=n_splits)
    cost_mape, cost_bias = [], []
    sav_mae, sav_bias, sav_rmse = [], [], []
    n_pairs_total = 0
    folds = []

    # Pre-compute pair test eligibility: a pair is only evaluable if BOTH baseline and batched
    # rows are in the same test fold.
    for fold_i, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        te_set = set(te)
        factory_out = model_factory()
        # Detect: LGB returns (single_model, log_flag); MLP returns (list_of_pipelines, log_flag)
        first = factory_out[0]
        is_ensemble = isinstance(first, list)

        y_train = np.log1p(y[tr]) if log_target else y[tr]
        if is_ensemble:
            pipes = first
            for p in pipes:
                p.fit(X[tr], y_train)
            cost_pred = np.mean([np.expm1(p.predict(X[te])) if log_target else p.predict(X[te])
                                  for p in pipes], axis=0)
            all_pred = np.mean([np.expm1(p.predict(X)) if log_target else p.predict(X)
                                 for p in pipes], axis=0)
        else:
            mdl = first
            mdl.fit(X[tr], y_train)
            raw_pred_te = mdl.predict(X[te])
            cost_pred = np.expm1(raw_pred_te) if log_target else raw_pred_te
            raw_pred_all = mdl.predict(X)
            all_pred = np.expm1(raw_pred_all) if log_target else raw_pred_all

        actual_te = y[te]
        # Cost-MAPE
        denom = np.maximum(1.0, actual_te)
        cost_mape.append(np.mean(np.abs(actual_te - cost_pred) / denom) * 100)
        cost_bias.append(np.mean(cost_pred - actual_te))

        # Saving-pairs: filter pairs where BOTH rows are in test fold
        pairs_in_fold = pair_df[pair_df["baseline_idx"].isin(te_set) & pair_df["batched_idx"].isin(te_set)]
        n_pairs_total += len(pairs_in_fold)
        if len(pairs_in_fold) >= 5:
            pred_base = all_pred[pairs_in_fold["baseline_idx"].values]
            pred_batch = all_pred[pairs_in_fold["batched_idx"].values]
            actual_base = y[pairs_in_fold["baseline_idx"].values]
            actual_batch = y[pairs_in_fold["batched_idx"].values]
            actual_sav = 100 * (actual_base - actual_batch) / np.maximum(actual_base, 1)
            pred_sav = 100 * (pred_base - pred_batch) / np.maximum(pred_base, 1)
            err = pred_sav - actual_sav
            sav_mae.append(np.mean(np.abs(err)))
            sav_bias.append(np.mean(err))
            sav_rmse.append(np.sqrt(np.mean(err ** 2)))
        else:
            sav_mae.append(np.nan); sav_bias.append(np.nan); sav_rmse.append(np.nan)

        folds.append({
            "variant": name,
            "fold": fold_i + 1,
            "n_test": len(te),
            "n_pairs_test": int(len(pairs_in_fold)),
            "cost_mape_pct": cost_mape[-1],
            "cost_bias_eur": cost_bias[-1],
            "sav_mae_pp": sav_mae[-1],
            "sav_bias_pp": sav_bias[-1],
            "sav_rmse_pp": sav_rmse[-1],
        })

    elapsed = time.time() - t0
    return {
        "variant": name,
        "log_target": log_target,
        "n_folds": n_splits,
        "n_pairs_total": int(n_pairs_total),
        "cost_mape_mean": float(np.mean(cost_mape)),
        "cost_mape_std": float(np.std(cost_mape)),
        "cost_bias_eur_mean": float(np.mean(cost_bias)),
        "sav_mae_pp_mean": float(np.nanmean(sav_mae)),
        "sav_mae_pp_std": float(np.nanstd(sav_mae)),
        "sav_bias_pp_mean": float(np.nanmean(sav_bias)),
        "sav_rmse_pp_mean": float(np.nanmean(sav_rmse)),
        "fit_seconds": float(elapsed),
        "folds_detail": folds,
    }


def fig_DB1_comparison(results: list[dict], out_path: Path):
    """Two-panel: Cost-MAPE + Saving-MAE per variant."""
    names = [r["variant"] for r in results]
    cost_mapes = [r["cost_mape_mean"] for r in results]
    cost_stds = [r["cost_mape_std"] for r in results]
    sav_maes = [r["sav_mae_pp_mean"] for r in results]
    sav_stds = [r["sav_mae_pp_std"] for r in results]
    sav_bias = [r["sav_bias_pp_mean"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    xs = np.arange(len(names))
    colors = ["#666", "#1f77b4", "#bbb", "#cb181d"]
    ax.bar(xs, cost_mapes, yerr=cost_stds, capsize=6, color=colors, edgecolor="k", lw=0.5)
    for i, v in enumerate(cost_mapes):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Cost-MAPE  [%]")
    ax.set_title("(a) Cost-MAPE (lower = better)", loc="left", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(xs, sav_maes, yerr=sav_stds, capsize=6, color=colors, edgecolor="k", lw=0.5)
    for i, v in enumerate(sav_maes):
        if not np.isnan(v):
            ax.text(i, v, f"{v:.2f} pp", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Saving-MAE  [pp]")
    ax.set_title("(b) Saving-MAE on natural batching pairs", loc="left", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    ax.bar(xs, sav_bias, color=colors, edgecolor="k", lw=0.5)
    for i, v in enumerate(sav_bias):
        if not np.isnan(v):
            ax.text(i, v, f"{v:+.2f} pp", ha="center",
                      va="bottom" if v > 0 else "top", fontsize=9, fontweight="bold")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Saving-Bias  [pp]  (predicted − actual)")
    ax.set_title("(c) Saving-Bias (closer to 0 = better)", loc="left", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Fig DB1 — Model-Level Debiasing: 4-Variant Comparison  (5-fold GroupKFold over PLZ)",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_report(results: list[dict], n_train: int, n_pairs: int, out_path: Path):
    lines = ["# Model-Level Debiasing — Target-Transformation Experiment\n"]
    lines.append(f"Setup: {n_train} Trainings-Rows, {n_pairs} natuerliche Batching-Pairs (innerhalb training_matrix), 5-fold GroupKFold ueber PLZ.\n")
    lines.append("Frage: Reduziert eine Log-Target-Transformation den Saving-Bias schon waehrend des Trainings?\n")

    lines.append("## Ergebnisse\n")
    lines.append("| Variant | Target | Cost-MAPE | Saving-MAE | Saving-Bias | Saving-RMSE | Fit-Time |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in results:
        target = "log1p" if r["log_target"] else "raw"
        sav_mae = f"{r['sav_mae_pp_mean']:.2f} pp" if not np.isnan(r['sav_mae_pp_mean']) else "n/a"
        sav_bias = f"{r['sav_bias_pp_mean']:+.2f} pp" if not np.isnan(r['sav_bias_pp_mean']) else "n/a"
        sav_rmse = f"{r['sav_rmse_pp_mean']:.2f} pp" if not np.isnan(r['sav_rmse_pp_mean']) else "n/a"
        lines.append(
            f"| **{r['variant']}** | {target} | {r['cost_mape_mean']:.2f}% +/- {r['cost_mape_std']:.2f} | "
            f"{sav_mae} | {sav_bias} | {sav_rmse} | {r['fit_seconds']:.1f}s |"
        )
    lines.append("")

    # Ranking
    by_sav_bias = sorted([r for r in results if not np.isnan(r['sav_bias_pp_mean'])],
                            key=lambda r: abs(r['sav_bias_pp_mean']))
    by_sav_mae = sorted([r for r in results if not np.isnan(r['sav_mae_pp_mean'])],
                            key=lambda r: r['sav_mae_pp_mean'])

    lines.append("## Ranking nach |Saving-Bias|\n")
    for i, r in enumerate(by_sav_bias, 1):
        lines.append(f"{i}. **{r['variant']}** ({'log' if r['log_target'] else 'raw'}): "
                       f"bias = {r['sav_bias_pp_mean']:+.2f} pp, MAE = {r['sav_mae_pp_mean']:.2f} pp")
    lines.append("")

    lines.append("## Interpretation\n")
    raw_lgb = next((r for r in results if r["variant"] == "LGB-raw"), None)
    log_lgb = next((r for r in results if r["variant"] == "LGB-log"), None)
    raw_mlp = next((r for r in results if r["variant"] == "MLP-raw"), None)
    log_mlp = next((r for r in results if r["variant"] == "MLP-log"), None)

    if raw_lgb and log_lgb:
        delta = abs(log_lgb["sav_bias_pp_mean"]) - abs(raw_lgb["sav_bias_pp_mean"])
        if delta < -0.5:
            lines.append(f"- **LGB**: Log-Target reduziert |bias| um {-delta:.2f} pp (von {abs(raw_lgb['sav_bias_pp_mean']):.2f} auf {abs(log_lgb['sav_bias_pp_mean']):.2f}). Wirkt.")
        elif delta > 0.5:
            lines.append(f"- **LGB**: Log-Target erhoeht |bias| um {delta:.2f} pp. Schlechter.")
        else:
            lines.append(f"- **LGB**: Log-Target aendert |bias| nur marginal (Δ = {delta:+.2f} pp).")

    if raw_mlp and log_mlp:
        delta = abs(log_mlp["sav_bias_pp_mean"]) - abs(raw_mlp["sav_bias_pp_mean"])
        if delta < -0.5:
            lines.append(f"- **MLP**: Log-Target reduziert |bias| um {-delta:.2f} pp (von {abs(raw_mlp['sav_bias_pp_mean']):.2f} auf {abs(log_mlp['sav_bias_pp_mean']):.2f}). Wirkt.")
        elif delta > 0.5:
            lines.append(f"- **MLP**: Log-Target erhoeht |bias| um {delta:.2f} pp. Schlechter.")
        else:
            lines.append(f"- **MLP**: Log-Target aendert |bias| nur marginal (Δ = {delta:+.2f} pp).")
    lines.append("")

    lines.append("## Empfehlung\n")
    best = by_sav_bias[0] if by_sav_bias else None
    if best:
        lines.append(f"- Beste Variante (kleinster |bias|): **{best['variant']}** mit "
                       f"bias = {best['sav_bias_pp_mean']:+.2f} pp, MAE = {best['sav_mae_pp_mean']:.2f} pp")
        if best["variant"] != "MLP-raw":
            lines.append(f"- Aktuelle Production ist MLP-raw. Switching to **{best['variant']}** "
                           f"koennte die Saving-Prediction-Qualitaet verbessern, ohne Calibration-Layer.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    train_csv = args.train_csv
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    assert train_csv.exists(), (
        f"training matrix not found: {train_csv}. Pass --train-csv explicitly "
        f"(default is {DEFAULT_TRAIN_CSV})."
    )

    print(f"Loading training_matrix from {train_csv} ...")
    tm = pd.read_csv(train_csv, dtype={"plz": str})
    tm = tm.dropna(subset=BASE_FEATURES + ["actual_cost_eur"]).reset_index(drop=True)
    print(f"  {len(tm)} rows after dropna")

    print("Building natural batching-pairs index...")
    pair_df = build_pair_index(tm)
    print(f"  {len(pair_df)} pairs covering {pair_df['plz'].nunique()} PLZ")
    pair_df.to_csv(out / "tab_pair_index.csv", index=False)

    X = tm[BASE_FEATURES].to_numpy()
    y = tm["actual_cost_eur"].to_numpy()
    groups = tm["plz"].to_numpy()

    results = []
    folds_all = []

    if LGB_AVAILABLE:
        print("\nEvaluating LGB-raw...")
        r = evaluate_variant("LGB-raw", X, y, groups, tm, pair_df,
                              model_factory=lambda: make_lgb(False),
                              log_target=False)
        results.append(r); folds_all.extend(r["folds_detail"])
        print(f"  cost-MAPE {r['cost_mape_mean']:.2f}%, sav-bias {r['sav_bias_pp_mean']:+.2f} pp, sav-MAE {r['sav_mae_pp_mean']:.2f} pp")

        print("\nEvaluating LGB-log...")
        r = evaluate_variant("LGB-log", X, y, groups, tm, pair_df,
                              model_factory=lambda: make_lgb(True),
                              log_target=True)
        results.append(r); folds_all.extend(r["folds_detail"])
        print(f"  cost-MAPE {r['cost_mape_mean']:.2f}%, sav-bias {r['sav_bias_pp_mean']:+.2f} pp, sav-MAE {r['sav_mae_pp_mean']:.2f} pp")

    print("\nEvaluating MLP-raw (3 seeds)...")
    r = evaluate_variant("MLP-raw", X, y, groups, tm, pair_df,
                          model_factory=lambda: make_mlp(False, n_seeds=3),
                          log_target=False)
    results.append(r); folds_all.extend(r["folds_detail"])
    print(f"  cost-MAPE {r['cost_mape_mean']:.2f}%, sav-bias {r['sav_bias_pp_mean']:+.2f} pp, sav-MAE {r['sav_mae_pp_mean']:.2f} pp")

    print("\nEvaluating MLP-log (3 seeds)...")
    r = evaluate_variant("MLP-log", X, y, groups, tm, pair_df,
                          model_factory=lambda: make_mlp(True, n_seeds=3),
                          log_target=True)
    results.append(r); folds_all.extend(r["folds_detail"])
    print(f"  cost-MAPE {r['cost_mape_mean']:.2f}%, sav-bias {r['sav_bias_pp_mean']:+.2f} pp, sav-MAE {r['sav_mae_pp_mean']:.2f} pp")

    pd.DataFrame(results).drop(columns=["folds_detail"]).to_csv(out / "tab_variant_comparison.csv", index=False)
    pd.DataFrame(folds_all).to_csv(out / "tab_per_fold.csv", index=False)

    print("\nRendering figure...")
    fig_DB1_comparison(results, out / "fig_DB1_variant_comparison.png")

    write_report(results, n_train=len(tm), n_pairs=len(pair_df), out_path=out / "REPORT.md")
    print(f"\nAll outputs in {out}")


if __name__ == "__main__":
    main()
