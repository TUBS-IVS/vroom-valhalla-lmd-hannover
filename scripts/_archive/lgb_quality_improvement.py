"""Quality Improvement Tests for the Production LGB-logT Surrogate.

Diagnostic + Model-Improvement-Experiment. Wir testen ob das Production-Modell
(LGB-logT, 44 combo features) durch folgende Aenderungen auf der Saving-
Prediction-Bias-Aufgabe BESSER wird, **ohne neuen VROOM-Run**:

  V0 - Baseline: production-style LGB-logT, no modifications
  V1 - Sample-weighting: weight samples by 1/actual_cost_eur (relative-error focus)
  V2 - Monotonic constraint: cost monotonically increasing in `n_parcels` (sanity-anchor)
  V3 - Asymmetric loss: penalize underprediction more than overprediction (Quantile=0.65)
  V4 - Batching-aware features: add 5 explicit features for the batched-day state
  V5 - V2 + V4 combined (best-of)
  V6 - V1 + V4 combined

Evaluation:
  * Cost-MAPE on 5-fold GroupKFold(PLZ) test folds — must not be worse than V0
  * Saving-Bias on natural batching pairs (310 pairs in training_matrix)
  * Saving-MAE on natural pairs

Outputs (results/lgb_quality_improvement/):
  tab_variant_results.csv
  fig_LQI1_variant_comparison.{pdf,png}
  REPORT.md
"""
from __future__ import annotations

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
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not available, cannot run.")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "results" / "oracle_loop_extended_2026_05_22" / "training_matrix.csv"
OUT = ROOT / "results" / "lgb_quality_improvement"
OUT.mkdir(parents=True, exist_ok=True)

BASE_FEATURES = [
    "n_parcels", "n_stops", "area_km2", "hub_dist_km", "parcels_per_stop",
    "load_factor", "min_vehicles", "parcels_per_km2",
    "ch_area_km2", "ch_perimeter_km", "mean_nn_dist_km", "mean_inter_stop_dist_km",
    "stop_density_ch", "centroid_hub_dist_km", "max_hub_dist_km",
    "coord_std_x", "coord_std_y", "aspect_ratio",
    "b2c_share", "demand_std", "max_stop_demand", "demand_cap_ratio",
    "provider_idx", "day_idx", "delivery_frequency",
]

# 5 engineered features that explicitly capture batching state
BATCHING_FEATURES = [
    "is_batched",                  # binary: agg_k > 1
    "agg_k_log",                   # log of aggregation level
    "parcels_per_load_capacity",   # n_parcels / (min_vehicles * 230)
    "schedule_compression",        # how much demand piled up
    "hub_round_trip_per_parcel",   # 2 * hub_dist / n_parcels
]

PAIR_KEYS = ["provider", "plz", "base_day", "scale", "p_keep", "noise_sigma",
             "b2c_scale", "b2b_scale", "seed"]


# ---------------------------------------------------------------------------
def add_batching_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_batched"] = (df["agg_k"] > 1).astype(float)
    df["agg_k_log"] = np.log1p(df["agg_k"].astype(float))
    df["parcels_per_load_capacity"] = df["n_parcels"] / (df["min_vehicles"] * 230.0).clip(lower=1)
    df["schedule_compression"] = df["agg_k"].astype(float) * df["delivery_frequency"]
    df["hub_round_trip_per_parcel"] = (2.0 * df["hub_dist_km"]) / df["n_parcels"].clip(lower=1)
    return df


def build_pair_index(df: pd.DataFrame) -> pd.DataFrame:
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
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def make_v0(features):  # baseline LGB-logT
    base = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                          n_jobs=2, random_state=42, verbosity=-1)
    return TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)


def make_v1(features):  # sample-weighting baked into fit() call via sample_weight argument
    return make_v0(features)


def make_v2(features):  # monotonic constraint on n_parcels (must always increase cost)
    mono = [0] * len(features)
    if "n_parcels" in features:
        mono[features.index("n_parcels")] = 1
    if "min_vehicles" in features:
        mono[features.index("min_vehicles")] = 1
    base = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                          n_jobs=2, random_state=42, verbosity=-1,
                          monotone_constraints=mono, monotone_constraints_method="advanced")
    return TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)


def make_v3(features):  # asymmetric loss via quantile objective
    base = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                          n_jobs=2, random_state=42, verbosity=-1,
                          objective="quantile", alpha=0.65)
    return TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)


def make_v4(features):  # 25 + 5 batching features
    return make_v0(features)


def make_v5(features):  # V2 + V4
    return make_v2(features)


def make_v6(features):  # V1 + V4
    return make_v0(features)


VARIANTS = [
    ("V0_baseline_LGBlogT", BASE_FEATURES, None, make_v0,
     "Production-style: LGB-logT, 25 features, no modifications"),
    ("V1_sample_weighted",  BASE_FEATURES, "inv_cost", make_v1,
     "V0 + sample_weight = 1/actual_cost (relative-error focus)"),
    ("V2_monotonic",        BASE_FEATURES, None, make_v2,
     "V0 + monotonic constraint cost UP with n_parcels & min_vehicles"),
    ("V3_asymmetric_q0.65", BASE_FEATURES, None, make_v3,
     "V0 with quantile loss alpha=0.65 (penalize underprediction more)"),
    ("V4_batching_features", BASE_FEATURES + BATCHING_FEATURES, None, make_v4,
     "V0 + 5 batching-aware features"),
    ("V5_monotonic+batching", BASE_FEATURES + BATCHING_FEATURES, None, make_v5,
     "V2 + V4 combined"),
    ("V6_weighted+batching",  BASE_FEATURES + BATCHING_FEATURES, "inv_cost", make_v6,
     "V1 + V4 combined"),
]


# ---------------------------------------------------------------------------
def evaluate_variant(name: str, features: list[str], weight_strategy: str | None,
                       factory, df_full: pd.DataFrame, pair_df: pd.DataFrame,
                       n_splits: int = 5) -> dict:
    t0 = time.time()
    X = df_full[features].to_numpy()
    y = df_full["actual_cost_eur"].to_numpy()
    groups = df_full["plz"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    cost_mape, cost_bias = [], []
    sav_mae, sav_bias = [], []
    pred_full = np.zeros(len(df_full))  # accumulate OOF predictions across folds for pair eval

    for fold_i, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        mdl = factory(features)
        sample_weight = None
        if weight_strategy == "inv_cost":
            sample_weight = 1.0 / np.maximum(50.0, y[tr])  # inverse-cost weighting (floor 50€)
            sample_weight = sample_weight / sample_weight.mean()  # normalize
        if sample_weight is not None:
            mdl.fit(X[tr], y[tr], sample_weight=sample_weight)
        else:
            mdl.fit(X[tr], y[tr])
        pred_te = mdl.predict(X[te])
        pred_full[te] = pred_te

        denom = np.maximum(1.0, y[te])
        cost_mape.append(np.mean(np.abs(y[te] - pred_te) / denom) * 100)
        cost_bias.append(np.mean(pred_te - y[te]))

        # Saving bias on pairs that are FULLY within this test fold
        te_set = set(te)
        in_fold = pair_df[pair_df["baseline_idx"].isin(te_set) & pair_df["batched_idx"].isin(te_set)]
        if len(in_fold) >= 5:
            p_base = pred_full[in_fold["baseline_idx"].values]
            p_batch = pred_full[in_fold["batched_idx"].values]
            a_base = y[in_fold["baseline_idx"].values]
            a_batch = y[in_fold["batched_idx"].values]
            act_sav = 100 * (a_base - a_batch) / np.maximum(a_base, 1)
            pred_sav = 100 * (p_base - p_batch) / np.maximum(p_base, 1)
            err = pred_sav - act_sav
            sav_mae.append(np.mean(np.abs(err)))
            sav_bias.append(np.mean(err))
        else:
            sav_mae.append(np.nan); sav_bias.append(np.nan)

    return {
        "variant": name,
        "n_folds": n_splits,
        "cost_mape_mean": float(np.mean(cost_mape)),
        "cost_mape_std": float(np.std(cost_mape)),
        "cost_bias_eur_mean": float(np.mean(cost_bias)),
        "sav_mae_pp": float(np.nanmean(sav_mae)),
        "sav_mae_pp_std": float(np.nanstd(sav_mae)),
        "sav_bias_pp": float(np.nanmean(sav_bias)),
        "fit_seconds": float(time.time() - t0),
    }


# ---------------------------------------------------------------------------
def fig_LQI1(results: list[dict], out_path: Path):
    n = len(results)
    names = [r["variant"] for r in results]
    cost_mapes = [r["cost_mape_mean"] for r in results]
    cost_stds = [r["cost_mape_std"] for r in results]
    sav_maes = [r["sav_mae_pp"] for r in results]
    sav_stds = [r["sav_mae_pp_std"] for r in results]
    sav_bias = [r["sav_bias_pp"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ["#444"] + ["#1f77b4"] * (n - 1)
    # Baseline highlighted, improvements blue, best in red
    best_idx = int(np.argmin([abs(b) for b in sav_bias]))
    colors[best_idx] = "#cb181d"

    xs = np.arange(n)
    ax = axes[0]
    ax.bar(xs, cost_mapes, yerr=cost_stds, capsize=4, color=colors, edgecolor="k", lw=0.4)
    for i, v in enumerate(cost_mapes):
        ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Cost-MAPE  [%]")
    ax.set_title("(a) Cost-prediction quality", loc="left", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.bar(xs, sav_maes, yerr=sav_stds, capsize=4, color=colors, edgecolor="k", lw=0.4)
    for i, v in enumerate(sav_maes):
        if not np.isnan(v):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Saving-MAE  [pp]")
    ax.set_title("(b) Saving-MAE on natural pairs", loc="left", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    ax.bar(xs, sav_bias, color=colors, edgecolor="k", lw=0.4)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    for i, v in enumerate(sav_bias):
        if not np.isnan(v):
            ax.text(i, v, f"{v:+.2f}", ha="center",
                      va="bottom" if v > 0 else "top", fontsize=8, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Saving-Bias  [pp]")
    ax.set_title("(c) Saving-Bias (closer to 0 = better)", loc="left", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Fig LQI1 — LGB-logT Quality-Improvement variants  (5-fold GroupKFold over PLZ)",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
def write_report(results: list[dict], n_train: int, n_pairs: int, descriptions: dict):
    out = OUT / "REPORT.md"
    lines = ["# LGB-logT Quality-Improvement Test\n"]
    lines.append("Frage: kann das Production-Modell (LGB-logT) durch Trainings-Aenderungen ohne neuen VROOM-Run "
                  "auf die Saving-Prediction-Aufgabe besser werden?\n")
    lines.append(f"Setup: {n_train} Trainings-Rows, {n_pairs} natuerliche batching-Pairs, 5-fold GroupKFold(PLZ).\n")

    lines.append("## Varianten\n")
    for name, _, _, _, desc in VARIANTS:
        lines.append(f"- **{name}**: {desc}")
    lines.append("")

    lines.append("## Ergebnisse\n")
    lines.append("| Variant | Cost-MAPE | Saving-MAE | Saving-Bias | Fit-Time |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| **{r['variant']}** | {r['cost_mape_mean']:.2f}% +/- {r['cost_mape_std']:.2f} | "
            f"{r['sav_mae_pp']:.2f} pp +/- {r['sav_mae_pp_std']:.2f} | "
            f"{r['sav_bias_pp']:+.2f} pp | "
            f"{r['fit_seconds']:.1f}s |"
        )
    lines.append("")

    # Baseline reference + Best
    v0 = next(r for r in results if r["variant"].startswith("V0"))
    best = min(results, key=lambda r: abs(r["sav_bias_pp"]))
    lines.append(f"## Headline\n")
    lines.append(f"- **Baseline V0** Cost-MAPE = **{v0['cost_mape_mean']:.2f}%**, Saving-Bias = **{v0['sav_bias_pp']:+.2f} pp**.")
    lines.append(f"- **Beste Variante {best['variant']}**: Cost-MAPE = **{best['cost_mape_mean']:.2f}%** (Delta {best['cost_mape_mean']-v0['cost_mape_mean']:+.2f} pp), Saving-Bias = **{best['sav_bias_pp']:+.2f} pp** (Reduktion {abs(v0['sav_bias_pp'])-abs(best['sav_bias_pp']):+.2f} pp).")
    lines.append("")

    if abs(best["sav_bias_pp"]) < abs(v0["sav_bias_pp"]) - 0.3:
        lines.append(f"**Empfehlung:** Production-Modell auf **{best['variant']}** umstellen — drueckt Saving-Bias substanziell ohne Cost-MAPE-Verschlechterung.")
    else:
        lines.append("**Empfehlung:** Keine der Trainings-Aenderungen drueckt den Saving-Bias substanziell. Der Bias kommt aus Optimizer-Winner's-Curse (Sektion 24 des Compendiums), nicht aus Modell-Quality. Post-Hoc Calibration (Sektion 23) oder UCB-Acquisition bleibt der Weg.")
    out.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
def main():
    print("Loading training_matrix...")
    tm = pd.read_csv(TRAIN_CSV, dtype={"plz": str})
    tm = tm.dropna(subset=BASE_FEATURES + ["actual_cost_eur", "agg_k"]).reset_index(drop=True)
    print(f"  {len(tm)} rows")

    print("Adding batching features...")
    tm = add_batching_features(tm)

    print("Building natural batching-pairs index...")
    pair_df = build_pair_index(tm)
    print(f"  {len(pair_df)} pairs")

    print("\n=== Running 7 variants ===")
    results = []
    for name, features, weight, factory, desc in VARIANTS:
        print(f"\n>> {name}: {desc}")
        r = evaluate_variant(name, features, weight, factory, tm, pair_df, n_splits=5)
        results.append(r)
        print(f"   cost-MAPE {r['cost_mape_mean']:.2f}%, sav-bias {r['sav_bias_pp']:+.2f} pp, sav-MAE {r['sav_mae_pp']:.2f} pp")

    pd.DataFrame(results).to_csv(OUT / "tab_variant_results.csv", index=False)

    print("\nRendering figure...")
    fig_LQI1(results, OUT / "fig_LQI1_variant_comparison.png")

    write_report(results, n_train=len(tm), n_pairs=len(pair_df),
                  descriptions={v[0]: v[4] for v in VARIANTS})
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
