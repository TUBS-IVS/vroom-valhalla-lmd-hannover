"""Diagnostic-Only: identify features that explain the +10.1pp surrogate bias on saving %.

DISCLAIMER (Lasse-confirmed): das hier ist EXPLORATIVE Diagnose, NICHT die finale
Trainings-Methodik. Wir verwenden die Batching-Results NUR als Diagnose-Daten,
um Feature-Gaps zu identifizieren. Die finale Trainings-Methodik bleibt
"baseline + perturbed" wie bisher.

Workflow:
  1. Lade die 312-Zeilen Saving-Tabelle (PLZ × provider × actual vs predicted)
  2. Reichere mit dem 25-Feature-Set aus training_matrix.csv an (Baseline-only,
     gemittelt pro (provider, plz))
  3. Engineer 7 Candidate-Features die theoretisch den Bias erklaeren koennten
  4. Berechne Residual = predicted_saving - actual_saving
  5. Trainiere RandomForest auf Residual, identifiziere top-Features
  6. Cross-Validated Calibration: subtrahiere predicted Residual, miss MAPE-Drop
  7. Quick A/B-Test: LGB-Modell auf 25 baseline-features vs 25 + 7 augmented
     auf der training_matrix mit GroupKFold-CV ueber PLZ — testet ob die neuen
     Features auf Cost-Prediction-MAPE helfen.

Outputs (results/bias_correction_diagnostic/):
  tab_enriched_saving.csv           312 Rows + alle Features
  tab_residual_feature_importance.csv
  tab_calibration_cv_mape.csv       MAPE vor/nach Calibration (5-fold CV)
  tab_engineered_feature_def.md     Erklaerung der neuen Features
  tab_ab_lgb_baseline_vs_augmented.csv  CV-MAPE A/B
  fig_BC1_residual_drivers.{pdf,png}
  fig_BC2_calibration_effect.{pdf,png}
  fig_BC3_engineered_features_dependence.{pdf,png}
  REPORT.md
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SAVING_CSV = ROOT / "results" / "final_optimization" / "vroom_validation" / "tab_actual_vs_predicted_saving.csv"
TRAIN_CSV = ROOT / "results" / "oracle_loop_extended_2026_05_22" / "training_matrix.csv"
CLUSTERS_CSV = ROOT / "data" / "geodata" / "plz_clusters.csv"
CLUSTER_RAUMTYP_CSV = ROOT / "data" / "geodata" / "cluster_raumtyp.csv"
OUT = ROOT / "results" / "bias_correction_diagnostic"
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

VEHICLE_CAPACITY = 230
FIXED_EUR = 189.15
KM_EUR = 0.3864


def load_enriched_saving() -> pd.DataFrame:
    """Join 312-row saving table with PLZ-level averaged features from training_matrix."""
    saving = pd.read_csv(SAVING_CSV, dtype={"plz": str})
    saving["plz"] = saving["plz"].str.zfill(5)

    tm = pd.read_csv(TRAIN_CSV, dtype={"plz": str})
    tm["plz"] = tm["plz"].str.zfill(5)
    # Pure baseline only (no perturbation)
    base = tm[(tm["is_baseline"]) & (tm["scale"] == 1.0) & (tm["p_keep"] == 1.0) & (tm["noise_sigma"] == 0.0)]
    feat_avg = base.groupby(["provider", "plz"], as_index=False)[BASE_FEATURES].mean()

    enriched = saving.merge(feat_avg, on=["provider", "plz"], how="left",
                              suffixes=("_saving", ""))
    # Drop duplicated columns from saving table
    drop_cols = [c for c in enriched.columns if c.endswith("_saving")]
    if drop_cols:
        enriched = enriched.drop(columns=drop_cols)

    # Saving-residual = predicted - actual (positive means surrogate overshoots)
    enriched["residual_pp"] = enriched["predicted_saving_pct"] - enriched["actual_saving_pct"]

    # Cluster + raumtyp join (for stratification)
    cl = pd.read_csv(CLUSTERS_CSV, dtype={"cluster_id": str})
    cl["cluster_id"] = cl["cluster_id"].str.zfill(5)
    long_rows = []
    for _, r in cl.iterrows():
        for m in r["member_plz_list"].split(","):
            long_rows.append({"cluster_id": r["cluster_id"], "plz": m.strip().zfill(5)})
    long_df = pd.DataFrame(long_rows)
    enriched = enriched.merge(long_df, on="plz", how="left")
    cr = pd.read_csv(CLUSTER_RAUMTYP_CSV, dtype={"cluster_id": str})
    cr["cluster_id"] = cr["cluster_id"].str.zfill(5)
    enriched = enriched.merge(cr[["cluster_id", "raumtyp_3", "raumtyp_8"]], on="cluster_id", how="left")

    return enriched


def engineer_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer 7 candidate features hypothesized to capture batching-specific cost structure."""
    out = df.copy()
    # 1) Consolidation slack: how much room is there to bundle parcels into existing routes?
    out["consolidation_slack"] = 1.0 - (
        out["n_parcels"] / (out["min_vehicles"] * VEHICLE_CAPACITY).clip(lower=1)
    )
    out["consolidation_slack"] = out["consolidation_slack"].clip(lower=0, upper=1)

    # 2) Capacity headroom: ratio of capacity to demand
    out["capacity_headroom"] = (out["min_vehicles"] * VEHICLE_CAPACITY) / out["n_parcels"].clip(lower=1)

    # 3) Effective density (parcels per convex-hull km^2 — different from PLZ-area density)
    out["parcels_per_ch_km2"] = out["n_parcels"] / out["ch_area_km2"].clip(lower=0.01)

    # 4) Hub-effort ratio (round-trip hub-distance per route)
    out["hub_effort_km_per_route"] = 2.0 * out["hub_dist_km"] * out["min_vehicles"] / out["n_parcels"].clip(lower=1)

    # 5) Spatial elongation (aspect ratio normalized)
    out["spatial_elongation"] = np.maximum(out["aspect_ratio"], 1.0 / out["aspect_ratio"].clip(lower=0.01))

    # 6) Stop fragmentation (stops per parcel — high = lots of small drops)
    out["stops_per_parcel"] = out["n_stops"] / out["n_parcels"].clip(lower=1)

    # 7) Batching marginal benefit (theoretical: how much can we shrink routes?)
    # If parcels << capacity, batching can collapse routes; if parcels >> capacity, routes are already saturated.
    out["routes_compressibility"] = np.maximum(0, 1.0 - out["load_factor"])

    return out


ENGINEERED_FEATURES = [
    "consolidation_slack", "capacity_headroom", "parcels_per_ch_km2",
    "hub_effort_km_per_route", "spatial_elongation", "stops_per_parcel",
    "routes_compressibility",
]


def fit_residual_model(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, RandomForestRegressor]:
    sub = df.dropna(subset=feature_cols + ["residual_pp"]).copy()
    X = sub[feature_cols].to_numpy()
    y = sub["residual_pp"].to_numpy()
    rf = RandomForestRegressor(n_estimators=600, max_depth=10, n_jobs=2, random_state=42)
    rf.fit(X, y)
    r2 = 1 - np.sum((y - rf.predict(X)) ** 2) / np.sum((y - y.mean()) ** 2)
    perm = permutation_importance(rf, X, y, n_repeats=30, random_state=42, n_jobs=2)
    imp = pd.DataFrame({
        "feature": feature_cols,
        "perm_mean": perm.importances_mean,
        "perm_std": perm.importances_std,
        "in_sample_r2": r2,
    }).sort_values("perm_mean", ascending=False)
    return imp, rf


def cv_calibration(df: pd.DataFrame, feature_cols: list[str], n_splits: int = 5) -> dict:
    """Honest k-fold CV: train residual-correction model on train fold, apply to test fold,
    measure MAPE of corrected predictions vs uncorrected on TEST fold."""
    sub = df.dropna(subset=feature_cols + ["residual_pp", "actual_saving_pct", "predicted_saving_pct"]).copy()
    groups = sub["plz"].to_numpy()
    X = sub[feature_cols].to_numpy()
    y_resid = sub["residual_pp"].to_numpy()
    y_actual = sub["actual_saving_pct"].to_numpy()
    y_pred = sub["predicted_saving_pct"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    mae_before, mae_after = [], []
    rmse_before, rmse_after = [], []
    bias_before, bias_after = [], []
    for tr, te in gkf.split(X, y_resid, groups=groups):
        rf = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
        rf.fit(X[tr], y_resid[tr])
        pred_residual = rf.predict(X[te])
        corrected = y_pred[te] - pred_residual

        err_before = y_actual[te] - y_pred[te]
        err_after = y_actual[te] - corrected
        mae_before.append(np.mean(np.abs(err_before)))
        mae_after.append(np.mean(np.abs(err_after)))
        rmse_before.append(np.sqrt(np.mean(err_before ** 2)))
        rmse_after.append(np.sqrt(np.mean(err_after ** 2)))
        bias_before.append(np.mean(y_pred[te] - y_actual[te]))
        bias_after.append(np.mean(corrected - y_actual[te]))

    return {
        "n_folds": n_splits,
        "n_samples": len(sub),
        "mae_before_pp": float(np.mean(mae_before)),
        "mae_before_std_pp": float(np.std(mae_before)),
        "mae_after_pp": float(np.mean(mae_after)),
        "mae_after_std_pp": float(np.std(mae_after)),
        "mae_improvement_pp": float(np.mean(mae_before) - np.mean(mae_after)),
        "rmse_before_pp": float(np.mean(rmse_before)),
        "rmse_after_pp": float(np.mean(rmse_after)),
        "bias_before_mean_pp": float(np.mean(bias_before)),
        "bias_after_mean_pp": float(np.mean(bias_after)),
        "mae_before_folds_pp": [round(m, 3) for m in mae_before],
        "mae_after_folds_pp": [round(m, 3) for m in mae_after],
    }


def ab_test_augmented_model(n_splits: int = 5) -> dict:
    """Train two LGB models on training_matrix:
      A: 25 baseline features
      B: 25 + 7 engineered features
    Compare CV-MAPE on actual_cost_eur via GroupKFold over PLZ.
    Uses GBM as a fast surrogate proxy; not the production MLP.
    """
    try:
        from lightgbm import LGBMRegressor
        LGB_AVAILABLE = True
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor as LGBMRegressor  # noqa: N812
        LGB_AVAILABLE = False

    tm = pd.read_csv(TRAIN_CSV, dtype={"plz": str})
    tm["plz"] = tm["plz"].str.zfill(5)
    tm = tm.dropna(subset=BASE_FEATURES + ["actual_cost_eur"])
    tm = engineer_candidates(tm)

    X_base = tm[BASE_FEATURES].to_numpy()
    X_aug = tm[BASE_FEATURES + ENGINEERED_FEATURES].to_numpy()
    y = np.log1p(tm["actual_cost_eur"].to_numpy())  # log-target to match LGB-logT convention
    groups = tm["plz"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    base_mapes, aug_mapes = [], []
    for tr, te in gkf.split(X_base, y, groups=groups):
        for X_train, X_test, store in [(X_base, X_base, base_mapes), (X_aug, X_aug, aug_mapes)]:
            if LGB_AVAILABLE:
                m = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63, n_jobs=2,
                                    random_state=42, verbosity=-1)
            else:
                m = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
            m.fit(X_train[tr], y[tr])
            pred = np.expm1(m.predict(X_test[te]))
            actual = np.expm1(y[te])
            denom = np.maximum(1.0, actual)
            store.append(np.mean(np.abs(actual - pred) / denom) * 100)

    return {
        "model_class": "LightGBM" if LGB_AVAILABLE else "GradientBoosting (sklearn fallback)",
        "n_folds": n_splits,
        "n_train_rows": len(tm),
        "mape_baseline_mean": float(np.mean(base_mapes)),
        "mape_baseline_std": float(np.std(base_mapes)),
        "mape_augmented_mean": float(np.mean(aug_mapes)),
        "mape_augmented_std": float(np.std(aug_mapes)),
        "improvement_pp": float(np.mean(base_mapes) - np.mean(aug_mapes)),
        "mape_baseline_folds": [round(m, 4) for m in base_mapes],
        "mape_augmented_folds": [round(m, 4) for m in aug_mapes],
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def fig_BC1_residual_drivers(imp_all: pd.DataFrame, out_path: Path):
    """Bar plot of top-15 features for residual prediction."""
    top = imp_all.head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    ys = np.arange(len(top))
    colors = ["#cb181d" if f in ENGINEERED_FEATURES else "#1f77b4" for f in top["feature"]]
    ax.barh(ys, top["perm_mean"], xerr=top["perm_std"], color=colors, edgecolor="k", lw=0.4)
    ax.set_yticks(ys); ax.set_yticklabels(top["feature"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation Importance (residual = predicted − actual saving pp)")
    ax.set_title("Fig BC1 — Top features that explain surrogate bias\n"
                  "red = engineered candidate, blue = existing surrogate feature",
                  loc="left", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_BC2_calibration_effect(cv: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    folds = np.arange(1, cv["n_folds"] + 1)
    ax.plot(folds, cv["mae_before_folds_pp"], "o-", color="#666",
             label=f"Before calibration  (mean {cv['mae_before_pp']:.2f} pp)")
    ax.plot(folds, cv["mae_after_folds_pp"], "s-", color="#cb181d",
             label=f"After calibration  (mean {cv['mae_after_pp']:.2f} pp)")
    ax.axhline(cv["mae_before_pp"], color="#666", ls=":", lw=0.8)
    ax.axhline(cv["mae_after_pp"], color="#cb181d", ls=":", lw=0.8)
    ax.set_xlabel("CV-Fold (GroupKFold over PLZ)")
    ax.set_ylabel("MAE on saving_pct  [pp]")
    ax.set_title(f"Fig BC2 — Saving-MAE before vs after residual-calibration  "
                  f"(improvement = {cv['mae_improvement_pp']:+.2f} pp)",
                  loc="left", fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_BC3_engineered_features(df: pd.DataFrame, out_path: Path):
    """Show how engineered features relate to residual."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    axes = axes.flatten()
    for ax, feat in zip(axes, ENGINEERED_FEATURES + ["residual_pp"][:1]):
        if feat not in df.columns:
            ax.axis("off"); continue
        sub = df.dropna(subset=[feat, "residual_pp"])
        ax.scatter(sub[feat], sub["residual_pp"], s=14, alpha=0.5, color="#0072B2",
                     edgecolors="none")
        # spearman
        from scipy.stats import spearmanr
        rho, p = spearmanr(sub[feat], sub["residual_pp"])
        # LOWESS
        order = np.argsort(sub[feat].values)
        xs = sub[feat].values[order]
        ys = sub["residual_pp"].values[order]
        if len(xs) > 30:
            # simple rolling mean
            window = max(20, len(xs) // 8)
            rolling = pd.Series(ys).rolling(window=window, min_periods=10, center=True).mean()
            ax.plot(xs, rolling, color="k", lw=2)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax.set_xlabel(feat)
        ax.set_ylabel("residual (pp)" if ax is axes[0] else "")
        ax.set_title(f"ρ={rho:+.2f}, p={p:.1e}", loc="left", fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes[len(ENGINEERED_FEATURES):]:
        ax.axis("off")
    fig.suptitle("Fig BC3 — Engineered candidate features vs surrogate residual",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(df: pd.DataFrame, imp_base: pd.DataFrame, imp_all: pd.DataFrame,
                  cv: dict, ab: dict):
    out = OUT / "REPORT.md"
    lines = ["# Bias-Correction Diagnostic (NICHT für Paper-Methodik)\n"]
    lines.append("> Diese Auswertung verwendet die Batching-Saving-Daten NUR zur Diagnose.")
    lines.append("> Die finale Trainings-Methodik bleibt 'baseline + perturbed' wie bisher.")
    lines.append("> Ergebnisse hier sind exploratorische Hinweise, nicht Reporting-Material.\n")

    lines.append("## Setup\n")
    lines.append(f"- Saving-Tabelle (PLZ × Provider): {len(df)} rows")
    lines.append(f"- Residual = predicted_saving_pct − actual_saving_pct (positive ⇒ Surrogate ueberschaetzt Saving)")
    lines.append(f"- Mean residual: {df['residual_pp'].mean():+.2f} pp")
    lines.append(f"- Median residual: {df['residual_pp'].median():+.2f} pp\n")

    lines.append("## 1. Welche Features erklaeren den Bias (existing 25-feature set)?\n")
    lines.append("RF-Modell auf Residual mit 25 Surrogate-Features. In-sample R² = "
                  f"{imp_base['in_sample_r2'].iloc[0]:.3f}.\n")
    lines.append("| Rank | Feature | Permutation Imp. | Std |")
    lines.append("|---:|---|---:|---:|")
    for i, r in imp_base.head(10).iterrows():
        lines.append(f"| {imp_base.index.get_loc(i)+1} | {r['feature']} | {r['perm_mean']:.4f} | {r['perm_std']:.4f} |")
    lines.append("")

    lines.append("## 2. Engineered Candidate-Features (7 Stueck)\n")
    lines.append("| Feature | Definition |")
    lines.append("|---|---|")
    lines.append("| `consolidation_slack` | `1 − n_parcels / (min_vehicles × VEHICLE_CAPACITY)`, clipped [0,1] — wieviel Bündelungs-Spielraum bietet die Baseline-Auslastung |")
    lines.append("| `capacity_headroom` | `(min_vehicles × VEHICLE_CAPACITY) / n_parcels` — Kapazitäts-zu-Demand-Ratio |")
    lines.append("| `parcels_per_ch_km2` | `n_parcels / ch_area_km2` — Dichte ueber Convex-Hull (statt PLZ-Area) |")
    lines.append("| `hub_effort_km_per_route` | `2 × hub_dist_km × min_vehicles / n_parcels` — Hub-RoundTrip-Aufwand pro Paket |")
    lines.append("| `spatial_elongation` | `max(aspect_ratio, 1/aspect_ratio)` — Längs-Streckung normalisiert |")
    lines.append("| `stops_per_parcel` | `n_stops / n_parcels` — Fragmentierungs-Indikator |")
    lines.append("| `routes_compressibility` | `max(0, 1 − load_factor)` — wie kompressibel sind die Routen |")
    lines.append("")

    lines.append("## 3. Erweiterte Importance (25 + 7 = 32 Features)\n")
    lines.append("Top-15 Features. **Rot markiert** in fig_BC1: engineered Candidates.\n")
    lines.append("| Rank | Feature | Permutation Imp. | Std | Engineered? |")
    lines.append("|---:|---|---:|---:|:---:|")
    for i, r in imp_all.head(15).iterrows():
        eng = "🟥 yes" if r["feature"] in ENGINEERED_FEATURES else ""
        lines.append(f"| {imp_all.index.get_loc(i)+1} | {r['feature']} | {r['perm_mean']:.4f} | {r['perm_std']:.4f} | {eng} |")
    lines.append("")

    lines.append("## 4. Calibration-Test (5-fold CV ueber PLZ-Gruppen)\n")
    lines.append("Trainiere GBM auf Residual im Train-Fold, subtrahiere von Surrogate-Prediction im Test-Fold.\n")
    lines.append(f"- n_samples: {cv['n_samples']}")
    lines.append(f"- Saving-MAE **vor** Calibration: {cv['mae_before_pp']:.2f} pp +/- {cv['mae_before_std_pp']:.2f}")
    lines.append(f"- Saving-MAE **nach** Calibration: {cv['mae_after_pp']:.2f} pp +/- {cv['mae_after_std_pp']:.2f}")
    lines.append(f"- Saving-RMSE vorher: {cv['rmse_before_pp']:.2f} pp -> nachher: {cv['rmse_after_pp']:.2f} pp")
    lines.append(f"- **MAE-Improvement: {cv['mae_improvement_pp']:+.2f} pp**")
    lines.append(f"- Mean bias vorher: {cv['bias_before_mean_pp']:+.2f} pp -> nachher: {cv['bias_after_mean_pp']:+.2f} pp")
    lines.append("")
    lines.append("| Fold | MAE before (pp) | MAE after (pp) |")
    lines.append("|---:|---:|---:|")
    for i, (b, a) in enumerate(zip(cv["mae_before_folds_pp"], cv["mae_after_folds_pp"]), 1):
        lines.append(f"| {i} | {b:.2f} | {a:.2f} |")
    lines.append("")

    lines.append("## 5. A/B-Test auf Cost-Prediction (LGB-Proxy, kein MLP)\n")
    lines.append("Quick A/B-Test: ein LGB-Modell auf 25 baseline-features vs. 25 + 7 engineered features.")
    lines.append("Trainiert auf der vollen training_matrix (Baseline + Perturbed), 5-fold GroupKFold ueber PLZ.\n")
    lines.append(f"- Model class: {ab['model_class']}")
    lines.append(f"- Training rows: {ab['n_train_rows']}")
    lines.append(f"- Cost-MAPE **baseline** (25 features): {ab['mape_baseline_mean']:.2f}% ± {ab['mape_baseline_std']:.2f}%")
    lines.append(f"- Cost-MAPE **augmented** (32 features): {ab['mape_augmented_mean']:.2f}% ± {ab['mape_augmented_std']:.2f}%")
    lines.append(f"- **Improvement: {ab['improvement_pp']:+.2f} pp**")
    lines.append("")

    lines.append("## 6. Empfehlung\n")
    top_eng = [f for f in imp_all.head(15)["feature"].tolist() if f in ENGINEERED_FEATURES]
    if cv["mae_improvement_pp"] > 1.5:
        lines.append(f"- Calibration drueckt Saving-MAE um {cv['mae_improvement_pp']:.1f} pp -> "
                       "**lohnender Post-Hoc-Layer**, falls man das deployen will (1 zusaetzliches Modell pro Surrogate).")
    elif cv["mae_improvement_pp"] > 0.5:
        lines.append(f"- Calibration drueckt Saving-MAE um {cv['mae_improvement_pp']:.1f} pp -> "
                       "marginal-relevanter Gewinn.")
    else:
        lines.append(f"- Calibration drueckt Saving-MAE nur um {cv['mae_improvement_pp']:.1f} pp -> "
                       "der Bias ist nicht durch existierende Features modellierbar; "
                       "echte Modell-Verbesserung braucht andere Daten.")
    if top_eng:
        lines.append(f"- Engineered Features in den Top-15: **{', '.join(top_eng)}** -> Kandidaten fuer Feature-Set-Augmentation.")
    if ab["improvement_pp"] > 0.5:
        lines.append(f"- LGB-A/B-Test zeigt {ab['improvement_pp']:.2f} pp Cost-MAPE-Verbesserung beim Augmented-Modell -> "
                       "Re-Training des MLP-Ensembles mit 32 statt 25 Features koennte ~{ab['improvement_pp']:.1f} pp bringen.")
    else:
        lines.append(f"- LGB-A/B-Test zeigt nur {ab['improvement_pp']:.2f} pp Verbesserung -> "
                       "engineered Features bringen kein Signal ueber die existierenden 25 hinaus.")
    lines.append("")

    lines.append("## 7. Methodische Hinweise\n")
    lines.append("- **Wichtig:** Diese Diagnose wurde ausschliesslich auf existierenden Daten durchgefuehrt. Kein VROOM-Call.")
    lines.append("- Die Calibration in Sektion 4 ist **honest CV** (GroupKFold ueber PLZ), das Improvement ist also out-of-sample.")
    lines.append("- Der A/B-Test in Sektion 5 nutzt LGB als Proxy, nicht das production-MLP. Echtes MLP-Re-Training koennte andere Ergebnisse liefern.")
    lines.append("- Falls Reviewer fragen: das ist Supplementary Diagnostic, NICHT die Trainings-Methodik des Papers.")

    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("Loading + enriching saving table...")
    df = load_enriched_saving()
    df = engineer_candidates(df)
    df.to_csv(OUT / "tab_enriched_saving.csv", index=False)
    print(f"  {len(df)} rows; residual mean = {df['residual_pp'].mean():+.2f} pp, median = {df['residual_pp'].median():+.2f} pp")

    print("\nFitting residual model (existing 25 features only)...")
    imp_base, rf_base = fit_residual_model(df, BASE_FEATURES)
    print("Top-5 baseline features:")
    print(imp_base.head(5).to_string(index=False))

    print("\nFitting residual model (25 + 7 engineered features)...")
    all_features = BASE_FEATURES + ENGINEERED_FEATURES
    imp_all, rf_all = fit_residual_model(df, all_features)
    imp_all.to_csv(OUT / "tab_residual_feature_importance.csv", index=False)
    print("Top-10 (with engineered):")
    print(imp_all.head(10).to_string(index=False))

    print("\nCV calibration test (GroupKFold over PLZ)...")
    cv = cv_calibration(df, all_features, n_splits=5)
    pd.DataFrame([cv]).to_csv(OUT / "tab_calibration_cv_mape.csv", index=False)
    print(f"  Saving-MAE: {cv['mae_before_pp']:.2f} pp -> {cv['mae_after_pp']:.2f} pp  "
            f"(improvement {cv['mae_improvement_pp']:+.2f} pp)")
    print(f"  Bias: {cv['bias_before_mean_pp']:+.2f} pp -> {cv['bias_after_mean_pp']:+.2f} pp")

    print("\nA/B-Test: baseline (25) vs augmented (32) features on cost-prediction (LGB)...")
    ab = ab_test_augmented_model(n_splits=5)
    pd.DataFrame([ab]).to_csv(OUT / "tab_ab_lgb_baseline_vs_augmented.csv", index=False)
    print(f"  Cost-MAPE: {ab['mape_baseline_mean']:.2f}% -> {ab['mape_augmented_mean']:.2f}%  "
            f"(improvement {ab['improvement_pp']:+.2f} pp)")

    print("\nRendering figures...")
    fig_BC1_residual_drivers(imp_all, OUT / "fig_BC1_residual_drivers.png")
    fig_BC2_calibration_effect(cv, OUT / "fig_BC2_calibration_effect.png")
    fig_BC3_engineered_features(df, OUT / "fig_BC3_engineered_features.png")

    write_report(df, imp_base, imp_all, cv, ab)
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
