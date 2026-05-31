"""Warum performt LGB-logT auf SA_ML-chosen schedules schlechter als auf
perturbed-baseline? Distribution-Shift-Diagnose.

Hypothesen:
  H1: Feature-Distribution-Shift — die SA_ML-Schedules erzeugen Feature-Werte,
      die in der training_matrix selten/nicht vorkommen (extrapolation).
  H2: Schedule-Frequency-Mismatch — Training hat agg_k in {1,2,3} weitgehend
      gleichverteilt, aber Optimizer pickt fast immer schedule_size=2 (= agg_k=3).
  H3: Hub-Bundling-Effekte — auf batched delivery days sind die per-day Feature-
      Werte (n_parcels, n_stops) viel groesser als auf Baseline → Modell muss
      extrapolieren.
  H4: Per-day variance is heteroscedastic — variance der actual cost waechst mit
      cost magnitude staerker als das Modell sieht.

Daten:
  * training_matrix.csv (11'523 rows, perturbed baseline + agg_k 1/2/3)
  * ml_vs_vroom_per_day.csv (1'283 rows, out-of-pool SA_ML + Fixed schedules)

Outputs (results/distribution_shift_diagnosis/):
  tab_feature_distribution_stats.csv
  tab_extrapolation_check.csv
  tab_per_bin_quality.csv
  fig_DS1_feature_histograms.{pdf,png}
  fig_DS2_extrapolation_map.{pdf,png}
  fig_DS3_per_bin_bias.{pdf,png}
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "distribution_shift_diagnosis"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = ROOT / "results" / "oracle_loop_extended_2026_05_22" / "training_matrix.csv"
MLV_CSV = ROOT / "results" / "final_optimization" / "ml_vs_vroom_per_day.csv"

KEY_FEATURES = [
    "n_parcels", "n_stops", "parcels_per_stop", "load_factor", "min_vehicles",
    "parcels_per_km2", "hub_dist_km", "area_km2",
]


def load_training_feature_distribution() -> pd.DataFrame:
    """Per-day features as the model sees them in training (raw per-row)."""
    tm = pd.read_csv(TRAIN_CSV)
    print(f"Training pool: {len(tm)} rows, agg_k distribution: {tm['agg_k'].value_counts().to_dict()}")
    return tm


def load_out_of_pool_states() -> pd.DataFrame:
    """For each (provider, plz, day) cell in ml_vs_vroom_per_day, reconstruct the
    per-day feature state as the LGB would see it during build_cost_matrices_ml.
    Since we don't have explicit per-(PLZ, schedule, day) features stored, we
    approximate by mapping the actual VROOM-routed n_parcels (= per-day demand
    aggregated for the schedule) to the closest training_matrix samples."""
    mlv = pd.read_csv(MLV_CSV, dtype={"plz": str})
    mlv = mlv[mlv["delivers_on_day"]].dropna(subset=["ml_pred_cost_eur", "vroom_actual_cost_eur"])
    mlv = mlv[mlv["vroom_actual_cost_eur"] > 0]
    return mlv


# ---------------------------------------------------------------------------
def compare_distributions(train: pd.DataFrame, mlv: pd.DataFrame) -> pd.DataFrame:
    """Compute distribution statistics for key features in training vs out-of-pool."""
    rows = []
    for feat in KEY_FEATURES:
        if feat not in train.columns:
            continue
        train_v = train[feat].dropna().values
        # For out-of-pool we use vroom_n_parcels / vroom_n_routes / vroom_distance_km as proxies
        # Map per-day features from training to those available in mlv:
        proxy_map = {
            "n_parcels": "vroom_n_parcels",
            "n_stops": None,  # not directly in mlv
            "parcels_per_stop": None,
        }
        proxy = proxy_map.get(feat)
        if proxy and proxy in mlv.columns:
            oop_v = mlv[proxy].dropna().values
        else:
            oop_v = None
        row = {
            "feature": feat,
            "train_n": len(train_v),
            "train_mean": float(np.mean(train_v)),
            "train_median": float(np.median(train_v)),
            "train_p05": float(np.percentile(train_v, 5)),
            "train_p95": float(np.percentile(train_v, 95)),
            "train_max": float(train_v.max()),
        }
        if oop_v is not None and len(oop_v):
            row.update({
                "oop_proxy": proxy,
                "oop_n": len(oop_v),
                "oop_mean": float(np.mean(oop_v)),
                "oop_median": float(np.median(oop_v)),
                "oop_p05": float(np.percentile(oop_v, 5)),
                "oop_p95": float(np.percentile(oop_v, 95)),
                "oop_max": float(oop_v.max()),
                "shift_p95_pct": float(100 * (np.percentile(oop_v, 95) - np.percentile(train_v, 95)) / np.maximum(1, np.percentile(train_v, 95))),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def extrapolation_check(train: pd.DataFrame, mlv: pd.DataFrame) -> pd.DataFrame:
    """How many OOP cells lie OUTSIDE the training feature range?
    Use n_parcels as primary indicator (it's the most informative aggregate)."""
    train_min, train_max = train["n_parcels"].min(), train["n_parcels"].max()
    train_p99 = train["n_parcels"].quantile(0.99)
    train_p95 = train["n_parcels"].quantile(0.95)
    oop = mlv["vroom_n_parcels"].dropna()
    rows = [
        {"metric": "train_min", "value": train_min},
        {"metric": "train_p05", "value": float(train["n_parcels"].quantile(0.05))},
        {"metric": "train_p50", "value": float(train["n_parcels"].quantile(0.50))},
        {"metric": "train_p95", "value": train_p95},
        {"metric": "train_p99", "value": train_p99},
        {"metric": "train_max", "value": train_max},
        {"metric": "oop_n", "value": float(len(oop))},
        {"metric": "oop_mean", "value": float(oop.mean())},
        {"metric": "oop_median", "value": float(oop.median())},
        {"metric": "oop_max", "value": float(oop.max())},
        {"metric": "n_oop_above_train_p95", "value": float((oop > train_p95).sum())},
        {"metric": "pct_oop_above_train_p95", "value": float(100 * (oop > train_p95).sum() / len(oop))},
        {"metric": "n_oop_above_train_max", "value": float((oop > train_max).sum())},
        {"metric": "pct_oop_above_train_max", "value": float(100 * (oop > train_max).sum() / len(oop))},
    ]
    return pd.DataFrame(rows)


def per_bin_quality(mlv: pd.DataFrame) -> pd.DataFrame:
    """Bias and MAPE per n_parcels bin (the dominant feature)."""
    mlv = mlv.copy()
    mlv["bin"] = pd.qcut(mlv["vroom_n_parcels"], q=10, duplicates="drop")
    mlv["signed_relerr_pct"] = 100 * (mlv["ml_pred_cost_eur"] - mlv["vroom_actual_cost_eur"]) / mlv["vroom_actual_cost_eur"].clip(lower=1)
    mlv["ape_pct"] = mlv["signed_relerr_pct"].abs()
    grp = mlv.groupby("bin").agg(
        n=("vroom_n_parcels", "size"),
        parcels_mean=("vroom_n_parcels", "mean"),
        parcels_min=("vroom_n_parcels", "min"),
        parcels_max=("vroom_n_parcels", "max"),
        cost_actual_mean=("vroom_actual_cost_eur", "mean"),
        cost_pred_mean=("ml_pred_cost_eur", "mean"),
        bias_pct=("signed_relerr_pct", "mean"),
        mape_pct=("ape_pct", "mean"),
        median_ape_pct=("ape_pct", "median"),
    ).reset_index().round(2)
    return grp


def per_scenario_bin_quality(mlv: pd.DataFrame) -> pd.DataFrame:
    """Bias per (scenario, n_parcels bin)."""
    mlv = mlv.copy()
    mlv["bin"] = pd.qcut(mlv["vroom_n_parcels"], q=6, duplicates="drop")
    mlv["signed_relerr_pct"] = 100 * (mlv["ml_pred_cost_eur"] - mlv["vroom_actual_cost_eur"]) / mlv["vroom_actual_cost_eur"].clip(lower=1)
    return mlv.groupby(["scenario", "bin"]).agg(
        n=("vroom_n_parcels", "size"),
        parcels_mean=("vroom_n_parcels", "mean"),
        bias_pct=("signed_relerr_pct", "mean"),
        mape_pct=("signed_relerr_pct", lambda x: x.abs().mean()),
    ).reset_index().round(2)


# ---------------------------------------------------------------------------
def fig_DS1_histograms(train: pd.DataFrame, mlv: pd.DataFrame, out_path: Path):
    """Overlay histograms: training vs out-of-pool for key per-day features."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    pairs = [
        ("n_parcels", "vroom_n_parcels", "(a) n_parcels per delivery"),
        ("actual_distance_km", "vroom_distance_km", "(b) distance_km"),
        ("actual_n_routes", "vroom_n_routes", "(c) n_routes per cell"),
        ("actual_cost_eur", "vroom_actual_cost_eur", "(d) cost_eur per cell"),
    ]
    for ax, (tcol, mcol, title) in zip(axes.flatten(), pairs):
        train_v = train[tcol].dropna() if tcol in train.columns else None
        oop_v = mlv[mcol].dropna() if mcol in mlv.columns else None
        if train_v is not None:
            ax.hist(train_v.clip(upper=train_v.quantile(0.99)), bins=40, alpha=0.5,
                      color="#1f77b4", label=f"training_matrix (n={len(train_v):,}, mean={train_v.mean():.0f})", density=True)
        if oop_v is not None:
            ax.hist(oop_v.clip(upper=train_v.quantile(0.99) if train_v is not None else oop_v.quantile(0.99)),
                     bins=40, alpha=0.5, color="#cb181d",
                     label=f"out-of-pool SA_ML+Fixed (n={len(oop_v):,}, mean={oop_v.mean():.0f})", density=True)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel(tcol); ax.set_ylabel("Density")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Fig DS1 — Feature distribution: training (perturbed baseline) vs out-of-pool (SA_ML + Fixed schedules)",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_DS2_extrapolation(train: pd.DataFrame, mlv: pd.DataFrame, out_path: Path):
    """Scatter: vroom_n_parcels vs vroom_distance_km, color by error.
    Overlay training pool extent."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # Training pool extent (shaded)
    ax.fill_between(
        [train["n_parcels"].quantile(0.01), train["n_parcels"].quantile(0.99)],
        train["actual_distance_km"].quantile(0.01),
        train["actual_distance_km"].quantile(0.99),
        color="#1f77b4", alpha=0.08,
        label=f"Training pool 1%-99% range (n={len(train):,})",
    )

    # Scatter OOP cells colored by signed error
    mlv = mlv.copy()
    mlv["signed_relerr_pct"] = 100 * (mlv["ml_pred_cost_eur"] - mlv["vroom_actual_cost_eur"]) / mlv["vroom_actual_cost_eur"].clip(lower=1)
    sc = ax.scatter(mlv["vroom_n_parcels"], mlv["vroom_distance_km"],
                      c=mlv["signed_relerr_pct"], cmap="RdBu_r", vmin=-30, vmax=30,
                      s=18, alpha=0.7, edgecolors="k", lw=0.2,
                      label=f"OOP cells (n={len(mlv):,})")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Signed relative error  [%]  (predicted − actual) / actual")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n_parcels per delivery cell")
    ax.set_ylabel("distance_km per delivery cell")
    ax.set_title("Fig DS2 — Out-of-pool cells in feature space — color shows ML cost prediction bias",
                  loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def fig_DS3_per_bin(per_bin: pd.DataFrame, per_sc_bin: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: bias per n_parcels bin (all scenarios)
    ax = axes[0]
    xs = np.arange(len(per_bin))
    colors = ["#1f77b4" if b < 0 else "#cb181d" for b in per_bin["bias_pct"]]
    ax.bar(xs, per_bin["bias_pct"], color=colors, edgecolor="k", lw=0.4)
    for i, (b, n) in enumerate(zip(per_bin["bias_pct"], per_bin["n"])):
        ax.text(i, b, f"{b:+.1f}\nn={int(n)}", ha="center",
                  va="bottom" if b > 0 else "top", fontsize=8)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    bin_labels = [f"[{int(r['parcels_min'])}-{int(r['parcels_max'])}]" for _, r in per_bin.iterrows()]
    ax.set_xticks(xs); ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=7)
    ax.set_xlabel("n_parcels bin")
    ax.set_ylabel("Cost prediction bias  [%]")
    ax.set_title("(a) Cost-bias per n_parcels bin (all OOP cells)", loc="left", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # Right: per scenario × bin
    ax = axes[1]
    scenarios = sorted(per_sc_bin["scenario"].unique())
    bins = sorted(per_sc_bin["bin"].unique(), key=lambda x: x.left if hasattr(x, "left") else 0)
    sc_color = {"Fixed Batch-Only": "#0072B2", "SA_ML Batch-Only": "#cb181d",
                  "Avg Batch-Only": "#888", "Worst Batch-Only": "#444"}
    width = 0.8 / len(scenarios)
    for i, sc in enumerate(scenarios):
        sub = per_sc_bin[per_sc_bin["scenario"] == sc].set_index("bin").reindex(bins)
        if sub.empty: continue
        biases = sub["bias_pct"].values
        offset = (i - (len(scenarios) - 1) / 2) * width
        ax.bar(np.arange(len(bins)) + offset, biases, width,
                 color=sc_color.get(sc, "k"), edgecolor="k", lw=0.4, label=sc)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels([str(b) for b in bins], rotation=30, ha="right", fontsize=7)
    ax.set_xlabel("n_parcels bin")
    ax.set_ylabel("Cost prediction bias  [%]")
    ax.set_title("(b) Bias per Scenario × n_parcels bin", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Fig DS3 — Cost-prediction bias per feature bin",
                  x=0.005, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
def write_report(dist: pd.DataFrame, extrap: pd.DataFrame, per_bin: pd.DataFrame,
                  per_sc_bin: pd.DataFrame, out_path: Path):
    lines = ["# Distribution-Shift-Diagnose: Warum performt LGB-logT auf SA_ML schlechter?\n"]
    lines.append("## Frage\n")
    lines.append("Auf der perturbed-baseline training_matrix erreicht LGB-logT 0.73 % Cost-MAPE (Sektion 5.6).")
    lines.append("Auf den out-of-pool SA_ML-gerouteten Schedules erreicht es nur 14.5 % MAPE und +10.1 pp aggregate")
    lines.append("Saving-Bias. Was ist der Mechanismus?\n")

    lines.append("## Hypothese 1: Feature-Distribution-Shift\n")
    lines.append("Vergleich Training (perturbed baseline) vs Out-of-Pool (SA_ML + Fixed):\n")
    lines.append("| Feature | train mean | train p95 | OOP proxy mean | OOP p95 | Shift p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in dist.iterrows():
        proxy = r.get("oop_proxy")
        if pd.isna(proxy):
            continue
        lines.append(
            f"| {r['feature']} | {r['train_mean']:.0f} | {r['train_p95']:.0f} | "
            f"{r['oop_mean']:.0f} | {r['oop_p95']:.0f} | {r['shift_p95_pct']:+.0f}% |"
        )
    lines.append("")

    lines.append("## Hypothese 2: Extrapolation auf high-volume cells\n")
    lines.append(f"- Training max(n_parcels): **{extrap[extrap['metric']=='train_max']['value'].iloc[0]:.0f}**")
    lines.append(f"- Training p95(n_parcels): **{extrap[extrap['metric']=='train_p95']['value'].iloc[0]:.0f}**")
    lines.append(f"- OOP cells above train p95: **{extrap[extrap['metric']=='n_oop_above_train_p95']['value'].iloc[0]:.0f}** ({extrap[extrap['metric']=='pct_oop_above_train_p95']['value'].iloc[0]:.1f} %)")
    lines.append(f"- OOP cells above train max: **{extrap[extrap['metric']=='n_oop_above_train_max']['value'].iloc[0]:.0f}** ({extrap[extrap['metric']=='pct_oop_above_train_max']['value'].iloc[0]:.1f} %)\n")

    lines.append("## Hypothese 3: Per-Bin Cost-Bias (alle OOP cells, n_parcels-binned)\n")
    lines.append("| Bin | n | Parcels mean | Cost actual | Cost pred | Bias % | MAPE % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in per_bin.iterrows():
        lines.append(
            f"| {r['parcels_min']:.0f}-{r['parcels_max']:.0f} | {int(r['n'])} | "
            f"{r['parcels_mean']:.0f} | {r['cost_actual_mean']:.0f} | {r['cost_pred_mean']:.0f} | "
            f"{r['bias_pct']:+.2f} | {r['mape_pct']:.2f} |"
        )
    lines.append("")

    lines.append("## Per-Scenario × Per-Bin Bias\n")
    lines.append("Welche Kombinationen (Scenario × Volume) sind besonders schlecht?\n")
    lines.append("| Scenario | Bin | n | Bias % | MAPE % |")
    lines.append("|---|---|---:|---:|---:|")
    for _, r in per_sc_bin.iterrows():
        lines.append(f"| {r['scenario']} | {r['bin']} | {int(r['n'])} | {r['bias_pct']:+.2f} | {r['mape_pct']:.2f} |")
    lines.append("")

    lines.append("## Conclusion\n")
    pct_above_p95 = extrap[extrap['metric']=='pct_oop_above_train_p95']['value'].iloc[0]
    if pct_above_p95 > 30:
        lines.append(f"**Extrapolation:** {pct_above_p95:.0f}% der OOP cells liegen ueber dem 95%-Quantile des Trainings. Das Modell **extrapoliert** auf einen substantiellen Teil der Test-Daten.")
    elif pct_above_p95 > 10:
        lines.append(f"**Moderate Extrapolation:** {pct_above_p95:.0f}% der OOP cells liegen ueber dem 95%-Quantile des Trainings. Teilweise Extrapolation.")
    else:
        lines.append(f"**Geringe Extrapolation:** nur {pct_above_p95:.0f}% der OOP cells liegen ueber train p95. Distribution-Shift ist NICHT die Hauptursache.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("Loading training pool + out-of-pool data...")
    train = load_training_feature_distribution()
    mlv = load_out_of_pool_states()
    print(f"  training: {len(train)} rows;  OOP: {len(mlv)} rows")

    print("\nComputing distribution stats...")
    dist = compare_distributions(train, mlv)
    dist.to_csv(OUT / "tab_feature_distribution_stats.csv", index=False)
    print(dist.to_string(index=False))

    print("\nExtrapolation check...")
    extrap = extrapolation_check(train, mlv)
    extrap.to_csv(OUT / "tab_extrapolation_check.csv", index=False)
    print(extrap.to_string(index=False))

    print("\nPer-bin quality...")
    per_bin = per_bin_quality(mlv)
    per_bin.to_csv(OUT / "tab_per_bin_quality.csv", index=False)
    print(per_bin.to_string(index=False))

    print("\nPer-scenario per-bin...")
    per_sc_bin = per_scenario_bin_quality(mlv)
    per_sc_bin.to_csv(OUT / "tab_per_scenario_bin.csv", index=False)
    print(per_sc_bin.to_string(index=False))

    print("\nRendering figures...")
    fig_DS1_histograms(train, mlv, OUT / "fig_DS1_feature_histograms.png")
    fig_DS2_extrapolation(train, mlv, OUT / "fig_DS2_extrapolation_map.png")
    fig_DS3_per_bin(per_bin, per_sc_bin, OUT / "fig_DS3_per_bin_bias.png")

    write_report(dist, extrap, per_bin, per_sc_bin, OUT / "REPORT.md")
    print(f"\nAll outputs in {OUT}")


if __name__ == "__main__":
    main()
