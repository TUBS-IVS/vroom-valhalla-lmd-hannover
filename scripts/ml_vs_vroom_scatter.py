"""ML-vs-VROOM Scatter Diagnostic — Daganzo-LGB-Hybrid v2-aug on training matrix.

For each (provider, plz, agg_k, scale, p_keep, noise, seed) sample, predict
the routing cost with our production model and plot against the actual VROOM
solution.

Splits:
  * agg_k=1  — daily baseline (no batching)
  * agg_k=2  — 3-delivery-per-week schedules
  * agg_k=3  — 2-delivery-per-week schedules

Note: this matrix was the training set for the model, so the headline scatter
is IN-SAMPLE fit. The model's out-of-sample MAPE (5-fold CV) is 0.72%
(separate diagnostic). Here we mainly show calibration quality and any
systematic bias across operating regimes.

Outputs (results/ml_vs_vroom/):
    fig_scatter_all.{png,pdf}                # global scatter (log-log)
    fig_scatter_per_aggk.{png,pdf}           # 3 panels, one per agg_k
    fig_scatter_per_provider.{png,pdf}       # 7 panels, one per LSP
    fig_residual_distribution.{png,pdf}      # residual histograms
    tab_diagnostic_per_aggk.csv              # MAPE, MAE, R^2, bias
    tab_diagnostic_per_provider.csv
"""
from __future__ import annotations
import pickle, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from batch_delivery.features import ALL_COLS  # noqa: E402

rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

OUT = ROOT / "results" / "ml_vs_vroom"
OUT.mkdir(parents=True, exist_ok=True)

TRAINING_CSV = ROOT / "results/oracle_loop_extended_2026_05_22/training_matrix.csv"
MODEL_PKL    = ROOT / "results/oracle_loop_extended_2026_05_22/daganzo_hybrid_v2aug.pkl"

PROV_COLOR = {"Amazon": "#003049", "DHL": "#d62828", "DPD": "#f77f00",
               "FedEx": "#5a189a", "GLS": "#2a9d8f", "Hermes": "#9d4edd",
               "UPS": "#7d5a50"}
AGGK_COLOR = {1: "#1f4f8f", 2: "#2a9d8f", 3: "#e76f51"}


def log(msg):
    print(msg, flush=True)


def load_model():
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_daganzo_hybrid import DaganzoLGBHybrid, _LGBIdentityWrap  # noqa
    import __main__
    __main__._LGBIdentityWrap = _LGBIdentityWrap
    with open(MODEL_PKL, "rb") as f:
        d = pickle.load(f)
    if d.get("kind") == "DaganzoLGBHybrid":
        return DaganzoLGBHybrid(model=d["model"], combo_cols=d["combo_cols"],
                                 alpha=d["alpha"])
    raise RuntimeError(f"Expected DaganzoLGBHybrid, got {d.get('kind')}")


def diagnostics(y_true, y_pred):
    err = y_pred - y_true
    mape = float(np.mean(np.abs(err) / np.maximum(1e-6, y_true)) * 100)
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    bias_pct = float(np.mean(err / np.maximum(1e-6, y_true)) * 100)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"n": len(y_true), "MAPE_pct": mape, "MAE_eur": mae,
            "bias_eur": bias, "bias_pct": bias_pct, "R2": r2}


def main():
    t0 = time.time()
    log(f"Loading model from {MODEL_PKL.relative_to(ROOT)} ...")
    model = load_model()
    log(f"  combo_cols: {len(model.combo_cols)}  alpha={model.alpha}")

    log(f"Loading training matrix from {TRAINING_CSV.relative_to(ROOT)} ...")
    df = pd.read_csv(TRAINING_CSV)
    df = df[df.vroom_status == "OK"].copy()
    log(f"  rows OK: {len(df)}")

    # Ensure all ALL_COLS exist; fill with 0 if missing
    missing = [c for c in ALL_COLS if c not in df.columns]
    if missing:
        log(f"  WARN: missing feature columns, filling with 0: {missing}")
        for c in missing:
            df[c] = 0.0

    log("Predicting ...")
    feat_df = df[ALL_COLS].copy()
    y_pred = model.predict(feat_df)
    df["ml_pred_cost_eur"] = y_pred
    df["err"] = df["ml_pred_cost_eur"] - df["actual_cost_eur"]
    df["abs_err"] = df["err"].abs()
    df["abs_pct_err"] = 100.0 * df.abs_err / df.actual_cost_eur.clip(lower=1e-6)

    df.to_csv(OUT / "tab_predictions.csv", index=False)

    # Diagnostics per agg_k
    rows_aggk = []
    for ak in sorted(df.agg_k.unique()):
        sub = df[df.agg_k == ak]
        d = diagnostics(sub.actual_cost_eur.values, sub.ml_pred_cost_eur.values)
        d = {"agg_k": int(ak), **d}
        rows_aggk.append(d)
    diag_aggk = pd.DataFrame(rows_aggk)
    diag_aggk.to_csv(OUT / "tab_diagnostic_per_aggk.csv", index=False)
    log("\nPer agg_k:")
    log(diag_aggk.round(2).to_string(index=False))

    rows_prov = []
    for p in sorted(df.provider.unique()):
        sub = df[df.provider == p]
        d = diagnostics(sub.actual_cost_eur.values, sub.ml_pred_cost_eur.values)
        d = {"provider": p, **d}
        rows_prov.append(d)
    diag_prov = pd.DataFrame(rows_prov)
    diag_prov.to_csv(OUT / "tab_diagnostic_per_provider.csv", index=False)
    log("\nPer provider:")
    log(diag_prov.round(2).to_string(index=False))

    # Global stats
    d_all = diagnostics(df.actual_cost_eur.values, df.ml_pred_cost_eur.values)
    log(f"\nGLOBAL: n={d_all['n']}  MAPE={d_all['MAPE_pct']:.2f}%  "
        f"MAE={d_all['MAE_eur']:.2f}€  bias={d_all['bias_pct']:+.2f}%  "
        f"R²={d_all['R2']:.4f}")

    # ----- Figures -----
    # 1) Global scatter (log-log)
    log("\nPlot 1: global scatter ...")
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for ak in sorted(df.agg_k.unique()):
        sub = df[df.agg_k == ak]
        ax.scatter(sub.actual_cost_eur, sub.ml_pred_cost_eur,
                    s=9, alpha=0.35, color=AGGK_COLOR[ak],
                    label=f"agg_k = {ak} (n={len(sub):,})", edgecolor="none")
    lo = min(df.actual_cost_eur.min(), df.ml_pred_cost_eur.min())
    hi = max(df.actual_cost_eur.max(), df.ml_pred_cost_eur.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.7, label="$y = x$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("VROOM routing cost [€]")
    ax.set_ylabel("Daganzo-LGB-Hybrid predicted cost [€]")
    ax.set_title(
        f"Daganzo-LGB-Hybrid predictions vs VROOM truth "
        f"(in-sample, n = {len(df):,})\n"
        f"MAPE = {d_all['MAPE_pct']:.2f}%   bias = {d_all['bias_pct']:+.2f}%   "
        f"R$^2$ = {d_all['R2']:.4f}"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig_scatter_all.png")
    fig.savefig(OUT / "fig_scatter_all.pdf")
    plt.close(fig)

    # 2) Per agg_k 3-panel
    log("Plot 2: per agg_k ...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ai, ak in enumerate(sorted(df.agg_k.unique())):
        sub = df[df.agg_k == ak]
        d = diag_aggk[diag_aggk.agg_k == ak].iloc[0]
        axes[ai].scatter(sub.actual_cost_eur, sub.ml_pred_cost_eur,
                            s=10, alpha=0.35, color=AGGK_COLOR[ak], edgecolor="none")
        axes[ai].plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.7)
        axes[ai].set_xscale("log")
        axes[ai].set_yscale("log")
        sched_label = {1: "daily (6/wk)", 2: "3 deliv/wk", 3: "2 deliv/wk"}.get(int(ak), str(ak))
        axes[ai].set_title(f"agg_k = {ak}  ({sched_label})\n"
                              f"MAPE = {d.MAPE_pct:.2f}%   bias = {d.bias_pct:+.2f}%   "
                              f"n = {int(d.n):,}")
        axes[ai].set_xlabel("VROOM cost [€]")
        axes[ai].grid(alpha=0.3, which="both")
    axes[0].set_ylabel("ML predicted cost [€]")
    fig.suptitle("Calibration across batching regimes (agg_k = days/delivery)",
                  fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_scatter_per_aggk.png")
    fig.savefig(OUT / "fig_scatter_per_aggk.pdf")
    plt.close(fig)

    # 3) Per provider 7-panel
    log("Plot 3: per provider ...")
    providers = sorted(df.provider.unique())
    fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for ai, prov in enumerate(providers):
        sub = df[df.provider == prov]
        d = diag_prov[diag_prov.provider == prov].iloc[0]
        axes[ai].scatter(sub.actual_cost_eur, sub.ml_pred_cost_eur,
                            s=9, alpha=0.4, color=PROV_COLOR[prov], edgecolor="none")
        axes[ai].plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.7)
        axes[ai].set_xscale("log")
        axes[ai].set_yscale("log")
        axes[ai].set_title(f"{prov}\nMAPE = {d.MAPE_pct:.2f}%   "
                              f"bias = {d.bias_pct:+.2f}%   n = {int(d.n):,}")
        axes[ai].grid(alpha=0.3, which="both")
    for j in range(len(providers), len(axes)):
        axes[j].axis("off")
    fig.text(0.5, -0.01, "VROOM routing cost [€]", ha="center", fontsize=11)
    fig.text(-0.005, 0.5, "Daganzo-LGB-Hybrid predicted cost [€]",
              va="center", rotation="vertical", fontsize=11)
    fig.suptitle("Calibration per logistics service provider", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig_scatter_per_provider.png")
    fig.savefig(OUT / "fig_scatter_per_provider.pdf")
    plt.close(fig)

    # 4) Residual distributions
    log("Plot 4: residual distribution ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.hist(df.err.values, bins=80, color="#1f4f8f", alpha=0.85,
              edgecolor="white", linewidth=0.5)
    ax1.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.axvline(d_all["bias_eur"], color="#b3261e", linewidth=1.2,
                  label=f"Mean bias = {d_all['bias_eur']:.1f}€ "
                         f"({d_all['bias_pct']:+.2f}%)")
    ax1.set_xlabel("ML prediction error  $\\hat{y} - y$  [€]")
    ax1.set_ylabel("Count")
    ax1.set_title("Absolute residual distribution")
    ax1.legend()
    ax1.grid(alpha=0.3)

    rel_err = (df.err.values / np.maximum(1e-6, df.actual_cost_eur.values) * 100)
    ax2.hist(rel_err, bins=80, range=(-15, 15), color="#2a9d8f", alpha=0.85,
              edgecolor="white", linewidth=0.5)
    ax2.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.axvline(d_all["bias_pct"], color="#b3261e", linewidth=1.2,
                  label=f"Mean rel. bias = {d_all['bias_pct']:+.2f}%")
    ax2.set_xlabel("Relative ML error  $(\\hat{y} - y) / y$  [%]")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Relative residual distribution  (MAPE = {d_all['MAPE_pct']:.2f}%)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_residual_distribution.png")
    fig.savefig(OUT / "fig_residual_distribution.pdf")
    plt.close(fig)

    log(f"\nDone in {time.time()-t0:.0f}s. Outputs in: {OUT}")
    for p in sorted(OUT.glob("*")):
        log(f"  {p.name}")


if __name__ == "__main__":
    main()
