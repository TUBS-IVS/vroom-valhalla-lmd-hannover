"""Predict optimized-scenario costs with Daganzo-LGB-Hybrid vs VROOM truth.

Uses the per-day resolved table from `results/final_optimization_v3_mergefix/
ml_vs_vroom_per_day.csv`, which contains:
  * the schedules picked by the v3 LGB-logT optimizer
  * the per-day VROOM-actual costs after re-solving

For each (scenario, provider, PLZ) tuple we:
  1. Aggregate VROOM cost over delivery days  →  weekly truth
  2. Re-predict the weekly cost with the **Daganzo-Hybrid** model
     by looking up `sched_cost[(prov, plz), schedule_idx]` in the cached matrix
  3. Also keep the original LGB-v3 weekly prediction (sum over delivery days)
     for an A/B comparison

Outputs (results/ml_vs_vroom_optimized/):
    fig_scatter_daganzo_hybrid.{png,pdf}     (NEW model vs VROOM)
    fig_scatter_lgb_v3.{png,pdf}             (OLD model vs VROOM, for context)
    fig_scatter_per_provider.{png,pdf}       (Daganzo-Hybrid per LSP)
    fig_residual_distribution.{png,pdf}
    tab_weekly_predictions.csv               (scenario, provider, plz, vroom, lgb_v3, daganzo)
    tab_diagnostics.csv                      (model × scenario MAPE/bias/R²)
"""
from __future__ import annotations
import pickle, sys, time, warnings
from collections import defaultdict
from itertools import combinations
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

rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

OUT = ROOT / "results" / "ml_vs_vroom_optimized"
OUT.mkdir(parents=True, exist_ok=True)

ML_VROOM_CSV = ROOT / "results/final_optimization_v3_mergefix/ml_vs_vroom_per_day.csv"
CACHE_NPZ    = ROOT / "results/penalty_sweep/sched_cost_cache.npz"

PROV_COLOR = {"Amazon": "#003049", "DHL": "#d62828", "DPD": "#f77f00",
               "FedEx": "#5a189a", "GLS": "#2a9d8f", "Hermes": "#9d4edd",
               "UPS": "#7d5a50"}
SCEN_COLOR = {"Fixed Batch-Only": "#1f4f8f", "SA_ML Batch-Only": "#2a9d8f",
               "Avg Batch-Only": "#e9c46a", "Worst Batch-Only": "#b3261e"}

N_DAYS = 6
MAX_HOLD = 3


def log(msg):
    print(msg, flush=True)


def enumerate_schedules():
    out = []
    for k in range(1, N_DAYS + 1):
        for combo in combinations(range(N_DAYS), k):
            days = sorted(combo)
            ok = True
            for i in range(len(days)):
                gap = (days[(i + 1) % len(days)] - days[i]) % N_DAYS
                if gap == 0:
                    gap = N_DAYS
                if gap > MAX_HOLD:
                    ok = False
                    break
            if ok:
                out.append(frozenset(days))
    return out


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
    log("[1] Loading per-day ML/VROOM table ...")
    df = pd.read_csv(ML_VROOM_CSV)
    df = df[df.delivers_on_day]  # only delivery days have non-zero costs
    log(f"  {len(df)} delivery-day rows  ({df.scenario.nunique()} scenarios, "
        f"{df.groupby(['scenario','provider','plz']).ngroups} (scen,prov,plz) cells)")

    # Aggregate to weekly (scenario, provider, plz)
    weekly = (df.groupby(["scenario", "provider", "plz", "schedule_days", "schedule_size"],
                          as_index=False)
                  .agg(vroom_cost_eur=("vroom_actual_cost_eur", "sum"),
                       lgb_v3_pred_eur=("ml_pred_cost_eur", "sum"),
                       n_parcels=("vroom_n_parcels", "sum"),
                       n_routes=("vroom_n_routes", "sum")))
    log(f"  weekly rows: {len(weekly)}")

    # Look up Daganzo-Hybrid cached weekly cost
    log("[2] Loading cached sched_cost matrix ...")
    cache = np.load(CACHE_NPZ)
    sched_cost = cache["sched_cost"]
    prov_order = list(cache["prov_order"])
    plz_order  = list(cache["plz_order"])
    pp_lookup = {(p, plz): i for i, (p, plz) in enumerate(zip(prov_order, plz_order))}
    log(f"  cache shape: {sched_cost.shape}")

    schedules = enumerate_schedules()
    sched_lookup = {",".join(str(d) for d in sorted(s)): i for i, s in enumerate(schedules)}
    log(f"  {len(schedules)} schedule patterns; sample lookup keys: "
        f"{list(sched_lookup.keys())[:3]}")

    # The cached costs are at fast_share=0 (everything batched), so they should
    # be apples-to-apples for the *Batch-Only* scenarios.
    daganzo_pred = []
    missing = 0
    for _, r in weekly.iterrows():
        key_pp = (r.provider, str(r.plz).zfill(5) if isinstance(r.plz, int) else str(r.plz))
        # PLZ might be int; cache uses str
        if key_pp not in pp_lookup:
            # try without zfill
            key_pp = (r.provider, str(r.plz))
        if key_pp not in pp_lookup:
            daganzo_pred.append(np.nan); missing += 1; continue
        pp_i = pp_lookup[key_pp]
        sched_key = str(r.schedule_days).strip()
        if sched_key not in sched_lookup:
            daganzo_pred.append(np.nan); missing += 1; continue
        si = sched_lookup[sched_key]
        daganzo_pred.append(float(sched_cost[pp_i, si]))
    weekly["daganzo_hybrid_pred_eur"] = daganzo_pred
    if missing:
        log(f"  WARN: {missing} rows had no cache lookup match")
    weekly = weekly.dropna(subset=["daganzo_hybrid_pred_eur"]).copy()
    log(f"  matched {len(weekly)} weekly rows")

    # Filter zero-VROOM rows (Avg/Worst scenarios have NaN per-day → sum 0)
    n_before = len(weekly)
    weekly = weekly[weekly.vroom_cost_eur > 0].copy()
    log(f"  filtered to {len(weekly)} rows with valid VROOM truth (dropped {n_before - len(weekly)})")
    log(f"  scenarios remaining: {weekly.scenario.value_counts().to_dict()}")
    weekly.to_csv(OUT / "tab_weekly_predictions.csv", index=False)

    # Diagnostics
    diag_rows = []
    for scen in sorted(weekly.scenario.unique()):
        sub = weekly[weekly.scenario == scen]
        for model_name, col in [("LGB-logT v3", "lgb_v3_pred_eur"),
                                  ("Daganzo-LGB-Hybrid v2-aug", "daganzo_hybrid_pred_eur")]:
            d = diagnostics(sub.vroom_cost_eur.values, sub[col].values)
            d = {"scenario": scen, "model": model_name, **d}
            diag_rows.append(d)
    # Global per-model
    for model_name, col in [("LGB-logT v3", "lgb_v3_pred_eur"),
                              ("Daganzo-LGB-Hybrid v2-aug", "daganzo_hybrid_pred_eur")]:
        d = diagnostics(weekly.vroom_cost_eur.values, weekly[col].values)
        d = {"scenario": "ALL", "model": model_name, **d}
        diag_rows.append(d)
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(OUT / "tab_diagnostics.csv", index=False)
    log("\nDiagnostics:")
    log(diag.round(2).to_string(index=False))

    # ----- Figures -----
    lo = min(weekly.vroom_cost_eur.min(), weekly.daganzo_hybrid_pred_eur.min(),
              weekly.lgb_v3_pred_eur.min())
    hi = max(weekly.vroom_cost_eur.max(), weekly.daganzo_hybrid_pred_eur.max(),
              weekly.lgb_v3_pred_eur.max())

    # 1) Daganzo-Hybrid main scatter (colored by scenario)
    log("\nPlot 1: Daganzo-Hybrid vs VROOM ...")
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for scen in sorted(weekly.scenario.unique()):
        sub = weekly[weekly.scenario == scen]
        ax.scatter(sub.vroom_cost_eur, sub.daganzo_hybrid_pred_eur,
                    s=24, alpha=0.55, color=SCEN_COLOR.get(scen, "grey"),
                    label=f"{scen}  (n={len(sub)})", edgecolor="none")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.7, label="$y = x$")
    d_dag = diag[(diag.model.str.startswith("Daganzo")) & (diag.scenario == "ALL")].iloc[0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("VROOM actual weekly cost [€]")
    ax.set_ylabel("Daganzo-LGB-Hybrid predicted weekly cost [€]")
    ax.set_title(
        f"Daganzo-LGB-Hybrid predictions vs VROOM truth on optimized schedules\n"
        f"MAPE = {d_dag['MAPE_pct']:.2f}%   bias = {d_dag['bias_pct']:+.2f}%   "
        f"R$^2$ = {d_dag['R2']:.4f}   n = {int(d_dag['n']):,}"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig_scatter_daganzo_hybrid.png")
    fig.savefig(OUT / "fig_scatter_daganzo_hybrid.pdf")
    plt.close(fig)

    # 2) Old LGB-v3 scatter (for context)
    log("Plot 2: LGB-v3 vs VROOM ...")
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for scen in sorted(weekly.scenario.unique()):
        sub = weekly[weekly.scenario == scen]
        ax.scatter(sub.vroom_cost_eur, sub.lgb_v3_pred_eur,
                    s=24, alpha=0.55, color=SCEN_COLOR.get(scen, "grey"),
                    label=f"{scen}  (n={len(sub)})", edgecolor="none")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.7, label="$y = x$")
    d_lgb = diag[(diag.model == "LGB-logT v3") & (diag.scenario == "ALL")].iloc[0]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("VROOM actual weekly cost [€]")
    ax.set_ylabel("LGB-logT v3 predicted weekly cost [€]")
    ax.set_title(
        f"Previous-generation LGB-logT v3 predictions vs VROOM truth\n"
        f"MAPE = {d_lgb['MAPE_pct']:.2f}%   bias = {d_lgb['bias_pct']:+.2f}%   "
        f"R$^2$ = {d_lgb['R2']:.4f}"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig_scatter_lgb_v3.png")
    fig.savefig(OUT / "fig_scatter_lgb_v3.pdf")
    plt.close(fig)

    # 3) Side-by-side comparison (Daganzo-Hybrid left, LGB-v3 right) for paper
    log("Plot 3: side-by-side model comparison ...")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, col, title, d in [
        (axA, "daganzo_hybrid_pred_eur",
         "Daganzo-LGB-Hybrid v2-aug (current model)", d_dag),
        (axB, "lgb_v3_pred_eur",
         "LGB-logT v3 (previous generation)", d_lgb),
    ]:
        for scen in sorted(weekly.scenario.unique()):
            sub = weekly[weekly.scenario == scen]
            ax.scatter(sub.vroom_cost_eur, sub[col],
                       s=20, alpha=0.55, color=SCEN_COLOR.get(scen, "grey"),
                       label=scen, edgecolor="none")
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("VROOM actual weekly cost [€]")
        ax.set_title(f"{title}\n"
                      f"MAPE = {d['MAPE_pct']:.2f}%   "
                      f"bias = {d['bias_pct']:+.2f}%   R$^2$ = {d['R2']:.4f}")
        ax.grid(alpha=0.3, which="both")
    axA.set_ylabel("Predicted weekly cost [€]")
    axA.legend(loc="upper left", fontsize=8)
    fig.suptitle("Surrogate calibration on optimized scenarios (weekly aggregates, "
                  f"n = {len(weekly):,})", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig_scatter_side_by_side.png")
    fig.savefig(OUT / "fig_scatter_side_by_side.pdf")
    plt.close(fig)

    # 4) Per-provider Daganzo-Hybrid
    log("Plot 4: per-provider ...")
    providers = sorted(weekly.provider.unique())
    fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for ai, prov in enumerate(providers):
        sub = weekly[weekly.provider == prov]
        d = diagnostics(sub.vroom_cost_eur.values,
                          sub.daganzo_hybrid_pred_eur.values)
        axes[ai].scatter(sub.vroom_cost_eur, sub.daganzo_hybrid_pred_eur,
                            s=18, alpha=0.55, color=PROV_COLOR[prov],
                            edgecolor="none")
        axes[ai].plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.7)
        axes[ai].set_xscale("log")
        axes[ai].set_yscale("log")
        axes[ai].set_title(f"{prov}\nMAPE = {d['MAPE_pct']:.2f}%   "
                              f"bias = {d['bias_pct']:+.2f}%   n = {d['n']}")
        axes[ai].grid(alpha=0.3, which="both")
    for j in range(len(providers), len(axes)):
        axes[j].axis("off")
    fig.text(0.5, -0.01, "VROOM actual weekly cost [€]", ha="center", fontsize=11)
    fig.text(-0.005, 0.5, "Daganzo-LGB-Hybrid predicted [€]",
              va="center", rotation="vertical", fontsize=11)
    fig.suptitle("Daganzo-LGB-Hybrid calibration by logistics service provider",
                  fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig_scatter_per_provider.png")
    fig.savefig(OUT / "fig_scatter_per_provider.pdf")
    plt.close(fig)

    # 5) Residual distribution
    log("Plot 5: residual distribution ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    weekly["err_daganzo"] = weekly.daganzo_hybrid_pred_eur - weekly.vroom_cost_eur
    weekly["relerr_daganzo"] = 100.0 * weekly.err_daganzo / weekly.vroom_cost_eur.clip(lower=1e-6)
    weekly["err_lgb"] = weekly.lgb_v3_pred_eur - weekly.vroom_cost_eur
    weekly["relerr_lgb"] = 100.0 * weekly.err_lgb / weekly.vroom_cost_eur.clip(lower=1e-6)

    bins = np.linspace(-40, 40, 81)
    ax1.hist(weekly.relerr_daganzo.values, bins=bins, color="#1f4f8f",
              alpha=0.7, edgecolor="white", label="Daganzo-LGB-Hybrid v2-aug")
    ax1.hist(weekly.relerr_lgb.values, bins=bins, color="#e76f51",
              alpha=0.55, edgecolor="white", label="LGB-logT v3")
    ax1.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Relative ML error  $(\\hat{y} - y)/y$  [%]")
    ax1.set_ylabel("Count")
    ax1.set_title("Relative residuals on optimized scenarios")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Cumulative absolute error
    abs_err_dag = np.sort(np.abs(weekly.relerr_daganzo.values))
    abs_err_lgb = np.sort(np.abs(weekly.relerr_lgb.values))
    pct = np.linspace(0, 100, len(weekly))
    ax2.plot(abs_err_dag, pct, color="#1f4f8f", linewidth=2,
              label=f"Daganzo-Hybrid  (median {np.median(abs_err_dag):.2f}%)")
    ax2.plot(abs_err_lgb, pct, color="#e76f51", linewidth=2,
              label=f"LGB-logT v3      (median {np.median(abs_err_lgb):.2f}%)")
    ax2.set_xlabel("Absolute relative error |$(\\hat{y} - y)/y$|  [%]")
    ax2.set_ylabel("Cumulative share of cells [%]")
    ax2.set_title("Error CDF — Daganzo-Hybrid dominates LGB-v3")
    ax2.set_xlim(0, 40)
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
