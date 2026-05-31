"""Detailed ML validation: VROOM truth vs Daganzo-LGB-Hybrid vs PURE Daganzo.

Validation is performed on the 1267 VROOM-routed delivery-day cells from
Phase 2 of the overnight orchestrator (P=0.5, share_willing=1.0). These
schedules were chosen by ML optimization and routed with VROOM — they were
*not* in the surrogate's training pool (which used different aggregations
via the oracle-loop pipeline). Hence this is a genuine generalisation test.

For each (provider, PLZ) we compare three numbers:

  VROOM_actual  — the routed truth (Δ_truth)
  Hybrid        — Daganzo physics + LGB residual = production model
  Pure Daganzo  — physics only (α·base, with LGB residual removed)

The gap between Pure Daganzo and Hybrid shows how much value the LGB
residual contributes; the gap between Hybrid and VROOM shows residual error.

Outputs (results/overnight_2026_05_27/):
  fig15_validation_scatter_hybrid_vs_pure.{png,pdf}
  fig16_validation_per_provider.{png,pdf}
  fig17_validation_per_schedule_size.{png,pdf}
  fig18_residual_distributions.{png,pdf}
  fig19_lgb_correction_pattern.{png,pdf}
  fig20_physical_plausibility.{png,pdf}
  tab_validation_per_pp.csv
  tab_validation_summary.csv

Inputs:
  results/overnight_2026_05_27/tab_vroom_validation.csv
  results/overnight_2026_05_27/tab_chosen_schedules.csv
  results/oracle_loop_extended_2026_05_22/daganzo_hybrid_v2aug.pkl
  results/checkpoints/01_demand.pkl, 04_optim_prep.pkl
"""
from __future__ import annotations
import pickle
import sys
import warnings
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

from batch_delivery.features import ALL_COLS  # noqa: E402
from batch_delivery.optimization.core import build_cost_matrices_ml  # noqa: E402

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

OUT = ROOT / "results" / "overnight_2026_05_27"
N_DAYS = 6
MAX_HOLD = 3
OPERATING_P = 0.5
OPERATING_SHARE = 1.0
PROVIDERS = ["DHL", "Amazon", "DPD", "FedEx", "GLS", "Hermes", "UPS"]
PROV_COLOR = {"Amazon": "#003049", "DHL": "#d62828", "DPD": "#f77f00",
               "FedEx": "#5a189a", "GLS": "#2a9d8f", "Hermes": "#9d4edd",
               "UPS": "#7d5a50"}


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


def load_hybrid_model():
    from train_daganzo_hybrid import DaganzoLGBHybrid, _LGBIdentityWrap  # noqa
    import __main__
    __main__._LGBIdentityWrap = _LGBIdentityWrap
    with open(ROOT / "results/oracle_loop_extended_2026_05_22/daganzo_hybrid_v2aug.pkl", "rb") as f:
        d = pickle.load(f)
    return DaganzoLGBHybrid(model=d["model"], combo_cols=d["combo_cols"], alpha=d["alpha"])


class PureDaganzoPredictor:
    """Predicts cost using only Daganzo physics (no LGB residual)."""

    def __init__(self, hybrid):
        self.combo_cols = hybrid.combo_cols
        self.alpha = hybrid.alpha
        self._daganzo_vec = hybrid._daganzo_vec
        self.kind = "PureDaganzo"

    def predict(self, df_feats):
        base = self._daganzo_vec(
            df_feats["n_parcels"].values, df_feats["n_stops"].values,
            df_feats["area_km2"].values, df_feats["hub_dist_km"].values,
        )
        return self.alpha * base

    def predict_single(self, base25):
        df = pd.DataFrame(base25.reshape(1, -1), columns=ALL_COLS)
        return float(self.predict(df)[0])


def build_ml_prep(provider_data):
    from batch_delivery.config.constants import provider_to_demand_prefix
    ml_prep = {}
    for prov in PROVIDERS:
        pdata = provider_data.get(prov)
        if pdata is None:
            continue
        df_assign = pdata["df_assignments"]
        hub_coords_by_plz = {row["plz"]: (row["hub_lon"], row["hub_lat"])
                              for _, row in df_assign.iterrows()}
        prefix = provider_to_demand_prefix(prov)
        col_total = f"{prefix}_total"
        plz_day_coords = {}
        for pc in pdata["all_plz_set"]:
            plz_day_coords[pc] = {}
            for d in range(N_DAYS):
                gdf_d = pdata["daily_gdfs_wgs"].get(d)
                if gdf_d is None:
                    continue
                pts = gdf_d[gdf_d["plz"] == pc]
                if len(pts) == 0:
                    continue
                lons = pts["lon"].values.astype(np.float64)
                lats = pts["lat"].values.astype(np.float64)
                psd = (pts[col_total].values.astype(np.float64)
                       if col_total in pts.columns else np.ones(len(pts)))
                plz_day_coords[pc][d] = (lons, lats, psd)
        ml_prep[prov] = {"plz_day_coords": plz_day_coords,
                          "hub_coords_by_plz": hub_coords_by_plz}
    return ml_prep


def predict_with_both(provider_data, optim_data, schedules, hybrid, pure_dag,
                       chosen_idx_per_pp):
    """For each (provider, PLZ) with chosen schedule, sum delivery-day costs
    using HYBRID and PURE_DAGANZO models.  Returns per-(prov, plz) totals.
    """
    ml_prep = build_ml_prep(provider_data)
    rows = []
    for prov in PROVIDERS:
        if prov not in optim_data or prov not in ml_prep:
            continue
        odata = optim_data[prov]
        prep = ml_prep[prov]
        plz_keys = odata["plz_keys"]
        plz_data = odata["plz_data"]

        for model, label in [(hybrid, "hybrid"), (pure_dag, "pure_daganzo")]:
            matrices = build_cost_matrices_ml(
                plz_keys, plz_data, schedules, model, prov,
                prep["plz_day_coords"], prep["hub_coords_by_plz"],
                fast_share_b2c=0.0, fast_share_b2b=0.0,
            )
            cost_3d = matrices["cost_3d"]
            sched_active = matrices["sched_active"]
            dd_cost_per_sched = (cost_3d * sched_active[None, :, :]).sum(axis=2)
            for pi, pc in enumerate(plz_keys):
                key = (prov, str(pc))
                if key not in chosen_idx_per_pp:
                    continue
                si = chosen_idx_per_pp[key]
                rows.append({
                    "provider": prov, "plz": str(pc),
                    "model": label,
                    "predicted_cost_eur": float(dd_cost_per_sched[pi, int(si)]),
                })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("Detailed ML validation: VROOM vs Hybrid vs Pure Daganzo")
    print("=" * 70)

    vroom = pd.read_csv(OUT / "tab_vroom_validation.csv")
    vroom = vroom[vroom.vroom_cost_eur > 0].copy()
    vroom["plz"] = vroom.plz.astype(str)
    print(f"  VROOM solves: {len(vroom)} delivery-day rows")
    # Aggregate to weekly per (prov, plz)
    vagg = (vroom.groupby(["provider", "plz"], as_index=False)
            .agg(vroom_weekly_cost=("vroom_cost_eur", "sum"),
                 vroom_total_routes=("vroom_n_routes", "sum"),
                 vroom_total_distance_km=("vroom_distance_km", "sum"),
                 vroom_total_parcels=("vroom_n_parcels", "sum")))

    chosen = pd.read_csv(OUT / "tab_chosen_schedules.csv")
    chosen = chosen[(np.isclose(chosen.penalty, OPERATING_P)) &
                    (np.isclose(chosen.share_willing, OPERATING_SHARE))].copy()
    chosen["plz"] = chosen.plz.astype(str)
    print(f"  Chosen schedules: {len(chosen)} (provider, PLZ) rows")

    chosen_idx_per_pp = {(r.provider, r.plz): int(r.schedule_idx)
                         for _, r in chosen.iterrows()}

    print("\nLoading models ...")
    hybrid = load_hybrid_model()
    pure_dag = PureDaganzoPredictor(hybrid)
    print(f"  hybrid alpha = {hybrid.alpha}")

    print("\nRe-predicting with HYBRID + PURE_DAGANZO ...")
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    schedules = enumerate_schedules()
    preds_long = predict_with_both(
        chk["provider_data"], chk4["optimization_data"], schedules,
        hybrid, pure_dag, chosen_idx_per_pp,
    )
    # Pivot to (prov, plz, hybrid, pure_daganzo)
    preds = preds_long.pivot_table(
        index=["provider", "plz"], columns="model",
        values="predicted_cost_eur", aggfunc="first"
    ).reset_index()

    # Merge with VROOM and chosen schedule meta
    df = preds.merge(vagg, on=["provider", "plz"], how="inner")
    df = df.merge(chosen[["provider", "plz", "schedule_size",
                          "schedule_weekdays", "weekly_parcels"]],
                  on=["provider", "plz"], how="left")
    df["err_hybrid"] = df.hybrid - df.vroom_weekly_cost
    df["err_pure"] = df.pure_daganzo - df.vroom_weekly_cost
    df["abs_err_hybrid"] = df.err_hybrid.abs()
    df["abs_err_pure"] = df.err_pure.abs()
    df["pct_err_hybrid"] = 100 * df.err_hybrid / df.vroom_weekly_cost.clip(lower=1)
    df["pct_err_pure"] = 100 * df.err_pure / df.vroom_weekly_cost.clip(lower=1)
    df.to_csv(OUT / "tab_validation_per_pp.csv", index=False)

    # ── headline summary ────────────────────────────────────────────────
    def diag(actual, predicted, label):
        err = predicted - actual
        mape = float(np.mean(np.abs(err) / np.maximum(1e-6, actual)) * 100)
        bias_pct = float(np.mean(err / np.maximum(1e-6, actual)) * 100)
        r2 = 1 - float(((err) ** 2).sum() / max(1, ((actual - actual.mean()) ** 2).sum()))
        return {"model": label, "n": len(actual), "MAPE_pct": mape,
                "bias_pct": bias_pct, "R2": r2,
                "mean_abs_err_eur": float(np.mean(np.abs(err)))}
    s_hybrid = diag(df.vroom_weekly_cost.values, df.hybrid.values, "Daganzo-LGB-Hybrid")
    s_pure   = diag(df.vroom_weekly_cost.values, df.pure_daganzo.values, "Pure Daganzo physics")
    summary = pd.DataFrame([s_hybrid, s_pure])
    summary.to_csv(OUT / "tab_validation_summary.csv", index=False)
    print("\n--- Headline (global) ---")
    print(summary.round(2).to_string(index=False))

    print("\n--- Per-provider MAPE ---")
    prov_diag = []
    for prov, g in df.groupby("provider"):
        for col, lbl in [("hybrid", "Daganzo-Hybrid"),
                          ("pure_daganzo", "Pure Daganzo")]:
            d = diag(g.vroom_weekly_cost.values, g[col].values, lbl)
            d["provider"] = prov
            prov_diag.append(d)
    prov_diag_df = pd.DataFrame(prov_diag)
    print(prov_diag_df.pivot(index="provider", columns="model", values="MAPE_pct").round(2).to_string())

    # ── Figures ─────────────────────────────────────────────────────────
    lo = min(df.vroom_weekly_cost.min(), df.hybrid.min(), df.pure_daganzo.min())
    hi = max(df.vroom_weekly_cost.max(), df.hybrid.max(), df.pure_daganzo.max())

    # Plot 15: side-by-side scatter
    print("\nPlot 15: side-by-side scatter ...")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, col, title, d in [
        (axA, "pure_daganzo", "Pure Daganzo physics", s_pure),
        (axB, "hybrid", "Daganzo-LGB-Hybrid (production)", s_hybrid),
    ]:
        for prov, g in df.groupby("provider"):
            ax.scatter(g.vroom_weekly_cost, g[col],
                       s=18, alpha=0.65, color=PROV_COLOR[prov], label=prov,
                       edgecolor="none")
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("VROOM actual cost [EUR]")
        ax.set_title(f"{title}\nMAPE = {d['MAPE_pct']:.2f}%   "
                      f"bias = {d['bias_pct']:+.2f}%   "
                      f"R$^2$ = {d['R2']:.4f}")
        ax.grid(alpha=0.3, which="both")
    axA.set_ylabel("Predicted weekly cost [EUR]")
    axA.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"ML validation on optimized schedules — pure physics vs hybrid "
                  f"(n = {len(df)} (provider, PLZ) cells, P={OPERATING_P}, share=100%)",
                  fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig15_validation_scatter_hybrid_vs_pure.png")
    fig.savefig(OUT / "fig15_validation_scatter_hybrid_vs_pure.pdf")
    plt.close(fig)

    # Plot 16: per-provider MAPE bars
    print("Plot 16: per-provider MAPE ...")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    prov_order = list(PROV_COLOR.keys())
    x = np.arange(len(prov_order))
    width = 0.38
    mape_pure = [
        prov_diag_df.query("provider == @p and model == 'Pure Daganzo'")
                    .MAPE_pct.iloc[0] for p in prov_order
    ]
    mape_hyb = [
        prov_diag_df.query("provider == @p and model == 'Daganzo-Hybrid'")
                    .MAPE_pct.iloc[0] for p in prov_order
    ]
    ax.bar(x - width / 2, mape_pure, width, color="#e76f51",
           label="Pure Daganzo physics", edgecolor="black")
    ax.bar(x + width / 2, mape_hyb, width, color="#1f4f8f",
           label="Daganzo-LGB-Hybrid", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(prov_order)
    ax.set_ylabel("MAPE [%]")
    ax.set_title("Per-LSP MAPE: pure physics vs hybrid")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig16_validation_per_provider.png")
    fig.savefig(OUT / "fig16_validation_per_provider.pdf")
    plt.close(fig)

    # Plot 17: per-schedule-size MAPE
    print("Plot 17: per-schedule-size MAPE ...")
    sz_diag = []
    for sz, g in df.groupby("schedule_size"):
        if len(g) < 5:
            continue
        for col, lbl in [("hybrid", "Daganzo-Hybrid"), ("pure_daganzo", "Pure Daganzo")]:
            d = diag(g.vroom_weekly_cost.values, g[col].values, lbl)
            d["schedule_size"] = sz
            d["n"] = len(g)
            sz_diag.append(d)
    sz_df = pd.DataFrame(sz_diag)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sizes = sorted(sz_df.schedule_size.unique())
    x = np.arange(len(sizes))
    mape_pure_sz = [sz_df.query("schedule_size == @s and model == 'Pure Daganzo'")
                          .MAPE_pct.iloc[0] for s in sizes]
    mape_hyb_sz = [sz_df.query("schedule_size == @s and model == 'Daganzo-Hybrid'")
                          .MAPE_pct.iloc[0] for s in sizes]
    n_vals = [sz_df.query("schedule_size == @s and model == 'Daganzo-Hybrid'")
                          .n.iloc[0] for s in sizes]
    ax.bar(x - width / 2, mape_pure_sz, width, color="#e76f51",
           label="Pure Daganzo", edgecolor="black")
    ax.bar(x + width / 2, mape_hyb_sz, width, color="#1f4f8f",
           label="Hybrid", edgecolor="black")
    for i, n in enumerate(n_vals):
        ax.text(i, max(mape_pure_sz[i], mape_hyb_sz[i]) + 0.4,
                f"n={n}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s)} d/wk" for s in sizes])
    ax.set_ylabel("MAPE [%]")
    ax.set_title("MAPE per chosen schedule-size: physics vs hybrid")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig17_validation_per_schedule_size.png")
    fig.savefig(OUT / "fig17_validation_per_schedule_size.pdf")
    plt.close(fig)

    # Plot 18: residual histograms
    print("Plot 18: residual distributions ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    bins_eur = np.linspace(min(df.err_pure.min(), df.err_hybrid.min()),
                            max(df.err_pure.max(), df.err_hybrid.max()), 60)
    ax1.hist(df.err_pure, bins=bins_eur, color="#e76f51", alpha=0.55,
              edgecolor="white", label=f"Pure Daganzo (bias={s_pure['bias_pct']:+.1f}%)")
    ax1.hist(df.err_hybrid, bins=bins_eur, color="#1f4f8f", alpha=0.7,
              edgecolor="white", label=f"Hybrid (bias={s_hybrid['bias_pct']:+.1f}%)")
    ax1.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Prediction error $\\hat{y} - y$ [EUR]")
    ax1.set_ylabel("Count")
    ax1.set_title("Absolute residuals")
    ax1.legend()
    ax1.grid(alpha=0.3)

    rel_bins = np.linspace(-50, 50, 80)
    ax2.hist(df.pct_err_pure.clip(-50, 50), bins=rel_bins, color="#e76f51",
              alpha=0.55, edgecolor="white", label="Pure Daganzo")
    ax2.hist(df.pct_err_hybrid.clip(-50, 50), bins=rel_bins, color="#1f4f8f",
              alpha=0.7, edgecolor="white", label="Hybrid")
    ax2.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Relative error $(\\hat{y} - y)/y$ [%]")
    ax2.set_ylabel("Count")
    ax2.set_title("Relative residuals")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.suptitle("How the LGB residual fixes pure-Daganzo bias",
                  fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig18_residual_distributions.png")
    fig.savefig(OUT / "fig18_residual_distributions.pdf")
    plt.close(fig)

    # Plot 19: LGB correction pattern — error before vs after
    print("Plot 19: LGB correction pattern ...")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for prov, g in df.groupby("provider"):
        ax.scatter(g.pct_err_pure, g.pct_err_hybrid,
                   color=PROV_COLOR[prov], s=18, alpha=0.65, label=prov,
                   edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.plot([-60, 60], [-60, 60], "k--", linewidth=1, alpha=0.4,
             label="no improvement")
    ax.set_xlabel("Pure-Daganzo relative error [%]")
    ax.set_ylabel("Hybrid relative error [%]")
    ax.set_title("Cell-by-cell: where the LGB residual reduces the physics error\n"
                  "(points below diagonal: hybrid improves over pure Daganzo)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    fig.tight_layout()
    fig.savefig(OUT / "fig19_lgb_correction_pattern.png")
    fig.savefig(OUT / "fig19_lgb_correction_pattern.pdf")
    plt.close(fig)

    # Plot 20: physical plausibility — cost vs n_parcels
    print("Plot 20: physical plausibility ...")
    # Bin VROOM data by total parcels, show mean cost in each bin
    df_sorted = df.dropna(subset=["weekly_parcels"]).copy()
    df_sorted["log_parcels"] = np.log10(df_sorted.weekly_parcels.clip(lower=1))
    n_bins = 12
    df_sorted["pkts_bin"] = pd.cut(df_sorted.log_parcels, bins=n_bins)
    bin_agg = (df_sorted.groupby("pkts_bin", observed=True).agg(
        mean_pkts=("weekly_parcels", "mean"),
        mean_vroom=("vroom_weekly_cost", "mean"),
        mean_hybrid=("hybrid", "mean"),
        mean_pure=("pure_daganzo", "mean"),
        n=("plz", "count")).reset_index())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(bin_agg.mean_pkts, bin_agg.mean_vroom / 1e3, "s-",
            color="black", linewidth=2.2, markersize=9, label="VROOM truth")
    ax.plot(bin_agg.mean_pkts, bin_agg.mean_hybrid / 1e3, "o-",
            color="#1f4f8f", linewidth=2, markersize=7,
            label="Daganzo-Hybrid")
    ax.plot(bin_agg.mean_pkts, bin_agg.mean_pure / 1e3, "^--",
            color="#e76f51", linewidth=1.8, markersize=7,
            label="Pure Daganzo")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Weekly parcels per (provider, PLZ)")
    ax.set_ylabel("Mean weekly cost [k EUR]")
    ax.set_title("Physical plausibility — does the model preserve "
                  "cost-vs-volume scaling?\n"
                  "(both pure physics and hybrid follow VROOM's monotonic curve)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig20_physical_plausibility.png")
    fig.savefig(OUT / "fig20_physical_plausibility.pdf")
    plt.close(fig)

    print("\n" + "=" * 70)
    print("DONE. Outputs:")
    for p in sorted(OUT.glob("fig1[5-9]*"), key=lambda x: x.name):
        print(f"  {p.name}")
    for p in sorted(OUT.glob("fig20*")):
        print(f"  {p.name}")
    print(f"  tab_validation_per_pp.csv ({len(df)} rows)")
    print(f"  tab_validation_summary.csv")


if __name__ == "__main__":
    main()
