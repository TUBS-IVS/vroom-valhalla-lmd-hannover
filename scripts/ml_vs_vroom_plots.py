"""ML-prediction vs VROOM-actual ground-truth analysis for the final pipeline.

Reads the per-(scenario, provider, PLZ, day) table produced by
``run_final_optimization.py`` and produces paper-ready figures + tables.

Input:
    results/final_optimization/ml_vs_vroom_per_day.csv
        Columns: scenario, provider, plz, day_idx, weekday, delivers_on_day,
                 schedule_size, schedule_days, ml_pred_cost_eur,
                 vroom_actual_cost_eur, vroom_n_routes, vroom_n_parcels, ...

Outputs in results/final_optimization/figures/ and tables/:
    figO1  ML pred vs VROOM actual scatter, all (PLZ, day) tuples, by scenario
    figO2  Per-carrier MAPE, best vs worst, bar chart
    figO3  Per-weekday MAPE pattern
    figO4  Residual histogram (best vs worst)
    figO5  Top-20 worst (PLZ, day) predictions by absolute residual
    figO6  Schedule-size distribution per scenario (best vs worst)
    figO7  Cost saved by carrier: baseline vs best (waterfall)
    figO8  Per-(LSP, day, scenario) heatmap of mean MAPE
    figO9  Pareto: cost vs avg-waiting-days across scenarios
    figO10 ML-prediction calibration for the production scenarios (best & worst)

Re-run anytime:
    python scripts/ml_vs_vroom_plots.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_helpers import safe_metrics, apply_style, PALETTE, PROVIDERS

apply_style()

OUT_DIR = ROOT / "results" / "final_optimization"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=160)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf/.png")


def main():
    gt_path = OUT_DIR / "ml_vs_vroom_per_day.csv"
    if not gt_path.exists():
        raise FileNotFoundError(f"{gt_path} not found — run run_final_optimization.py first")
    gt = pd.read_csv(gt_path)
    kpi_path = OUT_DIR / "scenario_comparison_kpis_with_worst.csv"
    df_kpi = pd.read_csv(kpi_path) if kpi_path.exists() else None
    print(f"loaded ground-truth: {len(gt):,} rows  "
           f"({gt['scenario'].nunique()} scenarios, {gt['provider'].nunique()} LSPs, "
           f"{gt['plz'].nunique()} PLZs)")

    # restrict to delivery days for the calibration analysis
    gt_d = gt[gt["delivers_on_day"] & (gt["vroom_actual_cost_eur"] > 0)
               & (gt["ml_pred_cost_eur"] > 0)].copy()
    gt_d["resid_pct"] = (gt_d["ml_pred_cost_eur"] - gt_d["vroom_actual_cost_eur"]) \
                          / gt_d["vroom_actual_cost_eur"] * 100
    gt_d["abs_err_eur"] = (gt_d["ml_pred_cost_eur"] - gt_d["vroom_actual_cost_eur"]).abs()

    # Scenario style table (covers Best / Fixed / Avg / Worst — batch only flavour)
    sc_style = {
        "SA_ML Batch-Only":  {"color": "#009E73", "label": "SA_ML Batch (best)"},
        "Fixed Batch-Only":  {"color": "#0072B2", "label": "Fixed Batch (carrier)"},
        "Avg Batch-Only":    {"color": "#999999", "label": "Avg Batch (median)"},
        "Worst Batch-Only":  {"color": "#D55E00", "label": "Worst Batch (max)"},
        # (Express variants kept for backward-compat — silently skipped if not in data)
        "SA_ML + Express":   {"color": "#56B4E9", "label": "SA_ML+Express"},
        "Worst + Express":   {"color": "#CC79A7", "label": "Worst+Express"},
    }
    scenarios = [s for s in sc_style if s in gt_d["scenario"].unique()]

    # ─── figO1: ML pred vs VROOM actual scatter (per scenario, log-log) ────
    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6), squeeze=False)
    for ax, sc in zip(axes.flat, scenarios):
        sub = gt_d[gt_d["scenario"] == sc]
        ax.scatter(sub["vroom_actual_cost_eur"], sub["ml_pred_cost_eur"],
                    s=4, alpha=0.45, color=sc_style[sc]["color"],
                    edgecolors="none", rasterized=True)
        lo = max(1.0, sub["vroom_actual_cost_eur"].min() * 0.9)
        hi = sub["vroom_actual_cost_eur"].max() * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        m = safe_metrics(sub["vroom_actual_cost_eur"], sub["ml_pred_cost_eur"])
        ax.set_title(f"{sc_style[sc]['label']}\nMAPE={m['mape']:.2f}%, n={m['n']:,}",
                       loc="left", fontsize=8.5)
        ax.set_xlabel("VROOM cost  [€/day]")
        if ax is axes.flat[0]:
            ax.set_ylabel("ML predicted cost  [€/day]")
    fig.suptitle("Fig O1 — ML prediction vs VROOM actual per (PLZ, day) — best & worst scenarios",
                  x=0.005, ha="left", fontsize=10)
    _save(fig, "figO1_pred_vs_actual_per_scenario")

    # ─── figO2: per-carrier MAPE, best vs worst ────
    per_lsp = (gt_d.groupby(["scenario", "provider"])
                .apply(lambda g: safe_metrics(g["vroom_actual_cost_eur"], g["ml_pred_cost_eur"])["mape"])
                .reset_index(name="mape"))
    per_lsp.to_csv(TAB_DIR / "tabO1_mape_per_scenario_provider.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    xs = np.arange(len(PROVIDERS))
    w = 0.8 / max(1, len(scenarios))
    for i, sc in enumerate(scenarios):
        vals = [per_lsp[(per_lsp.scenario == sc) & (per_lsp.provider == p)]["mape"].iloc[0]
                if len(per_lsp[(per_lsp.scenario == sc) & (per_lsp.provider == p)]) else np.nan
                for p in PROVIDERS]
        ax.bar(xs + (i - (len(scenarios) - 1) / 2) * w, vals, w,
                color=sc_style[sc]["color"], edgecolor="k", lw=0.3,
                label=sc_style[sc]["label"])
    ax.set_xticks(xs); ax.set_xticklabels(PROVIDERS, fontsize=9)
    ax.set_ylabel("MAPE  [%]"); ax.set_yscale("log")
    ax.set_title("Fig O2 — ML prediction MAPE per carrier, head-to-head best vs worst", loc="left")
    ax.legend(fontsize=7)
    _save(fig, "figO2_mape_per_carrier")

    # ─── figO3: per-weekday MAPE pattern ────
    per_wd = (gt_d.groupby(["scenario", "weekday"])
                .apply(lambda g: safe_metrics(g["vroom_actual_cost_eur"], g["ml_pred_cost_eur"])["mape"])
                .reset_index(name="mape"))
    per_wd.to_csv(TAB_DIR / "tabO2_mape_per_weekday.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    wd_order = ["Mo", "Tu", "We", "Th", "Fr", "Sa"]
    for sc in scenarios:
        d = per_wd[per_wd.scenario == sc].set_index("weekday").reindex(wd_order)
        ax.plot(wd_order, d["mape"], "o-", color=sc_style[sc]["color"],
                 label=sc_style[sc]["label"], lw=1.2, markersize=5)
    ax.set_ylabel("MAPE  [%]"); ax.set_xlabel("weekday")
    ax.legend(fontsize=7)
    ax.set_title("Fig O3 — MAPE per weekday across scenarios", loc="left")
    _save(fig, "figO3_mape_per_weekday")

    # ─── figO4: residual histogram (best vs worst) ────
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))
    best_scs   = [s for s in scenarios if "SA_ML" in s]
    worst_scs  = [s for s in scenarios if "Worst" in s]
    for ax, group, title in [(axes[0], best_scs,  "Best (ML-optimised) scenarios"),
                              (axes[1], worst_scs, "Worst (anti-optimised) scenarios")]:
        for sc in group:
            r = gt_d[gt_d.scenario == sc]["resid_pct"].clip(-50, 50)
            ax.hist(r, bins=60, alpha=0.55, color=sc_style[sc]["color"],
                     label=f"{sc_style[sc]['label']}  med={r.median():+.1f}%",
                     edgecolor="white", lw=0.2)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("rel. residual (pred − VROOM)/VROOM  [%]")
        ax.set_ylabel("count")
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.legend(fontsize=7)
    fig.suptitle("Fig O4 — ML-prediction residual distribution (per-day, clipped to ±50%)",
                  x=0.005, ha="left")
    _save(fig, "figO4_residual_histogram")

    # ─── figO5: top-20 worst predictions by abs error ────
    worst = (gt_d.sort_values("abs_err_eur", ascending=False).head(20).iloc[::-1])
    worst["label"] = worst["provider"] + "·" + worst["plz"].astype(str) + "·" + worst["weekday"] + "·" + worst["scenario"].str.replace("SA_ML ", "SA-")
    fig, ax = plt.subplots(figsize=(7.16, 4.5))
    colors = [PALETTE.get(p, "#444") for p in worst["provider"]]
    ax.barh(worst["label"], worst["abs_err_eur"], color=colors, edgecolor="k", lw=0.4)
    for i, (e, p, a) in enumerate(zip(worst["abs_err_eur"],
                                         worst["ml_pred_cost_eur"],
                                         worst["vroom_actual_cost_eur"])):
        ax.text(e, i, f"  pred={p:.0f}€ act={a:.0f}€", va="center", fontsize=7)
    ax.set_xlabel("|ML pred − VROOM actual|  [€]")
    ax.set_title("Fig O5 — Top-20 worst per-day predictions", loc="left")
    _save(fig, "figO5_top20_worst_predictions")
    worst.to_csv(TAB_DIR / "tabO3_worst_20_per_day_residuals.csv", index=False)

    # ─── figO6: schedule-size distribution per scenario ────
    sched_size = (gt[["scenario", "provider", "plz", "schedule_size"]]
                  .drop_duplicates(["scenario", "provider", "plz"]))
    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    for sc in scenarios:
        sub = sched_size[sched_size.scenario == sc]
        vc = sub["schedule_size"].value_counts().sort_index()
        ax.bar(vc.index + (scenarios.index(sc) - (len(scenarios) - 1) / 2) * 0.18,
                vc.values, 0.18, color=sc_style[sc]["color"], edgecolor="k", lw=0.3,
                label=sc_style[sc]["label"])
    ax.set_xlabel("schedule size  [delivery days / week]")
    ax.set_ylabel("number of (LSP, PLZ) tuples")
    ax.set_title("Fig O6 — Chosen schedule-size distribution per scenario", loc="left")
    ax.legend(fontsize=7)
    _save(fig, "figO6_schedule_size_distribution")

    # ─── figO7: cost waterfall — baseline -> best -> worst per carrier ───
    if df_kpi is not None:
        # Total cost per scenario
        try:
            scenarios_for_water = ["Baseline", "SA_ML + Express", "SA_ML Batch-Only",
                                     "Worst + Express", "Worst Batch-Only"]
            df_reset = df_kpi.reset_index() if df_kpi.index.name else df_kpi
            label_col = "scenario" if "scenario" in df_reset.columns else df_reset.columns[0]
            totals = {row[label_col]: float(row["cost_eur"])
                       for _, row in df_reset.iterrows()}
            order = [s for s in scenarios_for_water if s in totals]
            costs = [totals[s] for s in order]
            colors = ["#888888"] + [sc_style.get(s, {"color": "#555"})["color"] for s in order[1:]]

            fig, ax = plt.subplots(figsize=(7.16, 3.2))
            xs = np.arange(len(order))
            ax.bar(xs, costs, color=colors, edgecolor="k", lw=0.4)
            for i, (c, s) in enumerate(zip(costs, order)):
                delta_pct = 100 * (c - totals["Baseline"]) / totals["Baseline"] if "Baseline" in totals else 0
                ax.text(i, c, f"{c/1000:.1f}k€\n{delta_pct:+.1f}%",
                          ha="center", va="bottom", fontsize=8)
            ax.set_xticks(xs); ax.set_xticklabels(order, rotation=15, fontsize=8.5)
            ax.set_ylabel("Total weekly cost  [€]")
            ax.set_title("Fig O7 — Total weekly cost across scenarios (vs baseline %)",
                          loc="left")
            _save(fig, "figO7_total_cost_waterfall")
        except Exception as exc:
            print(f"  figO7 skipped: {exc}")

    # ─── figO8: heatmap (LSP × scenario) of mean MAPE ───
    pivot = per_lsp.pivot(index="provider", columns="scenario", values="mape")
    pivot = pivot.reindex(PROVIDERS)
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    im = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=8, rotation=15)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                          color="white" if v > 4 else "k", fontsize=8)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cb.set_label("MAPE  [%]")
    ax.set_title("Fig O8 — Per-(LSP × scenario) MAPE heatmap", loc="left")
    _save(fig, "figO8_lsp_x_scenario_mape_heatmap")
    pivot.to_csv(TAB_DIR / "tabO4_mape_lsp_x_scenario_pivot.csv")

    # ─── figO9: Pareto cost vs avg-waiting ───
    if df_kpi is not None:
        try:
            df_reset = df_kpi.reset_index() if df_kpi.index.name else df_kpi
            label_col = "scenario" if "scenario" in df_reset.columns else df_reset.columns[0]
            wait_col = ("avg_waiting_days" if "avg_waiting_days" in df_reset.columns
                          else "customer_wait_days")
            if wait_col in df_reset.columns and "cost_eur" in df_reset.columns:
                fig, ax = plt.subplots(figsize=(5.5, 3.2))
                for _, row in df_reset.iterrows():
                    label = str(row[label_col])
                    color = sc_style.get(label, {"color": "#444"})["color"]
                    ax.scatter(row[wait_col], row["cost_eur"], s=80, color=color,
                                edgecolors="k", linewidths=0.6)
                    ax.annotate(label, (row[wait_col], row["cost_eur"]),
                                  xytext=(6, 4), textcoords="offset points", fontsize=8)
                ax.set_xlabel("Avg. customer waiting days")
                ax.set_ylabel("Total weekly cost  [€]")
                ax.set_title("Fig O9 — Cost vs service-quality Pareto", loc="left")
                _save(fig, "figO9_cost_vs_wait_pareto")
        except Exception as exc:
            print(f"  figO9 skipped: {exc}")

    # ─── figO10: combined calibration grouped (defensive vs empty/NaN groups) ───
    groups_for_panels = [
        (best_scs,  "Best (ML-optimised) + Fixed"),
        (worst_scs, "Worst (anti-optimised)"),
    ]
    # if any group is empty, just show all data in one panel
    groups_for_panels = [g for g in groups_for_panels if g[0] and len(gt_d[gt_d["scenario"].isin(g[0])]) > 0]
    if len(groups_for_panels) == 0:
        groups_for_panels = [(scenarios, "All scenarios combined")]
    fig, axes = plt.subplots(1, len(groups_for_panels),
                                figsize=(3.5 * len(groups_for_panels), 3.4),
                                squeeze=False)
    for ax, (group, title) in zip(axes.flat, groups_for_panels):
        sub = gt_d[gt_d["scenario"].isin(group)]
        if len(sub) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        ax.scatter(sub["vroom_actual_cost_eur"], sub["ml_pred_cost_eur"],
                    s=4, alpha=0.4, c=sub["scenario"].map({sc: sc_style[sc]["color"]
                                                            for sc in group}),
                    edgecolors="none", rasterized=True)
        vmin = max(1.0, float(np.nanmin(sub["vroom_actual_cost_eur"])) * 0.9)
        vmax = float(np.nanmax(sub["vroom_actual_cost_eur"])) * 1.05
        if not (np.isfinite(vmin) and np.isfinite(vmax) and vmin < vmax):
            vmin, vmax = 1.0, 10000.0
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=0.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax); ax.set_aspect("equal")
        m = safe_metrics(sub["vroom_actual_cost_eur"], sub["ml_pred_cost_eur"])
        ax.set_title(f"{title}  n={m['n']:,}  MAPE={m['mape']:.2f}%  R²={m['r2']:.4f}",
                       loc="left", fontsize=9)
        ax.set_xlabel("VROOM cost  [€/day]")
        if ax is axes.flat[0]:
            ax.set_ylabel("ML predicted cost  [€/day]")
    fig.suptitle("Fig O10 — Combined calibration: ML pred vs VROOM actual on production scenarios",
                  x=0.005, ha="left", fontsize=10)
    _save(fig, "figO10_combined_calibration_best_worst")

    # ─── figO11: Best/Avg/Worst BAND plot per LSP ───
    # Total weekly cost per (scenario, LSP) from VROOM actuals
    if df_kpi is not None:
        try:
            df_reset = df_kpi.reset_index() if df_kpi.index.name else df_kpi
            # Need per-LSP breakdown — derive from gt by summing delivery-day VROOM costs
            band = (gt[gt["delivers_on_day"] & (gt["vroom_actual_cost_eur"] > 0)]
                    .groupby(["scenario", "provider"])["vroom_actual_cost_eur"].sum()
                    .reset_index())
            band_pivot = band.pivot(index="provider", columns="scenario",
                                       values="vroom_actual_cost_eur").reindex(PROVIDERS)
            band_pivot.to_csv(TAB_DIR / "tabO5_weekly_cost_per_lsp_per_scenario.csv")

            fig, ax = plt.subplots(figsize=(7.16, 3.4))
            xs = np.arange(len(PROVIDERS))
            best_col = "SA_ML Batch-Only"
            avg_col  = "Avg Batch-Only"
            worst_col = "Worst Batch-Only"
            if all(c in band_pivot.columns for c in [best_col, avg_col, worst_col]):
                best   = band_pivot[best_col].values
                avg    = band_pivot[avg_col].values
                worst  = band_pivot[worst_col].values
                # band from best to worst
                ax.fill_between(xs, best, worst, alpha=0.18, color="#888888",
                                  label="best-worst range")
                ax.plot(xs, worst, "s-", color="#D55E00", lw=1.4, markersize=6,
                          label="Worst (anti-optimised)")
                ax.plot(xs, avg,   "o-", color="#999999", lw=1.4, markersize=6,
                          label="Avg (median schedule)")
                ax.plot(xs, best,  "v-", color="#009E73", lw=1.6, markersize=6,
                          label="SA_ML Batch (best)")
                # Fixed schedule overlay
                if "Fixed Batch-Only" in band_pivot.columns:
                    fixed = band_pivot["Fixed Batch-Only"].values
                    ax.plot(xs, fixed, "x--", color="#0072B2", lw=1.0, markersize=8,
                              label="Fixed Batch (carrier)")
                ax.set_xticks(xs); ax.set_xticklabels(PROVIDERS, fontsize=9)
                ax.set_ylabel("Total weekly VROOM cost per LSP  [€]")
                ax.set_title("Fig O11 — Best vs Average vs Worst schedule  (weekly cost per LSP)",
                              loc="left")
                ax.legend(fontsize=8, loc="upper right")
                # annotate gap percentages
                for i, lsp in enumerate(PROVIDERS):
                    if i < len(best) and best[i] > 0:
                        gap = 100 * (worst[i] - best[i]) / best[i]
                        ax.annotate(f"+{gap:.0f}%", xy=(i, worst[i]),
                                      xytext=(0, 5), textcoords="offset points",
                                      ha="center", fontsize=7, color="#D55E00")
            _save(fig, "figO11_best_avg_worst_band")
        except Exception as exc:
            print(f"  figO11 skipped: {exc}")

    # ─── headline numbers ───
    headline = {}
    for sc in scenarios:
        sub = gt_d[gt_d.scenario == sc]
        m = safe_metrics(sub["vroom_actual_cost_eur"], sub["ml_pred_cost_eur"])
        headline[sc] = {"n_obs": m["n"], "mape": round(m["mape"], 3),
                          "r2": round(m["r2"], 4), "bias_pct": round(m["bias"], 3),
                          "mae_eur": round(m["mae"], 2)}
    (OUT_DIR / "ml_vs_vroom_headline.json").write_text(
        json.dumps(headline, indent=2, default=str))
    print(f"\nHeadline ML-vs-VROOM calibration per scenario:")
    print(json.dumps(headline, indent=2, default=str))


if __name__ == "__main__":
    main()
