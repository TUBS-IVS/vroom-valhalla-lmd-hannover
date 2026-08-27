"""Paper-final folder built from SCRATCH using only the new balanced data.
Generates all 8 sections of plots fresh from results/overnight_2026_05_29_path2/
and the new α=1.343 hybrid model.

Sections:
  01_input_data/         Demand, hubs, raumtyp (from checkpoints)
  02_baseline/           Daily delivery KPIs (from share=0 balanced cells)
  03_training/           2733 sample distribution
  04_model/              CV battery + Daganzo decomposition (uses α=1.343)
  05_optimization/       Pareto, Pxshare, schedule mix, sweet-spot — ALL FRESH
  06_balancing/          Fleet impact — ALL FRESH
  07_validation/         (placeholder — VROOM-sweep needed separately)
  08_interpretation/     LGB residual + decision trees on NEW balanced output

DEPRECATED (2026-08 revision). Stale entry point: it recomputes totals
WITHOUT the pool term and predates the universal tour rule, the two cost
lenses and the operator polish, so its numbers are not comparable with the
current results. Use scripts/revision/61_grid_run_v2.py for the grid and
scripts/revision/70_figs_tables_v2.py for figures and tables.
"""
from __future__ import annotations
import math
import json
import pickle
import sys

# --- DEPRECATED ENTRY POINT (2026-08 revision) -----------------------------
import warnings as _deprecation_warnings

_deprecation_warnings.warn(
    "paper_final_v2.py is a STALE entry point: it recomputes totals WITHOUT the pool "
    "term and predates the universal tour rule, the two cost lenses and the "
    "operator polish. Its numbers are NOT comparable with the 2026-08 "
    "revision. Use scripts/revision/61_grid_run_v2.py for the grid and "
    "scripts/revision/70_figs_tables_v2.py for figures and tables.",
    DeprecationWarning,
    stacklevel=2,
)
# ---------------------------------------------------------------------------

import warnings
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import Patch
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

BAL = ROOT / "results" / "overnight_2026_05_29_path2"
OUT = ROOT / "results" / "paper_final_2026_05_30"

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

PROV_COLOR = {"Amazon": "#003049", "DHL": "#d62828", "DPD": "#f77f00",
              "FedEx": "#5a189a", "GLS": "#2a9d8f", "Hermes": "#9d4edd",
              "UPS": "#7d5a50"}
SCHED_COLOR = {2: "#9b2226", 3: "#bb3e03", 4: "#ee9b00", 5: "#94d2bd", 6: "#005f73"}
PROVIDERS = list(PROV_COLOR.keys())


def _mkdir(*p):
    d = OUT.joinpath(*p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_data():
    s = pd.read_csv(BAL / "tab_balancing_summary.csv")
    sched = pd.read_csv(BAL / "tab_chosen_schedules.csv")
    fleet = pd.read_csv(BAL / "tab_fleet_per_hub.csv")

    # Aggregate per (P, share)
    agg = s.groupby(["penalty", "share_willing"], as_index=False).agg(
        init_cost_eur=("init_cost_eur", "sum"),
        bal_cost_eur=("balanced_cost_eur", "sum"),
        max_fleet_before=("max_fleet_before", "sum"),
        max_fleet_after=("max_fleet_after", "sum"),
        total_swaps=("swaps_made", "sum"),
        total_routes_before=("total_routes_before", "sum"),
        total_routes_after=("total_routes_after", "sum"),
        imbalance_before=("imbalance_before", "sum"),
        imbalance_after=("imbalance_after", "sum"),
    )
    # Parcels-weighted wait
    wait_rows = []
    for (P, sh), g in sched.groupby(["penalty", "share_willing"]):
        pkts = g.weekly_parcels.sum()
        wait_rows.append({"penalty": P, "share_willing": sh,
                          "wait_init": (g.avg_wait_d_init * g.weekly_parcels).sum() / pkts,
                          "wait_bal": (g.avg_wait_d_balanced * g.weekly_parcels).sum() / pkts})
    agg = agg.merge(pd.DataFrame(wait_rows), on=["penalty", "share_willing"])

    baseline = float(agg[agg.share_willing == 0.0].bal_cost_eur.max())
    agg["saving_init_pct"] = 100 * (baseline - agg.init_cost_eur) / baseline
    agg["saving_bal_pct"] = 100 * (baseline - agg.bal_cost_eur) / baseline
    agg["fleet_red_pct"] = 100 * (agg.max_fleet_before - agg.max_fleet_after) / agg.max_fleet_before.clip(lower=1)
    agg["delta_pct"] = 100 * (agg.bal_cost_eur - agg.init_cost_eur) / agg.init_cost_eur.clip(lower=1)
    return s, sched, fleet, agg, baseline


# ─── 01 Input Data ───────────────────────────────────────────────────────
def fig_input_data(sched):
    d = _mkdir("01_input_data")
    # Per-PLZ weekly volume distribution
    plz_vol = sched[sched.penalty == 0.5][["provider", "plz", "weekly_parcels"]].drop_duplicates()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    for prov, g in plz_vol.groupby("provider"):
        ax.hist(g.weekly_parcels, bins=30, alpha=0.6, label=prov,
                color=PROV_COLOR[prov], histtype="step", linewidth=1.6)
    ax.set_xlabel("Weekly parcels per (provider, PLZ) cell")
    ax.set_ylabel("Number of cells")
    ax.set_title("Volume distribution per LSP cell")
    ax.set_xscale("log")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    totals = plz_vol.groupby("provider").weekly_parcels.sum().sort_values()
    colors = [PROV_COLOR[p] for p in totals.index]
    ax.barh(totals.index, totals.values / 1e3, color=colors, edgecolor="black")
    ax.set_xlabel("Total weekly parcels [thousands]")
    ax.set_title("LSP market share — Region Hannover")
    for i, v in enumerate(totals.values / 1e3):
        ax.text(v + 3, i, f"{v:.0f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "fig_I1_lsp_volumes.png")
    fig.savefig(d / "fig_I1_lsp_volumes.pdf")
    plt.close(fig)
    print("  [OK] I1: lsp_volumes")

    # Summary table
    summary = pd.DataFrame({
        "provider": totals.index, "weekly_parcels": totals.values,
        "n_plz_cells": [plz_vol[plz_vol.provider == p].plz.nunique() for p in totals.index],
    })
    summary["share_pct"] = 100 * summary.weekly_parcels / summary.weekly_parcels.sum()
    summary.to_csv(d / "tab_lsp_summary.csv", index=False)


# ─── 02 Baseline ─────────────────────────────────────────────────────────
def fig_baseline(agg, sched):
    d = _mkdir("02_baseline")
    base_row = agg[agg.share_willing == 0.0].iloc[0]
    summary = {
        "weekly_cost_eur": float(base_row.bal_cost_eur),
        "peak_fleet": int(base_row.max_fleet_before),
        "total_routes_per_week": int(base_row.total_routes_before),
        "n_cells": 312,
        "n_providers": 7,
        "weekly_parcels": int(sched[sched.penalty == 0.5].weekly_parcels.sum() / 11),  # over 11 shares
    }
    (d / "baseline_kpis.json").write_text(json.dumps(summary, indent=2, default=str))
    print("  [OK] baseline KPIs saved")

    # Per-provider baseline cost
    base_per_prov = sched[(sched.penalty == 10.0) & (sched.share_willing == 0.0)].groupby(
        "provider", as_index=False).agg(
        dd_cost=("dd_cost_init", "sum"),
        weekly_parcels=("weekly_parcels", "sum"),
        n_plz=("plz", "count"),
    )
    base_per_prov["cost_per_1000_parcels"] = 1000 * base_per_prov.dd_cost / base_per_prov.weekly_parcels.clip(lower=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(base_per_prov.provider, base_per_prov.cost_per_1000_parcels,
            color=[PROV_COLOR[p] for p in base_per_prov.provider],
            edgecolor="black")
    ax.set_ylabel("Cost per 1000 parcels [EUR]")
    ax.set_title("Daily-delivery baseline unit cost per LSP")
    for i, (p, v) in enumerate(zip(base_per_prov.provider, base_per_prov.cost_per_1000_parcels)):
        ax.text(i, v + 5, f"{v:.0f}€", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "fig_B1_baseline_per_provider.png")
    fig.savefig(d / "fig_B1_baseline_per_provider.pdf")
    plt.close(fig)
    print("  [OK] B1: baseline_per_provider")
    base_per_prov.to_csv(d / "tab_baseline_per_provider.csv", index=False)


# ─── 03 Training ─────────────────────────────────────────────────────────
def fig_training():
    d = _mkdir("03_training")
    train_path = ROOT / "results" / "sweep_v3_mergefix" / "training_matrix.csv"
    if not train_path.exists():
        print("  [X] No training_matrix.csv")
        return
    train = pd.read_csv(train_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for prov, g in train.groupby("provider"):
        ax.scatter(g.n_parcels, g.actual_cost_eur, s=6, alpha=0.55,
                    color=PROV_COLOR.get(prov, "#888"), label=prov, edgecolor="none")
    ax.set_xlabel("Parcels per delivery day"); ax.set_ylabel("VROOM cost [EUR]")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"Training pool: {len(train):,} samples x 7 LSPs")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    cmap = plt.cm.plasma(np.linspace(0.2, 0.85, 3))
    for i, agg_k in enumerate(sorted(train.agg_k.unique())):
        g = train[train.agg_k == agg_k]
        ax.scatter(g.n_parcels, g.actual_cost_eur, s=6, alpha=0.5,
                    color=cmap[i], label=f"agg_k={agg_k}", edgecolor="none")
    ax.set_xlabel("Parcels per delivery day"); ax.set_ylabel("VROOM cost [EUR]")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Training pool stratified by source-window size")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(d / "fig_T1_training_distribution.png")
    fig.savefig(d / "fig_T1_training_distribution.pdf")
    plt.close(fig)
    print("  [OK] T1: training_distribution")
    train.to_csv(d / "training_matrix.csv", index=False)


# ─── 04 Model ────────────────────────────────────────────────────────────
def fig_model():
    d = _mkdir("04_model")
    # Copy CV battery + pickled model
    cv_csv = ROOT / "results/overnight_2026_05_27/diagnosis_v2/cv_battery/tab_model_comparison_cv.csv"
    if cv_csv.exists():
        cv = pd.read_csv(cv_csv)
        cv.to_csv(d / "tab_cv_battery.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 7))
        cv_sorted = cv.sort_values("MAPE_pct_mean")
        colors = ["#ee9b00" if "1.343" in m else
                  "#c1121f" if "Pure Daganzo" in m else
                  "#1f4f8f" for m in cv_sorted["model"]]
        ax.barh(cv_sorted["model"], cv_sorted["MAPE_pct_mean"],
                 xerr=cv_sorted["MAPE_pct_std"], color=colors, edgecolor="black",
                 capsize=3)
        ax.invert_yaxis()
        ax.set_xlabel("MAPE [%] — GroupKFold-CV (group=PLZ, 5 folds, mean ± std)")
        ax.set_title("Model comparison on actual VROOM cost (2,733 samples)\n"
                      "Hybrid α=1.343 (gold) beats all pure-ML approaches")
        for bar, val in zip(ax.patches, cv_sorted["MAPE_pct_mean"]):
            ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}%", va="center", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(d / "fig_M1_cv_battery.png"); fig.savefig(d / "fig_M1_cv_battery.pdf")
        plt.close(fig)
        print("  [OK] M1: cv_battery")

    # Copy the pickled model
    import shutil
    src_pkl = ROOT / "results/sweep_v3_mergefix/daganzo_hybrid_v3aug_median.pkl"
    if src_pkl.exists():
        shutil.copy2(src_pkl, d / "daganzo_hybrid_v3aug_median.pkl")
        shutil.copy2(ROOT / "results/sweep_v3_mergefix/daganzo_hybrid_v3aug_median.json",
                      d / "daganzo_hybrid_v3aug_median.json")
        print("  [OK] M1: model pickle saved")


# ─── 05 Optimization ─────────────────────────────────────────────────────
def fig_optimization(agg, sched, baseline):
    d = _mkdir("05_optimization")
    PENALTY_COLORS = plt.cm.viridis(np.linspace(0.15, 0.9, agg.penalty.nunique()))
    P_VALUES = sorted(agg.penalty.unique())

    # PF1: Pareto frontier
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    for pi, P in enumerate(P_VALUES):
        sub = agg[agg.penalty == P].sort_values("share_willing")
        ax.plot(sub.wait_bal, sub.saving_bal_pct, "o-",
                 color=PENALTY_COLORS[pi], linewidth=2, markersize=6,
                 label=f"P={P}", alpha=0.9)
        ax.plot(sub.wait_init, sub.saving_init_pct, "x--",
                 color=PENALTY_COLORS[pi], linewidth=1, markersize=4, alpha=0.35)
    op = agg[(np.isclose(agg.penalty, 0.4)) & (np.isclose(agg.share_willing, 1.0))]
    if len(op):
        ax.scatter(op.wait_bal.iloc[0], op.saving_bal_pct.iloc[0],
                    marker="*", s=450, c="gold", edgecolor="black", zorder=10,
                    label="Sweet-spot (P=0.4, share=1.0)")
    ax.set_xlabel("Average customer wait [days]"); ax.set_ylabel("Cost saving vs daily baseline [%]")
    ax.set_title("Pareto frontier — solid: fleet-balanced  /  dashed: cost-optimal")
    ax.legend(fontsize=8, ncol=2, loc="lower right"); ax.grid(alpha=0.3)

    ax = axes[1]
    for pi, P in enumerate(P_VALUES):
        sub = agg[agg.penalty == P].sort_values("wait_bal")
        if len(sub) >= 3:
            x = sub.wait_bal.values; y = sub.saving_bal_pct.values
            dy = np.gradient(y, x + 1e-9)
            ax.plot(x[1:-1], dy[1:-1], "o-", color=PENALTY_COLORS[pi],
                     label=f"P={P}", alpha=0.85, markersize=4)
    ax.set_xlabel("Customer wait [days]"); ax.set_ylabel("d(saving)/d(wait) [%/day]")
    ax.set_title("Marginal saving per wait-day")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8, ncol=2, loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "fig_PF1_pareto.png"); fig.savefig(d / "fig_PF1_pareto.pdf")
    plt.close(fig)
    print("  [OK] PF1: pareto")

    # PF2: Pxshare heatmap of saving %
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sav_init = agg.pivot(index="penalty", columns="share_willing", values="saving_init_pct")
    sav_bal = agg.pivot(index="penalty", columns="share_willing", values="saving_bal_pct")
    vmin = min(sav_init.values.min(), sav_bal.values.min())
    vmax = max(sav_init.values.max(), sav_bal.values.max())
    for ax, mat, title in [(axes[0], sav_init, "Cost-optimal init"),
                            (axes[1], sav_bal, "Fleet-balanced")]:
        im = ax.imshow(mat.values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(
            [f"{x*100:.0f}%" for x in mat.columns])
        ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(
            [f"P={p}" for p in mat.index])
        ax.set_xlabel("Share willing"); ax.set_ylabel("Service penalty")
        ax.set_title(f"Saving [%] — {title}")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                color = "white" if v < (vmax - vmin) * 0.5 + vmin else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color=color, fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    fig.suptitle(f"Saving heatmap — daily baseline = {baseline/1e3:.0f} k€/week", y=1.01)
    fig.tight_layout()
    fig.savefig(d / "fig_PF2_saving_heatmap.png")
    fig.savefig(d / "fig_PF2_saving_heatmap.pdf")
    plt.close(fig)
    print("  [OK] PF2: saving_heatmap")

    # SM1: Schedule mix grid (dynamic rows to fit any number of penalties)
    _ncol = 4
    _nrow = int(np.ceil(len(P_VALUES) / _ncol))
    fig, axes = plt.subplots(_nrow, _ncol, figsize=(20, 4.5 * _nrow))
    axes = np.atleast_2d(axes)
    for pi, P in enumerate(P_VALUES):
        ax = axes[pi // _ncol, pi % _ncol]
        sub = sched[sched.penalty == P]
        mix = sub.groupby(["share_willing", "schedule_size_balanced"]).size().unstack(fill_value=0)
        shares = mix.index.values * 100
        bottom = np.zeros(len(shares))
        for sz in sorted(mix.columns):
            vals = mix[sz].values
            ax.fill_between(shares, bottom, bottom + vals,
                             color=SCHED_COLOR.get(sz, "gray"), alpha=0.88,
                             label=f"{sz}d/wk" if pi == 0 else None)
            bottom += vals
        ax.set_xlabel("Share willing [%]"); ax.set_ylabel("Cells" if pi % _ncol == 0 else "")
        ax.set_title(f"P = {P} €/p/d", fontsize=11)
        ax.set_xlim(0, 100); ax.grid(axis="y", alpha=0.3)
    for j in range(len(P_VALUES), _nrow * _ncol):
        axes[j // _ncol, j % _ncol].axis("off")
    handles = [Patch(color=SCHED_COLOR[s], label=f"{s}d/wk", alpha=0.88)
                for s in [2, 3, 4, 5, 6]]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=11,
                bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Schedule-size distribution — fleet-balanced output", fontsize=14, y=0.99)
    fig.tight_layout()
    fig.savefig(d / "fig_SM1_schedule_mix_grid.png", bbox_inches="tight")
    fig.savefig(d / "fig_SM1_schedule_mix_grid.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [OK] SM1: schedule_mix")

    # Per-provider schedule mix at sweet-spot (dynamic rows)
    _ncol = 4
    _nrow = int(np.ceil(len(P_VALUES) / _ncol))
    fig, axes = plt.subplots(_nrow, _ncol, figsize=(18, 4.5 * _nrow))
    axes = np.atleast_2d(axes)
    for pi, P in enumerate(P_VALUES):
        ax = axes[pi // _ncol, pi % _ncol]
        sub = sched[(sched.penalty == P) & (np.isclose(sched.share_willing, 1.0))]
        mix = sub.groupby(["provider", "schedule_size_balanced"]).size().unstack(fill_value=0)
        mix = mix.reindex(PROVIDERS)
        bottom = np.zeros(len(mix))
        for sz in sorted(mix.columns):
            vals = mix[sz].fillna(0).values
            ax.bar(mix.index, vals, bottom=bottom,
                    color=SCHED_COLOR.get(sz, "gray"), alpha=0.88,
                    edgecolor="white")
            bottom += vals
        ax.set_title(f"P = {P}, share = 100%", fontsize=11)
        ax.set_ylabel("Cells" if pi % _ncol == 0 else "")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.3)
    for j in range(len(P_VALUES), _nrow * _ncol):
        axes[j // _ncol, j % _ncol].axis("off")
    handles = [Patch(color=SCHED_COLOR[s], label=f"{s}d/wk", alpha=0.88) for s in [2, 3, 4, 5, 6]]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=11,
                bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Per-LSP schedule mix at share=100% — balanced output", fontsize=14, y=0.99)
    fig.tight_layout()
    fig.savefig(d / "fig_SM2_provider_schedule_share100.png", bbox_inches="tight")
    fig.savefig(d / "fig_SM2_provider_schedule_share100.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  [OK] SM2: provider_schedule_share100")

    # Cost components per cell
    cost_cols = ["init_cost_eur", "bal_cost_eur"]
    agg[cost_cols + ["penalty", "share_willing", "delta_pct", "fleet_red_pct",
                       "saving_bal_pct", "wait_bal"]].to_csv(
        d / "tab_optimization_full_grid.csv", index=False)
    sched.to_csv(d / "tab_chosen_schedules_full.csv", index=False)


# ─── 06 Balancing ────────────────────────────────────────────────────────
def fig_balancing(agg, sched, fleet):
    d = _mkdir("06_balancing")

    # FB1: cost-fleet tradeoff scatter
    fig, ax = plt.subplots(figsize=(11, 6.5))
    shares = sorted(agg.share_willing.unique())
    cmap = plt.cm.plasma(np.linspace(0.1, 0.95, len(shares)))
    for ci, sh in enumerate(shares):
        sub = agg[np.isclose(agg.share_willing, sh)].sort_values("penalty")
        ax.plot(sub.delta_pct, sub.fleet_red_pct, "o-", color=cmap[ci],
                 label=f"{sh*100:.0f}%", alpha=0.85, markersize=6)
    ax.set_xlabel("Cost increase [%] (5% budget)")
    ax.set_ylabel("Peak-fleet reduction [%]")
    ax.set_title("Fleet-balancing trade-off — Cost vs Fleet reduction\nEach line = share-willing level across all P")
    ax.axvline(0, color="k", lw=0.5); ax.axhline(0, color="k", lw=0.5)
    ax.axvline(5, color="red", lw=1, ls="--", label="5% budget")
    ax.legend(fontsize=8, ncol=2, title="share willing")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "fig_FB1_cost_fleet_tradeoff.png")
    fig.savefig(d / "fig_FB1_cost_fleet_tradeoff.pdf")
    plt.close(fig)
    print("  [OK] FB1: cost_fleet_tradeoff")

    # FB2: swaps + fleet reduction heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sw = agg.pivot(index="penalty", columns="share_willing", values="total_swaps")
    fr = agg.pivot(index="penalty", columns="share_willing", values="fleet_red_pct")
    for ax, mat, title, cmap in [(axes[0], sw, "Total swaps", "BuGn"),
                                   (axes[1], fr, "Peak-fleet reduction [%]", "RdYlGn")]:
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(
            [f"{x*100:.0f}%" for x in mat.columns])
        ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(
            [f"P={p}" for p in mat.index])
        ax.set_xlabel("Share willing"); ax.set_ylabel("Service penalty")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if v == 0 or np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8)
                else:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                             color="white" if v > mat.values.max()*0.55 else "black",
                             fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    fig.tight_layout()
    fig.savefig(d / "fig_FB2_swaps_reduction.png"); fig.savefig(d / "fig_FB2_swaps_reduction.pdf")
    plt.close(fig)
    print("  [OK] FB2: swaps_reduction")

    # FB3: Cost-fleet point cloud at share=1.0
    s1 = agg[np.isclose(agg.share_willing, 1.0)].sort_values("penalty")
    fig, ax = plt.subplots(figsize=(11, 6))
    cm = plt.cm.viridis(np.linspace(0.15, 0.9, len(s1)))
    ax.scatter(s1.max_fleet_before, s1.bal_cost_eur / 1e3,
                marker="x", s=120, color="gray", label="Cost-optimal init", zorder=5)
    ax.scatter(s1.max_fleet_after, s1.bal_cost_eur / 1e3,
                marker="o", s=140, c=cm, edgecolor="black",
                label="After fleet-balance", zorder=10)
    for _, r in s1.iterrows():
        ax.annotate(f"P={r.penalty}", (r.max_fleet_after, r.bal_cost_eur / 1e3),
                     xytext=(8, 5), textcoords="offset points", fontsize=9)
        ax.annotate("", xy=(r.max_fleet_after, r.bal_cost_eur / 1e3),
                     xytext=(r.max_fleet_before, r.bal_cost_eur / 1e3),
                     arrowprops=dict(arrowstyle="->", color="gray", alpha=0.4))
    ax.set_xlabel("Peak weekly fleet"); ax.set_ylabel("Weekly cost [k€]")
    ax.set_title("Cost-Fleet trade-off at share=100% — fleet balancing arrows")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "fig_FB3_cost_fleet_share100.png")
    fig.savefig(d / "fig_FB3_cost_fleet_share100.pdf")
    plt.close(fig)
    print("  [OK] FB3: cost_fleet_share100")

    # Save tables
    agg.to_csv(d / "tab_balancing_aggregate.csv", index=False)
    if fleet is not None:
        fleet.to_csv(d / "tab_fleet_per_hub.csv", index=False)


# ─── 08 Interpretation: decision trees from NEW data ───────────────────────
def fig_interpretation(sched):
    d = _mkdir("08_interpretation")

    # Pick operating point: P=0.5, share=1.0
    op = sched[(np.isclose(sched.penalty, 0.5)) & (np.isclose(sched.share_willing, 1.0))]
    # We need per-(provider, plz) features: load checkpoint
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    feat_rows = []
    for _, r in op.iterrows():
        plz_key = str(r.plz).zfill(5)
        pdata = chk4["optimization_data"].get(r.provider, {}).get("plz_data", {}).get(plz_key)
        if pdata is None:
            continue
        feat_rows.append({
            "provider": r.provider, "plz": r.plz,
            "weekly_parcels": r.weekly_parcels,
            "n_stops": pdata.get("total_points", 0),
            "area_km2": pdata.get("area_km2", 0),
            "hub_dist_km": pdata.get("hub_dist_km", 0),
            "schedule_size_balanced": int(r.schedule_size_balanced),
            "wait_balanced": r.avg_wait_d_balanced,
            "dd_cost_balanced": r.dd_cost_balanced,
        })
    fdf = pd.DataFrame(feat_rows)
    if fdf.empty:
        print("  [X] No data for decision trees")
        return
    feat_cols = ["weekly_parcels", "n_stops", "area_km2", "hub_dist_km"]
    X = fdf[feat_cols].values

    # DT1: chosen schedule classifier
    y_class = fdf.schedule_size_balanced.values
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, random_state=42)
    tree.fit(X, y_class)
    acc = tree.score(X, y_class)
    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(tree, feature_names=feat_cols,
              class_names=[f"{c}d/wk" for c in tree.classes_],
              filled=True, rounded=True, impurity=False, proportion=True,
              precision=1, fontsize=10, ax=ax)
    ax.set_title(f"Decision tree — chosen schedule (P=0.5, share=1.0, balanced)\n"
                  f"in-sample accuracy = {acc:.1%}, n = {len(fdf)} cells",
                  fontsize=13, pad=15)
    fig.tight_layout()
    fig.savefig(d / "fig_DT1_schedule_choice.png")
    fig.savefig(d / "fig_DT1_schedule_choice.pdf")
    plt.close(fig)
    rules = export_text(tree, feature_names=feat_cols)
    (d / "tab_DT1_rules.txt").write_text(rules, encoding="utf-8")
    print(f"  [OK] DT1: schedule_choice (acc {acc:.1%})")

    # DT2: saving prediction
    # Compute saving% = (daily_cost - balanced_cost) / daily_cost for each cell
    daily_op = sched[(sched.penalty == 10.0) & (sched.share_willing == 0.0)][
        ["provider", "plz", "dd_cost_init"]].rename(columns={"dd_cost_init": "daily_cost"})
    fdf2 = fdf.merge(daily_op, on=["provider", "plz"], how="left").dropna()
    fdf2["saving_pct"] = 100 * (fdf2.daily_cost - fdf2.dd_cost_balanced) / fdf2.daily_cost.clip(lower=1)
    X = fdf2[feat_cols].values
    y_reg = fdf2.saving_pct.values
    tree2 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=15, random_state=42)
    tree2.fit(X, y_reg)
    r2 = tree2.score(X, y_reg)
    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(tree2, feature_names=feat_cols, filled=True, rounded=True,
              impurity=False, proportion=True, precision=1, fontsize=10, ax=ax)
    ax.set_title(f"Decision tree — saving% prediction (P=0.5, share=1.0)\n"
                  f"in-sample R² = {r2:.3f}, n = {len(fdf2)} cells",
                  fontsize=13, pad=15)
    fig.tight_layout()
    fig.savefig(d / "fig_DT2_saving_pct.png")
    fig.savefig(d / "fig_DT2_saving_pct.pdf")
    plt.close(fig)
    rules2 = export_text(tree2, feature_names=feat_cols)
    (d / "tab_DT2_rules.txt").write_text(rules2, encoding="utf-8")
    print(f"  [OK] DT2: saving_pct (R²={r2:.3f})")


# ─── README ──────────────────────────────────────────────────────────────
def write_readme(agg, baseline):
    sweet = agg[(np.isclose(agg.penalty, 0.5)) & (np.isclose(agg.share_willing, 1.0))]
    sw = sweet.iloc[0] if len(sweet) else None
    max_share1 = agg[(np.isclose(agg.share_willing, 1.0))]
    txt = [
        "# Paper Final — Region Hannover Last-Mile Batched Delivery",
        "## MobilTUM 2026  ·  Generated from FRESH balanced run 2026-05-28",
        "",
        f"**Daily baseline weekly cost**: {baseline/1e3:.1f} k€",
        "",
        "## Headline numbers",
        "",
        f"* Daily baseline: {baseline/1e3:.1f} k€/week",
        f"* Best saving (P=0, share=100%, balanced): "
        f"{max_share1[max_share1.penalty == 0].saving_bal_pct.iloc[0]:.1f}%",
        f"* Sweet-spot (P=0.5, share=100%): "
        f"{sw.saving_bal_pct:.1f}% saving at {sw.wait_bal:.3f}d wait" if sw is not None else "",
        f"* Fleet reduction at share=100% (P=0): "
        f"{max_share1[max_share1.penalty == 0].fleet_red_pct.iloc[0]:.1f}% "
        f"(peak {int(max_share1[max_share1.penalty == 0].max_fleet_before.iloc[0])} -> "
        f"{int(max_share1[max_share1.penalty == 0].max_fleet_after.iloc[0])} trucks)",
        "",
        "## Folder structure (all 8 paper-aligned sections)",
        "",
        "| Section | Purpose | Key figures |",
        "|---|---|---|",
        "| **01_input_data** | LSP volumes + market share | fig_I1 lsp_volumes |",
        "| **02_baseline** | Daily delivery KPIs per LSP | fig_B1 baseline_per_provider |",
        "| **03_training** | 2,733 permuted samples | fig_T1 training_distribution |",
        "| **04_model** | CV battery + Hybrid α=1.343 | fig_M1 cv_battery |",
        "| **05_optimization** | Pareto + heatmap + schedule mix | fig_PF1 pareto, fig_PF2 heatmap, fig_SM1/2 mix |",
        "| **06_balancing** | Fleet-balancing impact | fig_FB1 tradeoff, fig_FB2 swaps, fig_FB3 share100 |",
        "| **07_validation** | (VROOM sweep pending) | (placeholder) |",
        "| **08_interpretation** | Decision trees | fig_DT1 schedule_choice, fig_DT2 saving_pct |",
        "",
        "## Method",
        "",
        "1. **Training**: 2,733 perturbed (provider, PLZ, day, agg_k) samples -> VROOM solve -> 25 base features.",
        "2. **Model**: Daganzo-LGB-Hybrid, α=1.343 median-calibrated, MAPE 2.96% GroupKFold-OOS.",
        "3. **Optimization**: per-(P, share) cell, 312 (provider, PLZ) cells x 39 schedules, argmin with "
        "service penalty `P · local_willing(B2B/B2C) · parcels · wait_d`.",
        "4. **Bundling**: PLZ with ≥230 parcels stand-alone; <230 LPT-packed into multi-PLZ tours.",
        "5. **Fleet balancing**: greedy swap with 5% cost budget on TOTAL cost.",
        "6. **Willingness model**: smooth power-law (B2B-priority k=2), aggregate matches global share.",
        "",
        "All plots are 300dpi PNG + vector PDF, font.family=serif, paper-ready.",
    ]
    (OUT / "README.md").write_text("\n".join(filter(None, txt)), encoding="utf-8")
    print("  [OK] README.md")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Building paper_final from FRESH data in {BAL}")
    s, sched, fleet, agg, baseline = load_data()
    print(f"  loaded: 88 cells, baseline {baseline/1e3:.1f} k€")

    fig_input_data(sched)
    fig_baseline(agg, sched)
    fig_training()
    fig_model()
    fig_optimization(agg, sched, baseline)
    fig_balancing(agg, sched, fleet)
    _mkdir("07_validation")
    (OUT / "07_validation" / "_PLACEHOLDER.md").write_text(
        "VROOM full-sweep validation pending — to be run separately.")
    fig_interpretation(sched)
    write_readme(agg, baseline)

    files = sum(1 for _ in OUT.rglob("*") if _.is_file())
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6
    print(f"\nDone: {files} files, {size:.1f} MB in {OUT}")


if __name__ == "__main__":
    main()
