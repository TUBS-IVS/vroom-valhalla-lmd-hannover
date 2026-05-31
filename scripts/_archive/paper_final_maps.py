"""Regenerate spatial PLZ choropleth maps + delivery-frequency mix + value-of-
optimization on the NEW balanced data, into paper_final.

Adds:
  11_spatial_maps/  fig_MAP1 PLZ choropleth per P (median chosen freq, share=1.0)
                    fig_MAP2 PLZ choropleth saving% per P
  05_optimization/  fig_O5 delivery-freq mix vs P (stacked bar)
                    fig_O6 freq-mix vs share per P (multi-panel area)
                    fig_O7 value-of-optimization (best vs worst schedule)
"""
from __future__ import annotations
import pickle, sys, warnings
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
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
BAL = ROOT / "results" / "overnight_2026_05_27_balanced"
OUT = ROOT / "results" / "paper_final_2026_05_28"

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
FREQ_COLOR = {2: "#1d3557", 3: "#2a9d8f", 4: "#e9c46a", 5: "#f4a261", 6: "#e76f51"}
N_DAYS, MAX_HOLD = 6, 3


def enum_sched():
    out = []
    for k in range(1, N_DAYS + 1):
        for combo in combinations(range(N_DAYS), k):
            ds = sorted(combo); ok = True
            for i in range(len(ds)):
                gap = (ds[(i+1) % len(ds)] - ds[i]) % N_DAYS
                if gap == 0: gap = N_DAYS
                if gap > MAX_HOLD: ok = False; break
            if ok: out.append(frozenset(ds))
    return out


def load_geometry(chosen):
    import geopandas as gpd
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    plz_gdf = chk["gdf_plz"].copy()
    plz_gdf["plz"] = plz_gdf["plz"].astype(str).str.zfill(5)
    cl = pd.read_csv(ROOT / "data/geodata/plz_clusters.csv", dtype={"cluster_id": str})
    cl["members"] = cl["member_plz_list"].str.split(",")
    cl_long = cl.explode("members").rename(columns={"members": "plz"})
    cl_long["plz"] = cl_long["plz"].astype(str).str.zfill(5)
    cl_long["cluster_id"] = cl_long["cluster_id"].astype(str)
    plz_gdf = plz_gdf.merge(cl_long[["plz", "cluster_id"]], on="plz", how="left")
    plz_gdf["cluster_id"] = plz_gdf["cluster_id"].fillna(plz_gdf["plz"])
    in_scope = [str(p).zfill(5) for p in chosen.plz.astype(str).unique()]
    view = plz_gdf[plz_gdf["cluster_id"].isin(in_scope) | plz_gdf["plz"].isin(in_scope)].copy()
    return view


def fig_maps(chosen):
    import geopandas as gpd
    d = OUT / "11_spatial_maps"
    d.mkdir(parents=True, exist_ok=True)
    chosen["plz"] = chosen.plz.astype(str).str.zfill(5)
    pen_values = sorted(chosen.penalty.unique())
    view = load_geometry(chosen)

    # Per (P, cluster): median chosen freq at share=1.0
    op = chosen[np.isclose(chosen.share_willing, 1.0)]
    cmap = ListedColormap([FREQ_COLOR[s] for s in (2, 3, 4, 5, 6)])
    norm = BoundaryNorm([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], cmap.N)

    fig, axes = plt.subplots(1, len(pen_values), figsize=(4.2 * len(pen_values), 6))
    if len(pen_values) == 1:
        axes = [axes]
    for ax, P in zip(axes, pen_values):
        sub = op[op.penalty == P]
        med = sub.groupby("plz").schedule_size_balanced.median()
        vplot = view.copy()
        vplot["freq"] = vplot["cluster_id"].map(med).fillna(vplot["plz"].map(med))
        vplot_valid = vplot[vplot.freq.notna()]
        vplot_valid.plot(column="freq", ax=ax, cmap=cmap, norm=norm,
                          edgecolor="white", linewidth=0.3)
        vplot[vplot.freq.isna()].plot(ax=ax, color="#eeeeee", edgecolor="white", linewidth=0.3)
        ax.set_title(f"P = {P} €/p/d", fontsize=11)
        ax.axis("off")
    handles = [Patch(color=FREQ_COLOR[s], label=f"{s} d/wk") for s in (2, 3, 4, 5, 6)]
    fig.legend(handles=handles, title="Median chosen\ndelivery frequency",
                loc="center right", bbox_to_anchor=(1.005, 0.5), fontsize=10)
    fig.suptitle("Where do LSPs deliver how often? — PLZ median chosen frequency (share=100%)",
                  fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(d / "fig_MAP1_freq_choropleth_per_P.png", bbox_inches="tight")
    fig.savefig(d / "fig_MAP1_freq_choropleth_per_P.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ MAP1: freq_choropleth_per_P")

    # Map 2: saving % per PLZ per P
    daily = chosen[(chosen.penalty == 10.0) & (chosen.share_willing == 0.0)][
        ["provider", "plz", "dd_cost_init"]].rename(columns={"dd_cost_init": "daily"})
    fig, axes = plt.subplots(1, len(pen_values), figsize=(4.2 * len(pen_values), 6))
    if len(pen_values) == 1:
        axes = [axes]
    for ax, P in zip(axes, pen_values):
        sub = op[op.penalty == P].merge(daily, on=["provider", "plz"], how="left")
        sub["saving"] = 100 * (sub.daily - sub.dd_cost_balanced) / sub.daily.clip(lower=1)
        sav = sub.groupby("plz").saving.mean()
        vplot = view.copy()
        vplot["sav"] = vplot["cluster_id"].map(sav).fillna(vplot["plz"].map(sav))
        vv = vplot[vplot.sav.notna()]
        vv.plot(column="sav", ax=ax, cmap="RdYlGn", edgecolor="white",
                linewidth=0.3, vmin=0, vmax=40)
        vplot[vplot.sav.isna()].plot(ax=ax, color="#eeeeee", edgecolor="white", linewidth=0.3)
        ax.set_title(f"P = {P} €/p/d", fontsize=11)
        ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 40))
    fig.colorbar(sm, ax=axes, shrink=0.5, label="Saving %", pad=0.01)
    fig.suptitle("Spatial saving distribution — PLZ mean cost-saving (share=100%)",
                  fontsize=13, y=1.0)
    fig.savefig(d / "fig_MAP2_saving_choropleth_per_P.png", bbox_inches="tight")
    fig.savefig(d / "fig_MAP2_saving_choropleth_per_P.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ MAP2: saving_choropleth_per_P")


def fig_freq_mix(chosen):
    d = OUT / "05_optimization"
    pen_values = sorted(chosen.penalty.unique())

    # O5: delivery-freq mix vs P at share=1.0 (stacked bar)
    op = chosen[np.isclose(chosen.share_willing, 1.0)]
    fig, ax = plt.subplots(figsize=(11, 6))
    mix = op.groupby(["penalty", "schedule_size_balanced"]).size().unstack(fill_value=0)
    x = np.arange(len(mix.index))
    bottom = np.zeros(len(mix))
    for sz in sorted(mix.columns):
        ax.bar(x, mix[sz].values, bottom=bottom, color=FREQ_COLOR.get(int(sz), "gray"),
                label=f"{int(sz)} d/wk", edgecolor="white", width=0.85)
        bottom += mix[sz].values
    ax.set_xticks(x); ax.set_xticklabels([f"{p}" for p in mix.index])
    ax.set_xlabel("Service penalty P [€/parcel/day]")
    ax.set_ylabel("Count of (provider, PLZ) cells")
    ax.set_title("Delivery-frequency mix shifts with service penalty (share=100%)")
    ax.legend(title="Delivery days/week", loc="center right", bbox_to_anchor=(1.18, 0.5))
    fig.tight_layout()
    fig.savefig(d / "fig_O5_freq_mix_vs_P.png"); fig.savefig(d / "fig_O5_freq_mix_vs_P.pdf")
    plt.close(fig)
    print("  ✓ O5: freq_mix_vs_P")

    # O6: freq mix vs share per P (multi-panel stacked area %)
    fig, axes = plt.subplots(1, len(pen_values), figsize=(3.0 * len(pen_values), 4))
    if len(pen_values) == 1:
        axes = [axes]
    for ax, P in zip(axes, pen_values):
        sub = chosen[chosen.penalty == P]
        mix = sub.groupby(["share_willing", "schedule_size_balanced"]).size().unstack(fill_value=0)
        mix_pct = 100 * mix.div(mix.sum(axis=1), axis=0)
        shares = mix_pct.index.values * 100
        bottom = np.zeros(len(shares))
        for sz in sorted(mix_pct.columns):
            ax.fill_between(shares, bottom, bottom + mix_pct[sz].values,
                             color=FREQ_COLOR.get(int(sz), "gray"), alpha=0.9)
            bottom += mix_pct[sz].values
        ax.set_title(f"P = {P}", fontsize=10)
        ax.set_xlabel("Share willing [%]", fontsize=9)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    axes[0].set_ylabel("Delivery-freq mix [%]")
    handles = [Patch(color=FREQ_COLOR[s], label=f"{s}d/wk") for s in (2, 3, 4, 5, 6)]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.suptitle("Delivery-frequency mix vs willingness, across service penalties", y=1.02)
    fig.tight_layout()
    fig.savefig(d / "fig_O6_freq_mix_vs_share_per_P.png", bbox_inches="tight")
    fig.savefig(d / "fig_O6_freq_mix_vs_share_per_P.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ O6: freq_mix_vs_share_per_P")


def fig_value_of_optimization(chosen):
    """O7: value of optimization = best vs worst schedule cost, recomputed."""
    from penalty_sweep_pareto import build_pp_list, compute_sched_cost, load_model, avg_wait_days
    from batch_delivery.io.demand import get_source_days
    d = OUT / "05_optimization"

    schedules = enum_sched()
    sched_sizes = np.array([len(s) for s in schedules])
    sched_waits = np.array([avg_wait_days(sorted(s)) for s in schedules])
    sched_src = [{dd: get_source_days(dd, sorted(s)) for dd in sorted(s)} for s in schedules]
    pp_list = build_pp_list()
    model = load_model(ROOT / "results/sweep_v3_mergefix/daganzo_hybrid_v3aug_median.pkl")
    sched_cost = compute_sched_cost(model, pp_list, schedules, sched_src)
    n_pp = len(pp_list)
    pkts = np.array([pp["weekly_parcels"] for pp in pp_list], dtype=np.float64)

    P_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
    SHARE_GRID = np.arange(0, 1.01, 0.1)
    rows = []
    for P in P_GRID:
        for sh in SHARE_GRID:
            # objective = cost + P*share*pkts*wait
            obj = sched_cost + P * sh * pkts[:, None] * sched_waits[None, :]
            # batch_share = fraction of cells with schedule < 6
            # BEST = argmin obj; WORST = argmax cost among non-daily
            best_idx = obj.argmin(axis=1)
            best_cost = sched_cost[np.arange(n_pp), best_idx].sum()
            # worst (naive) = pick the most expensive non-daily schedule
            nondaily_mask = sched_sizes < 6
            worst_cost_per_cell = np.where(nondaily_mask[None, :], sched_cost, -np.inf).max(axis=1)
            # batch share = how many cells are non-daily in best solution
            batch_share = 100 * (sched_sizes[best_idx] < 6).mean()
            extra = (worst_cost_per_cell.sum() - best_cost) / 1e3
            rows.append({"penalty": P, "share": sh, "batch_share_pct": batch_share,
                          "extra_cost_kE": extra})
    vdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(P_GRID)))
    for ci, P in enumerate(P_GRID):
        sub = vdf[vdf.penalty == P].sort_values("batch_share_pct")
        ax.plot(sub.batch_share_pct, sub.extra_cost_kE, "o-", color=cmap[ci],
                 label=f"P = {P}", linewidth=2, markersize=6)
    ax.set_xlabel("Batch share [% of (provider, PLZ) cells batched]")
    ax.set_ylabel("Extra weekly cost when picking WORST schedule [k€]")
    ax.set_title("How much does optimization save vs naive schedule choice?")
    ax.legend(title="Service penalty")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "fig_O7_value_of_optimization.png")
    fig.savefig(d / "fig_O7_value_of_optimization.pdf")
    plt.close(fig)
    print("  ✓ O7: value_of_optimization")
    vdf.to_csv(d / "tab_value_of_optimization.csv", index=False)


def main():
    print("Regenerating spatial maps + freq-mix + value-of-optimization on NEW data...")
    chosen = pd.read_csv(BAL / "tab_chosen_schedules.csv")
    try:
        fig_maps(chosen.copy())
    except Exception as e:
        print(f"  ✗ maps failed: {e}")
    fig_freq_mix(chosen.copy())
    fig_value_of_optimization(chosen.copy())
    print(f"\nDone. {sum(1 for _ in OUT.rglob('*') if _.is_file())} files")


if __name__ == "__main__":
    main()
