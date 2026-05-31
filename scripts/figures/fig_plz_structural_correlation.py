"""Per-(provider, PLZ) knee-saving correlated with structural features.

Mechanistic complement to the provider-heterogeneity finding: shows that
the per-PLZ consolidation potential (knee saving %) is explained by three
structural drivers that operate ABOVE the LSP level:
  - hub distance       (longer stem -> more savings)
  - PLZ area           (more dispersion -> more savings)
  - parcels per stop   (denser drops -> LESS savings, the "DHL effect")

Uses the cached 312 x 39 Daganzo-Hybrid cost matrix (penalty-independent
cost-only path) plus the PLZ-level structural metadata in the
optimization-prep checkpoint. No new model calls.

Per-cell knee = chord-distance geometric knee on its own cost-vs-wait
Pareto frontier (same method as the system-level sweet-spot identification).

Output (results/EWGT_Results/, flat):
  fig_PLZ_structural_correlation.{png,pdf}
  tab_PLZ_knee_with_features.csv
"""
from __future__ import annotations
import pickle
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "EWGT_Results"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

N_DAYS = 6
MAX_HOLD = 3

RAUMTYP_PALETTE = {
    "urban":    "#1d3557",
    "suburban": "#2a9d8f",
    "rural":    "#e76f51",
}
RAUMTYP_ORDER = ["urban", "suburban", "rural"]


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


def avg_wait(s):
    if not s:
        return 0.0
    ds = sorted(s)
    total = 0.0
    for di in range(N_DAYS):
        next_dd = min(((d - di) % N_DAYS, d) for d in ds)[1]
        total += (next_dd - di) % N_DAYS
    return total / N_DAYS


def chord_knee(costs: np.ndarray, waits: np.ndarray) -> int:
    """Chord-distance geometric knee index on a 2-D Pareto frontier.
    Returns the index of the point with maximum orthogonal distance
    to the chord between the cost-min and wait-min extremes."""
    if len(costs) < 3:
        return int(np.argmin(costs))
    cn = (costs - costs.min()) / (costs.max() - costs.min() + 1e-12)
    wn = (waits - waits.min()) / (waits.max() - waits.min() + 1e-12)
    i_lo = int(np.argmin(wn))
    i_hi = int(np.argmin(cn))
    P1 = np.array([wn[i_lo], cn[i_lo]])
    P2 = np.array([wn[i_hi], cn[i_hi]])
    v = P2 - P1
    L = np.linalg.norm(v) + 1e-12
    best_i, best_d = i_lo, -np.inf
    for i in range(len(costs)):
        Q = np.array([wn[i], cn[i]])
        d = abs(v[0] * (Q[1] - P1[1]) - v[1] * (Q[0] - P1[0])) / L
        if d > best_d:
            best_d, best_i = d, i
    return best_i


def main():
    cache = np.load(ROOT / "results/penalty_sweep/sched_cost_cache.npz")
    sched_cost = cache["sched_cost"]
    prov_order = [str(x) for x in cache["prov_order"]]
    plz_order = [str(x) for x in cache["plz_order"]]
    n_pp = sched_cost.shape[0]
    print(f"Loaded sched_cost matrix: {sched_cost.shape}")

    schedules = enumerate_schedules()
    sched_sizes = np.array([len(s) for s in schedules])
    sched_waits = np.array([avg_wait(s) for s in schedules])
    daily_idx = int(np.where(sched_sizes == N_DAYS)[0][0])

    prep = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl",
                             "rb"))["optimization_data"]

    rows = []
    for i, (prov, plz) in enumerate(zip(prov_order, plz_order)):
        if prov not in prep or plz not in prep[prov]["plz_data"]:
            continue
        pd_meta = prep[prov]["plz_data"][plz]
        weekly_parcels = (sum(pd_meta["b2c"].values())
                          + sum(pd_meta["b2b"].values()))
        n_stops_per_day = float(pd_meta["n_stops_per_day"])
        stops_per_week = n_stops_per_day * N_DAYS
        parcels_per_stop = (weekly_parcels / stops_per_week
                            if stops_per_week > 0 else np.nan)
        hub_dist = float(pd_meta["hub_dist_km"])
        area = float(pd_meta["area_km2"])

        costs = sched_cost[i]
        daily_cost = float(costs[daily_idx])
        knee_si = chord_knee(costs, sched_waits)
        knee_cost = float(costs[knee_si])
        knee_wait = float(sched_waits[knee_si])
        knee_size = int(sched_sizes[knee_si])
        knee_sav_pct = 100.0 * (daily_cost - knee_cost) / daily_cost
        max_sav_pct = 100.0 * (daily_cost - costs[sched_sizes < N_DAYS].min()
                                ) / daily_cost

        rows.append({
            "provider": prov, "plz": plz,
            "weekly_parcels": weekly_parcels,
            "n_stops_per_day": n_stops_per_day,
            "parcels_per_stop": parcels_per_stop,
            "hub_dist_km": hub_dist,
            "area_km2": area,
            "daily_cost_eur": daily_cost,
            "knee_size": knee_size,
            "knee_wait_d": knee_wait,
            "knee_saving_pct": knee_sav_pct,
            "max_saving_pct": max_sav_pct,
        })

    df = pd.DataFrame(rows)
    # Join raumtyp_3 (urban/suburban/rural) per PLZ
    rt = pd.read_csv(ROOT / "data/geodata/plz_raumtyp.csv",
                      dtype={"plz": str})[["plz", "raumtyp_3"]]
    df = df.merge(rt, on="plz", how="left")
    n_missing = df.raumtyp_3.isna().sum()
    if n_missing > 0:
        print(f"WARN: {n_missing} cells without raumtyp_3 - dropped from plot")
    df.to_csv(OUT / "tab_PLZ_knee_with_features.csv", index=False)
    print(f"Built knee table for {len(df)} (provider, PLZ) cells")

    # Print raumtyp-level summary
    print("\nRaumtyp median knee saving %:")
    print(df.groupby("raumtyp_3")
            .agg(median_knee_sav=("knee_saving_pct", "median"),
                 median_max_sav=("max_saving_pct", "median"),
                 median_hub_dist=("hub_dist_km", "median"),
                 median_pps=("parcels_per_stop", "median"),
                 n=("plz", "nunique"))
            .round(2).to_string())

    # Spearman correlations
    print("\nSpearman rho (knee saving % vs feature):")
    for feat in ("hub_dist_km", "area_km2", "parcels_per_stop",
                  "weekly_parcels"):
        rho, p = spearmanr(df[feat], df["knee_saving_pct"])
        print(f"  {feat:<20}  rho={rho:+.3f}  p={p:.2e}")

    # ---- 3-panel figure, raumtyp-coloured ----
    df_plot = df.dropna(subset=["raumtyp_3"]).copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    feats = [
        ("hub_dist_km", "Hub distance [km]",
         "(a) Longer stem -> more potential saving"),
        ("area_km2", r"Postal-code area size [km$^2$]",
         "(b) Larger area -> more dispersion gain"),
        ("parcels_per_stop", "Parcels per drop-site [pkts/site/wk]",
         "(c) Denser drops -> structurally capped"),
    ]
    for ax, (feat, xlab, title) in zip(axes, feats):
        for rt in RAUMTYP_ORDER:
            sub = df_plot[df_plot.raumtyp_3 == rt]
            if len(sub) == 0:
                continue
            ax.scatter(sub[feat], sub.knee_saving_pct,
                        s=24, color=RAUMTYP_PALETTE[rt], alpha=0.75,
                        edgecolor="white", linewidth=0.4,
                        label=rt.capitalize(), rasterized=True)
        x = df_plot[feat].values
        y = df_plot.knee_saving_pct.values
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        bins = np.array_split(np.arange(len(xs)), 8)
        bx = [xs[b].mean() for b in bins if len(b) > 0]
        by = [np.median(ys[b]) for b in bins if len(b) > 0]
        ax.plot(bx, by, "-", color="black", linewidth=1.6, alpha=0.85,
                 label="Binned median")
        rho, _ = spearmanr(x, y)
        ax.text(0.97, 0.05, rf"Spearman $\rho = {rho:+.2f}$",
                 transform=ax.transAxes, ha="right", va="bottom",
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
                                         fc="white", ec="0.6"))
        ax.set_xlabel(xlab)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Per-postal-code knee saving vs daily baseline [%]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
                bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=10,
                title="Area type", title_fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=1.0)
    fig.savefig(OUT / "fig_PLZ_structural_correlation.png",
                 bbox_inches="tight")
    fig.savefig(OUT / "fig_PLZ_structural_correlation.pdf",
                 bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {OUT/'fig_PLZ_structural_correlation.png'}")


if __name__ == "__main__":
    main()
