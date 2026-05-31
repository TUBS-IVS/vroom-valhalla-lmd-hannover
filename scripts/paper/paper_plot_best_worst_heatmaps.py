"""BEST vs WORST plan-combination heatmaps + value-of-optimization analyses.

Builds two heatmaps with axes (batch_share × WAITING_PENALTY):

  BEST  → for each batched PLZ pick the cost-optimal schedule under
          the penalty (this is what the optimizer does)
  WORST → for each batched PLZ pick the worst schedule under the
          penalty (anti-optimization)

Same PLZs are batched in both cases (selected by best-saving rank),
so the difference isolates the value of correct SCHEDULE CHOICE,
holding the batching decision fixed.

Additional outputs:
  * fig13_best_minus_worst.{png,pdf}     — heatmap of (worst − best) gap
  * tab_best_worst_grid.csv              — long table with all metrics
  * fig14_value_of_optimization.{png,pdf} — gap as function of batch_share
                                            (panels per penalty)

Inputs:
  results/penalty_sweep/sched_cost_cache.npz   (312 × 39 Daganzo-Hybrid)
  results/checkpoints/01_demand.pkl
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "overnight_2026_05_27"

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

N_DAYS = 6
MAX_HOLD = 3
BATCH_SHARES = np.linspace(0.0, 1.0, 11)
PENALTIES = np.array([0.0, 0.25, 0.50, 0.75, 1.00])


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


def main():
    print("Loading cached sched_cost ...")
    cache = np.load(ROOT / "results/penalty_sweep/sched_cost_cache.npz")
    sched_cost = cache["sched_cost"]
    prov_order = list(cache["prov_order"])
    plz_order = list(cache["plz_order"])
    n_pp = sched_cost.shape[0]

    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    pld_all = {}
    for prov, pd_ in chk["provider_data"].items():
        pld_all[prov] = pd_["plz_demand"]

    weekly = []
    for prov, plz in zip(prov_order, plz_order):
        row = pld_all[prov]
        r = row[row.plz == plz]
        weekly.append(int(r.weekly_parcels.iloc[0]) if not r.empty else 0)
    weekly = np.array(weekly, dtype=np.float64)

    schedules = enumerate_schedules()
    sched_sizes = np.array([len(s) for s in schedules])
    sched_waits = np.array([avg_wait(s) for s in schedules])
    daily_idx = int(np.where(sched_sizes == N_DAYS)[0][0])
    daily_cost = sched_cost[:, daily_idx]

    print(f"  {n_pp} PLZ cells, {len(schedules)} schedules, "
          f"daily-only baseline {daily_cost.sum()/1e3:,.0f} k€")

    # Per (penalty): identify BEST and WORST schedule per PLZ
    rows = []
    best_grid = np.zeros((len(BATCH_SHARES), len(PENALTIES)))
    worst_grid = np.zeros_like(best_grid)
    sav_best_per_plz = {}    # penalty → array of savings per PLZ
    sav_worst_per_plz = {}

    # WORST is penalty-independent: argmax of PURE cost among non-daily
    # schedules (size < 6). Represents the most expensive possible batched
    # tour — what a naive picker would pick if they avoided daily delivery
    # but ignored cost. Pure cost of WORST is always >= BEST.
    nondaily_mask = sched_sizes < N_DAYS
    cost_nondaily = np.where(nondaily_mask[None, :], sched_cost, -np.inf)
    worst_si_constant = np.argmax(cost_nondaily, axis=1)
    cost_worst_const = sched_cost[np.arange(n_pp), worst_si_constant]

    for jP, P in enumerate(PENALTIES):
        # BEST: argmin of operational objective (cost + service-penalty term)
        obj = sched_cost + P * weekly[:, None] * sched_waits[None, :]
        best_si = np.argmin(obj, axis=1)
        cost_best = sched_cost[np.arange(n_pp), best_si]
        # WORST: pure-cost-pessimal non-daily schedule, penalty-independent.
        cost_worst = cost_worst_const.copy()

        # Saving per PLZ when going BATCHED instead of daily
        # (positive = saving; negative = batching this cell raises cost)
        sav_best = daily_cost - cost_best
        sav_worst = daily_cost - cost_worst
        sav_best_per_plz[P] = sav_best
        sav_worst_per_plz[P] = sav_worst

        # Rank PLZs by their BEST-case saving (used as batching priority)
        rank = np.argsort(-sav_best)   # highest saving first

        for iB, bs in enumerate(BATCH_SHARES):
            n_batched = int(round(bs * n_pp))
            batched_mask = np.zeros(n_pp, dtype=bool)
            batched_mask[rank[:n_batched]] = True

            # BEST scenario
            cost_total_best = (
                cost_best[batched_mask].sum()
                + daily_cost[~batched_mask].sum()
            )
            # WORST scenario (same PLZs batched, but each picks worst schedule)
            cost_total_worst = (
                cost_worst[batched_mask].sum()
                + daily_cost[~batched_mask].sum()
            )
            best_grid[iB, jP] = cost_total_best / 1e3
            worst_grid[iB, jP] = cost_total_worst / 1e3
            rows.append({
                "batch_share": float(bs),
                "penalty": float(P),
                "best_cost_keur": cost_total_best / 1e3,
                "worst_cost_keur": cost_total_worst / 1e3,
                "gap_keur": (cost_total_worst - cost_total_best) / 1e3,
                "best_saving_pct": 100.0 * (1 - cost_total_best / daily_cost.sum()),
                "worst_saving_pct": 100.0 * (1 - cost_total_worst / daily_cost.sum()),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tab_best_worst_grid.csv", index=False)

    # Plot 11: BEST heatmap
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(best_grid, aspect="auto", cmap="viridis_r",
                    vmin=best_grid.min(), vmax=worst_grid.max())
    ax.set_xticks(range(len(PENALTIES)))
    ax.set_xticklabels([f"{p:g}" for p in PENALTIES])
    ax.set_yticks(range(len(BATCH_SHARES)))
    ax.set_yticklabels([f"{b:.1f}" for b in BATCH_SHARES])
    ax.set_xlabel("WAITING_PENALTY [€/parcel/day]")
    ax.set_ylabel("batch_share (fraction of PLZ cells batched)")
    ax.set_title("BEST plan combination — total weekly cost [k€]\n"
                  "(optimizer picks cost-optimal schedule per batched PLZ)")
    for i in range(len(BATCH_SHARES)):
        for j in range(len(PENALTIES)):
            v = best_grid[i, j]
            color = "white" if (v - best_grid.min()) \
                / max(1, worst_grid.max() - best_grid.min()) < 0.55 else "black"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color=color, fontsize=9)
    plt.colorbar(im, ax=ax, label="Weekly cost [k€]")
    fig.tight_layout()
    fig.savefig(OUT / "fig11_heatmap_best.png")
    fig.savefig(OUT / "fig11_heatmap_best.pdf")
    plt.close(fig)

    # Plot 12: WORST heatmap
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(worst_grid, aspect="auto", cmap="viridis_r",
                    vmin=best_grid.min(), vmax=worst_grid.max())
    ax.set_xticks(range(len(PENALTIES)))
    ax.set_xticklabels([f"{p:g}" for p in PENALTIES])
    ax.set_yticks(range(len(BATCH_SHARES)))
    ax.set_yticklabels([f"{b:.1f}" for b in BATCH_SHARES])
    ax.set_xlabel("WAITING_PENALTY [€/parcel/day]")
    ax.set_ylabel("batch_share (fraction of PLZ cells batched)")
    ax.set_title("WORST plan combination — total weekly cost [k€]\n"
                  "(naive picker selects cost-maximal schedule per batched PLZ)")
    for i in range(len(BATCH_SHARES)):
        for j in range(len(PENALTIES)):
            v = worst_grid[i, j]
            color = "white" if (v - best_grid.min()) \
                / max(1, worst_grid.max() - best_grid.min()) < 0.55 else "black"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color=color, fontsize=9)
    plt.colorbar(im, ax=ax, label="Weekly cost [k€]")
    fig.tight_layout()
    fig.savefig(OUT / "fig12_heatmap_worst.png")
    fig.savefig(OUT / "fig12_heatmap_worst.pdf")
    plt.close(fig)

    # Plot 13: gap heatmap (WORST - BEST) — value of optimization
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    gap = worst_grid - best_grid
    im = ax.imshow(gap, aspect="auto", cmap="Reds")
    ax.set_xticks(range(len(PENALTIES)))
    ax.set_xticklabels([f"{p:g}" for p in PENALTIES])
    ax.set_yticks(range(len(BATCH_SHARES)))
    ax.set_yticklabels([f"{b:.1f}" for b in BATCH_SHARES])
    ax.set_xlabel("WAITING_PENALTY [€/parcel/day]")
    ax.set_ylabel("batch_share")
    ax.set_title("Value of optimization — extra weekly cost [k€]\n"
                  "if you batch the SAME PLZs but pick the wrong schedule")
    for i in range(len(BATCH_SHARES)):
        for j in range(len(PENALTIES)):
            v = gap[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v > gap.max() * 0.55 else "black",
                    fontsize=9)
    plt.colorbar(im, ax=ax, label="Cost gap WORST − BEST [k€]")
    fig.tight_layout()
    fig.savefig(OUT / "fig13_best_minus_worst.png")
    fig.savefig(OUT / "fig13_best_minus_worst.pdf")
    plt.close(fig)

    # Plot 14: value of optimization — per batch_share, lines per penalty
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(PENALTIES)))
    for j, P in enumerate(PENALTIES):
        ax.plot(BATCH_SHARES * 100, gap[:, j], "o-",
                color=colors[j], linewidth=2, markersize=5,
                label=f"$P={P:g}$")
    ax.set_xlabel("Batch share [% of (provider, PLZ) cells batched]")
    ax.set_ylabel("Extra weekly cost when picking WORST schedule [k€]")
    ax.set_title("How much does optimization save vs naive schedule choice?")
    ax.legend(title="Service penalty", loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig14_value_of_optimization.png")
    fig.savefig(OUT / "fig14_value_of_optimization.pdf")
    plt.close(fig)

    # Headline summary
    print()
    print("Saving range across the grid:")
    print(f"  BEST  cost min: {best_grid.min():.0f} k€,  max: {best_grid.max():.0f} k€")
    print(f"  WORST cost min: {worst_grid.min():.0f} k€,  max: {worst_grid.max():.0f} k€")
    print(f"  Max gap (WORST-BEST):  {gap.max():.0f} k€  "
          f"at batch_share={BATCH_SHARES[np.unravel_index(gap.argmax(), gap.shape)[0]]:.1f}, "
          f"P={PENALTIES[np.unravel_index(gap.argmax(), gap.shape)[1]]:g}")
    print(f"  Min gap (WORST-BEST):  {gap.min():.0f} k€")
    print()
    for p in OUT.glob("fig1[1-4]*"):
        print(f"  saved {p.name}")
    print(f"  tab_best_worst_grid.csv saved")


if __name__ == "__main__":
    main()
