"""Value-of-optimization plot for the EWGT paper: extra weekly cost when
picking the WORST schedule choice instead of the BEST, as a function of
batch share, with P = 10 added.

Same calculation logic as paper_plot_best_worst_heatmaps.py, but
  - includes P = 10 to show the high-penalty extreme
  - outputs flat into results/EWGT_Results/
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
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "EWGT_Results"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

N_DAYS = 6
MAX_HOLD = 3
BATCH_SHARES = np.linspace(0.0, 1.0, 11)
PENALTIES = np.array([0.0, 0.25, 0.50, 0.75, 1.00, 10.0])


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
                    ok = False; break
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
    cache = np.load(ROOT / "results/penalty_sweep/sched_cost_cache.npz")
    sched_cost = cache["sched_cost"]
    prov_order = list(cache["prov_order"])
    plz_order = list(cache["plz_order"])
    n_pp = sched_cost.shape[0]

    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    pld_all = {prov: pd_["plz_demand"]
               for prov, pd_ in chk["provider_data"].items()}
    weekly = []
    for prov, plz in zip(prov_order, plz_order):
        r = pld_all[prov][pld_all[prov].plz == plz]
        weekly.append(int(r.weekly_parcels.iloc[0]) if not r.empty else 0)
    weekly = np.array(weekly, dtype=np.float64)

    schedules = enumerate_schedules()
    sched_sizes = np.array([len(s) for s in schedules])
    sched_waits = np.array([avg_wait(s) for s in schedules])
    daily_idx = int(np.where(sched_sizes == N_DAYS)[0][0])
    daily_cost = sched_cost[:, daily_idx]
    print(f"{n_pp} PLZ cells, {len(schedules)} schedules, "
          f"daily baseline {daily_cost.sum()/1e3:,.0f} k€")

    # WORST is non-daily pure-cost-maximum (penalty-independent)
    nondaily_mask = sched_sizes < N_DAYS
    cost_nondaily = np.where(nondaily_mask[None, :], sched_cost, -np.inf)
    worst_si = np.argmax(cost_nondaily, axis=1)
    cost_worst = sched_cost[np.arange(n_pp), worst_si]

    # Plot: 2-panel - raw cost gap (a) and full-objective gap (b)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5),
                                    sharex=True)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(PENALTIES)))
    # Penalty-objective weight per non-daily worst schedule -- the WORST
    # schedule's penalty cost depends on its wait. Build wait of worst once
    # per P (P-independent here because the worst is the cost-max).
    wait_worst = sched_waits[worst_si]

    for jP, P in enumerate(PENALTIES):
        obj = sched_cost + P * weekly[:, None] * sched_waits[None, :]
        best_si = np.argmin(obj, axis=1)
        cost_best = sched_cost[np.arange(n_pp), best_si]
        wait_best = sched_waits[best_si]
        # Rank cells by best raw-cost saving (panel a's ordering kept
        # consistent across both panels).
        sav_best = daily_cost - cost_best
        rank = np.argsort(-sav_best)

        # Penalty cost per cell at the BEST and the WORST choice
        pen_best = P * weekly * wait_best
        pen_worst = P * weekly * wait_worst

        full_best = cost_best + pen_best
        full_worst = cost_worst + pen_worst

        gaps_raw = []
        gaps_full = []
        for bs in BATCH_SHARES:
            n_batched = int(round(bs * n_pp))
            idx = rank[:n_batched]
            gaps_raw.append(
                (cost_worst[idx].sum() - cost_best[idx].sum()) / 1e3)
            gaps_full.append(
                (full_worst[idx].sum() - full_best[idx].sum()) / 1e3)
        ax1.plot(BATCH_SHARES * 100, gaps_raw, "o-",
                  color=cmap[jP], markersize=6, linewidth=1.8,
                  label=rf"$P = {P:g}$")
        ax2.plot(BATCH_SHARES * 100, gaps_full, "o-",
                  color=cmap[jP], markersize=6, linewidth=1.8,
                  label=rf"$P = {P:g}$")

    for ax in (ax1, ax2):
        ax.set_xlabel("Batch share [% of (provider, PLZ) cells batched]")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)
    ax1.set_ylabel("Extra weekly raw cost [k€]")
    ax1.set_title("(a) Raw routing cost: WORST minus BEST")
    ax2.set_ylabel(r"Extra weekly objective"
                     r" (cost + $P \cdot \mathrm{pkts} \cdot \overline{w}$)"
                     " [k€]")
    ax2.set_title("(b) Full objective: WORST minus BEST")
    ax1.legend(title="Service penalty", loc="upper left", ncol=2,
                fontsize=9)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "fig_value_of_optimization.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_value_of_optimization.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'fig_value_of_optimization.png'}")


if __name__ == "__main__":
    main()
