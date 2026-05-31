"""Value of the full Path-2 pipeline vs naive per-PLZ heuristics.

Compares total weekly system cost across plausible scheduling policies
at theta=1 (full willingness), over the service-penalty sweep P. The aim
is to quantify how much the *coordinated* Path-2 pipeline (CD on hub-
bundled cost + frequency-preserving balancing + system smoothing) adds
on top of a naive per-PLZ choice.

Policies plotted:
  - Daily baseline           = every PLZ daily, no batching
  - Fixed {Mo,Wed,Fri}       = every PLZ on this fixed 3-day pattern
  - Naive per-PLZ argmin     = per-PLZ argmin of (cost + P * pkts * wait),
                                no hub coordination
  - Path-2 pipeline          = CD on bundled cost + balancing + smoothing
                                (from tab_balancing_summary.csv)

All four read raw cost from the cached Daganzo-Hybrid matrix to keep the
cost model identical, except Path-2 which we lift directly from the
pipeline output (so it carries the bundling correction and any
balancing-induced cost delta).

Outputs (results/EWGT_Results/, flat):
  fig_value_of_path2_pipeline.{png,pdf}
  tab_value_of_path2_policies.csv
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
PATH2 = ROOT / "results" / "overnight_2026_05_29_path2"
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
PENALTIES = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0])


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
    cache = np.load(ROOT / "results/penalty_sweep/sched_cost_cache.npz")
    sched_cost = cache["sched_cost"]
    prov_order = [str(x) for x in cache["prov_order"]]
    plz_order = [str(x) for x in cache["plz_order"]]
    n_pp = sched_cost.shape[0]

    schedules = enumerate_schedules()
    sched_sizes = np.array([len(s) for s in schedules])
    sched_waits = np.array([avg_wait(s) for s in schedules])
    daily_idx = int(np.where(sched_sizes == N_DAYS)[0][0])
    daily_cost_per_pp = sched_cost[:, daily_idx]
    daily_total = float(daily_cost_per_pp.sum())

    # Find {Mon, Wed, Fri} schedule index. Mon=0 .. Sat=5
    target = frozenset({0, 2, 4})
    mwf_idx = next(i for i, s in enumerate(schedules) if s == target)
    fixed_mwf_cost_per_pp = sched_cost[:, mwf_idx]
    fixed_mwf_total = float(fixed_mwf_cost_per_pp.sum())

    # Per-PLZ weekly parcels for penalty calculation
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl",
                            "rb"))
    pld_all = {prov: pd_["plz_demand"]
                for prov, pd_ in chk["provider_data"].items()}
    weekly = []
    for prov, plz in zip(prov_order, plz_order):
        r = pld_all[prov][pld_all[prov].plz == plz]
        weekly.append(int(r.weekly_parcels.iloc[0]) if not r.empty else 0)
    weekly = np.array(weekly, dtype=np.float64)

    # Path-2 actual: lift from tab_balancing_summary
    bal = pd.read_csv(PATH2 / "tab_balancing_summary.csv")
    bal_th1 = bal[np.isclose(bal.share_willing, 1.0)].copy()
    path2 = (bal_th1.groupby("penalty")
                  .agg(path2_init_eur=("init_cost_eur", "sum"),
                       path2_balanced_eur=("balanced_cost_eur", "sum"))
                  .reset_index()
                  .sort_values("penalty"))

    rows = []
    for P in PENALTIES:
        # naive per-PLZ argmin on (cost + P * pkts * wait)
        obj = sched_cost + P * weekly[:, None] * sched_waits[None, :]
        naive_si = np.argmin(obj, axis=1)
        naive_cost_per_pp = sched_cost[np.arange(n_pp), naive_si]
        naive_wait_per_pp = sched_waits[naive_si]
        naive_total = float(naive_cost_per_pp.sum())
        naive_wait_w = float(
            (naive_wait_per_pp * weekly).sum() / weekly.sum())

        # fixed Mon-Wed-Fri wait
        mwf_wait = float(sched_waits[mwf_idx])

        row = dict(
            penalty=P,
            daily_cost_eur=daily_total,
            daily_wait_d=0.0,
            fixed_mwf_cost_eur=fixed_mwf_total,
            fixed_mwf_wait_d=mwf_wait,
            naive_cost_eur=naive_total,
            naive_wait_d=naive_wait_w,
        )
        rows.append(row)
    pol = pd.DataFrame(rows)

    # Merge Path-2 actual where available
    pol = pol.merge(path2, on="penalty", how="left")

    pol["daily_sav_pct"] = 0.0
    pol["fixed_mwf_sav_pct"] = (100.0 *
        (daily_total - pol.fixed_mwf_cost_eur) / daily_total)
    pol["naive_sav_pct"] = (100.0 *
        (daily_total - pol.naive_cost_eur) / daily_total)
    pol["path2_sav_pct"] = (100.0 *
        (daily_total - pol.path2_balanced_eur) / daily_total)

    pol.to_csv(OUT / "tab_value_of_path2_policies.csv", index=False)
    print("Policy totals at theta=1 (k€/wk):")
    print((pol[["penalty", "daily_cost_eur", "fixed_mwf_cost_eur",
                 "naive_cost_eur", "path2_balanced_eur",
                 "daily_sav_pct", "fixed_mwf_sav_pct",
                 "naive_sav_pct", "path2_sav_pct"]] / 1.0).round(2)
              .to_string(index=False))

    # ---- Plot ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.0),
                                     sharex=True)
    col_daily = "#999999"
    col_mwf = "#4D4D4D"
    col_naive = "#E76F51"
    col_path2 = "#1D3557"

    # Left: absolute cost
    axL.axhline(daily_total / 1000.0, color=col_daily, linewidth=1.6,
                 linestyle=":", label="Daily baseline")
    axL.axhline(fixed_mwf_total / 1000.0, color=col_mwf, linewidth=1.6,
                 linestyle="--",
                 label=r"Fixed {Mo, We, Fr}")
    axL.plot(pol.penalty, pol.naive_cost_eur / 1000.0, "s-",
              color=col_naive, linewidth=1.8, markersize=6,
              label="Naive per-PLZ argmin")
    p2 = pol[~pol.path2_balanced_eur.isna()]
    axL.plot(p2.penalty, p2.path2_balanced_eur / 1000.0, "o-",
              color=col_path2, linewidth=2.0, markersize=7,
              label="Path-2 pipeline (real)")
    axL.set_xlabel(r"Service penalty $P$ [€/p/d]")
    axL.set_ylabel("Total weekly system cost [k€]")
    axL.set_title("(a) Absolute system cost at $\\theta = 1$")
    axL.grid(alpha=0.3)
    axL.legend(loc="lower right", framealpha=0.95)

    # Right: saving % vs daily
    axR.axhline(0, color=col_daily, linewidth=1.2, linestyle=":")
    axR.axhline(pol.fixed_mwf_sav_pct.iloc[0], color=col_mwf,
                 linewidth=1.6, linestyle="--",
                 label=r"Fixed {Mo, We, Fr}")
    axR.plot(pol.penalty, pol.naive_sav_pct, "s-",
              color=col_naive, linewidth=1.8, markersize=6,
              label="Naive per-PLZ argmin")
    axR.plot(p2.penalty, p2.path2_sav_pct, "o-",
              color=col_path2, linewidth=2.0, markersize=7,
              label="Path-2 pipeline (real)")
    # Sweet-spot annotation
    if len(p2[np.isclose(p2.penalty, 0.5)]) > 0:
        row = p2[np.isclose(p2.penalty, 0.5)].iloc[0]
        axR.scatter([0.5], [row.path2_sav_pct], marker="*", s=320,
                     color="gold", edgecolor="black", zorder=10,
                     label=r"Sweet-spot $P^\ast = 0.5$")
    axR.set_xlabel(r"Service penalty $P$ [€/p/d]")
    axR.set_ylabel("Saving vs daily baseline [%]")
    axR.set_title("(b) Saving relative to daily baseline")
    axR.grid(alpha=0.3)
    axR.legend(loc="upper right", framealpha=0.95)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "fig_value_of_path2_pipeline.png",
                 bbox_inches="tight")
    fig.savefig(OUT / "fig_value_of_path2_pipeline.pdf",
                 bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {OUT/'fig_value_of_path2_pipeline.png'}")


if __name__ == "__main__":
    main()
