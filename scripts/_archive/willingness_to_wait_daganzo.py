"""Willingness-to-Wait 2D Sensitivity — Daganzo-LGB-Hybrid @ P=0.5 €/parcel/day.

Two customer-side knobs:
  * share_willing  ∈ [0, 100%]  — fraction of demand that accepts being batched
  * postponement_window ∈ {1, 2, 3}  — max gap days the batched portion accepts

For each (window, share):
  * Allowed schedules = all valid weekly patterns with max gap ≤ window
  * Per (provider, PLZ) cost is the cost-weighted blend
        cost = (1 - share) * cost_6day_daily
             + share * min(cost_allowed + P * pkts * wait_days)
  * Wait days only accrue on the batched portion (express portion = 0 wait).

Reuses the cached `results/penalty_sweep/sched_cost_cache.npz` matrix
(n_pp × 39 schedules at MAX_HOLD=3, predicted by Daganzo-Hybrid v2-aug).

Outputs (results/willingness_p050/):
    figW1_cost_vs_share.{png,pdf}
    figW2_wait_vs_share.{png,pdf}
    figW3_schedule_size_mix.{png,pdf}                ← analog to v2preview headline
    figW4_pareto_cost_wait.{png,pdf}
    figW5_provider_cost_vs_share.{png,pdf}
    tab_grid.csv
    tab_chosen_at_p050.csv
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

from batch_delivery.features import ALL_COLS, _PROVIDER_IDX  # noqa: E402

rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

OUT = ROOT / "results" / "willingness_p050"
OUT.mkdir(parents=True, exist_ok=True)

N_DAYS = 6
MAX_HOLD_CACHED = 3
PENALTY = 0.5  # operating point
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

SHARE_GRID = np.linspace(0.0, 1.0, 11)
WINDOW_GRID = [1, 2, 3]

WIN_COLOR  = {1: "#003049", 2: "#2a9d8f", 3: "#e76f51"}
SIZE_COLOR = {1: "#3b1f4b", 2: "#1d3557", 3: "#2a9d8f",
              4: "#e9c46a", 5: "#f4a261", 6: "#e76f51"}


def log(msg):
    print(msg, flush=True)


def enumerate_schedules(max_hold):
    out = []
    for k in range(1, N_DAYS + 1):
        for combo in combinations(range(N_DAYS), k):
            days = sorted(combo)
            ok = True
            for i in range(len(days)):
                gap = (days[(i + 1) % len(days)] - days[i]) % N_DAYS
                if gap == 0:
                    gap = N_DAYS
                if gap > max_hold:
                    ok = False
                    break
            if ok:
                out.append(frozenset(days))
    return out


def avg_wait_days(schedule_days):
    if not schedule_days:
        return 0.0
    ds = sorted(schedule_days)
    total = 0.0
    for di in range(N_DAYS):
        next_dd = min(((d - di) % N_DAYS, d) for d in ds)[1]
        wait = (next_dd - di) % N_DAYS
        total += wait
    return total / N_DAYS


def main():
    t0 = time.time()
    log("=" * 72)
    log("Willingness-to-Wait 2D @ P=0.50 €/parcel/day  (Daganzo-LGB-Hybrid)")
    log("=" * 72)

    # 1) cached sched_cost (39 schedules @ MAX_HOLD=3)
    log("[1] Loading cached sched_cost ...")
    cache = np.load(ROOT / "results/penalty_sweep/sched_cost_cache.npz")
    sched_cost = cache["sched_cost"]
    prov_cache = list(cache["prov_order"])
    plz_cache  = list(cache["plz_order"])
    log(f"    cache shape: {sched_cost.shape}")

    # 2) reconstruct pp metadata
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    provider_data = chk["provider_data"]
    optim_data = chk4["optimization_data"]

    n_pp_arr = []
    pp_meta = []
    for prov, plz in zip(prov_cache, plz_cache):
        pld = provider_data[prov]["plz_demand"]
        row = pld[pld.plz == plz]
        if row.empty:
            continue
        row = row.iloc[0]
        n_pp_arr.append(int(row.weekly_parcels))
        pp_meta.append({"provider": prov, "plz": plz,
                          "weekly_parcels": int(row.weekly_parcels)})
    n_pp_arr = np.array(n_pp_arr, dtype=np.float64)
    n_pp = len(pp_meta)
    log(f"    pp_meta rows: {n_pp}")

    # 3) schedules @ MAX_HOLD=3 + masks for windows {1, 2, 3}
    schedules = enumerate_schedules(MAX_HOLD_CACHED)
    sched_sizes = np.array([len(s) for s in schedules])
    sched_waits = np.array([avg_wait_days(sorted(s)) for s in schedules])

    # For each window: which schedule-indices are valid?
    # Build a mask per window using the enumerate function
    window_mask = {}
    for w in WINDOW_GRID:
        valid = set(enumerate_schedules(w))
        mask = np.array([s in valid for s in schedules])
        window_mask[w] = mask
        log(f"    window={w}: {mask.sum()} / {len(schedules)} schedules valid")

    # daily 6-day reference cost: only the size-6 schedule
    daily_idx = int(np.where(sched_sizes == N_DAYS)[0][0])
    cost_daily = sched_cost[:, daily_idx]
    log(f"    daily-only ref cost @ all-PLZ: {cost_daily.sum()/1e3:.1f} k€")

    # 4) sweep (window, share)
    rows = []
    chosen_records_p050 = []
    for w in WINDOW_GRID:
        wm = window_mask[w]
        # For each pp: best batched schedule under penalty (under window constraint)
        combined = sched_cost + PENALTY * n_pp_arr[:, None] * sched_waits[None, :]
        combined_window = np.where(wm[None, :], combined, np.inf)
        best_si_window = np.argmin(combined_window, axis=1)
        cost_batched = sched_cost[np.arange(n_pp), best_si_window]
        wait_batched = sched_waits[best_si_window]
        size_batched = sched_sizes[best_si_window]

        # record chosen schedule for this window
        for ppi, pp in enumerate(pp_meta):
            chosen_records_p050.append({
                "window": w, "share_willing": 1.0,  # snapshot at 100% will = batched fully
                **pp,
                "schedule_size_batched": int(size_batched[ppi]),
                "schedule_idx": int(best_si_window[ppi]),
                "schedule_weekdays": ",".join(WEEKDAYS[d] for d in sorted(schedules[best_si_window[ppi]])),
                "cost_eur_batched": float(cost_batched[ppi]),
                "avg_wait_d_batched": float(wait_batched[ppi]),
            })

        for share in SHARE_GRID:
            blended_cost = (1 - share) * cost_daily + share * cost_batched
            blended_wait = share * wait_batched  # express portion = 0 wait
            tot_cost = float(blended_cost.sum())
            tot_pkts = float(n_pp_arr.sum())
            avg_wait_w = float((blended_wait * n_pp_arr).sum() / tot_pkts)

            # Schedule-size mix: at fractional share, count batched-PLZ at chosen
            # schedule size, ELSE daily. Conceptually: at share s, fraction s of
            # PLZ adopt the batched schedule, fraction (1-s) stay daily. Use the
            # weighted count for the size mix plot.
            size_counts = defaultdict(float)
            for ppi in range(n_pp):
                size_counts[int(size_batched[ppi])] += share
                size_counts[N_DAYS] += (1 - share)

            rows.append({
                "window": w,
                "share_willing": float(share),
                "total_cost_eur": tot_cost,
                "cost_savings_pct": 100.0 * (cost_daily.sum() - tot_cost) / cost_daily.sum(),
                "avg_wait_days_weighted": avg_wait_w,
                "n_pkts_weekly": int(tot_pkts),
                **{f"size_{k}_count": size_counts.get(k, 0) for k in range(1, 7)},
            })
            log(f"  window={w}  share={share*100:5.1f}%  "
                f"cost={tot_cost/1e3:6.1f}k€  wait={avg_wait_w:.3f}d  "
                f"daily-share={size_counts[N_DAYS]/n_pp:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tab_grid.csv", index=False)
    pd.DataFrame(chosen_records_p050).to_csv(OUT / "tab_chosen_at_p050.csv", index=False)

    # 5) Figures
    log("[2] Building figures ...")

    # figW1: cost vs share, per window
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for w in WINDOW_GRID:
        sub = df[df.window == w].sort_values("share_willing")
        ax.plot(sub.share_willing * 100, sub.total_cost_eur / 1e3,
                "o-", color=WIN_COLOR[w], linewidth=2, markersize=5,
                label=f"Postponement window = {w} day{'s' if w != 1 else ''}")
    ax.set_xlabel("Share of customers willing to wait [%]")
    ax.set_ylabel("Weekly routing cost [k€]")
    ax.set_title("Cost reduction as customer willingness rises\n"
                  f"(operating point $P={PENALTY}$ €/parcel/day, Daganzo-LGB-Hybrid)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figW1_cost_vs_share.png")
    fig.savefig(OUT / "figW1_cost_vs_share.pdf")
    plt.close(fig)

    # figW2: avg wait vs share
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for w in WINDOW_GRID:
        sub = df[df.window == w].sort_values("share_willing")
        ax.plot(sub.share_willing * 100, sub.avg_wait_days_weighted,
                "o-", color=WIN_COLOR[w], linewidth=2, markersize=5,
                label=f"Window = {w} day{'s' if w != 1 else ''}")
    ax.set_xlabel("Share of customers willing to wait [%]")
    ax.set_ylabel("Average customer wait, weighted by parcels [days]")
    ax.set_title("Wait-time burden grows with the willing-share and window")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figW2_wait_vs_share.png")
    fig.savefig(OUT / "figW2_wait_vs_share.pdf")
    plt.close(fig)

    # figW3: schedule-size mix per window (stacked area)  — analog v2preview headline
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ai, w in enumerate(WINDOW_GRID):
        sub = df[df.window == w].sort_values("share_willing")
        share_pct = sub.share_willing.values * 100
        bottom = np.zeros(len(sub))
        totals = sum(sub[f"size_{k}_count"].values for k in range(1, 7))
        totals = np.where(totals == 0, 1, totals)
        for sz in [2, 3, 4, 5, 6]:
            count = sub[f"size_{sz}_count"].values
            pct = 100.0 * count / totals
            axes[ai].fill_between(share_pct, bottom, bottom + pct,
                                    color=SIZE_COLOR[sz], alpha=0.88,
                                    label=f"{sz} day/wk")
            bottom = bottom + pct
        axes[ai].set_title(f"Postponement window = {w} day{'s' if w != 1 else ''}")
        axes[ai].set_xlabel("Share of customers willing to wait [%]")
        axes[ai].set_ylim(0, 100)
        axes[ai].grid(alpha=0.3)
    axes[0].set_ylabel("Delivery-frequency mix [%]")
    axes[-1].legend(title="Delivery days/week",
                    loc="upper right", frameon=True, fontsize=8)
    fig.suptitle(
        f"How willingness-to-wait reshapes the delivery-frequency mix "
        f"(operating point $P={PENALTY}$ €/parcel/day)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT / "figW3_schedule_size_mix.png")
    fig.savefig(OUT / "figW3_schedule_size_mix.pdf")
    plt.close(fig)

    # figW4: Pareto wait × cost
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for w in WINDOW_GRID:
        sub = df[df.window == w].sort_values("share_willing")
        ax.plot(sub.avg_wait_days_weighted, sub.total_cost_eur / 1e3,
                "o-", color=WIN_COLOR[w], linewidth=2, markersize=5,
                label=f"Window = {w} day{'s' if w != 1 else ''}")
        for _, r in sub.iterrows():
            if r.share_willing in (0.0, 0.5, 1.0):
                ax.annotate(f"{int(r.share_willing*100)}%",
                              xy=(r.avg_wait_days_weighted, r.total_cost_eur / 1e3),
                              xytext=(6, 4), textcoords="offset points",
                              fontsize=8, color=WIN_COLOR[w])
    ax.set_xlabel("Average customer wait, weighted by parcels [days]")
    ax.set_ylabel("Weekly routing cost [k€]")
    ax.set_title("Pareto trade-off across willingness and postponement-window")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figW4_pareto_cost_wait.png")
    fig.savefig(OUT / "figW4_pareto_cost_wait.pdf")
    plt.close(fig)

    # figW5: per-provider cost curve at window=3
    # Need per-provider cost reconstruction
    log("    Per-provider cost decomposition ...")
    w = 3
    wm = window_mask[w]
    combined = sched_cost + PENALTY * n_pp_arr[:, None] * sched_waits[None, :]
    combined_window = np.where(wm[None, :], combined, np.inf)
    best_si_window = np.argmin(combined_window, axis=1)
    cost_batched_w3 = sched_cost[np.arange(n_pp), best_si_window]

    prov_rows = []
    providers = sorted(set(pp["provider"] for pp in pp_meta))
    for share in SHARE_GRID:
        prov_costs = defaultdict(float)
        prov_pkts = defaultdict(float)
        for ppi, pp in enumerate(pp_meta):
            cd = cost_daily[ppi]
            cb = cost_batched_w3[ppi]
            blended = (1 - share) * cd + share * cb
            prov_costs[pp["provider"]] += blended
            prov_pkts[pp["provider"]] += n_pp_arr[ppi]
        for p in providers:
            prov_rows.append({"share": share, "provider": p,
                                "cost_eur": prov_costs[p], "pkts": prov_pkts[p]})
    df_prov = pd.DataFrame(prov_rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p in providers:
        sub = df_prov[df_prov.provider == p].sort_values("share")
        ax.plot(sub.share * 100, sub.cost_eur / 1e3, "o-",
                 linewidth=1.6, markersize=4, label=p)
    ax.set_xlabel("Share of customers willing to wait [%]")
    ax.set_ylabel("Weekly routing cost [k€]")
    ax.set_title(f"Cost trajectory by logistics provider (window = 3 days, $P={PENALTY}$)")
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figW5_provider_cost_vs_share.png")
    fig.savefig(OUT / "figW5_provider_cost_vs_share.pdf")
    plt.close(fig)

    # 6) REPORT
    log("[3] Writing REPORT.md ...")
    lines = [
        "# Willingness-to-Wait 2D Sensitivity (Daganzo-LGB-Hybrid @ P=0.5)",
        f"\nOperating point: WAITING_PENALTY P = **{PENALTY} €/parcel/day**",
        f"Postponement windows: {WINDOW_GRID}",
        f"Share grid: {len(SHARE_GRID)} points 0 → 100%",
        f"\n## Headline numbers (cost in k€, baseline = all-daily 1,977 k€)",
        "\n| Window | 0% willing | 50% willing | 100% willing | Saving @100% | ⌀ Wait @100% |",
        "|---|---|---|---|---|---|",
    ]
    base = float(df[df.share_willing == 0.0].iloc[0].total_cost_eur)
    for w in WINDOW_GRID:
        sub = df[df.window == w].set_index("share_willing")
        c0 = sub.loc[0.0, "total_cost_eur"] / 1e3
        c5 = sub.loc[0.5, "total_cost_eur"] / 1e3
        c1 = sub.loc[1.0, "total_cost_eur"] / 1e3
        w1 = sub.loc[1.0, "avg_wait_days_weighted"]
        sav = 100.0 * (1 - c1 / c0)
        lines.append(f"| {w} day | {c0:,.0f} | {c5:,.0f} | {c1:,.0f} | {sav:.1f}% | {w1:.3f}d |")
    lines.append(
        "\n*Cost blend formula:* `cost(share, window) = (1-share) · cost_daily + "
        "share · cost_batched(window, P=0.5)`. The batched portion picks the "
        "schedule minimising `cost + P · pkts · wait_days`. Wait days only "
        "accrue on the batched portion."
    )
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    log(f"\nDone in {time.time()-t0:.0f}s. Outputs in: {OUT}")
    for p in sorted(OUT.glob("*")):
        log(f"  {p.name}")


if __name__ == "__main__":
    main()
