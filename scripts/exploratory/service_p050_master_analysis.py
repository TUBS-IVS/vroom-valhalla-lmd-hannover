"""Master Analysis at WAITING_PENALTY = 0.50 €/parcel/day (Sweet-Spot).

Three sub-analyses in one folder `results/service_p050_final/`:

  sensitivity_break_even/
    tab_marginal_cost_per_wait_day.csv
    tab_break_even_per_plz.csv         (at which P does each PLZ flip daily→batched?)
    fig_break_even_curve.png/pdf
    fig_marginal_cost_curve.png/pdf

  schedule_analysis/
    tab_chosen_schedules.csv           (per (provider, plz): weekdays, mix, cost, wait)
    tab_schedule_pattern_counts.csv    (which weekday-combos are picked, how often)
    fig_schedule_size_distribution.png/pdf
    fig_schedule_weekday_heatmap.png/pdf  (active weekdays per PLZ × schedule)
    fig_provider_schedule_breakdown.png/pdf
    fig_feature_boxplots.png/pdf       (weekly_pkts / hub_dist / area / b2c by schedule size)
    tab_feature_importance.csv         (RF feature importance for schedule-size class)
    tab_feature_correlations.csv       (Spearman vs schedule-size)

  willingness_to_wait/
    tab_wait_distribution.csv
    fig_wait_histogram.png/pdf
    fig_wait_by_provider.png/pdf
    tab_max_hold_sensitivity.csv       (MAX_HOLD ∈ {2,3} comparison)
    fig_max_hold_sensitivity.png/pdf

Reuses cached `results/penalty_sweep/sched_cost_cache.npz` (size n_pp × 39).
"""
from __future__ import annotations
import pickle, sys, time, warnings
from collections import defaultdict, Counter
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
from sklearn.ensemble import RandomForestClassifier

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

# Paper-ready feature labels (technical name → human label)
FEATURE_LABELS = {
    "weekly_parcels":   "Weekly parcel volume [pkts/wk]",
    "n_stops":          "Unique drop-sites per week",
    "area_km2":         "Service area [km$^2$]",
    "hub_dist_km":      "Distance to depot [km]",
    "b2c_share":        "B2C share of parcels",
    "parcels_per_stop": "Parcels per drop-site [pkts/site/wk]",
    "parcels_per_km2":  "Parcel demand density [pkts/km$^2$/wk]",
    "provider":         "Logistics service provider",
}
SCHED_SIZE_LABEL = "Delivery days per week"
PENALTY_LABEL = "Service penalty $P$ [€/parcel/day]"
COST_LABEL = "Weekly routing cost [k€]"
WAIT_LABEL = "Average customer wait [days]"
MAX_WAIT_LABEL = "Maximum customer wait [days]"

OUT = ROOT / "results" / "service_p050_final"
OUT_SBE = OUT / "sensitivity_break_even"
OUT_SCH = OUT / "schedule_analysis"
OUT_WTW = OUT / "willingness_to_wait"
for d in [OUT_SBE, OUT_SCH, OUT_WTW]:
    d.mkdir(parents=True, exist_ok=True)

N_DAYS = 6
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
PENALTY_TARGET = 0.5

PENALTIES_FINE = np.array([
    0.0, 0.025, 0.05, 0.075,
    0.10, 0.125, 0.15, 0.20, 0.25, 0.30,
    0.40, 0.50, 0.60, 0.75,
    1.00, 1.50, 2.00, 3.00, 5.00,
])


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


def max_wait_days(schedule_days):
    if not schedule_days:
        return float(N_DAYS)
    ds = sorted(schedule_days)
    gaps = []
    for i in range(len(ds)):
        gap = (ds[(i + 1) % len(ds)] - ds[i]) % N_DAYS
        if gap == 0:
            gap = N_DAYS
        gaps.append(gap)
    return float(max(gaps) - 1)  # wait days, not gap days (gap 3 = wait 2)


def load_cache():
    cache_path = ROOT / "results/penalty_sweep/sched_cost_cache.npz"
    d = np.load(cache_path)
    return d["sched_cost"], list(d["prov_order"]), list(d["plz_order"])


def build_pp_meta():
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    provider_data = chk["provider_data"]
    optim_data = chk4["optimization_data"]

    pp_meta = []
    for prov, pdata in provider_data.items():
        plz_demand = pdata["plz_demand"]
        for plz in optim_data[prov]["plz_keys"]:
            od = optim_data[prov]["plz_data"][plz]
            row = plz_demand[plz_demand.plz == plz]
            if row.empty:
                continue
            row = row.iloc[0]
            weekly = int(row.weekly_parcels)
            n_stops = int(od["total_points"])
            pp_meta.append({
                "provider": prov,
                "plz": plz,
                "weekly_parcels": weekly,
                "n_stops": n_stops,
                "area_km2": float(od["area_km2"]),
                "hub_dist_km": float(od["hub_dist_km"]),
                "b2c_share": float(row.b2c_weekly / max(1, weekly)),
                "parcels_per_stop": weekly / max(1, n_stops),
                "parcels_per_km2": weekly / max(0.01, float(od["area_km2"])),
            })
    return pp_meta


def part_A_break_even(sched_cost, sched_sizes, sched_waits, n_pp_arr, pp_meta):
    log("=" * 72)
    log("PART A — Sensitivity / Break-Even")
    log("=" * 72)
    n_pp = sched_cost.shape[0]

    # A.1 Marginal cost of service along the penalty curve
    rows = []
    for P in PENALTIES_FINE:
        combined = sched_cost + P * n_pp_arr[:, None] * sched_waits[None, :]
        best_si = np.argmin(combined, axis=1)
        cost = float(sched_cost[np.arange(n_pp), best_si].sum())
        wait_w = float((sched_waits[best_si] * n_pp_arr).sum() / n_pp_arr.sum())
        n_daily = int((sched_sizes[best_si] == N_DAYS).sum())
        rows.append({"penalty": float(P), "cost_eur": cost, "avg_wait_d": wait_w,
                      "n_daily_plz": n_daily})
    mc = pd.DataFrame(rows).sort_values("penalty").reset_index(drop=True)
    # Marginal: dcost / dwait between successive points
    mc["dcost"] = -mc["cost_eur"].diff()
    mc["dwait"] = mc["avg_wait_d"].diff()
    mc["marg_cost_per_wait_d"] = mc["dcost"] / mc["dwait"].replace(0, np.nan)
    mc.to_csv(OUT_SBE / "tab_marginal_cost_per_wait_day.csv", index=False)
    log(f"  marginal cost table: {len(mc)} rows")

    # A.2 Break-even per PLZ: at which P does PLZ switch from non-daily to daily?
    daily_mask = sched_sizes == N_DAYS
    batched_mask = sched_sizes < N_DAYS
    break_evens = []
    for ppi in range(n_pp):
        best_daily_cost = sched_cost[ppi, daily_mask].min()
        # For each P find best non-daily combined cost
        # Solve: P*pkts*wait + batched_cost_min(P) = best_daily_cost
        # Search over a fine P grid where the chosen schedule switches
        # Simple: scan P grid and find first P where daily becomes best
        flip_p = None
        for P in PENALTIES_FINE:
            combined = sched_cost[ppi] + P * n_pp_arr[ppi] * sched_waits
            best_si = int(np.argmin(combined))
            if sched_sizes[best_si] == N_DAYS:
                flip_p = float(P)
                break
        # Even at P=PENALTIES_FINE.max() the PLZ might still prefer batched (unlikely with 5€/p/d)
        break_evens.append(flip_p if flip_p is not None else float("inf"))

    df_be = pd.DataFrame(pp_meta)
    df_be["break_even_penalty"] = break_evens
    df_be.to_csv(OUT_SBE / "tab_break_even_per_plz.csv", index=False)
    log(f"  break-even per PLZ saved ({len(df_be)} rows)")

    # Plot A.1 marginal cost curve
    fig, ax = plt.subplots(figsize=(7, 4.5))
    valid = mc.dropna(subset=["marg_cost_per_wait_d"])
    pen_mid = valid.penalty.values
    ax.plot(pen_mid, valid.marg_cost_per_wait_d.values / 1e3, "o-",
            linewidth=2, color="#0c4f7a")
    ax.axhline(1000, ls=":", color="grey", label="1 M€ per wait-day saved")
    ax.axvline(PENALTY_TARGET, ls="--", color="#b3261e",
                label=f"$P={PENALTY_TARGET}$ €/pkt/day (operating point)")
    ax.set_xlabel(PENALTY_LABEL)
    ax.set_ylabel("Marginal cost of service\n[k€ saved per wait-day gained]")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_title("Marginal cost of service along the Pareto frontier")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_SBE / "fig_marginal_cost_curve.png")
    fig.savefig(OUT_SBE / "fig_marginal_cost_curve.pdf")
    plt.close(fig)

    # Plot A.2 cumulative break-even: x = penalty, y = % of PLZ already on daily
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = np.array(break_evens)
    bp_sorted = np.sort(bp[np.isfinite(bp)])
    p_range = np.linspace(0, max(5.0, PENALTIES_FINE.max()), 200)
    pct = np.array([100 * (bp_sorted <= p).sum() / len(bp_sorted) for p in p_range])
    ax.plot(p_range, pct, "-", linewidth=2, color="#0c4f7a")
    ax.axvline(PENALTY_TARGET, ls="--", color="#b3261e",
                label=f"$P={PENALTY_TARGET}$ €/pkt/day")
    ax.set_xlabel(PENALTY_LABEL)
    ax.set_ylabel("Share of (provider, PLZ) cells\non daily delivery [%]")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_title("Adoption of daily delivery as service penalty rises")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_SBE / "fig_break_even_curve.png")
    fig.savefig(OUT_SBE / "fig_break_even_curve.pdf")
    plt.close(fig)


def part_B_schedule(sched_cost, schedules, sched_sizes, sched_waits, n_pp_arr, pp_meta):
    log("=" * 72)
    log("PART B — Schedule Analysis @ P=0.5")
    log("=" * 72)
    P = PENALTY_TARGET
    n_pp = sched_cost.shape[0]
    combined = sched_cost + P * n_pp_arr[:, None] * sched_waits[None, :]
    best_si = np.argmin(combined, axis=1)

    chosen = []
    for ppi, pp in enumerate(pp_meta):
        si = int(best_si[ppi])
        sched_set = sorted(schedules[si])
        chosen.append({
            **pp,
            "schedule_size": int(sched_sizes[si]),
            "schedule_weekdays": ",".join(WEEKDAYS[d] for d in sched_set),
            "schedule_idx": si,
            "ml_cost_eur": float(sched_cost[ppi, si]),
            "avg_wait_d": float(sched_waits[si]),
            "max_wait_d": max_wait_days(sched_set),
            "service_penalty_eur": float(P * n_pp_arr[ppi] * sched_waits[si]),
        })
    df_chosen = pd.DataFrame(chosen)
    df_chosen.to_csv(OUT_SCH / "tab_chosen_schedules.csv", index=False)
    log(f"  chosen schedules saved ({len(df_chosen)} rows)")

    # Schedule pattern counts
    pat_ct = (df_chosen.groupby(["schedule_size", "schedule_weekdays"], as_index=False)
              .size()
              .sort_values(["schedule_size", "size"], ascending=[True, False]))
    pat_ct.to_csv(OUT_SCH / "tab_schedule_pattern_counts.csv", index=False)
    log(f"  unique schedule patterns picked: {len(pat_ct)}")

    # Plot B.1 schedule-size distribution
    size_ct = df_chosen.schedule_size.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(size_ct.index, size_ct.values,
                   color=plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(size_ct))),
                   edgecolor="black")
    for b, v in zip(bars, size_ct.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, str(v),
                ha="center", fontsize=9)
    ax.set_xlabel(SCHED_SIZE_LABEL)
    ax.set_ylabel("Count of (provider, PLZ) cells")
    ax.set_title(f"Chosen delivery frequency at operating point $P={P}$ €/parcel/day")
    fig.tight_layout()
    fig.savefig(OUT_SCH / "fig_schedule_size_distribution.png")
    fig.savefig(OUT_SCH / "fig_schedule_size_distribution.pdf")
    plt.close(fig)

    # Plot B.2 weekday usage heatmap: rows = schedule_size, cols = weekday
    weekday_share = np.zeros((6, 6))  # schedule_size 1..6, weekday Mon..Sat
    weekday_count = np.zeros(6)  # total per schedule_size
    for _, r in df_chosen.iterrows():
        sz = r.schedule_size - 1
        weekday_count[sz] += 1
        for wd in r.schedule_weekdays.split(","):
            j = WEEKDAYS.index(wd)
            weekday_share[sz, j] += 1
    for i in range(6):
        if weekday_count[i] > 0:
            weekday_share[i] /= weekday_count[i]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(weekday_share, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels(WEEKDAYS)
    ax.set_yticks(range(6))
    ax.set_yticklabels([f"{k+1} day/wk ($n$={int(weekday_count[k])})" for k in range(6)])
    ax.set_xlabel("Weekday")
    ax.set_ylabel(SCHED_SIZE_LABEL)
    ax.set_title(f"Share of cells delivering on each weekday by delivery frequency\n(operating point $P={P}$ €/parcel/day)")
    for i in range(6):
        for j in range(6):
            v = weekday_share[i, j]
            if weekday_count[i] > 0:
                ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, label="Share of cells")
    fig.tight_layout()
    fig.savefig(OUT_SCH / "fig_schedule_weekday_heatmap.png")
    fig.savefig(OUT_SCH / "fig_schedule_weekday_heatmap.pdf")
    plt.close(fig)

    # Plot B.3 provider breakdown
    pv = (df_chosen.groupby(["provider", "schedule_size"])
          .size().unstack(fill_value=0).reindex(columns=range(1, 7), fill_value=0))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottoms = np.zeros(len(pv))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, 6))
    for k_idx, sz in enumerate(range(1, 7)):
        ax.bar(pv.index, pv[sz].values, bottom=bottoms,
                label=f"{sz} day/wk", color=colors[k_idx], edgecolor="white")
        bottoms += pv[sz].values
    ax.set_xlabel("Logistics service provider")
    ax.set_ylabel("Count of PLZ areas")
    ax.set_title(f"Delivery-frequency mix per provider at operating point $P={P}$ €/parcel/day")
    ax.legend(title=SCHED_SIZE_LABEL, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_SCH / "fig_provider_schedule_breakdown.png")
    fig.savefig(OUT_SCH / "fig_provider_schedule_breakdown.pdf")
    plt.close(fig)

    # Plot B.4 feature boxplots vs schedule size
    feats = ["weekly_parcels", "n_stops", "area_km2", "hub_dist_km",
              "b2c_share", "parcels_per_stop", "parcels_per_km2"]
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    axs = axs.flatten()
    for i, feat in enumerate(feats):
        data_by_size = [df_chosen[df_chosen.schedule_size == sz][feat].values
                        for sz in range(1, 7)]
        axs[i].boxplot(data_by_size, labels=[str(s) for s in range(1, 7)],
                        showfliers=False, patch_artist=True,
                        boxprops=dict(facecolor="#cfdcec", edgecolor="#1f4f8f"),
                        medianprops=dict(color="#b3261e", linewidth=1.5))
        axs[i].set_title(FEATURE_LABELS.get(feat, feat), fontsize=10)
        axs[i].set_xlabel(SCHED_SIZE_LABEL, fontsize=9)
        axs[i].grid(alpha=0.3)
    axs[-1].axis("off")
    fig.suptitle(
        f"Feature distribution by chosen delivery frequency "
        f"(operating point $P={P}$ €/parcel/day)",
        y=1.0, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_SCH / "fig_feature_boxplots.png")
    fig.savefig(OUT_SCH / "fig_feature_boxplots.pdf")
    plt.close(fig)

    # Spearman correlations
    corr_rows = []
    for feat in feats:
        rho, p = spearmanr(df_chosen[feat].values, df_chosen.schedule_size.values)
        corr_rows.append({"feature": feat, "spearman_rho": rho, "p_value": p})
    df_corr = pd.DataFrame(corr_rows).sort_values("spearman_rho", key=abs, ascending=False)
    df_corr.to_csv(OUT_SCH / "tab_feature_correlations.csv", index=False)
    log("  Spearman correlations with schedule_size (top 5):")
    for _, r in df_corr.head(5).iterrows():
        log(f"      {r.feature:25s}  rho={r.spearman_rho:+.3f}  p={r.p_value:.2e}")

    # Random Forest feature importance
    X = df_chosen[feats + ["provider"]].copy()
    X["provider"] = X["provider"].map(_PROVIDER_IDX).fillna(-1)
    y = df_chosen.schedule_size.values
    rf = RandomForestClassifier(n_estimators=400, max_depth=8, random_state=42,
                                  n_jobs=-1)
    rf.fit(X, y)
    fi = pd.DataFrame({"feature": list(X.columns), "importance": rf.feature_importances_}) \
        .sort_values("importance", ascending=False)
    fi.to_csv(OUT_SCH / "tab_feature_importance.csv", index=False)
    log(f"  RF accuracy on training set: {rf.score(X, y):.3f}")
    log("  Feature importance (top 5):")
    for _, r in fi.head(5).iterrows():
        log(f"      {r.feature:25s}  {r.importance:.3f}")


def part_C_wait(sched_cost_by_hold, schedules_by_hold, n_pp_arr, pp_meta):
    log("=" * 72)
    log("PART C — Willingness to Wait")
    log("=" * 72)
    P = PENALTY_TARGET
    # Use MAX_HOLD=3 base (default) for the wait distribution
    sched_cost, schedules = sched_cost_by_hold[3]
    sched_waits = np.array([avg_wait_days(sorted(s)) for s in schedules])
    sched_sizes = np.array([len(s) for s in schedules])

    n_pp = sched_cost.shape[0]
    combined = sched_cost + P * n_pp_arr[:, None] * sched_waits[None, :]
    best_si = np.argmin(combined, axis=1)

    rows = []
    for ppi, pp in enumerate(pp_meta):
        si = int(best_si[ppi])
        sched_set = sorted(schedules[si])
        rows.append({
            **pp,
            "schedule_size": int(sched_sizes[si]),
            "avg_wait_d": float(sched_waits[si]),
            "max_wait_d": max_wait_days(sched_set),
        })
    df_wait = pd.DataFrame(rows)
    df_wait.to_csv(OUT_WTW / "tab_wait_distribution.csv", index=False)

    # Plot C.1 wait histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.hist(df_wait.avg_wait_d.values, bins=20, color="#1f4f8f",
              edgecolor="white", alpha=0.85)
    ax1.set_xlabel(WAIT_LABEL)
    ax1.set_ylabel("Count of (provider, PLZ) cells")
    ax1.set_title(f"Average wait time across cells\n(operating point $P={P}$ €/parcel/day)")
    ax1.grid(alpha=0.3)

    ax2.hist(df_wait.max_wait_d.values, bins=range(0, 4),
              color="#b3261e", edgecolor="white", alpha=0.85)
    ax2.set_xlabel(MAX_WAIT_LABEL)
    ax2.set_ylabel("Count of (provider, PLZ) cells")
    ax2.set_title(f"Worst-case wait per cell\n(MAX_HOLDING_DAYS = 3 constraint)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_WTW / "fig_wait_histogram.png")
    fig.savefig(OUT_WTW / "fig_wait_histogram.pdf")
    plt.close(fig)

    # Plot C.2 wait by provider (boxplots)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    by_prov = df_wait.groupby("provider")["avg_wait_d"].apply(list)
    ax.boxplot(by_prov.values, labels=by_prov.index, showfliers=True,
                patch_artist=True,
                boxprops=dict(facecolor="#cfdcec", edgecolor="#1f4f8f"),
                medianprops=dict(color="#b3261e", linewidth=1.5))
    ax.set_xlabel("Logistics service provider")
    ax.set_ylabel(WAIT_LABEL)
    ax.set_title(f"Customer wait distribution per provider at $P={P}$ €/parcel/day")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_WTW / "fig_wait_by_provider.png")
    fig.savefig(OUT_WTW / "fig_wait_by_provider.pdf")
    plt.close(fig)

    # Plot C.3 MAX_HOLD sensitivity (MAX_HOLD=2 vs MAX_HOLD=3)
    rows = []
    for max_hold, (sc, sd) in sched_cost_by_hold.items():
        sw = np.array([avg_wait_days(sorted(s)) for s in sd])
        comb = sc + P * n_pp_arr[:, None] * sw[None, :]
        best = np.argmin(comb, axis=1)
        chosen_cost = float(sc[np.arange(n_pp), best].sum())
        chosen_wait = float((sw[best] * n_pp_arr).sum() / n_pp_arr.sum())
        sizes = np.array([len(s) for s in sd])
        n_daily = int((sizes[best] == N_DAYS).sum())
        rows.append({"max_hold": max_hold, "n_schedules": len(sd),
                      "total_cost_eur": chosen_cost,
                      "avg_wait_d": chosen_wait,
                      "n_daily_plz": n_daily,
                      "n_batched_plz": n_pp - n_daily})
    df_mh = pd.DataFrame(rows)
    df_mh.to_csv(OUT_WTW / "tab_max_hold_sensitivity.csv", index=False)
    log(f"  MAX_HOLD sensitivity:")
    for _, r in df_mh.iterrows():
        log(f"     MAX_HOLD={int(r.max_hold)}  cost={r.total_cost_eur/1e3:.1f}k€  "
            f"wait={r.avg_wait_d:.3f}d  daily={int(r.n_daily_plz)}")

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.bar(df_mh.max_hold.astype(str), df_mh.total_cost_eur / 1e3,
             color="#1f4f8f", alpha=0.7, label="Cost")
    ax1.set_xlabel("MAX_HOLDING_DAYS constraint")
    ax1.set_ylabel(COST_LABEL, color="#1f4f8f")
    ax1.tick_params(axis="y", labelcolor="#1f4f8f")
    ax2 = ax1.twinx()
    ax2.plot(df_mh.max_hold.astype(str), df_mh.avg_wait_d, "s-",
              color="#b3261e", linewidth=2, label="Wait")
    ax2.set_ylabel(WAIT_LABEL, color="#b3261e")
    ax2.tick_params(axis="y", labelcolor="#b3261e")
    ax2.spines["right"].set_visible(True)
    ax1.set_title(f"Sensitivity of cost and wait time to MAX_HOLDING_DAYS "
                  f"at $P={P}$ €/parcel/day")
    fig.tight_layout()
    fig.savefig(OUT_WTW / "fig_max_hold_sensitivity.png")
    fig.savefig(OUT_WTW / "fig_max_hold_sensitivity.pdf")
    plt.close(fig)


def main():
    t0 = time.time()
    log("Master Analysis @ P=0.50 €/parcel/day (Sweet-Spot)")

    # 1) load cached sched_cost (39 schedules @ MAX_HOLD=3)
    log("[1] Loading cached sched_cost ...")
    sched_cost_h3, prov_cache, plz_cache = load_cache()
    log(f"    sched_cost_h3.shape={sched_cost_h3.shape}")

    # 2) reconstruct pp_meta in cache order
    log("[2] Building pp_meta in cache order ...")
    pp_meta_full = build_pp_meta()
    by_key = {(pp["provider"], pp["plz"]): pp for pp in pp_meta_full}
    pp_meta = []
    for prov, plz in zip(prov_cache, plz_cache):
        key = (prov, plz)
        if key in by_key:
            pp_meta.append(by_key[key])
        else:
            log(f"    MISSING key in pp_meta_full: {key}")
    log(f"    pp_meta length: {len(pp_meta)} (cache has {len(prov_cache)})")
    n_pp_arr = np.array([pp["weekly_parcels"] for pp in pp_meta], dtype=np.float64)

    # 3) schedules for MAX_HOLD=3 and MAX_HOLD=2 (subset of h3)
    schedules_h3 = enumerate_schedules(3)
    schedules_h2 = enumerate_schedules(2)
    log(f"    {len(schedules_h3)} schedules @ MAX_HOLD=3, {len(schedules_h2)} @ MAX_HOLD=2")
    sched_sizes_h3 = np.array([len(s) for s in schedules_h3])
    sched_waits_h3 = np.array([avg_wait_days(sorted(s)) for s in schedules_h3])

    # 4) sched_cost @ MAX_HOLD=2: subset by index
    h3_idx = {s: i for i, s in enumerate(schedules_h3)}
    h2_indices = np.array([h3_idx[s] for s in schedules_h2 if s in h3_idx])
    sched_cost_h2 = sched_cost_h3[:, h2_indices]
    schedules_h2_in_h3_order = [schedules_h2[k] for k in range(len(schedules_h2))
                                  if schedules_h2[k] in h3_idx]
    log(f"    h2-subset matrix shape: {sched_cost_h2.shape}")

    sched_cost_by_hold = {
        3: (sched_cost_h3, schedules_h3),
        2: (sched_cost_h2, schedules_h2_in_h3_order),
    }

    # PART A
    part_A_break_even(sched_cost_h3, sched_sizes_h3, sched_waits_h3, n_pp_arr, pp_meta)
    # PART B
    part_B_schedule(sched_cost_h3, schedules_h3, sched_sizes_h3, sched_waits_h3,
                    n_pp_arr, pp_meta)
    # PART C
    part_C_wait(sched_cost_by_hold, None, n_pp_arr, pp_meta)

    log(f"\nDone in {time.time()-t0:.0f}s. Outputs in: {OUT}")
    for sub in [OUT_SBE, OUT_SCH, OUT_WTW]:
        log(f"  {sub.name}/")
        for p in sorted(sub.glob("*")):
            log(f"    {p.name}")


if __name__ == "__main__":
    main()
