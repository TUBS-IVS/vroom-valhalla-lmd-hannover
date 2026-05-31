"""1D Penalty Sweep — Service-Cost Pareto for the paper.

Per (provider, plz), pick the schedule minimizing
    combined = ml_cost + P * weekly_parcels * avg_wait_days

across a fine WAITING_PENALTY grid P. No fixed batch_share — the optimizer
naturally picks how aggressively to batch as P shrinks.

Outputs:
    results/penalty_sweep/
        sched_cost_cache.npz       (cached prediction matrix; reused on re-run)
        tab_penalty_pareto.csv     (one row per penalty)
        tab_pareto_optimal.csv     (Pareto-optimal subset)
        fig_pareto_cost_wait.{png,pdf}    (the main Pareto plot)
        fig_delivery_day_mix.{png,pdf}    (PLZ schedule-size distribution per penalty)
        fig_cost_vs_penalty.{png,pdf}     (cost trajectory + wait days as dual axis)
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
from batch_delivery.io.demand import get_source_days  # noqa: E402

rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

PENALTY_LABEL = "Service penalty $P$ [€/parcel/day]"
COST_LABEL    = "Weekly routing cost [k€]"
WAIT_LABEL    = "Average customer wait [days]"
SCHED_SIZE_LABEL = "Delivery days per week"

OUT = ROOT / "results" / "penalty_sweep"
OUT.mkdir(parents=True, exist_ok=True)
N_DAYS = 6
MAX_HOLD = 3
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

PENALTIES = np.array([
    0.0, 0.025, 0.05, 0.075,
    0.10, 0.125, 0.15, 0.20, 0.25, 0.30,
    0.40, 0.50, 0.60, 0.75,
    1.00, 1.50, 2.00, 3.00, 5.00, 10.0,
])

CACHE = OUT / "sched_cost_cache.npz"


def log(msg):
    print(msg, flush=True)


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


def load_model(model_path: Path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from train_daganzo_hybrid import DaganzoLGBHybrid, _LGBIdentityWrap  # noqa
    import __main__
    __main__._LGBIdentityWrap = _LGBIdentityWrap
    with open(model_path, "rb") as f:
        d = pickle.load(f)
    if d.get("kind") == "DaganzoLGBHybrid":
        return DaganzoLGBHybrid(model=d["model"], combo_cols=d["combo_cols"],
                                 alpha=d["alpha"])
    from batch_delivery.surrogate.lgb_adapter import LGBLogTSurrogate
    return LGBLogTSurrogate.load(model_path)


def build_pp_list():
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    provider_data = chk["provider_data"]
    optim_data = chk4["optimization_data"]

    pp_list = []
    for prov, pdata in provider_data.items():
        plz_demand = pdata["plz_demand"]
        for plz in optim_data[prov]["plz_keys"]:
            od = optim_data[prov]["plz_data"][plz]
            row = plz_demand[plz_demand.plz == plz]
            if row.empty:
                continue
            row = row.iloc[0]
            b2c = {i: int(row[f"arrivals_b2c_{WEEKDAYS[i]}"]) for i in range(6)}
            b2b = {i: int(row[f"arrivals_b2b_{WEEKDAYS[i]}"]) for i in range(6)}
            total_daily = {i: b2c[i] + b2b[i] for i in range(6)}
            base = {c: 0.0 for c in ALL_COLS}
            base.update({
                "area_km2": float(od["area_km2"]),
                "hub_dist_km": float(od["hub_dist_km"]),
                "n_stops": float(od["total_points"]),
                "b2c_share": float(row.b2c_weekly / max(1, row.weekly_parcels)),
                "provider_idx": float(_PROVIDER_IDX.get(prov, 0)),
                "centroid_hub_dist_km": float(od["hub_dist_km"]),
                "max_hub_dist_km": float(od["hub_dist_km"]) * 1.2,
                "demand_std": max(1.0, row.weekly_parcels / max(1, od["total_points"])) * 0.3,
                "max_stop_demand": max(1.0, row.weekly_parcels / max(1, od["total_points"])) * 2,
                "ch_area_km2": float(od["area_km2"]) * 0.6,
                "ch_perimeter_km": np.sqrt(float(od["area_km2"])) * 4,
                "mean_nn_dist_km": 0.15,
                "mean_inter_stop_dist_km": np.sqrt(float(od["area_km2"])) * 0.4,
                "stop_density_ch": od["total_points"] / max(0.01, od["area_km2"] * 0.6),
                "coord_std_x": np.sqrt(float(od["area_km2"])) * 0.3,
                "coord_std_y": np.sqrt(float(od["area_km2"])) * 0.3,
                "aspect_ratio": 1.2,
            })
            pp_list.append({
                "provider": prov, "plz": plz,
                "weekly_parcels": int(row.weekly_parcels),
                "base": base, "daily": total_daily,
            })
    return pp_list


def compute_sched_cost(model, pp_list, schedules, sched_source):
    n_pp = len(pp_list)
    n_s = len(schedules)
    sched_cost = np.zeros((n_pp, n_s), dtype=np.float64)
    t_predict_start = time.time()
    log_every = max(1, n_pp // 20)
    for ppi, pp in enumerate(pp_list):
        rows = []
        idx_map = []
        for si, sched in enumerate(schedules):
            sched_list = sorted(sched)
            for dd in sched_list:
                src_days = sched_source[si][dd]
                n_parcels_dd = sum(pp["daily"][d] for d in src_days)
                if n_parcels_dd <= 0:
                    continue
                f = pp["base"].copy()
                f["n_parcels"] = n_parcels_dd
                f["parcels_per_stop"] = n_parcels_dd / max(1, f["n_stops"])
                f["load_factor"] = n_parcels_dd / 230
                f["min_vehicles"] = max(1, int(np.ceil(n_parcels_dd / 230)))
                f["parcels_per_km2"] = n_parcels_dd / max(0.01, f["area_km2"])
                f["demand_cap_ratio"] = n_parcels_dd / (f["min_vehicles"] * 230)
                f["day_idx"] = dd
                f["delivery_frequency"] = len(sched)
                rows.append([f[c] for c in ALL_COLS])
                idx_map.append(si)
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=ALL_COLS)
        costs = model.predict(df)
        for k, si in enumerate(idx_map):
            sched_cost[ppi, si] += float(costs[k])
        if (ppi + 1) % log_every == 0 or ppi == n_pp - 1:
            elapsed = time.time() - t_predict_start
            eta = elapsed * (n_pp - ppi - 1) / max(1, ppi + 1)
            log(f"      [{ppi+1}/{n_pp}] {pp['provider']:>7s} {pp['plz']}  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")
    log(f"      Done predicting in {time.time()-t_predict_start:.0f}s")
    return sched_cost


def main():
    t_start = time.time()

    log("[1/5] Building (provider, plz) tuples + schedules...")
    pp_list = build_pp_list()
    log(f"      {len(pp_list)} (provider, plz) tuples")

    schedules = enumerate_schedules()
    sched_sizes = np.array([len(s) for s in schedules])
    sched_waits = np.array([avg_wait_days(sorted(s)) for s in schedules])
    sched_source = [
        {dd: get_source_days(dd, sorted(s)) for dd in sorted(s)}
        for s in schedules
    ]
    log(f"      {len(schedules)} valid schedules (max_hold={MAX_HOLD})")

    # Try cache first
    if CACHE.exists():
        log(f"[2/5] Loading cached sched_cost from {CACHE.name}...")
        d = np.load(CACHE)
        sched_cost = d["sched_cost"]
        plz_order = list(d["plz_order"])
        prov_order = list(d["prov_order"])
        # Sanity: re-key pp_list to cache order
        current_keys = [(pp["provider"], pp["plz"]) for pp in pp_list]
        cache_keys = list(zip(prov_order, plz_order))
        if current_keys != cache_keys:
            log("      WARN: pp_list mismatch with cache, recomputing...")
            sched_cost = None
        else:
            log(f"      Loaded {sched_cost.shape} cost matrix from cache")
    else:
        sched_cost = None

    if sched_cost is None:
        log("[2/5] Loading model + batch-predicting sched_cost...")
        model = load_model(ROOT / "results/sweep_v3_mergefix/daganzo_hybrid_v3aug_median.pkl")
        sched_cost = compute_sched_cost(model, pp_list, schedules, sched_source)
        np.savez_compressed(
            CACHE,
            sched_cost=sched_cost,
            plz_order=np.array([pp["plz"] for pp in pp_list]),
            prov_order=np.array([pp["provider"] for pp in pp_list]),
        )
        log(f"      Cached to {CACHE}")

    n_pp = len(pp_list)
    n_pp_arr = np.array([pp["weekly_parcels"] for pp in pp_list], dtype=np.float64)

    log("[3/5] Sweeping penalties — per-PLZ argmin(cost + P*pkts*wait)...")
    rows = []
    # Per-pp / per-penalty schedule choices, used later for mix figure
    pp_choices_per_pen = {}
    for P in PENALTIES:
        combined = sched_cost + P * n_pp_arr[:, None] * sched_waits[None, :]
        best_si = np.argmin(combined, axis=1)
        pp_choices_per_pen[float(P)] = best_si

        chosen_cost = sched_cost[np.arange(n_pp), best_si]
        chosen_wait = sched_waits[best_si]
        chosen_size = sched_sizes[best_si]

        total_cost = float(chosen_cost.sum())
        total_pkts = float(n_pp_arr.sum())
        avg_wait = float((chosen_wait * n_pp_arr).sum() / total_pkts)
        max_wait_plz = float(chosen_wait.max())

        mix = defaultdict(int)
        pkts_per_size = defaultdict(float)
        for sz, n_pk in zip(chosen_size, n_pp_arr):
            mix[int(sz)] += 1
            pkts_per_size[int(sz)] += float(n_pk)

        rows.append({
            "penalty": float(P),
            "total_cost_eur": total_cost,
            "avg_wait_days": avg_wait,
            "max_wait_days_per_plz": max_wait_plz,
            "n_pkts_weekly": int(total_pkts),
            "n_batched_plz": int((chosen_size < N_DAYS).sum()),
            "n_daily_plz": int((chosen_size == N_DAYS).sum()),
            **{f"mix_{k}day": mix.get(k, 0) for k in range(1, 7)},
            **{f"pkts_{k}day": pkts_per_size.get(k, 0.0) for k in range(1, 7)},
        })
        log(f"      P={P:6.3f}  cost={total_cost/1e3:7.1f}k€  wait={avg_wait:.3f}d  "
            f"batched={rows[-1]['n_batched_plz']}/{n_pp}")

    df = pd.DataFrame(rows)
    baseline_cost = float(df[df.penalty == df.penalty.max()].total_cost_eur.iloc[0])  # high P → all daily
    df["cost_savings_pct"] = 100.0 * (baseline_cost - df.total_cost_eur) / baseline_cost
    df.to_csv(OUT / "tab_penalty_pareto.csv", index=False)

    # Pareto subset
    df["is_pareto"] = False
    for i, r in df.iterrows():
        dominated = ((df.total_cost_eur < r.total_cost_eur) &
                     (df.avg_wait_days <= r.avg_wait_days)) | \
                    ((df.total_cost_eur <= r.total_cost_eur) &
                     (df.avg_wait_days < r.avg_wait_days))
        df.loc[i, "is_pareto"] = not dominated.any()
    df[df.is_pareto].to_csv(OUT / "tab_pareto_optimal.csv", index=False)
    log(f"      {df.is_pareto.sum()} / {len(df)} Pareto-optimal cells")

    log("[4/5] Plot 1 — Pareto front (cost vs avg wait days)...")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    s = df.sort_values("avg_wait_days")
    ax.plot(s.avg_wait_days, s.total_cost_eur / 1e3, "o-",
            color="#1f4f8f", linewidth=2, markersize=6, alpha=0.85)
    annotate_idx = list(range(0, len(s), 2))
    if len(s) - 1 not in annotate_idx:
        annotate_idx.append(len(s) - 1)
    for i in annotate_idx:
        r = s.iloc[i]
        ax.annotate(f"$P={r.penalty:g}$", (r.avg_wait_days, r.total_cost_eur / 1e3),
                    xytext=(8, 5), textcoords="offset points", fontsize=8.5,
                    color="black", alpha=0.9)
    p_highlight = 0.5
    sub = df[df.penalty == p_highlight]
    if not sub.empty:
        r = sub.iloc[0]
        ax.scatter(r.avg_wait_days, r.total_cost_eur / 1e3, marker="*", s=420,
                   c="gold", edgecolors="black", zorder=10,
                   label=f"Operating point $P={p_highlight}$ €/parcel/day")
        ax.legend(loc="upper right")
    ax.set_xlabel(WAIT_LABEL)
    ax.set_ylabel(COST_LABEL)
    ax.set_title("Service-cost Pareto frontier: penalty sensitivity\n"
                  "(Daganzo-LGB-Hybrid surrogate, 312 (provider, PLZ) cells)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pareto_cost_wait.png")
    fig.savefig(OUT / "fig_pareto_cost_wait.pdf")
    plt.close(fig)

    log("[5/5] Plot 2 — delivery-day mix vs penalty + Plot 3 — cost+wait dual-axis...")
    mix_cols = [f"mix_{k}day" for k in range(1, 7)]
    mix_arr = df[mix_cols].values
    pen_strs = [f"{p:g}" for p in df.penalty]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = np.zeros(len(df))
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, 6))
    for k in range(6):
        ax.bar(pen_strs, mix_arr[:, k], bottom=bottoms,
               label=f"{k+1} day/wk", color=colors[k], edgecolor="white", linewidth=0.3)
        bottoms += mix_arr[:, k]
    ax.set_xlabel(PENALTY_LABEL)
    ax.set_ylabel("Count of (provider, PLZ) cells")
    ax.set_title("Delivery-frequency mix shifts with the service penalty")
    ax.legend(title=SCHED_SIZE_LABEL,
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / "fig_delivery_day_mix.png")
    fig.savefig(OUT / "fig_delivery_day_mix.pdf")
    plt.close(fig)

    # Plot 3: cost vs penalty + avg wait days on second axis
    fig, ax1 = plt.subplots(figsize=(8, 5))
    s = df.sort_values("penalty")
    ax1.plot(s.penalty, s.total_cost_eur / 1e3, "o-", color="#1f4f8f",
             linewidth=2, label="Cost")
    ax1.set_xlabel(PENALTY_LABEL)
    ax1.set_ylabel(COST_LABEL, color="#1f4f8f")
    ax1.tick_params(axis="y", labelcolor="#1f4f8f")
    ax1.set_xscale("symlog", linthresh=0.05)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(s.penalty, s.avg_wait_days, "s--", color="#b3261e",
             linewidth=2, label="Wait")
    ax2.set_ylabel(WAIT_LABEL, color="#b3261e")
    ax2.tick_params(axis="y", labelcolor="#b3261e")
    ax2.spines["right"].set_visible(True)
    ax1.set_title("Cost and customer wait as functions of the service penalty\n"
                   "(log-scaled $P$ axis)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_cost_vs_penalty.png")
    fig.savefig(OUT / "fig_cost_vs_penalty.pdf")
    plt.close(fig)

    log(f"\nDone in {time.time()-t_start:.0f}s. Outputs:")
    for p in sorted(OUT.glob("*")):
        log(f"  {p.name}")


if __name__ == "__main__":
    main()
