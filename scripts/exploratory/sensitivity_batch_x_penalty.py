"""2D Sensitivity Sweep — batch_share x WAITING_PENALTY pareto front.

OPTIMIZED: batch all (schedule, delivery_day) predictions per (provider, plz)
in a single model.predict call. ~100x faster than one-by-one.

For each (batch_share, penalty) in 11x5 grid:
    1. Per (provider, plz), pick best schedule under combined cost
       total_cost = ml_pred + penalty * n_parcels_weekly * avg_wait_days
    2. Enforce batch_share constraint via greedy "biggest-gain" assignment

Outputs:
    results/sensitivity_2d/
        tab_grid_kpis.csv
        tab_pareto_optimal.csv
        fig_pareto_2d.{png,pdf}
        fig_heatmap_cost.{png,pdf}
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
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

OUT = ROOT / "results" / "sensitivity_2d"
OUT.mkdir(parents=True, exist_ok=True)
N_DAYS = 6
MAX_HOLD = 3
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

BATCH_SHARES = np.linspace(0.0, 1.0, 11)
PENALTIES = np.array([0.0, 0.25, 0.5, 1.0, 2.0])


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
                if gap == 0: gap = N_DAYS
                if gap > MAX_HOLD:
                    ok = False; break
            if ok:
                out.append(frozenset(days))
    return out


def avg_wait_days(schedule_days):
    if not schedule_days: return 0.0
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
    import __main__; __main__._LGBIdentityWrap = _LGBIdentityWrap
    with open(model_path, "rb") as f:
        d = pickle.load(f)
    if d.get("kind") == "DaganzoLGBHybrid":
        return DaganzoLGBHybrid(model=d["model"], combo_cols=d["combo_cols"],
                                 alpha=d["alpha"])
    from batch_delivery.surrogate.lgb_adapter import LGBLogTSurrogate
    return LGBLogTSurrogate.load(model_path)


def main():
    t_start = time.time()
    log("[1/6] Loading checkpoints...")
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    provider_data = chk["provider_data"]
    optim_data = chk4["optimization_data"]

    log("[2/6] Loading Daganzo-Hybrid model...")
    model = load_model(ROOT / "results/oracle_loop_extended_2026_05_22/daganzo_hybrid_v2aug.pkl")
    log(f"      Model: {model.__class__.__name__}")

    schedules = enumerate_schedules()
    log(f"      Valid schedules: {len(schedules)}")

    # Pre-compute (delivery_day → source_days) per schedule
    sched_source = []
    sched_waits = []
    for sched in schedules:
        sched_list = sorted(sched)
        src = {dd: get_source_days(dd, sched_list) for dd in sched_list}
        sched_source.append(src)
        sched_waits.append(avg_wait_days(sched_list))
    sched_waits = np.array(sched_waits)

    log("[3/6] Building per-(provider, plz) base feature dicts...")
    pp_list = []
    for prov, pdata in provider_data.items():
        plz_demand = pdata["plz_demand"]
        for plz in optim_data[prov]["plz_keys"]:
            od = optim_data[prov]["plz_data"][plz]
            row = plz_demand[plz_demand.plz == plz]
            if row.empty: continue
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
    log(f"      {len(pp_list)} (provider, plz) tuples")

    log("[4/6] Batch-predict cost(provider, plz, schedule) — this is the heavy part...")
    # For each (provider, plz), build a DataFrame with ALL (schedule, dd) feature rows
    # Predict in ONE model.predict call per pp. Then sum to schedule-level cost.
    n_pp = len(pp_list)
    sched_cost = np.zeros((n_pp, len(schedules)), dtype=np.float64)
    t_predict_start = time.time()
    log_every = max(1, n_pp // 20)
    for ppi, pp in enumerate(pp_list):
        rows = []
        idx_map = []  # which schedule each row belongs to
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
        # Sum costs per schedule
        for k, si in enumerate(idx_map):
            sched_cost[ppi, si] += float(costs[k])
        if (ppi + 1) % log_every == 0 or ppi == n_pp - 1:
            elapsed = time.time() - t_predict_start
            eta = elapsed * (n_pp - ppi - 1) / max(1, ppi + 1)
            log(f"      [{ppi+1}/{n_pp}] {pp['provider']:>7s} {pp['plz']}  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")
    log(f"      Done predicting in {time.time()-t_predict_start:.0f}s")

    # Find best schedule per (pp, agg_k bucket: daily=size 6, batched=<6)
    n_pp_arr = np.array([pp["weekly_parcels"] for pp in pp_list])
    sched_sizes = np.array([len(s) for s in schedules])
    daily_mask = sched_sizes == 6
    batched_mask = sched_sizes < 6

    # For each pp: best daily (size=6) cost and schedule index
    best_daily_idx = np.full(n_pp, -1, dtype=int)
    best_daily_cost = np.full(n_pp, np.inf)
    for ppi in range(n_pp):
        candidates = np.where(daily_mask)[0]
        if len(candidates) == 0: continue
        best_local = candidates[np.argmin(sched_cost[ppi, candidates])]
        best_daily_idx[ppi] = best_local
        best_daily_cost[ppi] = sched_cost[ppi, best_local]

    log("[5/6] Iterating 11x5 = 55 grid cells...")
    rows = []
    for bs in BATCH_SHARES:
        n_batched = int(round(bs * n_pp))
        # Greedy: pick n_batched PLZs with biggest savings from batching
        best_batched_min_cost = np.array([
            sched_cost[ppi, batched_mask].min() if batched_mask.any() else np.inf
            for ppi in range(n_pp)
        ])
        gains = best_daily_cost - best_batched_min_cost
        batched_set = set(np.argsort(-gains)[:n_batched].tolist())
        for penalty in PENALTIES:
            total_cost = 0.0; total_wait = 0.0; n_pkts = 0
            mix = defaultdict(int)
            for ppi in range(n_pp):
                if ppi in batched_set:
                    # Pick best batched under combined cost (cost + penalty × pkts × wait)
                    combined = sched_cost[ppi, :] + penalty * n_pp_arr[ppi] * sched_waits
                    # Restrict to batched
                    combined_b = np.where(batched_mask, combined, np.inf)
                    best_si = int(np.argmin(combined_b))
                else:
                    best_si = best_daily_idx[ppi]
                total_cost += sched_cost[ppi, best_si]
                total_wait += sched_waits[best_si] * n_pp_arr[ppi]
                n_pkts += n_pp_arr[ppi]
                mix[sched_sizes[best_si]] += 1
            avg_wait = total_wait / max(1, n_pkts)
            rows.append({
                "batch_share": bs, "penalty": penalty,
                "total_cost_eur": total_cost,
                "avg_wait_days": avg_wait,
                "n_pkts_weekly": n_pkts,
                **{f"mix_{k}day": mix.get(k, 0) for k in range(2, 7)},
            })
            log(f"      bs={bs:.2f}  pen={penalty:.2f}  cost={total_cost/1e3:6.1f}k€  wait={avg_wait:.3f}d")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tab_grid_kpis.csv", index=False)

    log("[6/6] Identifying Pareto-optimal cells + plotting...")
    df["is_pareto"] = False
    for i, r in df.iterrows():
        dominated = ((df.total_cost_eur < r.total_cost_eur) &
                      (df.avg_wait_days <= r.avg_wait_days)) | \
                     ((df.total_cost_eur <= r.total_cost_eur) &
                      (df.avg_wait_days < r.avg_wait_days))
        df.loc[i, "is_pareto"] = not dominated.any()
    df[df.is_pareto].to_csv(OUT / "tab_pareto_optimal.csv", index=False)

    # Pareto plot
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for pen in PENALTIES:
        sub = df[df.penalty == pen].sort_values("batch_share")
        ax.plot(sub.avg_wait_days, sub.total_cost_eur / 1e3, "o-",
                label=f"penalty={pen:.2f}€/p/d", alpha=0.75)
    pareto = df[df.is_pareto]
    ax.scatter(pareto.avg_wait_days, pareto.total_cost_eur / 1e3,
                marker="*", s=250, c="gold", edgecolors="black", zorder=10,
                label="Pareto-optimal")
    ax.set_xlabel("Avg customer waiting days")
    ax.set_ylabel("Total weekly cost [k€]")
    ax.set_title("Service-Cost Pareto Front (2D Sensitivity Sweep)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pareto_2d.png"); fig.savefig(OUT / "fig_pareto_2d.pdf")
    plt.close(fig)

    # Heatmap: cost
    cost_grid = df.pivot(index="batch_share", columns="penalty", values="total_cost_eur") / 1e3
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cost_grid.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(PENALTIES))); ax.set_xticklabels([f"{p:.2f}" for p in PENALTIES])
    ax.set_yticks(range(len(BATCH_SHARES))); ax.set_yticklabels([f"{b:.1f}" for b in BATCH_SHARES])
    ax.set_xlabel("WAITING_PENALTY €/parcel/day"); ax.set_ylabel("batch_share")
    ax.set_title("Total weekly cost [k€]")
    for i in range(len(BATCH_SHARES)):
        for j in range(len(PENALTIES)):
            v = cost_grid.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v < cost_grid.values.mean() else "black",
                    fontsize=7)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_heatmap_cost.png"); fig.savefig(OUT / "fig_heatmap_cost.pdf")
    plt.close(fig)

    log(f"\nDone in {time.time()-t_start:.0f}s. Output: {OUT}")
    log(f"Pareto cells: {df.is_pareto.sum()}")


if __name__ == "__main__":
    main()
