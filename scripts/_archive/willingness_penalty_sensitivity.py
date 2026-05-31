"""WAITING_PENALTY Sensitivity — wie verschiebt sich Schedule-Wahl bei monetärer Wartezeit-Strafe?

Erweitert die willingness-Analyse um eine Penalty-Dimension:
    cost_objective = ML_cost + PENALTY_EUR_PER_DAY * n_parcels_week * avg_wait_days

Test-Penalties: [0.0, 0.25, 0.5, 1.0, 2.0, 4.0] €/Paket/Wartetag

Per Penalty:
    - Re-optimize schedule pro (provider, PLZ) mit modifizierter Cost-Funktion
    - Track: schedule-size mix, avg_wait, total_cost, % daily-schedules
    - Identify break-even penalty where 6-day schedule wird optimal für die Mehrheit der PLZ

Output:
    results/willingness_penalty/
        figP1_schedule_mix_vs_penalty.{png,pdf}
        figP2_avg_wait_vs_penalty.{png,pdf}
        figP3_pareto_cost_wait.{png,pdf}
        figP4_breakeven_curves.{png,pdf}
        tab_penalty_grid.csv
        REPORT.md
"""
from __future__ import annotations
import argparse, os, pickle, sys, warnings
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

rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

MODE = os.environ.get("WW_MODE", "auto")
CHK_PROD = ROOT / "results" / "checkpoints"
CHK_V2 = ROOT / "results" / "checkpoints" / "archive" / "pre_merge_fix_2026_05_25"
CHK = CHK_V2 if MODE == "v2" else CHK_PROD
V3 = ROOT / "results" / "sweep_v3_mergefix"
V2_RUN = ROOT / "results" / "oracle_loop_extended_2026_05_22"
OUT_NAME = "willingness_penalty_v2" if MODE == "v2" else "willingness_penalty"
OUT = ROOT / "results" / OUT_NAME
OUT.mkdir(parents=True, exist_ok=True)

PROVIDERS = ["Amazon", "DHL", "DPD", "FedEx", "GLS", "Hermes", "UPS"]
DAYS_DE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa"]
SIZE_COLOR = {1: "#3b1f4b", 2: "#1d3557", 3: "#2a9d8f",
                4: "#e9c46a", 5: "#f4a261", 6: "#e76f51"}
N_DAYS = 6
MAX_HOLD = 3   # fix per user
FAST_SHARE = 0.06  # default HAGRID express share — not the variable here

PENALTIES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]  # €/Paket/Wartetag


def enumerate_schedules(max_hold):
    out = []
    min_freq = max(1, int(np.ceil(N_DAYS / max_hold)))
    for k in range(min_freq, N_DAYS + 1):
        for combo in combinations(range(N_DAYS), k):
            days = sorted(combo)
            ok = True
            for i in range(len(days)):
                gap = (days[(i + 1) % len(days)] - days[i]) % N_DAYS
                if gap == 0: gap = N_DAYS
                if gap > max_hold:
                    ok = False; break
            if ok: out.append(frozenset(days))
    return out


def load_state():
    from batch_delivery.surrogate.lgb_adapter import LGBLogTSurrogate
    cost_path = (V2_RUN / "production_lgb_logT_v2.pkl") if MODE == "v2" else (
        V3 / "production_lgb_logT_v3.pkl" if (V3 / "production_lgb_logT_v3.pkl").exists()
        else V2_RUN / "production_lgb_logT_v2.pkl"
    )
    cost_model = LGBLogTSurrogate.load(cost_path)
    optimization_data = pickle.load(open(CHK / "04_optim_prep.pkl", "rb"))["optimization_data"]
    provider_data = pickle.load(open(CHK / "01_demand.pkl", "rb"))["provider_data"]
    print(f"[model] {cost_path.name}")
    return optimization_data, cost_model, provider_data


def build_ml_prep(provider_data):
    from batch_delivery.config.constants import N_DAYS, provider_to_demand_prefix
    ml_prep = {}
    for prov in PROVIDERS:
        pdata = provider_data.get(prov)
        if pdata is None: continue
        df_assign = pdata["df_assignments"]
        hub_coords_by_plz = {row["plz"]: (row["hub_lon"], row["hub_lat"])
                              for _, row in df_assign.iterrows()}
        prefix = provider_to_demand_prefix(prov)
        col_total = f"{prefix}_total"
        plz_day_coords = {}
        for pc in pdata["all_plz_set"]:
            plz_day_coords[pc] = {}
            for d in range(N_DAYS):
                gdf_d = pdata["daily_gdfs_wgs"].get(d)
                if gdf_d is None: continue
                pts = gdf_d[gdf_d["plz"] == pc]
                if len(pts) == 0: continue
                lons = pts["lon"].values.astype(np.float64)
                lats = pts["lat"].values.astype(np.float64)
                psd = (pts[col_total].values.astype(np.float64) if col_total in pts.columns
                        else np.ones(len(pts)))
                plz_day_coords[pc][d] = (lons, lats, psd)
        ml_prep[prov] = {"plz_day_coords": plz_day_coords,
                          "hub_coords_by_plz": hub_coords_by_plz}
    return ml_prep


def solve_with_penalty(penalty: float, optimization_data, ml_prep, cost_model):
    """For each (provider, PLZ), pick min-cost schedule with waiting penalty."""
    from batch_delivery.optimization.core import build_cost_matrices_ml
    schedules = enumerate_schedules(MAX_HOLD)
    sched_sizes = np.array([len(s) for s in schedules])

    total_base_cost = 0.0
    total_penalty_cost = 0.0
    total_wait_weighted = 0.0
    total_parcels = 0
    chosen_rows = []

    for prov in PROVIDERS:
        if prov not in optimization_data or prov not in ml_prep:
            continue
        odata = optimization_data[prov]
        prep = ml_prep[prov]
        plz_keys = odata["plz_keys"]
        plz_data = odata["plz_data"]

        # Use existing fast_share for express handling
        matrices = build_cost_matrices_ml(
            plz_keys, plz_data, schedules, cost_model, prov,
            prep["plz_day_coords"], prep["hub_coords_by_plz"],
            fast_share_b2c=FAST_SHARE, fast_share_b2b=FAST_SHARE,
        )
        cost_3d = matrices["cost_3d"]        # (n_plz, n_sched, n_days)
        wait_mx = matrices["wait_mx"]         # (n_plz, n_sched) avg wait per parcel

        # Total weekly cost per (plz, sched)
        base_cost = cost_3d.sum(axis=2)       # (n_plz, n_sched)

        # Weekly parcels per PLZ (from plz_data)
        plz_weekly_parcels = np.array([
            sum(plz_data.get(pc, {}).get("b2c", {}).values()) +
            sum(plz_data.get(pc, {}).get("b2b", {}).values())
            for pc in plz_keys
        ])

        # Penalty cost per (plz, sched) = penalty € × n_parcels × wait_days
        # Note: wait_mx is in DAYS, plz_weekly_parcels is integer
        penalty_mx = penalty * plz_weekly_parcels[:, None] * wait_mx

        # Combined objective
        combined = base_cost + penalty_mx
        chosen_idx = combined.argmin(axis=1)

        for pi, sidx in enumerate(chosen_idx):
            sz = int(sched_sizes[sidx])
            bc = float(base_cost[pi, sidx])
            wt = float(wait_mx[pi, sidx])
            pc = float(penalty * plz_weekly_parcels[pi] * wt)
            total_base_cost += bc
            total_penalty_cost += pc
            total_wait_weighted += plz_weekly_parcels[pi] * wt
            total_parcels += plz_weekly_parcels[pi]
            chosen_rows.append({
                "penalty_eur_per_day": penalty,
                "provider": prov, "plz": plz_keys[pi],
                "schedule_size": sz,
                "base_cost_eur": bc,
                "wait_days_avg": wt,
                "weekly_parcels": int(plz_weekly_parcels[pi]),
                "penalty_cost_eur": pc,
                "combined_cost_eur": bc + pc,
            })

    avg_wait = total_wait_weighted / max(1, total_parcels)
    return {
        "penalty_eur_per_day": penalty,
        "total_base_cost_eur": total_base_cost,
        "total_penalty_cost_eur": total_penalty_cost,
        "total_parcels_per_week": total_parcels,
        "avg_wait_days": avg_wait,
        "n_plz": len(chosen_rows),
        "size_mix": dict(pd.Series([r["schedule_size"] for r in chosen_rows]).value_counts()),
        "chosen_rows": chosen_rows,
    }


def main():
    print("=" * 60)
    print("WAITING_PENALTY Sensitivity Analysis")
    print("=" * 60)
    optimization_data, cost_model, provider_data = load_state()
    ml_prep = build_ml_prep(provider_data)

    grid_rows = []
    all_chosen = []
    for pen in PENALTIES:
        res = solve_with_penalty(pen, optimization_data, ml_prep, cost_model)
        print(f"penalty={pen:5.2f}€/d  | cost {res['total_base_cost_eur']/1e3:6.1f}k€  "
                f"penalty_cost {res['total_penalty_cost_eur']/1e3:6.1f}k€  "
                f"avg_wait {res['avg_wait_days']:.3f}d  "
                f"size_mix {res['size_mix']}")
        all_chosen.extend(res["chosen_rows"])
        row = {k: v for k, v in res.items() if k not in ("size_mix", "chosen_rows")}
        for sz in [2, 3, 4, 5, 6]:
            row[f"n_size_{sz}"] = res["size_mix"].get(sz, 0)
        grid_rows.append(row)

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUT / "tab_penalty_grid.csv", index=False)
    pd.DataFrame(all_chosen).to_csv(OUT / "tab_penalty_chosen.csv", index=False)

    # ── figP1: schedule-size mix vs penalty
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sizes = [2, 3, 4, 5, 6]
    bot = np.zeros(len(grid))
    for sz in sizes:
        col = f"n_size_{sz}"
        if col not in grid: continue
        vals = grid[col].values
        total = grid[[f"n_size_{s}" for s in sizes if f"n_size_{s}" in grid]].sum(axis=1).clip(lower=1)
        pct = vals / total * 100
        ax.fill_between(grid["penalty_eur_per_day"], bot, bot + pct,
                          color=SIZE_COLOR[sz], alpha=0.85, label=f"{sz} Tage")
        bot = bot + pct
    ax.set_xlabel("Wartezeit-Penalty [€ / Paket / Tag]")
    ax.set_ylabel("Schedule-Größe Mix [%]")
    ax.set_ylim(0, 100)
    ax.set_title("Schedule-Wahl-Shift mit Wartezeit-Penalty")
    ax.legend(loc="center right", ncol=1, frameon=True, bbox_to_anchor=(1.18, 0.5))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figP1_schedule_mix_vs_penalty.png")
    fig.savefig(OUT / "figP1_schedule_mix_vs_penalty.pdf")
    plt.close(fig)

    # ── figP2: avg_wait vs penalty
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(grid["penalty_eur_per_day"], grid["avg_wait_days"],
             "o-", color="#9d0208", markersize=6)
    for _, r in grid.iterrows():
        ax.annotate(f"{r['avg_wait_days']:.2f}d",
                      xy=(r["penalty_eur_per_day"], r["avg_wait_days"]),
                      xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Wartezeit-Penalty [€ / Paket / Tag]")
    ax.set_ylabel("Avg waiting days")
    ax.set_title("Wartezeit-Reduktion durch Penalty")
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "figP2_avg_wait_vs_penalty.png")
    fig.savefig(OUT / "figP2_avg_wait_vs_penalty.pdf")
    plt.close(fig)

    # ── figP3: Pareto cost vs wait (base + penalty visualisation)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid["avg_wait_days"], grid["total_base_cost_eur"] / 1e3,
             "o-", color="#003049", markersize=8)
    for _, r in grid.iterrows():
        ax.annotate(f"€{r['penalty_eur_per_day']}/d",
                      xy=(r["avg_wait_days"], r["total_base_cost_eur"]/1e3),
                      xytext=(5, 5), textcoords="offset points", fontsize=8,
                      color="#666")
    ax.set_xlabel("Avg waiting days (achieved)")
    ax.set_ylabel("Total weekly base cost [thousand €]")
    ax.set_title("Cost-vs-Service Pareto under varying penalty")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figP3_pareto_cost_wait.png")
    fig.savefig(OUT / "figP3_pareto_cost_wait.pdf")
    plt.close(fig)

    # ── figP4: % daily-schedule + break-even line
    fig, ax = plt.subplots(figsize=(6, 4))
    total = grid[[f"n_size_{s}" for s in sizes if f"n_size_{s}" in grid]].sum(axis=1).clip(lower=1)
    pct_6d = grid["n_size_6"] / total * 100
    pct_2d = grid["n_size_2"] / total * 100
    ax.plot(grid["penalty_eur_per_day"], pct_6d, "o-", color=SIZE_COLOR[6],
             label="6-Tage-Schedule (täglich)", markersize=6)
    ax.plot(grid["penalty_eur_per_day"], pct_2d, "s-", color=SIZE_COLOR[2],
             label="2-Tage-Schedule (max Batching)", markersize=6)
    # Find break-even
    cross_idx = np.where(np.diff(np.sign(pct_6d.values - 50)))[0]
    if len(cross_idx) > 0:
        ax.axhline(50, color="black", linestyle=":", alpha=0.5)
        x_break = grid["penalty_eur_per_day"].iloc[cross_idx[0] + 1]
        ax.axvline(x_break, color="red", linestyle="--", alpha=0.7)
        ax.annotate(f"Break-even: 50% picken 6-day\n@ €{x_break:.2f}/parcel/day",
                      xy=(x_break, 50), xytext=(15, -30), textcoords="offset points",
                      fontsize=9, color="red",
                      arrowprops=dict(arrowstyle="->", color="red"))
    ax.set_xlabel("Wartezeit-Penalty [€ / Paket / Tag]")
    ax.set_ylabel("% PLZ-Cluster picking schedule")
    ax.set_title("Break-even — when does daily delivery win?")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figP4_breakeven_curves.png")
    fig.savefig(OUT / "figP4_breakeven_curves.pdf")
    plt.close(fig)
    print(f"\n[done] all figures in {OUT}")

    # REPORT
    lines = [
        "# Waiting-Penalty Sensitivity Report",
        f"\n**Mode**: {MODE} | **Output**: {OUT.name}",
        f"**Method**: cost_objective = base_cost + PENALTY * n_parcels * avg_wait_days",
        f"**Schedules tested**: 39 (MAX_HOLDING_DAYS=3)",
        f"**Penalty grid**: {PENALTIES} €/parcel/day",
        f"\n## Grid\n",
        grid.to_string(index=False),
    ]
    cross_idx = np.where(np.diff(np.sign(grid["n_size_6"] / total * 100 - 50).values))[0]
    if len(cross_idx) > 0:
        x_break = grid["penalty_eur_per_day"].iloc[cross_idx[0] + 1]
        lines.append(f"\n## Break-even: 50% wechseln zu daily delivery bei **€{x_break:.2f}/parcel/day**")
    else:
        lines.append(f"\n## Break-even: außerhalb des Test-Grids — Daily-Delivery wird nicht majoritär gewählt selbst bei €{PENALTIES[-1]}/parcel/day")
    lines.append(f"\n## Figures\n")
    for fp in sorted(OUT.glob("figP*.png")):
        lines.append(f"- `{fp.name}`")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"REPORT.md geschrieben")


if __name__ == "__main__":
    main()
