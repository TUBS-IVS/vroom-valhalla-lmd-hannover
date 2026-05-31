"""Comprehensive comparison: final Path-2 solution (CD + per-hub balance +
system smoothing) vs the two naive baselines (Random schedule, per-PLZ Argmin
on unbundled proxy), on the SAME hub-bundled cost basis, with both
COST and FLEET metrics.

For each test cell we compute, on the system aggregate (sum across providers):
  - Bundled cost saving vs daily baseline [%]
  - Peak vehicles (max over Mo-Sa days)
  - Mo-Sa fleet spread (max - min)
  - Total weekly fleet (sum over days)
  - Parcels-weighted mean wait [d]

Outputs:
  results/overnight_2026_05_29_path2/_final_vs_baselines.csv
  results/overnight_2026_05_29_path2/_final_vs_baselines.png
"""
from __future__ import annotations
import logging, pickle, sys, time
from pathlib import Path

logging.disable(logging.CRITICAL)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util as _iu
_spec = _iu.spec_from_file_location("ob", ROOT / "scripts" / "overnight_orchestrator_balanced.py")
mod = _iu.module_from_spec(_spec); _spec.loader.exec_module(mod)
logging.disable(logging.CRITICAL)

from batch_delivery.optimization.core import (  # noqa: E402
    balance_fleet_per_hub_ml, build_cost_matrices_ml, _daily_fleet_per_hub,
)

OUT = ROOT / "results" / "overnight_2026_05_29_path2"
BASE = 1909747.75

rcParams.update({
    "font.family": "serif", "font.size": 9,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 8.5, "axes.titlesize": 9.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "savefig.bbox": "tight", "savefig.dpi": 180, "pdf.fonttype": 42,
})

TEST_CELLS = [
    (0.0, 0.1),   # low P, low theta: bundling synergies dominant
    (0.0, 0.5),   # low P, mid theta
    (0.0, 1.0),   # low P, max theta: no express, original anomaly
    (0.5, 0.5),   # sweet-spot-ish
    (0.75, 1.0),  # higher P, max theta
]


def bundled_cost_zero_swaps(chosen, plz_keys, plz_hub_arr, hub_plz_list,
                             matrices, schedules):
    bal = balance_fleet_per_hub_ml(
        {"chosen": chosen.astype(np.int64), "best_cost": 0.0},
        plz_keys, plz_hub_arr, hub_plz_list,
        matrices, schedules,
        cost_budget_pct=5.0, max_swaps=0,
    )
    return float(bal["initial_total_cost"])


def fleet_summary(chosen, plz_hub_arr, hub_plz_list, veh_3d, schedules):
    """Returns (peak, spread, total_weekly, fleet_by_day)."""
    fleet_hub_day = _daily_fleet_per_hub(chosen, plz_hub_arr, hub_plz_list,
                                          veh_3d, schedules)
    sys_by_day = fleet_hub_day.sum(axis=0)
    return (float(sys_by_day.max()),
            float(sys_by_day.max() - sys_by_day.min()),
            float(sys_by_day.sum()),
            sys_by_day.astype(int).tolist())


def main() -> None:
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    pdata = chk["provider_data"]; odata = chk4["optimization_data"]
    model = mod.load_model(); mlp = mod.build_ml_prep(pdata)
    schedules = mod.enumerate_schedules()
    sched_waits = np.array([mod.avg_wait_days(sorted(s)) for s in schedules])
    sched_sizes = np.array([len(s) for s in schedules], dtype=np.float64)

    chosen_df = pd.read_csv(OUT / "tab_chosen_schedules.csv")
    chosen_df["plz"] = chosen_df.plz.astype(str)
    sm_path = OUT / "_tab_chosen_with_system_smoothing.csv"
    sm_df = (pd.read_csv(sm_path) if sm_path.exists() else None)
    if sm_df is not None:
        sm_df["plz"] = sm_df.plz.astype(str)

    rng = np.random.default_rng(20260530)
    rows = []
    for P, sh in TEST_CELLS:
        cell = chosen_df[(np.isclose(chosen_df.penalty, P)) &
                         (np.isclose(chosen_df.share_willing, sh))]
        if len(cell) == 0:
            print(f"  SKIP P={P} th={sh}, not yet in chosen CSV")
            continue
        sm_cell = (sm_df[(np.isclose(sm_df.penalty, P)) &
                         (np.isclose(sm_df.share_willing, sh))]
                   if sm_df is not None else None)
        has_smoothed = sm_cell is not None and len(sm_cell) > 0

        t = time.time()
        fs_b2c_v = mod.fs_b2c(sh); fs_b2b_v = mod.fs_b2b(sh)

        # Accumulators
        agg = {bl: {"cost": 0.0, "peak": 0.0, "weekly": 0.0,
                     "spread_sysday": np.zeros(mod.N_DAYS),
                     "wait_weighted_num": 0.0, "wait_weighted_den": 0.0}
               for bl in ("baseline", "random", "argmin", "final")}
        daily_si = next(i for i, s in enumerate(schedules)
                        if len(s) == mod.N_DAYS)

        for prov in mod.PROVIDERS:
            if prov not in odata or prov not in mlp:
                continue
            od = odata[prov]; prep = mlp[prov]
            plz_keys = od["plz_keys"]; plz_data = od["plz_data"]
            plz_hub_arr = od["plz_hub_arr"]; hub_plz_list = od["hub_plz_list"]
            m = build_cost_matrices_ml(
                plz_keys, plz_data, schedules, model, prov,
                prep["plz_day_coords"], prep["hub_coords_by_plz"],
                fast_share_b2c=fs_b2c_v, fast_share_b2b=fs_b2b_v,
            )
            total_cost_mx = m["cost_3d"].sum(axis=2)
            wk = np.array([
                sum(plz_data[pc]["b2c"].values()) + sum(plz_data[pc]["b2b"].values())
                for pc in plz_keys
            ], dtype=np.float64)
            b2cs = m.get("plz_b2c_share", None)
            lw = ((b2cs * (1 - fs_b2c_v) + (1 - b2cs) * (1 - fs_b2b_v))
                  if b2cs is not None else np.full(len(plz_keys), sh))
            obj = total_cost_mx + P * lw[:, None] * wk[:, None] * sched_waits[None, :]

            # RANDOM
            chosen_rnd = rng.integers(0, len(schedules), size=len(plz_keys),
                                       dtype=np.int64)
            # ARGMIN
            obj_min = obj.min(axis=1, keepdims=True)
            near_tied = obj <= obj_min * 1.005
            score = np.where(near_tied, sched_sizes[None, :], -np.inf)
            chosen_arg = score.argmax(axis=1).astype(np.int64)
            if sh == 0.0:
                daily_si = next(i for i, s in enumerate(schedules)
                                if len(s) == mod.N_DAYS)
                chosen_arg = np.full(len(plz_keys), daily_si, dtype=np.int64)
            # FINAL = system-smoothed if present, else balanced (post-hub)
            sub_p = cell[cell.provider == prov].set_index("plz")
            sub_p.index = sub_p.index.astype(str)
            if has_smoothed:
                sub_s = sm_cell[sm_cell.provider == prov].set_index("plz")
                sub_s.index = sub_s.index.astype(str)
                chosen_fin = np.array(
                    [int(sub_s.loc[str(pc), "schedule_idx_system_smoothed"])
                     for pc in plz_keys], dtype=np.int64,
                )
            else:
                chosen_fin = np.array(
                    [int(sub_p.loc[str(pc), "schedule_idx_balanced"])
                     for pc in plz_keys], dtype=np.int64,
                )

            chosen_base = np.full(len(plz_keys), daily_si, dtype=np.int64)
            for name, chosen_arr in (("baseline", chosen_base),
                                       ("random", chosen_rnd),
                                       ("argmin", chosen_arg),
                                       ("final", chosen_fin)):
                cost = bundled_cost_zero_swaps(
                    chosen_arr, plz_keys, plz_hub_arr, hub_plz_list, m, schedules)
                peak, spread, weekly, by_day = fleet_summary(
                    chosen_arr, plz_hub_arr, hub_plz_list, m["veh_3d"], schedules)
                agg[name]["cost"] += cost
                agg[name]["spread_sysday"] += by_day  # accumulate provider days
                wait = float((sched_waits[chosen_arr] * wk).sum())
                agg[name]["wait_weighted_num"] += wait
                agg[name]["wait_weighted_den"] += float(wk.sum())

        for name in ("baseline", "random", "argmin", "final"):
            sys_by_day = agg[name]["spread_sysday"]
            agg[name]["peak"] = float(sys_by_day.max())
            agg[name]["weekly"] = float(sys_by_day.sum())
            agg[name]["spread"] = float(sys_by_day.max() - sys_by_day.min())
            agg[name]["wait"] = (agg[name]["wait_weighted_num"]
                                  / max(1, agg[name]["wait_weighted_den"]))

        row = {"penalty": P, "share_willing": sh}
        for name in ("baseline", "random", "argmin", "final"):
            sav = 100 * (BASE - agg[name]["cost"]) / BASE
            row[f"{name}_sav_pct"] = sav
            row[f"{name}_peak"] = agg[name]["peak"]
            row[f"{name}_spread"] = agg[name]["spread"]
            row[f"{name}_weekly"] = agg[name]["weekly"]
            row[f"{name}_wait"] = agg[name]["wait"]
        rows.append(row)
        print(f"  P={P:<5g} sh={sh:<4g}  "
              f"base[sav={row['baseline_sav_pct']:5.1f}% peak={row['baseline_peak']:4.0f} spread={row['baseline_spread']:4.0f}]  "
              f"random[sav={row['random_sav_pct']:5.1f}% peak={row['random_peak']:4.0f} spread={row['random_spread']:4.0f}]  "
              f"argmin[sav={row['argmin_sav_pct']:5.1f}% peak={row['argmin_peak']:4.0f} spread={row['argmin_spread']:4.0f}]  "
              f"final[sav={row['final_sav_pct']:5.1f}% peak={row['final_peak']:4.0f} spread={row['final_spread']:4.0f}]  "
              f"t={time.time()-t:.0f}s",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "_final_vs_baselines.csv", index=False)

    # 4-panel comparison bar chart
    metrics = [
        ("sav_pct",  "Cost saving [%]",          "{:.1f}",  None),
        ("peak",     "Peak Mo-Sa vehicles",      "{:.0f}",  None),
        ("spread",   "Mo-Sa fleet spread",       "{:.0f}",  None),
        ("wait",     r"Mean wait [d]",            "{:.2f}",  None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    x = np.arange(len(df)); w = 0.21
    cols = {"baseline": "#1d3557", "random": "#888",
            "argmin": "#f4a261", "final": "#2a9d8f"}
    labels = {"baseline": "Baseline (daily)",
              "random": "Random",
              "argmin": "Per-PLZ Argmin",
              "final": "Final (Pfad 2 + Smooth)"}
    names = ("baseline", "random", "argmin", "final")

    for ax, (key, label, fmt, _) in zip(axes, metrics):
        for i, name in enumerate(names):
            vals = df[f"{name}_{key}"].values
            ax.bar(x + (i - 1.5) * w, vals, w, color=cols[name],
                   edgecolor="black", linewidth=0.4,
                   label=labels[name])
            for j, v in enumerate(vals):
                ax.text(j + (i - 1.5) * w, v + 0.02 * max(vals),
                        fmt.format(v), ha="center", fontsize=6, rotation=0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"P={r.penalty:g}\n$\\theta$={r.share_willing:g}"
                            for _, r in df.iterrows()], fontsize=7)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    axes[0].axhline(0, color="black", lw=0.4)
    fig.suptitle("Final Pfad-2-Lösung vs. naive Baselines — Kosten und Flotte",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "_final_vs_baselines.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {OUT/'_final_vs_baselines.csv'}")
    print(f"saved {OUT/'_final_vs_baselines.png'}")


if __name__ == "__main__":
    main()
