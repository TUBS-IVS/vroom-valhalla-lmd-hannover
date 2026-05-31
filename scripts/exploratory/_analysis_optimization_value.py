"""For a handful of representative (P, theta) cells, compare three baselines
on the same hub-bundled cost basis:

  - RANDOM: each PLZ gets a uniformly random schedule (rng.integers).
  - ARGMIN: per-PLZ separable argmin on the OLD unbundled objective
            (total_cost_mx + P*penalty), i.e. what Pfad 1 used.
  - CD_ML:  the Pfad 2 output (chosen_init from tab_chosen_schedules.csv).

Reports the bundled cost saving vs daily baseline for each baseline, so the
reader can see how much the bundling-aware coordinate descent contributes on
top of the naive argmin start (and vs an uninformed random start).

Outputs (under results/overnight_2026_05_29_path2/):
    _opt_value.csv         per-cell, per-baseline saving %
    _opt_value.png         grouped bar chart
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util as _iu
_spec = _iu.spec_from_file_location("ob", ROOT / "scripts" / "overnight_orchestrator_balanced.py")
mod = _iu.module_from_spec(_spec); _spec.loader.exec_module(mod)
logging.disable(logging.CRITICAL)

from batch_delivery.optimization.core import (  # noqa: E402
    balance_fleet_per_hub_ml, build_cost_matrices_ml,
)

OUT = ROOT / "results" / "overnight_2026_05_29_path2"
BASE = 1909747.75
TEST_CELLS = [
    (0.0, 0.1),   # low P, low theta : where bundling matters most
    (0.0, 0.5),   # low P, mid theta
    (0.0, 1.0),   # low P, max theta : no express, no bundling
    (0.5, 0.5),   # mid P, mid theta : balanced regime
    (0.75, 1.0),  # higher P : penalty constrains optimization
]


def bundled_cost(chosen, plz_keys, plz_hub_arr, hub_plz_list, matrices, schedules):
    bal = balance_fleet_per_hub_ml(
        {"chosen": chosen.astype(np.int64), "best_cost": 0.0},
        plz_keys, plz_hub_arr, hub_plz_list,
        matrices, schedules,
        cost_budget_pct=5.0, max_swaps=0,
    )
    return float(bal["initial_total_cost"])


def main() -> None:
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    pdata = chk["provider_data"]; odata = chk4["optimization_data"]
    model = mod.load_model(); mlp = mod.build_ml_prep(pdata)
    schedules = mod.enumerate_schedules()
    sched_waits = np.array([mod.avg_wait_days(sorted(s)) for s in schedules])
    sched_sizes = np.array([len(s) for s in schedules], dtype=np.float64)

    cd_df = pd.read_csv(OUT / "tab_chosen_schedules.csv")
    cd_df["plz"] = cd_df.plz.astype(str)

    rng = np.random.default_rng(20260530)
    rows = []

    for P, sh in TEST_CELLS:
        cell = cd_df[(np.isclose(cd_df.penalty, P)) &
                     (np.isclose(cd_df.share_willing, sh))]
        if len(cell) == 0:
            print(f"  SKIP ({P},{sh}) — not yet in chosen CSV")
            continue
        t = time.time()
        fs_b2c_v = mod.fs_b2c(sh); fs_b2b_v = mod.fs_b2b(sh)
        per_prov_random = 0.0
        per_prov_argmin = 0.0
        per_prov_cd = 0.0
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
            per_prov_random += bundled_cost(chosen_rnd, plz_keys, plz_hub_arr,
                                             hub_plz_list, m, schedules)

            # ARGMIN (per-PLZ on unbundled obj, with tie-breaker as orchestrator)
            obj_min = obj.min(axis=1, keepdims=True)
            near_tied = obj <= obj_min * 1.005
            score = np.where(near_tied, sched_sizes[None, :], -np.inf)
            chosen_arg = score.argmax(axis=1).astype(np.int64)
            if sh == 0.0:
                daily_si = next(i for i, s in enumerate(schedules)
                                if len(s) == mod.N_DAYS)
                chosen_arg = np.full(len(plz_keys), daily_si, dtype=np.int64)
            per_prov_argmin += bundled_cost(chosen_arg, plz_keys, plz_hub_arr,
                                             hub_plz_list, m, schedules)

            # CD_ML (Pfad 2)
            sub = cell[cell.provider == prov].set_index("plz")
            sub.index = sub.index.astype(str)
            chosen_cd = np.array(
                [int(sub.loc[str(pc), "schedule_idx_init"]) for pc in plz_keys],
                dtype=np.int64,
            )
            per_prov_cd += bundled_cost(chosen_cd, plz_keys, plz_hub_arr,
                                         hub_plz_list, m, schedules)

        sav_rnd = 100 * (BASE - per_prov_random) / BASE
        sav_arg = 100 * (BASE - per_prov_argmin) / BASE
        sav_cd = 100 * (BASE - per_prov_cd) / BASE
        rows.append({
            "penalty": P, "share_willing": sh,
            "random_sav": sav_rnd,
            "argmin_sav": sav_arg,
            "cd_sav": sav_cd,
            "gain_argmin_over_random": sav_arg - sav_rnd,
            "gain_cd_over_argmin": sav_cd - sav_arg,
        })
        print(f"  P={P} sh={sh}  random={sav_rnd:+.1f}%  argmin={sav_arg:+.1f}%  "
              f"CD={sav_cd:+.1f}%  (CD gain over argmin: {sav_cd-sav_arg:+.2f}pp)  "
              f"t={time.time()-t:.0f}s",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "_opt_value.csv", index=False)

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df))
    w = 0.27
    ax.bar(x - w, df.random_sav, w, color="#888", label="Random schedule",
           edgecolor="black", linewidth=0.4)
    ax.bar(x, df.argmin_sav, w, color="#f4a261", label="Per-PLZ argmin (unbundled proxy)",
           edgecolor="black", linewidth=0.4)
    ax.bar(x + w, df.cd_sav, w, color="#2a9d8f",
           label="CD_ML on bundled+penalty (Path 2)", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P={r.penalty:g}\n$\\theta$={r.share_willing:g}"
                        for _, r in df.iterrows()])
    ax.set_ylabel("Bundled cost saving vs daily baseline [%]")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_title("Wert der gebündelten Optimierung — Random vs Per-PLZ-Argmin vs CD_ML",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, r in df.iterrows():
        ax.text(i + w, r.cd_sav + 0.5, f"+{r.gain_cd_over_argmin:.1f}",
                ha="center", color="#2a9d8f", fontsize=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "_opt_value.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"\nsaved {OUT/'_opt_value.csv'}")
    print(f"saved {OUT/'_opt_value.png'}")


if __name__ == "__main__":
    main()
