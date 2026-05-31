"""Path 2 smoke: run optimize_cd_ml on one (P, share) cell with the penalty
term baked into dd_cost_mx, compare the resulting bundled cost and schedule mix
to (a) the per-PLZ argmin (current init, from backup CSV) and (b) the bundled
cost of that argmin selection. Writes a JSON report; touches nothing else.
"""
from __future__ import annotations
import argparse, json, logging, pickle, sys, time
from pathlib import Path

logging.disable(logging.CRITICAL)
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
    balance_fleet_per_hub_ml, build_cost_matrices_ml, optimize_cd_ml,
)

OUT = ROOT / "results" / "overnight_2026_05_27_balanced"


def run_cell(P: float, share: float) -> dict:
    fs_b2c_v = mod.fs_b2c(share); fs_b2b_v = mod.fs_b2b(share)
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    pdata = chk["provider_data"]; odata = chk4["optimization_data"]
    model = mod.load_model(); mlp = mod.build_ml_prep(pdata)
    schedules = mod.enumerate_schedules()
    sched_waits = np.array([mod.avg_wait_days(sorted(s)) for s in schedules])
    sched_sizes = np.array([len(s) for s in schedules])

    chosen = pd.read_csv(OUT / "tab_chosen_schedules.csv")
    chosen["plz"] = chosen.plz.astype(str)
    cell = chosen[(np.isclose(chosen.penalty, P)) &
                  (np.isclose(chosen.share_willing, share))]

    out = {"penalty": P, "share": share, "providers": []}
    tot_p1_bun = 0.0; tot_p2_bun = 0.0
    sizes_p1: list[int] = []; sizes_p2: list[int] = []
    waits_p1: list[float] = []; waits_p2: list[float] = []
    pkts_all: list[int] = []
    t0_cell = time.time()

    for prov in mod.PROVIDERS:
        if prov not in odata or prov not in mlp: continue
        od = odata[prov]; prep = mlp[prov]
        plz_keys = od["plz_keys"]; plz_data = od["plz_data"]
        plz_hub_arr = od["plz_hub_arr"]; hub_plz_list = od["hub_plz_list"]

        m = build_cost_matrices_ml(
            plz_keys, plz_data, schedules, model, prov,
            prep["plz_day_coords"], prep["hub_coords_by_plz"],
            fast_share_b2c=fs_b2c_v, fast_share_b2b=fs_b2b_v,
        )
        dd_orig = m["dd_cost_mx"].copy()
        wk = np.array([
            sum(plz_data[pc]["b2c"].values()) + sum(plz_data[pc]["b2b"].values())
            for pc in plz_keys
        ], dtype=np.float64)
        b2cs = m.get("plz_b2c_share", None)
        lw = ((b2cs * (1 - fs_b2c_v) + (1 - b2cs) * (1 - fs_b2b_v))
              if b2cs is not None else np.full(len(plz_keys), share))
        pen_mx = P * lw[:, None] * wk[:, None] * sched_waits[None, :]

        # ── Path 1 init (per-PLZ argmin) from existing CSV
        sub = cell[cell.provider == prov].set_index("plz")
        sub.index = sub.index.astype(str)
        chosen_p1 = np.array(
            [int(sub.loc[str(pc), "schedule_idx_init"]) for pc in plz_keys],
            dtype=np.int64,
        )
        # bundled cost of path-1 selection
        bal0 = balance_fleet_per_hub_ml(
            {"chosen": chosen_p1, "best_cost": 0.0},
            plz_keys, plz_hub_arr, hub_plz_list, m, schedules,
            cost_budget_pct=5.0, max_swaps=0,
        )
        p1_bundled = float(bal0["initial_total_cost"])

        # ── Path 2: bake penalty into dd_cost_mx, run CD on bundled+penalty
        m_pen = dict(m); m_pen["dd_cost_mx"] = dd_orig + pen_mx
        t_p = time.time()
        cd = optimize_cd_ml(
            plz_keys, plz_hub_arr, hub_plz_list, m_pen, schedules,
            max_rounds=15, shuffle_plz=True, seed=1234,
            pair_polish=True, pair_polish_rounds=2, pair_polish_max_pairs=200,
        )
        t_p2 = time.time() - t_p
        chosen_p2 = cd["chosen"]
        # Strip penalty back out to get pure bundled routing cost
        bal2 = balance_fleet_per_hub_ml(
            {"chosen": chosen_p2, "best_cost": 0.0},
            plz_keys, plz_hub_arr, hub_plz_list, m, schedules,
            cost_budget_pct=5.0, max_swaps=0,
        )
        p2_bundled = float(bal2["initial_total_cost"])
        n_changed = int((chosen_p1 != chosen_p2).sum())
        tot_p1_bun += p1_bundled; tot_p2_bun += p2_bundled
        sizes_p1 += sched_sizes[chosen_p1].tolist()
        sizes_p2 += sched_sizes[chosen_p2].tolist()
        waits_p1 += sched_waits[chosen_p1].tolist()
        waits_p2 += sched_waits[chosen_p2].tolist()
        pkts_all += wk.astype(int).tolist()
        out["providers"].append({
            "provider": prov, "n_plz": len(plz_keys),
            "p1_bundled": p1_bundled, "p2_bundled": p2_bundled,
            "gain": p1_bundled - p2_bundled,
            "n_changed": n_changed, "cd_seconds": round(t_p2, 1),
        })
        print(f"  {prov:>6}: p1={p1_bundled/1e3:7.1f}k  p2={p2_bundled/1e3:7.1f}k "
              f"(gain={p1_bundled-p2_bundled:+.0f})  changed={n_changed}/{len(plz_keys)}  "
              f"CD={t_p2:.1f}s", flush=True)

    base = 1909747.75
    p = np.array(pkts_all, float)
    out["totals"] = {
        "p1_sum_bundled": tot_p1_bun, "p2_sum_bundled": tot_p2_bun,
        "p1_sav_pct": 100 * (base - tot_p1_bun) / base,
        "p2_sav_pct": 100 * (base - tot_p2_bun) / base,
        "p1_mean_size": float(np.mean(sizes_p1)),
        "p2_mean_size": float(np.mean(sizes_p2)),
        "p1_wait_pkts_weighted": float((np.array(waits_p1) * p).sum() / p.sum()),
        "p2_wait_pkts_weighted": float((np.array(waits_p2) * p).sum() / p.sum()),
        "total_seconds": round(time.time() - t0_cell, 1),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=float, required=True)
    ap.add_argument("--share", type=float, required=True)
    args = ap.parse_args()
    res = run_cell(args.P, args.share)
    print("\n=== TOTALS ===")
    for k, v in res["totals"].items():
        print(f"  {k}: {v}")
    out_path = OUT / f"_path2_smoke_P{args.P}_sh{args.share}.json"
    json.dump(res, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
