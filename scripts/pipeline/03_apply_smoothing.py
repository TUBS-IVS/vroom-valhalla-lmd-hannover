"""Post-processing: apply system_smooth_pass on top of the per-hub balanced
schedules in tab_chosen_schedules.csv / tab_balancing_summary.csv. Writes new
columns (schedule_idx_system_smoothed, etc.) and a separate
tab_balancing_summary_smoothed.csv so the original Path-2 output is preserved
verbatim for diff/audit.

Idempotent: re-runs only cells not yet present in the output cache.
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
    build_cost_matrices_ml, system_smooth_pass, _daily_fleet_per_hub,
)

OUT = ROOT / "results" / "overnight_2026_05_29_path2"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


def process_cell(P: float, share: float, pdata: dict, odata: dict, mlp: dict,
                 model, schedules: list, sched_waits: np.ndarray,
                 cell_chosen: pd.DataFrame, cost_budget_pct: float):
    """Run system_smooth_pass on each provider in a cell, return per-PLZ and
    per-cell results."""
    fs_b2c_v = mod.fs_b2c(share); fs_b2b_v = mod.fs_b2b(share)
    plz_rows = []
    summary_rows = []
    sys_fleet_before = np.zeros(mod.N_DAYS)
    sys_fleet_after = np.zeros(mod.N_DAYS)

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
        sub = cell_chosen[cell_chosen.provider == prov].set_index("plz")
        sub.index = sub.index.astype(str)
        chosen_bal = np.array(
            [int(sub.loc[str(pc), "schedule_idx_balanced"]) for pc in plz_keys],
            dtype=np.int64,
        )
        wk = np.array([
            sum(plz_data[pc]["b2c"].values()) + sum(plz_data[pc]["b2b"].values())
            for pc in plz_keys
        ], dtype=np.float64)
        b2cs = m.get("plz_b2c_share", None)
        lw = ((b2cs * (1 - fs_b2c_v) + (1 - b2cs) * (1 - fs_b2b_v))
              if b2cs is not None else np.full(len(plz_keys), share))
        pen_mx = P * lw[:, None] * wk[:, None] * sched_waits[None, :]

        fb = _daily_fleet_per_hub(chosen_bal, plz_hub_arr, hub_plz_list,
                                    m["veh_3d"], schedules).sum(axis=0)
        sys_fleet_before += fb

        res = system_smooth_pass(
            chosen_bal, plz_keys, plz_hub_arr, hub_plz_list, m, schedules,
            cost_budget_pct=cost_budget_pct, max_iterations=400,
            penalty_mx=pen_mx,
        )
        new_chosen = res["chosen"]
        fa = _daily_fleet_per_hub(new_chosen, plz_hub_arr, hub_plz_list,
                                    m["veh_3d"], schedules).sum(axis=0)
        sys_fleet_after += fa

        dd_cost = (m["cost_3d"] * m["sched_active"][None, :, :]).sum(axis=2)
        for pi, pc in enumerate(plz_keys):
            si = int(new_chosen[pi])
            plz_rows.append({
                "penalty": P, "share_willing": share,
                "provider": prov, "plz": str(pc),
                "schedule_idx_system_smoothed": si,
                "schedule_size_system_smoothed": len(schedules[si]),
                "weekdays_system_smoothed": ",".join(
                    WEEKDAYS[d] for d in sorted(schedules[si])),
                "avg_wait_d_system_smoothed": float(sched_waits[si]),
                "dd_cost_system_smoothed": float(dd_cost[pi, si]),
            })
        per_prov_spread_before = float(fb.max() - fb.min())
        per_prov_spread_after = float(fa.max() - fa.min())
        summary_rows.append({
            "penalty": P, "share_willing": share, "provider": prov,
            "system_smooth_swaps": int(res["swaps_made"]),
            "system_smoothed_cost_eur": float(res["cost"]),
            "cost_pre_smoothing": float(res["initial_total_cost"]),
            "provider_spread_before": per_prov_spread_before,
            "provider_spread_after": per_prov_spread_after,
        })

    sys_spread_before = float(sys_fleet_before.max() - sys_fleet_before.min())
    sys_spread_after = float(sys_fleet_after.max() - sys_fleet_after.min())
    return plz_rows, summary_rows, sys_spread_before, sys_spread_after


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1.0,
                    help="additional cost-budget on top of per-hub balanced "
                         "(percent of cost+penalty objective)")
    ap.add_argument("--cells", type=str, default=None,
                    help="comma-separated list of (P,share) to process; "
                         "default = all available")
    args = ap.parse_args()

    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    pdata = chk["provider_data"]; odata = chk4["optimization_data"]
    model = mod.load_model(); mlp = mod.build_ml_prep(pdata)
    schedules = mod.enumerate_schedules()
    sched_waits = np.array([mod.avg_wait_days(sorted(s)) for s in schedules])

    chosen_p = OUT / "tab_chosen_schedules.csv"
    chosen = pd.read_csv(chosen_p); chosen["plz"] = chosen.plz.astype(str)
    cells_all = sorted(set(zip(chosen.penalty.tolist(), chosen.share_willing.tolist())))
    if args.cells:
        wanted = {tuple(map(float, c.split("_")))
                  for c in args.cells.split(",")}
        cells_all = [c for c in cells_all if c in wanted]
    print(f"processing {len(cells_all)} cells with budget=+{args.budget}%")

    plz_out_p = OUT / "_tab_chosen_with_system_smoothing.csv"
    summ_out_p = OUT / "_tab_balancing_summary_with_smoothing.csv"
    sys_out_p = OUT / "_system_spread_per_cell.csv"

    existing_plz_keys = set()
    existing_plz = []
    existing_summ = []
    existing_sys = []
    if plz_out_p.exists():
        existing_plz = pd.read_csv(plz_out_p)
        existing_plz_keys = set(zip(existing_plz.penalty, existing_plz.share_willing))
        existing_plz = existing_plz.to_dict("records")
    if summ_out_p.exists():
        existing_summ = pd.read_csv(summ_out_p).to_dict("records")
    if sys_out_p.exists():
        existing_sys = pd.read_csv(sys_out_p).to_dict("records")

    plz_rows = list(existing_plz)
    summ_rows = list(existing_summ)
    sys_rows = list(existing_sys)
    todo = [(P, s) for (P, s) in cells_all
            if (P, s) not in existing_plz_keys]
    print(f"  {len(existing_plz_keys)} cells already cached, {len(todo)} to do")

    t_total = time.time()
    for i, (P, share) in enumerate(todo, 1):
        t = time.time()
        cell_chosen = chosen[(np.isclose(chosen.penalty, P)) &
                             (np.isclose(chosen.share_willing, share))]
        pr, sr, sb, sa = process_cell(
            P, share, pdata, odata, mlp, model, schedules, sched_waits,
            cell_chosen, args.budget,
        )
        plz_rows.extend(pr); summ_rows.extend(sr)
        sys_rows.append({
            "penalty": P, "share_willing": share,
            "system_spread_before_smoothing": sb,
            "system_spread_after_smoothing": sa,
            "reduction_pct": (100 * (sb - sa) / max(1, sb)),
        })
        # Incremental save (resumable)
        pd.DataFrame(plz_rows).to_csv(plz_out_p, index=False)
        pd.DataFrame(summ_rows).to_csv(summ_out_p, index=False)
        pd.DataFrame(sys_rows).to_csv(sys_out_p, index=False)
        elapsed = time.time() - t_total
        eta = elapsed * (len(todo) - i) / max(1, i) / 60
        print(f"  [{i:3d}/{len(todo)}] P={P:<5g} sh={share:<4g}  "
              f"sys_spread {sb:6.0f}->{sa:6.0f} ({100*(sb-sa)/max(1,sb):4.1f}%)  "
              f"t={time.time()-t:.0f}s  eta={eta:.1f}min", flush=True)

    print(f"\nwrote {plz_out_p} ({len(plz_rows)} rows)")
    print(f"wrote {summ_out_p} ({len(summ_rows)} rows)")
    print(f"wrote {sys_out_p} ({len(sys_rows)} rows)")


if __name__ == "__main__":
    main()
