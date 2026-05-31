"""Recompute init_cost_eur as the BUNDLED cost (dd + hub-bundled express) of
the existing chosen_init schedules, replacing the per-PLZ UNBUNDLED total used
previously. Reads tab_chosen_schedules.csv (already restored from the backup,
penalty-blind balancing semantics) and overwrites the provider-level
init_cost_eur in tab_balancing_summary.csv. Schedules do not change. This
makes init and balanced report on the same (bundled) cost basis, removing the
apples-to-oranges gap in the saving heatmap.

If a (P, share) tuple is given via --only, processes only that cell (smoke).
"""
from __future__ import annotations
import argparse, logging, pickle, sys, time
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

from batch_delivery.optimization.core import balance_fleet_per_hub_ml, build_cost_matrices_ml  # noqa: E402

OUT = ROOT / "results" / "overnight_2026_05_27_balanced"


def _provider_init_bundled(prov: str, fs_b2c_v: float, fs_b2b_v: float,
                            cell_df: pd.DataFrame, odata: dict, prep: dict,
                            schedules, model) -> float:
    """Return the bundled cost (dd + hub-bundled express) of the chosen_init
    schedules for this provider, via balance_fleet_per_hub_ml(max_swaps=0)."""
    plz_keys = odata["plz_keys"]; plz_data = odata["plz_data"]
    plz_hub_arr = odata["plz_hub_arr"]; hub_plz_list = odata["hub_plz_list"]

    sub = cell_df[cell_df.provider == prov].set_index("plz")
    sub.index = sub.index.astype(str)
    chosen_init = np.array(
        [int(sub.loc[str(pc), "schedule_idx_init"]) for pc in plz_keys],
        dtype=np.int64,
    )

    matrices = build_cost_matrices_ml(
        plz_keys, plz_data, schedules, model, prov,
        prep["plz_day_coords"], prep["hub_coords_by_plz"],
        fast_share_b2c=fs_b2c_v, fast_share_b2b=fs_b2b_v,
    )
    bal = balance_fleet_per_hub_ml(
        {"chosen": chosen_init, "best_cost": 0.0},
        plz_keys, plz_hub_arr, hub_plz_list,
        matrices, schedules,
        cost_budget_pct=5.0, max_swaps=0,
    )
    return float(bal["initial_total_cost"])


def main(only: tuple[float, float] | None) -> None:
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    pdata = chk["provider_data"]; odata = chk4["optimization_data"]
    model = mod.load_model(); mlp = mod.build_ml_prep(pdata)
    schedules = mod.enumerate_schedules()

    summary = pd.read_csv(OUT / "tab_balancing_summary.csv")
    chosen = pd.read_csv(OUT / "tab_chosen_schedules.csv")
    chosen["plz"] = chosen.plz.astype(str)

    cells = sorted(set(zip(summary.penalty.tolist(), summary.share_willing.tolist())))
    if only is not None:
        cells = [(P, s) for P, s in cells
                 if np.isclose(P, only[0]) and np.isclose(s, only[1])]

    print(f"recomputing bundled init for {len(cells)} cell(s)", flush=True)
    out_rows = []
    t0 = time.time()
    for ci, (P, share) in enumerate(cells, 1):
        t_cell = time.time()
        fs_b2c_v = mod.fs_b2c(share); fs_b2b_v = mod.fs_b2b(share)
        cell_chosen = chosen[(np.isclose(chosen.penalty, P)) &
                             (np.isclose(chosen.share_willing, share))]
        per_prov_costs = {}
        for prov in mod.PROVIDERS:
            if prov not in odata or prov not in mlp:
                continue
            c = _provider_init_bundled(
                prov, fs_b2c_v, fs_b2b_v, cell_chosen, odata[prov], mlp[prov],
                schedules, model,
            )
            per_prov_costs[prov] = c
            out_rows.append({"penalty": P, "share_willing": share,
                              "provider": prov, "init_cost_eur_bundled": c})
        dt = time.time() - t_cell
        tot_min = (time.time() - t0) / 60
        eta_min = dt * (len(cells) - ci) / 60
        print(f"  [{ci:3d}/{len(cells)}] P={P:<5g} share={share:<3g}  "
              f"sum_init_bundled={sum(per_prov_costs.values())/1e3:7.1f}k  "
              f"t_cell={dt:.0f}s  elapsed={tot_min:.1f}min  eta={eta_min:.1f}min",
              flush=True)
    df_new = pd.DataFrame(out_rows)
    out_path = OUT / "_init_bundled_cache.csv"
    df_new.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  ({len(df_new)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated 'P,share' for smoke test on one cell")
    args = ap.parse_args()
    only = tuple(map(float, args.only.split(","))) if args.only else None
    main(only)
