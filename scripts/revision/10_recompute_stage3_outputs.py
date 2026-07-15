"""Recompute fleet/cost/wait for the system-smoothed (Stage-3) schedules.

Deterministic: rebuilds the ML cost matrices exactly like the production
orchestrator and evaluates the stored Stage-3 schedule choices. Two hard
gates abort on any accounting drift:
  gate A: recomputed Stage-2 per-hub/day fleet == tab_fleet_per_hub.fleet_after
  gate B: recomputed Stage-3 system spread   == _system_spread_per_cell.csv

Resumable: after every completed cell the four CSVs are appended to and the
cell is recorded in state_recompute.json, so an interrupted run continues
where it left off (same pattern as the production orchestrator).
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _stage3_common as C

sys.path.insert(0, str(C.ROOT / "src"))
from batch_delivery.optimization.core import (
    build_cost_matrices_ml,
    _daily_fleet_per_hub,
)
from batch_delivery.optimization.costs import _hub_express_day_ml

C.OUT_DIR.mkdir(parents=True, exist_ok=True)

STATE_JSON = C.OUT_DIR / "state_recompute.json"
OUT_FILES = {
    "fleet": C.OUT_DIR / "tab_fleet_per_hub_smoothed.csv",
    "cost": C.OUT_DIR / "tab_costs_smoothed.csv",
    "expr": C.OUT_DIR / "tab_express_smoothed.csv",
    "wait": C.OUT_DIR / "tab_wait_smoothed.csv",
}


def _load_state() -> list[list[float]]:
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text())["completed"]
    return []


def _save_state(completed: list[list[float]]) -> None:
    tmp = STATE_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"completed": completed}, indent=1))
    tmp.replace(STATE_JSON)


def _append(key: str, rows: list[dict]) -> None:
    path = OUT_FILES[key]
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(),
                              index=False)


def main() -> None:
    provider_data, optim_data = C.load_checkpoints()
    model = C.load_model()
    ml_prep = C.build_ml_prep(provider_data)
    schedules = C.enumerate_schedules()
    sched_waits = np.array([C.avg_wait_days(sorted(s)) for s in schedules])

    chosen = pd.read_csv(C.RUN_DIR / "tab_chosen_schedules.csv")
    chosen["plz"] = chosen.plz.astype(str)
    smoothed = pd.read_csv(C.RUN_DIR / "_tab_chosen_with_system_smoothing.csv")
    smoothed["plz"] = smoothed.plz.astype(str)
    ref_fleet = pd.read_csv(C.RUN_DIR / "tab_fleet_per_hub.csv")
    ref_spread = pd.read_csv(C.RUN_DIR / "_system_spread_per_cell.csv")

    cells = sorted(
        set(zip(smoothed.penalty, smoothed.share_willing)) - {(0.4, s) for s in set(smoothed.share_willing)}
    )

    # --- env guard for smoke-testing a limited cell subset ---
    # STAGE3_CELLS="0.0,1.0;0.5,1.0" restricts the run to the listed (P, theta)
    # pairs, e.g. for a fast smoke check before the full ~1-2h grid run.
    _cells_env = os.environ.get("STAGE3_CELLS")
    if _cells_env:
        wanted = set()
        for pair in _cells_env.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            p_str, s_str = pair.split(",")
            wanted.add((float(p_str), float(s_str)))
        cells = [c for c in cells if any(
            np.isclose(c[0], w[0]) and np.isclose(c[1], w[1]) for w in wanted
        )]
        print(f"[env guard] STAGE3_CELLS={_cells_env!r} -> {len(cells)} cell(s): {cells}", flush=True)

    # --- resume support: skip cells already completed in a previous run ---
    completed = _load_state()
    n_target = len(cells)
    todo = [c for c in cells if not any(
        np.isclose(c[0], w[0]) and np.isclose(c[1], w[1]) for w in completed
    )]
    if len(todo) < n_target:
        print(f"[resume] {n_target - len(todo)} cell(s) already done, "
              f"{len(todo)} remaining", flush=True)

    t0 = time.time()

    for i, (P, share) in enumerate(todo, 1):
        fleet_rows, cost_rows, expr_rows, wait_rows = [], [], [], []
        fs_b2c_v, fs_b2b_v = C.fs_b2c(share), C.fs_b2b(share)
        sys_fleet_s3 = np.zeros(C.N_DAYS)
        wait_num = wait_den = 0.0

        for prov in C.PROVIDERS:
            od, prep = optim_data[prov], ml_prep[prov]
            plz_keys = od["plz_keys"]
            plz_hub_arr, hub_plz_list = od["plz_hub_arr"], od["hub_plz_list"]

            m = build_cost_matrices_ml(
                plz_keys, od["plz_data"], schedules, model, prov,
                prep["plz_day_coords"], prep["hub_coords_by_plz"],
                fast_share_b2c=fs_b2c_v, fast_share_b2b=fs_b2b_v,
            )

            def _idx(df, col):
                sub = df[(np.isclose(df.penalty, P))
                         & (np.isclose(df.share_willing, share))
                         & (df.provider == prov)].set_index("plz")
                return np.array([int(sub.loc[str(pc), col]) for pc in plz_keys],
                                dtype=np.int64)

            chosen_s2 = _idx(chosen, "schedule_idx_balanced")
            chosen_s3 = _idx(smoothed, "schedule_idx_system_smoothed")

            # ---- Gate A: reproduce the Stage-2 fleet exactly ----
            fleet_s2 = _daily_fleet_per_hub(chosen_s2, plz_hub_arr, hub_plz_list,
                                            m["veh_3d"], schedules)
            ref = ref_fleet[(np.isclose(ref_fleet.penalty, P))
                            & (np.isclose(ref_fleet.share_willing, share))
                            & (ref_fleet.provider == prov)]
            ref_piv = ref.pivot_table(index="hub", columns="day",
                                      values="fleet_after", aggfunc="first")
            hub_names = [prep["hub_name_by_plz"].get(plz_keys[int(h[0])], f"hub_{hi}")
                         if len(h) else f"hub_{hi}"
                         for hi, h in enumerate(hub_plz_list)]
            for hi, h in enumerate(hub_plz_list):
                if len(h) == 0:
                    continue
                got = fleet_s2[hi]
                want = ref_piv.loc[hub_names[hi]].values.astype(float)
                assert np.allclose(got, want, atol=1e-6), (
                    f"GATE A FAIL P={P} sh={share} {prov} hub={hub_names[hi]}: "
                    f"recomputed {got} != stored {want}"
                )

            # ---- Stage-3 fleet, dd cost, express ----
            fleet_s3 = _daily_fleet_per_hub(chosen_s3, plz_hub_arr, hub_plz_list,
                                            m["veh_3d"], schedules)
            sys_fleet_s3 += fleet_s3.sum(axis=0)

            dd_mx = (m["cost_3d"] * m["sched_active"][None, :, :]).sum(axis=2)
            dd_total = float(dd_mx[np.arange(len(plz_keys)), chosen_s3].sum())

            cache: dict = {}
            expr_total = 0.0
            for hi in range(len(hub_plz_list)):
                for d in range(C.N_DAYS):
                    v = _hub_express_day_ml(
                        hi, d, chosen_s3, hub_plz_list, schedules,
                        m["raw_express"], m["expr_stops"], m, cache, 1.0,
                    )
                    expr_total += v
                    expr_rows.append(dict(penalty=P, share_willing=share,
                                          provider=prov, hub=hub_names[hi],
                                          day=d, express_cost_stage3_eur=v))
                    fleet_rows.append(dict(penalty=P, share_willing=share,
                                           provider=prov, hub=hub_names[hi], day=d,
                                           fleet_stage2=float(fleet_s2[hi, d]),
                                           fleet_stage3=float(fleet_s3[hi, d])))

            cost_rows.append(dict(penalty=P, share_willing=share, provider=prov,
                                  dd_cost_stage3_eur=dd_total,
                                  express_stage3_eur=expr_total,
                                  total_stage3_eur=dd_total + expr_total))

            wk = np.array([sum(od["plz_data"][pc]["b2c"].values())
                           + sum(od["plz_data"][pc]["b2b"].values())
                           for pc in plz_keys])
            wait_num += float((sched_waits[chosen_s3] * wk).sum())
            wait_den += float(wk.sum())

        # ---- Gate B: reproduce the recorded Stage-3 system spread ----
        want = ref_spread[(np.isclose(ref_spread.penalty, P))
                          & (np.isclose(ref_spread.share_willing, share))]
        if len(want):
            got_spread = float(sys_fleet_s3.max() - sys_fleet_s3.min())
            want_spread = float(want.iloc[0].system_spread_after_smoothing)
            assert abs(got_spread - want_spread) < 1e-6, (
                f"GATE B FAIL P={P} sh={share}: spread {got_spread} != {want_spread}"
            )

        wait_rows.append(dict(penalty=P, share_willing=share,
                              avg_wait_d_stage3=wait_num / max(wait_den, 1.0)))

        # --- incremental persist: append this cell's rows + record state ---
        _append("fleet", fleet_rows)
        _append("cost", cost_rows)
        _append("expr", expr_rows)
        _append("wait", wait_rows)
        completed.append([float(P), float(share)])
        _save_state(completed)

        print(f"[{i:2d}/{len(todo)}] P={P:<5g} sh={share:<4g} gates OK "
              f"({len(completed)}/{n_target} total, {time.time()-t0:.0f}s)",
              flush=True)

    if len(completed) >= n_target:
        print("ALL GATES PASSED — outputs written to", C.OUT_DIR)
    else:
        print(f"PARTIAL: {len(completed)}/{n_target} cells done — re-run to resume")


if __name__ == "__main__":
    main()
