"""Recompute fleet and wait metrics with two reporting bugs fixed.

Both bugs affect ONLY the reported metrics, never the optimisation itself
(the objective's penalty term already uses the local willing share, and the
cost path already bundles the express tour). So no re-optimisation is needed.

BUG 1 — fleet double counting
  veh_3d assigns >= 1 vehicle to EVERY non-delivering cell on every day
  (costs.py: veh_3d[active] = max(1, nr), active includes express_demand on
  non-delivery days). But the COST of those parcels is computed as ONE pooled
  hub tour (_hub_express_day_ml). Fleet and cost therefore disagree.
  FIX: fleet(hub, day) = sum(dd vehicles of delivering cells)
                       + ceil(pooled express demand / Q)   [one shared tour]

BUG 2 — wait over-weighting
  Reported wait = sum(sched_wait * ALL parcels) / sum(ALL parcels), but only
  the willing fraction actually waits; standard parcels are delivered daily.
  FIX: wait = sum(sched_wait * willing parcels) / sum(ALL parcels)
  (identical at theta = 1, where willing == all)

Outputs (results/revision_2026_07/):
  tab_fleet_per_hub_fixed.csv   penalty, share_willing, provider, hub, day,
                                fleet_old, fleet_fixed, dd_veh, expr_veh_old,
                                expr_veh_fixed
  tab_wait_fixed.csv            penalty, share_willing, wait_old, wait_fixed,
                                willing_parcels, total_parcels

DEPRECATED (2026-08 revision): superseded by scripts/revision/61_grid_run_v2.py,
67_validate_vroom_v2.py, 70_figs_tables_v2.py and 73_tables_ops_v2.py.
"""
from __future__ import annotations
import argparse, math, sys, time

# --- DEPRECATED ENTRY POINT (2026-08 revision) -----------------------------
import warnings as _deprecation_warnings

_deprecation_warnings.warn(
    "50_recompute_fleet_wait_fixed.py is a STALE entry point: it recomputes totals WITHOUT the pool "
    "term and predates the universal tour rule, the two cost lenses and the "
    "operator polish. Its numbers are NOT comparable with the 2026-08 "
    "revision. Use scripts/revision/61_grid_run_v2.py for the grid, "
    "scripts/revision/67_validate_vroom_v2.py for VROOM validation, "
    "scripts/revision/70_figs_tables_v2.py for figures and tables, and "
    "scripts/revision/73_tables_ops_v2.py for the v2 ops/knee/value-of-"
    "stage-2 tables.",
    DeprecationWarning,
    stacklevel=2,
)
# ---------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _stage3_common as C

sys.path.insert(0, str(C.ROOT / "src"))
from batch_delivery.config.constants import VEHICLE_CAPACITY
from batch_delivery.optimization.core import build_cost_matrices_ml

OUT = C.OUT_DIR
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None,
                    help="comma list like 10.0_0.1,0.0_1.0 (default: all)")
    args = ap.parse_args()

    provider_data, optim_data = C.load_checkpoints()
    model = C.load_model()
    ml_prep = C.build_ml_prep(provider_data)
    schedules = C.enumerate_schedules()
    sched_waits = np.array([C.avg_wait_days(sorted(s)) for s in schedules])
    sched_active = np.zeros((len(schedules), C.N_DAYS), dtype=bool)
    for si, s in enumerate(schedules):
        for d in s:
            sched_active[si, d] = True

    sm = pd.read_csv(C.RUN_DIR / "_tab_chosen_with_system_smoothing.csv")
    sm["plz"] = sm.plz.astype(str)

    cells = sorted({(float(p), float(s)) for p, s in
                    zip(sm.penalty, sm.share_willing)})
    cells = [c for c in cells if not np.isclose(c[0], 0.4)]
    if args.cells:
        want = {tuple(map(float, c.split("_"))) for c in args.cells.split(",")}
        cells = [c for c in cells if c in want]

    # Resumable: keep finished cells, skip them (harness kills detached runs
    # after ~59 min, so a long sweep must survive a restart).
    fleet_rows, wait_rows = [], []
    done: set[tuple[float, float]] = set()
    fp, wp = OUT / "tab_fleet_per_hub_fixed.csv", OUT / "tab_wait_fixed.csv"
    if fp.exists() and wp.exists():
        fdf, wdf = pd.read_csv(fp), pd.read_csv(wp)
        done = {(float(p), float(s)) for p, s in zip(wdf.penalty, wdf.share_willing)}
        # keep only rows of completed cells (drops a half-written cell)
        keep = fdf.apply(lambda r: (float(r.penalty), float(r.share_willing)) in done, axis=1)
        fleet_rows = fdf[keep].to_dict("records")
        wait_rows = wdf.to_dict("records")
        print(f"[resume] {len(done)} cell(s) already done")
    cells = [c for c in cells if c not in done]
    if not cells:
        print("nothing to do — all cells present")
        return
    t0 = time.time()

    for i, (P, th) in enumerate(cells, 1):
        fs_b2c_v, fs_b2b_v = C.fs_b2c(th), C.fs_b2b(th)
        w_num = w_num_old = w_den = 0.0
        willing_tot = 0.0

        for prov in C.PROVIDERS:
            od, prep = optim_data[prov], ml_prep[prov]
            plz_keys = od["plz_keys"]
            hub_plz_list = od["hub_plz_list"]

            m = build_cost_matrices_ml(
                plz_keys, od["plz_data"], schedules, model, prov,
                prep["plz_day_coords"], prep["hub_coords_by_plz"],
                fast_share_b2c=fs_b2c_v, fast_share_b2b=fs_b2b_v,
            )
            veh_3d = m["veh_3d"]
            raw_express = m["raw_express"]

            sub = sm[(np.isclose(sm.penalty, P)) & (np.isclose(sm.share_willing, th))
                     & (sm.provider == prov)].set_index("plz")
            chosen = np.array([int(sub.loc[str(pc), "schedule_idx_system_smoothed"])
                               for pc in plz_keys], dtype=np.int64)

            # ---- wait (bug 2) ----
            wk = np.array([sum(od["plz_data"][pc]["b2c"].values())
                           + sum(od["plz_data"][pc]["b2b"].values())
                           for pc in plz_keys], dtype=np.float64)
            b2cs = m.get("plz_b2c_share")
            if b2cs is not None:
                local_willing = (b2cs * (1.0 - fs_b2c_v)
                                 + (1.0 - b2cs) * (1.0 - fs_b2b_v))
            else:
                local_willing = np.full(len(plz_keys), th)
            w_num += float((sched_waits[chosen] * wk * local_willing).sum())
            w_num_old += float((sched_waits[chosen] * wk).sum())
            w_den += float(wk.sum())
            willing_tot += float((wk * local_willing).sum())

            # ---- fleet (bug 1) ----
            hub_names = []
            for hi, h in enumerate(hub_plz_list):
                hub_names.append(prep["hub_name_by_plz"].get(plz_keys[int(h[0])], f"hub_{hi}")
                                 if len(h) else f"hub_{hi}")
            for hi, h_ps in enumerate(hub_plz_list):
                if len(h_ps) == 0:
                    continue
                for d in range(C.N_DAYS):
                    delivering = sched_active[chosen[h_ps], d]
                    # delivery-day vehicles of the cells that run their own tour
                    dd_veh = float(veh_3d[h_ps[delivering], chosen[h_ps[delivering]], d].sum()) \
                        if delivering.any() else 0.0
                    # express of the cells WITHOUT own tour that day
                    nd = h_ps[~delivering]
                    ex_dem = raw_express[nd, d] if len(nd) else np.array([])
                    ex_mask = ex_dem > 0
                    # OLD: >=1 vehicle per cell (what veh_3d encodes)
                    ex_old = float(veh_3d[nd[ex_mask], chosen[nd[ex_mask]], d].sum()) \
                        if ex_mask.any() else 0.0
                    # FIXED: one pooled tour for the hub
                    tot_ex = float(ex_dem[ex_mask].sum()) if ex_mask.any() else 0.0
                    ex_fix = float(math.ceil(tot_ex / VEHICLE_CAPACITY)) if tot_ex > 0 else 0.0
                    fleet_rows.append(dict(
                        penalty=P, share_willing=th, provider=prov,
                        hub=hub_names[hi], day=d,
                        dd_veh=dd_veh, expr_veh_old=ex_old, expr_veh_fixed=ex_fix,
                        fleet_old=dd_veh + ex_old, fleet_fixed=dd_veh + ex_fix,
                    ))

        wait_rows.append(dict(penalty=P, share_willing=th,
                              wait_old=w_num_old / max(w_den, 1.0),
                              wait_fixed=w_num / max(w_den, 1.0),
                              willing_parcels=willing_tot, total_parcels=w_den))
        el = time.time() - t0
        print(f"[{i:2d}/{len(cells)}] P={P:<5g} th={th:<4g} "
              f"wait {wait_rows[-1]['wait_old']:.3f}->{wait_rows[-1]['wait_fixed']:.3f} "
              f"({el:.0f}s)", flush=True)
        pd.DataFrame(fleet_rows).to_csv(OUT / "tab_fleet_per_hub_fixed.csv", index=False)
        pd.DataFrame(wait_rows).to_csv(OUT / "tab_wait_fixed.csv", index=False)

    print(f"\nwrote {OUT/'tab_fleet_per_hub_fixed.csv'} ({len(fleet_rows)} rows)")
    print(f"wrote {OUT/'tab_wait_fixed.csv'} ({len(wait_rows)} rows)")


if __name__ == "__main__":
    main()
