"""CD restart-spread analysis (reviewer: no global-optimum guarantee).

The production orchestrator (scripts/pipeline/02_optimize_grid.py, Path 2)
calls ``optimize_cd_ml`` with ``n_restarts=2`` and only ever records the
*best* restart's cost — there is no evidence in the run outputs of how much
the coordinate-descent result varies across restarts. ``orchestrator.log``
does not help either: the package logger was silenced, so no restart lines
were ever written.

This script recomputes the restart spread directly at four validation
points (penalty in {0.0, 0.25, 0.5, 0.75} at share_willing=1.0, the express
end of the grid where hub-bundling coupling is strongest), using
``n_restarts=5`` instead of the production run's 2, and reports how far the
worst restart's cost is from the best restart's cost.

Resumable: rows are appended to ``tab_cd_restart_spread.csv`` after every
(penalty, share_willing, provider) triple, and any triple already present in
the CSV is skipped on re-run. The matrices build (~30s/provider) dominates
runtime; the full 4-cell x 7-provider grid is expected to take 30-60 min.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _stage3_common as C  # noqa: E402

from batch_delivery.optimization.core import (  # noqa: E402
    build_cost_matrices_ml,
    optimize_cd_ml,
)

C.OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = C.OUT_DIR / "tab_cd_restart_spread.csv"

CELLS = [(0.0, 1.0), (0.25, 1.0), (0.5, 1.0), (0.75, 1.0)]
N_RESTARTS = 5
COLUMNS = [
    "penalty", "share_willing", "provider", "n_restarts",
    "best_cost", "worst_cost", "spread_eur", "spread_pct",
]


def _load_done() -> set[tuple[float, float, str]]:
    if not OUT_CSV.exists():
        return set()
    df = pd.read_csv(OUT_CSV)
    return {
        (round(float(p), 6), round(float(s), 6), str(prov))
        for p, s, prov in zip(df.penalty, df.share_willing, df.provider)
    }


def _append_row(row: dict) -> None:
    pd.DataFrame([row], columns=COLUMNS).to_csv(
        OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=False
    )


def main() -> None:
    provider_data, optim_data = C.load_checkpoints()
    model = C.load_model()
    ml_prep = C.build_ml_prep(provider_data)
    schedules = C.enumerate_schedules()
    sched_waits = np.array([C.avg_wait_days(sorted(s)) for s in schedules])
    sched_size_arr = np.array([len(s) for s in schedules], dtype=np.float64)

    done = _load_done()
    n_target = len(CELLS) * len(C.PROVIDERS)
    if done:
        print(f"[resume] {len(done)}/{n_target} (penalty, share, provider) "
              f"triple(s) already present -> skipping", flush=True)

    t0 = time.time()
    n_done_now = 0

    for P, share in CELLS:
        fs_b2c_v = C.fs_b2c(share)
        fs_b2b_v = C.fs_b2b(share)

        for prov in C.PROVIDERS:
            key = (round(float(P), 6), round(float(share), 6), prov)
            if key in done:
                continue
            if prov not in optim_data or prov not in ml_prep:
                continue

            t_cell = time.time()
            odata = optim_data[prov]
            prep = ml_prep[prov]
            plz_keys = odata["plz_keys"]
            plz_data = odata["plz_data"]
            plz_hub_arr = odata["plz_hub_arr"]
            hub_plz_list = odata["hub_plz_list"]

            # ── Build cost matrices with current operating point (same as
            # production orchestrator, scripts/pipeline/02_optimize_grid.py) ──
            m = build_cost_matrices_ml(
                plz_keys, plz_data, schedules, model, prov,
                prep["plz_day_coords"], prep["hub_coords_by_plz"],
                fast_share_b2c=fs_b2c_v, fast_share_b2b=fs_b2b_v,
            )
            total_cost_mx = m["cost_3d"].sum(axis=2)

            # ── Initial cost-optimal selection (under penalty) — VERBATIM
            # from scripts/pipeline/02_optimize_grid.py lines 236-260, minus
            # the share==0.0 special case (all four cells here have
            # share_willing=1.0, so that branch never fires) ──
            weekly_pkts = np.array([
                sum(plz_data[pc]["b2c"].values()) + sum(plz_data[pc]["b2b"].values())
                for pc in plz_keys
            ], dtype=np.float64)
            plz_b2c_share = m.get("plz_b2c_share", None)
            if plz_b2c_share is not None:
                local_willing = (plz_b2c_share * (1.0 - fs_b2c_v)
                                  + (1.0 - plz_b2c_share) * (1.0 - fs_b2b_v))
            else:
                local_willing = np.full(len(plz_keys), share)
            obj = (total_cost_mx
                   + P * local_willing[:, None] * weekly_pkts[:, None] * sched_waits[None, :])
            obj_min = obj.min(axis=1, keepdims=True)
            near_tied = obj <= obj_min * 1.005
            score = np.where(near_tied, sched_size_arr[None, :], -np.inf)
            chosen_init = score.argmax(axis=1).astype(np.int64)

            # ── Path-2 penalized CD refinement, with more restarts than
            # production (n_restarts=5 vs. production's 2) ──
            penalty_mx = (P * local_willing[:, None] * weekly_pkts[:, None]
                          * sched_waits[None, :])
            mat_pen = dict(m)
            mat_pen["dd_cost_mx"] = m["dd_cost_mx"] + penalty_mx
            cd = optimize_cd_ml(
                plz_keys, plz_hub_arr, hub_plz_list, mat_pen, schedules,
                fixed_assignment=chosen_init.astype(np.intp),
                max_rounds=8, shuffle_plz=True, seed=42,
                pair_polish=True, pair_polish_rounds=3, pair_polish_max_pairs=300,
                n_restarts=N_RESTARTS,
            )
            rc = cd["restart_costs"]
            best_cost = float(min(rc))
            worst_cost = float(max(rc))
            row = dict(
                penalty=P, share_willing=share, provider=prov,
                n_restarts=N_RESTARTS, best_cost=best_cost, worst_cost=worst_cost,
                spread_eur=worst_cost - best_cost,
                spread_pct=100.0 * (worst_cost - best_cost) / best_cost,
            )
            _append_row(row)
            done.add(key)
            n_done_now += 1
            print(f"[{len(done):2d}/{n_target}] P={P:<5g} sh={share:<4g} "
                  f"{prov:<8s} best={best_cost:,.0f} worst={worst_cost:,.0f} "
                  f"spread={row['spread_pct']:.3f}%  t={time.time()-t_cell:.0f}s "
                  f"(elapsed {time.time()-t0:.0f}s)", flush=True)

    if len(done) >= n_target:
        print(f"ALL {n_target} rows done — output at {OUT_CSV}")
        df = pd.read_csv(OUT_CSV)
        print(f"max spread_pct = {df.spread_pct.max():.4f}%  "
              f"median spread_pct = {df.spread_pct.median():.4f}%")
    else:
        print(f"PARTIAL: {len(done)}/{n_target} rows done "
              f"({n_done_now} this run) — re-run to resume")


if __name__ == "__main__":
    main()
