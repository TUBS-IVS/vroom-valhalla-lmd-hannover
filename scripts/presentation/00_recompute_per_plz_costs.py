"""Persist per-PLZ direct-delivery costs for the Stage-3 schedules at theta=1.

Why theta=1 only: the express (fast-lane) cost component is bundled per hub and
is NOT decomposable to single postal-code areas, so a per-PLZ saving built on a
cell with express > 0 would silently mix a bundled quantity into a per-PLZ one.
At share_willing = 1.0 the express component is exactly 0.0 for every provider
and every penalty level (verified against tab_costs_smoothed.csv), so the total
cost equals the direct-delivery cost and a per-PLZ decomposition is exact.

The baseline reference is the full-week schedule (Monday-Saturday, i.e. daily
delivery), which is schedule index len(schedules)-1 under the pinned
enumerate_schedules() ordering.

Two hard gates abort on any accounting drift:
  gate A: sum of per-PLZ Stage-3 dd cost == tab_costs_smoothed.dd_cost_stage3_eur
  gate B: sum of per-PLZ baseline dd cost == tab_baseline_per_provider.dd_cost

Resumable: after every completed cell the CSV is appended to and the cell is
recorded in state_per_plz.json, so an interrupted run continues where it left
off (same pattern as scripts/revision/10_recompute_stage3_outputs.py).

Output: results/revision_2026_07/tab_per_plz_costs_theta1.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "revision"))
import _stage3_common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "src"))
from batch_delivery.optimization.core import build_cost_matrices_ml  # noqa: E402

THETA = 1.0
PENALTIES = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]

OUT_CSV = C.OUT_DIR / "tab_per_plz_costs_theta1.csv"
STATE_JSON = C.OUT_DIR / "state_per_plz.json"
BASELINE_CSV = (C.ROOT / "results" / "paper_outputs_2026_05_30" / "02_baseline"
                / "tab_baseline_per_provider.csv")

# Gate tolerances. Gate A reconciles two float64 sums of the same matrix
# entries, so it is tight. Gate B compares against a CSV rounded at write
# time, so it gets a relative tolerance.
GATE_A_ATOL = 1e-6
GATE_B_RTOL = 1e-9


def _load_state() -> list[list[float]]:
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text())["completed"]
    return []


def _save_state(completed: list[list[float]]) -> None:
    tmp = STATE_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"completed": completed}, indent=1))
    tmp.replace(STATE_JSON)


def _prune_partial(completed: list[list[float]]) -> None:
    """Self-heal: drop rows of any cell not recorded as completed."""
    if not OUT_CSV.exists():
        return
    df = pd.read_csv(OUT_CSV, dtype={"plz": str})
    if len(df) == 0:
        return
    mask = np.zeros(len(df), dtype=bool)
    for P, s in completed:
        mask |= np.isclose(df.penalty, P) & np.isclose(df.share_willing, s)
    if not mask.all():
        print(f"[self-heal] dropping {int((~mask).sum())} row(s) from "
              f"partially-appended cell(s)", flush=True)
        df[mask].to_csv(OUT_CSV, index=False)


def _full_week_index(schedules: list) -> int:
    """Index of the Monday-Saturday (daily delivery) schedule."""
    full = frozenset(range(C.N_DAYS))
    hits = [i for i, s in enumerate(schedules) if s == full]
    assert len(hits) == 1, f"expected exactly one full-week schedule, got {hits}"
    return hits[0]


def main() -> None:
    provider_data, optim_data = C.load_checkpoints()
    model = C.load_model()
    ml_prep = C.build_ml_prep(provider_data)
    schedules = C.enumerate_schedules()
    assert len(schedules) == 39, f"expected 39 patterns, got {len(schedules)}"
    full_idx = _full_week_index(schedules)
    sched_waits = np.array([C.avg_wait_days(sorted(s)) for s in schedules])

    smoothed = pd.read_csv(C.RUN_DIR / "_tab_chosen_with_system_smoothing.csv")
    smoothed["plz"] = smoothed.plz.astype(str)
    ref_cost = pd.read_csv(C.OUT_DIR / "tab_costs_smoothed.csv")
    ref_base = pd.read_csv(BASELINE_CSV)

    # Guard the premise of this whole script before spending any compute.
    expr_at_theta = ref_cost[np.isclose(ref_cost.share_willing, THETA)]
    assert (expr_at_theta.express_stage3_eur == 0.0).all(), (
        "express is not exactly 0 at theta=%s -- per-PLZ decomposition would "
        "mix a hub-bundled quantity into a per-PLZ one" % THETA
    )
    print(f"premise OK: express == 0.0 for all {len(expr_at_theta)} "
          f"(P, provider) rows at theta={THETA}", flush=True)

    cells = [(P, THETA) for P in PENALTIES]

    _cells_env = os.environ.get("PRES_CELLS")
    if _cells_env:
        wanted = {float(x.strip()) for x in _cells_env.split(";") if x.strip()}
        cells = [c for c in cells if any(np.isclose(c[0], w) for w in wanted)]
        print(f"[env guard] PRES_CELLS={_cells_env!r} -> {len(cells)} cell(s)",
              flush=True)

    completed = _load_state()
    _prune_partial(completed)
    n_target = len(cells)
    todo = [c for c in cells if not any(
        np.isclose(c[0], w[0]) and np.isclose(c[1], w[1]) for w in completed
    )]
    if len(todo) < n_target:
        print(f"[resume] {n_target - len(todo)} cell(s) already done, "
              f"{len(todo)} remaining", flush=True)

    fs_b2c_v, fs_b2b_v = C.fs_b2c(THETA), C.fs_b2b(THETA)
    t0 = time.time()

    for i, (P, share) in enumerate(todo, 1):
        rows: list[dict] = []

        for prov in C.PROVIDERS:
            od, prep = optim_data[prov], ml_prep[prov]
            plz_keys = od["plz_keys"]

            m = build_cost_matrices_ml(
                plz_keys, od["plz_data"], schedules, model, prov,
                prep["plz_day_coords"], prep["hub_coords_by_plz"],
                fast_share_b2c=fs_b2c_v, fast_share_b2b=fs_b2b_v,
            )

            sub = smoothed[(np.isclose(smoothed.penalty, P))
                           & (np.isclose(smoothed.share_willing, share))
                           & (smoothed.provider == prov)].set_index("plz")
            chosen = np.array(
                [int(sub.loc[str(pc), "schedule_idx_system_smoothed"])
                 for pc in plz_keys], dtype=np.int64)

            dd_mx = (m["cost_3d"] * m["sched_active"][None, :, :]).sum(axis=2)
            dd_s3 = dd_mx[np.arange(len(plz_keys)), chosen]
            dd_base = dd_mx[:, full_idx]

            # ---- Gate A: per-PLZ sum reproduces the provider Stage-3 total ----
            want = ref_cost[(np.isclose(ref_cost.penalty, P))
                            & (np.isclose(ref_cost.share_willing, share))
                            & (ref_cost.provider == prov)]
            assert len(want) == 1, f"no reference cost row for {P}/{share}/{prov}"
            want_dd = float(want.iloc[0].dd_cost_stage3_eur)
            got_dd = float(dd_s3.sum())
            assert abs(got_dd - want_dd) < GATE_A_ATOL, (
                f"GATE A FAIL P={P} {prov}: per-PLZ sum {got_dd!r} != "
                f"stored dd_cost_stage3_eur {want_dd!r} "
                f"(delta {got_dd - want_dd:.6e})"
            )

            # ---- Gate B: baseline sum reproduces the pinned daily-delivery cost ----
            want_base = float(
                ref_base[ref_base.provider == prov].iloc[0].dd_cost)
            got_base = float(dd_base.sum())
            assert np.isclose(got_base, want_base, rtol=GATE_B_RTOL), (
                f"GATE B FAIL {prov}: baseline sum {got_base!r} != "
                f"tab_baseline_per_provider dd_cost {want_base!r} "
                f"(rel {abs(got_base - want_base) / want_base:.3e})"
            )

            hub_name_by_plz = prep["hub_name_by_plz"]
            for j, pc in enumerate(plz_keys):
                pd_ = od["plz_data"][pc]
                b2c_wk = float(sum(pd_["b2c"].values()))
                b2b_wk = float(sum(pd_["b2b"].values()))
                wk = b2c_wk + b2b_wk
                sched = sorted(schedules[int(chosen[j])])
                rows.append(dict(
                    penalty=P, share_willing=share, provider=prov, plz=str(pc),
                    hub=hub_name_by_plz.get(pc, ""),
                    schedule_size_stage3=len(sched),
                    schedule_days_stage3="".join(str(d) for d in sched),
                    avg_wait_d_stage3=float(sched_waits[int(chosen[j])]),
                    dd_cost_stage3_eur=float(dd_s3[j]),
                    dd_cost_baseline_eur=float(dd_base[j]),
                    saving_abs_eur=float(dd_base[j] - dd_s3[j]),
                    saving_pct=float(
                        (1.0 - dd_s3[j] / dd_base[j]) * 100.0)
                    if dd_base[j] > 0 else np.nan,
                    weekly_parcels=wk,
                    b2c_share=(b2c_wk / wk) if wk > 0 else np.nan,
                    area_km2=float(pd_["area_km2"]),
                    hub_dist_km=float(pd_["hub_dist_km"]),
                    n_stops_per_day=float(pd_["n_stops_per_day"]),
                    demand_per_area=(wk / float(pd_["area_km2"]))
                    if pd_["area_km2"] > 0 else np.nan,
                ))

        pd.DataFrame(rows).to_csv(OUT_CSV, mode="a",
                                  header=not OUT_CSV.exists(), index=False)
        completed.append([float(P), float(share)])
        _save_state(completed)
        print(f"[{i:2d}/{len(todo)}] P={P:<5g} theta={share:<4g} gates A+B OK "
              f"({len(completed)}/{n_target} total, {time.time() - t0:.0f}s)",
              flush=True)

    if len(completed) >= n_target:
        print(f"ALL GATES PASSED -- {OUT_CSV}")
    else:
        print(f"INCOMPLETE: {len(completed)}/{n_target} cells done; rerun to resume")


if __name__ == "__main__":
    main()
