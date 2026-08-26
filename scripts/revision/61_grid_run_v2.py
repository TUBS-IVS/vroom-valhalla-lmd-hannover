"""61: full grid under the realistic-tour rule (base run: head=None).

Re-runs the whole (P, theta, provider) grid through stage 1 -> 2 -> 3 with the
rev1 realistic-tour machinery (per-cell express cost, partition-priced hub
pooling, express-exact fleet objective). Where ``10_recompute_stage3_outputs``
only RE-PRICES the stored 2026-05-29 schedule choices, this script OPTIMIZES:

  stage 1  argmin warm start + ``optimize_cd_ml``   -> schedule_idx_stage1
  stage 2  ``balance_fleet_per_hub_ml``             -> schedule_idx_balanced
  stage 3  ``system_smooth_pass``                   -> schedule_idx_system_smoothed

The calls, argument construction and penalty wiring are copied verbatim from
the canonical production run (``scripts/pipeline/02_optimize_grid.py`` for
stages 1-2, ``scripts/pipeline/03_apply_smoothing.py`` for stage 3); only the
matrices are new. ``matrices["bundle_head"]`` stays absent by design in this
base run, so ``price_group`` falls back to Sigma-pricing.

ONE deliberate deviation from the canonical wiring, see ``--init-proxy``: the
stage-1 warm-start proxy reads ``cost_3d_raw`` (the unpooled per-cell
prediction, i.e. exactly what ``cost_3d`` meant before the rev1 small-delivery
rule) rather than the now-zeroed ``cost_3d``. Reading the zeroed matrix would
price a sub-threshold delivery instance at 0 in the warm start and bias the
initial selection toward schedules that shrink instances. Pass
``--init-proxy pooled`` to reproduce the literal canonical expression.

Loop order is theta-outer / provider-inner / penalty-innermost: the cost
matrices depend on (theta, provider) but NOT on P, so one matrix build is
amortized over all 8 penalties. Matrices are released between blocks.

Resumable: completed (P, theta, provider) triples are skipped. Run OUTSIDE the
agent harness (~59-min kill rule):
  Start-Process .venv\\Scripts\\python.exe -ArgumentList "scripts/revision/61_grid_run_v2.py" -RedirectStandardOutput results/revision_2026_08/61.log

Outputs (results/revision_2026_08/):
  _tab_chosen_v2.csv        penalty, share_willing, provider, plz,
                            schedule_idx_stage1, schedule_idx_balanced,
                            schedule_idx_system_smoothed
  tab_costs_v2.csv          per (P, theta, provider): dd / express / pool split
                            at the system-smoothed choice (+ stage totals)
  tab_fleet_per_hub_v2.csv  per (P, theta, provider, hub, day): partition-aware
                            fleet = veh_3d part + _hub_express_vehicles
  tab_wait_v2.csv           per (P, theta, provider): willing-weighted wait
                            numerator/denominator (50_'s fixed formula); the
                            cell-level wait is sum(num)/sum(den) over providers
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")   # LGBM feature-name notices, one per predict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _stage3_common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "src"))
from batch_delivery.optimization.core import build_cost_matrices_ml  # noqa: E402
from batch_delivery.optimization.coordinate_descent import optimize_cd_ml  # noqa: E402
from batch_delivery.optimization.balancing import (  # noqa: E402
    balance_fleet_per_hub_ml,
    system_smooth_pass,
    _daily_fleet_per_hub,
)
from batch_delivery.optimization.costs import (  # noqa: E402
    _hub_express_day_ml,
    _hub_express_vehicles,
    _hub_smallday_pool_ml,
)
from batch_delivery.optimization.schedules import enumerate_valid_schedules  # noqa: E402

logging.disable(logging.INFO)  # silence the package's INFO/DEBUG chatter

OUT = C.ROOT / "results" / "revision_2026_08"
CHOSEN = OUT / "_tab_chosen_v2.csv"
COSTS = OUT / "tab_costs_v2.csv"
FLEET = OUT / "tab_fleet_per_hub_v2.csv"
WAIT = OUT / "tab_wait_v2.csv"
# CHOSEN is written LAST per triple and is therefore the completion marker;
# _prune_partial() drops orphan rows from the other three on resume.
SIDE_FILES = (COSTS, FLEET, WAIT)

FLEET_COST_BUDGET_PCT = 5.0   # 02_optimize_grid.py:60 (paper revision 2026-05-27)
SMOOTH_BUDGET_PCT = 1.0       # 03_apply_smoothing.py --budget default


# ─────────────────────────────────────────────────────────────────────────────
# Atomic-ish IO with the retry/backoff writer from 20_validate_vroom_smoothed
# ─────────────────────────────────────────────────────────────────────────────

def _retry_write(path: Path, fn) -> None:
    """Run *fn* (a writer closure); retry on transient Windows file locks.

    The 2026-07-16 overnight run died with PermissionError [Errno 13] on a
    checkpoint append (backup/AV/sync briefly locking the CSV). Losing hours
    of resumable progress to a transient lock is unacceptable, so retry with
    backoff for up to ~5 minutes before giving up.
    """
    last_err = None
    for attempt in range(60):
        try:
            fn()
            return
        except PermissionError as e:      # transient lock — wait and retry
            last_err = e
            if attempt == 0:
                print(f"  WARNING: {path.name} locked ({e}); retrying up to 5 min",
                      flush=True)
            time.sleep(5)
    raise last_err


def append_rows(path: Path, rows: list[dict]) -> None:
    """Append rows to *path*, writing the header only on first creation."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    _retry_write(path, lambda: df.to_csv(
        path, mode="a", header=not path.exists(), index=False))


def _rewrite(path: Path, df: pd.DataFrame) -> None:
    _retry_write(path, lambda: df.to_csv(path, index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Resume bookkeeping
# ─────────────────────────────────────────────────────────────────────────────

def _key(P: float, th: float, prov: str) -> tuple[float, float, str]:
    return (round(float(P), 4), round(float(th), 4), str(prov))


def load_done() -> set[tuple[float, float, str]]:
    if not CHOSEN.exists():
        return set()
    d = pd.read_csv(CHOSEN)
    return {_key(r.penalty, r.share_willing, r.provider) for r in d.itertuples()}


def prune_partial(done: set) -> None:
    """Drop rows of any triple missing from CHOSEN (the completion marker).

    The four appends per triple are not atomic; a kill between them leaves a
    partially-written triple that would be redone on resume, duplicating rows.
    """
    for path in SIDE_FILES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if len(df) == 0:
            continue
        keep = np.array([_key(p, s, v) in done for p, s, v
                         in zip(df.penalty, df.share_willing, df.provider)])
        if not keep.all():
            print(f"[self-heal] {path.name}: dropping {int((~keep).sum())} "
                  f"row(s) from partially-appended triple(s)", flush=True)
            _rewrite(path, df[keep])


# ─────────────────────────────────────────────────────────────────────────────
# One (P, theta, provider) triple: stage 1 -> 2 -> 3 + output rows
# ─────────────────────────────────────────────────────────────────────────────

def run_triple(P: float, th: float, prov: str, od: dict, prep: dict, m: dict,
               schedules: list, sched_waits: np.ndarray, args) -> tuple[dict, dict]:
    """Optimize one triple against the pre-built matrices *m*.

    Returns ``(rows_by_table, timings)``.
    """
    plz_keys = od["plz_keys"]
    plz_data = od["plz_data"]
    plz_hub_arr = od["plz_hub_arr"]
    hub_plz_list = od["hub_plz_list"]
    n_plz = len(plz_keys)
    pidx = np.arange(n_plz)

    fs_b2c_v, fs_b2b_v = C.fs_b2c(th), C.fs_b2b(th)
    veh_3d = m["veh_3d"]
    sched_active = m["sched_active"]
    raw_express = m["raw_express"]
    expr_stops = m["expr_stops"]

    # ── penalty wiring, verbatim from 02_optimize_grid.py:236-268 ────────
    weekly_pkts = np.array([
        sum(plz_data[pc]["b2c"].values()) + sum(plz_data[pc]["b2b"].values())
        for pc in plz_keys
    ], dtype=np.float64)
    # FIX 2026-05-27: wait penalty uses PER-PLZ local willing fraction.
    plz_b2c_share = m.get("plz_b2c_share", None)
    if plz_b2c_share is not None:
        local_willing = (plz_b2c_share * (1.0 - fs_b2c_v)
                         + (1.0 - plz_b2c_share) * (1.0 - fs_b2b_v))
    else:
        local_willing = np.full(n_plz, th)
    penalty_mx = (P * local_willing[:, None] * weekly_pkts[:, None]
                  * sched_waits[None, :])

    # ── STAGE 1: argmin warm start, then bundled CD refinement ──────────
    t0 = time.perf_counter()
    # See the module docstring: cost_3d_raw is the pre-rev1 meaning of cost_3d.
    cost_src = m["cost_3d_raw"] if args.init_proxy == "raw" else m["cost_3d"]
    total_cost_mx = cost_src.sum(axis=2)
    obj = (total_cost_mx
           + P * local_willing[:, None] * weekly_pkts[:, None] * sched_waits[None, :])
    # Tie-breaker: among schedules within 0.5% of the cell minimum, pick the
    # largest (daily as natural baseline at share=0 where all schedules ~ tied).
    sched_size_arr = np.array([len(s) for s in schedules], dtype=np.float64)
    obj_min = obj.min(axis=1, keepdims=True)
    near_tied = obj <= obj_min * 1.005
    score = np.where(near_tied, sched_size_arr[None, :], -np.inf)
    chosen_s1 = score.argmax(axis=1).astype(np.int64)
    # At share=0 no parcel is willing to wait -> enforce daily as baseline.
    if th == 0.0:
        daily_si = next(i for i, s in enumerate(schedules) if len(s) == C.N_DAYS)
        chosen_s1 = np.full(n_plz, daily_si, dtype=chosen_s1.dtype)

    if th > 0.0:
        mat_pen = dict(m)
        mat_pen["dd_cost_mx"] = m["dd_cost_mx"] + penalty_mx
        cd = optimize_cd_ml(
            plz_keys, plz_hub_arr, hub_plz_list, mat_pen, schedules,
            fixed_assignment=chosen_s1.astype(np.intp),
            max_rounds=8, shuffle_plz=True, seed=42,
            pair_polish=True, pair_polish_rounds=3, pair_polish_max_pairs=300,
            n_restarts=2,
        )
        chosen_s1 = cd["chosen"].astype(np.int64)
    t_s1 = time.perf_counter() - t0
    print(f"    P={P:<5g} th={th:<4g} {prov:<7s} stage1 {t_s1:7.1f}s", flush=True)

    # ── STAGE 2: per-hub fleet balancing ────────────────────────────────
    t0 = time.perf_counter()
    # init_cost = bundled total (dd + hub-bundled express + pool) of the
    # refined init; balance_fleet_per_hub_ml(max_swaps=0) computes exactly that.
    bal0 = balance_fleet_per_hub_ml(
        {"chosen": chosen_s1, "best_cost": 0.0},
        plz_keys, plz_hub_arr, hub_plz_list, m, schedules,
        cost_budget_pct=FLEET_COST_BUDGET_PCT, max_swaps=0,
    )
    init_cost = float(bal0["initial_total_cost"])

    bal = balance_fleet_per_hub_ml(
        {"chosen": chosen_s1, "best_cost": init_cost},
        plz_keys, plz_hub_arr, hub_plz_list, m, schedules,
        cost_budget_pct=FLEET_COST_BUDGET_PCT,
        penalty_mx=penalty_mx,
        preserve_frequency=True,
    )
    chosen_s2 = bal["chosen"].astype(np.int64)
    t_s2 = time.perf_counter() - t0
    print(f"    P={P:<5g} th={th:<4g} {prov:<7s} stage2 {t_s2:7.1f}s "
          f"({bal['swaps_made']} swaps)", flush=True)

    # ── STAGE 3: system-level smoothing ─────────────────────────────────
    t0 = time.perf_counter()
    res = system_smooth_pass(
        chosen_s2, plz_keys, plz_hub_arr, hub_plz_list, m, schedules,
        cost_budget_pct=args.budget, max_iterations=400,
        penalty_mx=penalty_mx,
    )
    chosen_s3 = res["chosen"].astype(np.int64)
    t_s3 = time.perf_counter() - t0
    print(f"    P={P:<5g} th={th:<4g} {prov:<7s} stage3 {t_s3:7.1f}s "
          f"({res['swaps_made']} swaps)", flush=True)

    # ── OUTPUT: cost split, partition-aware fleet, willing-weighted wait ─
    t0 = time.perf_counter()
    express_cache: dict = {}
    pool_cache: dict = {}

    def _ev(hi: int, d: int, ch: np.ndarray) -> float:
        return _hub_express_vehicles(hi, d, ch, hub_plz_list, schedules,
                                     raw_express, m, express_cache)

    dd_total = float(m["dd_cost_mx"][pidx, chosen_s3].sum())
    expr_total = 0.0
    pool_total = 0.0
    hub_names = [
        prep["hub_name_by_plz"].get(plz_keys[int(h[0])], f"hub_{hi}")
        if len(h) else f"hub_{hi}"
        for hi, h in enumerate(hub_plz_list)
    ]
    fleet_rows = []
    fleet_idx: list[tuple[int, int]] = []
    for hi, h_ps in enumerate(hub_plz_list):
        for d in range(C.N_DAYS):
            expr_total += _hub_express_day_ml(
                hi, d, chosen_s3, hub_plz_list, schedules,
                raw_express, expr_stops, m, express_cache, 1.0)
            pool_total += _hub_smallday_pool_ml(
                hi, d, chosen_s3, hub_plz_list, schedules, m, pool_cache)
            if len(h_ps) == 0:
                continue
            dd_veh = float(veh_3d[h_ps, chosen_s3[h_ps], d].sum())
            ex_veh = float(_ev(hi, d, chosen_s3))     # cache hit
            fleet_rows.append(dict(
                penalty=P, share_willing=th, provider=prov,
                hub=hub_names[hi], day=d,
                dd_veh=dd_veh, express_veh=ex_veh, fleet=dd_veh + ex_veh,
            ))
            fleet_idx.append((hi, d))

    # Gate: the recorded fleet must BE _daily_fleet_per_hub's output (Task 7 G3).
    fleet_s3 = _daily_fleet_per_hub(
        chosen_s3, plz_hub_arr, hub_plz_list, veh_3d, schedules,
        express_veh_fn=_ev)
    for (hi, d), r in zip(fleet_idx, fleet_rows):
        assert abs(fleet_s3[hi, d] - r["fleet"]) < 1e-9, (
            f"fleet mismatch P={P} th={th} {prov} hub={r['hub']} d={d}: "
            f"{fleet_s3[hi, d]} != {r['fleet']}")

    # Gate: the smoother's incrementally-tracked routing cost must equal the
    # independent recomputation of dd + express + pool at its own choice.
    routing_total = dd_total + expr_total + pool_total
    assert abs(res["cost"] - routing_total) <= 1e-6 * max(1.0, abs(routing_total)), (
        f"cost bookkeeping drift P={P} th={th} {prov}: "
        f"tracked {res['cost']:.6f} != recomputed {routing_total:.6f}")

    cost_rows = [dict(
        penalty=P, share_willing=th, provider=prov,
        dd_cost_eur=dd_total,
        express_cost_eur=expr_total,
        pool_cost_eur=pool_total,
        routing_total_eur=routing_total,
        penalty_eur=float(penalty_mx[pidx, chosen_s3].sum()),
        cost_stage1_eur=init_cost,
        cost_stage2_eur=float(bal["cost"]),
        cost_stage3_eur=float(res["cost"]),
        imbalance_before=float(bal["imbalance_before"]),
        imbalance_after=float(bal["imbalance_after"]),
        system_spread_before=float(res["system_spread_before"]),
        system_spread_after=float(res["system_spread_after"]),
        swaps_balance=int(bal["swaps_made"]),
        swaps_smooth=int(res["swaps_made"]),
    )]

    # Willing-weighted wait — formula ported from 50_recompute_fleet_wait_fixed
    # (BUG 2: only the willing fraction actually waits). The cell-level metric
    # is sum(wait_num_willing) / sum(total_parcels) over the 7 providers, so
    # the per-provider numerator and denominator are stored, not the ratio.
    wk = weekly_pkts
    wait_rows = [dict(
        penalty=P, share_willing=th, provider=prov,
        wait_num_willing=float((sched_waits[chosen_s3] * wk * local_willing).sum()),
        wait_num_all=float((sched_waits[chosen_s3] * wk).sum()),
        total_parcels=float(wk.sum()),
        willing_parcels=float((wk * local_willing).sum()),
    )]

    chosen_rows = [dict(
        penalty=P, share_willing=th, provider=prov, plz=str(pc),
        schedule_idx_stage1=int(chosen_s1[pi]),
        schedule_idx_balanced=int(chosen_s2[pi]),
        schedule_idx_system_smoothed=int(chosen_s3[pi]),
    ) for pi, pc in enumerate(plz_keys)]
    t_out = time.perf_counter() - t0

    rows = {"chosen": chosen_rows, "costs": cost_rows,
            "fleet": fleet_rows, "wait": wait_rows}
    return rows, {"s1": t_s1, "s2": t_s2, "s3": t_s3, "out": t_out}


# ─────────────────────────────────────────────────────────────────────────────

def parse_only(spec: str) -> tuple[float | None, float | None, str | None]:
    """``P=0.5,th=0.1,prov=DPD`` -> (0.5, 0.1, 'DPD'); any key may be omitted."""
    P = th = prov = None
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        k = k.strip().lower()
        if k == "p":
            P = float(v)
        elif k in ("th", "theta", "share", "share_willing"):
            th = float(v)
        elif k in ("prov", "provider"):
            prov = v.strip()
        else:
            raise SystemExit(f"--only: unknown key {k!r} in {spec!r}")
    return P, th, prov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="restrict the grid, e.g. P=0.5,th=0.1,prov=DPD")
    ap.add_argument("--budget", type=float, default=SMOOTH_BUDGET_PCT,
                    help="stage-3 system-smoothing cost budget in %% "
                         "(03_apply_smoothing.py default)")
    ap.add_argument("--init-proxy", choices=("raw", "pooled"), default="raw",
                    help="stage-1 warm-start proxy matrix: 'raw' = cost_3d_raw "
                         "(pre-rev1 semantics, default), 'pooled' = the zeroed "
                         "cost_3d (literal canonical expression)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    only_P, only_th, only_prov = (parse_only(args.only) if args.only
                                  else (None, None, None))

    # Grid: the (P, theta) cells of the canonical run, regrouped theta-outer so
    # the (theta, provider) matrix build amortizes over all penalties.
    grid = (pd.read_csv(C.RUN_DIR / "_tab_chosen_with_system_smoothing.csv")
            [["penalty", "share_willing"]].drop_duplicates().values.tolist())
    pairs = [(float(p), float(t)) for p, t in grid]
    thetas = sorted({t for _, t in pairs})
    if only_th is not None:
        thetas = [t for t in thetas if np.isclose(t, only_th)]
    providers = [p for p in C.PROVIDERS
                 if only_prov is None or p == only_prov]
    if not thetas or not providers:
        raise SystemExit(f"--only {args.only!r} matched no grid point")

    done = load_done()
    prune_partial(done)

    todo: list[tuple[float, float, str]] = []
    for th in thetas:
        Ps = sorted({p for p, t in pairs if np.isclose(t, th)})
        if only_P is not None:
            Ps = [p for p in Ps if np.isclose(p, only_P)]
        for prov in providers:
            for P in Ps:
                if _key(P, th, prov) not in done:
                    todo.append((P, th, prov))
    n_total = len(todo)
    print(f"[grid] {len(pairs)} (P, theta) cells x {len(providers)} provider(s); "
          f"{len(done)} triple(s) already done; {n_total} to run", flush=True)
    if n_total == 0:
        print("nothing to do", flush=True)
        return

    print("[load] checkpoints + model ...", flush=True)
    t_load = time.perf_counter()
    provider_data, optim_data = C.load_checkpoints()
    model = C.load_model()
    ml_prep = C.build_ml_prep(provider_data)
    del provider_data
    gc.collect()
    schedules = C.enumerate_schedules()
    assert len(schedules) == 39, f"expected 39 schedules, got {len(schedules)}"
    assert schedules == enumerate_valid_schedules(), (
        "schedule ordering differs from batch_delivery.optimization.schedules "
        "— stored schedule_idx_* columns would be meaningless")
    sched_waits = np.array([C.avg_wait_days(sorted(s)) for s in schedules])
    print(f"[load] done in {time.perf_counter() - t_load:.0f}s "
          f"({len(schedules)} schedules, init_proxy={args.init_proxy}, "
          f"smooth_budget={args.budget}%)", flush=True)

    t_run = time.perf_counter()
    n_done = 0
    for th in thetas:
        Ps = sorted({p for p, t in pairs if np.isclose(t, th)})
        if only_P is not None:
            Ps = [p for p in Ps if np.isclose(p, only_P)]
        fs_b2c_v, fs_b2b_v = C.fs_b2c(th), C.fs_b2b(th)
        for prov in providers:
            block = [P for P in Ps if _key(P, th, prov) not in done]
            if not block:
                continue
            od, prep = optim_data[prov], ml_prep[prov]

            t0 = time.perf_counter()
            m = build_cost_matrices_ml(
                od["plz_keys"], od["plz_data"], schedules, model, prov,
                prep["plz_day_coords"], prep["hub_coords_by_plz"],
                fast_share_b2c=fs_b2c_v, fast_share_b2b=fs_b2b_v)
            assert m.get("bundle_head") is None, (
                "base run must price with the Sigma fallback (head=None)")
            t_mtx = time.perf_counter() - t0
            print(f"[mtx] th={th:<4g} {prov:<7s} built in {t_mtx:.1f}s "
                  f"({len(block)} penalty value(s) to run)", flush=True)

            for P in block:
                rows, tt = run_triple(P, th, prov, od, prep, m, schedules,
                                      sched_waits, args)
                t0 = time.perf_counter()
                # CHOSEN last: it is the completion marker prune_partial reads.
                append_rows(COSTS, rows["costs"])
                append_rows(FLEET, rows["fleet"])
                append_rows(WAIT, rows["wait"])
                append_rows(CHOSEN, rows["chosen"])
                t_w = time.perf_counter() - t0
                done.add(_key(P, th, prov))
                n_done += 1
                el = time.perf_counter() - t_run
                eta = el * (n_total - n_done) / max(1, n_done) / 60.0
                print(f"[{n_done:3d}/{n_total}] P={P:<5g} th={th:<4g} {prov:<7s} "
                      f"mtx={t_mtx:5.1f}s s1={tt['s1']:6.1f}s s2={tt['s2']:6.1f}s "
                      f"s3={tt['s3']:6.1f}s out={tt['out']:5.1f}s write={t_w:4.1f}s "
                      f"| tot={tt['s1'] + tt['s2'] + tt['s3'] + tt['out'] + t_w:6.1f}s "
                      f"eta={eta:.1f}min", flush=True)
                t_mtx = 0.0     # amortized: only the first P of a block pays it

            del m
            gc.collect()

    print(f"\n[done] {n_done} triple(s) in {(time.perf_counter() - t_run) / 60:.1f}min",
          flush=True)
    for p in (CHOSEN, COSTS, FLEET, WAIT):
        if p.exists():
            print(f"  {p} ({len(pd.read_csv(p))} rows)", flush=True)


if __name__ == "__main__":
    main()
