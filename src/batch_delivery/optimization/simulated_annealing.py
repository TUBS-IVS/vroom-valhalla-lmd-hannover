"""Simulated-annealing schedule optimisers.

Two SA variants live here:

* ``sa_optimize`` — over the Daganzo proxy cost. Legacy path.
* ``sa_optimize_ml`` (wraps ``_sa_optimize_ml_single``) — over the ML
  cost matrices. ``_sa_optimize_ml_LEGACY`` is the pre-2026-05-22 control.

The production path now uses :func:`coordinate_descent.optimize_cd_ml`
because CD outperforms SA on the (P, theta) grid; the SA variants are
kept for ablation and reproducibility.
"""

import itertools
import math
import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from batch_delivery.config.constants import (
    N_DAYS, WEEKDAYS, MAX_HOLDING_DAYS,
    VEHICLE_CAPACITY, FIXED_COST_EUR, COST_PER_KM_EUR,
    SERVICE_TIME_PER_PARCEL, SERVICE_TIME_CAP,
    AVAILABLE_WORK_S, LINE_HAUL_SPEED_KMH,
    FAST_SHARE_B2C, FAST_SHARE_B2B,
    SA_ITERATIONS, SA_T_INIT, SA_ALPHA, SA_SEED,
    FLEET_BALANCE_MAX_SWAPS,
    CARRIER_DAYS, CARRIER_FIXED_INDICES,
)
from batch_delivery.config.constants import COST_SCALE
from batch_delivery.legacy.daganzo import predict_vec, CalibratedDaganzo
from batch_delivery.io.demand import get_source_days, compute_shifted_demand_plz
from batch_delivery.features import (
    compute_tier2_features, ALL_COLS, TIER2_COLS, _PROVIDER_IDX,
)
from batch_delivery.utils import log

from batch_delivery.optimization.costs import _hub_express_day, _hub_express_day_ml




# ─────────────────────────────────────────────────────────────────────────────
# Step 1: SA — pure cost optimisation (no fleet penalty)
# ─────────────────────────────────────────────────────────────────────────────

def sa_optimize(
    plz_keys: list[str],
    plz_hub_arr: np.ndarray,
    hub_plz_list: list[np.ndarray],
    matrices: dict,
    schedules: list[frozenset[int]],
    cal: CalibratedDaganzo,
    max_iter: int = SA_ITERATIONS,
    t_init: float = SA_T_INIT,
    alpha: float = SA_ALPHA,
    seed: int = SA_SEED,
    batch_only: bool = False,
    express_scale: float = 1.0,
    fixed_assignment: np.ndarray | None = None,
) -> dict:
    """Simulated annealing: minimise Daganzo cost (delivery-day + express).

    Pure cost optimisation — no fleet-balance penalty.
    Fleet balancing is a separate Step 2 (see ``balance_fleet_per_hub``).

    Parameters
    ----------
    batch_only : bool
        If True, express cost is zero (all parcels batched).

    Returns
    -------
    dict with keys: chosen, best_cost, history, accepted, improved,
                    schedules_per_plz.
    """
    dd_cost_mx = matrices["dd_cost_mx"]
    raw_express = matrices["raw_express"]
    expr_stops = matrices["expr_stops"]
    area_arr = matrices["area_arr"]
    hd_arr = matrices["hd_arr"]
    corr_arr = matrices["corr_arr"]
    sched_active = matrices["sched_active"]

    n_plz = len(plz_keys)
    n_sched = len(schedules)
    n_hubs = len(hub_plz_list)
    params = cal.params
    use_jabali = cal.use_jabali

    # Early-stopping: stop if no improvement over PATIENCE iterations
    ES_PATIENCE = max_iter // 6          # ~50k for 300k iterations
    ES_MIN_IMPROVE_PCT = 0.01           # 0.01 % relative improvement

    rng = np.random.default_rng(seed)
    if fixed_assignment is not None:
        chosen = fixed_assignment.copy()
    else:
        chosen = np.argmin(dd_cost_mx, axis=1).copy()

    # Express cache: hub_day_cost[hi, d]
    ecache = np.zeros((n_hubs, N_DAYS))
    if not batch_only:
        for hi in range(n_hubs):
            for d in range(N_DAYS):
                ecache[hi, d] = _hub_express_day(
                    hi, d, chosen, hub_plz_list, schedules,
                    raw_express, expr_stops, area_arr, hd_arr, corr_arr,
                    params, use_jabali, express_scale,
                    sched_active=sched_active,
                )

    _plz_idx = np.arange(n_plz)
    cur_dd = float(dd_cost_mx[_plz_idx, chosen].sum())
    cur_cost = cur_dd + ecache.sum()

    best = chosen.copy()
    best_cost = cur_cost
    T = t_init
    accepted = 0
    improved = 0
    iter_since_improve = 0
    cost_at_last_check = best_cost
    history = [(0, best_cost, T)]

    t0 = time.perf_counter()
    pbar = tqdm(
        range(max_iter), desc="SA optimisation",
        unit_scale=True, leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] cost={postfix}",
    )
    for it in pbar:
        iter_since_improve += 1

        # ── Early-stopping check ─────────────────────────────────────
        if iter_since_improve >= ES_PATIENCE:
            rel_improve = abs(cost_at_last_check - best_cost) / max(1.0, abs(cost_at_last_check))
            if rel_improve < ES_MIN_IMPROVE_PCT / 100:
                log.debug(
                    f"SA early-stop at iter {it:,}: no significant "
                    f"improvement for {ES_PATIENCE:,} iterations "
                    f"(Δ={rel_improve:.6%})"
                )
                break
            cost_at_last_check = best_cost
            iter_since_improve = 0

        pi = int(rng.integers(0, n_plz))
        old_si = int(chosen[pi])
        new_si = int(rng.integers(0, n_sched))
        if new_si == old_si:
            continue

        # Delivery-day cost delta (O(1) lookup)
        delta = dd_cost_mx[pi, new_si] - dd_cost_mx[pi, old_si]

        # Express delta: recompute only affected days at this hub
        if not batch_only:
            hi = int(plz_hub_arr[pi])
            old_days = schedules[old_si]
            new_days = schedules[new_si]
            affected = old_days.symmetric_difference(new_days)

            new_expr_vals: dict[int, float] = {}
            if affected:
                chosen[pi] = new_si  # tentatively apply
                for d in affected:
                    nv = _hub_express_day(
                        hi, d, chosen, hub_plz_list, schedules,
                        raw_express, expr_stops, area_arr, hd_arr, corr_arr,
                        params, use_jabali, express_scale,
                        sched_active=sched_active,
                    )
                    delta += nv - ecache[hi, d]
                    new_expr_vals[d] = nv
                chosen[pi] = old_si  # revert
        else:
            new_expr_vals = {}

        # Metropolis acceptance
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            chosen[pi] = new_si
            cur_cost += delta
            for d, val in new_expr_vals.items():
                ecache[int(plz_hub_arr[pi]), d] = val
            accepted += 1
            if cur_cost < best_cost:
                best = chosen.copy()
                best_cost = cur_cost
                improved += 1
                iter_since_improve = 0
                cost_at_last_check = best_cost

        T *= alpha
        if it % 50_000 == 0:
            history.append((it, best_cost, T))
            pbar.set_postfix_str(f"{best_cost:,.0f}")

    pbar.close()
    sa_iters_done = it + 1 if 'it' in dir() else 0

    # ── Greedy descent polish ────────────────────────────────────────
    # Deterministic sweep: for each PLZ try every schedule; accept any
    # improvement.  Repeat until convergence (max 5 rounds).
    chosen = best.copy()

    # Rebuild express cache for best solution
    if not batch_only:
        for hi in range(n_hubs):
            for d in range(N_DAYS):
                ecache[hi, d] = _hub_express_day(
                    hi, d, chosen, hub_plz_list, schedules,
                    raw_express, expr_stops, area_arr, hd_arr, corr_arr,
                    params, use_jabali, express_scale,
                    sched_active=sched_active,
                )

    cur_cost = best_cost
    polish_improved = True
    polish_rounds = 0

    while polish_improved:
        polish_improved = False
        polish_rounds += 1
        for pi in range(n_plz):
            old_si = int(chosen[pi])
            best_si = old_si
            best_delta = 0.0
            best_expr_new: dict[int, float] = {}

            for new_si in range(n_sched):
                if new_si == old_si:
                    continue
                delta = dd_cost_mx[pi, new_si] - dd_cost_mx[pi, old_si]

                if not batch_only:
                    hi = int(plz_hub_arr[pi])
                    old_days = schedules[old_si]
                    new_days = schedules[new_si]
                    affected = old_days.symmetric_difference(new_days)
                    expr_new: dict[int, float] = {}
                    if affected:
                        chosen[pi] = new_si
                        for d in affected:
                            nv = _hub_express_day(
                                hi, d, chosen, hub_plz_list, schedules,
                                raw_express, expr_stops, area_arr, hd_arr, corr_arr,
                                params, use_jabali, express_scale,
                                sched_active=sched_active,
                            )
                            delta += nv - ecache[hi, d]
                            expr_new[d] = nv
                        chosen[pi] = old_si
                else:
                    expr_new = {}

                if delta < best_delta:
                    best_delta = delta
                    best_si = new_si
                    best_expr_new = expr_new

            if best_si != old_si:
                chosen[pi] = best_si
                cur_cost += best_delta
                for d, val in best_expr_new.items():
                    ecache[int(plz_hub_arr[pi]), d] = val
                polish_improved = True

        if cur_cost < best_cost:
            best = chosen.copy()
            best_cost = cur_cost

        # Safety: max 5 polish rounds
        if polish_rounds >= 5:
            break

    elapsed = time.perf_counter() - t0
    accept_rate = accepted / max(1, sa_iters_done) * 100
    log.debug(
        f"SA complete: {sa_iters_done:,}/{max_iter:,} iter in {elapsed:.0f}s, "
        f"accepted={accept_rate:.1f}%, improved={improved:,}, "
        f"polish_rounds={polish_rounds}, best_cost={best_cost:,.0f}"
    )

    # Build per-PLZ schedule mapping
    schedules_per_plz = {
        plz_keys[pi]: schedules[int(best[pi])] for pi in range(n_plz)
    }

    return {
        "chosen": best,
        "best_cost": best_cost,
        "history": history,
        "accepted": accepted,
        "improved": improved,
        "polish_rounds": polish_rounds,
        "schedules_per_plz": schedules_per_plz,
    }




# ─────────────────────────────────────────────────────────────────────────────
# SA optimisation — ML version
# ─────────────────────────────────────────────────────────────────────────────

def _sa_optimize_ml_single(
    plz_keys: list[str],
    plz_hub_arr: np.ndarray,
    hub_plz_list: list[np.ndarray],
    matrices: dict,
    schedules: list[frozenset[int]],
    init_chosen: np.ndarray,
    max_iter: int,
    t_init: float,
    alpha: float,
    seed: int,
    batch_only: bool,
    express_scale: float,
) -> dict:
    """Single SA run for ML cost with enhanced move strategy.

    Improvements over textbook SA:
    1. Cost-biased schedule proposal (Boltzmann-weighted neighbor selection)
    2. No-waste move: sample from {0..n_sched-1} \\ {old_si} directly
    3. Hub-swap move: 20 % of moves swap two PLZ at the same hub
    4. Reheating: when stuck (no improvement for reheat_patience iterations),
       temperature is reset to t_init × reheat_fraction
    5. Greedy descent polish: after SA cools, deterministic sweep tries
       every schedule for every PLZ until convergence

    Parameters
    ----------
    init_chosen : ndarray
        Starting schedule assignment (one schedule index per PLZ).
    """
    dd_cost_mx = matrices["dd_cost_mx"]
    raw_express = matrices["raw_express"]
    expr_stops = matrices["expr_stops"]

    n_plz = len(plz_keys)
    n_sched = len(schedules)
    n_hubs = len(hub_plz_list)

    # ── SA hyperparameters ───────────────────────────────────────────
    REHEAT_PATIENCE = max_iter // 10       # reheat after 10 % without improvement
    REHEAT_FRACTION = 0.3                  # reheat to 30 % of initial temp
    HUB_SWAP_PROB = 0.20                   # 20 % hub-swap moves
    BOLTZMANN_TEMP_MULT = 2.0              # softmax temperature for schedule proposal

    rng = np.random.default_rng(seed)
    chosen = init_chosen.copy()

    ecache = np.zeros((n_hubs, N_DAYS))
    express_pred_cache: dict = {}

    if not batch_only:
        for hi in range(n_hubs):
            for d in range(N_DAYS):
                ecache[hi, d] = _hub_express_day_ml(
                    hi, d, chosen, hub_plz_list, schedules,
                    raw_express, expr_stops, matrices,
                    express_pred_cache, express_scale,
                )

    _plz_idx = np.arange(n_plz)
    cur_dd = float(dd_cost_mx[_plz_idx, chosen].sum())
    cur_cost = cur_dd + ecache.sum()

    best = chosen.copy()
    best_cost = cur_cost
    T = t_init
    accepted = 0
    improved = 0
    reheats = 0
    iter_since_improve = 0
    history = [(0, best_cost, T)]

    # ── Pre-compute Boltzmann proposal weights per PLZ (vectorised) ─
    # Softmax over dd_cost_mx rows → bias toward cheaper schedules.
    stds = np.maximum(
        1.0, dd_cost_mx.std(axis=1, keepdims=True) * BOLTZMANN_TEMP_MULT,
    )
    _logits = -dd_cost_mx / stds
    _logits -= _logits.max(axis=1, keepdims=True)  # numerical stability
    _w = np.exp(_logits)
    proposal_weights = _w / _w.sum(axis=1, keepdims=True)

    t0 = time.perf_counter()
    pbar = tqdm(
        range(max_iter), desc=f"SA_ML (seed={seed})",
        unit_scale=True, leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] cost={postfix}",
    )
    for it in pbar:
        iter_since_improve += 1

        # ── Reheating: escape local minimum ──────────────────────────
        if iter_since_improve >= REHEAT_PATIENCE:
            T = t_init * REHEAT_FRACTION
            iter_since_improve = 0
            reheats += 1
            # Restore best solution to continue from the best known
            chosen = best.copy()
            cur_dd = float(dd_cost_mx[_plz_idx, chosen].sum())
            if not batch_only:
                for hi in range(n_hubs):
                    for d in range(N_DAYS):
                        ecache[hi, d] = _hub_express_day_ml(
                            hi, d, chosen, hub_plz_list, schedules,
                            raw_express, expr_stops, matrices,
                            express_pred_cache, express_scale,
                        )
            cur_cost = cur_dd + ecache.sum()

        # ── Move selection ───────────────────────────────────────────
        use_hub_swap = (
            not batch_only
            and n_plz > 1
            and rng.random() < HUB_SWAP_PROB
        )

        if use_hub_swap:
            # Hub-swap: pick a hub, pick two PLZ, swap their schedules
            hi = int(rng.integers(0, n_hubs))
            h_ps = hub_plz_list[hi]
            if len(h_ps) < 2:
                continue
            pair = rng.choice(h_ps, size=2, replace=False)
            pi_a, pi_b = int(pair[0]), int(pair[1])
            old_a, old_b = int(chosen[pi_a]), int(chosen[pi_b])
            if old_a == old_b:
                continue
            # Swap: pi_a gets old_b's schedule, pi_b gets old_a's schedule
            delta = (
                dd_cost_mx[pi_a, old_b] - dd_cost_mx[pi_a, old_a]
                + dd_cost_mx[pi_b, old_a] - dd_cost_mx[pi_b, old_b]
            )

            # Express delta for hub-swap
            old_days_a = schedules[old_a]
            old_days_b = schedules[old_b]
            affected = old_days_a.symmetric_difference(old_days_b)
            new_expr_vals_swap: dict[int, float] = {}
            if affected:
                chosen[pi_a] = old_b
                chosen[pi_b] = old_a
                for d_aff in affected:
                    nv = _hub_express_day_ml(
                        hi, d_aff, chosen, hub_plz_list, schedules,
                        raw_express, expr_stops, matrices,
                        express_pred_cache, express_scale,
                    )
                    delta += nv - ecache[hi, d_aff]
                    new_expr_vals_swap[d_aff] = nv
                chosen[pi_a] = old_a
                chosen[pi_b] = old_b

            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                chosen[pi_a] = old_b
                chosen[pi_b] = old_a
                cur_cost += delta
                for d_val, val in new_expr_vals_swap.items():
                    ecache[hi, d_val] = val
                accepted += 1
                if cur_cost < best_cost:
                    best = chosen.copy()
                    best_cost = cur_cost
                    improved += 1
                    iter_since_improve = 0
        else:
            # Standard single-PLZ move with cost-biased schedule proposal
            pi = int(rng.integers(0, n_plz))
            old_si = int(chosen[pi])

            # Boltzmann-weighted schedule selection (excluding current)
            pw = proposal_weights[pi].copy()
            pw[old_si] = 0.0
            pw_sum = pw.sum()
            if pw_sum <= 0:
                continue
            pw /= pw_sum
            new_si = int(rng.choice(n_sched, p=pw))

            delta = dd_cost_mx[pi, new_si] - dd_cost_mx[pi, old_si]

            if not batch_only:
                hi = int(plz_hub_arr[pi])
                old_days = schedules[old_si]
                new_days = schedules[new_si]
                affected = old_days.symmetric_difference(new_days)

                new_expr_vals: dict[int, float] = {}
                if affected:
                    chosen[pi] = new_si
                    for d_aff in affected:
                        nv = _hub_express_day_ml(
                            hi, d_aff, chosen, hub_plz_list, schedules,
                            raw_express, expr_stops, matrices,
                            express_pred_cache, express_scale,
                        )
                        delta += nv - ecache[hi, d_aff]
                        new_expr_vals[d_aff] = nv
                    chosen[pi] = old_si
            else:
                new_expr_vals = {}

            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                chosen[pi] = new_si
                cur_cost += delta
                for d_val, val in new_expr_vals.items():
                    ecache[int(plz_hub_arr[pi]), d_val] = val
                accepted += 1
                if cur_cost < best_cost:
                    best = chosen.copy()
                    best_cost = cur_cost
                    improved += 1
                    iter_since_improve = 0

        T *= alpha
        if it % 50_000 == 0:
            history.append((it, best_cost, T))
            pbar.set_postfix_str(f"{best_cost:,.0f}")

    pbar.close()

    # ── Greedy descent polish ────────────────────────────────────────
    # Deterministic sweep: for each PLZ try every schedule, accept any
    # improvement.  Repeat until no PLZ improves (convergence).
    chosen = best.copy()

    # Rebuild express cache for best solution
    if not batch_only:
        for hi in range(n_hubs):
            for d in range(N_DAYS):
                ecache[hi, d] = _hub_express_day_ml(
                    hi, d, chosen, hub_plz_list, schedules,
                    raw_express, expr_stops, matrices,
                    express_pred_cache, express_scale,
                )

    cur_cost = best_cost
    polish_improved = True
    polish_rounds = 0
    while polish_improved:
        polish_improved = False
        polish_rounds += 1
        for pi in range(n_plz):
            old_si = int(chosen[pi])
            best_si = old_si
            best_delta = 0.0
            best_expr_new: dict[int, float] = {}

            for new_si in range(n_sched):
                if new_si == old_si:
                    continue
                delta = dd_cost_mx[pi, new_si] - dd_cost_mx[pi, old_si]

                if not batch_only:
                    hi = int(plz_hub_arr[pi])
                    old_days = schedules[old_si]
                    new_days = schedules[new_si]
                    affected = old_days.symmetric_difference(new_days)
                    expr_new: dict[int, float] = {}
                    if affected:
                        chosen[pi] = new_si
                        for d_aff in affected:
                            nv = _hub_express_day_ml(
                                hi, d_aff, chosen, hub_plz_list, schedules,
                                raw_express, expr_stops, matrices,
                                express_pred_cache, express_scale,
                            )
                            delta += nv - ecache[hi, d_aff]
                            expr_new[d_aff] = nv
                        chosen[pi] = old_si
                else:
                    expr_new = {}

                if delta < best_delta:
                    best_delta = delta
                    best_si = new_si
                    best_expr_new = expr_new

            if best_si != old_si:
                chosen[pi] = best_si
                cur_cost += best_delta
                for d_val, val in best_expr_new.items():
                    ecache[int(plz_hub_arr[pi]), d_val] = val
                polish_improved = True

        if cur_cost < best_cost:
            best = chosen.copy()
            best_cost = cur_cost

        # Safety: max 5 polish rounds to avoid infinite loops
        if polish_rounds >= 5:
            break

    elapsed = time.perf_counter() - t0
    accept_rate = accepted / max(1, max_iter) * 100
    log.debug(
        f"SA_ML run (seed={seed}): {max_iter:,} iter in {elapsed:.0f}s, "
        f"accepted={accept_rate:.1f}%, improved={improved:,}, "
        f"reheats={reheats}, polish_rounds={polish_rounds}, "
        f"best_cost={best_cost:,.0f}, cache_size={len(express_pred_cache):,}"
    )

    schedules_per_plz = {
        plz_keys[pi]: schedules[int(best[pi])] for pi in range(n_plz)
    }

    return {
        "chosen": best,
        "best_cost": best_cost,
        "history": history,
        "accepted": accepted,
        "improved": improved,
        "reheats": reheats,
        "polish_rounds": polish_rounds,
        "schedules_per_plz": schedules_per_plz,
    }




def sa_optimize_ml(
    plz_keys: list[str],
    plz_hub_arr: np.ndarray,
    hub_plz_list: list[np.ndarray],
    matrices: dict,
    schedules: list[frozenset[int]],
    max_iter: int = SA_ITERATIONS,
    t_init: float = SA_T_INIT,
    alpha: float = SA_ALPHA,
    seed: int = SA_SEED,
    batch_only: bool = False,
    express_scale: float = 1.0,
    n_restarts: int = 3,
    fixed_assignment: np.ndarray | None = None,
) -> dict:
    """Legacy wrapper — delegates to coordinate descent (optimize_cd_ml)."""
    return optimize_cd_ml(
        plz_keys=plz_keys,
        plz_hub_arr=plz_hub_arr,
        hub_plz_list=hub_plz_list,
        matrices=matrices,
        schedules=schedules,
        batch_only=batch_only,
        express_scale=express_scale,
        fixed_assignment=fixed_assignment,
    )




def _sa_optimize_ml_LEGACY(
    plz_keys: list[str],
    plz_hub_arr: np.ndarray,
    hub_plz_list: list[np.ndarray],
    matrices: dict,
    schedules: list[frozenset[int]],
    max_iter: int = SA_ITERATIONS,
    t_init: float = SA_T_INIT,
    alpha: float = SA_ALPHA,
    seed: int = SA_SEED,
    batch_only: bool = False,
    express_scale: float = 1.0,
    n_restarts: int = 3,
    fixed_assignment: np.ndarray | None = None,
) -> dict:
    """SA minimising ML-predicted cost with multi-restart strategy (LEGACY).

    Runs ``n_restarts`` independent SA optimisations with different seeds
    and initial assignments, then returns the best result.

    Restart strategy:
    - Restart 0: greedy argmin initialisation (cheapest schedule per PLZ)
    - Restart 1: fixed-schedule assignment (if *fixed_assignment* provided)
    - Restart 2+: random initialisation with coprime seed spacing

    Parameters
    ----------
    n_restarts : int
        Number of independent SA runs (default 3).
    fixed_assignment : ndarray, optional
        Schedule indices from the fixed-schedule scenario.  Used as
        starting point for the second restart.

    Returns
    -------
    dict with keys: chosen, best_cost, history, accepted, improved,
                    schedules_per_plz, restart_costs.
    """
    dd_cost_mx = matrices["dd_cost_mx"]
    n_plz = len(plz_keys)
    n_sched = len(schedules)

    results: list[dict] = []
    restart_costs: list[float] = []

    for restart_idx in range(n_restarts):
        # ── Determine initial assignment for this restart ────────────
        if restart_idx == 0 and fixed_assignment is not None:
            # Start from carrier's fixed schedule (e.g. Mon/Wed/Fri)
            init = fixed_assignment.copy()
        elif restart_idx == 0 or (restart_idx == 1 and fixed_assignment is not None):
            # Greedy: assign each PLZ its cheapest schedule
            init = np.argmin(dd_cost_mx, axis=1).copy()
        else:
            # Random initialisation with coprime seed spacing
            r_rng = np.random.default_rng(seed + restart_idx * 7919)
            init = r_rng.integers(0, n_sched, size=n_plz)

        restart_seed = seed + restart_idx * 7919

        res = _sa_optimize_ml_single(
            plz_keys, plz_hub_arr, hub_plz_list, matrices, schedules,
            init_chosen=init,
            max_iter=max_iter, t_init=t_init, alpha=alpha,
            seed=restart_seed, batch_only=batch_only,
            express_scale=express_scale,
        )
        results.append(res)
        restart_costs.append(res["best_cost"])
        log.info(
            f"SA_ML restart {restart_idx + 1}/{n_restarts}: "
            f"cost={res['best_cost']:,.0f}"
        )

    # Pick the best run
    best_idx = int(np.argmin(restart_costs))
    best_result = results[best_idx]
    best_result["restart_costs"] = restart_costs

    log.info(
        f"SA_ML multi-restart complete: best restart={best_idx + 1}, "
        f"costs={[f'{c:,.0f}' for c in restart_costs]}"
    )

    return best_result
