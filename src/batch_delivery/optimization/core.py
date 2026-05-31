"""Schedule optimisation: enumeration, SA (pure cost), fleet balancing.

Two-step approach:
  Step 1 — SA minimises total Daganzo proxy cost (delivery-day + hub-bundled
           express) without any fleet-balance penalty.
  Step 2 — Swap-based post-processing equalises daily vehicle counts per hub
           while respecting a cost-increase budget.
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


# ─────────────────────────────────────────────────────────────────────────────
# Valid schedule enumeration
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_valid_schedules() -> list[frozenset[int]]:
    """Return all delivery-day subsets satisfying the holding-day constraint.

    A schedule is valid iff the maximum gap between consecutive delivery
    days (cyclic) does not exceed ``MAX_HOLDING_DAYS``.
    """
    min_freq = max(2, math.ceil(N_DAYS / MAX_HOLDING_DAYS))
    valid: list[frozenset[int]] = []
    for size in range(min_freq, N_DAYS + 1):
        for combo in itertools.combinations(range(N_DAYS), size):
            days_sorted = list(combo)
            ok = True
            for i in range(len(days_sorted)):
                gap = (
                    days_sorted[(i + 1) % len(days_sorted)] - days_sorted[i]
                ) % N_DAYS
                if gap == 0:
                    gap = N_DAYS
                if gap > MAX_HOLDING_DAYS:
                    ok = False
                    break
            if ok:
                valid.append(frozenset(combo))
    log.debug(f"Valid schedules: {len(valid)} (min freq {min_freq})")
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised waiting-time helper (shared by Daganzo & ML matrices)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_wait_mx(
    sched_active: np.ndarray,
    schedules: list[frozenset[int]],
    daily_demand: np.ndarray,
    daily_b2c: np.ndarray,
    daily_b2b: np.ndarray,
    fast_share_b2c: float,
    fast_share_b2b: float,
) -> np.ndarray:
    """Vectorised waiting-time matrix: shape (n_plz, n_sched).

    For each (PLZ, schedule) pair, computes the demand-weighted average
    number of waiting days for batch (non-express) parcels that arrive
    on non-delivery days.
    """
    n_sched = len(schedules)

    # Wait-days per (schedule, day): distance to next delivery day
    wait_per_day_mx = np.zeros((n_sched, N_DAYS), dtype=np.float64)
    for si, sched in enumerate(schedules):
        for d in range(N_DAYS):
            if d in sched:
                continue
            w = 0
            check = (d + 1) % N_DAYS
            while check != d:
                w += 1
                if check in sched:
                    break
                check = (check + 1) % N_DAYS
            wait_per_day_mx[si, d] = w

    # Express parcels per day (removed from batch waiting)
    expr_day = (
        np.round(daily_b2c * fast_share_b2c)
        + np.round(daily_b2b * fast_share_b2b)
    )  # (n_plz, N_DAYS)

    # Batch demand = total - express (on non-delivery days; delivery-day
    # contributions are zeroed by wait_per_day_mx == 0 on those days)
    batch_demand = daily_demand - expr_day  # (n_plz, N_DAYS)

    # Total parcels per PLZ (all days, schedule-independent)
    total_p = daily_demand.sum(axis=1)  # (n_plz,)

    # Weighted wait: (n_plz, 1, N_DAYS) * (1, n_sched, N_DAYS) → sum axis=2
    total_w = (batch_demand[:, None, :] * wait_per_day_mx[None, :, :]).sum(axis=2)

    return total_w / np.maximum(1.0, total_p[:, None])


# ─────────────────────────────────────────────────────────────────────────────
# Cost / vehicle matrix construction
# ─────────────────────────────────────────────────────────────────────────────

def build_cost_matrices(
    plz_keys: list[str],
    plz_data: dict,
    schedules: list[frozenset[int]],
    cal: CalibratedDaganzo,
    fast_share_b2c: float = FAST_SHARE_B2C,
    fast_share_b2b: float = FAST_SHARE_B2B,
    plz_corrections_override: dict | None = None,
) -> dict:
    """Build vectorised cost, vehicle and waiting-time matrices.

    Returns a dict with keys:
        dd_cost_mx    (n_plz, n_sched)         delivery-day costs only
        cost_3d       (n_plz, n_sched, N_DAYS)  per-day costs
        veh_3d        (n_plz, n_sched, N_DAYS)  per-day vehicles
        wait_mx       (n_plz, n_sched)          avg waiting days
        raw_express   (n_plz, N_DAYS)           express demand per PLZ/day
        expr_stops    (n_plz, N_DAYS)           express stops per PLZ/day
        area_arr      (n_plz,)
        hd_arr        (n_plz,)
        corr_arr      (n_plz,)
        sched_active  (n_sched, N_DAYS)
        daily_demand  (n_plz, N_DAYS)
    """
    n_plz = len(plz_keys)
    n_sched = len(schedules)
    params = cal.params
    plz_corrections = plz_corrections_override if plz_corrections_override is not None else cal.plz_corrections
    use_jabali = cal.use_jabali

    # 1) Source-day accumulation matrices
    S_all = np.zeros((n_sched, N_DAYS, N_DAYS), dtype=np.float64)
    sched_active = np.zeros((n_sched, N_DAYS), dtype=bool)
    for si, sched in enumerate(schedules):
        ds = sorted(sched)
        for dd in ds:
            sched_active[si, dd] = True
            for d in get_source_days(dd, ds):
                S_all[si, dd, d] = 1.0
    n_source = S_all.sum(axis=2)
    non_delivery = ~sched_active

    NDD = S_all * non_delivery[:, None, :].astype(np.float64)

    # 2) Per-PLZ feature vectors
    daily_b2c = np.zeros((n_plz, N_DAYS), dtype=np.float64)
    daily_b2b = np.zeros((n_plz, N_DAYS), dtype=np.float64)
    daily_demand = np.zeros((n_plz, N_DAYS), dtype=np.float64)
    area_arr = np.empty(n_plz, dtype=np.float64)
    hd_arr = np.empty(n_plz, dtype=np.float64)
    spd_arr = np.empty(n_plz, dtype=np.float64)
    tp_arr = np.empty(n_plz, dtype=np.float64)
    corr_arr = np.empty(n_plz, dtype=np.float64)

    for pi, pc in enumerate(plz_keys):
        pd_ = plz_data[pc]
        for d in range(N_DAYS):
            b2c_d = pd_["b2c"].get(d, 0)
            b2b_d = pd_["b2b"].get(d, 0)
            daily_b2c[pi, d] = b2c_d
            daily_b2b[pi, d] = b2b_d
            daily_demand[pi, d] = b2c_d + b2b_d
        area_arr[pi] = max(0.01, pd_["area_km2"])
        hd_arr[pi] = pd_["hub_dist_km"]
        spd_arr[pi] = pd_["n_stops_per_day"]
        tp_arr[pi] = pd_["total_points"]
        corr_arr[pi] = plz_corrections.get(pc, 1.0)

    fast_share_blend = 0.5 * fast_share_b2c + 0.5 * fast_share_b2b

    # 3) Delivery-day demand with express subtraction
    shifted_raw = np.einsum("sdk,pk->psd", S_all, daily_demand)
    express_sub_b2c = np.einsum("sdk,pk->psd", NDD, daily_b2c)
    express_sub_b2b = np.einsum("sdk,pk->psd", NDD, daily_b2b)
    express_sub = (
        np.round(express_sub_b2c * fast_share_b2c)
        + np.round(express_sub_b2b * fast_share_b2b)
    )
    shifted_dd = np.maximum(0, shifted_raw - express_sub)

    # 4) Express-only demand on non-delivery days
    express_demand = (
        np.round(daily_b2c[:, None, :] * fast_share_b2c)
        + np.round(daily_b2b[:, None, :] * fast_share_b2b)
    )
    express_demand = express_demand * non_delivery[None, :, :].astype(np.float64)

    # 5) Combined demand & stops
    combined_demand = (
        shifted_dd * sched_active[None, :, :].astype(np.float64) + express_demand
    )
    # FIX 2026-05-27: dd_stops must scale with willing fraction.
    # Old (buggy): dd_stops = stops_per_day × n_source (share-independent — overestimated
    #              stops at low share since not all source-day customers actually
    #              consolidate when share<1).
    # New: stops_today + willing × prior_source_stops, capped at total customers.
    # At share=0  (willing_blend=0): dd_stops = stops_per_day  (today only)
    # At share=1  (willing_blend=1): dd_stops = stops_per_day × n_source  (matches training agg_k)
    willing_blend = 1.0 - fast_share_blend
    dd_stops = np.minimum(
        spd_arr[:, None, None] * (1.0 + willing_blend * (n_source[None, :, :] - 1.0)),
        tp_arr[:, None, None],
    )
    ndd_stops = np.maximum(1.0, spd_arr[:, None, None] * fast_share_blend)
    combined_stops = (
        dd_stops * sched_active[None, :, :].astype(np.float64)
        + ndd_stops * non_delivery[None, :, :].astype(np.float64)
    )

    # 6) Active mask
    active = combined_demand > 0
    n_active = int(active.sum())
    if n_active == 0:
        return {
            "dd_cost_mx": np.zeros((n_plz, n_sched)),
            "cost_3d": np.zeros((n_plz, n_sched, N_DAYS)),
            "veh_3d": np.zeros((n_plz, n_sched, N_DAYS)),
            "wait_mx": np.zeros((n_plz, n_sched)),
            "raw_express": np.zeros((n_plz, N_DAYS)),
            "expr_stops": np.zeros((n_plz, N_DAYS)),
            "area_arr": area_arr, "hd_arr": hd_arr, "corr_arr": corr_arr,
            "sched_active": sched_active, "daily_demand": daily_demand,
        }

    # 7) Flatten → single predict_vec call → scatter
    flat_np = combined_demand[active]
    flat_ns = combined_stops[active]
    flat_area = np.broadcast_to(area_arr[:, None, None], combined_demand.shape)[active]
    flat_hd = np.broadcast_to(hd_arr[:, None, None], combined_demand.shape)[active]
    flat_pps = flat_np / np.maximum(1.0, flat_ns)
    flat_ok = (flat_np > 0) & (flat_ns > 0)

    flat_cost, flat_nr, _ = predict_vec(
        params, flat_np, flat_ns, flat_area, flat_hd, flat_pps, flat_ok,
        use_jabali=use_jabali,
    )

    flat_corr = np.broadcast_to(corr_arr[:, None, None], combined_demand.shape)[active]
    flat_cost *= flat_corr

    cost_3d = np.zeros_like(combined_demand)
    cost_3d[active] = flat_cost
    veh_3d = np.zeros(combined_demand.shape, dtype=np.float64)
    veh_3d[active] = np.maximum(1, flat_nr)

    # 8) Waiting-time matrix (vectorised)
    wait_mx = _compute_wait_mx(
        sched_active, schedules, daily_demand,
        daily_b2c, daily_b2b, fast_share_b2c, fast_share_b2b,
    )

    # Delivery-day-only cost matrix
    dd_cost_mx = (cost_3d * sched_active[None, :, :].astype(np.float64)).sum(axis=2)

    # Raw express arrays (schedule-independent, for SA hub-bundled computation)
    raw_express = (
        np.round(daily_b2c * fast_share_b2c) + np.round(daily_b2b * fast_share_b2b)
    )
    expr_stops = np.maximum(1.0, spd_arr[:, None] * fast_share_blend * np.ones((1, N_DAYS)))

    return {
        "dd_cost_mx": dd_cost_mx,
        "cost_3d": cost_3d,
        "veh_3d": veh_3d,
        "wait_mx": wait_mx,
        "raw_express": raw_express,
        "expr_stops": expr_stops,
        "area_arr": area_arr,
        "hd_arr": hd_arr,
        "corr_arr": corr_arr,
        "params": params,
        "use_jabali": use_jabali,
        "sched_active": sched_active,
        "daily_demand": daily_demand,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hub-bundled express cost (used by SA)
# ─────────────────────────────────────────────────────────────────────────────

def _hub_express_day(
    hi: int, d: int, chosen: np.ndarray,
    hub_plz_list: list[np.ndarray],
    schedules: list[frozenset[int]],
    raw_express: np.ndarray,
    expr_stops: np.ndarray,
    area_arr: np.ndarray,
    hd_arr: np.ndarray,
    corr_arr: np.ndarray,
    params: np.ndarray,
    use_jabali: bool = False,
    express_scale: float = 1.0,
    sched_active: np.ndarray | None = None,
) -> float:
    """Hub-level express cost for one day (Daganzo proxy), vectorised.

    Uses NumPy boolean masking instead of a Python for-loop over PLZs.
    ``sched_active`` (n_sched, N_DAYS) is an optional pre-computed mask;
    if not provided, the function falls back to per-PLZ ``d in schedule``
    checks.
    """
    h_ps = hub_plz_list[hi]

    # Boolean mask: PLZs whose chosen schedule does NOT include day d
    if sched_active is not None:
        is_non_delivery = ~sched_active[chosen[h_ps], d]
    else:
        is_non_delivery = np.array(
            [d not in schedules[int(chosen[pi])] for pi in h_ps],
            dtype=bool,
        )

    # Express demand on this day for those PLZs
    expr_demand = raw_express[h_ps, d]
    mask = is_non_delivery & (expr_demand > 0)

    if not mask.any():
        return 0.0

    active_ps = h_ps[mask]
    tot_dem = float(expr_demand[mask].sum())
    tot_stp = float(expr_stops[active_ps, d].sum())
    tot_area = float(area_arr[active_ps].sum())
    corr_w = float(corr_arr[active_ps].sum())
    n_contr = int(mask.sum())

    avg_c = corr_w / n_contr if n_contr > 0 else 1.0
    pps = tot_dem / max(1.0, tot_stp)
    hub_hd = hd_arr[h_ps[0]]

    cc, _, _ = predict_vec(
        params,
        np.array([tot_dem]), np.array([tot_stp]),
        np.array([tot_area]), np.array([hub_hd]),
        np.array([pps]), np.array([True]),
        use_jabali=use_jabali,
    )
    return cc[0] * avg_c * express_scale


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
# Step 2: Fleet balancing post-processing (swap-based)
# ─────────────────────────────────────────────────────────────────────────────

def _daily_fleet_per_hub(
    chosen: np.ndarray,
    plz_hub_arr: np.ndarray,
    hub_plz_list: list[np.ndarray],
    veh_3d: np.ndarray,
    schedules: list[frozenset[int]],
) -> np.ndarray:
    """Compute daily vehicle count per hub: shape (n_hubs, N_DAYS)."""
    n_hubs = len(hub_plz_list)
    plz_veh = veh_3d[np.arange(len(chosen)), chosen, :]  # (n_plz, N_DAYS)
    fleet = np.zeros((n_hubs, N_DAYS), dtype=np.float64)
    np.add.at(fleet, plz_hub_arr, plz_veh)
    return fleet


def _fleet_imbalance(fleet: np.ndarray) -> float:
    """Sum of per-hub range (max - min daily vehicles)."""
    return float(np.sum(fleet.max(axis=1) - fleet.min(axis=1)))


def balance_fleet_per_hub(
    sa_result: dict,
    plz_keys: list[str],
    plz_hub_arr: np.ndarray,
    hub_plz_list: list[np.ndarray],
    matrices: dict,
    schedules: list[frozenset[int]],
    cost_budget_pct: float = 1.0,
    max_swaps: int = FLEET_BALANCE_MAX_SWAPS,
    seed: int = SA_SEED + 1,
    express_scale: float = 1.0,
) -> dict:
    """Swap-based fleet balancing per hub (Step 2).

    For each hub, try swapping a PLZ to a different schedule that
    reduces the fleet range (max - min daily vehicles) while keeping
    the cost increase within *cost_budget_pct* % of SA Step 1 cost.

    Parameters
    ----------
    cost_budget_pct : float
        Maximum allowed cost increase as % of SA best cost.

    Returns
    -------
    dict with: chosen, cost, imbalance_before, imbalance_after,
               schedules_per_plz, swaps_made.
    """
    dd_cost_mx = matrices["dd_cost_mx"]
    veh_3d = matrices["veh_3d"]
    raw_express = matrices["raw_express"]
    expr_stops = matrices["expr_stops"]
    area_arr = matrices["area_arr"]
    hd_arr = matrices["hd_arr"]
    corr_arr = matrices["corr_arr"]
    params = matrices["params"]
    use_jabali = matrices["use_jabali"]
    sched_active = matrices["sched_active"]
    n_plz = len(plz_keys)
    n_sched = len(schedules)

    chosen = sa_result["chosen"].copy()
    sa_cost = sa_result["best_cost"]
    max_cost = sa_cost * (1 + cost_budget_pct / 100.0)

    cur_dd_cost = float(dd_cost_mx[np.arange(n_plz), chosen].sum())
    ecache = np.zeros((len(hub_plz_list), N_DAYS))
    for hi in range(len(hub_plz_list)):
        for d in range(N_DAYS):
            ecache[hi, d] = _hub_express_day(
                hi, d, chosen, hub_plz_list, schedules,
                raw_express, expr_stops, area_arr, hd_arr, corr_arr,
                params, use_jabali, express_scale,
                sched_active=sched_active,
            )
    cur_cost = cur_dd_cost + ecache.sum()

    fleet = _daily_fleet_per_hub(chosen, plz_hub_arr, hub_plz_list, veh_3d, schedules)
    imbalance_before = _fleet_imbalance(fleet)

    rng = np.random.default_rng(seed)
    swaps_made = 0

    pbar = tqdm(
        range(max_swaps), desc="Fleet balancing",
        unit="swap", leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )
    for _ in pbar:
        # Pick the most imbalanced hub
        hub_ranges = fleet.max(axis=1) - fleet.min(axis=1)
        hi = int(np.argmax(hub_ranges))
        if hub_ranges[hi] <= 1.0:
            break  # all hubs balanced

        h_ps = hub_plz_list[hi]
        if len(h_ps) == 0:
            continue

        # Pick a random PLZ at this hub
        pi = int(rng.choice(h_ps))
        old_si = int(chosen[pi])
        old_cost_pi = dd_cost_mx[pi, old_si]
        old_days = schedules[old_si]

        # Try all alternative schedules, pick the one that best reduces fleet range
        best_new_si = None
        best_new_range = hub_ranges[hi]
        best_delta = 0.0
        best_expr_vals: dict[int, float] = {}

        for new_si in range(n_sched):
            if new_si == old_si:
                continue
            delta_c = dd_cost_mx[pi, new_si] - old_cost_pi
            delta_total = delta_c

            new_days = schedules[new_si]
            affected = old_days.symmetric_difference(new_days)
            new_expr_vals: dict[int, float] = {}
            if affected:
                chosen[pi] = new_si
                for d in affected:
                    nv = _hub_express_day(
                        hi, d, chosen, hub_plz_list, schedules,
                        raw_express, expr_stops, area_arr, hd_arr, corr_arr,
                        params, use_jabali, express_scale,
                        sched_active=sched_active,
                    )
                    delta_total += nv - ecache[hi, d]
                    new_expr_vals[d] = nv
                chosen[pi] = old_si

            if cur_cost + delta_total > max_cost:
                continue

            # Tentative fleet change
            old_veh = veh_3d[pi, old_si, :]
            new_veh = veh_3d[pi, new_si, :]
            new_fleet_hub = fleet[hi] - old_veh + new_veh
            new_range = float(new_fleet_hub.max() - new_fleet_hub.min())
            if new_range < best_new_range:
                best_new_range = new_range
                best_new_si = new_si
                best_delta = delta_total
                best_expr_vals = new_expr_vals

        if best_new_si is not None:
            fleet[hi] -= veh_3d[pi, old_si, :]
            fleet[hi] += veh_3d[pi, best_new_si, :]
            chosen[pi] = best_new_si
            cur_cost += best_delta
            for d, val in best_expr_vals.items():
                ecache[hi, d] = val
            swaps_made += 1
            pbar.set_postfix_str(f"swaps={swaps_made}, imb={_fleet_imbalance(fleet):.0f}")

    imbalance_after = _fleet_imbalance(fleet)
    log.debug(
        f"Fleet balancing: {swaps_made} swaps, "
        f"imbalance {imbalance_before:.0f} → {imbalance_after:.0f} vehicles, "
        f"cost delta {(cur_cost / sa_cost - 1) * 100:+.2f}%"
    )

    schedules_per_plz = {
        plz_keys[pi]: schedules[int(chosen[pi])] for pi in range(n_plz)
    }

    return {
        "chosen": chosen,
        "cost": cur_cost,
        "imbalance_before": imbalance_before,
        "imbalance_after": imbalance_after,
        "schedules_per_plz": schedules_per_plz,
        "swaps_made": swaps_made,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scenario-level correction factors (two-phase recalibration)
# ─────────────────────────────────────────────────────────────────────────────

def compute_scenario_corrections(
    provider: str,
    df_routes_fixed: pd.DataFrame,
    fixed_schedules: dict[str, set[int]],
    plz_data: dict,
    plz_keys: list[str],
    plz_hub_arr: np.ndarray,
    hub_plz_list: list[np.ndarray],
    schedules: list[frozenset[int]],
    cal: CalibratedDaganzo,
    fast_share_b2c: float = FAST_SHARE_B2C,
    fast_share_b2b: float = FAST_SHARE_B2B,
) -> dict:
    """Compute frequency-adaptive correction factors from VROOM Fixed+Express.

    The Daganzo proxy is calibrated on baseline (6×/week) routes. When
    applied to reduced schedules (2–3×/week) the BHH-sqrt scaling and
    route-count quantisation behave differently — corrections trained
    on daily demand over-/under-predict accumulated demand.

    This function compares Daganzo *raw* predictions (correction=1.0) for
    the Fixed schedule against actual VROOM costs and derives:
      - **per-PLZ DD corrections** — delivery-day cost ratio
      - **per-provider express correction** — single scaling factor

    Parameters
    ----------
    provider : str
        Provider name (for logging).
    df_routes_fixed : DataFrame
        VROOM routes for Fixed+Express scenario (this provider only).
    fixed_schedules : dict
        {plz: set[int]} delivery days from ``build_fixed_schedules``.
    plz_data : dict
        Per-PLZ feature data from ``prepare_plz_data``.
    plz_keys : list[str]
        Sorted PLZ codes.
    plz_hub_arr, hub_plz_list : arrays
        Hub membership from ``build_hub_arrays``.
    schedules : list[frozenset[int]]
        Valid schedules from ``enumerate_valid_schedules``.
    cal : CalibratedDaganzo
        Calibrated model (provides params, use_jabali).
    fast_share_b2c, fast_share_b2b : float
        Express share fractions.

    Returns
    -------
    dict with keys:
        dd_plz_corrections  : dict[str, float] — per-PLZ delivery-day factor
        express_correction  : float             — provider-wide express scale
        df_diagnostic       : DataFrame         — per-PLZ comparison table
    """
    n_hubs = len(hub_plz_list)
    params = cal.params
    use_jabali = cal.use_jabali

    # ── 1) Per-PLZ delivery-day corrections ──────────────────────────
    diag_rows = []
    dd_plz_corrections: dict[str, float] = {}

    for pc in plz_keys:
        if pc not in plz_data or pc not in fixed_schedules:
            dd_plz_corrections[pc] = cal.plz_corrections.get(pc, 1.0)
            continue

        pd_ = plz_data[pc]
        sched = fixed_schedules[pc]
        baseline_corr = cal.plz_corrections.get(pc, 1.0)

        # Daganzo raw prediction for this PLZ under the fixed schedule
        shifted = compute_shifted_demand_plz(
            sched, pd_["b2c"], pd_["b2b"], fast_share_b2c, fast_share_b2b,
        )
        dg_raw_dd = 0.0
        for dd_info in shifted.values():
            if dd_info.get("express_only", False):
                continue
            n_parcels = dd_info["total"]
            if n_parcels <= 0:
                continue
            n_stops = min(
                int(pd_["n_stops_per_day"] * len(dd_info["source_days"])),
                pd_["total_points"],
            )
            if n_stops <= 0:
                continue
            dg_raw_dd += cal.cost(
                n_parcels, n_stops, pd_["area_km2"], pd_["hub_dist_km"],
                correction=1.0,
            )

        # VROOM actual DD cost for this PLZ
        plz_routes = df_routes_fixed[
            (df_routes_fixed["plz"].astype(str) == str(pc))
            & (~df_routes_fixed["is_express"])
        ]
        vroom_dd = plz_routes["cost"].sum() / COST_SCALE if len(plz_routes) > 0 else 0.0

        # Correction factor
        if dg_raw_dd > 0 and vroom_dd > 0:
            sc_corr = vroom_dd / dg_raw_dd
        else:
            sc_corr = baseline_corr

        dd_plz_corrections[pc] = sc_corr

        diag_rows.append({
            "plz": pc,
            "baseline_corr": round(baseline_corr, 4),
            "scenario_corr": round(sc_corr, 4),
            "ratio": round(sc_corr / max(0.01, baseline_corr), 4),
            "dg_raw_dd": round(dg_raw_dd, 1),
            "vroom_dd": round(vroom_dd, 1),
            "vroom_dd_routes": len(plz_routes),
        })

    # ── 2) Per-provider express correction ───────────────────────────
    # Build fixed schedule → chosen index array
    sched_list = list(schedules)
    sched_idx_map = {s: i for i, s in enumerate(sched_list)}
    chosen_fixed = np.zeros(len(plz_keys), dtype=np.int64)
    for pi, pc in enumerate(plz_keys):
        fs = frozenset(fixed_schedules.get(pc, set()))
        chosen_fixed[pi] = sched_idx_map.get(fs, 0)

    # Build raw arrays (without corrections) for express Daganzo
    fast_share_blend = 0.5 * fast_share_b2c + 0.5 * fast_share_b2b
    n_plz = len(plz_keys)
    raw_express = np.zeros((n_plz, N_DAYS), dtype=np.float64)
    expr_stops = np.zeros((n_plz, N_DAYS), dtype=np.float64)
    area_arr = np.zeros(n_plz, dtype=np.float64)
    hd_arr = np.zeros(n_plz, dtype=np.float64)
    corr_ones = np.ones(n_plz, dtype=np.float64)  # no correction → raw

    for pi, pc in enumerate(plz_keys):
        if pc not in plz_data:
            continue
        pd_ = plz_data[pc]
        for d in range(N_DAYS):
            b2c_d = pd_["b2c"].get(d, 0)
            b2b_d = pd_["b2b"].get(d, 0)
            raw_express[pi, d] = round(b2c_d * fast_share_b2c) + round(b2b_d * fast_share_b2b)
        expr_stops[pi, :] = max(1.0, pd_["n_stops_per_day"] * fast_share_blend)
        area_arr[pi] = max(0.01, pd_["area_km2"])
        hd_arr[pi] = pd_["hub_dist_km"]

    # Daganzo raw express cost (summed over all hubs × days)
    # Build sched_active for the schedule list used in this context
    _sa_corr = np.zeros((len(sched_list), N_DAYS), dtype=bool)
    for si, s in enumerate(sched_list):
        for d in s:
            _sa_corr[si, d] = True

    dg_express_total = 0.0
    for hi in range(n_hubs):
        for d in range(N_DAYS):
            dg_express_total += _hub_express_day(
                hi, d, chosen_fixed, hub_plz_list, sched_list,
                raw_express, expr_stops, area_arr, hd_arr, corr_ones,
                params, use_jabali, express_scale=1.0,
                sched_active=_sa_corr,
            )

    # VROOM actual express cost
    vroom_express_total = (
        df_routes_fixed[df_routes_fixed["is_express"]]["cost"].sum() / COST_SCALE
    )

    if dg_express_total > 0 and vroom_express_total > 0:
        express_correction = vroom_express_total / dg_express_total
    else:
        express_correction = 1.0

    # ── 3) Diagnostics ───────────────────────────────────────────────
    df_diag = pd.DataFrame(diag_rows)
    if len(df_diag) > 0:
        corr_vals = df_diag["scenario_corr"].values
        bl_vals = df_diag["baseline_corr"].values
        log.info(
            f"  [{provider}] Scenario corrections computed:\n"
            f"    DD corrections:  n={len(corr_vals)}, "
            f"median={np.median(corr_vals):.3f}, "
            f"mean={np.mean(corr_vals):.3f}, "
            f"std={np.std(corr_vals):.3f}, "
            f"range=[{np.min(corr_vals):.3f}, {np.max(corr_vals):.3f}]\n"
            f"    Baseline corrs:  "
            f"median={np.median(bl_vals):.3f}, "
            f"mean={np.mean(bl_vals):.3f}\n"
            f"    Express factor:  {express_correction:.4f}"
            f"  (Daganzo_raw={dg_express_total:,.0f}, VROOM={vroom_express_total:,.0f})"
        )

    return {
        "dd_plz_corrections": dd_plz_corrections,
        "express_correction": express_correction,
        "df_diagnostic": df_diag,
        "dg_express_raw": dg_express_total,
        "vroom_express": vroom_express_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-schedule helper
# ─────────────────────────────────────────────────────────────────────────────

def build_fixed_schedules(
    plz_keys: list[str],
    carrier: str = "DHL",
) -> dict[str, set[int]]:
    """Build fixed delivery schedules for all PLZ from CARRIER_DAYS."""
    days = CARRIER_FIXED_INDICES.get(carrier) or CARRIER_FIXED_INDICES["DHL"]
    return {pc: set(days) for pc in plz_keys}


# ─────────────────────────────────────────────────────────────────────────────
# Hub assignment arrays (utility for SA setup)
# ─────────────────────────────────────────────────────────────────────────────

def build_hub_arrays(
    plz_keys: list[str],
    df_assignments: pd.DataFrame,
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    """Build PLZ-to-hub index array and per-hub PLZ lists.

    Returns
    -------
    (plz_hub_arr, hub_plz_list, hub_names)
    """
    plz_to_hub: dict[str, str] = {}
    for pc in plz_keys:
        hr = df_assignments[df_assignments["plz"] == pc]
        if len(hr) > 0:
            plz_to_hub[pc] = hr.iloc[0]["hub_name"]
    hub_names = sorted(set(plz_to_hub.values()))
    hub_idx = {h: i for i, h in enumerate(hub_names)}
    n_hubs = len(hub_names)
    plz_hub_arr = np.array(
        [hub_idx.get(plz_to_hub.get(p, ""), 0) for p in plz_keys]
    )
    hub_plz_list = [np.where(plz_hub_arr == hi)[0] for hi in range(n_hubs)]
    return plz_hub_arr, hub_plz_list, hub_names


# ═══════════════════════════════════════════════════════════════════════════
# ML-based cost matrices and SA optimisation
# ═══════════════════════════════════════════════════════════════════════════

def build_cost_matrices_ml(
    plz_keys: list[str],
    plz_data: dict,
    schedules: list[frozenset[int]],
    ml_predictor,
    provider: str,
    plz_day_coords: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    hub_coords_by_plz: dict[str, tuple[float, float]],
    fast_share_b2c: float = FAST_SHARE_B2C,
    fast_share_b2b: float = FAST_SHARE_B2B,
) -> dict:
    """Build cost matrices using MLP Ensemble predictions.

    Same demand logic as ``build_cost_matrices`` but replaces the Daganzo
    ``predict_vec`` with full 25-feature ML prediction.

    Additional parameters
    ---------------------
    ml_predictor : MLCostPredictor
        Trained MLP Ensemble predictor.
    provider : str
        Provider name (for ``provider_idx`` feature).
    plz_day_coords : dict
        ``{plz_code: {day: (lon, lat, per_stop_demand)}}`` — WGS-84.
    hub_coords_by_plz : dict
        ``{plz_code: (hub_lon, hub_lat)}`` in WGS-84.
    """
    from batch_delivery.features import compute_all_features, ALL_COLS

    n_plz = len(plz_keys)
    n_sched = len(schedules)

    # ── 1) Source-day accumulation matrices ───────────────────────────
    S_all = np.zeros((n_sched, N_DAYS, N_DAYS), dtype=np.float64)
    sched_active = np.zeros((n_sched, N_DAYS), dtype=bool)
    for si, sched in enumerate(schedules):
        ds = sorted(sched)
        for dd in ds:
            sched_active[si, dd] = True
            for d in get_source_days(dd, ds):
                S_all[si, dd, d] = 1.0
    n_source = S_all.sum(axis=2)
    non_delivery = ~sched_active
    NDD = S_all * non_delivery[:, None, :].astype(np.float64)

    # ── 2) Per-PLZ arrays ─────────────────────────────────────────────
    daily_b2c = np.zeros((n_plz, N_DAYS), dtype=np.float64)
    daily_b2b = np.zeros((n_plz, N_DAYS), dtype=np.float64)
    daily_demand = np.zeros((n_plz, N_DAYS), dtype=np.float64)
    area_arr = np.empty(n_plz, dtype=np.float64)
    hd_arr = np.empty(n_plz, dtype=np.float64)
    spd_arr = np.empty(n_plz, dtype=np.float64)
    tp_arr = np.empty(n_plz, dtype=np.float64)

    for pi, pc in enumerate(plz_keys):
        pd_ = plz_data[pc]
        for d in range(N_DAYS):
            b2c_d = pd_["b2c"].get(d, 0)
            b2b_d = pd_["b2b"].get(d, 0)
            daily_b2c[pi, d] = b2c_d
            daily_b2b[pi, d] = b2b_d
            daily_demand[pi, d] = b2c_d + b2b_d
        area_arr[pi] = max(0.01, pd_["area_km2"])
        hd_arr[pi] = pd_["hub_dist_km"]
        spd_arr[pi] = pd_["n_stops_per_day"]
        tp_arr[pi] = pd_["total_points"]

    # FIX 2026-05-27: per-PLZ b2c share for weighted blend (was hardcoded 50/50).
    # Effective non-willing fraction at a PLZ depends on its actual b2c/b2b mix:
    #   B2C-heavy PLZ (e.g. 91/9): with B2B preferring waits, local willing ratio
    #                              is dominated by B2C (less batching benefit).
    #   B2B-heavy PLZ (e.g. 33/67): local willing ratio is high (more batching).
    _plz_total_weekly = daily_demand.sum(axis=1)
    plz_b2c_share_arr = np.where(
        _plz_total_weekly > 0,
        daily_b2c.sum(axis=1) / np.maximum(_plz_total_weekly, 1.0),
        0.5,
    )
    fast_share_blend_arr = (
        plz_b2c_share_arr * fast_share_b2c
        + (1.0 - plz_b2c_share_arr) * fast_share_b2b
    )  # shape (n_plz,)

    # ── 3) Delivery-day demand with express subtraction ───────────────
    shifted_raw = np.einsum("sdk,pk->psd", S_all, daily_demand)
    express_sub_b2c = np.einsum("sdk,pk->psd", NDD, daily_b2c)
    express_sub_b2b = np.einsum("sdk,pk->psd", NDD, daily_b2b)
    express_sub = (
        np.round(express_sub_b2c * fast_share_b2c)
        + np.round(express_sub_b2b * fast_share_b2b)
    )
    shifted_dd = np.maximum(0, shifted_raw - express_sub)

    # ── 4) Express-only demand on non-delivery days ───────────────────
    express_demand = (
        np.round(daily_b2c[:, None, :] * fast_share_b2c)
        + np.round(daily_b2b[:, None, :] * fast_share_b2b)
    )
    express_demand = express_demand * non_delivery[None, :, :].astype(np.float64)

    # ── 5) Combined demand & stops ────────────────────────────────────
    combined_demand = (
        shifted_dd * sched_active[None, :, :].astype(np.float64) + express_demand
    )
    # FIX 2026-05-27: dd_stops scales with PER-PLZ willing fraction.
    # At share=0 (willing_blend=0): dd_stops = stops_per_day (today only)
    # At share=1 (willing_blend=1): dd_stops = stops_per_day × n_source (matches training agg_k)
    willing_blend_arr = (1.0 - fast_share_blend_arr)[:, None, None]  # (n_plz, 1, 1)
    dd_stops = np.minimum(
        spd_arr[:, None, None] * (1.0 + willing_blend_arr * (n_source[None, :, :] - 1.0)),
        tp_arr[:, None, None],
    )
    ndd_stops = np.maximum(
        1.0, spd_arr[:, None, None] * fast_share_blend_arr[:, None, None]
    )
    combined_stops = (
        dd_stops * sched_active[None, :, :].astype(np.float64)
        + ndd_stops * non_delivery[None, :, :].astype(np.float64)
    )

    # ── 6) Active mask ────────────────────────────────────────────────
    active = combined_demand > 0
    n_active = int(active.sum())
    if n_active == 0:
        empty = {
            "dd_cost_mx": np.zeros((n_plz, n_sched)),
            "cost_3d": np.zeros((n_plz, n_sched, N_DAYS)),
            "veh_3d": np.zeros((n_plz, n_sched, N_DAYS)),
            "wait_mx": np.zeros((n_plz, n_sched)),
            "raw_express": np.zeros((n_plz, N_DAYS)),
            "expr_stops": np.zeros((n_plz, N_DAYS)),
            "area_arr": area_arr, "hd_arr": hd_arr,
            "sched_active": sched_active, "daily_demand": daily_demand,
        }
        return empty

    # ── 7) ML prediction (vectorised feature construction) ──────────
    # NOTE: compute_tier2_features, ALL_COLS, TIER2_COLS, _PROVIDER_IDX, and
    # get_source_days are all imported at module scope (see top of file).
    # Do NOT add function-local imports here — Python's scoping rules would
    # then treat get_source_days as a local variable throughout this function,
    # causing UnboundLocalError at line 1087 in the earlier section.
    n_t2 = len(TIER2_COLS)

    # Single-day Tier 2 (express / non-delivery cells)
    tier2_mx = np.zeros((n_plz, N_DAYS, n_t2), dtype=np.float64)
    for pi in range(n_plz):
        pc = plz_keys[pi]
        hlon, hlat = hub_coords_by_plz.get(pc, (9.73, 52.38))
        pc_coords = plz_day_coords.get(pc, {})
        for d in range(N_DAYS):
            coords = pc_coords.get(d)
            if coords is not None and len(coords[0]) > 0:
                t2 = compute_tier2_features(
                    coords[0], coords[1], hlon, hlat, coords[2],
                )
                for j, col in enumerate(TIER2_COLS):
                    tier2_mx[pi, d, j] = t2[col]

    # FW6.A FIX 2026-05-26 (Audit E BUG-11 + BUG-12): For DELIVERY cells,
    # Tier-2 (geometry) AND Tier-3 (demand_std/max_stop_demand) must both
    # come from the UNION of source-day stops (deduplicated by lon/lat with
    # summed psd). This mirrors sweep/perturb.py:aggregate_days. Without
    # this, production predictions systematically under-estimate batched
    # delivery costs by ~8% (median) and ~30% on multi-polygon urban
    # clusters (30159/30167/30449), driving the saving-table gap.
    # Cache by (pi, si, dd) tuple — only active multi-source delivery cells.
    tier_delivery_cache: dict[tuple[int, int, int], dict] = {}
    for si, sched in enumerate(schedules):
        sched_days = sorted(sched)
        for dd in sched_days:
            src_days = get_source_days(dd, sched_days)
            if len(src_days) <= 1:
                # Single-source: identical to express cache, skip
                continue
            for pi in range(n_plz):
                pc = plz_keys[pi]
                hlon, hlat = hub_coords_by_plz.get(pc, (9.73, 52.38))
                pc_coords = plz_day_coords.get(pc, {})
                all_lons, all_lats, all_psd = [], [], []
                for sd in src_days:
                    c = pc_coords.get(sd)
                    if c is not None and len(c[0]) > 0:
                        all_lons.append(c[0])
                        all_lats.append(c[1])
                        all_psd.append(c[2])
                if not all_lons:
                    log.warning(
                        "FW6.A cache: no source-day coords for "
                        "(plz=%s, schedule=%s, delivery_day=%d)",
                        pc, sched_days, dd,
                    )
                    continue
                u_lon = np.concatenate(all_lons)
                u_lat = np.concatenate(all_lats)
                u_psd = np.concatenate(all_psd)
                # Dedupe by (lon, lat) — same customer appearing on multiple
                # source days collapses to one stop with summed demand
                pts = pd.DataFrame({"lon": u_lon, "lat": u_lat, "psd": u_psd})
                pts = pts.groupby(["lon", "lat"], as_index=False)["psd"].sum()
                ded_lon = pts["lon"].values
                ded_lat = pts["lat"].values
                ded_psd = pts["psd"].values
                t2 = compute_tier2_features(ded_lon, ded_lat, hlon, hlat, ded_psd)
                tier_delivery_cache[(pi, si, dd)] = {
                    "tier2": np.array(
                        [t2[c] for c in TIER2_COLS], dtype=np.float64
                    ),
                    "psd_std": float(ded_psd.std()) if len(ded_psd) > 1 else 0.0,
                    "psd_max": float(ded_psd.max()) if len(ded_psd) > 0 else 0.0,
                }

    # B2C share per PLZ (for Tier 3)
    plz_total = daily_demand.sum(axis=1)
    plz_b2c_share = np.where(plz_total > 0, daily_b2c.sum(axis=1) / plz_total, 0.5)

    # Pre-compute PSD statistics per (PLZ, day)
    _has_psd = np.zeros((n_plz, N_DAYS), dtype=bool)
    _psd_std = np.zeros((n_plz, N_DAYS), dtype=np.float64)
    _psd_max = np.zeros((n_plz, N_DAYS), dtype=np.float64)
    for pi in range(n_plz):
        pc_coords = plz_day_coords.get(plz_keys[pi], {})
        for d in range(N_DAYS):
            cd = pc_coords.get(d)
            if cd is not None and len(cd[2]) > 0:
                _has_psd[pi, d] = True
                _psd_std[pi, d] = float(cd[2].std()) if len(cd[2]) > 1 else 0.0
                _psd_max[pi, d] = float(cd[2].max())

    # Active cell indices
    pi_arr, si_arr, d_arr = np.where(active)
    n_act = len(pi_arr)
    log.info("Computing ML features for %d active cells …", n_act)

    # Build feature matrix directly (25 columns)
    n_feat = len(ALL_COLS)
    feat_mx = np.empty((n_act, n_feat), dtype=np.float64)

    np_raw = combined_demand[pi_arr, si_arr, d_arr]
    np_f = np.trunc(np_raw).astype(np.float64)     # match int() cast
    ns_f = np.maximum(1.0, np.trunc(combined_stops[pi_arr, si_arr, d_arr]))
    area_f = np.maximum(0.01, area_arr[pi_arr])
    hd_f = hd_arr[pi_arr]

    # Tier 1 (cols 0–7)
    feat_mx[:, 0] = np_f                               # n_parcels
    feat_mx[:, 1] = ns_f                               # n_stops
    feat_mx[:, 2] = area_f                             # area_km2
    feat_mx[:, 3] = hd_f                               # hub_dist_km
    feat_mx[:, 4] = np_f / ns_f                        # parcels_per_stop
    feat_mx[:, 5] = np_f / VEHICLE_CAPACITY            # load_factor
    feat_mx[:, 6] = np.ceil(np_f / VEHICLE_CAPACITY)   # min_vehicles
    feat_mx[:, 7] = np_f / area_f                      # parcels_per_km2

    # Tier 2 (cols 8–17): per-day single-source baseline. For multi-source
    # delivery cells, FIX 2026-05-27 blends with the union-of-source-days cache
    # by willing fraction so share-dependent batching is reflected geometrically.
    # At share=0: pure single-day geometry (no consolidation happens)
    # At share=1: pure union geometry (matches training agg_k semantics)
    feat_mx[:, 8:8 + n_t2] = tier2_mx[pi_arr, d_arr, :]
    # Need per-active willing_blend now (defined below for tier 3); compute here.
    _wb_active = 1.0 - fast_share_blend_arr[pi_arr]
    for k in range(n_act):
        if sched_active[si_arr[k], d_arr[k]]:
            key = (int(pi_arr[k]), int(si_arr[k]), int(d_arr[k]))
            cached = tier_delivery_cache.get(key)
            if cached is not None:
                wb = float(_wb_active[k])
                single_t2 = tier2_mx[pi_arr[k], d_arr[k], :]
                feat_mx[k, 8:8 + n_t2] = (1.0 - wb) * single_t2 + wb * cached["tier2"]

    # Tier 3 (cols 18–24)
    # FIX 2026-05-27: delivery_frequency scales with PER-PLZ willing fraction.
    # At share=0 (no batching), every tour delivers today's parcels only → freq=1
    # At share=1 (full batching), freq = n_source (matches training agg_k)
    # For non-delivery days, freq = 1 (single-day residual semantics).
    n_src_active_freq = n_source[si_arr, d_arr]
    # Per-PLZ willing fraction (consistent with stops calc above)
    willing_blend_pi = (1.0 - fast_share_blend_arr[pi_arr])
    freq_f = (1.0 + willing_blend_pi
              * np.maximum(0.0, n_src_active_freq - 1.0)).astype(np.float64)
    freq_f = np.maximum(1.0, freq_f)
    b2c_int = np.trunc(np_f * plz_b2c_share[pi_arr])
    total_int = np.maximum(1.0, np_f)
    min_veh = np.maximum(1.0, np.ceil(np_f / VEHICLE_CAPACITY))
    hp = _has_psd[pi_arr, d_arr]

    feat_mx[:, 18] = b2c_int / total_int                                # b2c_share
    # FIX 2026-05-27: per-stop demand stats scale with willing fraction
    # (not blindly with n_source as before).
    # At share=0: scale=1 (per-day stats unchanged, no batching happens)
    # At share=1: scale=n_source (matches training agg_k semantics)
    n_src_active = n_source[si_arr, d_arr]
    willing_blend_active = 1.0 - fast_share_blend_arr[pi_arr]
    psd_scale = 1.0 + willing_blend_active * np.maximum(0.0, n_src_active - 1.0)
    feat_mx[:, 19] = np.where(hp, _psd_std[pi_arr, d_arr] * psd_scale, 0.0)   # demand_std
    feat_mx[:, 20] = np.where(hp, _psd_max[pi_arr, d_arr] * psd_scale, np_f)  # max_stop_demand
    feat_mx[:, 21] = np_f / (min_veh * VEHICLE_CAPACITY)               # demand_cap_ratio
    feat_mx[:, 22] = float(_PROVIDER_IDX.get(provider, 0))             # provider_idx
    feat_mx[:, 23] = d_arr.astype(np.float64)                          # day_idx
    feat_mx[:, 24] = freq_f                                            # delivery_frequency

    # FW6.A FIX 2026-05-26 (BUG-12): Union-cache psd values reflect FULL batching
    # (deduped union of source-day customers). FIX 2026-05-27: blend with the
    # single-day baseline by willing fraction so share<1 is consistent.
    #   blended = (1 - willing_blend) × single_day + willing_blend × union_cache
    # At share=0: pure single-day (no batching happens)
    # At share=1: pure union-cache (matches training agg_k)
    for k in range(n_act):
        if sched_active[si_arr[k], d_arr[k]]:
            key = (int(pi_arr[k]), int(si_arr[k]), int(d_arr[k]))
            cached = tier_delivery_cache.get(key)
            if cached is not None:
                wb = float(willing_blend_active[k])
                single_psd_std = float(_psd_std[pi_arr[k], d_arr[k]])
                single_psd_max = float(_psd_max[pi_arr[k], d_arr[k]])
                feat_mx[k, 19] = (1.0 - wb) * single_psd_std + wb * cached["psd_std"]
                feat_mx[k, 20] = (1.0 - wb) * single_psd_max + wb * cached["psd_max"]

    df_feats = pd.DataFrame(feat_mx, columns=ALL_COLS)
    ml_costs = ml_predictor.predict(df_feats)

    cost_3d = np.zeros_like(combined_demand)
    veh_3d = np.zeros(combined_demand.shape, dtype=np.float64)
    cost_3d[pi_arr, si_arr, d_arr] = ml_costs
    veh_3d[pi_arr, si_arr, d_arr] = np.maximum(1, np.ceil(np_raw / VEHICLE_CAPACITY))

    log.info("ML prediction complete: %d cells, cost range [%.0f, %.0f]",
             len(ml_costs), ml_costs.min(), ml_costs.max())

    # ── 8) Waiting-time matrix (vectorised) ─────────────────────────
    wait_mx = _compute_wait_mx(
        sched_active, schedules, daily_demand,
        daily_b2c, daily_b2b, fast_share_b2c, fast_share_b2b,
    )

    # ── 9) DD cost matrix & express arrays ────────────────────────────
    dd_cost_mx = (cost_3d * sched_active[None, :, :].astype(np.float64)).sum(axis=2)
    raw_express = (
        np.round(daily_b2c * fast_share_b2c)
        + np.round(daily_b2b * fast_share_b2b)
    )
    expr_stops = np.maximum(
        1.0, spd_arr[:, None] * fast_share_blend_arr[:, None] * np.ones((1, N_DAYS)),
    )

    # Per-PLZ per-day coordinate arrays for _hub_express_day_ml
    plz_day_lon: list[list[np.ndarray]] = []
    plz_day_lat: list[list[np.ndarray]] = []
    plz_day_psd: list[list[np.ndarray]] = []
    for pi, pc in enumerate(plz_keys):
        pc_coords = plz_day_coords.get(pc, {})
        lons_d, lats_d, psd_d = [], [], []
        for d in range(N_DAYS):
            cd = pc_coords.get(d)
            if cd is not None:
                lons_d.append(cd[0])
                lats_d.append(cd[1])
                psd_d.append(cd[2])
            else:
                lons_d.append(np.array([], dtype=np.float64))
                lats_d.append(np.array([], dtype=np.float64))
                psd_d.append(np.array([], dtype=np.float64))
        plz_day_lon.append(lons_d)
        plz_day_lat.append(lats_d)
        plz_day_psd.append(psd_d)

    hub_lon_arr = np.array([
        hub_coords_by_plz.get(pc, (9.73, 52.38))[0] for pc in plz_keys
    ])
    hub_lat_arr = np.array([
        hub_coords_by_plz.get(pc, (9.73, 52.38))[1] for pc in plz_keys
    ])

    return {
        "dd_cost_mx": dd_cost_mx,
        "cost_3d": cost_3d,
        "veh_3d": veh_3d,
        "wait_mx": wait_mx,
        "raw_express": raw_express,
        "expr_stops": expr_stops,
        "area_arr": area_arr,
        "hd_arr": hd_arr,
        "sched_active": sched_active,
        "daily_demand": daily_demand,
        # ML-specific data for SA express
        "plz_day_lon": plz_day_lon,
        "plz_day_lat": plz_day_lat,
        "plz_day_psd": plz_day_psd,
        "hub_lon_arr": hub_lon_arr,
        "hub_lat_arr": hub_lat_arr,
        "plz_b2c_share": plz_b2c_share,
        "ml_predictor": ml_predictor,
        "provider": provider,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hub-bundled express cost — ML version
# ─────────────────────────────────────────────────────────────────────────────

def _hub_express_day_ml(
    hi: int, d: int, chosen: np.ndarray,
    hub_plz_list: list[np.ndarray],
    schedules: list[frozenset[int]],
    raw_express: np.ndarray,
    expr_stops: np.ndarray,
    matrices: dict,
    express_cache: dict,
    express_scale: float = 1.0,
) -> float:
    """Hub-level express cost for one day using ML prediction (vectorised).

    Aggregates express stops from non-delivering PLZ at the hub, computes
    full 25-feature vector from merged coordinates, and predicts with the
    MLP Ensemble.  Results are cached by ``(hub, day, contributing_plz)``.
    """
    from batch_delivery.features import (
        compute_tier1_features, compute_tier2_features,
        compute_tier3_features, ALL_COLS,
    )

    h_ps = hub_plz_list[hi]

    # Vectorised: identify contributing PLZs via boolean masking
    sched_active = matrices.get("sched_active")
    if sched_active is not None:
        is_non_delivery = ~sched_active[chosen[h_ps], d]
    else:
        is_non_delivery = np.array(
            [d not in schedules[int(chosen[pi])] for pi in h_ps],
            dtype=bool,
        )
    expr_demand = raw_express[h_ps, d]
    mask = is_non_delivery & (expr_demand > 0)

    if not mask.any():
        return 0.0

    contributing = h_ps[mask].tolist()
    tot_dem = float(expr_demand[mask].sum())
    tot_stp = float(expr_stops[h_ps[mask], d].sum())

    cache_key = (hi, d, frozenset(contributing))
    cached = express_cache.get(cache_key)
    if cached is not None:
        return cached * express_scale

    # Merge coordinates from contributing PLZ
    plz_day_lon = matrices["plz_day_lon"]
    plz_day_lat = matrices["plz_day_lat"]
    plz_day_psd = matrices["plz_day_psd"]
    hub_lon_arr = matrices["hub_lon_arr"]
    hub_lat_arr = matrices["hub_lat_arr"]
    plz_b2c_share = matrices["plz_b2c_share"]
    ml_predictor = matrices["ml_predictor"]
    provider = matrices["provider"]
    area_arr = matrices["area_arr"]

    all_lon = [plz_day_lon[pi][d] for pi in contributing if len(plz_day_lon[pi][d]) > 0]
    all_lat = [plz_day_lat[pi][d] for pi in contributing if len(plz_day_lat[pi][d]) > 0]
    all_psd = [plz_day_psd[pi][d] for pi in contributing if len(plz_day_psd[pi][d]) > 0]

    if all_lon:
        merged_lon = np.concatenate(all_lon)
        merged_lat = np.concatenate(all_lat)
        merged_psd = np.concatenate(all_psd)
    else:
        merged_lon = np.array([], dtype=np.float64)
        merged_lat = np.array([], dtype=np.float64)
        merged_psd = np.array([tot_dem])

    # Hub coordinates (from first contributing PLZ)
    hlon = float(hub_lon_arr[contributing[0]])
    hlat = float(hub_lat_arr[contributing[0]])

    # Aggregate area (sum of contributing PLZ areas)
    tot_area = sum(float(area_arr[pi]) for pi in contributing)

    # Weighted B2C share
    dem_by_plz = [raw_express[pi, d] for pi in contributing]
    total = sum(dem_by_plz)
    b2c_share_w = sum(
        plz_b2c_share[pi] * raw_express[pi, d] for pi in contributing
    ) / max(1, total)

    # Compute features
    # FIX 2026-05-25: hub_dist_km=0.0 for express routes (they start at the hub).
    # Matches training-side feature in features/core.py:657 (xpr_hd = 0.0).
    # Previously passed area_arr[contributing[0]] as hub_dist_km (positional-arg
    # bug) which corrupted express cost predictions by ~2 orders of magnitude.
    t1 = compute_tier1_features(int(tot_dem), int(max(1, tot_stp)), tot_area, 0.0)
    if len(merged_lon) > 0:
        t2 = compute_tier2_features(merged_lon, merged_lat, hlon, hlat, merged_psd)
    else:
        from batch_delivery.features import TIER2_COLS
        t2 = {c: 0.0 for c in TIER2_COLS}
    t3 = compute_tier3_features(
        merged_psd if len(merged_psd) > 0 else np.array([tot_dem]),
        int(tot_dem * b2c_share_w), int(tot_dem),
        provider, d, 1,
    )

    feats = {**t1, **t2, **t3}
    base25 = np.array([feats[c] for c in ALL_COLS], dtype=np.float64)
    # Numpy-only single-row predict (avoids DataFrame overhead in hot loop).
    cost = float(ml_predictor.predict_single(base25))

    express_cache[cache_key] = cost
    return cost * express_scale


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


def balance_fleet_per_hub_ml(
    sa_result: dict,
    plz_keys: list[str],
    plz_hub_arr: np.ndarray,
    hub_plz_list: list[np.ndarray],
    matrices: dict,
    schedules: list[frozenset[int]],
    cost_budget_pct: float = 1.0,
    max_swaps: int = FLEET_BALANCE_MAX_SWAPS,
    seed: int = SA_SEED + 1,
    express_scale: float = 1.0,
    penalty_mx: np.ndarray | None = None,
    preserve_frequency: bool = False,
) -> dict:
    """Fleet balancing post-processing using ML express predictions.

    When ``preserve_frequency`` is True, candidate swaps are restricted to
    schedules of the SAME size as the current one (same number of delivery days
    per week). Balancing then only redistributes WHICH days are served, never
    HOW MANY, so the per-area delivery frequency, average wait, and service
    penalty are held at their cost-optimal (init) values. This keeps fleet
    smoothing service-neutral and prevents the balancer from trading service
    quality for the (hub-bundled) routing savings of lower frequencies — the
    cost model used for selection is per-PLZ unbundled, so under bundling lower
    frequencies look cheaper and would otherwise be re-introduced here.

    When ``penalty_mx`` (shape ``(n_plz, n_sched)``,
    ``P * willing_pi * pkts_pi * wait_si``) is supplied, the cost budget is
    enforced on the PENALIZED objective (routing cost + service penalty), the
    same objective the initial selection minimizes. The reported ``cost`` stays
    routing-only (dd + express) so it remains comparable to ``init_cost_eur``.
    Passing ``None`` reproduces the legacy penalty-blind behavior.
    """
    dd_cost_mx = matrices["dd_cost_mx"]
    veh_3d = matrices["veh_3d"]
    raw_express = matrices["raw_express"]
    expr_stops = matrices["expr_stops"]
    n_plz = len(plz_keys)

    chosen = sa_result["chosen"].copy()
    cur_dd_cost = float(dd_cost_mx[np.arange(n_plz), chosen].sum())
    express_pred_cache: dict = {}
    ecache = np.zeros((len(hub_plz_list), N_DAYS))
    for hi in range(len(hub_plz_list)):
        for d in range(N_DAYS):
            ecache[hi, d] = _hub_express_day_ml(
                hi, d, chosen, hub_plz_list, schedules,
                raw_express, expr_stops, matrices,
                express_pred_cache, express_scale,
            )
    cur_cost = cur_dd_cost + ecache.sum()
    # FIX 2026-05-27: budget on TOTAL cost (dd + express), not dd-only.
    # Previously max_cost used sa_result['best_cost'] which was only the
    # delivery-day cost — leading to immediate budget violation since cur_cost
    # always exceeds dd-only by the express residual.
    initial_total_cost = cur_cost
    # FIX 2026-05-29: enforce the budget on the PENALIZED objective (routing +
    # service penalty) so balancing cannot trade service quality for routing cost.
    # cur_cost stays routing-only for reporting; cur_obj drives acceptance.
    use_pen = penalty_mx is not None
    cur_pen = float(penalty_mx[np.arange(n_plz), chosen].sum()) if use_pen else 0.0
    cur_obj = cur_cost + cur_pen
    initial_obj = cur_obj
    max_obj = initial_obj * (1 + cost_budget_pct / 100.0)

    fleet = _daily_fleet_per_hub(chosen, plz_hub_arr, hub_plz_list, veh_3d, schedules)
    imbalance_before = _fleet_imbalance(fleet)

    rng = np.random.default_rng(seed)
    swaps_made = 0

    pbar = tqdm(
        range(max_swaps), desc="Fleet balancing (ML)",
        unit="swap", leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )
    for _ in pbar:
        hub_ranges = fleet.max(axis=1) - fleet.min(axis=1)
        hi = int(np.argmax(hub_ranges))
        if hub_ranges[hi] <= 1.0:
            break

        h_ps = hub_plz_list[hi]
        if len(h_ps) == 0:
            continue

        pi = int(rng.choice(h_ps))
        old_si = int(chosen[pi])
        old_cost_pi = dd_cost_mx[pi, old_si]
        old_days = schedules[old_si]

        best_new_si = None
        best_new_range = hub_ranges[hi]
        best_delta = 0.0
        best_delta_obj = 0.0
        best_expr_vals: dict[int, float] = {}

        for new_si in range(len(schedules)):
            if new_si == old_si:
                continue
            if preserve_frequency and len(schedules[new_si]) != len(old_days):
                continue
            delta_c = dd_cost_mx[pi, new_si] - old_cost_pi
            delta_total = delta_c

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
                    delta_total += nv - ecache[hi, d_aff]
                    new_expr_vals[d_aff] = nv
                chosen[pi] = old_si

            delta_obj = delta_total + (
                float(penalty_mx[pi, new_si] - penalty_mx[pi, old_si])
                if use_pen else 0.0
            )
            if cur_obj + delta_obj > max_obj:
                continue

            old_veh = veh_3d[pi, old_si, :]
            new_veh = veh_3d[pi, new_si, :]
            new_fleet_hub = fleet[hi] - old_veh + new_veh
            new_range = float(new_fleet_hub.max() - new_fleet_hub.min())
            if new_range < best_new_range:
                best_new_range = new_range
                best_new_si = new_si
                best_delta = delta_total
                best_delta_obj = delta_obj
                best_expr_vals = new_expr_vals

        if best_new_si is not None:
            fleet[hi] -= veh_3d[pi, old_si, :]
            fleet[hi] += veh_3d[pi, best_new_si, :]
            chosen[pi] = best_new_si
            cur_cost += best_delta
            cur_obj += best_delta_obj
            for d_val, val in best_expr_vals.items():
                ecache[hi, d_val] = val
            swaps_made += 1
            pbar.set_postfix_str(f"swaps={swaps_made}, imb={_fleet_imbalance(fleet):.0f}")

    imbalance_after = _fleet_imbalance(fleet)
    log.debug(
        f"Fleet balancing (ML): {swaps_made} swaps, "
        f"imbalance {imbalance_before:.0f} → {imbalance_after:.0f}, "
        f"routing delta {(cur_cost / initial_total_cost - 1) * 100:+.2f}%, "
        f"objective delta {(cur_obj / initial_obj - 1) * 100:+.2f}%"
    )

    schedules_per_plz = {
        plz_keys[pi]: schedules[int(chosen[pi])] for pi in range(n_plz)
    }

    return {
        "chosen": chosen,
        "cost": cur_cost,                       # TOTAL cost after balancing (dd + express)
        "initial_total_cost": initial_total_cost,  # TOTAL cost before balancing
        "imbalance_before": imbalance_before,
        "imbalance_after": imbalance_after,
        "schedules_per_plz": schedules_per_plz,
        "swaps_made": swaps_made,
    }


# ─────────────────────────────────────────────────────────────────────────────
# System-level fleet smoothing (provider aggregate Mo-Sa flattening)
# ─────────────────────────────────────────────────────────────────────────────

def system_smooth_pass(
    chosen: np.ndarray,
    plz_keys: list[str],
    plz_hub_arr: np.ndarray,
    hub_plz_list: list[np.ndarray],
    matrices: dict,
    schedules: list[frozenset[int]],
    cost_budget_pct: float = 1.0,
    max_iterations: int = 200,
    seed: int = SA_SEED + 17,
    express_scale: float = 1.0,
    penalty_mx: np.ndarray | None = None,
) -> dict:
    """System-level fleet smoothing pass (provider aggregate, frequency-preserving).

    Operates AFTER ``balance_fleet_per_hub_ml`` (which is greedy on the per-hub
    spread). The per-hub greedy can leave the provider-aggregate Mo-Sa fleet
    pattern uneven when hubs synchronise their peak days. This pass performs
    additional same-frequency swaps that pull mass off the system-peak day onto
    the system-trough day, until no improving swap exists or the additional
    cost budget is exhausted.

    Constraints
    -----------
    - Swaps are frequency-preserving (``len(new_si) == len(old_si)``) so the
      service quality at the PLZ does not change.
    - Each swap respects the **penalised** objective budget: the post-swap
      total objective must stay within ``cost_budget_pct`` of the input
      objective. With ``penalty_mx=None`` the budget reduces to the routing
      cost alone (dd + bundled express).
    - Candidate swaps must move the PLZ OFF the current system-peak day; we do
      not require landing on the system-trough day to keep the candidate pool
      broad.

    Returns the same schema as ``balance_fleet_per_hub_ml`` so it can be
    chained directly after it.
    """
    dd_cost_mx = matrices["dd_cost_mx"]
    veh_3d = matrices["veh_3d"]
    raw_express = matrices["raw_express"]
    expr_stops = matrices["expr_stops"]
    n_plz = len(plz_keys)

    chosen = chosen.copy()
    cur_dd_cost = float(dd_cost_mx[np.arange(n_plz), chosen].sum())
    express_pred_cache: dict = {}
    ecache = np.zeros((len(hub_plz_list), N_DAYS))
    for hi in range(len(hub_plz_list)):
        for d in range(N_DAYS):
            ecache[hi, d] = _hub_express_day_ml(
                hi, d, chosen, hub_plz_list, schedules,
                raw_express, expr_stops, matrices,
                express_pred_cache, express_scale,
            )
    cur_cost = cur_dd_cost + ecache.sum()
    initial_total_cost = cur_cost
    use_pen = penalty_mx is not None
    cur_pen = float(penalty_mx[np.arange(n_plz), chosen].sum()) if use_pen else 0.0
    cur_obj = cur_cost + cur_pen
    initial_obj = cur_obj
    max_obj = initial_obj * (1 + cost_budget_pct / 100.0)

    fleet = _daily_fleet_per_hub(chosen, plz_hub_arr, hub_plz_list, veh_3d, schedules)
    sys_fleet = fleet.sum(axis=0)
    system_spread_initial = float(sys_fleet.max() - sys_fleet.min())

    rng = np.random.default_rng(seed)
    swaps_made = 0

    for _it in range(max_iterations):
        d_max = int(np.argmax(sys_fleet))
        d_min = int(np.argmin(sys_fleet))
        cur_spread = float(sys_fleet[d_max] - sys_fleet[d_min])
        if cur_spread <= 1.0:
            break

        # PLZs whose current schedule includes the system-peak day. These are
        # the candidates whose move could lower the peak. Cap exploration to
        # keep per-iteration cost bounded.
        cand_plz = [pi for pi in range(n_plz)
                    if d_max in schedules[int(chosen[pi])]]
        if len(cand_plz) == 0:
            break
        rng.shuffle(cand_plz)
        cand_plz = cand_plz[:min(len(cand_plz), 250)]

        best_si = None
        best_pi = None
        best_delta_obj = 0.0
        best_delta_cost = 0.0
        best_expr_new: dict[tuple[int, int], float] = {}
        best_reduction = 0.0

        for pi in cand_plz:
            old_si = int(chosen[pi])
            old_days = schedules[old_si]
            old_size = len(old_days)
            hi = int(plz_hub_arr[pi])
            old_veh = veh_3d[pi, old_si, :]

            for new_si in range(len(schedules)):
                if new_si == old_si:
                    continue
                new_days = schedules[new_si]
                # Frequency-preserving + must move off d_max
                if len(new_days) != old_size:
                    continue
                if d_max in new_days:
                    continue

                # System-spread of swap (cheap check before paying for cost eval)
                new_veh = veh_3d[pi, new_si, :]
                new_sys_fleet = sys_fleet + (new_veh - old_veh)
                new_spread = float(new_sys_fleet.max() - new_sys_fleet.min())
                reduction = cur_spread - new_spread
                if reduction <= best_reduction:
                    continue

                # Cost & objective delta (full evaluation only for promising swaps)
                delta_cost = dd_cost_mx[pi, new_si] - dd_cost_mx[pi, old_si]
                affected = old_days.symmetric_difference(new_days)
                expr_new: dict[tuple[int, int], float] = {}
                if affected:
                    chosen[pi] = new_si
                    for d_aff in affected:
                        nv = _hub_express_day_ml(
                            hi, d_aff, chosen, hub_plz_list, schedules,
                            raw_express, expr_stops, matrices,
                            express_pred_cache, express_scale,
                        )
                        delta_cost += nv - ecache[hi, d_aff]
                        expr_new[(hi, d_aff)] = nv
                    chosen[pi] = old_si

                delta_obj = float(delta_cost) + (
                    float(penalty_mx[pi, new_si] - penalty_mx[pi, old_si])
                    if use_pen else 0.0
                )
                if cur_obj + delta_obj > max_obj:
                    continue

                best_reduction = reduction
                best_si = new_si
                best_pi = pi
                best_delta_obj = delta_obj
                best_delta_cost = float(delta_cost)
                best_expr_new = expr_new

        if best_si is None:
            break

        # Apply best swap
        old_si = int(chosen[best_pi])
        hi = int(plz_hub_arr[best_pi])
        fleet[hi] = fleet[hi] - veh_3d[best_pi, old_si, :] + veh_3d[best_pi, best_si, :]
        sys_fleet = fleet.sum(axis=0)
        chosen[best_pi] = best_si
        cur_cost += best_delta_cost
        cur_obj += best_delta_obj
        for (hi_aff, d_aff), val in best_expr_new.items():
            ecache[hi_aff, d_aff] = val
        swaps_made += 1

    system_spread_final = float(sys_fleet.max() - sys_fleet.min())
    schedules_per_plz = {
        plz_keys[pi]: schedules[int(chosen[pi])] for pi in range(n_plz)
    }

    log.debug(
        f"System smoothing: {swaps_made} swaps, "
        f"system spread {system_spread_initial:.0f} → {system_spread_final:.0f}, "
        f"routing delta {(cur_cost / initial_total_cost - 1) * 100:+.2f}%, "
        f"objective delta {(cur_obj / initial_obj - 1) * 100:+.2f}%"
    )

    return {
        "chosen": chosen,
        "cost": cur_cost,
        "initial_total_cost": initial_total_cost,
        "system_spread_before": system_spread_initial,
        "system_spread_after": system_spread_final,
        "schedules_per_plz": schedules_per_plz,
        "swaps_made": swaps_made,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate-descent optimiser (replaces SA for ML cost)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_cd_ml(
    plz_keys: list[str],
    plz_hub_arr: np.ndarray,
    hub_plz_list: list[np.ndarray],
    matrices: dict,
    schedules: list[frozenset[int]],
    batch_only: bool = False,
    express_scale: float = 1.0,
    max_rounds: int = 10,
    fixed_assignment: np.ndarray | None = None,
    n_restarts: int = 1,
    shuffle_plz: bool = False,
    seed: int = 42,
    pair_polish: bool = False,
    pair_polish_rounds: int = 2,
    pair_polish_max_pairs: int = 200,
) -> dict:
    """Optimise schedule assignment via multi-start coordinate descent.

    For **batch-only** scenarios the per-PLZ costs are independent, so
    ``argmin(dd_cost_mx, axis=1)`` is globally optimal and returned
    immediately.

    For **express** scenarios the hub-level bundling cost couples PLZ at the
    same hub.  Each restart initialises differently, then iterates: for
    each PLZ (in shuffled order if *shuffle_plz*), try every schedule, pick
    the one that yields the lowest total cost (dd + express).  Repeat until
    no PLZ improves.  The best result across all restarts is returned.

    Parameters
    ----------
    n_restarts : int
        Number of independent restarts.  Restart 0 uses *fixed_assignment*
        (or greedy argmin); restarts 1..N-1 use random initialisation.
    shuffle_plz : bool
        If True, randomise PLZ iteration order each round to reduce
        sequential bias.
    seed : int
        Base random seed for reproducibility.
    """
    dd_cost_mx = matrices["dd_cost_mx"]
    raw_express = matrices["raw_express"]
    expr_stops = matrices["expr_stops"]

    n_plz = len(plz_keys)
    n_sched = len(schedules)
    n_hubs = len(hub_plz_list)

    # ── Batch-only: argmin is globally optimal (no coupling) ─────────
    if batch_only:
        chosen = np.argmin(dd_cost_mx, axis=1).copy()
        _pi = np.arange(n_plz)
        best_cost = float(dd_cost_mx[_pi, chosen].sum())
        schedules_per_plz = {
            plz_keys[pi]: schedules[int(chosen[pi])] for pi in range(n_plz)
        }
        log.info(
            f"CD_ML batch-only: argmin optimal, cost={best_cost:,.0f}"
        )
        return {
            "chosen": chosen,
            "best_cost": best_cost,
            "history": [(0, best_cost)],
            "accepted": 0,
            "improved": 0,
            "reheats": 0,
            "polish_rounds": 0,
            "schedules_per_plz": schedules_per_plz,
            "restart_costs": [best_cost],
        }

    # ── Multi-start coordinate descent for express scenarios ─────────
    rng = np.random.default_rng(seed)
    global_best_cost = float("inf")
    global_best_chosen = None
    global_history: list[tuple[int, float]] = []
    restart_costs: list[float] = []
    global_improved = 0
    t0_global = time.perf_counter()

    for restart in range(n_restarts):
        t0 = time.perf_counter()

        # ── Initialisation ───────────────────────────────────────────
        if restart == 0:
            if fixed_assignment is not None:
                chosen = fixed_assignment.copy()
            else:
                chosen = np.argmin(dd_cost_mx, axis=1).copy()
            init_label = "fixed/argmin"
        else:
            chosen = rng.integers(0, n_sched, size=n_plz, dtype=np.intp)
            init_label = "random"

        # ── Build initial express cache ──────────────────────────────
        express_pred_cache: dict = {}
        ecache = np.zeros((n_hubs, N_DAYS))
        for hi in range(n_hubs):
            for d in range(N_DAYS):
                ecache[hi, d] = _hub_express_day_ml(
                    hi, d, chosen, hub_plz_list, schedules,
                    raw_express, expr_stops, matrices,
                    express_pred_cache, express_scale,
                )

        _pi = np.arange(n_plz)
        cur_cost = float(dd_cost_mx[_pi, chosen].sum()) + ecache.sum()
        best_cost = cur_cost
        best_chosen = chosen.copy()
        total_improved = 0

        for rnd in range(max_rounds):
            round_improved = 0
            plz_order = (
                rng.permutation(n_plz) if shuffle_plz
                else np.arange(n_plz)
            )
            for pi in plz_order:
                pi = int(pi)
                old_si = int(chosen[pi])
                hi = int(plz_hub_arr[pi])
                old_days = schedules[old_si]

                best_si = old_si
                best_delta = 0.0
                best_expr_new: dict[int, float] = {}

                for new_si in range(n_sched):
                    if new_si == old_si:
                        continue
                    delta = dd_cost_mx[pi, new_si] - dd_cost_mx[pi, old_si]

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

                    if delta < best_delta:
                        best_delta = delta
                        best_si = new_si
                        best_expr_new = expr_new

                if best_si != old_si:
                    chosen[pi] = best_si
                    cur_cost += best_delta
                    for d_val, val in best_expr_new.items():
                        ecache[hi, d_val] = val
                    round_improved += 1

            total_improved += round_improved
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_chosen = chosen.copy()

            if round_improved == 0:
                break

        # ── Pair-PLZ polish: refine via two-PLZ moves with day-toggle nbrs ─
        if pair_polish:
            nbr_idx = _day_toggle_neighbors(schedules)
            for k_polish in range(pair_polish_rounds):
                d, n_acc = _pair_polish_round(
                    chosen, plz_hub_arr, hub_plz_list, schedules,
                    matrices, ecache, express_pred_cache, express_scale,
                    nbr_idx, rng, max_pairs=pair_polish_max_pairs,
                )
                cur_cost += d
                total_improved += n_acc
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_chosen = chosen.copy()
                if n_acc == 0:
                    break
                log.debug(
                    f"CD_ML pair-polish round {k_polish + 1}: "
                    f"{n_acc} swaps, Δcost={d:,.1f}"
                )

        elapsed = time.perf_counter() - t0
        restart_costs.append(best_cost)
        global_improved += total_improved

        if n_restarts > 1:
            log.info(
                f"CD_ML restart {restart + 1}/{n_restarts} ({init_label}): "
                f"{rnd + 1} rounds, {total_improved} improved, "
                f"cost={best_cost:,.0f}, {elapsed:.1f}s"
            )

        if best_cost < global_best_cost:
            global_best_cost = best_cost
            global_best_chosen = best_chosen.copy()
        global_history.append((restart, global_best_cost))

    schedules_per_plz = {
        plz_keys[pi]: schedules[int(global_best_chosen[pi])]
        for pi in range(n_plz)
    }

    elapsed_total = time.perf_counter() - t0_global
    if n_restarts > 1:
        log.info(
            f"CD_ML complete: {n_restarts} restarts, "
            f"best={global_best_cost:,.0f}, worst={max(restart_costs):,.0f}, "
            f"spread={max(restart_costs) - min(restart_costs):,.0f}, "
            f"{elapsed_total:.1f}s"
        )
    else:
        log.info(
            f"CD_ML complete: {len(global_history)} restart(s), "
            f"{global_improved} improvements, cost={global_best_cost:,.0f}, "
            f"{elapsed_total:.1f}s"
        )

    return {
        "chosen": global_best_chosen,
        "best_cost": global_best_cost,
        "history": global_history,
        "accepted": global_improved,
        "improved": global_improved,
        "reheats": 0,
        "polish_rounds": n_restarts,
        "schedules_per_plz": schedules_per_plz,
        "restart_costs": restart_costs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Day-toggle neighborhood + Pair-PLZ-Move polish
# ─────────────────────────────────────────────────────────────────────────────

def _day_toggle_neighbors(
    schedules: list[frozenset[int]],
) -> list[np.ndarray]:
    """For each schedule index, list indices that differ by exactly one day.

    Result: ``neighbors[si] = ndarray of int64 indices into ``schedules``
    such that ``schedules[si] △ schedules[sj]`` has cardinality 1
    (i.e. they differ by exactly one day toggled on/off).  Schedules that
    are not in the valid 39-pattern set are simply absent.

    These local moves explore the schedule lattice in single-day steps —
    finer than the full 39-pattern jump, useful as a fast polish operator.
    """
    n = len(schedules)
    out: list[np.ndarray] = []
    for si in range(n):
        s_i = schedules[si]
        nbr: list[int] = []
        for sj in range(n):
            if sj == si:
                continue
            if len(s_i.symmetric_difference(schedules[sj])) == 1:
                nbr.append(sj)
        out.append(np.asarray(nbr, dtype=np.int64))
    return out


def _pair_polish_round(
    chosen: np.ndarray,
    plz_hub_arr: np.ndarray,
    hub_plz_list: list[np.ndarray],
    schedules: list[frozenset[int]],
    matrices: dict,
    ecache: np.ndarray,
    express_pred_cache: dict,
    express_scale: float,
    nbr_idx: list[np.ndarray],
    rng: np.random.Generator,
    max_pairs: int,
) -> tuple[float, int]:
    """One round of pair-PLZ moves, restricted to day-toggle neighbours.

    Iterates over a sample of co-located PLZ pairs ``(pi, pj)`` (same hub).
    For each pair, tries neighbour patterns of *both* PLZ simultaneously and
    accepts the move yielding the lowest joint cost delta.  Modifies
    ``chosen`` and ``ecache`` in place.

    Returns ``(cost_delta, n_accepted)`` — total improvement and number of
    accepted swaps in this round.
    """
    dd_cost_mx = matrices["dd_cost_mx"]
    raw_express = matrices["raw_express"]
    expr_stops = matrices["expr_stops"]

    # Build candidate pair list: all (pi, pj) at the same hub
    pair_pool: list[tuple[int, int]] = []
    for h_ps in hub_plz_list:
        if len(h_ps) < 2:
            continue
        ps = list(map(int, h_ps))
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                pair_pool.append((ps[i], ps[j]))
    if not pair_pool:
        return 0.0, 0

    # Sub-sample for speed
    if len(pair_pool) > max_pairs:
        idx = rng.choice(len(pair_pool), size=max_pairs, replace=False)
        pair_pool = [pair_pool[k] for k in idx]

    total_delta = 0.0
    n_accept = 0

    for pi, pj in pair_pool:
        hi = int(plz_hub_arr[pi])  # same hub by construction
        old_si, old_sj = int(chosen[pi]), int(chosen[pj])
        nbrs_i = nbr_idx[old_si]
        nbrs_j = nbr_idx[old_sj]
        if len(nbrs_i) == 0 or len(nbrs_j) == 0:
            continue

        old_days_i = schedules[old_si]
        old_days_j = schedules[old_sj]

        best_si, best_sj = old_si, old_sj
        best_delta = 0.0
        best_expr_new: dict[int, float] = {}

        for new_si in nbrs_i:
            for new_sj in nbrs_j:
                new_si_i = int(new_si)
                new_sj_i = int(new_sj)
                # dd-cost delta
                d_dd = (
                    (dd_cost_mx[pi, new_si_i] - dd_cost_mx[pi, old_si])
                    + (dd_cost_mx[pj, new_sj_i] - dd_cost_mx[pj, old_sj])
                )

                new_days_i = schedules[new_si_i]
                new_days_j = schedules[new_sj_i]
                affected = (
                    old_days_i.symmetric_difference(new_days_i)
                    | old_days_j.symmetric_difference(new_days_j)
                )

                expr_new: dict[int, float] = {}
                d_expr = 0.0
                if affected:
                    chosen[pi] = new_si_i
                    chosen[pj] = new_sj_i
                    for d_aff in affected:
                        nv = _hub_express_day_ml(
                            hi, d_aff, chosen, hub_plz_list, schedules,
                            raw_express, expr_stops, matrices,
                            express_pred_cache, express_scale,
                        )
                        d_expr += nv - ecache[hi, d_aff]
                        expr_new[d_aff] = nv
                    chosen[pi] = old_si
                    chosen[pj] = old_sj

                delta = d_dd + d_expr
                if delta < best_delta:
                    best_delta = delta
                    best_si, best_sj = new_si_i, new_sj_i
                    best_expr_new = expr_new

        if best_si != old_si or best_sj != old_sj:
            chosen[pi] = best_si
            chosen[pj] = best_sj
            for d_val, val in best_expr_new.items():
                ecache[hi, d_val] = val
            total_delta += best_delta
            n_accept += 1

    return total_delta, n_accept
