"""Fleet balancing per hub + system-level smoothing.

Production path (spec v3 §4.3, Tasks 6e/6f):

* :func:`operator_polish` — local search over (cell, schedule) moves that
  minimises what the operator actually pays for the week: the routing
  cost's variable part plus one weekly fixed bill per hub PEAK vehicle.
* :func:`operator_polish_best_of_n` — the production wrapper: that polish,
  frequency-FREE, from three starts (stage 1, the range-balanced plan, and
  the frequency-preserving best-of-two plan), keeping the cheapest end state.

Kept for ablation against it:

* :func:`operator_polish_best_of_two` — the frequency-PRESERVING wrapper
  grid v4 shipped; it is also the third candidate start above.

Kept for ablation against both (the pre-6e two-stage balancing):

* :func:`balance_fleet_per_hub` / :func:`balance_fleet_per_hub_ml` —
  swap-based postprocessing that equalises daily vehicle counts within
  each hub while respecting a cost-increase budget.
* :func:`system_smooth_pass` — system-level smoothing that exchanges
  schedules across hubs when the imbalance crosses a threshold.

``_daily_fleet_per_hub`` and ``_fleet_imbalance`` are vectorised
helpers used by every stage.
"""


import numpy as np
from tqdm.auto import tqdm

from batch_delivery.config.constants import (
    FIXED_COST_EUR,
    FLEET_BALANCE_MAX_SWAPS,
    N_DAYS,
    SA_SEED,
)
from batch_delivery.optimization.costs import (
    _hub_delivery_pool_vehicles,
    _hub_express_day,
    _hub_express_day_ml,
    _hub_express_vehicles,
    _hub_smallday_pool_ml,
    _pool_affected_days,
)
from batch_delivery.utils import log

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Fleet balancing post-processing (swap-based)
# ─────────────────────────────────────────────────────────────────────────────

def _sched_active_mask(schedules: list[frozenset[int]]) -> np.ndarray:
    """``(n_sched, N_DAYS)`` delivery-day mask, derived from *schedules*.

    Fallback for call sites that do not carry ``matrices["sched_active"]``
    (hand-built matrices in tests, the legacy Daganzo path). Cheap: 39x6.
    """
    sa = np.zeros((len(schedules), N_DAYS), dtype=bool)
    for si, sched in enumerate(schedules):
        for d in sched:
            sa[si, d] = True
    return sa


def _daily_fleet_per_hub(
    chosen: np.ndarray,
    plz_hub_arr: np.ndarray,
    hub_plz_list: list[np.ndarray],
    veh_3d: np.ndarray,
    schedules: list[frozenset[int]],
    pool_veh_fn=None,
    sched_active: np.ndarray | None = None,
) -> np.ndarray:
    """Compute daily vehicle count per hub: shape (n_hubs, N_DAYS).

    A hub's POOLED tours — the express partition on non-delivery days
    (Task 4) and the small-delivery groups on delivery days (spec §4.3 v3) —
    run invisibly to the per-cell count unless ``pool_veh_fn(hi, d, chosen)
    -> float`` is supplied; it adds that hub-day's pooled vehicles on top
    (D2 fix). ``pool_veh_fn=None`` reproduces the legacy count over the raw
    ``veh_3d`` slice (the non-ML ``balance_fleet_per_hub`` Daganzo path
    keeps this).

    ``veh_3d`` no longer carries the pooled small-delivery members — see
    ``build_cost_matrices_ml`` — so the per-cell term and the closure count
    disjoint sets of tours by construction.

    DOUBLE-COUNT FIX (2026-08-26): ``veh_3d`` is written for every ACTIVE
    instance, and on a NON-delivery day a cell's ``combined_demand`` IS its
    express residual (> 0 whenever theta < 1) — so the per-cell slice already
    carries >= 1 vehicle for every non-delivering express cell. Summing it
    over all six days AND adding the pooled express term counted every
    express vehicle twice (measured on DPD 0.5/0.1: recorded 109-118 vs
    69-116 correct, peak +58 %, hub spread 9 vs 47 true; it invalidated the
    base grid's stage-2/3 choices). With a pooled closure supplied the
    per-cell term is therefore masked to DELIVERY days — exactly the
    reporting fix in ``scripts/revision/50_recompute_fleet_wait_fixed.py``
    lines 138-151, now applied to the objective itself. ``sched_active``
    defaults to a mask derived from *schedules*, so no call site can forget
    it; pass ``matrices["sched_active"]`` to skip the (tiny) derivation.
    """
    n_hubs = len(hub_plz_list)
    plz_veh = veh_3d[np.arange(len(chosen)), chosen, :]  # (n_plz, N_DAYS)
    if pool_veh_fn is not None:
        sa = _sched_active_mask(schedules) if sched_active is None else sched_active
        plz_veh = plz_veh * sa[chosen, :]
    fleet = np.zeros((n_hubs, N_DAYS), dtype=np.float64)
    np.add.at(fleet, plz_hub_arr, plz_veh)
    if pool_veh_fn is not None:
        for hi in range(n_hubs):
            for d in range(N_DAYS):
                fleet[hi, d] += pool_veh_fn(hi, d, chosen)
    return fleet




def _fleet_imbalance(fleet: np.ndarray) -> float:
    """Sum of per-hub range (max - min daily vehicles)."""
    return float(np.sum(fleet.max(axis=1) - fleet.min(axis=1)))




# ─────────────────────────────────────────────────────────────────────────────
# Operator-cost objective (spec v3 §4.3, Task 6e)
# ─────────────────────────────────────────────────────────────────────────────

#: Weekly fixed bill per PEAK vehicle at a hub. Six operating days (Mon-Sat)
#: at ``FIXED_COST_EUR`` each: the van is leased and the driver employed for
#: the whole week, so the hub's busiest day sizes the fleet it pays for.
WEEK_FIXED_COST_EUR: float = FIXED_COST_EUR * N_DAYS


def _pool_veh_closure(
    hub_plz_list: list[np.ndarray],
    schedules: list[frozenset[int]],
    matrices: dict,
    express_pred_cache: dict,
    pool_pred_cache: dict,
):
    """``(hi, d, chosen) -> pooled vehicles``: express + small-delivery.

    Shares *express_pred_cache* / *pool_pred_cache* with the
    ``_hub_express_day_ml`` / ``_hub_smallday_pool_ml`` calls of the same
    pass, so vehicle counts come from cache hits rather than a second round
    of partition/surrogate calls.
    """
    def _fn(hi: int, d: int, ch: np.ndarray) -> float:
        return (
            _hub_express_vehicles(
                hi, d, ch, hub_plz_list, schedules, matrices["raw_express"],
                matrices, express_pred_cache,
            )
            + _hub_delivery_pool_vehicles(
                hi, d, ch, hub_plz_list, schedules, matrices, pool_pred_cache,
            )
        )
    return _fn


def operator_cost_breakdown(
    chosen: np.ndarray,
    plz_keys: list[str],
    plz_hub_arr: np.ndarray,
    hub_plz_list: list[np.ndarray],
    matrices: dict,
    schedules: list[frozenset[int]],
    penalty_mx: np.ndarray | None = None,
    express_scale: float = 1.0,
    fixed_cost: float = FIXED_COST_EUR,
    week_fixed: float = WEEK_FIXED_COST_EUR,
) -> dict:
    """From-scratch operator-cost decomposition of one *chosen* vector.

    No optimisation, no incremental state: everything is rebuilt from
    *chosen*, which is what makes this the independent reference the
    bookkeeping gate of :func:`operator_polish` compares against, and the
    reporting path the runner writes its ``operator_cost_eur`` column from.

    The cost model as the labels really contain it (Task 6e brief, verified
    on the training pool)::

        VROOM_cost = 189.15 * n_vehicles + 0.3864 * km + 36.00 * route_hours

    Every predicted cost therefore carries a per-vehicle-DAY fixed term.
    Subtracting ``fixed_cost * vehicle_days`` leaves the distance + time
    part, and the fixed bill is re-charged once per week per hub PEAK
    vehicle — below the peak an extra vehicle-day costs only its variable
    part, because the van is owned and the driver employed anyway.

    Returns a dict with ``dd_cost`` / ``express_cost`` / ``pool_cost`` /
    ``routing_cost``, ``vehicle_days`` / ``fixed_cost`` / ``variable_cost``,
    ``sum_hub_peak`` / ``week_fixed_cost``, ``penalty``, ``operator_cost``
    (variable + weekly fixed, WITHOUT the penalty — the money the operator
    pays) and ``opcost`` (``operator_cost + penalty`` — the objective the
    polish minimises; the wait penalty is a shadow price, not a cost, so the
    two are reported separately). ``fleet`` carries the profile itself.
    """
    n_plz = len(plz_keys)
    n_hubs = len(hub_plz_list)
    dd_cost_mx = matrices["dd_cost_mx"]
    veh_3d = matrices["veh_3d"]
    raw_express = matrices["raw_express"]
    expr_stops = matrices["expr_stops"]
    sa_mx = matrices.get("sched_active")
    if sa_mx is None:
        sa_mx = _sched_active_mask(schedules)

    chosen = np.asarray(chosen)
    express_pred_cache: dict = {}
    pool_pred_cache: dict = {}
    pool_veh_fn = _pool_veh_closure(
        hub_plz_list, schedules, matrices, express_pred_cache, pool_pred_cache)

    dd_cost = float(dd_cost_mx[np.arange(n_plz), chosen].sum())
    express_cost = 0.0
    pool_cost = 0.0
    for hi in range(n_hubs):
        for d in range(N_DAYS):
            express_cost += _hub_express_day_ml(
                hi, d, chosen, hub_plz_list, schedules,
                raw_express, expr_stops, matrices,
                express_pred_cache, express_scale,
            )
            pool_cost += _hub_smallday_pool_ml(
                hi, d, chosen, hub_plz_list, schedules, matrices,
                pool_pred_cache,
            )
    routing_cost = dd_cost + express_cost + pool_cost

    fleet = _daily_fleet_per_hub(
        chosen, plz_hub_arr, hub_plz_list, veh_3d, schedules,
        pool_veh_fn=pool_veh_fn, sched_active=sa_mx,
    )
    vehicle_days = float(fleet.sum())
    sum_hub_peak = float(fleet.max(axis=1).sum())
    fixed = fixed_cost * vehicle_days
    variable = routing_cost - fixed
    week_fixed_cost = week_fixed * sum_hub_peak
    penalty = (float(penalty_mx[np.arange(n_plz), chosen].sum())
               if penalty_mx is not None else 0.0)
    operator_cost = variable + week_fixed_cost

    return {
        "dd_cost": dd_cost,
        "express_cost": express_cost,
        "pool_cost": pool_cost,
        "routing_cost": routing_cost,
        "vehicle_days": vehicle_days,
        "fixed_cost": fixed,
        "variable_cost": variable,
        "sum_hub_peak": sum_hub_peak,
        "week_fixed_cost": week_fixed_cost,
        "penalty": penalty,
        "operator_cost": operator_cost,
        "opcost": operator_cost + penalty,
        "fleet": fleet,
        "imbalance": _fleet_imbalance(fleet),
    }


def operator_polish(
    sa_result: dict,
    plz_keys: list[str],
    plz_hub_arr: np.ndarray,
    hub_plz_list: list[np.ndarray],
    matrices: dict,
    schedules: list[frozenset[int]],
    max_swaps: int = FLEET_BALANCE_MAX_SWAPS,
    max_sweeps: int = 50,
    seed: int = SA_SEED + 1,
    express_scale: float = 1.0,
    penalty_mx: np.ndarray | None = None,
    preserve_frequency: bool = False,
    fixed_cost: float = FIXED_COST_EUR,
    week_fixed: float = WEEK_FIXED_COST_EUR,
    accept_eps: float = 1e-9,
) -> dict:
    """Stage 2 as an operator-cost minimisation (spec v3 §4.3, Task 6e).

    Replaces :func:`balance_fleet_per_hub_ml`'s "flatten the per-hub range
    inside a cost budget" with the one economic objective the operator
    actually pays::

        OpCost   = variable + W * Sigma_h peak_h  (+ penalty)
        variable = routing_cost - 189.15 * vehicle_days
        W        = 6 * 189.15 EUR per peak vehicle per hub per week
        peak_h   = max_d fleet[h, d]

    ``fleet`` is the v3 partition-aware profile of
    :func:`_daily_fleet_per_hub`: per-cell tours masked to delivery days,
    plus one ceil per pooled express tour and per pooled small-delivery
    group. The range objective and the cost budget are both gone; a move is
    accepted iff it strictly lowers OpCost::

        dOpCost = d(variable) + W * d(peak_h) + d(penalty)
        d(variable) = d(routing_cost) - 189.15 * d(vehicle_days_h)

    Only the hub of the moved cell can change — neither pool is priced or
    counted across hubs — so both fleet terms are evaluated on that one hub
    row. The candidate machinery (masked ``veh_3d`` rows, ``ecache`` /
    ``pcache`` / ``pvcache`` mirrors, exact pooled-vehicle deltas over
    ``symmetric_difference | _pool_affected_days``) is the one
    :func:`balance_fleet_per_hub_ml` uses; only the accept rule differs.

    Search: sweeps over a seeded permutation of the cells, best improving
    schedule per cell, applied immediately; it stops when a whole sweep
    finds nothing (``max_swaps_binding`` False) or when *max_swaps* /
    *max_sweeps* run out (``max_swaps_binding`` True — report it, the result
    is then not a local optimum).

    ``preserve_frequency=True`` restricts candidates to schedules of the same
    SIZE, so stage 2 only redistributes WHICH days a cell is served on, never
    how many. That pins each cell's delivery FREQUENCY — and with it the
    theta=0 baseline, which would otherwise be batched away (at theta=0 nobody
    waits, so ``penalty_mx`` is identically zero for every P and an
    unrestricted polish would face an unpriced service dimension).

    **Since Task 6f the pin is NOT the production setting at theta > 0.** The
    production caller (``61_grid_run_v2.stage2_plan`` via
    :func:`operator_polish_best_of_n`) runs frequency-FREE there, because
    everything a frequency change touches is priced — ``Delta variable``,
    ``W * Delta peak``, ``Delta penalty`` — and the pin was blocking the moves
    that carry most of the operator-lens value (a hub serving a single cell has
    nothing to rotate; its peak only falls with MORE delivery days). The pin
    survives as the theta=0 rule and as the ``operator-freqpres`` /
    ``operator-solo`` ablations. Do not describe it as the canonical wiring.

    It does NOT pin the wait. Schedules of equal size differ in average wait
    (size 3: 0.50-0.67 days; size 4: 0.33-0.50), so moving a cell from
    ``{Mon, Wed, Fri}`` to ``{Mon, Tue, Thu}`` changes the service metric
    without changing the frequency — measured on the v3 grid, the willing-
    weighted wait moves at stage 2 in 94 of 176 theta > 0 triples, by up to
    -17.65 %. This is priced, not ignored: ``Delta penalty`` is part of every
    accept decision. Any downstream text must say "frequency-preserving", never
    "wait-invariant".

    Returns :func:`balance_fleet_per_hub_ml`'s dict plus ``opcost_before`` /
    ``opcost_after`` (objective, penalty included), ``operator_cost_before`` /
    ``_after`` (penalty excluded), ``variable_before`` / ``_after``,
    ``sum_hub_peak_before`` / ``_after``, ``vehicle_days_before`` / ``_after``,
    ``penalty_before`` / ``_after``, ``accepted_deltas`` (one dOpCost per
    accepted move), ``sweeps`` and ``max_swaps_binding``.
    """
    dd_cost_mx = matrices["dd_cost_mx"]
    veh_3d = matrices["veh_3d"]
    raw_express = matrices["raw_express"]
    expr_stops = matrices["expr_stops"]
    n_plz = len(plz_keys)
    n_hubs = len(hub_plz_list)
    n_sched = len(schedules)
    pidx = np.arange(n_plz)
    # Delivery-day mask: every per-cell fleet term below is masked with it,
    # so the pooled vehicles added on top are not double-counted
    # (see _daily_fleet_per_hub).
    sa_mx = matrices.get("sched_active")
    if sa_mx is None:
        sa_mx = _sched_active_mask(schedules)

    chosen = sa_result["chosen"].copy()
    express_pred_cache: dict = {}
    pool_pred_cache: dict = {}
    ecache = np.zeros((n_hubs, N_DAYS))
    pcache = np.zeros((n_hubs, N_DAYS))
    # Pooled-vehicle mirror of ecache/pcache, so the accept GATE (not just the
    # accepted state) can correct its tentative fleet row for the pooled
    # vehicles a move silently shifts (I1 fix, inherited from the sibling).
    pvcache = np.zeros((n_hubs, N_DAYS))
    _pool_veh_fn = _pool_veh_closure(
        hub_plz_list, schedules, matrices, express_pred_cache, pool_pred_cache)

    for hi in range(n_hubs):
        for d in range(N_DAYS):
            ecache[hi, d] = _hub_express_day_ml(
                hi, d, chosen, hub_plz_list, schedules,
                raw_express, expr_stops, matrices,
                express_pred_cache, express_scale,
            )
            pcache[hi, d] = _hub_smallday_pool_ml(
                hi, d, chosen, hub_plz_list, schedules, matrices,
                pool_pred_cache,
            )
            pvcache[hi, d] = _pool_veh_fn(hi, d, chosen)

    use_pen = penalty_mx is not None
    fleet = _daily_fleet_per_hub(
        chosen, plz_hub_arr, hub_plz_list, veh_3d, schedules,
        pool_veh_fn=_pool_veh_fn, sched_active=sa_mx,
    )

    def _state() -> tuple[float, float, float, float, float]:
        """The five tracked scalars, each re-derived from the live caches.

        Deriving rather than accumulating is what makes the reported numbers
        equal to a from-scratch recomputation (G-6e-3): ``ecache`` /
        ``pcache`` / ``fleet`` are updated exactly on every accepted move, and
        everything else is a sum over them — so no running total can drift.
        Cheap: O(n_plz + n_hubs * N_DAYS) per accepted move.
        """
        cost = (float(dd_cost_mx[pidx, chosen].sum())
                + float(ecache.sum()) + float(pcache.sum()))
        pen = float(penalty_mx[pidx, chosen].sum()) if use_pen else 0.0
        vdays = float(fleet.sum())
        peaks = float(fleet.max(axis=1).sum())
        opc = cost - fixed_cost * vdays + week_fixed * peaks + pen
        return cost, pen, vdays, peaks, opc

    cur_cost, cur_pen, veh_days, peak_sum, opcost = _state()
    initial_total_cost = cur_cost
    opcost_before = opcost
    variable_before = cur_cost - fixed_cost * veh_days
    veh_days_before = veh_days
    peak_before = peak_sum
    pen_before = cur_pen
    imbalance_before = _fleet_imbalance(fleet)

    rng = np.random.default_rng(seed)
    swaps_made = 0
    sweeps = 0
    accepted_deltas: list[float] = []
    hit_bound = False

    # ``max_swaps=0`` is the pure-measurement call (the runner's stage-1 cost
    # anchor): report the input state, search nothing, and do not pretend the
    # bound says anything about local optimality.
    pbar = tqdm(
        range(max_sweeps if max_swaps > 0 else 0), desc="Operator polish",
        unit="sweep", leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )
    for _sweep in pbar:
        sweeps += 1
        improved = False
        for pi in rng.permutation(n_plz):
            if swaps_made >= max_swaps:
                hit_bound = True
                break
            pi = int(pi)
            hi = int(plz_hub_arr[pi])
            old_si = int(chosen[pi])
            old_days = schedules[old_si]
            old_cost_pi = dd_cost_mx[pi, old_si]
            hub_row = fleet[hi]
            old_row_sum = float(hub_row.sum())
            old_row_peak = float(hub_row.max())

            best_new_si = None
            best_delta_op = -accept_eps
            best_expr_vals: dict[int, float] = {}
            best_pool_vals: dict[int, float] = {}

            for new_si in range(n_sched):
                if new_si == old_si:
                    continue
                if preserve_frequency and len(schedules[new_si]) != len(old_days):
                    continue

                delta_cost = float(dd_cost_mx[pi, new_si] - old_cost_pi)
                new_days = schedules[new_si]
                affected = old_days.symmetric_difference(new_days)
                pool_affected = _pool_affected_days(pi, old_si, new_si, matrices)
                new_expr_vals: dict[int, float] = {}
                new_pool_vals: dict[int, float] = {}
                new_pool_veh: dict[int, float] = {}
                if affected or pool_affected:
                    chosen[pi] = new_si
                    for d_aff in affected:
                        nv = _hub_express_day_ml(
                            hi, d_aff, chosen, hub_plz_list, schedules,
                            raw_express, expr_stops, matrices,
                            express_pred_cache, express_scale,
                        )
                        delta_cost += nv - ecache[hi, d_aff]
                        new_expr_vals[d_aff] = nv
                    for d_aff in pool_affected:
                        pv = _hub_smallday_pool_ml(
                            hi, d_aff, chosen, hub_plz_list, schedules,
                            matrices, pool_pred_cache,
                        )
                        delta_cost += pv - pcache[hi, d_aff]
                        new_pool_vals[d_aff] = pv
                    # Still in the TRIAL state and served from the caches the
                    # two loops above just warmed for this exact
                    # (hi, d_aff, chosen): the true pooled-vehicle count, for
                    # the gate and not merely for the accepted state. Express
                    # moves on the schedules' symmetric difference, the
                    # delivery pool on _pool_affected_days (a day served by
                    # BOTH schedules still changes the pool when the batched
                    # demand behind it changes) -- hence the union.
                    for d_aff in set(affected) | set(pool_affected):
                        new_pool_veh[d_aff] = _pool_veh_fn(hi, d_aff, chosen)
                    chosen[pi] = old_si

                delta_pen = (
                    float(penalty_mx[pi, new_si] - penalty_mx[pi, old_si])
                    if use_pen else 0.0
                )

                # Delivery-day-masked, to match how `fleet` was built (the
                # unmasked slice would re-introduce the express double count).
                new_row = (hub_row
                           - veh_3d[pi, old_si, :] * sa_mx[old_si]
                           + veh_3d[pi, new_si, :] * sa_mx[new_si])
                for d_aff, v in new_pool_veh.items():
                    new_row[d_aff] += v - pvcache[hi, d_aff]

                delta_vdays = float(new_row.sum()) - old_row_sum
                delta_peak = float(new_row.max()) - old_row_peak
                delta_op = (
                    (delta_cost - fixed_cost * delta_vdays)
                    + week_fixed * delta_peak
                    + delta_pen
                )
                if delta_op < best_delta_op:
                    best_delta_op = delta_op
                    best_new_si = new_si
                    best_expr_vals = new_expr_vals
                    best_pool_vals = new_pool_vals

            if best_new_si is None:
                continue

            # ── accept ──────────────────────────────────────────────────
            chosen[pi] = best_new_si
            for d_val, val in best_expr_vals.items():
                ecache[hi, d_val] = val
            for d_val, val in best_pool_vals.items():
                pcache[hi, d_val] = val
            # Full hub-row refresh (not an incremental veh_3d-only delta) so
            # pooled vehicles -- which can shift for OTHER members of the
            # hub's partitions, not just `pi` -- stay exact (D2 fix). Cheap:
            # _pool_veh_fn hits the cache entries the candidate evaluation
            # above already populated for this exact (hi, d, chosen).
            h_ps = hub_plz_list[hi]
            for d in range(N_DAYS):
                pv = _pool_veh_fn(hi, d, chosen)
                pvcache[hi, d] = pv
                deliv = h_ps[sa_mx[chosen[h_ps], d]]
                fleet[hi, d] = (
                    float(veh_3d[deliv, chosen[deliv], d].sum()) + pv)
            cur_cost, cur_pen, veh_days, peak_sum, opcost = _state()
            accepted_deltas.append(best_delta_op)
            swaps_made += 1
            improved = True
            pbar.set_postfix_str(
                f"swaps={swaps_made}, opcost={opcost:.0f}")

        if not improved:
            break
        if swaps_made >= max_swaps:
            hit_bound = True
            break
    else:
        # Every sweep improved something and the sweep budget ran out: like
        # the swap bound, this is NOT a local optimum -- say so.
        hit_bound = hit_bound or sweeps >= max_sweeps
    pbar.close()

    imbalance_after = _fleet_imbalance(fleet)
    variable_after = cur_cost - fixed_cost * veh_days
    log.debug(
        f"Operator polish: {swaps_made} move(s) in {sweeps} sweep(s), "
        f"OpCost {opcost_before:,.0f} → {opcost:,.0f} "
        f"({(opcost / opcost_before - 1) * 100 if opcost_before else 0:+.2f}%), "
        f"Sigma hub peak {peak_before:.0f} → {peak_sum:.0f}, "
        f"vehicle-days {veh_days_before:.0f} → {veh_days:.0f}, "
        f"routing {(cur_cost / initial_total_cost - 1) * 100 if initial_total_cost else 0:+.2f}%"
        + (", BOUND BINDING" if hit_bound else "")
    )

    schedules_per_plz = {
        plz_keys[pi]: schedules[int(chosen[pi])] for pi in range(n_plz)
    }

    return {
        "chosen": chosen,
        "cost": cur_cost,                          # routing total (dd+expr+pool)
        "initial_total_cost": initial_total_cost,
        "imbalance_before": imbalance_before,
        "imbalance_after": imbalance_after,
        "schedules_per_plz": schedules_per_plz,
        "swaps_made": swaps_made,
        # ── the operator lens ───────────────────────────────────────────
        "opcost_before": opcost_before,
        "opcost_after": opcost,
        "operator_cost_before": opcost_before - pen_before,
        "operator_cost_after": opcost - cur_pen,
        "variable_before": variable_before,
        "variable_after": variable_after,
        "sum_hub_peak_before": peak_before,
        "sum_hub_peak_after": peak_sum,
        "vehicle_days_before": veh_days_before,
        "vehicle_days_after": veh_days,
        "penalty_before": pen_before,
        "penalty_after": cur_pen,
        "accepted_deltas": accepted_deltas,
        "sweeps": sweeps,
        "max_swaps_binding": hit_bound,
    }




#: Cost budget of the range balancer when it is used as a CANDIDATE START for
#: :func:`operator_polish_best_of_two` — the value the canonical production run
#: uses (``scripts/pipeline/02_optimize_grid.py``, paper revision 2026-05-27).
RANGE_START_BUDGET_PCT: float = 5.0

#: Candidate starts of the production stage-2 polish (Task 6f), in tie-break
#: order: the stage-1 plan, the range-balanced plan, and the
#: frequency-PRESERVING best-of-two plan (what grid v4 shipped). The third one
#: is what makes ``OpCost(v5) <= OpCost(v4)`` hold by construction.
BEST_OF_N_STARTS: tuple[str, ...] = ("stage1", "range", "freqpres")


def _measurement_only(measured: dict, branches: tuple[str, ...],
                      winner: str) -> dict:
    """Package a ``max_swaps=0`` measurement in the best-of-N result schema.

    ``operator_polish(max_swaps=0)`` searches nothing and reports the plan it
    was handed — its documented pure-measurement contract, which the runner's
    stage-1 anchor depends on. The wrappers honour it: no other candidate
    start is built (the range balancer carries its OWN swap budget and would
    return a different plan), nothing is compared, and every branch field but
    the measured one is ``nan`` / 0 rather than a fabricated value.
    """
    res = dict(measured)
    res["stage2_start_winner"] = winner
    for b in branches:
        res[f"opcost_from_{b}"] = (measured["opcost_after"] if b == winner
                                   else float("nan"))
        res[f"swaps_from_{b}"] = 0
        res[f"sweeps_from_{b}"] = 0
    res["opcost_range_start"] = float("nan")
    res["swaps_range_balancer"] = 0
    if "freqpres" in branches:
        res["opcost_freqpres_start"] = float("nan")
        res["swaps_freqpres_plan"] = 0
    res["max_swaps_binding_any"] = False
    return res


def operator_polish_best_of_two(
    sa_result: dict,
    plz_keys: list[str],
    plz_hub_arr: np.ndarray,
    hub_plz_list: list[np.ndarray],
    matrices: dict,
    schedules: list[frozenset[int]],
    max_swaps: int = FLEET_BALANCE_MAX_SWAPS,
    max_sweeps: int = 50,
    seed: int = SA_SEED + 1,
    express_scale: float = 1.0,
    penalty_mx: np.ndarray | None = None,
    preserve_frequency: bool = False,
    fixed_cost: float = FIXED_COST_EUR,
    week_fixed: float = WEEK_FIXED_COST_EUR,
    accept_eps: float = 1e-9,
    range_budget_pct: float = RANGE_START_BUDGET_PCT,
    range_max_swaps: int = FLEET_BALANCE_MAX_SWAPS,
) -> dict:
    """:func:`operator_polish` from TWO starts; keep the cheaper end state.

    Why a second start is needed
    ----------------------------
    ``W = 1 134.90`` EUR per peak vehicle dwarfs the variable cost a single
    cell can move, so the operator objective is **flat in whole plateaus**:
    until a move empties the last vehicle off the peak day, ``Delta peak = 0``
    and only the (tiny) ``Delta variable`` is visible. A strict single-cell
    descent cannot cross such a plateau. The range objective
    (``max - min``) has no plateau — it gives gradient toward the peak day
    all the way — so the old heuristic sometimes lands in a basin the polish
    cannot reach, and the polish's own early purchases of cheap below-peak
    vehicle-days can lock the peak in.

    Measured on the v3 grid: on 10 of 222 triples (4.5 %, all single-hub
    providers) the pre-6e range heuristic beat a polish started from stage 1
    in OPERATOR cost, worst case +3 046 EUR (FedEx P=0, theta=0.3, peak
    80 -> 85).

    The fix
    -------
    Run the polish from ``chosen`` (stage 1) AND from the range balancer's
    output, and return whichever END state has the lower OpCost. Since the
    range state is itself a candidate start and :func:`operator_polish` never
    worsens OpCost, the result satisfies

        OpCost(best-of-two) <= OpCost(range balancer)   [asserted below]

    by construction — the old heuristic can no longer win. Ties go to the
    stage-1 start.

    Reporting
    ---------
    All ``*_before`` fields are normalised to the STAGE-1 anchor regardless of
    which start won, so ``before -> after`` always reads "stage 1 -> final"
    and ``initial_total_cost`` stays the stage-1 routing cost the grid gates
    on. Adds ``stage2_start_winner`` (``"stage1"`` / ``"range"``),
    ``opcost_from_stage1`` / ``opcost_from_range`` (the two end states),
    ``opcost_range_start`` (the range balancer's own end state — the
    guarantee's reference), and the per-branch move counts.

    ``max_swaps=0`` is the pure-measurement contract of :func:`operator_polish`,
    honoured here too: the range balancer is NOT run (it has its own swap
    budget and would return a different state), no branch is compared, and the
    input plan is reported unchanged with the branch fields set to the
    measured value / ``nan``.
    """
    kw = dict(
        max_swaps=max_swaps, max_sweeps=max_sweeps, seed=seed,
        express_scale=express_scale, penalty_mx=penalty_mx,
        preserve_frequency=preserve_frequency, fixed_cost=fixed_cost,
        week_fixed=week_fixed, accept_eps=accept_eps,
    )
    chosen0 = sa_result["chosen"]

    from_s1 = operator_polish(
        {"chosen": chosen0}, plz_keys, plz_hub_arr, hub_plz_list,
        matrices, schedules, **kw)

    if max_swaps <= 0:
        return _measurement_only(from_s1, ("stage1", "range"), "stage1")

    rng_bal = balance_fleet_per_hub_ml(
        {"chosen": chosen0, "best_cost": 0.0}, plz_keys, plz_hub_arr,
        hub_plz_list, matrices, schedules,
        cost_budget_pct=range_budget_pct, max_swaps=range_max_swaps,
        seed=seed, express_scale=express_scale, penalty_mx=penalty_mx,
        preserve_frequency=preserve_frequency)
    from_rng = operator_polish(
        {"chosen": rng_bal["chosen"]}, plz_keys, plz_hub_arr, hub_plz_list,
        matrices, schedules, **kw)

    # The polish's own warm-up measured this before it moved anything: the
    # OpCost of the range balancer's end state, on exactly the same caches.
    opcost_range_start = from_rng["opcost_before"]

    range_wins = from_rng["opcost_after"] < from_s1["opcost_after"]
    res = dict(from_rng if range_wins else from_s1)

    # Fail loud rather than silently returning a state the old heuristic
    # beats: this is the whole point of the second start.
    assert res["opcost_after"] <= opcost_range_start + 1e-6 * max(
            1.0, abs(opcost_range_start)), (
        f"best-of-two returned {res['opcost_after']:.6f} > range-balancer "
        f"state {opcost_range_start:.6f} — operator_polish must never worsen "
        "OpCost from the start it is given")

    # ``before`` is always the stage-1 anchor, whichever branch won.
    for key in ("initial_total_cost", "imbalance_before", "opcost_before",
                "operator_cost_before", "variable_before",
                "sum_hub_peak_before", "vehicle_days_before",
                "penalty_before"):
        res[key] = from_s1[key]

    res.update({
        "stage2_start_winner": "range" if range_wins else "stage1",
        "opcost_from_stage1": from_s1["opcost_after"],
        "opcost_from_range": from_rng["opcost_after"],
        "opcost_range_start": opcost_range_start,
        "swaps_from_stage1": from_s1["swaps_made"],
        "swaps_from_range": from_rng["swaps_made"],
        "swaps_range_balancer": rng_bal["swaps_made"],
        "sweeps_from_stage1": from_s1["sweeps"],
        "sweeps_from_range": from_rng["sweeps"],
    })
    log.debug(
        f"Operator polish (best-of-two): start '{res['stage2_start_winner']}' "
        f"wins — from stage 1 {from_s1['opcost_after']:,.0f}, from range "
        f"{from_rng['opcost_after']:,.0f} (range state itself "
        f"{opcost_range_start:,.0f})"
    )
    return res




def operator_polish_best_of_n(
    sa_result: dict,
    plz_keys: list[str],
    plz_hub_arr: np.ndarray,
    hub_plz_list: list[np.ndarray],
    matrices: dict,
    schedules: list[frozenset[int]],
    starts: tuple[str, ...] = BEST_OF_N_STARTS,
    max_swaps: int = FLEET_BALANCE_MAX_SWAPS,
    max_sweeps: int = 50,
    seed: int = SA_SEED + 1,
    express_scale: float = 1.0,
    penalty_mx: np.ndarray | None = None,
    preserve_frequency: bool = False,
    fixed_cost: float = FIXED_COST_EUR,
    week_fixed: float = WEEK_FIXED_COST_EUR,
    accept_eps: float = 1e-9,
    range_budget_pct: float = RANGE_START_BUDGET_PCT,
    range_max_swaps: int = FLEET_BALANCE_MAX_SWAPS,
) -> dict:
    """The production stage 2 (Task 6f): a FREQUENCY-FREE polish from N starts.

    Why frequency-free
    ------------------
    With ``preserve_frequency=True`` (the v4 wiring) stage 2 may only re-TIME a
    cell's delivery days, never add or drop one. That blocks the moves which
    carry most of the operator-lens value, because a hub with a single cell has
    nothing to rotate: its profile ``0 0 33 0 0 29`` is the same under every
    rotation of a two-day pattern, and 9 of DHL's 16 hubs are exactly that. Its
    peak only falls if the cell is served on MORE days. Measured on the v3
    grid (DHL, P=0, theta=1): freeing the frequency moved OpCost
    814 314 -> 595 067 EUR, Sigma hub peak 654 -> 447, at +4.2 % routing cost
    and a LOWER wait (0.952 -> 0.621 parcel-weighted days).

    Nothing about a frequency change is unpriced at theta > 0: the objective
    already carries ``Delta variable``, ``W * Delta peak`` and
    ``Delta penalty = P * willing * parcels * Delta wait``. The one case where
    it IS unpriced is theta = 0 — nobody is willing to wait, so ``penalty_mx``
    is identically zero for every P and an unrestricted polish would batch the
    daily baseline away. That case is a stage-2 NO-OP, decided by the caller
    (``scripts/revision/61_grid_run_v2.stage2_plan``), not by this function.

    The start set
    -------------
    ``starts`` names the candidate starts, all polished with the SAME
    (frequency-free) rule; the cheapest end state wins, ties going to the
    earlier name in :data:`BEST_OF_N_STARTS`:

    * ``"stage1"`` — the stage-1 plan (mandatory: it is also the reporting
      anchor every ``*_before`` field is normalised to).
    * ``"range"`` — :func:`balance_fleet_per_hub_ml`'s output. ``W`` dwarfs the
      variable cost one cell can move, so the operator objective is flat in
      plateaus a strict descent cannot cross; ``max - min`` has no plateau and
      reaches basins the polish alone does not (see
      :func:`operator_polish_best_of_two`).
    * ``"freqpres"`` — :func:`operator_polish_best_of_two` with the frequency
      PIN, i.e. exactly the plan grid v4 shipped. Because the polish never
      worsens the state it is given, including it makes

          OpCost(this function) <= OpCost(the v4 plan)

      hold **by construction** — asserted below, so a v5 grid can never be
      worse than v4 on the objective it optimises.

    Returns
    -------
    :func:`operator_polish`'s dict for the WINNING branch, with every
    ``*_before`` field (and ``initial_total_cost`` / ``imbalance_before``)
    normalised to the stage-1 anchor, plus ``stage2_start_winner``,
    ``opcost_from_{stage1,range,freqpres}`` (the three end states),
    ``opcost_range_start`` / ``opcost_freqpres_start`` (the two candidate
    plans' own OpCost — the guarantees' references, and the v4 comparison
    number for free), the per-branch move counts
    (``swaps_from_*``, ``sweeps_from_*``, ``swaps_range_balancer``,
    ``swaps_freqpres_plan``) and ``max_swaps_binding_any`` (True if ANY branch
    hit its bound, whereas ``max_swaps_binding`` describes the winner alone).
    Fields belonging to a start that was not run are ``nan`` / 0.

    ``max_swaps=0`` is :func:`operator_polish`'s pure-measurement contract and
    is honoured here: no candidate is built, nothing is compared, the input
    plan is reported unchanged.
    """
    unknown = [s for s in starts if s not in BEST_OF_N_STARTS]
    assert not unknown, (
        f"unknown start(s) {unknown} — expected a subset of {BEST_OF_N_STARTS}")
    assert "stage1" in starts, (
        "the stage-1 start is mandatory: it is the anchor every *_before "
        "field is normalised to")

    kw = dict(
        max_swaps=max_swaps, max_sweeps=max_sweeps, seed=seed,
        express_scale=express_scale, penalty_mx=penalty_mx,
        preserve_frequency=preserve_frequency, fixed_cost=fixed_cost,
        week_fixed=week_fixed, accept_eps=accept_eps,
    )
    chosen0 = sa_result["chosen"]

    from_s1 = operator_polish(
        {"chosen": chosen0}, plz_keys, plz_hub_arr, hub_plz_list,
        matrices, schedules, **kw)
    if max_swaps <= 0:
        return _measurement_only(from_s1, BEST_OF_N_STARTS, "stage1")

    ends: dict[str, dict] = {"stage1": from_s1}
    nan = float("nan")
    opcost_range_start = nan
    opcost_freqpres_start = nan
    swaps_range_balancer = 0
    swaps_freqpres_plan = 0

    if "range" in starts:
        rng_bal = balance_fleet_per_hub_ml(
            {"chosen": chosen0, "best_cost": 0.0}, plz_keys, plz_hub_arr,
            hub_plz_list, matrices, schedules,
            cost_budget_pct=range_budget_pct, max_swaps=range_max_swaps,
            seed=seed, express_scale=express_scale, penalty_mx=penalty_mx,
            preserve_frequency=preserve_frequency)
        swaps_range_balancer = int(rng_bal["swaps_made"])
        ends["range"] = operator_polish(
            {"chosen": rng_bal["chosen"]}, plz_keys, plz_hub_arr,
            hub_plz_list, matrices, schedules, **kw)
        # The polish's own warm-up measured the range plan before it moved
        # anything, on exactly the same caches.
        opcost_range_start = ends["range"]["opcost_before"]

    if "freqpres" in starts:
        # The v4 plan itself: the frequency-PRESERVING best-of-two, regardless
        # of this call's own `preserve_frequency` (with the pin on, the two
        # coincide). Only the polish that follows it is frequency-free.
        fp = operator_polish_best_of_two(
            {"chosen": chosen0}, plz_keys, plz_hub_arr, hub_plz_list,
            matrices, schedules,
            max_swaps=max_swaps, max_sweeps=max_sweeps, seed=seed,
            express_scale=express_scale, penalty_mx=penalty_mx,
            preserve_frequency=True, fixed_cost=fixed_cost,
            week_fixed=week_fixed, accept_eps=accept_eps,
            range_budget_pct=range_budget_pct, range_max_swaps=range_max_swaps)
        swaps_freqpres_plan = int(fp["swaps_made"])
        ends["freqpres"] = operator_polish(
            {"chosen": fp["chosen"]}, plz_keys, plz_hub_arr, hub_plz_list,
            matrices, schedules, **kw)
        opcost_freqpres_start = ends["freqpres"]["opcost_before"]

    order = [s for s in BEST_OF_N_STARTS if s in ends]
    winner = order[0]
    for name in order[1:]:
        if ends[name]["opcost_after"] < ends[winner]["opcost_after"]:
            winner = name
    res = dict(ends[winner])

    # Fail loud rather than silently shipping a plan one of the candidate
    # STATES already beats — the whole point of carrying them.
    best_end = min(e["opcost_after"] for e in ends.values())
    assert res["opcost_after"] <= best_end + 1e-9 * max(1.0, abs(best_end)), (
        f"best-of-{len(ends)} returned {res['opcost_after']:.6f} > the "
        f"cheapest branch {best_end:.6f}")
    for ref_name, ref in (("range balancer", opcost_range_start),
                          ("frequency-preserving best-of-two",
                           opcost_freqpres_start)):
        if np.isnan(ref):
            continue
        assert res["opcost_after"] <= ref + 1e-6 * max(1.0, abs(ref)), (
            f"best-of-{len(ends)} returned {res['opcost_after']:.6f} > the "
            f"{ref_name} state {ref:.6f} — operator_polish must never worsen "
            "OpCost from the start it is given")

    # ``before`` is always the stage-1 anchor, whichever branch won.
    for key in ("initial_total_cost", "imbalance_before", "opcost_before",
                "operator_cost_before", "variable_before",
                "sum_hub_peak_before", "vehicle_days_before",
                "penalty_before"):
        res[key] = from_s1[key]

    res.update({
        "stage2_start_winner": winner,
        "opcost_range_start": opcost_range_start,
        "opcost_freqpres_start": opcost_freqpres_start,
        "swaps_range_balancer": swaps_range_balancer,
        "swaps_freqpres_plan": swaps_freqpres_plan,
        "max_swaps_binding_any": any(e["max_swaps_binding"]
                                     for e in ends.values()),
    })
    for name in BEST_OF_N_STARTS:
        end = ends.get(name)
        res[f"opcost_from_{name}"] = end["opcost_after"] if end else nan
        res[f"swaps_from_{name}"] = int(end["swaps_made"]) if end else 0
        res[f"sweeps_from_{name}"] = int(end["sweeps"]) if end else 0

    log.debug(
        f"Operator polish (best-of-{len(ends)}, "
        f"{'frequency-preserving' if preserve_frequency else 'frequency-free'}"
        f"): start '{winner}' wins — "
        + ", ".join(f"{n} {ends[n]['opcost_after']:,.0f}" for n in order)
        + f" (range plan {opcost_range_start:,.0f}, v4/freqpres plan "
        f"{opcost_freqpres_start:,.0f})"
    )
    return res




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

    Since Task 6f this is not only the pre-6e ablation: it is also the second
    CANDIDATE START of the production polish
    (:func:`operator_polish_best_of_n`), so its own prose is production prose.

    When ``preserve_frequency`` is True, candidate swaps are restricted to
    schedules of the SAME size as the current one (same number of delivery days
    per week). Balancing is then FREQUENCY-PRESERVING: it redistributes WHICH
    days are served, never HOW MANY, so a cell's delivery frequency stays at
    its cost-optimal (init) value and the balancer cannot trade service
    quality for the (hub-bundled) routing savings of lower frequencies — the
    cost model used for selection is per-PLZ unbundled, so under bundling
    lower frequencies look cheaper and would otherwise be re-introduced here.

    It is NOT wait-preserving, and must never be described as
    "service-neutral". Schedules of equal size differ in average wait (size 3
    spans 0.50-0.67 days, size 4 spans 0.33-0.50), so moving a cell from
    ``{Mon, Wed, Fri}`` to ``{Mon, Tue, Thu}`` changes the service metric at
    constant frequency — measured on the v3 grid, the willing-weighted wait
    moves at stage 2 in 94 of 176 theta > 0 triples, by up to -17.65 %. In
    :func:`operator_polish` that change is PRICED (``Delta penalty`` is part of
    every accept decision). Here it is not: this function ranks candidates by
    fleet RANGE and only uses ``penalty_mx`` as a budget ceiling, so the wait
    may drift anywhere inside that budget. That is one more reason its output
    is a candidate start to be polished and re-priced, never a final plan.

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
    # Delivery-day mask: every per-cell fleet term below is masked with it,
    # so the pooled express vehicles added on top are not double-counted
    # (see _daily_fleet_per_hub).
    sa_mx = matrices.get("sched_active")
    if sa_mx is None:
        sa_mx = _sched_active_mask(schedules)

    chosen = sa_result["chosen"].copy()
    cur_dd_cost = float(dd_cost_mx[np.arange(n_plz), chosen].sum())
    express_pred_cache: dict = {}
    pool_pred_cache: dict = {}
    ecache = np.zeros((len(hub_plz_list), N_DAYS))
    pcache = np.zeros((len(hub_plz_list), N_DAYS))
    # I1 fix: pooled-vehicle mirror of ecache/pcache. Needed so the swap GATE
    # (not just the accepted state) can correct its tentative per-day fleet
    # row for the pooled vehicles a swap silently moves -- see _pool_veh_fn.
    pvcache = np.zeros((len(hub_plz_list), N_DAYS))

    def _pool_veh_fn(hi: int, d: int, ch: np.ndarray) -> float:
        """Pooled vehicles for hub `hi`/day `d`: express + small-delivery.

        Shares ``express_pred_cache`` / ``pool_pred_cache`` with the
        ``_hub_express_day_ml`` / ``_hub_smallday_pool_ml`` calls above and
        below, so vehicle counts come from cache hits rather than a second
        round of partition/surrogate calls.
        """
        return (
            _hub_express_vehicles(
                hi, d, ch, hub_plz_list, schedules, raw_express, matrices,
                express_pred_cache,
            )
            + _hub_delivery_pool_vehicles(
                hi, d, ch, hub_plz_list, schedules, matrices, pool_pred_cache,
            )
        )

    for hi in range(len(hub_plz_list)):
        for d in range(N_DAYS):
            ecache[hi, d] = _hub_express_day_ml(
                hi, d, chosen, hub_plz_list, schedules,
                raw_express, expr_stops, matrices,
                express_pred_cache, express_scale,
            )
            pcache[hi, d] = _hub_smallday_pool_ml(
                hi, d, chosen, hub_plz_list, schedules, matrices,
                pool_pred_cache,
            )
            pvcache[hi, d] = _pool_veh_fn(hi, d, chosen)

    cur_cost = cur_dd_cost + ecache.sum() + pcache.sum()
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

    fleet = _daily_fleet_per_hub(
        chosen, plz_hub_arr, hub_plz_list, veh_3d, schedules,
        pool_veh_fn=_pool_veh_fn, sched_active=sa_mx,
    )
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
        best_pool_vals: dict[int, float] = {}

        for new_si in range(len(schedules)):
            if new_si == old_si:
                continue
            if preserve_frequency and len(schedules[new_si]) != len(old_days):
                continue
            delta_c = dd_cost_mx[pi, new_si] - old_cost_pi
            delta_total = delta_c

            new_days = schedules[new_si]
            affected = old_days.symmetric_difference(new_days)
            pool_affected = _pool_affected_days(pi, old_si, new_si, matrices)
            new_expr_vals: dict[int, float] = {}
            new_pool_vals: dict[int, float] = {}
            new_pool_veh: dict[int, float] = {}
            if affected or pool_affected:
                chosen[pi] = new_si
                for d_aff in affected:
                    nv = _hub_express_day_ml(
                        hi, d_aff, chosen, hub_plz_list, schedules,
                        raw_express, expr_stops, matrices,
                        express_pred_cache, express_scale,
                    )
                    delta_total += nv - ecache[hi, d_aff]
                    new_expr_vals[d_aff] = nv
                for d_aff in pool_affected:
                    pv = _hub_smallday_pool_ml(
                        hi, d_aff, chosen, hub_plz_list, schedules,
                        matrices, pool_pred_cache,
                    )
                    delta_total += pv - pcache[hi, d_aff]
                    new_pool_vals[d_aff] = pv
                # I1 fix: still inside the TRIAL state, and served from the
                # caches the two loops above just warmed for this exact
                # (hi, d_aff, chosen) -- so this records the true pooled
                # vehicle count for the gate below, not just for the accepted
                # state. Express moves on the schedules' symmetric difference,
                # the delivery pool on _pool_affected_days (a day served by
                # BOTH schedules still changes the pool when the batched
                # demand behind it changes), hence the union.
                for d_aff in set(affected) | set(pool_affected):
                    new_pool_veh[d_aff] = _pool_veh_fn(hi, d_aff, chosen)
                chosen[pi] = old_si

            delta_obj = delta_total + (
                float(penalty_mx[pi, new_si] - penalty_mx[pi, old_si])
                if use_pen else 0.0
            )
            if cur_obj + delta_obj > max_obj:
                continue

            # Delivery-day-masked, to match how `fleet` was built (the
            # unmasked slice would re-introduce the express double count).
            old_veh = veh_3d[pi, old_si, :] * sa_mx[old_si]
            new_veh = veh_3d[pi, new_si, :] * sa_mx[new_si]
            new_fleet_hub = fleet[hi] - old_veh + new_veh
            # I1 fix: the veh_3d-only delta above misses the POOLED-vehicle
            # movement a swap causes (pi joins day d_aff's express pool when
            # it stops delivering there, leaves it when it starts; and its
            # small-delivery group re-forms around a changed demand) --
            # correct the tentative row with the true value before
            # ranking/gating.
            for d_aff, v in new_pool_veh.items():
                new_fleet_hub[d_aff] += v - pvcache[hi, d_aff]
            new_range = float(new_fleet_hub.max() - new_fleet_hub.min())
            if new_range < best_new_range:
                best_new_range = new_range
                best_new_si = new_si
                best_delta = delta_total
                best_delta_obj = delta_obj
                best_expr_vals = new_expr_vals
                best_pool_vals = new_pool_vals

        if best_new_si is not None:
            # D2 fix: full hub-row refresh (not the old incremental
            # veh_3d-only delta) so pooled vehicles — which can shift for
            # OTHER members of the hub's partitions, not just `pi` — stay
            # exact. Cheap: _pool_veh_fn hits the cache entries the candidate
            # evaluation above already populated for this exact
            # (hi, d, chosen) state.
            # I1 fix: pvcache is refreshed in the same loop (not just fleet)
            # so the NEXT swap's gate correction reads the post-accept
            # pooled-vehicle state, not a stale pre-swap one.
            chosen[pi] = best_new_si
            for d in range(N_DAYS):
                pv = _pool_veh_fn(hi, d, chosen)
                pvcache[hi, d] = pv
                deliv = h_ps[sa_mx[chosen[h_ps], d]]
                fleet[hi, d] = (
                    float(veh_3d[deliv, chosen[deliv], d].sum()) + pv)
            cur_cost += best_delta
            cur_obj += best_delta_obj
            for d_val, val in best_expr_vals.items():
                ecache[hi, d_val] = val
            for d_val, val in best_pool_vals.items():
                pcache[hi, d_val] = val
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
    sa_mx = matrices.get("sched_active")
    if sa_mx is None:
        sa_mx = _sched_active_mask(schedules)
    # Pruning-heuristic proxy only — see the pre-check below. Falls back to
    # veh_3d for hand-built matrices that predate the pooled-vehicle rule.
    veh_proxy = matrices.get("veh_3d_raw")
    if veh_proxy is None:
        veh_proxy = veh_3d

    chosen = chosen.copy()
    cur_dd_cost = float(dd_cost_mx[np.arange(n_plz), chosen].sum())
    express_pred_cache: dict = {}
    pool_pred_cache: dict = {}
    ecache = np.zeros((len(hub_plz_list), N_DAYS))
    pcache = np.zeros((len(hub_plz_list), N_DAYS))
    # I1 fix: pooled-vehicle mirror of ecache/pcache. Needed so the swap GATE
    # (not just the accepted state) can correct its tentative system-fleet row
    # for the pooled vehicles a swap silently moves -- see _pool_veh_fn below.
    pvcache = np.zeros((len(hub_plz_list), N_DAYS))

    def _pool_veh_fn(hi: int, d: int, ch: np.ndarray) -> float:
        """Pooled vehicles for hub `hi`/day `d`: express + small-delivery.

        Shares ``express_pred_cache`` / ``pool_pred_cache`` with the
        ``_hub_express_day_ml`` / ``_hub_smallday_pool_ml`` calls above and
        below, so vehicle counts come from cache hits rather than a second
        round of partition/surrogate calls.
        """
        return (
            _hub_express_vehicles(
                hi, d, ch, hub_plz_list, schedules, raw_express, matrices,
                express_pred_cache,
            )
            + _hub_delivery_pool_vehicles(
                hi, d, ch, hub_plz_list, schedules, matrices, pool_pred_cache,
            )
        )

    for hi in range(len(hub_plz_list)):
        for d in range(N_DAYS):
            ecache[hi, d] = _hub_express_day_ml(
                hi, d, chosen, hub_plz_list, schedules,
                raw_express, expr_stops, matrices,
                express_pred_cache, express_scale,
            )
            pcache[hi, d] = _hub_smallday_pool_ml(
                hi, d, chosen, hub_plz_list, schedules, matrices,
                pool_pred_cache,
            )
            pvcache[hi, d] = _pool_veh_fn(hi, d, chosen)

    cur_cost = cur_dd_cost + ecache.sum() + pcache.sum()
    initial_total_cost = cur_cost
    use_pen = penalty_mx is not None
    cur_pen = float(penalty_mx[np.arange(n_plz), chosen].sum()) if use_pen else 0.0
    cur_obj = cur_cost + cur_pen
    initial_obj = cur_obj
    max_obj = initial_obj * (1 + cost_budget_pct / 100.0)

    fleet = _daily_fleet_per_hub(
        chosen, plz_hub_arr, hub_plz_list, veh_3d, schedules,
        pool_veh_fn=_pool_veh_fn, sched_active=sa_mx,
    )
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
        best_pool_new: dict[tuple[int, int], float] = {}
        best_reduction = 0.0

        for pi in cand_plz:
            old_si = int(chosen[pi])
            old_days = schedules[old_si]
            old_size = len(old_days)
            hi = int(plz_hub_arr[pi])
            # Delivery-day-masked, to match how `fleet` was built.
            old_veh = veh_3d[pi, old_si, :] * sa_mx[old_si]
            old_veh_p = veh_proxy[pi, old_si, :] * sa_mx[old_si]

            for new_si in range(len(schedules)):
                if new_si == old_si:
                    continue
                new_days = schedules[new_si]
                # Frequency-preserving + must move off d_max
                if len(new_days) != old_size:
                    continue
                if d_max in new_days:
                    continue

                # System-spread of swap (cheap, pool-BLIND check before
                # paying for cost eval -- a pure pruning heuristic: it only
                # decides whether this candidate is worth the expensive
                # partition/surrogate evaluation below. Never used as the
                # final accept decision -- see the I1 fix after it.)
                # It runs on ``veh_proxy`` (the UNPOOLED per-cell count):
                # ``veh_3d`` is zero for a pooled small-delivery member, so a
                # cell whose every instance pools would show a zero delta here
                # and be pruned before its true (pooled) effect is ever
                # evaluated. The proxy only ranks candidates for evaluation;
                # the accept gate below stays exact.
                new_veh_p = veh_proxy[pi, new_si, :] * sa_mx[new_si]
                proxy_sys_fleet = sys_fleet + (new_veh_p - old_veh_p)
                new_spread = float(
                    proxy_sys_fleet.max() - proxy_sys_fleet.min())
                reduction = cur_spread - new_spread
                if reduction <= best_reduction:
                    continue

                # Cost & objective delta (full evaluation only for promising swaps)
                delta_cost = dd_cost_mx[pi, new_si] - dd_cost_mx[pi, old_si]
                affected = old_days.symmetric_difference(new_days)
                pool_affected = _pool_affected_days(
                    pi, old_si, new_si, matrices)
                expr_new: dict[tuple[int, int], float] = {}
                pool_new: dict[tuple[int, int], float] = {}
                pool_veh_new: dict[int, float] = {}
                if affected or pool_affected:
                    chosen[pi] = new_si
                    for d_aff in affected:
                        nv = _hub_express_day_ml(
                            hi, d_aff, chosen, hub_plz_list, schedules,
                            raw_express, expr_stops, matrices,
                            express_pred_cache, express_scale,
                        )
                        delta_cost += nv - ecache[hi, d_aff]
                        expr_new[(hi, d_aff)] = nv
                    for d_aff in pool_affected:
                        pv = _hub_smallday_pool_ml(
                            hi, d_aff, chosen, hub_plz_list, schedules,
                            matrices, pool_pred_cache,
                        )
                        delta_cost += pv - pcache[hi, d_aff]
                        pool_new[(hi, d_aff)] = pv
                    # I1 fix: still in the TRIAL state, served from the caches
                    # the two loops above just warmed for this exact
                    # (hi, d_aff, chosen) -- the true pooled-vehicle count for
                    # the exact gate correction below. Express moves on the
                    # symmetric difference, the delivery pool on
                    # _pool_affected_days, hence the union.
                    for d_aff in set(affected) | set(pool_affected):
                        pool_veh_new[d_aff] = _pool_veh_fn(hi, d_aff, chosen)
                    chosen[pi] = old_si

                delta_obj = float(delta_cost) + (
                    float(penalty_mx[pi, new_si] - penalty_mx[pi, old_si])
                    if use_pen else 0.0
                )
                if cur_obj + delta_obj > max_obj:
                    continue

                # I1 fix: the accept decision itself must be true-profile-
                # exact, not the blind pre-check above. Rebuild the tentative
                # SYSTEM fleet row from the POOLED-EXACT per-cell counts (only
                # hub `hi`'s entries can move -- both pools are priced per
                # hub), correct it with the true pooled-vehicle delta just
                # computed, then re-derive spread/reduction from that.
                true_sys_fleet = sys_fleet + (
                    veh_3d[pi, new_si, :] * sa_mx[new_si] - old_veh)
                for d_aff, v in pool_veh_new.items():
                    true_sys_fleet[d_aff] += v - pvcache[hi, d_aff]
                true_spread = float(true_sys_fleet.max() - true_sys_fleet.min())
                true_reduction = cur_spread - true_spread
                if true_reduction <= best_reduction:
                    continue

                best_reduction = true_reduction
                best_si = new_si
                best_pi = pi
                best_delta_obj = delta_obj
                best_delta_cost = float(delta_cost)
                best_expr_new = expr_new
                best_pool_new = pool_new

        if best_si is None:
            break

        # Apply best swap. D2 fix: full hub-row refresh (not the old
        # incremental veh_3d-only delta) so pooled vehicles — which can shift
        # for OTHER members of the hub's partitions, not just `best_pi` —
        # stay exact. Cheap: _pool_veh_fn hits the cache entries the candidate
        # evaluation above already populated for this exact (hi, d, chosen).
        # I1 fix: pvcache is refreshed in the same loop (not just fleet) so
        # the NEXT iteration's gate correction reads the post-accept
        # pooled-vehicle state, not a stale pre-swap one.
        hi = int(plz_hub_arr[best_pi])
        chosen[best_pi] = best_si
        h_ps = hub_plz_list[hi]
        for d in range(N_DAYS):
            pv = _pool_veh_fn(hi, d, chosen)
            pvcache[hi, d] = pv
            deliv = h_ps[sa_mx[chosen[h_ps], d]]
            fleet[hi, d] = float(veh_3d[deliv, chosen[deliv], d].sum()) + pv
        sys_fleet = fleet.sum(axis=0)
        cur_cost += best_delta_cost
        cur_obj += best_delta_obj
        for (hi_aff, d_aff), val in best_expr_new.items():
            ecache[hi_aff, d_aff] = val
        for (hi_aff, d_aff), val in best_pool_new.items():
            pcache[hi_aff, d_aff] = val
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
