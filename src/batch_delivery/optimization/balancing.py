"""Fleet balancing per hub + system-level smoothing.

Two-stage balancing:

* :func:`balance_fleet_per_hub` / :func:`balance_fleet_per_hub_ml` —
  swap-based postprocessing that equalises daily vehicle counts within
  each hub while respecting a cost-increase budget.
* :func:`system_smooth_pass` — system-level smoothing that exchanges
  schedules across hubs when the imbalance crosses a threshold.

``_daily_fleet_per_hub`` and ``_fleet_imbalance`` are vectorised
helpers used by both stages.
"""


import numpy as np
from tqdm.auto import tqdm

from batch_delivery.config.constants import (
    FLEET_BALANCE_MAX_SWAPS,
    N_DAYS,
    SA_SEED,
)
from batch_delivery.optimization.costs import (
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
    express_veh_fn=None,
    sched_active: np.ndarray | None = None,
) -> np.ndarray:
    """Compute daily vehicle count per hub: shape (n_hubs, N_DAYS).

    A hub's express partition (rev1 hub-bundled non-delivery tours, Task 4)
    runs invisibly to the per-cell count unless ``express_veh_fn(hi, d,
    chosen) -> float`` is supplied — it adds that hub-day's POOLED vehicles
    on top (D2 fix). ``express_veh_fn=None`` reproduces the legacy count over
    the raw ``veh_3d`` slice (the non-ML ``balance_fleet_per_hub`` Daganzo
    path keeps this).

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
    if express_veh_fn is not None:
        sa = _sched_active_mask(schedules) if sched_active is None else sched_active
        plz_veh = plz_veh * sa[chosen, :]
    fleet = np.zeros((n_hubs, N_DAYS), dtype=np.float64)
    np.add.at(fleet, plz_hub_arr, plz_veh)
    if express_veh_fn is not None:
        for hi in range(n_hubs):
            for d in range(N_DAYS):
                fleet[hi, d] += express_veh_fn(hi, d, chosen)
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
    # I1 fix: express-vehicle mirror of ecache. Needed so the swap GATE (not
    # just the accepted state) can correct its tentative per-day fleet row for
    # the express vehicles a swap silently moves -- see _express_veh_fn below.
    evcache = np.zeros((len(hub_plz_list), N_DAYS))

    def _express_veh_fn(hi: int, d: int, ch: np.ndarray) -> float:
        """Express-partition vehicles for hub `hi`/day `d` (D2 fix).

        Shares ``express_pred_cache`` with the ``_hub_express_day_ml`` calls
        above and below, so vehicle counts come from cache hits rather than a
        second round of partition/surrogate calls.
        """
        return _hub_express_vehicles(
            hi, d, ch, hub_plz_list, schedules, raw_express, matrices,
            express_pred_cache,
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
            evcache[hi, d] = _express_veh_fn(hi, d, chosen)

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
        express_veh_fn=_express_veh_fn, sched_active=sa_mx,
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
            new_expr_veh: dict[int, float] = {}
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
                    # I1 fix: pure cache hit -- the _hub_express_day_ml call
                    # just above already warmed express_pred_cache for this
                    # exact (hi, d_aff, chosen) state, so this costs nothing
                    # extra. Records the true express-vehicle count so the
                    # gate below can be corrected, not just the accepted state.
                    new_expr_veh[d_aff] = _express_veh_fn(hi, d_aff, chosen)
                for d_aff in pool_affected:
                    pv = _hub_smallday_pool_ml(
                        hi, d_aff, chosen, hub_plz_list, schedules,
                        matrices, pool_pred_cache,
                    )
                    delta_total += pv - pcache[hi, d_aff]
                    new_pool_vals[d_aff] = pv
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
            # I1 fix: the veh_3d-only delta above misses the express-vehicle
            # movement a swap causes (pi joins day d_aff's express pool when
            # it stops delivering there, leaves it when it starts) -- correct
            # the tentative row with the true value before ranking/gating.
            for d_aff, v in new_expr_veh.items():
                new_fleet_hub[d_aff] += v - evcache[hi, d_aff]
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
            # veh_3d-only delta) so express-partition vehicles — which can
            # shift for OTHER members of the hub's non-delivery partition,
            # not just `pi` — stay exact. Cheap: express_veh_fn hits the
            # express_pred_cache entries the candidate evaluation above
            # already populated for this exact (hi, d, chosen) state.
            # I1 fix: evcache is refreshed in the same loop (not just fleet)
            # so the NEXT swap's gate correction reads the post-accept
            # express-vehicle state, not a stale pre-swap one.
            chosen[pi] = best_new_si
            for d in range(N_DAYS):
                ev = _express_veh_fn(hi, d, chosen)
                evcache[hi, d] = ev
                deliv = h_ps[sa_mx[chosen[h_ps], d]]
                fleet[hi, d] = (
                    float(veh_3d[deliv, chosen[deliv], d].sum()) + ev)
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

    chosen = chosen.copy()
    cur_dd_cost = float(dd_cost_mx[np.arange(n_plz), chosen].sum())
    express_pred_cache: dict = {}
    pool_pred_cache: dict = {}
    ecache = np.zeros((len(hub_plz_list), N_DAYS))
    pcache = np.zeros((len(hub_plz_list), N_DAYS))
    # I1 fix: express-vehicle mirror of ecache. Needed so the swap GATE (not
    # just the accepted state) can correct its tentative system-fleet row for
    # the express vehicles a swap silently moves -- see _express_veh_fn below.
    evcache = np.zeros((len(hub_plz_list), N_DAYS))

    def _express_veh_fn(hi: int, d: int, ch: np.ndarray) -> float:
        """Express-partition vehicles for hub `hi`/day `d` (D2 fix).

        Shares ``express_pred_cache`` with the ``_hub_express_day_ml`` calls
        above and below, so vehicle counts come from cache hits rather than a
        second round of partition/surrogate calls.
        """
        return _hub_express_vehicles(
            hi, d, ch, hub_plz_list, schedules, raw_express, matrices,
            express_pred_cache,
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
            evcache[hi, d] = _express_veh_fn(hi, d, chosen)

    cur_cost = cur_dd_cost + ecache.sum() + pcache.sum()
    initial_total_cost = cur_cost
    use_pen = penalty_mx is not None
    cur_pen = float(penalty_mx[np.arange(n_plz), chosen].sum()) if use_pen else 0.0
    cur_obj = cur_cost + cur_pen
    initial_obj = cur_obj
    max_obj = initial_obj * (1 + cost_budget_pct / 100.0)

    fleet = _daily_fleet_per_hub(
        chosen, plz_hub_arr, hub_plz_list, veh_3d, schedules,
        express_veh_fn=_express_veh_fn, sched_active=sa_mx,
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

            for new_si in range(len(schedules)):
                if new_si == old_si:
                    continue
                new_days = schedules[new_si]
                # Frequency-preserving + must move off d_max
                if len(new_days) != old_size:
                    continue
                if d_max in new_days:
                    continue

                # System-spread of swap (cheap, express-BLIND check before
                # paying for cost eval -- a pure pruning heuristic: it only
                # decides whether this candidate is worth the expensive
                # partition/surrogate evaluation below. Never used as the
                # final accept decision -- see the I1 fix after it.)
                new_veh = veh_3d[pi, new_si, :] * sa_mx[new_si]
                new_sys_fleet = sys_fleet + (new_veh - old_veh)
                new_spread = float(new_sys_fleet.max() - new_sys_fleet.min())
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
                expr_veh_new: dict[int, float] = {}
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
                        # I1 fix: pure cache hit (the _hub_express_day_ml call
                        # above just warmed this exact (hi, d_aff, chosen)
                        # key) -- the true express-vehicle count, for the
                        # exact gate correction below.
                        expr_veh_new[d_aff] = _express_veh_fn(hi, d_aff, chosen)
                    for d_aff in pool_affected:
                        pv = _hub_smallday_pool_ml(
                            hi, d_aff, chosen, hub_plz_list, schedules,
                            matrices, pool_pred_cache,
                        )
                        delta_cost += pv - pcache[hi, d_aff]
                        pool_new[(hi, d_aff)] = pv
                    chosen[pi] = old_si

                delta_obj = float(delta_cost) + (
                    float(penalty_mx[pi, new_si] - penalty_mx[pi, old_si])
                    if use_pen else 0.0
                )
                if cur_obj + delta_obj > max_obj:
                    continue

                # I1 fix: the accept decision itself must be true-profile-
                # exact, not the blind pre-check above. Correct the tentative
                # SYSTEM fleet row (only hub `hi`'s entries can move -- express
                # is priced per hub) with the true express-vehicle delta just
                # computed, then re-derive spread/reduction from that.
                true_sys_fleet = new_sys_fleet.copy()
                for d_aff, v in expr_veh_new.items():
                    true_sys_fleet[d_aff] += v - evcache[hi, d_aff]
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
        # incremental veh_3d-only delta) so express-partition vehicles —
        # which can shift for OTHER members of the hub's non-delivery
        # partition, not just `best_pi` — stay exact. Cheap: express_veh_fn
        # hits the express_pred_cache entries the candidate evaluation above
        # already populated for this exact (hi, d, chosen) state.
        # I1 fix: evcache is refreshed in the same loop (not just fleet) so
        # the NEXT iteration's gate correction reads the post-accept
        # express-vehicle state, not a stale pre-swap one.
        hi = int(plz_hub_arr[best_pi])
        chosen[best_pi] = best_si
        h_ps = hub_plz_list[hi]
        for d in range(N_DAYS):
            ev = _express_veh_fn(hi, d, chosen)
            evcache[hi, d] = ev
            deliv = h_ps[sa_mx[chosen[h_ps], d]]
            fleet[hi, d] = float(veh_3d[deliv, chosen[deliv], d].sum()) + ev
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
