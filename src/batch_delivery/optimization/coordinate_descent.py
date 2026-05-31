"""Coordinate-descent schedule optimiser (production path).

The CD optimiser sweeps (PLZ, schedule) cells one at a time, accepting
moves that reduce hub-bundled cost. ``_day_toggle_neighbors`` builds the
local neighbourhood (single-day flips inside the holding-day envelope),
and ``_pair_polish_round`` adds an O(K^2) pair-swap polish at the end.

Used by ``scripts/pipeline/02_optimize_grid.py`` as Stage 2 of the
paper pipeline.
"""

import time

import numpy as np

from batch_delivery.config.constants import (
    N_DAYS,
)
from batch_delivery.optimization.costs import _hub_express_day_ml
from batch_delivery.utils import log

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
