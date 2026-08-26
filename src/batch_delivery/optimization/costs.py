"""Cost-matrix construction (Daganzo and ML paths).

Builds the (n_plz, n_sched) cost / vehicle / wait matrices used by every
optimiser in the package. Two parallel paths are kept here because they
share the same vectorised demand/stops accumulator:

* ``build_cost_matrices`` + ``_hub_express_day`` — Daganzo continuum
  proxy. Legacy path, used for ablation against the ML surrogate.
* ``build_cost_matrices_ml`` + ``_hub_express_day_ml`` — production
  Daganzo-LGB-Hybrid surrogate. The main path for every paper number.

``compute_scenario_corrections`` is the Daganzo per-PLZ calibration
helper invoked when the legacy path is exercised.
"""


import numpy as np
import pandas as pd

from batch_delivery.config.constants import (
    COST_SCALE,
    FAST_SHARE_B2B,
    FAST_SHARE_B2C,
    MIN_TOUR_PARCELS,
    N_DAYS,
    VEHICLE_CAPACITY,
)
from batch_delivery.features import (
    _PROVIDER_IDX,
    ALL_COLS,
    TIER2_COLS,
    compute_tier2_features,
)
from batch_delivery.io.demand import compute_shifted_demand_plz, get_source_days
from batch_delivery.legacy.daganzo import CalibratedDaganzo, predict_vec
from batch_delivery.optimization.schedules import _compute_wait_mx
from batch_delivery.utils import log

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

    # ── 5b) Geometry lookups shared by every downstream consumer ──────
    # Built here (not after the ML block) so the ``n_active == 0`` early
    # return can hand back a dict with exactly the same keys as the full
    # return path — a missing key there used to surface as a KeyError deep
    # inside _hub_express_day_ml / price_group.
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

    # Per-cell centroid (first day with geometry) — the nearest-neighbour
    # seed coordinates ``build_partition`` packs tours by. Degree space on
    # purpose (see optimization/partition.py); cells without any geometry
    # keep (0, 0) and are therefore never picked as close neighbours.
    _cent_lon = np.zeros(n_plz, dtype=np.float64)
    _cent_lat = np.zeros(n_plz, dtype=np.float64)
    for pi in range(n_plz):
        for dd in range(N_DAYS):
            L = plz_day_lon[pi][dd]
            if len(L):
                _cent_lon[pi] = float(L.mean())
                _cent_lat[pi] = float(plz_day_lat[pi][dd].mean())
                break

    # B2C share per PLZ (for Tier 3)
    plz_total = daily_demand.sum(axis=1)
    plz_b2c_share = np.where(plz_total > 0, daily_b2c.sum(axis=1) / plz_total, 0.5)

    # ── 6) Active mask ────────────────────────────────────────────────
    active = combined_demand > 0
    n_active = int(active.sum())
    if n_active == 0:
        empty = {
            "dd_cost_mx": np.zeros((n_plz, n_sched)),
            "cost_3d": np.zeros((n_plz, n_sched, N_DAYS)),
            "cost_3d_raw": np.zeros((n_plz, n_sched, N_DAYS)),
            "veh_3d": np.zeros((n_plz, n_sched, N_DAYS)),
            "veh_3d_raw": np.zeros((n_plz, n_sched, N_DAYS)),
            "wait_mx": np.zeros((n_plz, n_sched)),
            "raw_express": np.zeros((n_plz, N_DAYS)),
            "expr_stops": np.zeros((n_plz, N_DAYS)),
            "express_cost": np.zeros((n_plz, N_DAYS)),
            "fast_share_blend_arr": fast_share_blend_arr,
            "area_arr": area_arr, "hd_arr": hd_arr,
            "sched_active": sched_active, "daily_demand": daily_demand,
            "small_delivery_mask": np.zeros(
                (n_plz, n_sched, N_DAYS), dtype=bool),
            "small_delivery_price": np.zeros((n_plz, n_sched, N_DAYS)),
            "combined_demand": combined_demand,
            "combined_stops": combined_stops,
            "_pool_dem": np.zeros((n_plz, n_sched, N_DAYS)),
            "_pool_stp": np.zeros((n_plz, n_sched, N_DAYS)),
            "_cent_lon": _cent_lon,
            "_cent_lat": _cent_lat,
            "plz_day_lon": plz_day_lon,
            "plz_day_lat": plz_day_lat,
            "plz_day_psd": plz_day_psd,
            "hub_lon_arr": hub_lon_arr,
            "hub_lat_arr": hub_lat_arr,
            "plz_b2c_share": plz_b2c_share,
            "ml_predictor": ml_predictor,
            "provider": provider,
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

    # (``plz_b2c_share`` for Tier 3 is computed in §5b, above the early return.)

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

    # ── 8b) Small-delivery rule (rev1 realistic tours) ────────────────
    # A delivery instance below one vehicle load never gets a tour of its
    # own — it rides with its neighbours. Such instances are therefore NOT
    # separable and must not appear in the per-cell matrix; the pooled twin
    # ``_hub_smallday_pool_ml`` prices them at the hub instead. The zeroing
    # happens BEFORE dd_cost_mx is summed, otherwise a small instance would
    # be charged twice (separable matrix AND pooled twin).
    #
    # Restricted to delivery days on purpose: on non-delivery days the
    # instance is an express residual, already pooled hub-wide by
    # ``_hub_express_day_ml`` off ``raw_express`` / ``express_cost``.
    # ``cost_3d_raw`` keeps the unpooled per-cell prediction for diagnostics.
    cost_3d_raw = cost_3d.copy()
    small_delivery_mask = (
        active
        & (combined_demand < MIN_TOUR_PARCELS)
        & sched_active[None, :, :]
    )
    cost_3d[small_delivery_mask] = 0.0        # pooled twin owns these
    # spec §4.3 v3: a pooled group is ONE tour, so its vehicles are counted by
    # the group (``_hub_delivery_pool_vehicles``), never per member — exactly
    # the rule the express partition already follows. The per-cell count would
    # otherwise charge one vehicle to each member of a shared tour.
    # ``veh_3d_raw`` keeps the unpooled per-cell count for diagnostics and for
    # the cheap pruning heuristics that only need a movement magnitude.
    veh_3d_raw = veh_3d.copy()
    veh_3d[small_delivery_mask] = 0.0

    # Pool signature per (plz, schedule, day): the parcels/stops this cell
    # contributes to the hub pool, 0 when it does not participate. Two
    # schedules with an identical signature leave the pool value untouched —
    # ``_pool_affected_days`` uses that to skip recomputation exactly.
    _pool_dem = np.where(small_delivery_mask, combined_demand, 0.0)
    _pool_stp = np.where(
        small_delivery_mask, np.maximum(1.0, combined_stops), 0.0)

    # ── 9) DD cost matrix & express arrays ────────────────────────────
    dd_cost_mx = (cost_3d * sched_active[None, :, :].astype(np.float64)).sum(axis=2)
    raw_express = (
        np.round(daily_b2c * fast_share_b2c)
        + np.round(daily_b2b * fast_share_b2b)
    )
    expr_stops = np.maximum(
        1.0, spd_arr[:, None] * fast_share_blend_arr[:, None] * np.ones((1, N_DAYS)),
    )

    # ── 9b) Per-cell express cost (rev1 realistic-tour rule) ────────────
    # The express instance is a *scaled single-day instance of the same
    # cell* — the pool's scale/p_keep augmentation family. Real hub_dist and
    # area (D3a fix), single-day tier2 geometry, psd stats scaled by the
    # standard share (D3b fix). G2: assert the domain, never extrapolate
    # silently.
    express_cost = np.zeros((n_plz, N_DAYS), dtype=np.float64)
    xi, xd = np.where(raw_express > 0)
    if len(xi):
        assert np.all(area_arr[xi] > 0), "G2: zero area in express instance"
        assert np.all(hd_arr[xi] > 0), "G2: zero hub_dist in express instance"
        xf = np.empty((len(xi), len(ALL_COLS)), dtype=np.float64)
        npx = raw_express[xi, xd]
        nsx = np.maximum(1.0, np.trunc(expr_stops[xi, xd]))
        arx = np.maximum(0.01, area_arr[xi])
        xf[:, 0] = np.trunc(npx)
        xf[:, 1] = nsx
        xf[:, 2] = arx
        xf[:, 3] = hd_arr[xi]
        xf[:, 4] = np.trunc(npx) / nsx
        xf[:, 5] = np.trunc(npx) / VEHICLE_CAPACITY
        xf[:, 6] = np.ceil(np.trunc(npx) / VEHICLE_CAPACITY)
        xf[:, 7] = np.trunc(npx) / arx
        xf[:, 8:8 + n_t2] = tier2_mx[xi, xd, :]
        b2cx = np.trunc(np.trunc(npx) * plz_b2c_share[xi])
        xf[:, 18] = b2cx / np.maximum(1.0, np.trunc(npx))
        xhp = _has_psd[xi, xd]
        xf[:, 19] = np.where(xhp, _psd_std[xi, xd] * fast_share_blend_arr[xi], 0.0)
        xf[:, 20] = np.where(xhp, _psd_max[xi, xd] * fast_share_blend_arr[xi],
                             np.trunc(npx))
        min_vx = np.maximum(1.0, np.ceil(np.trunc(npx) / VEHICLE_CAPACITY))
        xf[:, 21] = np.trunc(npx) / (min_vx * VEHICLE_CAPACITY)
        xf[:, 22] = float(_PROVIDER_IDX.get(provider, 0))
        xf[:, 23] = xd.astype(np.float64)
        xf[:, 24] = 1.0                      # single-day residual semantics
        express_cost[xi, xd] = ml_predictor.predict(
            pd.DataFrame(xf, columns=ALL_COLS))

    # ── 9c) Per-instance pooled-delivery prices (perf, bit-identical) ───
    # At ``head=None`` — the base run — ``price_group`` prices a pooled group
    # as the Sigma of its members' SINGLETON prices (surrogate/bundle.py:174-179),
    # so the price is partition-independent and every term depends only on
    # (plz, schedule, day). Precomputing that table once per matrix build turns
    # ``_hub_smallday_pool_ml`` — which misses its cache on essentially every
    # trial move — from "build_partition + one bundle_features + one surrogate
    # call PER MEMBER" into a Sigma over array lookups.
    #
    # Identical by construction rather than by re-derivation: the row below is
    # exactly the one ``price_group`` -> ``bundle_features`` builds for a lone
    # member of the pool (same kind, same parcels/stops overrides, freq=1.0),
    # and it is priced with the same ``predict_single``.
    from batch_delivery.surrogate.bundle import bundle_features  # cycle: import here
    small_delivery_price = np.zeros_like(cost_3d)
    # bundle_features reads only these keys for a delivery singleton; passing a
    # view keeps the precompute independent of the dict assembled below.
    _bf_view = {
        "plz_day_lon": plz_day_lon, "plz_day_lat": plz_day_lat,
        "plz_day_psd": plz_day_psd, "hub_lon_arr": hub_lon_arr,
        "hub_lat_arr": hub_lat_arr, "area_arr": area_arr, "hd_arr": hd_arr,
        "plz_b2c_share": plz_b2c_share, "daily_demand": daily_demand,
        "provider": provider, "fast_share_blend_arr": fast_share_blend_arr,
        "raw_express": raw_express, "expr_stops": expr_stops,
    }
    _sd_parcels = np.zeros(n_plz, dtype=np.float64)
    _sd_stops = np.zeros(n_plz, dtype=np.float64)
    _sz, _ss, _sd = np.where(small_delivery_mask)
    for _z, _s, _d in zip(_sz.tolist(), _ss.tolist(), _sd.tolist()):
        _sd_parcels[_z] = combined_demand[_z, _s, _d]
        _sd_stops[_z] = max(1.0, combined_stops[_z, _s, _d])
        small_delivery_price[_z, _s, _d] = float(ml_predictor.predict_single(
            bundle_features((_z,), _d, _bf_view, kind="delivery",
                            parcels_by_cell=_sd_parcels,
                            stops_by_cell=_sd_stops, freq=1.0)))
        _sd_parcels[_z] = 0.0
        _sd_stops[_z] = 0.0
    log.info("Pooled small-delivery prices: %d instance(s) precomputed",
             len(_sz))

    return {
        "dd_cost_mx": dd_cost_mx,
        "cost_3d": cost_3d,
        "cost_3d_raw": cost_3d_raw,
        "veh_3d": veh_3d,
        "veh_3d_raw": veh_3d_raw,
        "wait_mx": wait_mx,
        "raw_express": raw_express,
        "expr_stops": expr_stops,
        "express_cost": express_cost,
        "fast_share_blend_arr": fast_share_blend_arr,
        "area_arr": area_arr,
        "hd_arr": hd_arr,
        "sched_active": sched_active,
        "daily_demand": daily_demand,
        "small_delivery_mask": small_delivery_mask,
        "small_delivery_price": small_delivery_price,
        "combined_demand": combined_demand,
        "combined_stops": combined_stops,
        "_pool_dem": _pool_dem,
        "_pool_stp": _pool_stp,
        "_cent_lon": _cent_lon,
        "_cent_lat": _cent_lat,
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

def _express_partition(
    contributing: list[int], d: int,
    raw_express: np.ndarray, expr_stops: np.ndarray, matrices: dict,
) -> tuple[tuple[int, ...], ...]:
    """Who rides with whom in hub-day *d*'s express pool.

    Factored out of :func:`_hub_express_day_ml` so the cost path (which does
    not need it at ``head=None``) and the vehicle path (which always does) can
    call it independently. Pure function of its arguments — same inputs, same
    grouping, whoever asks first.
    """
    from batch_delivery.optimization.partition import build_partition
    return build_partition(
        np.array(contributing), raw_express[:, d], expr_stops[:, d],
        matrices["area_arr"], matrices["hd_arr"],
        matrices["_cent_lon"], matrices["_cent_lat"],
        pts_lon={z: matrices["plz_day_lon"][z][d] for z in contributing},
        pts_lat={z: matrices["plz_day_lat"][z][d] for z in contributing},
    )


def _express_partition_vehicles(
    parts: tuple[tuple[int, ...], ...], d: int, raw_express: np.ndarray,
) -> float:
    """Vehicles of an express partition: one ceil per tour, never summed first."""
    return float(sum(
        np.ceil(sum(np.trunc(raw_express[z, d]) for z in g) / VEHICLE_CAPACITY)
        for g in parts
    ))


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
    """Hub-level express cost for one day — partition first, then price.

    rev1 realistic-tour rule: the non-delivering cells of a hub do not merge
    into one giant instance. ``build_partition`` decides who rides with whom
    (min-load, stops/area/hull caps) and ``price_group`` prices each tour.
    Results are cached by ``(hub, day, contributing_plz)`` as
    ``(cost, vehicles)``; ``_hub_express_vehicles`` reads the second slot.
    """
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
    cache_key = (hi, d, frozenset(contributing))
    cached = express_cache.get(cache_key)
    if cached is not None:
        return cached[0] * express_scale

    head = matrices.get("bundle_head")        # None until Gate U passes
    if head is None:
        # Sigma-fallback regime: ``price_group(kind="express", head=None)``
        # returns ``sum(express_cost[z, d])`` over the group's members
        # (surrogate/bundle.py:169-173), so summing over ANY partition of
        # ``contributing`` gives the same total — the grouping is not
        # load-bearing for the PRICE. Skipping build_partition here (a
        # ConvexHull per candidate merge over up to ~47 co-hub cells) is what
        # makes the coordinate-descent inner loop affordable; the partition's
        # one real consumer, the vehicle count, is deferred to
        # ``_hub_express_vehicles``, which fills the ``None`` slot in place.
        # Only the summation ORDER differs from the partition path (cell order
        # instead of group by group) — the terms are identical.
        ec = matrices["express_cost"]
        cost = float(sum(ec[z, d] for z in contributing))
        express_cache[cache_key] = (cost, None)
        return cost * express_scale

    from batch_delivery.surrogate.bundle import price_group

    parts = _express_partition(contributing, d, raw_express, expr_stops,
                               matrices)
    cost = float(sum(
        price_group(g, d, matrices, kind="express", head=head) for g in parts))
    express_cache[cache_key] = (
        cost, _express_partition_vehicles(parts, d, raw_express))
    return cost * express_scale


def _hub_express_vehicles(
    hi: int, d: int, chosen: np.ndarray,
    hub_plz_list: list[np.ndarray],
    schedules: list[frozenset[int]],
    raw_express: np.ndarray,
    matrices: dict,
    express_cache: dict,
) -> float:
    """Vehicles of the hub's express partition on day *d*.

    Shares ``express_cache`` with :func:`_hub_express_day_ml` — one entry per
    ``(hub, day, contributing)`` holding ``(cost, vehicles)``, so asking for
    the fleet never costs a second round of surrogate calls.

    The vehicles slot is ``None`` when the cost was priced partition-free (see
    :func:`_hub_express_day_ml`); this function is the only consumer that needs
    the grouping, so it builds the partition then — once — and upgrades the
    entry in place. The value is the one the eager path used to store.
    """
    h_ps = hub_plz_list[hi]
    sa = matrices.get("sched_active")
    nd = ~sa[chosen[h_ps], d] if sa is not None else np.array(
        [d not in schedules[int(chosen[p])] for p in h_ps], dtype=bool)
    mask = nd & (raw_express[h_ps, d] > 0)
    if not mask.any():
        return 0.0
    contributing = h_ps[mask].tolist()
    key = (hi, d, frozenset(contributing))
    cached = express_cache.get(key)
    if cached is None:
        _hub_express_day_ml(
            hi, d, chosen, hub_plz_list, schedules, raw_express,
            matrices["expr_stops"], matrices, express_cache, 1.0)
        cached = express_cache[key]
    if cached[1] is None:                     # partition deferred — pay it now
        parts = _express_partition(
            contributing, d, raw_express, matrices["expr_stops"], matrices)
        cached = (cached[0], _express_partition_vehicles(parts, d, raw_express))
        express_cache[key] = cached
    return cached[1]


def _smallday_members(
    hi: int, d: int, chosen: np.ndarray,
    hub_plz_list: list[np.ndarray], matrices: dict,
) -> tuple[list[int], tuple | None]:
    """The hub-day's pooled small-delivery cells and their cache key.

    ``([], None)`` when the hub pools nothing that day (or the matrices
    predate the rule) — both twins then answer 0.0 without touching the
    cache. Shared so the cost twin and the vehicle twin can never derive
    different keys for the same hub-day state.

    The key carries ``(cell, schedule)`` pairs, not just the membership set
    the express twin keys on: a member's pooled demand depends on how many
    source days its own schedule holds behind this one.
    """
    mask3 = matrices.get("small_delivery_mask")
    if mask3 is None:                         # legacy / hand-built matrices
        return [], None
    h_ps = hub_plz_list[hi]
    sa = matrices["sched_active"]
    small = [int(z) for z in h_ps
             if sa[int(chosen[z]), d] and mask3[z, int(chosen[z]), d]]
    if not small:
        return [], None
    return small, (hi, d, tuple(sorted((z, int(chosen[z])) for z in small)))


def _smallday_partition(
    hi: int, d: int, chosen: np.ndarray, small: list[int], matrices: dict,
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray, np.ndarray]:
    """Who rides with whom in hub-day *d*'s pooled small-delivery groups.

    Returns ``(parts, parcels, stops)`` — the grouping plus the per-cell
    parcel/stop vectors it was built from, which the pricing path hands to
    ``price_group`` and the vehicle path reduces with one ceil per group.
    Factored out of :func:`_hub_smallday_pool_ml` so the cost path (which
    skips it at ``head=None``) and :func:`_hub_delivery_pool_vehicles` (which
    always needs it) build the identical grouping — same inputs, same tours,
    whoever asks first.

    ``hi`` is carried for call-site symmetry with the twins; the grouping is
    fully determined by ``(d, small, chosen)``.
    """
    from batch_delivery.optimization.partition import build_partition
    cd = matrices["combined_demand"]
    cs = matrices["combined_stops"]
    parcels = np.zeros(cd.shape[0])
    stops = np.zeros(cd.shape[0])
    for z in small:
        parcels[z] = cd[z, int(chosen[z]), d]
        stops[z] = max(1.0, cs[z, int(chosen[z]), d])
    parts = build_partition(
        np.array(small), parcels, stops, matrices["area_arr"],
        matrices["hd_arr"], matrices["_cent_lon"], matrices["_cent_lat"],
        pts_lon={z: matrices["plz_day_lon"][z][d] for z in small},
        pts_lat={z: matrices["plz_day_lat"][z][d] for z in small},
    )
    return parts, parcels, stops


def _delivery_partition_vehicles(
    parts: tuple[tuple[int, ...], ...], parcels: np.ndarray,
) -> float:
    """Vehicles of a pooled delivery partition: one ceil per tour.

    The delivery-side twin of :func:`_express_partition_vehicles` — the group
    is ONE tour, so its members' parcels are summed BEFORE the ceil, never
    rounded up individually (spec §4.3 v3).
    """
    return float(sum(
        np.ceil(sum(np.trunc(parcels[z]) for z in g) / VEHICLE_CAPACITY)
        for g in parts
    ))


def _hub_smallday_pool_ml(
    hi: int, d: int, chosen: np.ndarray,
    hub_plz_list: list[np.ndarray],
    schedules: list[frozenset[int]],
    matrices: dict,
    pool_cache: dict,
) -> float:
    """Delivery-day twin: pool sub-threshold DELIVERING instances (rev1 rule).

    Instances zeroed out of ``cost_3d`` by ``small_delivery_mask`` are priced
    here instead: they are partitioned with their co-located neighbours at the
    same hub and each resulting tour is priced once.

    Demand depends on each member's schedule (own arrivals + held willing
    ones), so the cache key carries ``(cell, schedule)`` pairs — not just the
    membership set the express twin keys on.

    Results are cached as ``(cost, vehicles)``;
    :func:`_hub_delivery_pool_vehicles` reads the second slot.

    At ``head=None`` the group price is a Sigma over per-member singleton
    prices, so the partition is skipped and the precomputed
    ``matrices["small_delivery_price"]`` table is summed instead; see the
    fast-path comment below and ``build_cost_matrices_ml`` §9c.

    OPEN ITEM (spec'd, not yet resolved): ``freq=1.0`` is passed to
    ``bundle_features`` even though the pooled parcels ARE batched over
    ``n_source`` days — the separable matrix feeds the surrogate
    ``1 + willing_blend * (n_source - 1)`` for the very same instance
    (§7 Tier 3). A group can mix members on different schedules, so the
    generalisation needs a weighting rule rather than a copy of the scalar.
    Until that is decided, pooled instances are priced as single-day ones.
    """
    small, key = _smallday_members(hi, d, chosen, hub_plz_list, matrices)
    if not small:
        return 0.0
    hit = pool_cache.get(key)
    if hit is not None:
        return hit[0]

    head = matrices.get("bundle_head")
    sdp = matrices.get("small_delivery_price")
    if head is None and sdp is not None:
        # Sigma-fallback regime: ``price_group(kind="delivery", head=None)``
        # prices a group as the sum of its members' SINGLETON prices
        # (surrogate/bundle.py:174-179), and a singleton's price depends only
        # on (cell, schedule, day) — never on who else is in the group. The
        # table was precomputed in ``build_cost_matrices_ml`` §9c from exactly
        # that expression, so the partition, the per-member ``bundle_features``
        # and the per-member surrogate call are all dead work here. This cache
        # misses on essentially every trial move (the key carries each member's
        # schedule), which is why the miss path had to become cheap.
        # Summation order (cell order instead of group by group) is the only
        # difference from the partition path.
        cost = float(sum(sdp[z, int(chosen[z]), d] for z in small))
        # Partition deferred: its one remaining consumer is the vehicle
        # count, which fills the None slot in place (see the twin below).
        pool_cache[key] = (cost, None)
        return cost

    from batch_delivery.surrogate.bundle import price_group

    parts, parcels, stops = _smallday_partition(
        hi, d, chosen, small, matrices)
    cost = float(sum(
        price_group(g, d, matrices, kind="delivery",
                    parcels_by_cell=parcels, stops_by_cell=stops,
                    freq=1.0, head=head)
        for g in parts
    ))
    pool_cache[key] = (cost, _delivery_partition_vehicles(parts, parcels))
    return cost


def _hub_delivery_pool_vehicles(
    hi: int, d: int, chosen: np.ndarray,
    hub_plz_list: list[np.ndarray],
    schedules: list[frozenset[int]],
    matrices: dict,
    pool_cache: dict,
) -> float:
    """Vehicles of the hub's pooled small-delivery groups on day *d*.

    spec §4.3 v3: a pooled group is ONE tour and contributes
    ``ceil(Sigma_members trunc(combined_demand[z, chosen[z], d]) / Q)``
    vehicles — the same counting rule the express partition already uses,
    the same rule VROOM's ``n_routes`` reports in validation. Singleton
    (>= MIN_TOUR_PARCELS) delivery instances keep their per-cell ``veh_3d``
    count, which ``build_cost_matrices_ml`` zeroes for pooled members so the
    two never overlap.

    Shares ``pool_cache`` with :func:`_hub_smallday_pool_ml` — one entry per
    hub-day state holding ``(cost, vehicles)``. The vehicles slot is ``None``
    when the cost was priced partition-free (``head=None``); this function is
    the only consumer that needs the grouping, so it builds the partition
    then — once — and upgrades the entry in place. The value is the one the
    eager (``head`` installed) path stores.
    """
    small, key = _smallday_members(hi, d, chosen, hub_plz_list, matrices)
    if not small:
        return 0.0
    cached = pool_cache.get(key)
    if cached is None:
        _hub_smallday_pool_ml(
            hi, d, chosen, hub_plz_list, schedules, matrices, pool_cache)
        cached = pool_cache[key]
    if cached[1] is None:                     # partition deferred — pay it now
        parts, parcels, _ = _smallday_partition(
            hi, d, chosen, small, matrices)
        cached = (cached[0], _delivery_partition_vehicles(parts, parcels))
        pool_cache[key] = cached
    return cached[1]


def _pool_affected_days(
    pi: int, old_si: int, new_si: int, matrices: dict,
) -> list[int]:
    """Days whose small-delivery pool changes when *pi* moves old_si -> new_si.

    NOT the schedules' symmetric difference: a day served by BOTH schedules
    still changes the pool, because the batched demand of that day depends on
    how many source days the schedule holds behind it. A day is unaffected
    only when *pi*'s participation AND its pooled demand/stops are identical
    under both schedules.
    """
    dem = matrices.get("_pool_dem")
    if dem is None:                           # legacy / hand-built matrices
        return []
    stp = matrices["_pool_stp"]
    diff = (dem[pi, old_si] != dem[pi, new_si]) | (stp[pi, old_si] != stp[pi, new_si])
    return np.flatnonzero(diff).tolist()
