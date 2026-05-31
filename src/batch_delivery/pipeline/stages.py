"""Seven pipeline stages, one ``step_*`` function each.

Each stage takes a :class:`PipelineState`, mutates ``state.artefacts``,
and returns the same state. Stages can be invoked individually from
notebooks or tests; the orchestrator chains them.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from batch_delivery.config.constants import (
    EXPRESS_SCENARIOS,
    FAST_SHARE_B2B,
    FAST_SHARE_B2C,
    N_DAYS,
    NON_BASELINE_SCENARIOS,
    RESULTS_DIR,
    SC_BASELINE,
    SC_FIXED_BATCH,
    SC_FIXED_EXPRESS,
    SC_SA_ML_BATCH,
    SC_SA_ML_EXPRESS,
    SCENARIO_NAMES,
    provider_to_demand_prefix,
)
from batch_delivery.utils import (
    compute_weighted_speed_factor,
    get_logger,
    load_checkpoint,
    save_checkpoint,
)

log = get_logger(__name__, level=logging.INFO)

from batch_delivery.pipeline.state import (
    PipelineState,
)

# ─── Stage 1: demand + hub network ──────────────────────────────────────────


def step_load_demand_and_hubs(state: PipelineState) -> PipelineState:
    """Load HAGRID demand and hub assignments per provider.

    Mirrors notebook cell 3 (``01_demand`` checkpoint).
    """
    from batch_delivery.io import demand, hubs

    ck = load_checkpoint("01_demand")
    if ck is not None:
        provider_data = ck["provider_data"]
        gdf_plz = ck["gdf_plz"]
        log.info("[checkpoint] Restored %d providers", len(provider_data))
    else:
        gdf_plz = demand.load_plz_areas()
        provider_data: dict[str, dict] = {}
        for provider in state.config.providers:
            daily_gdfs, plz_day_b2c, plz_day_b2b = demand.load_daily_demand(lsp_prefix=provider)
            gdf_provider = demand.build_unified_gdf(daily_gdfs)
            gdf_provider, plz_day_b2c, plz_day_b2b, merge_map = demand.merge_small_plz(
                gdf_provider, gdf_plz, plz_day_b2c, plz_day_b2b, daily_gdfs=daily_gdfs,
            )
            all_plz_set = set(gdf_provider["plz"].unique()) - {"00000"}

            # Optional subset filter (used by integration tests).
            if state.config.subset_plz:
                keep = set(state.config.subset_plz)
                all_plz_set &= keep
                if not all_plz_set:
                    raise ValueError(
                        f"subset_plz {sorted(keep)} produced empty PLZ set for {provider}"
                    )
                gdf_provider = gdf_provider[gdf_provider["plz"].isin(all_plz_set)].copy()
                plz_day_b2c = {pc: v for pc, v in plz_day_b2c.items() if pc in all_plz_set}
                plz_day_b2b = {pc: v for pc, v in plz_day_b2b.items() if pc in all_plz_set}
                for d in list(daily_gdfs.keys()):
                    daily_gdfs[d] = daily_gdfs[d][daily_gdfs[d]["plz"].isin(all_plz_set)].copy()
                log.info("  %s: subset_plz reduced to %d PLZ", provider, len(all_plz_set))

            demand.fill_missing_days(all_plz_set, plz_day_b2c, plz_day_b2b)

            gdf_hubs = hubs.load_hubs(provider=provider)
            df_assignments = hubs.assign_plz_to_hubs(
                gdf_plz, gdf_hubs, all_plz_set, merge_map=merge_map,
            )

            plz_demand, fast_share, weekday_shares = demand.compute_demand_stats(
                all_plz_set, plz_day_b2c, plz_day_b2b, df_assignments
            )
            daily_gdfs_wgs = demand.convert_daily_gdfs_to_wgs(daily_gdfs)

            provider_data[provider] = {
                "daily_gdfs": daily_gdfs,
                "daily_gdfs_wgs": daily_gdfs_wgs,
                "gdf_provider": gdf_provider,
                "gdf_hubs": gdf_hubs,
                "df_assignments": df_assignments,
                "plz_day_b2c": plz_day_b2c,
                "plz_day_b2b": plz_day_b2b,
                "plz_demand": plz_demand,
                "all_plz_set": all_plz_set,
                "fast_share": fast_share,
                "weekday_shares": weekday_shares,
                "merge_map": merge_map,
            }
            log.info(
                "  %s: %d PLZ, %d hubs, %d weekly parcels",
                provider,
                len(all_plz_set),
                df_assignments["hub_name"].nunique(),
                int(plz_demand["weekly_parcels"].sum()),
            )
        save_checkpoint("01_demand", provider_data=provider_data, gdf_plz=gdf_plz)

    state.artefacts["provider_data"] = provider_data
    state.artefacts["gdf_plz"] = gdf_plz
    return state


# ─── Stage 2: two-pass baseline VROOM solve ─────────────────────────────────


def step_solve_baseline(state: PipelineState) -> PipelineState:
    """Two-pass baseline: raw (speed_factor=1) → derive wsf → traffic-adjusted.

    Mirrors notebook cell 5 (``02_baseline`` checkpoint).
    """
    from batch_delivery.routing import (
        build_scenario_requests,
        parse_routes,
        solve_scenario,
    )
    from batch_delivery.routing.core import compute_baseline_job_caps

    provider_data: dict[str, dict] = state.artefacts["provider_data"]

    ck = load_checkpoint("02_baseline")
    if ck is not None:
        state.artefacts.update({
            "provider_data": ck["provider_data"],
            "baseline_job_caps": ck["baseline_job_caps"],
            "df_routes_baseline": ck["df_routes_baseline"],
            "baseline_solutions_by_provider": ck["baseline_solutions_by_provider"],
            "baseline_solve_by_provider": ck["baseline_solve_by_provider"],
            "wsf": ck.get("wsf", 1.0),
            "wtf": ck.get("wtf", 0.0),
        })
        log.info("[checkpoint] Restored baseline — %d routes", len(ck["df_routes_baseline"]))
        return state

    # ── Pass 1: raw baseline, speed_factor=1.0 ───────────────────────────
    log.info("Step 2a · Raw baseline (no traffic, speed_factor=1.0)")
    raw_routes_parts = []
    for i, (provider, pdata) in enumerate(provider_data.items(), 1):
        log.info("  [%d/%d] %s (%d PLZ × %d days)",
                 i, len(provider_data), provider, len(pdata["all_plz_set"]), N_DAYS)
        baseline_schedules = {pc: set(range(N_DAYS)) for pc in sorted(pdata["all_plz_set"])}
        raw_requests, _ = build_scenario_requests(
            baseline_schedules, pdata["df_assignments"], pdata["daily_gdfs_wgs"],
            fast_share=1.0, speed_factor=1.0,
            scenario_name=f"{SC_BASELINE}_raw {provider}",
        )
        raw_solutions, _ = solve_scenario(
            raw_requests, f"{SC_BASELINE}_raw {provider}",
            cache_tag=f"baseline_raw_{provider.lower()}",
            save_intermediate=RESULTS_DIR / "baseline" / f"{provider.lower()}_raw",
        )
        df_raw = parse_routes(raw_solutions, SC_BASELINE, pdata["df_assignments"],
                              provider_name=provider)
        raw_routes_parts.append(df_raw)

    df_routes_raw = pd.concat(raw_routes_parts, ignore_index=True)
    wtf, wsf = compute_weighted_speed_factor(df_routes_raw)
    log.info("  ➜ Weighted speed factor: %.3f (traffic factor %.4f)", wsf, wtf)

    # ── Pass 2: traffic-adjusted baseline ─────────────────────────────────
    log.info("Step 2b · Traffic baseline (speed_factor=%.3f)", wsf)
    baseline_routes_parts = []
    baseline_solutions_by_provider: dict[str, Any] = {}
    baseline_solve_by_provider: dict[str, Any] = {}
    baseline_job_caps: dict[str, dict] = {}

    for i, (provider, pdata) in enumerate(provider_data.items(), 1):
        log.info("  [%d/%d] %s (%d PLZ × %d days)",
                 i, len(provider_data), provider, len(pdata["all_plz_set"]), N_DAYS)
        baseline_schedules = {pc: set(range(N_DAYS)) for pc in sorted(pdata["all_plz_set"])}
        baseline_requests, _ = build_scenario_requests(
            baseline_schedules, pdata["df_assignments"], pdata["daily_gdfs_wgs"],
            fast_share=1.0, speed_factor=wsf,
            scenario_name=f"{SC_BASELINE} {provider}",
        )
        baseline_solutions, df_baseline_solve = solve_scenario(
            baseline_requests, f"{SC_BASELINE} {provider}",
            cache_tag=f"baseline_{provider.lower()}",
            save_intermediate=RESULTS_DIR / "baseline" / provider.lower(),
        )
        df_routes_provider = parse_routes(
            baseline_solutions, SC_BASELINE, pdata["df_assignments"], provider_name=provider,
        )

        pdata["baseline_schedules"] = baseline_schedules
        pdata["baseline_routes"] = df_routes_provider
        pdata["baseline_solutions"] = baseline_solutions
        pdata["baseline_solve"] = df_baseline_solve
        baseline_solutions_by_provider[provider] = baseline_solutions
        baseline_solve_by_provider[provider] = df_baseline_solve
        baseline_routes_parts.append(df_routes_provider)
        baseline_job_caps[provider] = compute_baseline_job_caps(baseline_solutions)
        log.info("      %d routes | %.0f parcels | %.0f km",
                 len(df_routes_provider),
                 df_routes_provider["parcels"].sum(),
                 df_routes_provider["distance_km"].sum())

    df_routes_baseline = pd.concat(baseline_routes_parts, ignore_index=True)
    log.info("Baseline complete | %d routes | wsf=%.3f", len(df_routes_baseline), wsf)

    save_checkpoint(
        "02_baseline",
        provider_data=provider_data,
        baseline_job_caps=baseline_job_caps,
        df_routes_baseline=df_routes_baseline,
        baseline_solutions_by_provider=baseline_solutions_by_provider,
        baseline_solve_by_provider=baseline_solve_by_provider,
        wsf=wsf, wtf=wtf,
    )

    state.artefacts.update({
        "baseline_job_caps": baseline_job_caps,
        "df_routes_baseline": df_routes_baseline,
        "baseline_solutions_by_provider": baseline_solutions_by_provider,
        "baseline_solve_by_provider": baseline_solve_by_provider,
        "wsf": wsf,
        "wtf": wtf,
    })
    return state


# ─── Stage 3: optimisation prep ────────────────────────────────────────────


def step_prepare_optimisation(state: PipelineState) -> PipelineState:
    """Pre-compute per-provider PLZ structures, schedule list, hub arrays, coords.

    Mirrors notebook cell 8 §1–§2 (``04_optim_prep`` checkpoint).
    """
    from batch_delivery.io import demand as demand_io
    from batch_delivery.optimization import build_hub_arrays, enumerate_valid_schedules

    provider_data = state.artefacts["provider_data"]
    gdf_plz = state.artefacts["gdf_plz"]

    ck = load_checkpoint("04_optim_prep")
    if ck is not None:
        optimization_data = ck["optimization_data"]
        log.info("[checkpoint] Restored optimisation prep for %d providers", len(optimization_data))
    else:
        optimization_data = {}
        for provider, pdata in provider_data.items():
            plz_data = demand_io.prepare_plz_data(
                pdata["all_plz_set"], pdata["df_assignments"], gdf_plz,
                pdata["gdf_provider"], pdata["daily_gdfs_wgs"],
                pdata["plz_day_b2c"], pdata["plz_day_b2b"],
                merge_map=pdata.get("merge_map"),
            )
            plz_keys = sorted(plz_data.keys())
            schedules = enumerate_valid_schedules()
            plz_hub_arr, hub_plz_list, hub_names = build_hub_arrays(
                plz_keys, pdata["df_assignments"],
            )
            optimization_data[provider] = {
                "plz_data": plz_data,
                "plz_keys": plz_keys,
                "schedules": schedules,
                "plz_hub_arr": plz_hub_arr,
                "hub_plz_list": hub_plz_list,
                "hub_names": hub_names,
            }
            log.info("  %s: %d PLZ × %d schedules × %d hubs",
                     provider, len(plz_keys), len(schedules), len(hub_names))
        save_checkpoint("04_optim_prep", optimization_data=optimization_data)

    # ── Build plz_day_coords and hub_coords_by_plz per provider ──────────
    ml_prep: dict[str, dict] = {}
    for provider in state.config.providers:
        pdata = provider_data[provider]
        odata = optimization_data[provider]

        hub_coords_by_plz: dict[str, tuple[float, float]] = {}
        for _, hr in pdata["df_assignments"].iterrows():
            hub_coords_by_plz[hr["plz"]] = (hr["hub_lon"], hr["hub_lat"])

        prefix = provider_to_demand_prefix(provider)
        col_total = f"{prefix}_total"
        daily_wgs = pdata["daily_gdfs_wgs"]
        plz_day_coords: dict[str, dict] = {}

        for plz_code in odata["plz_keys"]:
            plz_day_coords[plz_code] = {}
            for d in range(N_DAYS):
                gdf_d = daily_wgs.get(d)
                if gdf_d is None:
                    continue
                pts = gdf_d[gdf_d["plz"] == plz_code]
                if len(pts) == 0:
                    continue
                lons = pts["lon"].values.astype(np.float64)
                lats = pts["lat"].values.astype(np.float64)
                psd = (
                    pts[col_total].values.astype(np.float64)
                    if col_total in pts.columns else np.ones(len(pts))
                )
                plz_day_coords[plz_code][d] = (lons, lats, psd)

        ml_prep[provider] = {
            "plz_day_coords": plz_day_coords,
            "hub_coords_by_plz": hub_coords_by_plz,
        }

    state.artefacts["optimization_data"] = optimization_data
    state.artefacts["ml_prep"] = ml_prep
    return state


# ─── Stage 4: train MLP surrogate ──────────────────────────────────────────


def step_train_surrogate(state: PipelineState) -> PipelineState:
    """Build the baseline feature matrix and fit the 5-seed MLP ensemble.

    Mirrors notebook cell 8 §3 (``07_ml_predictor`` checkpoint).
    """
    from batch_delivery.features import ALL_COLS, build_feature_matrix
    from batch_delivery.surrogate import MLCostPredictor

    provider_data = state.artefacts["provider_data"]
    optimization_data = state.artefacts["optimization_data"]
    gdf_plz = state.artefacts["gdf_plz"]
    df_routes_baseline = state.artefacts["df_routes_baseline"]

    baseline_sched_by_provider: dict[str, dict] = {}
    for provider in state.config.providers:
        baseline_sched_by_provider[provider] = {
            SC_BASELINE: {
                pc: set(range(N_DAYS)) for pc in optimization_data[provider]["plz_keys"]
            }
        }
    baseline_routes_dict = {SC_BASELINE: df_routes_baseline}

    ck = load_checkpoint("07_ml_predictor")
    if ck is not None:
        ml_predictor = ck["ml_predictor"]
        log.info("[checkpoint] ML predictor loaded: %d pipelines, %d combo features",
                 len(ml_predictor.pipelines), len(ml_predictor.combo_cols))
    else:
        log.info("Step 4 · Training initial ML model from baseline routes …")
        t0 = time.perf_counter()
        df_train = build_feature_matrix(
            provider_data=provider_data,
            optimization_data=optimization_data,
            cal_by_provider=None,
            all_routes=baseline_routes_dict,
            scenario_schedules_by_provider=baseline_sched_by_provider,
            gdf_plz=gdf_plz,
            providers=list(state.config.providers),
            scenarios=[SC_BASELINE],
            tier=3,
        )
        X_train = df_train[ALL_COLS].copy()
        y_train = df_train["actual_cost"].values
        log.info("  Feature matrix: %d rows × %d features", len(df_train), X_train.shape[1])

        sc = state.config.surrogate
        ml_predictor = MLCostPredictor.train(
            X_train, y_train,
            alpha=sc.alpha,
            arch=tuple(sc.hidden_layers),
        )
        pred = ml_predictor.predict(X_train)
        mape = float(np.mean(np.abs((y_train - pred) / np.clip(y_train, 1, None))) * 100)
        r2 = 1 - np.sum((y_train - pred) ** 2) / np.sum((y_train - y_train.mean()) ** 2)
        log.info("  Trained in %.0fs | MAPE=%.2f%% | R²=%.4f",
                 time.perf_counter() - t0, mape, r2)
        save_checkpoint("07_ml_predictor", ml_predictor=ml_predictor)
        ml_predictor.save(RESULTS_DIR / "checkpoints" / "ml_predictor.pkl")

    state.artefacts["ml_predictor"] = ml_predictor
    state.artefacts["baseline_sched_by_provider"] = baseline_sched_by_provider
    return state


# ─── Stage 5: ML-CD optimisation ───────────────────────────────────────────


def step_optimize(state: PipelineState) -> PipelineState:
    """Coordinate descent over the 39 valid weekly schedules for each provider.

    Produces ``scenario_schedules_by_provider`` for the four non-baseline
    scenarios (Fixed/Batch × Express/Batch-Only).

    Mirrors notebook cell 10 (``08_sa_ml_optimization`` checkpoint).
    """
    from batch_delivery.optimization import (
        balance_fleet_per_hub_ml,
        build_cost_matrices_ml,
        build_fixed_schedules,
        optimize_cd_ml,
    )

    provider_data = state.artefacts["provider_data"]
    optimization_data = state.artefacts["optimization_data"]
    ml_prep = state.artefacts["ml_prep"]
    ml_predictor = state.artefacts["ml_predictor"]

    scenario_schedules_by_provider: dict[str, dict] = {}
    for provider in state.config.providers:
        scenario_schedules_by_provider[provider] = {
            SC_BASELINE: {
                pc: set(range(N_DAYS)) for pc in optimization_data[provider]["plz_keys"]
            }
        }

    ck = load_checkpoint("08_sa_ml_optimization")
    if ck is not None:
        for p in state.config.providers:
            scenario_schedules_by_provider[p].update(ck["ml_schedules"][p])
        ml_optimization_data = ck["ml_optimization_data"]
        log.info("[checkpoint] Restored ML-CD optimisation")
    else:
        ml_optimization_data: dict[str, dict] = {}
        log.info("Step 5 · ML-CD optimisation")
        for i, provider in enumerate(state.config.providers, 1):
            odata = optimization_data[provider]
            prep = ml_prep[provider]
            plz_keys = odata["plz_keys"]
            schedules = odata["schedules"]
            log.info("  [%d/%d] %s (%d PLZ)",
                     i, len(state.config.providers), provider, len(plz_keys))

            fixed_schedules = build_fixed_schedules(plz_keys, carrier=provider)
            fixed_batch_schedules = build_fixed_schedules(plz_keys, carrier=provider)

            sched_to_idx = {s: si for si, s in enumerate(schedules)}
            fixed_asn = np.array(
                [sched_to_idx.get(frozenset(fixed_schedules.get(pk, set(range(N_DAYS)))), 0)
                 for pk in plz_keys],
                dtype=np.intp,
            )

            matrices_ml_expr = build_cost_matrices_ml(
                plz_keys, odata["plz_data"], schedules, ml_predictor,
                provider, prep["plz_day_coords"], prep["hub_coords_by_plz"],
                fast_share_b2c=FAST_SHARE_B2C, fast_share_b2b=FAST_SHARE_B2B,
            )
            matrices_ml_batch = build_cost_matrices_ml(
                plz_keys, odata["plz_data"], schedules, ml_predictor,
                provider, prep["plz_day_coords"], prep["hub_coords_by_plz"],
                fast_share_b2c=0.0, fast_share_b2b=0.0,
            )

            sa_ml_expr = optimize_cd_ml(
                plz_keys, odata["plz_hub_arr"], odata["hub_plz_list"],
                matrices_ml_expr, schedules, batch_only=False,
                fixed_assignment=fixed_asn,
                pair_polish=state.config.optimization.pair_polish,
                pair_polish_rounds=state.config.optimization.pair_polish_rounds,
                pair_polish_max_pairs=state.config.optimization.pair_polish_max_pairs,
            )
            bal_ml_expr = balance_fleet_per_hub_ml(
                sa_ml_expr, plz_keys, odata["plz_hub_arr"], odata["hub_plz_list"],
                matrices_ml_expr, schedules,
            )
            sa_ml_batch = optimize_cd_ml(
                plz_keys, odata["plz_hub_arr"], odata["hub_plz_list"],
                matrices_ml_batch, schedules, batch_only=True,
                fixed_assignment=fixed_asn,
            )
            bal_ml_batch = balance_fleet_per_hub_ml(
                sa_ml_batch, plz_keys, odata["plz_hub_arr"], odata["hub_plz_list"],
                matrices_ml_batch, schedules,
            )

            scenario_schedules_by_provider[provider][SC_FIXED_EXPRESS] = fixed_schedules
            scenario_schedules_by_provider[provider][SC_SA_ML_EXPRESS] = bal_ml_expr["schedules_per_plz"]
            scenario_schedules_by_provider[provider][SC_FIXED_BATCH] = fixed_batch_schedules
            scenario_schedules_by_provider[provider][SC_SA_ML_BATCH] = bal_ml_batch["schedules_per_plz"]

            ml_optimization_data[provider] = {
                "matrices_ml_expr": matrices_ml_expr,
                "matrices_ml_batch": matrices_ml_batch,
                "sa_ml_expr": sa_ml_expr, "bal_ml_expr": bal_ml_expr,
                "sa_ml_batch": sa_ml_batch, "bal_ml_batch": bal_ml_batch,
                "fixed_schedules": fixed_schedules,
                "fixed_batch_schedules": fixed_batch_schedules,
            }
            log.info("      ML+Express %.0f EUR | ML Batch %.0f EUR",
                     bal_ml_expr["cost"], bal_ml_batch["cost"])

            # ── Export per-provider optimisation tables + plots ─────────
            try:
                from batch_delivery.optimization.results import (
                    export_optimization_results,
                )
                opt_root = state.out_dir / "optimization" / provider.lower()
                for sc_tag, sc_result, sc_matrices in (
                    ("sa_ml_express", bal_ml_expr, matrices_ml_expr),
                    ("sa_ml_batch", bal_ml_batch, matrices_ml_batch),
                ):
                    export_optimization_results(
                        plz_keys=plz_keys,
                        plz_hub_arr=odata["plz_hub_arr"],
                        hub_names=odata["hub_names"],
                        hub_plz_list=odata["hub_plz_list"],
                        schedules=schedules,
                        matrices=sc_matrices,
                        cd_result=sc_result,
                        out_dir=opt_root / sc_tag,
                        summary_extras={
                            "provider": provider,
                            "scenario": sc_tag,
                            "total_cost": float(sc_result.get("cost",
                                                              sc_result.get("best_cost", 0.0))),
                        },
                    )
            except Exception as exc:
                log.warning("export_optimization_results failed for %s: %s",
                            provider, exc)

        save_checkpoint(
            "08_sa_ml_optimization",
            ml_schedules={
                p: {
                    SC_SA_ML_EXPRESS: scenario_schedules_by_provider[p][SC_SA_ML_EXPRESS],
                    SC_SA_ML_BATCH: scenario_schedules_by_provider[p][SC_SA_ML_BATCH],
                }
                for p in state.config.providers
            },
            ml_optimization_data=ml_optimization_data,
        )

    state.artefacts["scenario_schedules_by_provider"] = scenario_schedules_by_provider
    state.artefacts["ml_optimization_data"] = ml_optimization_data
    return state


# ─── Stage 6: VROOM solve for every non-baseline scenario ───────────────────


def step_solve_scenarios(state: PipelineState) -> PipelineState:
    """Resolve VRPs for every (provider, scenario) outside the baseline.

    Mirrors notebook cell 15 (``06_scenario_routing`` checkpoint).
    """
    from batch_delivery.routing import (
        build_scenario_requests,
        parse_routes,
        solve_scenario,
    )
    from batch_delivery.routing.core import MAX_JOBS_PER_REQUEST

    provider_data = state.artefacts["provider_data"]
    scenario_schedules_by_provider = state.artefacts["scenario_schedules_by_provider"]
    baseline_job_caps = state.artefacts["baseline_job_caps"]
    wsf = state.artefacts["wsf"]

    ck = load_checkpoint("06_scenario_routing")
    if ck is not None:
        state.artefacts.update({
            "all_routes": ck["all_routes"],
            "all_solutions_by_provider": ck["all_solutions_by_provider"],
            "df_solve_summary": ck["df_solve_summary"],
        })
        log.info("[checkpoint] Restored scenario routing — %d scenarios", len(ck["all_routes"]))
        return state

    all_routes_parts: dict[str, list] = {name: [] for name in SCENARIO_NAMES}
    all_solutions_by_provider: dict[tuple[str, str], Any] = {}
    scenario_solve_dfs = []

    all_caps = [c for caps in baseline_job_caps.values() for c in caps.values()]
    global_job_cap = max(all_caps) if all_caps else MAX_JOBS_PER_REQUEST
    log.info("Step 6 · Scenario routing (global job cap = %d)", global_job_cap)

    for i, (provider, pdata) in enumerate(provider_data.items(), 1):
        log.info("  [%d/%d] %s (%d PLZ × %d scenarios)",
                 i, len(provider_data), provider,
                 len(pdata["all_plz_set"]), len(NON_BASELINE_SCENARIOS))

        all_routes_parts[SC_BASELINE].append(pdata["baseline_routes"])
        all_solutions_by_provider[(provider, SC_BASELINE)] = pdata["baseline_solutions"]
        provider_caps = baseline_job_caps.get(provider, {})

        for sc_name in NON_BASELINE_SCENARIOS:
            sc_schedules = scenario_schedules_by_provider[provider][sc_name]
            sc_fast_share = pdata["fast_share"] if sc_name in EXPRESS_SCENARIOS else 0.0
            cache_tag = f"{provider.lower()}_{sc_name.lower().replace(' ', '_').replace('+', 'plus')}"
            result_dir = (state.out_dir / sc_name.lower().replace(" ", "_").replace("+", "plus")
                          / provider.lower())

            sc_requests, _ = build_scenario_requests(
                sc_schedules, pdata["df_assignments"], pdata["daily_gdfs_wgs"],
                fast_share=sc_fast_share,
                scenario_name=f"{sc_name} {provider}",
                speed_factor=wsf,
                max_jobs_per_plz=provider_caps,
            )
            sc_solutions, df_sc_solve = solve_scenario(
                sc_requests, f"{sc_name} {provider}",
                cache_tag=cache_tag, save_intermediate=result_dir,
            )
            df_sc_routes = parse_routes(sc_solutions, sc_name, pdata["df_assignments"],
                                         provider_name=provider)
            all_routes_parts[sc_name].append(df_sc_routes)
            all_solutions_by_provider[(provider, sc_name)] = sc_solutions
            scenario_solve_dfs.append(df_sc_solve.assign(provider=provider, scenario=sc_name))
            n_ok = (len(df_sc_solve[df_sc_solve["status"].isin(["OK", "CACHED"])])
                    if len(df_sc_solve) > 0 else 0)
            log.info("      %-20s %d/%d OK", sc_name, n_ok, len(df_sc_solve))

    all_routes = {
        name: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for name, parts in all_routes_parts.items()
    }
    df_solve_summary = (pd.concat(scenario_solve_dfs, ignore_index=True)
                        if scenario_solve_dfs else pd.DataFrame())

    save_checkpoint(
        "06_scenario_routing",
        all_routes=all_routes,
        all_solutions_by_provider=all_solutions_by_provider,
        df_solve_summary=df_solve_summary,
    )
    state.artefacts.update({
        "all_routes": all_routes,
        "all_solutions_by_provider": all_solutions_by_provider,
        "df_solve_summary": df_solve_summary,
    })
    return state


# ─── Stage 7: KPIs + comparison ────────────────────────────────────────────


def step_evaluate(state: PipelineState) -> PipelineState:
    """Compute scenario-level + per-provider KPIs and write reports.

    Mirrors notebook cell 17 (no checkpoint; cheap to recompute).
    """
    from batch_delivery.evaluation import (
        compare_scenarios,
        compute_scenario_kpis,
        save_kpis,
    )
    from batch_delivery.io.demand import compute_avg_waiting_days

    provider_data = state.artefacts["provider_data"]
    optimization_data = state.artefacts["optimization_data"]
    scenario_schedules_by_provider = state.artefacts["scenario_schedules_by_provider"]
    all_routes = state.artefacts["all_routes"]

    # Demand-weighted waiting time per scenario (and per provider).
    scenario_waiting: dict[str, float] = {SC_BASELINE: 0.0}
    provider_waiting_rows = []
    for sc_name in NON_BASELINE_SCENARIOS:
        total_w = 0.0
        weighted_wait = 0.0
        for provider, pdata in provider_data.items():
            sc_schedules = scenario_schedules_by_provider[provider][sc_name]
            weekly_map = pdata["plz_demand"].set_index("plz")["weekly_parcels"].to_dict()
            fs_b2c = FAST_SHARE_B2C if sc_name in EXPRESS_SCENARIOS else 0.0
            fs_b2b = FAST_SHARE_B2B if sc_name in EXPRESS_SCENARIOS else 0.0
            p_weight = 0.0
            p_weighted = 0.0
            for pc in optimization_data[provider]["plz_keys"]:
                if pc not in sc_schedules:
                    continue
                wait_days = compute_avg_waiting_days(
                    sc_schedules[pc],
                    pdata["plz_day_b2c"].get(pc, {}),
                    pdata["plz_day_b2b"].get(pc, {}),
                    fs_b2c, fs_b2b,
                )
                w = float(weekly_map.get(pc, 0.0))
                p_weight += w
                p_weighted += wait_days * w
                total_w += w
                weighted_wait += wait_days * w
            provider_waiting_rows.append({
                "provider": provider, "scenario": sc_name,
                "customer_wait_days": round(p_weighted / max(1.0, p_weight), 3),
            })
        scenario_waiting[sc_name] = weighted_wait / max(1.0, total_w)

    df_provider_wait = pd.DataFrame(provider_waiting_rows)

    kpi_list = []
    provider_kpi_list = []
    for sc_name, df_r in all_routes.items():
        kpi_list.append(compute_scenario_kpis(
            df_r, sc_name, avg_waiting_days=scenario_waiting.get(sc_name, 0.0),
        ))
        for provider in state.config.providers:
            if "provider" not in df_r.columns:
                continue
            df_provider_routes = df_r[df_r["provider"] == provider]
            if len(df_provider_routes) == 0:
                continue
            provider_wait = 0.0
            if sc_name != SC_BASELINE:
                hit = df_provider_wait[
                    (df_provider_wait["provider"] == provider)
                    & (df_provider_wait["scenario"] == sc_name)
                ]
                if len(hit) > 0:
                    provider_wait = float(hit.iloc[0]["customer_wait_days"])
            pk = compute_scenario_kpis(df_provider_routes, sc_name, avg_waiting_days=provider_wait)
            pk["provider"] = provider
            provider_kpi_list.append(pk)

    df_kpi = compare_scenarios(kpi_list, baseline_name=SC_BASELINE)
    df_kpi_provider = pd.DataFrame(provider_kpi_list)

    state.out_dir.mkdir(parents=True, exist_ok=True)
    save_kpis(df_kpi)  # writes to legacy RESULTS_DIR/scenario_comparison_kpis.csv
    df_kpi.to_csv(state.out_dir / "scenario_comparison_kpis.csv", index=False)
    df_kpi_provider.to_csv(state.out_dir / "scenario_comparison_kpis_by_provider.csv", index=False)
    log.info("  Wrote KPIs to %s", state.out_dir)

    state.artefacts["df_kpi"] = df_kpi
    state.artefacts["df_kpi_provider"] = df_kpi_provider
    state.artefacts["scenario_waiting"] = scenario_waiting
    return state
