"""Run the full schedule-optimisation pipeline using the production LGB-logT
surrogate, then add a custom WORST-CASE scenario for both extremes.

Pre-conditions:
  * Oracle-loop process has stopped (no VROOM contention).
  * Production LGB-logT model is trained (or this script trains it).
  * VROOM + Valhalla docker services are up (port 3000 + 8002).

What it does:
  1.   (Optional) train production LGB-logT on the final pool.
  2.   Run pipeline stages 1-3 (load demand, baseline VROOM, optim prep).
  3.   Inject LGB-logT as ml_predictor (skip MLP retraining).
  4.   Run stage 5 (ML coordinate-descent optimisation).
  5.   Build WORST-case schedules: per (provider, PLZ), pick the schedule with
       max ML-predicted cost. This represents the antipode of optimisation.
  6.   Run stage 6 (VROOM resolve) for all stock scenarios.
  7.   Add worst-case scenarios manually and VROOM-solve them.
  8.   Stage 7 + custom KPI table (Baseline vs best vs worst).

Output:
  results/final_optimization/
      scenario_comparison_kpis.csv
      scenario_comparison_kpis_by_provider.csv
      worst_vs_best_per_lsp.csv
      worst_vs_best_summary.json
      figures/ ...
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from batch_delivery.config import load_config
from batch_delivery.config.constants import (
    FAST_SHARE_B2B, FAST_SHARE_B2C, N_DAYS, RESULTS_DIR,
    SC_BASELINE, SC_FIXED_EXPRESS, SC_SA_ML_EXPRESS,
    SC_FIXED_BATCH, SC_SA_ML_BATCH, NON_BASELINE_SCENARIOS, EXPRESS_SCENARIOS,
)
# Monkey-patch: drop Express scenarios entirely (user decision after Amazon's
# single-hub express bundling hit VROOM budget). Only batch scenarios remain.
import batch_delivery.pipeline as _pl
_pl.NON_BASELINE_SCENARIOS = [SC_FIXED_BATCH, SC_SA_ML_BATCH]
_pl.EXPRESS_SCENARIOS = set()
from batch_delivery.pipeline import (
    PipelineState, step_load_demand_and_hubs, step_solve_baseline,
    step_prepare_optimisation, step_optimize, step_solve_scenarios, step_evaluate,
)
from batch_delivery.routing import build_scenario_requests, parse_routes, solve_scenario
from batch_delivery.evaluation import compare_scenarios, compute_scenario_kpis
from batch_delivery.io.demand import compute_avg_waiting_days
from batch_delivery.optimization import build_fixed_schedules
from batch_delivery.utils import get_logger
from batch_delivery.surrogate.lgb_adapter import LGBLogTSurrogate

log = get_logger(__name__)

# Custom scenario names (best / avg / worst — all batch-only)
SC_WORST_EXPRESS  = "Worst + Express"
SC_WORST_BATCH    = "Worst Batch-Only"
SC_AVG_BATCH      = "Avg Batch-Only"      # median-cost schedule per PLZ

CONFIG_PATH = ROOT / "conf" / "default.yaml"
RUN_DIR     = ROOT / "results" / "oracle_loop_extended_2026_05_22"
# Prefer the v3 model (trained on merge-fix sweep). Fall back to v2 if not yet trained.
SWEEP_V3_DIR = ROOT / "results" / "sweep_v3_mergefix"
LGB_V3_PATH  = SWEEP_V3_DIR / "production_lgb_logT_v3.pkl"
LGB_V2_PATH  = RUN_DIR / "production_lgb_logT_v2.pkl"
LGB_PATH     = LGB_V3_PATH if LGB_V3_PATH.exists() else LGB_V2_PATH
OUT_DIR      = ROOT / "results" / "final_optimization_v3_mergefix"
OUT_DIR.mkdir(parents=True, exist_ok=True)
log.info("Output dir: %s", OUT_DIR)


# ── helper: build worst-case schedule per PLZ from ML cost matrix ───────────
def build_worst_schedules(matrices_ml, schedules, plz_keys):
    """For each PLZ, pick the schedule with the maximum ML-predicted cost."""
    cost_3d = matrices_ml["cost_3d"]      # (n_plz, n_schedules, n_days)
    cost_2d = cost_3d.sum(axis=2)         # (n_plz, n_schedules)
    worst_schedule_idx = cost_2d.argmax(axis=1)
    return {plz_keys[i]: schedules[int(worst_schedule_idx[i])] for i in range(len(plz_keys))}


def build_average_schedules(matrices_ml, schedules, plz_keys):
    """For each PLZ, pick the *median-cost* schedule (50th-percentile by predicted total cost).

    This represents the 'average' schedule that a random / uninformed planner
    might pick — sits between the optimum (SA_ML) and the worst case.
    """
    cost_3d = matrices_ml["cost_3d"]      # (n_plz, n_schedules, n_days)
    cost_2d = cost_3d.sum(axis=2)         # (n_plz, n_schedules)
    # rank schedules per PLZ by cost ascending; pick the middle rank
    ranks = cost_2d.argsort(axis=1)       # (n_plz, n_schedules)
    median_rank = ranks[:, ranks.shape[1] // 2]    # middle-rank schedule index
    return {plz_keys[i]: schedules[int(median_rank[i])] for i in range(len(plz_keys))}


# ── stage runner with checkpoint reporting ─────────────────────────────────
def run_stage(stage_fn, state, label):
    t0 = time.perf_counter()
    log.info("► %s", label)
    state = stage_fn(state)
    log.info("✓ %s done in %.0fs", label, time.perf_counter() - t0)
    return state


def main():
    log.info("=" * 70)
    log.info("FINAL OPTIMISATION PIPELINE V2 (augmented LGB-logT)")
    log.info("=" * 70)

    # ── v3 mergefix: ALL checkpoints already archived externally; pipeline
    #    will rebuild Stage 1-8 with the corrected merge_small_plz that
    #    propagates plz codes to daily_gdfs (see src/batch_delivery/io/demand.py
    #    fix 2026-05-25 — was dropping ~61% of merged-cluster demand from routing).
    log.info("v3 mergefix run: expecting Stage 1-8 to rebuild from scratch")

    # ── boot the pipeline state ────────────────────────────────────────
    cfg = load_config(CONFIG_PATH)
    state = PipelineState(config=cfg, out_dir=OUT_DIR)

    # ── stages 1-3 (cached where possible) ─────────────────────────────
    state = run_stage(step_load_demand_and_hubs, state, "Stage 1 · demand + hubs")
    state = run_stage(step_solve_baseline,        state, "Stage 2 · baseline VROOM")
    state = run_stage(step_prepare_optimisation,  state, "Stage 3 · optim prep")

    # ── inject LGB-logT instead of training MLP ────────────────────────
    log.info("► Stage 4 · loading production LGB-logT surrogate (no MLP retrain)")
    if not LGB_PATH.exists():
        raise FileNotFoundError(
            f"Production LGB model not found at {LGB_PATH}. "
            "Run scripts/train_production_lgb.py first."
        )
    state.artefacts["ml_predictor"] = LGBLogTSurrogate.load(LGB_PATH)
    state.artefacts["baseline_sched_by_provider"] = {
        p: {SC_BASELINE: {pc: set(range(N_DAYS))
                          for pc in state.artefacts["optimization_data"][p]["plz_keys"]}}
        for p in cfg.providers
    }
    log.info("✓ LGB-logT loaded (%d combo features)",
              len(state.artefacts["ml_predictor"].combo_cols))

    # ── stage 5 · ML coordinate-descent optimisation ────────────────────
    state = run_stage(step_optimize, state, "Stage 5 · ML coordinate descent")

    # ── Patch: when stage 5 loaded from cache, the Fixed-* scenarios
    #   are missing because save_checkpoint("08_sa_ml_optimization", ...)
    #   only persists the SA_ML schedules. Re-build them here so stage 6
    #   doesn't KeyError on 'Fixed + Express' / 'Fixed Batch-Only'.
    sched_by_prov = state.artefacts["scenario_schedules_by_provider"]
    for provider in cfg.providers:
        plz_keys = state.artefacts["optimization_data"][provider]["plz_keys"]
        if SC_FIXED_EXPRESS not in sched_by_prov[provider]:
            sched_by_prov[provider][SC_FIXED_EXPRESS] = build_fixed_schedules(plz_keys, carrier=provider)
        if SC_FIXED_BATCH not in sched_by_prov[provider]:
            sched_by_prov[provider][SC_FIXED_BATCH] = build_fixed_schedules(plz_keys, carrier=provider)
    log.info("✓ Fixed-* schedules ensured for all providers")

    # ── BUILD WORST-CASE SCHEDULES ─────────────────────────────────────
    log.info("► Building WORST-case schedules per (provider, PLZ)")
    ml_opt = state.artefacts["ml_optimization_data"]
    odata  = state.artefacts["optimization_data"]
    scenario_schedules_by_provider = state.artefacts["scenario_schedules_by_provider"]

    for provider in cfg.providers:
        plz_keys = odata[provider]["plz_keys"]
        schedules = odata[provider]["schedules"]
        worst_expr = build_worst_schedules(
            ml_opt[provider]["matrices_ml_expr"], schedules, plz_keys)
        worst_batch = build_worst_schedules(
            ml_opt[provider]["matrices_ml_batch"], schedules, plz_keys)
        avg_batch = build_average_schedules(
            ml_opt[provider]["matrices_ml_batch"], schedules, plz_keys)
        scenario_schedules_by_provider[provider][SC_WORST_EXPRESS] = worst_expr
        scenario_schedules_by_provider[provider][SC_WORST_BATCH]   = worst_batch
        scenario_schedules_by_provider[provider][SC_AVG_BATCH]     = avg_batch
        log.info("  %s — schedules built (mean days/week: worst=%.2f, avg=%.2f, sa_ml=%.2f)",
                  provider,
                  np.mean([len(v) for v in worst_batch.values()]),
                  np.mean([len(v) for v in avg_batch.values()]),
                  np.mean([len(v) for v in scenario_schedules_by_provider[provider][SC_SA_ML_BATCH].values()]))

    # ── stage 6 · VROOM solve for all stock scenarios (4 non-baseline) ─
    state = run_stage(step_solve_scenarios, state, "Stage 6 · VROOM solve stock scenarios")

    # ── add average + worst-case scenarios via direct VROOM calls ─────
    # User decision: skip VROOM solve for Avg+Worst — these are diagnostic
    # only (we already know Worst ≈ Baseline mathematically, and Avg is a
    # median pick for threshold analysis). Schedules remain built so the
    # ground-truth table can include their ML predictions.
    log.info("► Skipping VROOM solve for AVG + WORST (kept as predicted schedules only)")
    provider_data = state.artefacts["provider_data"]
    baseline_job_caps = state.artefacts["baseline_job_caps"]
    wsf = state.artefacts["wsf"]
    all_routes = state.artefacts["all_routes"]

    for sc_name in []:    # was [SC_AVG_BATCH, SC_WORST_BATCH] — skipped by user request
        parts = []
        for i, (provider, pdata) in enumerate(provider_data.items(), 1):
            sc_schedules = scenario_schedules_by_provider[provider][sc_name]
            sc_fast_share = pdata["fast_share"] if sc_name == SC_WORST_EXPRESS else 0.0
            cache_tag = f"{provider.lower()}_{sc_name.lower().replace(' ','_').replace('+','plus')}"
            result_dir = (OUT_DIR / sc_name.lower().replace(' ','_').replace('+','plus')
                            / provider.lower())
            sc_requests, _ = build_scenario_requests(
                sc_schedules, pdata["df_assignments"], pdata["daily_gdfs_wgs"],
                fast_share=sc_fast_share,
                scenario_name=f"{sc_name} {provider}",
                speed_factor=wsf,
                max_jobs_per_plz=baseline_job_caps.get(provider, {}),
            )
            sc_solutions, df_sc_solve = solve_scenario(
                sc_requests, f"{sc_name} {provider}",
                cache_tag=cache_tag, save_intermediate=result_dir,
            )
            df_sc_routes = parse_routes(sc_solutions, sc_name, pdata["df_assignments"],
                                          provider_name=provider)
            parts.append(df_sc_routes)
            log.info("    %s %s — %d routes", sc_name, provider, len(df_sc_routes))
        all_routes[sc_name] = pd.concat(parts, ignore_index=True)
    state.artefacts["all_routes"] = all_routes

    # ── stage 7 stock KPIs ─────────────────────────────────────────────
    state = run_stage(step_evaluate, state, "Stage 7 · stock KPI evaluation")

    # ── custom KPI table including worst-case scenarios ────────────────
    log.info("► Custom KPI comparison (baseline vs best ML vs worst case)")
    all_scenarios = list(all_routes.keys())  # includes both worst-case
    scenario_waiting = state.artefacts.get("scenario_waiting", {})

    # waiting times for worst-case scenarios
    optimization_data = state.artefacts["optimization_data"]
    for sc_name in []:    # Avg + Worst skipped — no VROOM, no KPI rows
        total_w, weighted_wait = 0.0, 0.0
        for provider, pdata in provider_data.items():
            sc_schedules = scenario_schedules_by_provider[provider][sc_name]
            weekly_map = pdata["plz_demand"].set_index("plz")["weekly_parcels"].to_dict()
            fs_b2c = 0.0
            fs_b2b = 0.0
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
                total_w += w
                weighted_wait += wait_days * w
        scenario_waiting[sc_name] = weighted_wait / max(1.0, total_w)
    state.artefacts["scenario_waiting"] = scenario_waiting

    # build full comparison
    kpi_list = []
    for sc_name in all_scenarios:
        kpi_list.append(compute_scenario_kpis(
            all_routes[sc_name], sc_name,
            avg_waiting_days=scenario_waiting.get(sc_name, 0.0),
        ))
    df_kpi_all = compare_scenarios(kpi_list, baseline_name=SC_BASELINE)
    df_kpi_all.to_csv(OUT_DIR / "scenario_comparison_kpis_with_worst.csv", index=False)
    log.info("Wrote %s", OUT_DIR / "scenario_comparison_kpis_with_worst.csv")
    print("\n=== Full KPI comparison ===")
    print(df_kpi_all.round(2).to_string(index=False))

    # ── extreme summary: best vs worst gap ─────────────────────────────
    cost_per_sc = {row["scenario"]: row["cost_eur"]
                    for _, row in df_kpi_all.iterrows()
                    if "scenario" in df_kpi_all.columns}
    if not cost_per_sc:
        df_reset = df_kpi_all.reset_index()
        cost_per_sc = {row[df_reset.columns[0]]: row["cost_eur"]
                        for _, row in df_reset.iterrows()}
    extremes = {
        "baseline_cost_eur":    cost_per_sc.get(SC_BASELINE, np.nan),
        "best_ml_express_cost": cost_per_sc.get(SC_SA_ML_EXPRESS, np.nan),
        "best_ml_batch_cost":   cost_per_sc.get(SC_SA_ML_BATCH, np.nan),
        "worst_express_cost":   cost_per_sc.get(SC_WORST_EXPRESS, np.nan),
        "worst_batch_cost":     cost_per_sc.get(SC_WORST_BATCH, np.nan),
    }
    if not np.isnan(extremes["best_ml_express_cost"]) and not np.isnan(extremes["worst_express_cost"]):
        extremes["gap_best_vs_worst_express_pct"] = round(
            100 * (extremes["worst_express_cost"] - extremes["best_ml_express_cost"])
                  / extremes["best_ml_express_cost"], 2)
    if not np.isnan(extremes["best_ml_batch_cost"]) and not np.isnan(extremes["worst_batch_cost"]):
        extremes["gap_best_vs_worst_batch_pct"] = round(
            100 * (extremes["worst_batch_cost"] - extremes["best_ml_batch_cost"])
                  / extremes["best_ml_batch_cost"], 2)
    if not np.isnan(extremes["best_ml_express_cost"]) and not np.isnan(extremes["baseline_cost_eur"]):
        extremes["savings_best_vs_baseline_pct"] = round(
            100 * (extremes["baseline_cost_eur"] - extremes["best_ml_express_cost"])
                  / extremes["baseline_cost_eur"], 2)

    (OUT_DIR / "extremes_summary.json").write_text(json.dumps(extremes, indent=2, default=str))
    log.info("Wrote %s", OUT_DIR / "extremes_summary.json")
    print("\n=== Extremes summary ===")
    print(json.dumps(extremes, indent=2, default=str))

    # ── BUILD PER-(SCENARIO, PROVIDER, PLZ, DAY) ML-vs-VROOM GROUND TRUTH ──
    log.info("► Building per-(scenario, provider, PLZ, day) ML-vs-VROOM comparison")
    gt = build_groundtruth_table(
        scenario_schedules_by_provider=scenario_schedules_by_provider,
        ml_optimization_data=ml_opt,
        all_routes=all_routes,
        optimization_data=optimization_data,
        scenarios=[SC_FIXED_BATCH, SC_SA_ML_BATCH, SC_AVG_BATCH, SC_WORST_BATCH],
    )
    gt.to_csv(OUT_DIR / "ml_vs_vroom_per_day.csv", index=False)
    gt.to_parquet(OUT_DIR / "ml_vs_vroom_per_day.parquet")
    log.info("Wrote %s  (%d rows)", OUT_DIR / "ml_vs_vroom_per_day.csv", len(gt))

    # ── Save state pickle for downstream re-analysis ──────────────────
    import pickle as _pkl
    state_pkl = {
        "scenario_schedules_by_provider": scenario_schedules_by_provider,
        "scenario_waiting": state.artefacts.get("scenario_waiting"),
        "df_kpi_all": df_kpi_all,
        "extremes": extremes,
        # we save the big artefacts separately so this pkl stays manageable:
        "providers": list(cfg.providers),
        "run_dir": str(OUT_DIR),
    }
    (OUT_DIR / "final_state.pkl").write_bytes(_pkl.dumps(state_pkl))
    log.info("Wrote %s", OUT_DIR / "final_state.pkl")

    # ── Auto-trigger ml_vs_vroom_plots ────────────────────────────────
    try:
        log.info("► Generating ML-vs-VROOM plots (auto)")
        import subprocess as _sp
        _sp.run([sys.executable, str(ROOT / "scripts" / "ml_vs_vroom_plots.py")],
                  check=False)
    except Exception as exc:
        log.warning("ml_vs_vroom_plots failed: %s — run it manually later.", exc)

    log.info("=" * 70)
    log.info("PIPELINE DONE")


def build_groundtruth_table(
    scenario_schedules_by_provider: dict,
    ml_optimization_data: dict,
    all_routes: dict,
    optimization_data: dict,
    scenarios: list[str],
) -> pd.DataFrame:
    """Per-(scenario, provider, PLZ, day) ML-pred vs VROOM-actual table.

    Hub-bundled express routes (`is_express=True` in parse_routes output) are
    excluded from the per-day aggregation — they confound the per-PLZ
    attribution. Their contribution shows up in the scenario-total KPIs.
    """
    rows = []
    for sc_name in scenarios:
        is_express = sc_name in EXPRESS_SCENARIOS or sc_name == SC_WORST_EXPRESS
        matrices_key = "matrices_ml_expr" if is_express else "matrices_ml_batch"
        df_routes = all_routes.get(sc_name, pd.DataFrame())

        # build per-(plz, day) cost aggregation from VROOM routes (non-express only)
        if len(df_routes):
            routes_non_express = df_routes[df_routes.get("is_express", False) == False]
            agg = (routes_non_express
                    .groupby(["provider", "plz", "day_idx"])
                    .agg(vroom_cost_cents=("cost", "sum"),
                          vroom_n_routes=("vehicle_id", "count"),
                          vroom_n_parcels=("parcels", "sum"),
                          vroom_distance_km=("distance_km", "sum"),
                          vroom_duration_h=("total_h", "sum"))
                    .reset_index())
            agg["vroom_actual_cost_eur"] = agg["vroom_cost_cents"] / 100.0
            actual_lookup = {(r.provider, r.plz, r.day_idx): r
                              for r in agg.itertuples(index=False)}
        else:
            actual_lookup = {}

        for provider, sched_map in scenario_schedules_by_provider.items():
            if sc_name not in sched_map:
                continue
            sched_by_plz = sched_map[sc_name]
            ml_opt = ml_optimization_data[provider]
            schedules = optimization_data[provider]["schedules"]
            plz_keys = optimization_data[provider]["plz_keys"]
            mat = ml_opt[matrices_key]
            cost_3d = mat["cost_3d"]  # CORRECT shape: (n_plz, n_schedules, n_days)
            sched_to_idx = {s: i for i, s in enumerate(schedules)}

            for pi, plz in enumerate(plz_keys):
                sched = sched_by_plz.get(plz)
                if sched is None:
                    continue
                sidx = sched_to_idx.get(frozenset(sched))
                sched_days = sorted(int(d) for d in sched)
                for day in range(N_DAYS):
                    delivers = (day in sched)
                    ml_pred = float(cost_3d[pi, sidx, day]) if sidx is not None else np.nan
                    a = actual_lookup.get((provider, plz, day))
                    rows.append({
                        "scenario":           sc_name,
                        "provider":           provider,
                        "plz":                plz,
                        "day_idx":            day,
                        "weekday":            ["Mo", "Tu", "We", "Th", "Fr", "Sa"][day],
                        "delivers_on_day":    delivers,
                        "schedule_size":      len(sched),
                        "schedule_days":      ",".join(str(d) for d in sched_days),
                        "ml_pred_cost_eur":   ml_pred,
                        "vroom_actual_cost_eur": a.vroom_actual_cost_eur if a else (0.0 if delivers else np.nan),
                        "vroom_n_routes":     int(a.vroom_n_routes) if a else 0,
                        "vroom_n_parcels":    int(a.vroom_n_parcels) if a else 0,
                        "vroom_distance_km":  float(a.vroom_distance_km) if a else 0.0,
                        "vroom_duration_h":   float(a.vroom_duration_h) if a else 0.0,
                    })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
