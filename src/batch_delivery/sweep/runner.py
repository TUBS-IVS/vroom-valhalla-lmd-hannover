"""Mass-routing sweep driver.

Generates a training matrix for the surrogate model by routing many
perturbed-demand instances through VROOM and recording features +
ground-truth cost per (provider, day-window, PLZ, perturbation seed)
combination.
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from batch_delivery.config import constants as C
from batch_delivery.features.core import (
    ALL_COLS,
    compute_tier1_features,
    compute_tier2_features,
    compute_tier3_features,
)
from batch_delivery.io import demand as demand_io
from batch_delivery.io import hubs as hubs_io
from batch_delivery.routing.core import (
    build_vroom_jobs,
    build_vroom_vehicles,
    solve_single_plz,
)
from batch_delivery.runtime import RunContext
from batch_delivery.sweep.config import SweepConfig
from batch_delivery.sweep.perturb import (
    SweepCombination,
    aggregate_days,
    enumerate_combinations,
    perturb_demand,
)
from batch_delivery.utils import log


# A combo is the *unperturbed* HAGRID baseline iff all three perturbation
# knobs are at their identity values. Such combos produce identical routing
# input regardless of the seed, so we keep only one of them per
# (provider, base_day, agg_k, plz).
def _is_identity(combo: SweepCombination) -> bool:
    return (
        combo.scale == 1.0
        and combo.p_keep == 1.0
        and combo.noise_sigma == 0.0
        and combo.b2c_scale == 1.0
        and combo.b2b_scale == 1.0
    )


def _dedupe_identity_combos(
    combos: list[SweepCombination],
) -> tuple[list[SweepCombination], int]:
    """Drop redundant identity-combos with non-canonical seeds.

    Returns (deduped_list, n_dropped). The kept identity combo is the one
    with the smallest seed for each (provider, base_day, agg_k, plz) group.
    Baselines are placed at the *front* of the returned list so that small
    ``max_combinations`` caps always include them first.
    """
    seen_identity: dict[tuple, int] = {}
    dropped = 0
    for c in combos:
        if not _is_identity(c):
            continue
        key = (c.provider, c.base_day, c.agg_k, c.plz)
        prev = seen_identity.get(key)
        if prev is None or c.seed < prev:
            seen_identity[key] = c.seed

    baselines: list[SweepCombination] = []
    perturbed: list[SweepCombination] = []
    emitted: set[tuple] = set()
    for c in combos:
        if _is_identity(c):
            key = (c.provider, c.base_day, c.agg_k, c.plz)
            if c.seed == seen_identity[key] and key not in emitted:
                baselines.append(c)
                emitted.add(key)
            else:
                dropped += 1
        else:
            perturbed.append(c)
    return baselines + perturbed, dropped


# ---------------------------------------------------------------------------
# Per-provider base data
# ---------------------------------------------------------------------------

def _prepare_provider_data(provider: str, gdf_plz: gpd.GeoDataFrame) -> dict[str, Any]:
    """Load + project HAGRID demand and hub assignments for a single provider.

    CRITICAL FIX 2026-05-25: previously the sweep operated on UN-merged daily
    data while production inference uses merged-cluster data — causing every
    training row for merged-cluster representatives (30159, 30163, etc.) to
    have wrong topology (n_parcels, n_stops, area, hub_dist, etc.). The fix:
    (a) call merge_small_plz so daily_gdfs / plz_day_b2c / plz_day_b2b reflect
    the cluster IDs, (b) pass merge_map to assign_plz_to_hubs, (c) build
    plz_area_km2 from cluster-summed polygons.

    Returns a dict with:
      ``daily_pts``  : dict[day_idx, DataFrame(str_idx, lon, lat, plz,
                       dhl_total, dhl_b2c, dhl_b2b)]  (WGS84) — plz codes are
                       CLUSTER IDs
      ``df_assign``  : DataFrame(plz, hub_lon, hub_lat, hub_typ, dist_km) — plz
                       is cluster_id, dist_km uses cluster centroid
      ``plz_area_km2``: dict[cluster_id, float] — summed polygon area of all
                       cluster members
      ``merge_map``  : dict[small_plz, cluster_id]
    """
    daily_gdfs, plz_day_b2c, plz_day_b2b = demand_io.load_daily_demand(lsp_prefix=provider)
    gdf_hubs = hubs_io.load_hubs(provider=provider)

    # Apply merge_small_plz so daily_gdfs, demand dicts, and unified gdf all
    # reflect cluster IDs — mirrors what step_load_demand_and_hubs does in the
    # production pipeline.
    gdf_unified = demand_io.build_unified_gdf(daily_gdfs)
    gdf_unified, plz_day_b2c, plz_day_b2b, merge_map = demand_io.merge_small_plz(
        gdf_unified, gdf_plz, plz_day_b2c, plz_day_b2b, daily_gdfs=daily_gdfs,
    )

    # Project once per provider (after the merge has rewritten plz codes)
    daily_pts: dict[int, pd.DataFrame] = {}
    plz_with_demand: set[str] = set()
    for day_idx, gdf in daily_gdfs.items():
        gdf_wgs = gdf.to_crs(epsg=4326)
        df = pd.DataFrame({
            "str_idx": gdf_wgs["str_idx"].astype(str).values,
            "plz": gdf_wgs["plz"].astype(str).values,
            "lon": gdf_wgs.geometry.x.values,
            "lat": gdf_wgs.geometry.y.values,
            "dhl_total": gdf_wgs["dhl_total"].astype(int).values,
            "dhl_b2c": gdf_wgs["dhl_b2c"].astype(int).values,
            "dhl_b2b": gdf_wgs["dhl_b2b"].astype(int).values,
        })
        daily_pts[day_idx] = df
        plz_with_demand.update(df["plz"].unique().tolist())

    df_assign = hubs_io.assign_plz_to_hubs(
        gdf_plz, gdf_hubs, plz_with_demand, merge_map=merge_map,
    )
    df_assign = df_assign.set_index("plz")

    # PLZ area in km^2 — cluster-summed (members + representative)
    cluster_members: dict[str, list[str]] = {}
    for s_plz, t_plz in merge_map.items():
        cluster_members.setdefault(t_plz, []).append(s_plz)
    plz_area_km2: dict[str, float] = {}
    if "plz" in gdf_plz.columns:
        plz_to_area = {}
        for plz_code, sub in gdf_plz.dissolve(by="plz").iterrows():
            try:
                plz_to_area[str(plz_code)] = float(sub.geometry.area) / 1e6
            except Exception:
                plz_to_area[str(plz_code)] = float("nan")
        for cluster_id in plz_with_demand:
            members = [cluster_id] + cluster_members.get(cluster_id, [])
            total = sum(plz_to_area.get(m, 0.0) for m in members if m in plz_to_area)
            plz_area_km2[cluster_id] = total if total > 0 else float("nan")

    return {
        "daily_pts": daily_pts,
        "df_assign": df_assign,
        "plz_area_km2": plz_area_km2,
        "merge_map": merge_map,
        "plz_day_b2c": plz_day_b2c,
        "plz_day_b2b": plz_day_b2b,
    }


# ---------------------------------------------------------------------------
# Single combination → one training row
# ---------------------------------------------------------------------------

def _route_combination(
    combo: SweepCombination,
    provider_data: dict[str, Any],
    cfg: SweepConfig,
) -> dict[str, Any] | None:
    """Run a single sweep combination through VROOM and return one row.

    Returns ``None`` if the combination does not yield a valid sample
    (e.g. PLZ not in demand, too few stops, VROOM failure).
    """
    daily_pts = provider_data["daily_pts"]
    df_assign = provider_data["df_assign"]
    plz_area_km2 = provider_data["plz_area_km2"]

    if combo.plz not in df_assign.index:
        return None

    # 1. aggregate days
    pts_all = aggregate_days(daily_pts, combo.base_day, combo.agg_k)
    pts = pts_all[pts_all["plz"] == combo.plz].reset_index(drop=True)
    if len(pts) < cfg.min_stops:
        return None

    # 2. perturb
    rng = np.random.default_rng(combo.seed)
    pts = perturb_demand(
        pts,
        scale=combo.scale,
        p_keep=combo.p_keep,
        noise_sigma=combo.noise_sigma,
        b2c_scale=combo.b2c_scale,
        b2b_scale=combo.b2b_scale,
        rng=rng,
    )
    if len(pts) < cfg.min_stops:
        return None
    if int(pts["dhl_total"].sum()) < cfg.min_parcels:
        return None

    # 3. build VROOM request
    hub_row = df_assign.loc[combo.plz]
    jobs, total_demand = build_vroom_jobs(pts)
    if not jobs:
        return None

    seed_key = combo.cache_tag()
    vehicles, n_veh = build_vroom_vehicles(
        hub=hub_row,
        total_demand=total_demand,
        day_idx=combo.base_day,
        seed_key=seed_key,
        n_jobs=len(jobs),
    )
    request_body = {"jobs": jobs, "vehicles": vehicles}

    # 4. solve
    cache_tag = f"sweep_{combo.provider}" if cfg.use_cache else None
    result = solve_single_plz(
        plz_code=combo.plz,
        request_body=request_body,
        idx=0,
        total=1,
        day_name=C.WEEKDAYS[combo.base_day] if combo.base_day < len(C.WEEKDAYS) else "",
        cache_tag=cache_tag,
    )
    if result["status"] not in ("OK", "CACHED") or result["cost"] is None:
        return None
    if result["n_unassigned"] and result["n_unassigned"] > 0:
        return None

    # 5. compute features
    n_parcels = int(pts["dhl_total"].sum())
    n_stops = len(pts)
    area_km2 = float(plz_area_km2.get(combo.plz, 1.0)) or 1.0
    hub_dist_km = float(hub_row["dist_km"])
    hub_lon = float(hub_row["hub_lon"])
    hub_lat = float(hub_row["hub_lat"])

    tier1 = compute_tier1_features(n_parcels, n_stops, area_km2, hub_dist_km)
    tier2 = compute_tier2_features(
        pts["lon"].to_numpy(),
        pts["lat"].to_numpy(),
        hub_lon,
        hub_lat,
        per_stop_demand=pts["dhl_total"].to_numpy(),
    )
    tier3 = compute_tier3_features(
        per_stop_demand=pts["dhl_total"].to_numpy(),
        b2c_parcels=int(pts["dhl_b2c"].sum()),
        total_parcels=n_parcels,
        provider=combo.provider,
        day_idx=combo.base_day,
        delivery_frequency=combo.agg_k,
    )

    row: dict[str, Any] = {
        "provider": combo.provider,
        "plz": combo.plz,
        "base_day": combo.base_day,
        "agg_k": combo.agg_k,
        "scale": combo.scale,
        "p_keep": combo.p_keep,
        "noise_sigma": combo.noise_sigma,
        "b2c_scale": combo.b2c_scale,
        "b2b_scale": combo.b2b_scale,
        "seed": combo.seed,
        "is_baseline": bool(_is_identity(combo)),
        # targets
        "actual_cost_eur": float(result["cost"]) / C.COST_SCALE,
        "actual_distance_km": float(result["distance_m"]) / 1000.0,
        "actual_duration_h": float(result["duration_s"]) / 3600.0,
        "actual_n_routes": int(result["n_routes"]),
        "n_vehicles_planned": int(n_veh),
        "solve_time_s": float(result["solve_time_s"]),
        "vroom_status": str(result["status"]),
    }
    row.update(tier1)
    row.update(tier2)
    row.update(tier3)
    return row


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_sweep(
    cfg: SweepConfig,
    *,
    ctx: RunContext | None = None,
) -> pd.DataFrame:
    """Execute a full sweep and return the resulting training matrix.

    Side effects: writes ``cfg.out_csv`` (and optionally ``cfg.out_parquet``)
    into ``cfg.out_dir``.  When ``ctx`` is provided, structured events are
    appended to its JSONL log.
    """
    t_start = time.perf_counter()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Always need the PLZ polygons once for area + hub assignment
    gdf_plz = demand_io.load_plz_areas()

    # Pre-load per provider (expensive — do it once)
    provider_cache: dict[str, dict[str, Any]] = {}
    for provider in cfg.providers:
        log.info(f"sweep: preparing provider data for {provider}")
        if ctx is not None:
            ctx.log_event("sweep_provider_load", provider=provider)
        provider_cache[provider] = _prepare_provider_data(provider, gdf_plz)

    # Resolve PLZ list
    if cfg.plzs is None:
        plzs: list[str] = sorted(
            {
                p
                for pdata in provider_cache.values()
                for p in pdata["df_assign"].index.tolist()
            }
        )
    else:
        plzs = list(cfg.plzs)

    combos: Iterable[SweepCombination] = enumerate_combinations(
        providers=cfg.providers,
        base_days=cfg.base_days,
        agg_ks=cfg.agg_ks,
        plzs=plzs,
        scales=cfg.scales,
        p_keeps=cfg.p_keeps,
        noise_sigmas=cfg.noise_sigmas,
        seeds=cfg.seeds,
        b2c_scales=cfg.b2c_scales,
        b2b_scales=cfg.b2b_scales,
    )

    # Materialise so we can show a progress bar with a known total
    combo_list: list[SweepCombination] = list(combos)
    # Drop redundant identity (scale=1, p_keep=1, noise=0) combos that only
    # differ by seed — they all produce the same routing input. We keep the
    # smallest-seed one per (provider, base_day, agg_k, plz).
    combo_list, n_identity_dropped = _dedupe_identity_combos(combo_list)
    if n_identity_dropped:
        log.info(
            f"sweep: dropped {n_identity_dropped} redundant identity-seed "
            f"combos (kept one baseline per provider/day/K/plz)"
        )
    if cfg.shuffle_seed is not None:
        import random as _rnd
        from collections import defaultdict
        # Three-tier ordering:
        #   1) priority combos (active-sampling hot regions)
        #   2) STRATIFIED baselines — one per (provider, agg_k, base_day)
        #      cell where possible, capped at ~25 % of the per-iter budget,
        #      so every provider × agg_k slice has at least one identity
        #      tour to anchor the perturbation deltas.
        #   3) STRATIFIED perturbed — shuffled within each (provider, agg_k)
        #      bucket and filled round-robin across buckets so no provider
        #      or aggregation-K is starved by ``max_combinations``.
        priority_set: set[tuple[str, int, str, int]] = (
            {tuple(k) for k in cfg.priority_keys}  # type: ignore[misc]
            if cfg.priority_keys else set()
        )
        priority: list[SweepCombination] = []
        baseline: list[SweepCombination] = []
        perturbed: list[SweepCombination] = []
        for c in combo_list:
            key = (c.provider, c.base_day, c.plz, c.agg_k)
            if key in priority_set:
                priority.append(c)
            elif _is_identity(c):
                baseline.append(c)
            else:
                perturbed.append(c)
        rng = _rnd.Random(cfg.shuffle_seed)

        # ── Stratified baseline cap (~25 % of budget) ─────────────────────
        if cfg.max_combinations is not None and cfg.max_combinations > 0:
            base_quota = max(1, cfg.max_combinations // 4)
            base_quota = min(base_quota, len(baseline))
        else:
            base_quota = len(baseline)
        # Group baselines by (provider, agg_k, base_day) so we round-robin
        # across cells when picking the kept quota — guarantees coverage of
        # every provider × agg_k × day slice present in the enumeration.
        base_buckets: dict[tuple[str, int, int], list[SweepCombination]] = (
            defaultdict(list)
        )
        for c in baseline:
            base_buckets[(c.provider, c.agg_k, c.base_day)].append(c)
        for v in base_buckets.values():
            rng.shuffle(v)
        kept_baselines: list[SweepCombination] = []
        bucket_keys = list(base_buckets.keys())
        rng.shuffle(bucket_keys)
        idx = 0
        while len(kept_baselines) < base_quota and bucket_keys:
            bk = bucket_keys[idx % len(bucket_keys)]
            if base_buckets[bk]:
                kept_baselines.append(base_buckets[bk].pop())
                idx += 1
            else:
                bucket_keys.remove(bk)

        # ── Stratified perturbed shuffle (round-robin over provider×agg_k) ─
        pert_buckets: dict[tuple[str, int], list[SweepCombination]] = (
            defaultdict(list)
        )
        for c in perturbed:
            pert_buckets[(c.provider, c.agg_k)].append(c)
        for v in pert_buckets.values():
            rng.shuffle(v)
        pert_keys = list(pert_buckets.keys())
        rng.shuffle(pert_keys)
        kept_perturbed: list[SweepCombination] = []
        # Stop only when ALL buckets are empty (truncation by
        # max_combinations happens later).
        while any(pert_buckets[k] for k in pert_keys):
            for k in pert_keys:
                if pert_buckets[k]:
                    kept_perturbed.append(pert_buckets[k].pop())

        combo_list = priority + kept_baselines + kept_perturbed

        # ── Coverage diagnostics for the iter log ─────────────────────────
        if cfg.max_combinations is not None and cfg.max_combinations > 0:
            preview = combo_list[: cfg.max_combinations]
        else:
            preview = combo_list
        cell_counts: dict[tuple[str, int], int] = defaultdict(int)
        scale_counts: dict[float, int] = defaultdict(int)
        pkeep_counts: dict[float, int] = defaultdict(int)
        for c in preview:
            cell_counts[(c.provider, c.agg_k)] += 1
            scale_counts[c.scale] += 1
            pkeep_counts[c.p_keep] += 1
        log.info(
            f"sweep: stratified shuffle (seed={cfg.shuffle_seed}): "
            f"{len(priority):,} priority + {len(kept_baselines):,}/"
            f"{len(baseline):,} baselines + {len(kept_perturbed):,} "
            f"perturbed -> budget cap {cfg.max_combinations}"
        )
        log.info(
            "sweep: per (provider, agg_k) coverage in next "
            f"{len(preview):,} combos:"
        )
        for (prov, ak), n in sorted(cell_counts.items()):
            log.info(f"    {prov:>7s} k={ak}: {n:4d}")
        log.info(
            "sweep: scale distribution: "
            + ", ".join(f"{s:g}={n}" for s, n in sorted(scale_counts.items()))
        )
        log.info(
            "sweep: p_keep distribution: "
            + ", ".join(f"{p:g}={n}" for p, n in sorted(pkeep_counts.items()))
        )
    if cfg.max_combinations is not None:
        combo_list = combo_list[: cfg.max_combinations]
    total = len(combo_list)
    n_baseline_combos = sum(1 for c in combo_list if _is_identity(c))
    log.info(
        f"sweep: enumerating {total:,} combinations "
        f"({n_baseline_combos:,} baseline, "
        f"{total - n_baseline_combos:,} perturbed)"
    )
    if ctx is not None:
        ctx.log_event(
            "sweep_start",
            total_combinations=total,
            n_baseline=n_baseline_combos,
            n_perturbed=total - n_baseline_combos,
            providers=cfg.providers,
            base_days=cfg.base_days,
            agg_ks=cfg.agg_ks,
            n_plzs=len(plzs),
        )

    rows: list[dict[str, Any]] = []
    n_skipped = 0

    def _process_one(combo: SweepCombination) -> dict[str, Any] | None:
        try:
            return _route_combination(combo, provider_cache[combo.provider], cfg)
        except Exception as exc:  # pragma: no cover — diagnostic only
            log.warning(f"sweep: combination {combo} failed: {exc}")
            return None

    n_jobs = int(cfg.parallel_jobs)
    if n_jobs == 1 or total <= 1:
        iterator = combo_list
        if cfg.progress:
            iterator = tqdm(combo_list, desc="sweep", unit="combo")
        results_iter = (_process_one(c) for c in iterator)
    else:
        try:
            from joblib import Parallel, delayed
        except ImportError:  # pragma: no cover — joblib is a hard dep here
            log.warning(
                "sweep: joblib unavailable, falling back to serial execution"
            )
            iterator = combo_list
            if cfg.progress:
                iterator = tqdm(combo_list, desc="sweep", unit="combo")
            results_iter = (_process_one(c) for c in iterator)
        else:
            log.info(
                f"sweep: parallel n_jobs={n_jobs} backend={cfg.parallel_backend}"
            )
            parallel = Parallel(
                n_jobs=n_jobs,
                backend=cfg.parallel_backend,
                return_as="generator_unordered",
            )
            gen = parallel(delayed(_process_one)(c) for c in combo_list)
            if cfg.progress:
                gen = tqdm(gen, total=total, desc="sweep", unit="combo")
            results_iter = gen

    for row in results_iter:
        if row is None:
            n_skipped += 1
            continue
        rows.append(row)

    df = pd.DataFrame(rows)

    # Per-provider summary (rows + skipped + mean cost)
    if len(df) and "provider" in df.columns:
        per_prov = (
            df.groupby("provider")
            .agg(
                n_rows=("provider", "size"),
                n_baseline=("is_baseline", "sum"),
                mean_cost_eur=("actual_cost_eur", "mean"),
                mean_n_routes=("actual_n_routes", "mean"),
            )
            .round(2)
        )
        log.info(f"sweep: per-provider summary\n{per_prov.to_string()}")
        if ctx is not None:
            ctx.log_event(
                "sweep_per_provider",
                **{
                    str(p): {
                        "n_rows": int(row["n_rows"]),
                        "n_baseline": int(row["n_baseline"]),
                        "mean_cost_eur": float(row["mean_cost_eur"]),
                        "mean_n_routes": float(row["mean_n_routes"]),
                    }
                    for p, row in per_prov.iterrows()
                },
            )

    # Always include the standard feature columns even if a tier is empty
    expected = ALL_COLS + [
        "provider", "plz", "base_day", "agg_k", "scale", "p_keep",
        "noise_sigma", "seed", "is_baseline",
        "actual_cost_eur", "actual_distance_km", "actual_duration_h",
        "actual_n_routes", "n_vehicles_planned", "solve_time_s", "vroom_status",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = pd.Series(dtype="float64")

    csv_path = cfg.out_dir / cfg.out_csv
    df.to_csv(csv_path, index=False)
    log.info(f"sweep: wrote {len(df):,} rows -> {csv_path}")

    if cfg.out_parquet:
        try:
            parquet_path = cfg.out_dir / cfg.out_parquet
            df.to_parquet(parquet_path, index=False)
            log.info(f"sweep: wrote parquet -> {parquet_path}")
        except Exception as exc:  # pragma: no cover — pyarrow optional
            log.warning(f"sweep: parquet write skipped ({exc})")

    elapsed = time.perf_counter() - t_start
    if ctx is not None:
        ctx.log_event(
            "sweep_done",
            n_rows=len(df),
            n_skipped=n_skipped,
            duration_s=round(elapsed, 2),
            csv_path=str(csv_path),
        )
    log.info(
        f"sweep: done in {elapsed:.1f}s — "
        f"{len(df):,} rows, {n_skipped:,} skipped"
    )
    return df
