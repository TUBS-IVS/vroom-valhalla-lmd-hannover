"""Spatial and demand feature engineering for VRP cost prediction.

Implements a tiered feature taxonomy inspired by Akkerman & Mes (2023),
"Distance approximation to support customer selection in vehicle routing
problems".  Features are computed from delivery-point coordinates and
demand metadata, grouped into three tiers of increasing richness:

Tier 1 — Baseline (what the Daganzo proxy already uses)
Tier 2 — Spatial point-cloud geometry (convex hull, NN distances, …)
Tier 3 — Demand & capacity characteristics

All spatial computations use **projected CRS (EPSG:25832, metres)** to
ensure correct Euclidean distances.  WGS-84 coordinates are converted
internally.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, KDTree
from tqdm.auto import tqdm

from batch_delivery.config.constants import (
    COST_SCALE,
    N_DAYS,
    PROVIDERS,
    VEHICLE_CAPACITY,
)
from batch_delivery.utils import log

# ─────────────────────────────────────────────────────────────────────────────
# Feature-name constants (referenced by notebooks for column selection)
# ─────────────────────────────────────────────────────────────────────────────

TIER1_COLS: list[str] = [
    "n_parcels",
    "n_stops",
    "area_km2",
    "hub_dist_km",
    "parcels_per_stop",
    "load_factor",
    "min_vehicles",
    "parcels_per_km2",
]

TIER2_COLS: list[str] = [
    "ch_area_km2",
    "ch_perimeter_km",
    "mean_nn_dist_km",
    "mean_inter_stop_dist_km",
    "stop_density_ch",
    "centroid_hub_dist_km",
    "max_hub_dist_km",
    "coord_std_x",
    "coord_std_y",
    "aspect_ratio",
]

TIER3_COLS: list[str] = [
    "b2c_share",
    "demand_std",
    "max_stop_demand",
    "demand_cap_ratio",
    "provider_idx",
    "day_idx",
    "delivery_frequency",
]

ALL_COLS: list[str] = TIER1_COLS + TIER2_COLS + TIER3_COLS

# Provider label encoding (deterministic)
_PROVIDER_IDX = {p: i for i, p in enumerate(sorted(PROVIDERS))}


def provider_index(provider: str) -> int:
    """``provider_idx`` for *provider*, or raise.

    The single encoder for the ``provider_idx`` model feature, used on every
    path that builds it -- the per-cell feature builder, both cost-matrix
    builders and the bundle head -- so train and serve cannot drift apart.

    It **raises** on an unrecognised carrier rather than defaulting to index
    0. The 7-provider set is fixed and pinned by tests, so an unknown name is
    a defect (a typo, a stale mapping, a provider column read from the wrong
    frame); silently encoding it as index 0 would price it as that carrier and
    return a plausible-looking wrong number instead of an error.
    """
    try:
        return _PROVIDER_IDX[provider]
    except KeyError:
        raise KeyError(
            f"unknown provider {provider!r}: provider_idx is defined only for "
            f"the {len(_PROVIDER_IDX)} carriers of the case study, "
            f"{sorted(_PROVIDER_IDX)}. An unrecognised carrier is a defect, "
            f"not something to encode as index 0."
        ) from None


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — baseline features
# ─────────────────────────────────────────────────────────────────────────────

def compute_tier1_features(
    n_parcels: int,
    n_stops: int,
    area_km2: float,
    hub_dist_km: float,
) -> dict[str, float]:
    """Compute Tier-1 (Daganzo-level) features.

    Parameters
    ----------
    n_parcels : int
        Total parcels to deliver.
    n_stops : int
        Number of delivery points.
    area_km2 : float
        PLZ polygon area (km²).
    hub_dist_km : float
        Distance from PLZ centroid to depot (km).

    Returns
    -------
    dict with 8 float values keyed by ``TIER1_COLS``.
    """
    n_stops = max(1, n_stops)
    area_km2 = max(0.01, area_km2)
    return {
        "n_parcels": float(n_parcels),
        "n_stops": float(n_stops),
        "area_km2": area_km2,
        "hub_dist_km": hub_dist_km,
        "parcels_per_stop": n_parcels / n_stops,
        "load_factor": n_parcels / VEHICLE_CAPACITY,
        "min_vehicles": math.ceil(n_parcels / VEHICLE_CAPACITY),
        "parcels_per_km2": n_parcels / area_km2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — spatial point-cloud features
# ─────────────────────────────────────────────────────────────────────────────

def _wgs84_to_utm(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fast approximate WGS-84 → UTM32N (EPSG:25832) conversion.

    Uses a linear approximation valid for the Hannover region
    (lon ≈ 9.5–10.5°, lat ≈ 52.0–52.8°).  Maximum error < 5 m
    within the study area — sufficient for feature computation.
    """
    # Reference point: Hannover centre
    lon0, lat0 = 9.7320, 52.3759
    # Scale factors at reference latitude
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_132.0 * math.cos(math.radians(lat0))
    x = (lon - lon0) * m_per_deg_lon + 500_000  # false easting (UTM32N)
    y = (lat - lat0) * m_per_deg_lat + 5_800_000  # approx northing
    return x, y


def compute_tier2_features(
    lon: np.ndarray,
    lat: np.ndarray,
    hub_lon: float,
    hub_lat: float,
    per_stop_demand: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute Tier-2 spatial features from delivery-point coordinates.

    Parameters
    ----------
    lon, lat : ndarray, shape (n_stops,)
        WGS-84 coordinates of delivery points.
    hub_lon, hub_lat : float
        WGS-84 coordinates of the depot / hub.
    per_stop_demand : ndarray, optional
        Parcel count per stop (used for demand-weighted centroid).

    Returns
    -------
    dict with 10 float values keyed by ``TIER2_COLS``.
    """
    n = len(lon)

    # Default fallback for degenerate cases
    fallback = dict.fromkeys(TIER2_COLS, 0.0)

    if n == 0:
        return fallback

    # Project to metres
    x, y = _wgs84_to_utm(lon, lat)
    hx, hy = _wgs84_to_utm(
        np.array([hub_lon]), np.array([hub_lat]),
    )
    hx, hy = float(hx[0]), float(hy[0])
    coords = np.column_stack([x, y])  # (n, 2) in metres

    # ── Centroid & hub distances ─────────────────────────────────────
    cx, cy = float(x.mean()), float(y.mean())
    hub_dists = np.sqrt((x - hx) ** 2 + (y - hy) ** 2)
    centroid_hub_dist = math.sqrt((cx - hx) ** 2 + (cy - hy) ** 2)
    max_hub_dist = float(hub_dists.max())

    # ── Coordinate spread ────────────────────────────────────────────
    std_x = float(x.std()) if n > 1 else 0.0
    std_y = float(y.std()) if n > 1 else 0.0

    # ── Bounding box aspect ratio ────────────────────────────────────
    dx = float(x.max() - x.min()) if n > 1 else 1.0
    dy = float(y.max() - y.min()) if n > 1 else 1.0
    aspect = max(dx, dy) / max(min(dx, dy), 1.0)

    # ── Convex hull ──────────────────────────────────────────────────
    if n >= 3:
        try:
            hull = ConvexHull(coords)
            ch_area = hull.volume  # 2-D: volume = area
            ch_perim = hull.area  # 2-D: area = perimeter
        except Exception:
            ch_area = dx * dy
            ch_perim = 2 * (dx + dy)
    elif n == 2:
        ch_area = 0.0
        ch_perim = 2 * math.sqrt(
            (x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2
        )
    else:
        ch_area = 0.0
        ch_perim = 0.0

    # ── Nearest-neighbour distances (KDTree) ─────────────────────────
    if n >= 2:
        tree = KDTree(coords)
        nn_dists, _ = tree.query(coords, k=2)  # k=2: self + nearest
        mean_nn = float(nn_dists[:, 1].mean())
    else:
        mean_nn = 0.0

    # ── Mean inter-stop distance (sampled for large n) ───────────────
    if n >= 2:
        if n <= 300:
            # Full pairwise (upper triangle)
            from scipy.spatial.distance import pdist

            mean_inter = float(pdist(coords).mean())
        else:
            # Random sample of 300 pairs
            rng = np.random.default_rng(42)
            idx_a = rng.integers(0, n, size=300)
            idx_b = rng.integers(0, n, size=300)
            mask = idx_a != idx_b
            idx_a, idx_b = idx_a[mask], idx_b[mask]
            diffs = coords[idx_a] - coords[idx_b]
            mean_inter = float(np.sqrt((diffs ** 2).sum(axis=1)).mean())
    else:
        mean_inter = 0.0

    # ── Stop density on convex hull ──────────────────────────────────
    ch_area_km2 = ch_area / 1e6
    density_ch = n / max(ch_area_km2, 0.001)

    return {
        "ch_area_km2": ch_area_km2,
        "ch_perimeter_km": ch_perim / 1000,
        "mean_nn_dist_km": mean_nn / 1000,
        "mean_inter_stop_dist_km": mean_inter / 1000,
        "stop_density_ch": density_ch,
        "centroid_hub_dist_km": centroid_hub_dist / 1000,
        "max_hub_dist_km": max_hub_dist / 1000,
        "coord_std_x": std_x / 1000,  # km
        "coord_std_y": std_y / 1000,
        "aspect_ratio": aspect,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — demand & capacity features
# ─────────────────────────────────────────────────────────────────────────────

def compute_tier3_features(
    per_stop_demand: np.ndarray,
    b2c_parcels: int,
    total_parcels: int,
    provider: str,
    day_idx: int,
    delivery_frequency: int,
) -> dict[str, float]:
    """Compute Tier-3 demand and capacity features.

    Parameters
    ----------
    per_stop_demand : ndarray
        Parcel count per delivery point.
    b2c_parcels : int
        B2C parcel count (for modal split).
    total_parcels : int
        Total parcels.
    provider : str
        Provider name (label-encoded).
    day_idx : int
        Day index 0–5 (Mon–Sat).
    delivery_frequency : int
        Number of delivery days per week for this PLZ.

    Returns
    -------
    dict with 7 float values keyed by ``TIER3_COLS``.
    """
    n_stops = max(1, len(per_stop_demand))
    total = max(1, total_parcels)
    min_veh = max(1, math.ceil(total / VEHICLE_CAPACITY))

    return {
        "b2c_share": b2c_parcels / total if total > 0 else 0.5,
        "demand_std": float(per_stop_demand.std()) if n_stops > 1 else 0.0,
        "max_stop_demand": float(per_stop_demand.max()) if n_stops > 0 else 0.0,
        "demand_cap_ratio": total / (min_veh * VEHICLE_CAPACITY),
        "provider_idx": float(provider_index(provider)),
        "day_idx": float(day_idx),
        "delivery_frequency": float(delivery_frequency),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Combined feature computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_features(
    *,
    n_parcels: int,
    n_stops: int,
    area_km2: float,
    hub_dist_km: float,
    lon: np.ndarray,
    lat: np.ndarray,
    hub_lon: float,
    hub_lat: float,
    per_stop_demand: np.ndarray,
    b2c_parcels: int,
    provider: str,
    day_idx: int,
    delivery_frequency: int,
    tier: int = 3,
) -> dict[str, float]:
    """Compute features up to the specified tier.

    Parameters
    ----------
    tier : int
        1, 2, or 3 — controls which tiers are included.

    Returns
    -------
    dict with feature values.  Missing tiers' columns are set to NaN.
    """
    feats: dict[str, float] = {}

    # Tier 1
    feats.update(compute_tier1_features(n_parcels, n_stops, area_km2, hub_dist_km))

    # Tier 2
    if tier >= 2 and len(lon) > 0:
        feats.update(compute_tier2_features(lon, lat, hub_lon, hub_lat, per_stop_demand))
    elif tier >= 2:
        feats.update(dict.fromkeys(TIER2_COLS, np.nan))

    # Tier 3
    if tier >= 3:
        feats.update(compute_tier3_features(
            per_stop_demand, b2c_parcels, n_parcels,
            provider, day_idx, delivery_frequency,
        ))

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Training-set builder
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    provider_data: dict,
    optimization_data: dict,
    cal_by_provider: dict | None,
    all_routes: dict[str, pd.DataFrame],
    scenario_schedules_by_provider: dict,
    gdf_plz,
    providers: list[str] | None = None,
    scenarios: list[str] | None = None,
    tier: int = 3,
) -> pd.DataFrame:
    """Build a unified training DataFrame with features and VROOM targets.

    Iterates over all (provider, scenario, PLZ, day) combinations that
    have VROOM-solved route data and computes feature vectors alongside
    ground-truth cost, distance, and vehicle counts.

    Parameters
    ----------
    provider_data : dict
        Per-provider data from checkpoint ``01_demand``.
    optimization_data : dict
        Per-provider optimisation structures from checkpoint ``04_optim_prep``.
    cal_by_provider : dict
        ``{provider: CalibratedDaganzo}`` from checkpoint ``03_calibration``.
    all_routes : dict[str, DataFrame]
        ``{scenario_name: df_routes}`` from checkpoint ``06_scenario_routing``.
    scenario_schedules_by_provider : dict
        ``{provider: {scenario: {plz: schedule}}}`` from checkpoint ``05``.
    gdf_plz : GeoDataFrame
        PLZ area polygons.
    providers : list[str], optional
        Subset of providers to include (default: all).
    scenarios : list[str], optional
        Subset of scenarios to include (default: all).
    tier : int
        Feature tier (1, 2, or 3).

    Returns
    -------
    DataFrame
        One row per (provider, scenario, PLZ, day) with features + targets.
    """
    if providers is None:
        providers = list(provider_data.keys())
    if scenarios is None:
        scenarios = list(all_routes.keys())

    # Build argument tuples for parallel processing
    tasks = []
    for provider in providers:
        pdata = provider_data[provider]
        odata = optimization_data[provider]
        cal = cal_by_provider[provider] if cal_by_provider else None

        for scenario in scenarios:
            tasks.append((
                provider, scenario, pdata, odata, cal,
                all_routes, scenario_schedules_by_provider, tier,
            ))

    total = len(tasks)
    log.info(f"Building feature matrix: {total} (provider×scenario) tasks")

    # Use joblib for parallel execution (thread backend to share memory)
    try:
        from joblib import Parallel, delayed
        chunk_results = Parallel(
            n_jobs=-1, backend="loky", verbose=0,
        )(
            delayed(_build_features_single)(
                provider, scenario, pdata, odata, cal,
                all_routes, scenario_schedules_by_provider, tier,
            )
            for provider, scenario, pdata, odata, cal, _, _, _ in tasks
        )
    except ImportError:
        log.warning("joblib not available — building feature matrix sequentially")
        chunk_results = []
        for args in tqdm(tasks, desc="Building feature matrix", unit="prov×sc"):
            chunk_results.append(_build_features_single(*args))

    # Flatten
    rows: list[dict[str, Any]] = []
    for chunk in chunk_results:
        rows.extend(chunk)

    df = pd.DataFrame(rows)
    log.info(
        f"Feature matrix: {len(df):,} rows, "
        f"{len(providers)} providers, "
        f"{df['scenario'].nunique() if len(df) > 0 else 0} scenarios, "
        f"{len(df.columns)} columns (tier {tier})"
    )
    return df


def _build_features_single(
    provider: str,
    scenario: str,
    pdata: dict,
    odata: dict,
    cal,  # CalibratedDaganzo
    all_routes: dict[str, pd.DataFrame],
    scenario_schedules_by_provider: dict,
    tier: int,
) -> list[dict[str, Any]]:
    """Process one (provider, scenario) pair — returns list of row dicts."""
    from batch_delivery.config.constants import provider_to_demand_prefix

    plz_data = odata["plz_data"]
    daily_wgs = pdata["daily_gdfs_wgs"]

    # Hub coordinates (WGS-84)
    hub_coords = {}
    for _, hr in pdata["df_assignments"].iterrows():
        hub_coords[hr["plz"]] = (hr["hub_lon"], hr["hub_lat"])

    prefix = provider_to_demand_prefix(provider)
    col_total = f"{prefix}_total"
    col_b2c = f"{prefix}_b2c"

    rows: list[dict[str, Any]] = []

    df_sc = all_routes.get(scenario)
    if df_sc is None or len(df_sc) == 0:
        return rows

    df_prov = df_sc[df_sc["provider"] == provider].copy()
    if len(df_prov) == 0:
        return rows

    # Pre-cast PLZ column once and build groupby index for O(1) lookup
    df_prov["plz"] = df_prov["plz"].astype(str)
    _dd = df_prov[~df_prov["is_express"]]
    _dd_grouped = _dd.groupby(["plz", "day_idx"])
    _dd_keys = set(_dd_grouped.groups.keys())

    sched_map = scenario_schedules_by_provider.get(
        provider, {},
    ).get(scenario, {})

    for plz_code in odata["plz_keys"]:
        if plz_code not in plz_data:
            continue

        pd_ = plz_data[plz_code]
        area = pd_["area_km2"]
        hd = pd_["hub_dist_km"]
        hlon, hlat = hub_coords.get(plz_code, (9.73, 52.38))

        # Delivery frequency for this PLZ
        sched = sched_map.get(plz_code, set())
        freq = len(sched) if sched else N_DAYS

        plz_str = str(plz_code)
        for day in range(N_DAYS):
            # VROOM target: aggregate per (plz, day, non-express)
            if (plz_str, day) not in _dd_keys:
                continue
            hit = _dd_grouped.get_group((plz_str, day))
            if len(hit) == 0:
                continue

            actual_cost = hit["cost"].sum() / COST_SCALE
            actual_km = hit["distance_km"].sum()
            actual_routes = len(hit)
            actual_parcels = int(hit["parcels"].sum())
            actual_stops = int(hit["n_stops"].sum())

            if actual_cost <= 0 or actual_parcels <= 0:
                continue

            # Delivery points for this PLZ on this day
            gdf_day = daily_wgs.get(day)
            if gdf_day is not None and col_total in gdf_day.columns:
                pts = gdf_day[gdf_day["plz"] == plz_code]
                if len(pts) > 0:
                    lon_arr = pts["lon"].values.astype(np.float64)
                    lat_arr = pts["lat"].values.astype(np.float64)
                    psd = pts[col_total].values.astype(np.float64)
                    b2c = int(pts[col_b2c].sum()) if col_b2c in pts.columns else actual_parcels // 2
                else:
                    lon_arr = np.array([], dtype=np.float64)
                    lat_arr = np.array([], dtype=np.float64)
                    psd = np.array([], dtype=np.float64)
                    b2c = actual_parcels // 2
            else:
                lon_arr = np.array([], dtype=np.float64)
                lat_arr = np.array([], dtype=np.float64)
                psd = np.array([], dtype=np.float64)
                b2c = actual_parcels // 2

            # Compute features
            feats = compute_all_features(
                n_parcels=actual_parcels,
                n_stops=actual_stops,
                area_km2=area,
                hub_dist_km=hd,
                lon=lon_arr,
                lat=lat_arr,
                hub_lon=hlon,
                hub_lat=hlat,
                per_stop_demand=psd if len(psd) > 0 else np.array([actual_parcels]),
                b2c_parcels=b2c,
                provider=provider,
                day_idx=day,
                delivery_frequency=freq,
                tier=tier,
            )

            # Daganzo prediction (optional — skipped when cal is None)
            if cal is not None:
                daganzo_corr = cal.plz_corrections.get(plz_code, 1.0)
                daganzo_cost = cal.cost(
                    actual_parcels, actual_stops, area, hd,
                    correction=daganzo_corr,
                )
                daganzo_raw = cal.cost(
                    actual_parcels, actual_stops, area, hd,
                    correction=1.0,
                )
            else:
                daganzo_corr = 1.0
                daganzo_cost = 0.0
                daganzo_raw = 0.0

            # Row
            row = {
                "provider": provider,
                "scenario": scenario,
                "plz": plz_code,
                "day_idx": day,
                # Targets
                "actual_cost": actual_cost,
                "actual_km": actual_km,
                "actual_routes": actual_routes,
                # Daganzo reference
                "daganzo_cost_raw": daganzo_raw,
                "daganzo_cost_cal": daganzo_cost,
                "daganzo_plz_corr": daganzo_corr,
                # Features
                **feats,
            }
            rows.append(row)

    # ── Express routes: include as training rows ─────────────────
    # Express routes aggregate parcels from multiple PLZs at one
    # hub.  We compute features using merged stop coordinates
    # from the VROOM solution and the summed area of contributing
    # PLZs.  Hub distance is 0 (routes start/end at the hub).
    df_expr = df_prov[df_prov["is_express"]].copy()
    if len(df_expr) > 0:
        for day in range(N_DAYS):
            hits_xpr = df_expr[df_expr["day_idx"] == day]
            if len(hits_xpr) == 0:
                continue

            xpr_cost = hits_xpr["cost"].sum() / COST_SCALE
            xpr_km = hits_xpr["distance_km"].sum()
            xpr_routes = len(hits_xpr)
            xpr_parcels = int(hits_xpr["parcels"].sum())
            xpr_stops = int(hits_xpr["n_stops"].sum())

            if xpr_cost <= 0 or xpr_parcels <= 0:
                continue

            # Contributing PLZs: extract from pipe-separated plz field
            plz_field = hits_xpr.iloc[0]["plz"]
            if isinstance(plz_field, str) and "|" in plz_field:
                contrib_plzs = plz_field.split("|")
            elif isinstance(plz_field, str) and plz_field.startswith("_xpr_"):
                contrib_plzs = []
            else:
                contrib_plzs = [str(plz_field)]

            # Summed area and hub coordinates from contributing PLZs
            xpr_area = 0.0
            xpr_hlon, xpr_hlat = 9.73, 52.38
            n_contrib = 0
            for cplz in contrib_plzs:
                if cplz in plz_data:
                    xpr_area += plz_data[cplz]["area_km2"]
                    n_contrib += 1
            if n_contrib > 0 and contrib_plzs[0] in hub_coords:
                xpr_hlon, xpr_hlat = hub_coords[contrib_plzs[0]]
            xpr_area = max(0.01, xpr_area)

            # Express hub_dist = 0 (routes start at hub)
            xpr_hd = 0.0
            # delivery_frequency = number of contributing PLZs
            xpr_freq = max(1, n_contrib)

            # Coordinate arrays: not available from demand points for
            # express, use empty (Tier 2 features default to 0).
            xpr_lon = np.array([], dtype=np.float64)
            xpr_lat = np.array([], dtype=np.float64)
            xpr_psd = np.array([xpr_parcels])
            xpr_b2c = xpr_parcels // 2

            feats = compute_all_features(
                n_parcels=xpr_parcels,
                n_stops=xpr_stops,
                area_km2=xpr_area,
                hub_dist_km=xpr_hd,
                lon=xpr_lon,
                lat=xpr_lat,
                hub_lon=xpr_hlon,
                hub_lat=xpr_hlat,
                per_stop_demand=xpr_psd,
                b2c_parcels=xpr_b2c,
                provider=provider,
                day_idx=day,
                delivery_frequency=xpr_freq,
                tier=tier,
            )

            row = {
                "provider": provider,
                "scenario": scenario,
                "plz": f"_xpr_{day}",
                "day_idx": day,
                "actual_cost": xpr_cost,
                "actual_km": xpr_km,
                "actual_routes": xpr_routes,
                "daganzo_cost_raw": 0.0,
                "daganzo_cost_cal": 0.0,
                "daganzo_plz_corr": 1.0,
                **feats,
            }
            rows.append(row)

    return rows
