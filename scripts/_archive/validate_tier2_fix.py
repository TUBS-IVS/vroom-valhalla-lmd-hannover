"""Validate Tier-2 inference-feature bug fix without re-running final_optimization.

For each (provider, plz, schedule) chosen by SA_ML Batch-Only optimization,
re-compute Tier-2 features using ACCUMULATED source-day stops (the union of
days that flow into each delivery day) instead of single base-day stops.

Then predict cost with production_lgb_logT_v3 and compare:
    - ml_pred_original  : current (buggy) prediction from ml_vs_vroom_per_day
    - ml_pred_corrected : new prediction with fixed Tier-2 features
    - vroom_actual      : ground truth

If corrected predictions close the saving-table gap for FW6.A clusters,
the bug fix is validated and we patch the production code.
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from batch_delivery.features import (  # noqa: E402
    ALL_COLS, TIER2_COLS, _PROVIDER_IDX,
    compute_tier1_features, compute_tier2_features, compute_tier3_features,
)
from batch_delivery.surrogate import build_combo_features  # noqa: E402
from batch_delivery.io.demand import get_source_days  # noqa: E402
from batch_delivery.config.constants import (  # noqa: E402
    N_DAYS, FAST_SHARE_B2C, FAST_SHARE_B2B, VEHICLE_CAPACITY,
)

WEEKDAY_TO_IDX = {"Mo": 0, "Di": 1, "Mi": 2, "Do": 3, "Fr": 4, "Sa": 5,
                  "Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5,
                  "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                  "Friday": 4, "Saturday": 5}


def parse_schedule_days(s: str) -> list[int]:
    """Parse '0,3' or '[0, 3]' → [0, 3]."""
    s = s.strip().strip("[]").strip()
    if not s:
        return []
    return sorted(int(p.strip()) for p in s.replace(" ", "").split(","))


def compute_corrected_features(
    provider: str, plz_code: str, delivery_day: int,
    schedule_days: list[int],
    plz_day_coords: dict, hub_coords_by_plz: dict,
    plz_demand_row: pd.Series,
    area_km2: float, hub_dist_km: float,
    n_stops_per_day: float, total_points: int,
    daily_b2c: dict[int, int], daily_b2b: dict[int, int],
    fast_share_b2c: float = FAST_SHARE_B2C,
    fast_share_b2b: float = FAST_SHARE_B2B,
) -> dict[str, float]:
    """Compute features for a (plz, schedule, delivery_day) using ACCUMULATED stops."""
    src_days = get_source_days(delivery_day, schedule_days)

    # Accumulated demand for delivery day (batch part)
    accum_b2c = sum(daily_b2c.get(d, 0) for d in src_days)
    accum_b2b = sum(daily_b2b.get(d, 0) for d in src_days)
    # Express subtraction from non-delivery source days
    nondel_src = [d for d in src_days if d != delivery_day]
    express_sub = (
        round(sum(daily_b2c.get(d, 0) for d in nondel_src) * fast_share_b2c)
        + round(sum(daily_b2b.get(d, 0) for d in nondel_src) * fast_share_b2b)
    )
    n_parcels = max(0, accum_b2c + accum_b2b - express_sub)

    # Union of source-day stop coords (THE FIX)
    pc_coords = plz_day_coords.get(plz_code, {})
    all_lons, all_lats, all_psd = [], [], []
    for d in src_days:
        c = pc_coords.get(d)
        if c is not None and len(c[0]) > 0:
            all_lons.append(c[0]); all_lats.append(c[1]); all_psd.append(c[2])
    if not all_lons:
        return None
    union_lon = np.concatenate(all_lons)
    union_lat = np.concatenate(all_lats)
    union_psd = np.concatenate(all_psd)
    # Dedupe by (lon, lat) — same customer appearing on multiple days = one stop
    df_pts = pd.DataFrame({"lon": union_lon, "lat": union_lat, "psd": union_psd})
    df_dedup = df_pts.groupby(["lon", "lat"], as_index=False)["psd"].sum()
    lons = df_dedup["lon"].values.astype(np.float64)
    lats = df_dedup["lat"].values.astype(np.float64)
    per_stop_demand = df_dedup["psd"].values.astype(np.float64)
    n_stops = len(lons)
    if n_stops == 0:
        return None

    hlon, hlat = hub_coords_by_plz.get(plz_code, (9.73, 52.38))

    # Tier 1 (use ACCUMULATED n_parcels, n_stops from union)
    feats = compute_tier1_features(n_parcels, n_stops, area_km2, hub_dist_km)
    # Tier 2 — KEY FIX: features from UNION of source-day stops
    t2 = compute_tier2_features(lons, lats, hlon, hlat, per_stop_demand)
    feats.update(t2)
    # Tier 3
    b2c_share = accum_b2c / max(1, accum_b2c + accum_b2b)
    min_veh = max(1, int(np.ceil(n_parcels / VEHICLE_CAPACITY)))
    demand_std = float(per_stop_demand.std()) if len(per_stop_demand) > 1 else 0.0
    max_stop_demand = float(per_stop_demand.max()) if len(per_stop_demand) > 0 else float(n_parcels)
    t3 = compute_tier3_features(
        per_stop_demand=per_stop_demand,
        b2c_parcels=int(round(b2c_share * n_parcels)),
        total_parcels=int(n_parcels),
        provider=provider,
        day_idx=delivery_day,
        delivery_frequency=len(schedule_days),
    )
    feats.update(t3)
    return feats


def main():
    # Load checkpoints
    print("Loading checkpoints...")
    chk1 = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    provider_data = chk1["provider_data"]
    optim_data = chk4["optimization_data"]

    # Build ml_prep equivalent (plz_day_coords + hub_coords)
    from batch_delivery.config.constants import N_DAYS
    ml_prep = {}
    for provider, pdata in provider_data.items():
        prefix = {"DHL": "dhl", "Amazon": "ama", "DPD": "dpd", "FedEx": "fed",
                  "GLS": "gls", "Hermes": "her", "UPS": "ups"}[provider]
        col_total = f"{prefix}_total"
        col_b2c = f"{prefix}_b2c"
        col_b2b = f"{prefix}_b2b"
        daily_wgs = pdata["daily_gdfs_wgs"]
        hub_coords_by_plz = {}
        for _, hr in pdata["df_assignments"].iterrows():
            hub_coords_by_plz[hr["plz"]] = (hr["hub_lon"], hr["hub_lat"])
        plz_day_coords = {}
        for plz_code in optim_data[provider]["plz_keys"]:
            plz_day_coords[plz_code] = {}
            for d in range(N_DAYS):
                gdf_d = daily_wgs.get(d)
                if gdf_d is None: continue
                pts = gdf_d[gdf_d["plz"] == plz_code]
                if len(pts) == 0: continue
                lons = pts["lon"].values.astype(np.float64)
                lats = pts["lat"].values.astype(np.float64)
                psd = pts[col_total].values.astype(np.float64) if col_total in pts.columns else np.ones(len(pts))
                plz_day_coords[plz_code][d] = (lons, lats, psd)
        ml_prep[provider] = {
            "plz_day_coords": plz_day_coords,
            "hub_coords_by_plz": hub_coords_by_plz,
        }

    # Load model
    import sys as _sys
    model_choice = _sys.argv[1] if len(_sys.argv) > 1 else "original"
    if model_choice == "fullpool":
        model_path = ROOT / "results/sweep_v3_mergefix/production_lgb_logT_v3_fullpool.pkl"
        print(f"Loading {model_path.name} ...")
    else:
        model_path = ROOT / "results/sweep_v3_mergefix/production_lgb_logT_v3.pkl"
        print(f"Loading {model_path.name} (original with holdout) ...")
    model_data = pickle.load(open(model_path, "rb"))
    model = model_data["model"]

    # Load SA_ML predictions to know chosen schedules
    print("Loading ml_vs_vroom_per_day...")
    infer = pd.read_csv(ROOT / "results/final_optimization_v3_mergefix/ml_vs_vroom_per_day.csv",
                       dtype={"plz": str})
    infer["plz"] = infer["plz"].astype(str).str.zfill(5)
    saml = infer[(infer["scenario"] == "SA_ML Batch-Only") & infer["delivers_on_day"]].copy()
    print(f"SA_ML batch delivery-days: {len(saml)}")

    # For each row, compute corrected features
    rows_out = []
    n_fail = 0
    for ridx, row in saml.iterrows():
        prov = row["provider"]
        plz = row["plz"]
        dd_str = row["weekday"]
        # day_idx is the inference's delivery day
        dd = int(row["day_idx"])
        sched_str = str(row["schedule_days"])
        sched_days = parse_schedule_days(sched_str)
        if dd not in sched_days:
            n_fail += 1; continue

        pdata = provider_data[prov]
        odata = optim_data[prov]
        plz_demand = pdata["plz_demand"]
        plz_row = plz_demand[plz_demand["plz"] == plz]
        if plz_row.empty:
            n_fail += 1; continue
        prefix = {"DHL": "Monday", "Amazon": "Monday"}.get(prov, "Monday")
        # Build daily b2c/b2b dicts (day_idx -> count)
        weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
        daily_b2c = {i: int(plz_row[f"arrivals_b2c_{weekdays[i]}"].iloc[0]) for i in range(6)}
        daily_b2b = {i: int(plz_row[f"arrivals_b2b_{weekdays[i]}"].iloc[0]) for i in range(6)}

        od = odata["plz_data"][plz]
        area_km2 = float(od["area_km2"])
        hub_dist_km = float(od["hub_dist_km"])
        n_stops_per_day = float(od["n_stops_per_day"])
        total_points = int(od["total_points"])

        feats = compute_corrected_features(
            provider=prov, plz_code=plz, delivery_day=dd,
            schedule_days=sched_days,
            plz_day_coords=ml_prep[prov]["plz_day_coords"],
            hub_coords_by_plz=ml_prep[prov]["hub_coords_by_plz"],
            plz_demand_row=plz_row.iloc[0],
            area_km2=area_km2, hub_dist_km=hub_dist_km,
            n_stops_per_day=n_stops_per_day, total_points=total_points,
            daily_b2c=daily_b2c, daily_b2b=daily_b2b,
        )
        if feats is None:
            n_fail += 1; continue

        X = pd.DataFrame([{c: feats.get(c, 0.0) for c in ALL_COLS}])
        Xc = build_combo_features(X)
        pred_corrected = float(model.predict(Xc.values)[0])

        rows_out.append({
            "provider": prov, "plz": plz, "weekday": dd_str,
            "delivery_day": dd,
            "schedule_size": int(row["schedule_size"]),
            "schedule_days": sched_str,
            "ml_pred_original": float(row["ml_pred_cost_eur"]),
            "ml_pred_corrected": pred_corrected,
            "vroom_actual": float(row["vroom_actual_cost_eur"]),
            "vroom_n_parcels": int(row["vroom_n_parcels"]),
            "n_parcels_features": feats["n_parcels"],
            "n_stops_features": feats["n_stops"],
        })

    print(f"\nSuccess: {len(rows_out)}  Failed: {n_fail}")

    df = pd.DataFrame(rows_out)
    df["err_original_pct"] = 100 * (df["ml_pred_original"] - df["vroom_actual"]) / df["vroom_actual"].clip(lower=1)
    df["err_corrected_pct"] = 100 * (df["ml_pred_corrected"] - df["vroom_actual"]) / df["vroom_actual"].clip(lower=1)

    out = ROOT / "results/final_optimization_v3_mergefix/tier2_fix_validation.csv"
    df.to_csv(out, index=False)

    print(f"\n=== Overall Bias ===")
    print(f"  ORIGINAL  median {df.err_original_pct.median():+.2f}%  abs-median {df.err_original_pct.abs().median():.2f}%  abs-mean {df.err_original_pct.abs().mean():.2f}%")
    print(f"  CORRECTED median {df.err_corrected_pct.median():+.2f}%  abs-median {df.err_corrected_pct.abs().median():.2f}%  abs-mean {df.err_corrected_pct.abs().mean():.2f}%")

    print(f"\n=== FW6.A Clusters (30159/30167/30449) ===")
    fw = df[df.plz.isin(["30159","30167","30449"])]
    print(f"  ORIGINAL  median {fw.err_original_pct.median():+.2f}%  abs-median {fw.err_original_pct.abs().median():.2f}%")
    print(f"  CORRECTED median {fw.err_corrected_pct.median():+.2f}%  abs-median {fw.err_corrected_pct.abs().median():.2f}%")
    print()
    print(fw[["provider","plz","weekday","schedule_size","n_parcels_features","ml_pred_original","ml_pred_corrected","vroom_actual","err_original_pct","err_corrected_pct"]].sort_values(["plz","provider"]).to_string(index=False))


if __name__ == "__main__":
    main()
