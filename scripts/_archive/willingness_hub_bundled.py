"""Willingness-to-Wait with HUB-BUNDLED EXPRESS — adaptive bin-packing.

Key improvement over willingness_to_wait_2d.py:
  - Non-delivery-day express demand is BUNDLED per (hub, day) into adaptive
    groups, each kept in the training-distribution range (~2000 parcels, ~100 km²)
  - Each bundle gets ONE LGB cost prediction (hub-level tour) instead of per-PLZ
  - Matches what production pipeline does in `_hub_express_day_ml`

Bin-packing strategy (greedy first-fit-decreasing on parcels):
  - Sort PLZ at (hub, day) by express-parcels descending
  - Pack into bundles aiming for TARGET_PARCELS_PER_BUNDLE
  - Hard cap at MAX_PARCELS_PER_BUNDLE (= 95th-percentile of training)
  - If only 1 PLZ at hub: still bundled (= itself); flag if tiny
  - Track per-bundle: (n_parcels, area, density) → distance to training distribution

Tracks training-distribution status:
  - Each bundle reports if it's WITHIN training [min,max] of n_parcels & area
  - Flagged bundles count percentage at each (max_hold, fast_share)
  - Reported in REPORT.md

Outputs (results/willingness_hub_bundled/):
    figH1_cost_vs_share_per_window.{png,pdf}
    figH2_distance_vs_share.{png,pdf}
    figH3_fleet_vs_share.{png,pdf}
    figH4_pareto_distance_fleet.{png,pdf}
    figH5_bundle_quality.{png,pdf}        — % bundles in-distribution + bundle-size histogram
    figH6_per_provider_cost.{png,pdf}
    tab_2d_curve.csv                       (with hub-bundled vs per-PLZ comparison column)
    tab_bundle_stats.csv                   per-bundle features + ID-status
    REPORT.md
"""
from __future__ import annotations
import os, pickle, sys, warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "savefig.bbox": "tight", "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 1.4,
})

MODE = os.environ.get("WW_MODE", "auto")
CHK_PROD = ROOT / "results" / "checkpoints"
CHK_V2 = ROOT / "results" / "checkpoints" / "archive" / "pre_merge_fix_2026_05_25"
CHK = CHK_V2 if MODE == "v2" else CHK_PROD
V3 = ROOT / "results" / "sweep_v3_mergefix"
V2_RUN = ROOT / "results" / "oracle_loop_extended_2026_05_22"
OUT_NAME = "willingness_hub_bundled_v2" if MODE == "v2" else "willingness_hub_bundled"
OUT = ROOT / "results" / OUT_NAME
OUT.mkdir(parents=True, exist_ok=True)
print(f"[mode] WW_MODE={MODE} -> CHK={CHK.relative_to(ROOT)}, out={OUT.name}")

PROVIDERS = ["Amazon", "DHL", "DPD", "FedEx", "GLS", "Hermes", "UPS"]
DAYS_DE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa"]
WIN_COLOR = {1: "#003049", 2: "#2a9d8f", 3: "#e76f51"}
N_DAYS = 6

# Bin-packing parameters (from training distribution analysis)
TARGET_PARCELS_PER_BUNDLE = 2000     # ~median × 2, well-supported
MAX_PARCELS_PER_BUNDLE = 4000        # p90 of training (3873 rounded up)
MAX_AREA_PER_BUNDLE = 140             # p90 of training area_km2

# In-distribution check ranges (training p5-p95)
ID_RANGES = {
    "n_parcels": (100, 4000),
    "area_km2": (1.5, 140),
    "parcels_per_km2": (5, 700),
}


def enumerate_schedules(max_hold):
    out = []
    min_freq = max(1, int(np.ceil(N_DAYS / max_hold))) if max_hold > 0 else N_DAYS
    for k in range(min_freq, N_DAYS + 1):
        for combo in combinations(range(N_DAYS), k):
            days = sorted(combo)
            ok = True
            for i in range(len(days)):
                gap = (days[(i + 1) % len(days)] - days[i]) % N_DAYS
                if gap == 0:
                    gap = N_DAYS
                if gap > max_hold:
                    ok = False
                    break
            if ok:
                out.append(frozenset(days))
    return out


def load_state():
    from batch_delivery.surrogate.lgb_adapter import LGBLogTSurrogate
    cost_path = (V2_RUN / "production_lgb_logT_v2.pkl") if MODE == "v2" else (
        V3 / "production_lgb_logT_v3.pkl" if (V3 / "production_lgb_logT_v3.pkl").exists()
        else V2_RUN / "production_lgb_logT_v2.pkl"
    )
    print(f"[model] cost: {cost_path.relative_to(ROOT)}")
    cost_model = LGBLogTSurrogate.load(cost_path)

    chk_04 = CHK / "04_optim_prep.pkl"
    chk_01 = CHK / "01_demand.pkl"
    if not chk_04.exists() or not chk_01.exists():
        raise FileNotFoundError("Need 01_demand and 04_optim_prep checkpoints")
    optimization_data = pickle.load(open(chk_04, "rb"))["optimization_data"]
    provider_data = pickle.load(open(chk_01, "rb"))["provider_data"]
    return optimization_data, cost_model, provider_data


def build_ml_prep(provider_data):
    from batch_delivery.config.constants import N_DAYS, provider_to_demand_prefix
    ml_prep = {}
    for prov in PROVIDERS:
        pdata = provider_data.get(prov)
        if pdata is None: continue
        df_assign = pdata["df_assignments"]
        hub_coords_by_plz = {row["plz"]: (row["hub_lon"], row["hub_lat"])
                              for _, row in df_assign.iterrows()}
        hub_name_by_plz = dict(zip(df_assign["plz"], df_assign["hub_name"]))
        prefix = provider_to_demand_prefix(prov)
        col_total = f"{prefix}_total"
        plz_day_coords = {}
        for pc in pdata["all_plz_set"]:
            plz_day_coords[pc] = {}
            for d in range(N_DAYS):
                gdf_d = pdata["daily_gdfs_wgs"].get(d)
                if gdf_d is None: continue
                pts = gdf_d[gdf_d["plz"] == pc]
                if len(pts) == 0: continue
                lons = pts["lon"].values.astype(np.float64)
                lats = pts["lat"].values.astype(np.float64)
                psd = (pts[col_total].values.astype(np.float64) if col_total in pts.columns
                        else np.ones(len(pts)))
                plz_day_coords[pc][d] = (lons, lats, psd)
        ml_prep[prov] = {"plz_day_coords": plz_day_coords,
                          "hub_coords_by_plz": hub_coords_by_plz,
                          "hub_name_by_plz": hub_name_by_plz}
    return ml_prep


def adaptive_bin_pack(plz_demands: list[tuple[str, int, float]],
                       target: int = TARGET_PARCELS_PER_BUNDLE,
                       max_cap: int = MAX_PARCELS_PER_BUNDLE) -> list[list[tuple]]:
    """Balanced partitioning (LPT scheduling) — keeps all bundles similarly sized.

    Fix 2026-05-26: previously FFD would leave a tiny leftover bundle out-of-
    distribution. New approach:
      1. Compute total parcels at (hub, day).
      2. Decide n_bundles = ceil(total / target), capped so each bundle ≤ max_cap.
      3. Sort PLZ desc by parcels; assign each to the bundle with currently
         lowest total (Longest-Processing-Time-first, LPT). Yields bundles
         within ~max(plz_parcels) of each other — typically much closer to
         training-distribution.
      4. If total area would exceed bundle_area_cap, increase n_bundles.

    plz_demands : list of (plz, n_express_parcels, area_km2)
    Returns: list of bundles, each is list of (plz, parcels, area)
    """
    if not plz_demands:
        return []

    total_parcels = sum(p[1] for p in plz_demands)
    total_area = sum(p[2] for p in plz_demands)
    if total_parcels == 0:
        return []

    # Decide bundle count: enough to satisfy max_cap AND target
    n_by_parcels = max(1, int(np.ceil(total_parcels / target)))
    n_by_max = max(1, int(np.ceil(total_parcels / max_cap)))
    n_by_area = max(1, int(np.ceil(total_area / MAX_AREA_PER_BUNDLE)))
    n_bundles = max(n_by_parcels, n_by_max, n_by_area)

    # Longest-Processing-Time-first balanced assignment
    sorted_pd = sorted(plz_demands, key=lambda x: -x[1])
    bundles = [[] for _ in range(n_bundles)]
    bundle_parcels = np.zeros(n_bundles, dtype=np.float64)
    bundle_areas = np.zeros(n_bundles, dtype=np.float64)
    for plz, parcels, area in sorted_pd:
        # Assign to bundle with current minimum parcels (LPT rule)
        idx = int(np.argmin(bundle_parcels))
        bundles[idx].append((plz, parcels, area))
        bundle_parcels[idx] += parcels
        bundle_areas[idx] += area

    # Drop empty bundles (can happen if n_bundles > n_plz)
    return [b for b in bundles if len(b) > 0]


def compute_bundle_features(bundle: list[tuple], day: int, ml_prep_prov: dict,
                              plz_data_prov: dict, fast_share: float, provider: str):
    """Build the 25-feature vector for a hub-express bundle (mirrors _hub_express_day_ml).

    Each PLZ contributes its express demand (fast_share × today_demand at that PLZ).
    """
    from batch_delivery.features import (
        compute_tier1_features, compute_tier2_features,
        compute_tier3_features, _PROVIDER_IDX, ALL_COLS, TIER2_COLS,
    )

    plz_codes = [b[0] for b in bundle]
    plz_parcels = np.array([b[1] for b in bundle], dtype=np.float64)
    plz_areas = np.array([b[2] for b in bundle], dtype=np.float64)
    tot_dem = int(round(plz_parcels.sum()))
    tot_area = float(plz_areas.sum())

    if tot_dem <= 0:
        return None

    # Merge per-stop coordinates from contributing PLZ (scaled by fast_share)
    coords_lon, coords_lat, coords_psd = [], [], []
    n_stops_est = 0
    b2c_total = 0
    for pc, parcels, _ in bundle:
        coords = ml_prep_prov["plz_day_coords"].get(pc, {}).get(day)
        if coords is None or len(coords[0]) == 0:
            n_stops_est += max(1, parcels // 25)  # fallback heuristic
            continue
        lons, lats, psd = coords
        coords_lon.append(lons)
        coords_lat.append(lats)
        coords_psd.append(psd * fast_share)
        n_stops_est += len(lons)
        pd_ = plz_data_prov.get(pc, {})
        b2c_share = pd_.get("b2c", {}).get(day, 0)
        b2b_share = pd_.get("b2b", {}).get(day, 0)
        if b2c_share + b2b_share > 0:
            b2c_total += parcels * b2c_share / (b2c_share + b2b_share)

    if not coords_lon:
        return None

    merged_lon = np.concatenate(coords_lon)
    merged_lat = np.concatenate(coords_lat)
    merged_psd = np.concatenate(coords_psd)

    # Hub coords from first PLZ in bundle (all share the same hub by construction)
    hlon, hlat = ml_prep_prov["hub_coords_by_plz"].get(plz_codes[0], (9.73, 52.38))

    # Features
    t1 = compute_tier1_features(tot_dem, max(1, n_stops_est), tot_area, 0.0)  # hub_dist=0
    t2 = compute_tier2_features(merged_lon, merged_lat, hlon, hlat, merged_psd)
    t3 = compute_tier3_features(
        merged_psd if len(merged_psd) > 0 else np.array([tot_dem]),
        int(b2c_total), tot_dem,
        provider, day, 1,  # delivery_frequency=1 for express
    )
    feats = {**t1, **t2, **t3}
    return {
        "feats": np.array([feats[c] for c in ALL_COLS], dtype=np.float64),
        "n_parcels": tot_dem,
        "n_stops": max(1, n_stops_est),
        "area_km2": tot_area,
        "parcels_per_km2": tot_dem / max(0.01, tot_area),
        "n_plz": len(bundle),
    }


def predict_bundle_cost(features: np.ndarray, cost_model) -> float:
    """Predict cost for a single bundle via the LGB model.

    `features` is the 25-element BASE feature vector. predict_single builds
    the 44-combo internally — do NOT pre-build it.
    """
    if hasattr(cost_model, "predict_single"):
        return float(cost_model.predict_single(features))
    from batch_delivery.features import ALL_COLS
    df = pd.DataFrame(features.reshape(1, -1), columns=ALL_COLS)
    return float(cost_model.predict(df)[0])


def is_in_distribution(bundle_stats: dict) -> bool:
    """Check if bundle features are within training p5-p95 range."""
    for k, (lo, hi) in ID_RANGES.items():
        if k in bundle_stats:
            v = bundle_stats[k]
            if not (lo <= v <= hi):
                return False
    return True


def solve_with_hub_bundled_express(max_hold: int, fast_share: float,
                                     optimization_data, ml_prep, cost_model):
    """Per-PLZ batched delivery + hub-bundled express.

    Returns total cost, distance estimate, fleet estimate, plus bundle stats.
    """
    from batch_delivery.optimization.core import build_cost_matrices_ml

    schedules = enumerate_schedules(max_hold)
    total_cost_batched = 0.0
    total_cost_express = 0.0
    total_dist_approx = 0.0
    fleet_per_hub_day = defaultdict(lambda: np.zeros(N_DAYS))
    bundle_records = []

    # Cost-to-distance / routes ratios for cost-only model approximation
    COST_TO_DIST_KM = 0.155      # km per €
    COST_TO_ROUTES = 0.0035       # routes per €

    for prov in PROVIDERS:
        if prov not in optimization_data or prov not in ml_prep:
            continue
        odata = optimization_data[prov]
        prep = ml_prep[prov]
        plz_keys = odata["plz_keys"]
        plz_data = odata["plz_data"]

        # ── A) Compute delivery-day costs using batch-only mode (fast_share=0)
        #    so cost_3d contains the (1-fast_share)-scaled accumulated demand cost.
        #    Then we scale per delivery day cell by (1 - fast_share)... no actually
        #    we want batched-cost = cost_of_routing((1-fast_share) × accumulated_demand).
        #    The cleanest is to call build_cost_matrices_ml with fast_share which
        #    correctly subtracts express from delivery-day demand, and ZERO OUT
        #    the non-delivery-day cells (which would be per-PLZ express).
        matrices = build_cost_matrices_ml(
            plz_keys, plz_data, schedules, cost_model, prov,
            prep["plz_day_coords"], prep["hub_coords_by_plz"],
            fast_share_b2c=fast_share, fast_share_b2b=fast_share,
        )
        cost_3d = matrices["cost_3d"]            # (n_plz, n_sched, n_days)
        sched_active = matrices["sched_active"]  # (n_sched, n_days)

        # Only delivery-day cells = batched cost
        delivery_cost_3d = cost_3d * sched_active[None, :, :].astype(np.float64)
        delivery_cost_per_sched = delivery_cost_3d.sum(axis=2)  # (n_plz, n_sched)

        # Pick optimal schedule per PLZ on delivery-day cost only;
        # express cost is added later (hub-bundled, distributed back)
        chosen_idx = delivery_cost_per_sched.argmin(axis=1)
        chosen_delivery_cost = delivery_cost_per_sched[
            np.arange(len(plz_keys)), chosen_idx
        ]
        total_cost_batched += float(chosen_delivery_cost.sum())

        # ── B) Hub-bundled express
        # For each PLZ, which days is it on a non-delivery day?
        # non_delivery_days_per_plz[pi] = [days where schedule does not deliver]
        if fast_share > 0:
            # Per-PLZ today-demand × fast_share gives express parcels for each day
            for d in range(N_DAYS):
                # Find PLZ that have express today (chosen schedule does NOT deliver on d)
                per_hub_express = defaultdict(list)  # hub_name -> [(plz, parcels, area), ...]
                for pi, pc in enumerate(plz_keys):
                    sched = schedules[int(chosen_idx[pi])]
                    if d in sched:
                        continue  # this PLZ delivers today, no express
                    pd_ = plz_data.get(pc, {})
                    b2c_today = pd_.get("b2c", {}).get(d, 0)
                    b2b_today = pd_.get("b2b", {}).get(d, 0)
                    express_parcels = int(round(
                        b2c_today * fast_share + b2b_today * fast_share
                    ))
                    if express_parcels <= 0:
                        continue
                    hub = prep["hub_name_by_plz"].get(pc, "?")
                    area = pd_.get("area_km2", 1.0)
                    per_hub_express[hub].append((pc, express_parcels, area))

                # Bin-pack each hub's express PLZ into bundles
                for hub, plzs in per_hub_express.items():
                    bundles = adaptive_bin_pack(plzs)
                    for bundle in bundles:
                        bf = compute_bundle_features(
                            bundle, d, prep, plz_data, fast_share, prov,
                        )
                        if bf is None:
                            continue
                        cost = predict_bundle_cost(bf["feats"], cost_model)
                        in_dist = is_in_distribution(bf)
                        total_cost_express += cost
                        total_dist_approx += cost * COST_TO_DIST_KM
                        fleet_per_hub_day[(prov, hub)][d] += cost * COST_TO_ROUTES
                        bundle_records.append({
                            "max_hold": max_hold,
                            "fast_share": fast_share,
                            "share_willing": 1 - fast_share,
                            "provider": prov, "hub": hub, "day": d,
                            "n_plz_in_bundle": bf["n_plz"],
                            "n_parcels": bf["n_parcels"],
                            "area_km2": bf["area_km2"],
                            "parcels_per_km2": bf["parcels_per_km2"],
                            "cost": cost,
                            "in_distribution": in_dist,
                        })

        # Add delivery-day distance approximation
        total_dist_approx += chosen_delivery_cost.sum() * COST_TO_DIST_KM
        for pi, sidx in enumerate(chosen_idx):
            pc = plz_keys[pi]
            hub = prep["hub_name_by_plz"].get(pc, "?")
            for d in range(N_DAYS):
                if sched_active[int(sidx), d]:
                    fleet_per_hub_day[(prov, hub)][d] += (
                        cost_3d[pi, int(sidx), d] * COST_TO_ROUTES
                    )

    fleet_total = sum(np.max(v) for v in fleet_per_hub_day.values())
    return {
        "max_hold": max_hold,
        "fast_share": fast_share,
        "share_willing": 1 - fast_share,
        "total_cost_eur": total_cost_batched + total_cost_express,
        "cost_batched": total_cost_batched,
        "cost_express": total_cost_express,
        "total_dist_km_approx": total_dist_approx,
        "fleet_size_approx": fleet_total,
        "n_bundles": len(bundle_records),
        "n_bundles_in_dist": sum(1 for b in bundle_records if b["in_distribution"]),
        "bundle_records": bundle_records,
    }


def main():
    optimization_data, cost_model, provider_data = load_state()
    ml_prep = build_ml_prep(provider_data)
    print(f"[prep] {len(ml_prep)} providers")

    share_grid = np.linspace(0.0, 1.0, 11)
    max_hold_grid = [1, 2, 3]
    rows = []
    all_bundles = []

    for mh in max_hold_grid:
        n_sched = len(enumerate_schedules(mh))
        print(f"\n=== max_hold={mh} ({n_sched} schedules) ===")
        for fs in share_grid:
            res = solve_with_hub_bundled_express(
                mh, fs, optimization_data, ml_prep, cost_model,
            )
            n_in = res["n_bundles_in_dist"]
            n_tot = max(1, res["n_bundles"])
            pct_in = 100.0 * n_in / n_tot
            print(f"  share={1-fs:.2f}  cost={res['total_cost_eur']/1e3:6.1f}k EUR  "
                    f"(batch={res['cost_batched']/1e3:5.1f}  expr={res['cost_express']/1e3:5.1f})  "
                    f"bundles={res['n_bundles']:4d}  in-dist={pct_in:.0f}%")
            row = {k: v for k, v in res.items() if k != "bundle_records"}
            rows.append(row)
            all_bundles.extend(res["bundle_records"])

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tab_2d_curve.csv", index=False)
    pd.DataFrame(all_bundles).to_csv(OUT / "tab_bundle_stats.csv", index=False)

    # ── figH1: cost vs share, 3 windows
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for mh in max_hold_grid:
        sub = df[df["max_hold"] == mh].sort_values("share_willing")
        ax.plot(sub["share_willing"] * 100, sub["total_cost_eur"] / 1e3,
                 "o-", color=WIN_COLOR[mh], label=f"{mh}-Tage-Fenster")
    ax.set_xlabel("Share willing to wait [%]")
    ax.set_ylabel("Total weekly cost [thousand EUR]")
    ax.set_title("Cost vs. willingness — HUB-BUNDLED Express")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "figH1_cost_vs_share_per_window.png")
    fig.savefig(OUT / "figH1_cost_vs_share_per_window.pdf")
    plt.close(fig)

    # ── figH2: distance vs share
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for mh in max_hold_grid:
        sub = df[df["max_hold"] == mh].sort_values("share_willing")
        ax.plot(sub["share_willing"] * 100, sub["total_dist_km_approx"] / 1e3,
                 "o-", color=WIN_COLOR[mh], label=f"{mh}-Tage-Fenster")
    ax.set_xlabel("Share willing to wait [%]")
    ax.set_ylabel("Total weekly distance [thousand km]")
    ax.set_title("Distance vs. willingness — HUB-BUNDLED")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "figH2_distance_vs_share.png")
    fig.savefig(OUT / "figH2_distance_vs_share.pdf")
    plt.close(fig)

    # ── figH3: fleet
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for mh in max_hold_grid:
        sub = df[df["max_hold"] == mh].sort_values("share_willing")
        ax.plot(sub["share_willing"] * 100, sub["fleet_size_approx"],
                 "o-", color=WIN_COLOR[mh], label=f"{mh}-Tage-Fenster")
    ax.set_xlabel("Share willing to wait [%]")
    ax.set_ylabel("Peak fleet size (estimated)")
    ax.set_title("Fleet size vs. willingness — HUB-BUNDLED")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "figH3_fleet_vs_share.png")
    fig.savefig(OUT / "figH3_fleet_vs_share.pdf")
    plt.close(fig)

    # ── figH4: Pareto
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for mh in max_hold_grid:
        sub = df[df["max_hold"] == mh].sort_values("share_willing")
        ax.plot(sub["total_dist_km_approx"] / 1e3, sub["fleet_size_approx"],
                 "o-", color=WIN_COLOR[mh], label=f"{mh}-Tage-Fenster")
    ax.set_xlabel("Distance [thousand km]")
    ax.set_ylabel("Peak fleet")
    ax.set_title("Pareto: Distance x Fleet (HUB-BUNDLED)")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "figH4_pareto_distance_fleet.png")
    fig.savefig(OUT / "figH4_pareto_distance_fleet.pdf")
    plt.close(fig)

    # ── figH5: bundle quality (in-distribution check)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    bundle_df = pd.DataFrame(all_bundles)
    if len(bundle_df) > 0:
        # (a) % in-distribution per (max_hold, share)
        ax = axes[0]
        for mh in max_hold_grid:
            sub = bundle_df[bundle_df["max_hold"] == mh]
            g = sub.groupby("share_willing")["in_distribution"].mean() * 100
            ax.plot(g.index * 100, g.values, "o-", color=WIN_COLOR[mh],
                     label=f"{mh}-Tage")
        ax.set_xlabel("Share willing [%]")
        ax.set_ylabel("% bundles in training-distribution")
        ax.set_title("Bundle quality — % within p5-p95 of training")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(alpha=0.3)

        # (b) Bundle size histogram (n_parcels), at max_hold=3, share=0.5
        ax = axes[1]
        sub = bundle_df[(bundle_df["max_hold"] == 3) &
                         (bundle_df["share_willing"].round(1) == 0.5)]
        if len(sub) > 0:
            ax.hist(sub["n_parcels"], bins=30, color="#2a9d8f", alpha=0.8,
                     edgecolor="white")
            ax.axvline(TARGET_PARCELS_PER_BUNDLE, color="black",
                        linestyle="--", label="Target")
            ax.axvline(MAX_PARCELS_PER_BUNDLE, color="red",
                        linestyle="--", label="Max")
        ax.set_xlabel("Bundle size [n_parcels]")
        ax.set_ylabel("Count")
        ax.set_title("Bundle-size distribution at max_hold=3, share=50%")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figH5_bundle_quality.png")
    fig.savefig(OUT / "figH5_bundle_quality.pdf")
    plt.close(fig)
    print(f"\n[ok] All figures in {OUT}")

    # REPORT
    lines = [
        "# Willingness-to-Wait with Hub-Bundled Express",
        f"\n**Mode**: {MODE} | **Output**: {OUT.name}",
        f"**Bundle target**: ~{TARGET_PARCELS_PER_BUNDLE} parcels/bundle, max {MAX_PARCELS_PER_BUNDLE}, area ≤ {MAX_AREA_PER_BUNDLE} km²",
        f"**Bin-packing**: Greedy first-fit-decreasing on parcels, constrained by parcel+area caps",
        f"**In-distribution check**: bundle features within training p5-p95 ranges",
        f"\n## Headline numbers per window\n",
        "| Window | Cost (0% will) | Cost (100% will) | Saving | Dist (100%) | Fleet (100%) | % in-dist |",
        "|---|---|---|---|---|---|---|",
    ]
    for mh in max_hold_grid:
        sub = df[df["max_hold"] == mh]
        c0 = float(sub[sub["share_willing"] == 0]["total_cost_eur"].iloc[0] / 1e3)
        c1 = float(sub[sub["share_willing"] == 1]["total_cost_eur"].iloc[0] / 1e3)
        d1 = float(sub[sub["share_willing"] == 1]["total_dist_km_approx"].iloc[0] / 1e3)
        f1 = float(sub[sub["share_willing"] == 1]["fleet_size_approx"].iloc[0])
        bundle_df_sub = pd.DataFrame(all_bundles)
        if len(bundle_df_sub) > 0:
            mh_bundles = bundle_df_sub[bundle_df_sub["max_hold"] == mh]
            pct_id = 100 * mh_bundles["in_distribution"].mean() if len(mh_bundles) else 0
        else:
            pct_id = 0
        saving = (1 - c1 / c0) * 100 if c0 > 0 else 0
        lines.append(f"| **{mh} Tage** | {c0:.0f}k€ | {c1:.0f}k€ | {saving:.1f}% | "
                       f"{d1:.1f}k km | {f1:.0f} | {pct_id:.0f}% |")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] REPORT.md")


if __name__ == "__main__":
    main()
