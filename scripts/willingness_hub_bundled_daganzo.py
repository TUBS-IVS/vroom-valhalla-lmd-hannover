"""Willingness-to-Wait with HUB-BUNDLED EXPRESS — Daganzo-Hybrid + Smart Threshold.

Improvements over `willingness_hub_bundled.py`:
  * Loads **Daganzo-LGB-Hybrid v2-aug** (production model).
  * Adds `MIN_STANDALONE_EXPRESS_PARCELS` threshold: a PLZ with express
    volume above this threshold runs as STANDALONE delivery (its own ML
    cost call). Only PLZs below the threshold go into the bundling pool,
    so bundles are "the small ones" — exactly as the user asked
    ("nicht zu groß nur die kleinen").
  * Paper-ready labels (English, units, italic P).
  * 11-step willingness grid (0%, 10%, …, 100%) × 3 postponement windows.

How a non-delivery day is routed:
  1. Per (provider, hub, day), gather the PLZs whose chosen schedule
     does NOT deliver on this day.
  2. Compute each PLZ's express volume = `fast_share_b2c · b2c[d] +
     fast_share_b2b · b2b[d]`.
  3. PLZ with express ≥ `MIN_STANDALONE_EXPRESS_PARCELS` → its own ML
     prediction.
  4. Remaining (small) PLZs are LPT-balanced into bundles of size
     `~TARGET_PARCELS_PER_BUNDLE` (cap `MAX_PARCELS_PER_BUNDLE`, area
     cap `MAX_AREA_PER_BUNDLE`). Each bundle → one ML prediction.

Outputs (`results/willingness_hub_bundled_daganzo/`):
    figW1_cost_vs_share_per_window.{png,pdf}
    figW2_express_decomposition.{png,pdf}
    figW3_bundle_size_distribution.{png,pdf}
    figW4_in_distribution_quality.{png,pdf}
    figW5_provider_cost_share.{png,pdf}
    tab_grid.csv
    tab_bundle_stats.csv
    REPORT.md
"""
from __future__ import annotations
import pickle, sys, time, warnings
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
sys.path.insert(0, str(ROOT / "scripts"))

from batch_delivery.optimization.core import build_cost_matrices_ml  # noqa: E402

rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

OUT = ROOT / "results" / "willingness_hub_bundled_daganzo"
OUT.mkdir(parents=True, exist_ok=True)

PROVIDERS = ["DHL", "Amazon", "DPD", "FedEx", "GLS", "Hermes", "UPS"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
N_DAYS = 6
PENALTY_TARGET = 0.5
SHARE_GRID = np.linspace(0.0, 1.0, 11)   # 0.0, 0.1, ..., 1.0  (11 steps)
WINDOW_GRID = [1, 2, 3]
WIN_COLOR = {1: "#003049", 2: "#2a9d8f", 3: "#e76f51"}

# Bundling parameters
# TARGET = ~5 vehicles worth (VEHICLE_CAPACITY=230 → ~5×230 ≈ 1150 round to 1000)
# MAX    = ~10 vehicles worth, hard cap before bundle is too big for one tour-bundle
TARGET_PARCELS_PER_BUNDLE = 1000
MAX_PARCELS_PER_BUNDLE = 2000
MAX_AREA_PER_BUNDLE = 100  # km² — also tightened for smaller bundles

# Only the SMALL express loads go into bundling pool.
# Threshold: a PLZ with > ~65% of one vehicle's worth (150 of 230) of express
# parcels can stand on its own; below that it pays to merge with neighbours.
MIN_STANDALONE_EXPRESS_PARCELS = 150

ID_RANGES = {
    "n_parcels": (100, 4000),
    "area_km2": (1.5, 140),
    "parcels_per_km2": (5, 700),
}


def log(msg):
    print(msg, flush=True)


def enumerate_schedules(max_hold):
    out = []
    for k in range(1, N_DAYS + 1):
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


def avg_wait_days(schedule_days):
    if not schedule_days:
        return 0.0
    ds = sorted(schedule_days)
    total = 0.0
    for di in range(N_DAYS):
        next_dd = min(((d - di) % N_DAYS, d) for d in ds)[1]
        wait = (next_dd - di) % N_DAYS
        total += wait
    return total / N_DAYS


def load_model():
    from train_daganzo_hybrid import DaganzoLGBHybrid, _LGBIdentityWrap  # noqa
    import __main__
    __main__._LGBIdentityWrap = _LGBIdentityWrap
    with open(ROOT / "results/oracle_loop_extended_2026_05_22/daganzo_hybrid_v2aug.pkl", "rb") as f:
        d = pickle.load(f)
    if d.get("kind") == "DaganzoLGBHybrid":
        return DaganzoLGBHybrid(model=d["model"], combo_cols=d["combo_cols"],
                                 alpha=d["alpha"])
    raise RuntimeError(f"Expected DaganzoLGBHybrid, got {d.get('kind')}")


def build_ml_prep(provider_data):
    from batch_delivery.config.constants import provider_to_demand_prefix
    ml_prep = {}
    for prov in PROVIDERS:
        pdata = provider_data.get(prov)
        if pdata is None:
            continue
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
                if gdf_d is None:
                    continue
                pts = gdf_d[gdf_d["plz"] == pc]
                if len(pts) == 0:
                    continue
                lons = pts["lon"].values.astype(np.float64)
                lats = pts["lat"].values.astype(np.float64)
                psd = (pts[col_total].values.astype(np.float64)
                       if col_total in pts.columns else np.ones(len(pts)))
                plz_day_coords[pc][d] = (lons, lats, psd)
        ml_prep[prov] = {"plz_day_coords": plz_day_coords,
                          "hub_coords_by_plz": hub_coords_by_plz,
                          "hub_name_by_plz": hub_name_by_plz}
    return ml_prep


def adaptive_bin_pack(plz_demands, target=TARGET_PARCELS_PER_BUNDLE,
                       max_cap=MAX_PARCELS_PER_BUNDLE):
    """LPT-balanced partitioning of small express loads into bundles.

    plz_demands : list of (plz, n_express_parcels, area_km2)
    """
    if not plz_demands:
        return []
    total_parcels = sum(p[1] for p in plz_demands)
    total_area = sum(p[2] for p in plz_demands)
    if total_parcels == 0:
        return []

    n_by_parcels = max(1, int(np.ceil(total_parcels / target)))
    n_by_max = max(1, int(np.ceil(total_parcels / max_cap)))
    n_by_area = max(1, int(np.ceil(total_area / MAX_AREA_PER_BUNDLE)))
    n_bundles = max(n_by_parcels, n_by_max, n_by_area)

    sorted_pd = sorted(plz_demands, key=lambda x: -x[1])
    bundles = [[] for _ in range(n_bundles)]
    bundle_parcels = np.zeros(n_bundles)
    for plz, parcels, area in sorted_pd:
        idx = int(np.argmin(bundle_parcels))
        bundles[idx].append((plz, parcels, area))
        bundle_parcels[idx] += parcels
    return [b for b in bundles if b]


def compute_bundle_features(bundle, day, ml_prep_prov, plz_data_prov,
                              fast_share_b2c, fast_share_b2b, provider):
    """Build the 25-feature vector for one hub-express bundle (or standalone PLZ)."""
    from batch_delivery.features import (
        compute_tier1_features, compute_tier2_features,
        compute_tier3_features, ALL_COLS,
    )

    plz_codes = [b[0] for b in bundle]
    plz_parcels = np.array([b[1] for b in bundle], dtype=np.float64)
    plz_areas = np.array([b[2] for b in bundle], dtype=np.float64)
    tot_dem = int(round(plz_parcels.sum()))
    tot_area = float(plz_areas.sum())
    if tot_dem <= 0:
        return None

    fast_share_blend = 0.5 * fast_share_b2c + 0.5 * fast_share_b2b

    coords_lon, coords_lat, coords_psd = [], [], []
    n_stops_est = 0.0
    b2c_total = 0
    hub_dists = []
    for pc, parcels, _ in bundle:
        coords = ml_prep_prov["plz_day_coords"].get(pc, {}).get(day)
        if coords is None or len(coords[0]) == 0:
            n_stops_est += max(1, parcels // 25)
            continue
        lons, lats, psd = coords
        coords_lon.append(lons)
        coords_lat.append(lats)
        coords_psd.append(psd * fast_share_blend)
        # FIX: only fast_share of HAGRID stops are visited on express days.
        # Mirrors `ndd_stops = spd × fast_share_blend` in build_cost_matrices_ml.
        n_stops_est += len(lons) * fast_share_blend
        pd_ = plz_data_prov.get(pc, {})
        b2c_d = pd_.get("b2c", {}).get(day, 0)
        b2b_d = pd_.get("b2b", {}).get(day, 0)
        if b2c_d + b2b_d > 0:
            b2c_total += parcels * b2c_d / (b2c_d + b2b_d)
        # FIX 2026-05-26: record hub_dist per PLZ so the bundle uses
        # the average travel distance (NOT zero — a multi-PLZ tour still
        # travels hub → PLZ → ... → hub).
        hd_pc = pd_.get("hub_dist_km", None)
        if hd_pc is not None:
            hub_dists.append(float(hd_pc))
    if not coords_lon:
        return None
    n_stops_est = max(1, int(round(n_stops_est)))

    merged_lon = np.concatenate(coords_lon)
    merged_lat = np.concatenate(coords_lat)
    merged_psd = np.concatenate(coords_psd)
    hlon, hlat = ml_prep_prov["hub_coords_by_plz"].get(plz_codes[0], (9.73, 52.38))

    # Use the average hub-distance across PLZs in the bundle. Falls back to
    # a small positive number if no PLZ-level data is available.
    if hub_dists:
        bundle_hub_dist_km = float(np.mean(hub_dists))
    else:
        bundle_hub_dist_km = 5.0

    t1 = compute_tier1_features(tot_dem, max(1, n_stops_est), tot_area, bundle_hub_dist_km)
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
        "is_standalone": len(bundle) == 1,
    }


def is_in_distribution(bf):
    if not (ID_RANGES["n_parcels"][0] <= bf["n_parcels"] <= ID_RANGES["n_parcels"][1]):
        return False
    if not (ID_RANGES["area_km2"][0] <= bf["area_km2"] <= ID_RANGES["area_km2"][1]):
        return False
    if not (ID_RANGES["parcels_per_km2"][0] <= bf["parcels_per_km2"]
            <= ID_RANGES["parcels_per_km2"][1]):
        return False
    return True


def predict_batch(model, list_of_feats):
    """Predict cost for a list of 25-element feature vectors via Daganzo-Hybrid."""
    from batch_delivery.features import ALL_COLS
    if not list_of_feats:
        return np.array([])
    mx = np.vstack(list_of_feats)
    df = pd.DataFrame(mx, columns=ALL_COLS)
    return np.asarray(model.predict(df), dtype=np.float64)


def solve_window_share(max_hold, fast_share, optimization_data, ml_prep,
                        model, sched_waits_all):
    """Solve for one (window, share) cell using Daganzo-Hybrid + hub-bundled express."""
    schedules = enumerate_schedules(max_hold)
    sched_waits = np.array([avg_wait_days(sorted(s)) for s in schedules])

    fs_b2c = fast_share
    fs_b2b = fast_share * 0.5   # mirrors the abstract's 10%/5% asymmetry

    total_cost_batched = 0.0
    total_cost_standalone_express = 0.0
    total_cost_bundled_express = 0.0
    bundle_records = []
    per_pp_records = []

    for prov in PROVIDERS:
        if prov not in optimization_data or prov not in ml_prep:
            continue
        odata = optimization_data[prov]
        prep = ml_prep[prov]
        plz_keys = odata["plz_keys"]
        plz_data = odata["plz_data"]

        # ── A) Delivery-day cost matrix (real ML, fast_share subtracted)
        matrices = build_cost_matrices_ml(
            plz_keys, plz_data, schedules, model, prov,
            prep["plz_day_coords"], prep["hub_coords_by_plz"],
            fast_share_b2c=fs_b2c, fast_share_b2b=fs_b2b,
        )
        cost_3d = matrices["cost_3d"]
        sched_active = matrices["sched_active"]
        delivery_cost_3d = cost_3d * sched_active[None, :, :].astype(np.float64)
        dd_cost_per_sched = delivery_cost_3d.sum(axis=2)  # (n_plz, n_sched) batched portion

        # Pick best schedule per PLZ using TOTAL cost (delivery + per-PLZ express
        # estimate from build_cost_matrices_ml) + service penalty.  This honours
        # the trade-off that more aggressive batching leaves more express residual;
        # the per-PLZ express cost is then replaced by the (lower) bundled cost
        # downstream — but the schedule decision sees the full operating cost.
        total_cost_per_sched = cost_3d.sum(axis=2)         # (n_plz, n_sched) full
        weekly_pkts = np.array([
            sum(plz_data[pc]["b2c"].values()) + sum(plz_data[pc]["b2b"].values())
            for pc in plz_keys
        ], dtype=np.float64)
        objective = total_cost_per_sched + PENALTY_TARGET * weekly_pkts[:, None] * sched_waits[None, :]
        chosen_idx = objective.argmin(axis=1)
        chosen_dd = dd_cost_per_sched[np.arange(len(plz_keys)), chosen_idx]
        total_cost_batched += float(chosen_dd.sum())

        # ── B) Hub-bundled express on non-delivery days
        if fs_b2c > 0 or fs_b2b > 0:
            for d in range(N_DAYS):
                per_hub_standalone = defaultdict(list)  # large PLZ → own pred
                per_hub_bundle_pool = defaultdict(list)  # small PLZ → bundled
                for pi, pc in enumerate(plz_keys):
                    if d in schedules[int(chosen_idx[pi])]:
                        continue  # delivery day, no express
                    pd_ = plz_data.get(pc, {})
                    b2c_d = pd_.get("b2c", {}).get(d, 0)
                    b2b_d = pd_.get("b2b", {}).get(d, 0)
                    express_p = int(round(b2c_d * fs_b2c + b2b_d * fs_b2b))
                    if express_p <= 0:
                        continue
                    hub = prep["hub_name_by_plz"].get(pc, "?")
                    area = pd_.get("area_km2", 1.0)
                    if express_p >= MIN_STANDALONE_EXPRESS_PARCELS:
                        per_hub_standalone[hub].append((pc, express_p, area))
                    else:
                        per_hub_bundle_pool[hub].append((pc, express_p, area))

                # Standalone predictions (one feature row per large PLZ)
                standalone_features = []
                standalone_meta = []
                for hub, lst in per_hub_standalone.items():
                    for plz, p, a in lst:
                        bf = compute_bundle_features(
                            [(plz, p, a)], d, prep, plz_data, fs_b2c, fs_b2b, prov,
                        )
                        if bf is None:
                            continue
                        standalone_features.append(bf["feats"])
                        standalone_meta.append((hub, bf))
                if standalone_features:
                    costs = predict_batch(model, standalone_features)
                    for (hub, bf), c in zip(standalone_meta, costs):
                        total_cost_standalone_express += float(c)
                        bundle_records.append({
                            "provider": prov, "hub": hub, "day": d,
                            "mode": "standalone",
                            "n_plz": bf["n_plz"], "n_parcels": bf["n_parcels"],
                            "area_km2": bf["area_km2"],
                            "parcels_per_km2": bf["parcels_per_km2"],
                            "cost": float(c),
                            "in_distribution": is_in_distribution(bf),
                        })

                # Bundled predictions
                for hub, lst in per_hub_bundle_pool.items():
                    bundles = adaptive_bin_pack(lst)
                    bundle_features = []
                    bundle_meta = []
                    for bundle in bundles:
                        bf = compute_bundle_features(
                            bundle, d, prep, plz_data, fs_b2c, fs_b2b, prov,
                        )
                        if bf is None:
                            continue
                        bundle_features.append(bf["feats"])
                        bundle_meta.append((hub, bf))
                    if bundle_features:
                        costs = predict_batch(model, bundle_features)
                        for (hub, bf), c in zip(bundle_meta, costs):
                            total_cost_bundled_express += float(c)
                            bundle_records.append({
                                "provider": prov, "hub": hub, "day": d,
                                "mode": "bundled",
                                "n_plz": bf["n_plz"], "n_parcels": bf["n_parcels"],
                                "area_km2": bf["area_km2"],
                                "parcels_per_km2": bf["parcels_per_km2"],
                                "cost": float(c),
                                "in_distribution": is_in_distribution(bf),
                            })

        for pi, pc in enumerate(plz_keys):
            si = int(chosen_idx[pi])
            per_pp_records.append({
                "provider": prov, "plz": pc,
                "weekly_parcels": int(weekly_pkts[pi]),
                "schedule_size": len(schedules[si]),
                "schedule_weekdays": ",".join(WEEKDAYS[d] for d in sorted(schedules[si])),
                "avg_wait_d": float(sched_waits[si]),
                "dd_cost_eur": float(chosen_dd[pi]),
            })

    return {
        "total_cost_eur": total_cost_batched + total_cost_standalone_express + total_cost_bundled_express,
        "cost_batched": total_cost_batched,
        "cost_express_standalone": total_cost_standalone_express,
        "cost_express_bundled": total_cost_bundled_express,
        "bundle_records": bundle_records,
        "per_pp_records": per_pp_records,
        "n_standalone": sum(1 for r in bundle_records if r["mode"] == "standalone"),
        "n_bundled": sum(1 for r in bundle_records if r["mode"] == "bundled"),
    }


def main():
    t0 = time.time()
    log("Willingness × Hub-Bundled Express  —  Daganzo-LGB-Hybrid v2-aug")
    log(f"  MIN_STANDALONE_EXPRESS_PARCELS = {MIN_STANDALONE_EXPRESS_PARCELS}")
    log(f"  TARGET_PARCELS_PER_BUNDLE      = {TARGET_PARCELS_PER_BUNDLE}")
    log(f"  MAX_PARCELS_PER_BUNDLE         = {MAX_PARCELS_PER_BUNDLE}")
    log(f"  Window grid: {WINDOW_GRID} | Share grid: {SHARE_GRID}")

    log("\n[1] Loading checkpoints + model ...")
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    provider_data = chk["provider_data"]
    optim_data = chk4["optimization_data"]
    model = load_model()
    ml_prep = build_ml_prep(provider_data)
    sched_waits_all = {mh: np.array([avg_wait_days(sorted(s))
                                     for s in enumerate_schedules(mh)])
                       for mh in WINDOW_GRID}

    log("\n[2] Sweeping {} windows × {} shares = {} cells ...".format(
        len(WINDOW_GRID), len(SHARE_GRID), len(WINDOW_GRID) * len(SHARE_GRID)))
    rows = []
    all_bundles = []
    for mh in WINDOW_GRID:
        for share in SHARE_GRID:
            t_cell = time.time()
            res = solve_window_share(
                mh, share, optim_data, ml_prep, model, sched_waits_all,
            )
            n_total = max(1, len(res["bundle_records"]))
            in_dist = sum(1 for r in res["bundle_records"] if r["in_distribution"])
            rows.append({
                "window": mh,
                "fast_share": float(share),
                "share_willing": 1.0 - float(share),
                "total_cost_eur": res["total_cost_eur"],
                "cost_batched": res["cost_batched"],
                "cost_express_standalone": res["cost_express_standalone"],
                "cost_express_bundled": res["cost_express_bundled"],
                "n_express_standalone": res["n_standalone"],
                "n_express_bundled": res["n_bundled"],
                "pct_bundles_in_dist": 100 * in_dist / n_total,
            })
            for b in res["bundle_records"]:
                b.update({"window": mh, "fast_share": float(share),
                          "share_willing": 1.0 - float(share)})
                all_bundles.append(b)
            log(f"  window={mh}  share={share*100:5.1f}%  "
                f"cost={res['total_cost_eur']/1e3:6.1f}k€  "
                f"(batch={res['cost_batched']/1e3:.0f} | "
                f"stand={res['cost_express_standalone']/1e3:.0f} | "
                f"bundle={res['cost_express_bundled']/1e3:.0f}) "
                f"n_stand={res['n_standalone']} n_bund={res['n_bundled']} "
                f"in_dist={100*in_dist/n_total:.0f}%  "
                f"t={time.time()-t_cell:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tab_grid.csv", index=False)
    pd.DataFrame(all_bundles).to_csv(OUT / "tab_bundle_stats.csv", index=False)

    log("\n[3] Plotting ...")
    # figW1: cost vs share, per window
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for mh in WINDOW_GRID:
        sub = df[df.window == mh].sort_values("share_willing")
        ax.plot(sub.share_willing * 100, sub.total_cost_eur / 1e3,
                "o-", color=WIN_COLOR[mh], linewidth=2, markersize=6,
                label=f"Postponement window = {mh} day{'s' if mh != 1 else ''}")
    ax.set_xlabel("Share of customers willing to wait [%]")
    ax.set_ylabel("Weekly routing cost [k€]")
    ax.set_title("Cost vs. willingness — hub-bundled express residual\n"
                  f"Daganzo-LGB-Hybrid, $P={PENALTY_TARGET}$ €/parcel/day, "
                  f"standalone if $\\geq${MIN_STANDALONE_EXPRESS_PARCELS} pkts, "
                  f"bundle target ${TARGET_PARCELS_PER_BUNDLE}$, cap ${MAX_PARCELS_PER_BUNDLE}$")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figW1_cost_vs_share_per_window.png")
    fig.savefig(OUT / "figW1_cost_vs_share_per_window.pdf")
    plt.close(fig)

    # figW2: express decomposition at window=3
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df[df.window == 3].sort_values("share_willing")
    x = sub.share_willing * 100
    ax.fill_between(x, 0, sub.cost_batched / 1e3,
                     color="#1f4f8f", alpha=0.85, label="Batched delivery-day cost")
    ax.fill_between(x, sub.cost_batched / 1e3,
                     (sub.cost_batched + sub.cost_express_standalone) / 1e3,
                     color="#e9c46a", alpha=0.85, label="Standalone express ($\\geq$ threshold)")
    ax.fill_between(x,
                     (sub.cost_batched + sub.cost_express_standalone) / 1e3,
                     (sub.cost_batched + sub.cost_express_standalone + sub.cost_express_bundled) / 1e3,
                     color="#e76f51", alpha=0.85, label="Hub-bundled express (small loads)")
    ax.set_xlabel("Share of customers willing to wait [%]")
    ax.set_ylabel("Weekly cost component [k€]")
    ax.set_title("Cost decomposition at window = 3 days")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figW2_express_decomposition.png")
    fig.savefig(OUT / "figW2_express_decomposition.pdf")
    plt.close(fig)

    # figW3: bundle size distribution
    bdf = pd.DataFrame(all_bundles)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sub = bdf[(bdf.window == 3) & (np.isclose(bdf.fast_share, 0.10))]
    if len(sub):
        axes[0].hist([sub[sub["mode"] == "standalone"].n_parcels,
                       sub[sub["mode"] == "bundled"].n_parcels],
                       bins=30, stacked=True,
                       color=["#e9c46a", "#e76f51"],
                       label=["Standalone", "Bundled"], edgecolor="white")
        axes[0].axvline(MIN_STANDALONE_EXPRESS_PARCELS, color="black",
                          linestyle="--", label=f"Threshold ({MIN_STANDALONE_EXPRESS_PARCELS})")
        axes[0].axvline(TARGET_PARCELS_PER_BUNDLE, color="grey",
                          linestyle=":", label=f"Bundle target ({TARGET_PARCELS_PER_BUNDLE})")
        axes[0].set_xlabel("Parcels per express tour")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Bundle size distribution (window = 3, $f_s = 0.10$)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Right panel: parcels per km² (training-distribution view)
        axes[1].hist([sub[sub["mode"] == "standalone"].parcels_per_km2,
                       sub[sub["mode"] == "bundled"].parcels_per_km2],
                       bins=30, stacked=True,
                       color=["#e9c46a", "#e76f51"],
                       label=["Standalone", "Bundled"], edgecolor="white")
        axes[1].axvspan(ID_RANGES["parcels_per_km2"][0],
                          ID_RANGES["parcels_per_km2"][1],
                          color="green", alpha=0.08, label="In-training range")
        axes[1].set_xlabel("Parcels per km$^2$ per tour")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Spatial density of express tours")
        axes[1].set_xscale("log")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figW3_bundle_size_distribution.png")
    fig.savefig(OUT / "figW3_bundle_size_distribution.pdf")
    plt.close(fig)

    # figW4: in-distribution quality vs share
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mh in WINDOW_GRID:
        sub = df[df.window == mh].sort_values("share_willing")
        ax.plot(sub.share_willing * 100, sub.pct_bundles_in_dist,
                "o-", color=WIN_COLOR[mh], linewidth=2, markersize=5,
                label=f"Window = {mh} day{'s' if mh != 1 else ''}")
    ax.set_xlabel("Share of customers willing to wait [%]")
    ax.set_ylabel("Bundles in training-distribution [%]")
    ax.set_title("Bundle quality — share of tours within training $p_5$–$p_{95}$ ranges")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figW4_in_distribution_quality.png")
    fig.savefig(OUT / "figW4_in_distribution_quality.pdf")
    plt.close(fig)

    # figW5: cost saving overlay (no-bundling vs bundling)
    # We approximate "no bundling" = total cost where every small PLZ would route alone
    # via build_cost_matrices_ml's non-delivery-day cost (express_aware_sensitivity).
    # For paper purposes here we just report the absolute total cost.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for mh in WINDOW_GRID:
        sub = df[df.window == mh].sort_values("share_willing")
        savings = 100 * (sub.total_cost_eur.iloc[0] - sub.total_cost_eur) / sub.total_cost_eur.iloc[0]
        ax.plot(sub.share_willing * 100, savings, "o-",
                 color=WIN_COLOR[mh], linewidth=2, markersize=6,
                 label=f"Window = {mh} day{'s' if mh != 1 else ''}")
    ax.set_xlabel("Share of customers willing to wait [%]")
    ax.set_ylabel("Cost saving vs no-willingness baseline [%]")
    ax.set_title("Realised saving with hub-bundled express residual")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figW5_provider_cost_share.png")
    fig.savefig(OUT / "figW5_provider_cost_share.pdf")
    plt.close(fig)

    log("\n[4] REPORT.md ...")
    base_rows = df[df.window == 3].set_index("share_willing")
    c0 = float(base_rows.loc[0.0, "total_cost_eur"])
    lines = [
        "# Willingness × Hub-Bundled Express  — Daganzo-Hybrid v2-aug",
        "",
        f"**Operating point:** $P = {PENALTY_TARGET}$ €/parcel/day",
        f"**B2C express share:** `fast_share_b2c = s` (slider), "
        f"**B2B express share:** `fast_share_b2b = s / 2` (mirrors abstract 10%/5%)",
        f"**Standalone threshold:** if express parcels ≥ "
        f"{MIN_STANDALONE_EXPRESS_PARCELS} → standalone ML prediction",
        f"**Bundle target:** {TARGET_PARCELS_PER_BUNDLE} pkts, "
        f"max {MAX_PARCELS_PER_BUNDLE}, area ≤ {MAX_AREA_PER_BUNDLE} km²",
        "",
        "## Headline (window = 3 days, all 7 LSPs)",
        "",
        "| Share willing | f_s (B2C) | Total cost [k€] | Batched | Standalone Express | Bundled Express | Saving vs s=100% |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in base_rows.reset_index().iterrows():
        sav = 100.0 * (c0 - r.total_cost_eur) / c0
        lines.append(
            f"| {r.share_willing*100:.0f}% | {r.fast_share:.2f} | "
            f"{r.total_cost_eur/1e3:,.0f} | {r.cost_batched/1e3:,.0f} | "
            f"{r.cost_express_standalone/1e3:,.0f} | {r.cost_express_bundled/1e3:,.0f} | "
            f"{sav:+.2f}% |"
        )
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    log(f"\nDone in {time.time()-t0:.0f}s.  Outputs in {OUT}")
    for p in sorted(OUT.glob("*")):
        log(f"  {p.name}")


if __name__ == "__main__":
    main()
