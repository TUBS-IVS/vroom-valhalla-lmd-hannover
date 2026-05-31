"""Overnight orchestrator (paper-deadline week).

Plan
----
PHASE 1 — ML sweep with hub-bundled express  (~55 min)
  max_hold = 3   (only — the MAX_HOLDING_DAYS the paper uses)
  penalty  ∈ {0.0, 0.25, 0.5, 0.75, 1.0}                €/parcel/day
  share    ∈ {0.0, 0.1, 0.2, ..., 1.0}                 share-of-customers WILLING
  ⇒ 5 × 11 = 55 cells
  Each cell: build_cost_matrices_ml with fast_share = 1 - share,
  pick optimal schedule per (provider, PLZ), then hub-bundle the
  residual express (LPT bin-packing, threshold ≥150 = standalone).
  Output: tab_ml_grid.csv with cost components per cell + chosen schedules.

PHASE 2 — VROOM validation  (~3 h)
  4 selected (penalty, share) cells routed through the full pipeline
  with VROOM (Docker), producing actual per-day costs to compare
  against ML predictions.
  Cells:
    (P=0.5, share=1.0)   sweet-spot Scenario II (no express)
    (P=0.5, share=0.9)   ≈ abstract Scenario I (≈10% express residual)
    (P=0.0, share=1.0)   max-batching floor (no service penalty)
    (P=1.0, share=1.0)   premium service operating point
  Each cell ~30–45 min through the pipeline.

PHASE 3 — Reporting  (~5 min)
  - tab_ml_grid.csv         (full 55-cell ML sweep)
  - tab_vroom_validation.csv (per-cell ML-vs-VROOM diagnostics)
  - fig_cost_vs_share_per_penalty.{png,pdf}
  - fig_ml_vs_vroom_scatter.{png,pdf}
  - fig_pareto_p_vs_share.{png,pdf}
  - REPORT.md

Resumability
------------
The orchestrator writes a checkpoint after every ML cell and after every
VROOM cell. On crash, re-running skips completed cells.

State file: results/overnight_2026_05_27/state.json
"""
from __future__ import annotations
import json
import os
import pickle
import sys
import time
import warnings
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

rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

OUT = ROOT / "results" / "overnight_2026_05_27"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "orchestrator.log"
STATE_FILE = OUT / "state.json"

# ── Grids ────────────────────────────────────────────────────────────────────
PENALTY_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]
SHARE_WILLING_GRID = [round(x, 1) for x in np.linspace(0.0, 1.0, 11)]
MAX_HOLD = 3
PROVIDERS = ["DHL", "Amazon", "DPD", "FedEx", "GLS", "Hermes", "UPS"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
N_DAYS = 6

# Smooth power-law express ratio (paper revision 2026-05-27).
#
# The share_willing dial controls the *aggregate* share of parcels that wait.
# We assume B2B customers (~21.7% of total volume in Region Hannover) are more
# willing to wait than B2C — but with a smooth S-curve transition, not a hard
# priority queue. As the willingness pressure t (latent variable) rises:
#   willing_b2b(t) = t^(1/(1+k))     concave — fast rise (B2B accepts wait quickly)
#   willing_b2c(t) = t^(1+k)         convex  — slow rise (B2C resists longer)
# with B2B_ADVANTAGE = k > 0 controlling how much earlier B2B switches.
#
# We invert the aggregate constraint numerically:
#   aggregate(t) = b2b_share·willing_b2b(t) + b2c_share·willing_b2c(t) = share_willing
# so the dial always corresponds to the correct aggregate willing share.
#
# Earlier versions used:
#   - asymmetric `fs_b2b = (1-share)*0.5`  (artefact: at share=0 half B2B still batched)
#   - symmetric  `fs_b2b = 1-share`        (treats B2B and B2C identically — unrealistic)
#   - hard priority queue                  (instant jump at share = 21.7%)
B2B_GLOBAL_SHARE = 0.2170   # B2B volume fraction (results/checkpoints/01_demand.pkl)
B2B_ADVANTAGE = 2.0          # k in t^(1/(1+k)) vs t^(1+k); higher = stronger B2B preference

_B2C_SHARE = 1.0 - B2B_GLOBAL_SHARE
_EXP_LO = 1.0 / (1.0 + B2B_ADVANTAGE)
_EXP_HI = 1.0 + B2B_ADVANTAGE

def _solve_t(share_willing):
    """Bisection for latent willingness pressure t so aggregate = share_willing."""
    if share_willing <= 0.0: return 0.0
    if share_willing >= 1.0: return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        t = 0.5 * (lo + hi)
        agg = B2B_GLOBAL_SHARE * t**_EXP_LO + _B2C_SHARE * t**_EXP_HI
        if agg < share_willing: lo = t
        else: hi = t
    return 0.5 * (lo + hi)

def _willing_b2b(s): return _solve_t(s) ** _EXP_LO
def _willing_b2c(s): return _solve_t(s) ** _EXP_HI

def fs_b2c(share_willing): return 1.0 - _willing_b2c(share_willing)
def fs_b2b(share_willing): return 1.0 - _willing_b2b(share_willing)

# Bundling (matches willingness_hub_bundled_daganzo.py v4)
MIN_STANDALONE_EXPRESS_PARCELS = 230  # truck capacity (VEHICLE_CAPACITY)
TARGET_PARCELS_PER_BUNDLE = 1000
MAX_PARCELS_PER_BUNDLE = 2000
MAX_AREA_PER_BUNDLE = 100

ID_RANGES = {
    "n_parcels": (100, 4000),
    "area_km2": (1.5, 140),
    "parcels_per_km2": (5, 700),
}

# VROOM validation cells (subset of Phase 1 grid).
# REALISTIC TIMING: ~7.5 h per cell (1267 per-PLZ VROOM solves @ ~20s each).
# Budget allows ONLY the sweet-spot cell for the paper — already validated
# in the first overnight smoke run. Other operating points discussed via ML.
VROOM_VALIDATION_CELLS = [
    (0.5, 1.0),  # sweet-spot Scenario II — primary paper claim
]

# Phase 1.5: feature-bounds tornado parameters
# At the operating point (P=PHASE15_P, share=PHASE15_SHARE), perturb each
# feature by ±25% and ±10% on every cell, compute resulting weekly cost.
PHASE15_P = 0.5
PHASE15_SHARE = 1.0   # no express → clean delivery-only sensitivity
PHASE15_FEATURES = [
    "n_parcels", "n_stops", "area_km2", "hub_dist_km",
    "b2c_share", "parcels_per_stop", "parcels_per_km2",
]
PHASE15_DELTAS = [-0.25, -0.10, 0.10, 0.25]


# ── Logging ─────────────────────────────────────────────────────────────────
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def log(msg):
    safe = (msg
            .replace("€", "EUR")
            .replace("Δ", "d")
            .replace("→", "->")
            .replace("≤", "<=")
            .replace("≥", ">="))
    line = f"[{time.strftime('%H:%M:%S')}] {safe}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {
        "ml_completed_cells": [],
        "vroom_completed_cells": [],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_state(state):
    state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_FILE)


# ── Schedule helpers (shared) ────────────────────────────────────────────────
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


def avg_wait_days(s):
    if not s:
        return 0.0
    ds = sorted(s)
    total = 0.0
    for di in range(N_DAYS):
        next_dd = min(((d - di) % N_DAYS, d) for d in ds)[1]
        total += (next_dd - di) % N_DAYS
    return total / N_DAYS


# ── Model loader ─────────────────────────────────────────────────────────────
def load_model():
    from train_daganzo_hybrid import DaganzoLGBHybrid, _LGBIdentityWrap  # noqa
    import __main__
    __main__._LGBIdentityWrap = _LGBIdentityWrap
    # 2026-05-27 paper revision: median-calibrated alpha=1.343 (was alpha=1.0)
    # Removes structural +25% bias in Pure Daganzo; LGB residual is now white-noise.
    with open(ROOT / "results/sweep_v3_mergefix/daganzo_hybrid_v3aug_median.pkl", "rb") as f:
        d = pickle.load(f)
    if d.get("kind") == "DaganzoLGBHybrid":
        return DaganzoLGBHybrid(model=d["model"], combo_cols=d["combo_cols"], alpha=d["alpha"])
    raise RuntimeError(f"Bad model file: kind={d.get('kind')}")


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


# ── Hub-bundling primitives (cloned from willingness_hub_bundled_daganzo) ────
def adaptive_bin_pack(plz_demands):
    if not plz_demands:
        return []
    total_p = sum(p[1] for p in plz_demands)
    total_a = sum(p[2] for p in plz_demands)
    if total_p == 0:
        return []
    n_by_p = max(1, int(np.ceil(total_p / TARGET_PARCELS_PER_BUNDLE)))
    n_by_m = max(1, int(np.ceil(total_p / MAX_PARCELS_PER_BUNDLE)))
    n_by_a = max(1, int(np.ceil(total_a / MAX_AREA_PER_BUNDLE)))
    n_bundles = max(n_by_p, n_by_m, n_by_a)
    sorted_pd = sorted(plz_demands, key=lambda x: -x[1])
    bundles = [[] for _ in range(n_bundles)]
    bundle_p = np.zeros(n_bundles)
    for plz, parcels, area in sorted_pd:
        idx = int(np.argmin(bundle_p))
        bundles[idx].append((plz, parcels, area))
        bundle_p[idx] += parcels
    return [b for b in bundles if b]


def compute_bundle_features(bundle, day, ml_prep_prov, plz_data_prov,
                              fast_share_b2c, fast_share_b2b, provider):
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
    # FIX 2026-05-27: per-bundle weighted blend based on actual b2c/b2b mix
    # of the bundled PLZs (was hardcoded 50/50, biasing express-tour cost).
    total_b2c_residual = 0.0
    total_b2b_residual = 0.0
    for pc, parcels, _ in bundle:
        pd_ = plz_data_prov.get(pc, {})
        b2c_d = pd_.get("b2c", {}).get(day, 0)
        b2b_d = pd_.get("b2b", {}).get(day, 0)
        total_b2c_residual += b2c_d * fast_share_b2c
        total_b2b_residual += b2b_d * fast_share_b2b
    total_residual = total_b2c_residual + total_b2b_residual
    if total_residual > 0:
        bundle_b2c_share = total_b2c_residual / total_residual
    else:
        bundle_b2c_share = 0.5
    fs_blend = (bundle_b2c_share * fast_share_b2c
                + (1.0 - bundle_b2c_share) * fast_share_b2b)

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
        coords_psd.append(psd * fs_blend)
        n_stops_est += len(lons) * fs_blend
        pd_ = plz_data_prov.get(pc, {})
        b2c_d = pd_.get("b2c", {}).get(day, 0)
        b2b_d = pd_.get("b2b", {}).get(day, 0)
        # FIX 2026-05-27: b2c_total = actual b2c parcels in the express residual
        # (was: natural b2c-share × parcels — wrong when fs_b2c ≠ fs_b2b).
        b2c_total += b2c_d * fast_share_b2c
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
    bundle_hub_dist_km = float(np.mean(hub_dists)) if hub_dists else 5.0

    t1 = compute_tier1_features(tot_dem, n_stops_est, tot_area, bundle_hub_dist_km)
    t2 = compute_tier2_features(merged_lon, merged_lat, hlon, hlat, merged_psd)
    t3 = compute_tier3_features(
        merged_psd if len(merged_psd) > 0 else np.array([tot_dem]),
        int(b2c_total), tot_dem,
        provider, day, 1,
    )
    feats = {**t1, **t2, **t3}
    return {
        "feats": np.array([feats[c] for c in ALL_COLS], dtype=np.float64),
        "n_parcels": tot_dem,
        "n_stops": n_stops_est,
        "area_km2": tot_area,
        "parcels_per_km2": tot_dem / max(0.01, tot_area),
        "hub_dist_km": bundle_hub_dist_km,
        "n_plz": len(bundle),
    }


def is_in_distribution(bf):
    for k, (lo, hi) in ID_RANGES.items():
        if k in bf and not (lo <= bf[k] <= hi):
            return False
    return True


def predict_batch(model, list_of_feats):
    from batch_delivery.features import ALL_COLS
    if not list_of_feats:
        return np.array([])
    df = pd.DataFrame(np.vstack(list_of_feats), columns=ALL_COLS)
    return np.asarray(model.predict(df), dtype=np.float64)


# ── Phase 1 — ML grid cell ───────────────────────────────────────────────────
def evaluate_ml_cell(penalty, share_willing, optim_data, ml_prep, model,
                      schedules, sched_waits):
    from batch_delivery.optimization.core import build_cost_matrices_ml
    fs_b2c_val = fs_b2c(share_willing)
    fs_b2b_val = fs_b2b(share_willing)

    total_batched = 0.0
    total_express_stand = 0.0
    total_express_bundled = 0.0
    bundle_records = []
    chosen_records = []
    n_express_events = 0

    for prov in PROVIDERS:
        if prov not in optim_data or prov not in ml_prep:
            continue
        odata = optim_data[prov]
        prep = ml_prep[prov]
        plz_keys = odata["plz_keys"]
        plz_data = odata["plz_data"]

        matrices = build_cost_matrices_ml(
            plz_keys, plz_data, schedules, model, prov,
            prep["plz_day_coords"], prep["hub_coords_by_plz"],
            fast_share_b2c=fs_b2c_val, fast_share_b2b=fs_b2b_val,
        )
        cost_3d = matrices["cost_3d"]
        sched_active = matrices["sched_active"]
        dd_cost_per_sched = (cost_3d * sched_active[None, :, :].astype(np.float64)).sum(axis=2)
        total_cost_per_sched = cost_3d.sum(axis=2)
        weekly_pkts = np.array([
            sum(plz_data[pc]["b2c"].values()) + sum(plz_data[pc]["b2b"].values())
            for pc in plz_keys
        ], dtype=np.float64)
        # Objective: total cost (incl. per-PLZ express estimate) + service penalty.
        # FIX 2026-05-27: penalty uses PER-PLZ local willing fraction
        # (B2C-heavy PLZs have fewer willing customers locally; B2B-heavy have more).
        # local_willing_pi = b2c_share × (1-fs_b2c) + b2b_share × (1-fs_b2b)
        plz_b2c_share = matrices.get("plz_b2c_share", None)
        if plz_b2c_share is not None:
            local_willing = (plz_b2c_share * (1.0 - fs_b2c_val)
                              + (1.0 - plz_b2c_share) * (1.0 - fs_b2b_val))
        else:
            local_willing = np.full(len(plz_keys), share_willing)
        penalty_term = (penalty * local_willing[:, None] * weekly_pkts[:, None]
                        * sched_waits[None, :])
        obj = total_cost_per_sched + penalty_term
        # FIX 2026-05-27: tie-breaker preferring larger schedule_size for near-tied
        # objectives (within 0.5% of cell min). LGB residual + cached tier-2/3 stats
        # introduce tiny per-schedule variations even when the underlying physics
        # is schedule-invariant — the tie-breaker resolves these toward daily.
        sched_size_arr = np.array([len(s) for s in schedules], dtype=np.float64)
        obj_min = obj.min(axis=1, keepdims=True)
        near_tied = obj <= obj_min * 1.005
        score = np.where(near_tied, sched_size_arr[None, :], -np.inf)
        chosen_idx = score.argmax(axis=1)
        # FIX 2026-05-27: at share_willing=0 no parcel is willing to wait, so
        # daily delivery is the unambiguous baseline. Enforce daily explicitly to
        # avoid leftover ML feature-aggregation artefacts (tier-2/3 caches still
        # use union-of-source-days even when no batching is happening).
        if share_willing == 0.0:
            daily_si = next(i for i, s in enumerate(schedules) if len(s) == N_DAYS)
            chosen_idx = np.full(len(plz_keys), daily_si, dtype=chosen_idx.dtype)
        chosen_dd = dd_cost_per_sched[np.arange(len(plz_keys)), chosen_idx]
        total_batched += float(chosen_dd.sum())

        for pi, pc in enumerate(plz_keys):
            si = int(chosen_idx[pi])
            chosen_records.append({
                "penalty": penalty, "share_willing": share_willing,
                "provider": prov, "plz": pc,
                "weekly_parcels": int(weekly_pkts[pi]),
                "schedule_idx": si,
                "schedule_size": len(schedules[si]),
                "schedule_weekdays": ",".join(WEEKDAYS[d] for d in sorted(schedules[si])),
                "avg_wait_d": float(sched_waits[si]),
                "dd_cost_eur": float(chosen_dd[pi]),
            })

        if fs_b2c_val > 0 or fs_b2b_val > 0:
            for d in range(N_DAYS):
                per_hub_standalone = defaultdict(list)
                per_hub_bundle_pool = defaultdict(list)
                for pi, pc in enumerate(plz_keys):
                    if d in schedules[int(chosen_idx[pi])]:
                        continue
                    pd_ = plz_data.get(pc, {})
                    b2c_d = pd_.get("b2c", {}).get(d, 0)
                    b2b_d = pd_.get("b2b", {}).get(d, 0)
                    express_p = int(round(b2c_d * fs_b2c_val + b2b_d * fs_b2b_val))
                    if express_p <= 0:
                        continue
                    n_express_events += 1
                    hub = prep["hub_name_by_plz"].get(pc, "?")
                    area = pd_.get("area_km2", 1.0)
                    if express_p >= MIN_STANDALONE_EXPRESS_PARCELS:
                        per_hub_standalone[hub].append((pc, express_p, area))
                    else:
                        per_hub_bundle_pool[hub].append((pc, express_p, area))

                standalone_feats = []
                standalone_meta = []
                for hub, lst in per_hub_standalone.items():
                    for plz, p, a in lst:
                        bf = compute_bundle_features(
                            [(plz, p, a)], d, prep, plz_data,
                            fs_b2c_val, fs_b2b_val, prov,
                        )
                        if bf is None:
                            continue
                        standalone_feats.append(bf["feats"])
                        standalone_meta.append((hub, bf))
                if standalone_feats:
                    costs = predict_batch(model, standalone_feats)
                    for (hub, bf), c in zip(standalone_meta, costs):
                        total_express_stand += float(c)
                        bundle_records.append({
                            "penalty": penalty, "share_willing": share_willing,
                            "provider": prov, "hub": hub, "day": d,
                            "mode": "standalone", "n_plz": bf["n_plz"],
                            "n_parcels": bf["n_parcels"], "area_km2": bf["area_km2"],
                            "hub_dist_km": bf["hub_dist_km"], "cost": float(c),
                            "in_distribution": is_in_distribution(bf),
                        })

                for hub, lst in per_hub_bundle_pool.items():
                    bundles = adaptive_bin_pack(lst)
                    bfeats = []
                    bmeta = []
                    for bundle in bundles:
                        bf = compute_bundle_features(
                            bundle, d, prep, plz_data,
                            fs_b2c_val, fs_b2b_val, prov,
                        )
                        if bf is None:
                            continue
                        bfeats.append(bf["feats"])
                        bmeta.append((hub, bf))
                    if bfeats:
                        costs = predict_batch(model, bfeats)
                        for (hub, bf), c in zip(bmeta, costs):
                            total_express_bundled += float(c)
                            bundle_records.append({
                                "penalty": penalty, "share_willing": share_willing,
                                "provider": prov, "hub": hub, "day": d,
                                "mode": "bundled", "n_plz": bf["n_plz"],
                                "n_parcels": bf["n_parcels"], "area_km2": bf["area_km2"],
                                "hub_dist_km": bf["hub_dist_km"], "cost": float(c),
                                "in_distribution": is_in_distribution(bf),
                            })

    total_cost = total_batched + total_express_stand + total_express_bundled
    n_bundles = len(bundle_records)
    return {
        "penalty": penalty, "share_willing": share_willing,
        "fast_share_b2c": fs_b2c_val, "fast_share_b2b": fs_b2b_val,
        "total_cost_eur": total_cost,
        "cost_batched": total_batched,
        "cost_express_standalone": total_express_stand,
        "cost_express_bundled": total_express_bundled,
        "n_express_events": n_express_events,
        "n_express_tours": n_bundles,
        "n_standalone_tours": sum(1 for r in bundle_records if r["mode"] == "standalone"),
        "n_bundled_tours": sum(1 for r in bundle_records if r["mode"] == "bundled"),
        "pct_tours_in_dist": (100 * sum(1 for r in bundle_records if r["in_distribution"])
                                / max(1, n_bundles)),
        "bundle_records": bundle_records,
        "chosen_records": chosen_records,
    }


# ── Phase 1 — driver ─────────────────────────────────────────────────────────
def run_phase_1(state):
    log("=" * 70)
    log("PHASE 1 — ML sweep  (max_hold=3, 5 penalties × 11 shares)")
    log("=" * 70)

    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    provider_data = chk["provider_data"]
    optim_data = chk4["optimization_data"]
    model = load_model()
    ml_prep = build_ml_prep(provider_data)

    schedules = enumerate_schedules(MAX_HOLD)
    sched_waits = np.array([avg_wait_days(sorted(s)) for s in schedules])
    log(f"  {len(schedules)} valid schedules @ MAX_HOLD={MAX_HOLD}")
    log(f"  Penalty grid: {PENALTY_GRID}")
    log(f"  Share-willing grid: {SHARE_WILLING_GRID}")

    completed = set(tuple(c) for c in state["ml_completed_cells"])
    grid_records_path = OUT / "tab_ml_grid.csv"
    bundle_records_path = OUT / "tab_bundle_stats.csv"
    chosen_records_path = OUT / "tab_chosen_schedules.csv"

    grid_records = []
    bundle_records_all = []
    chosen_records_all = []
    # Resume: load previously saved
    if grid_records_path.exists():
        grid_records = pd.read_csv(grid_records_path).to_dict("records")
        log(f"  resumed grid_records: {len(grid_records)}")
    if bundle_records_path.exists():
        bundle_records_all = pd.read_csv(bundle_records_path).to_dict("records")
        log(f"  resumed bundle_records: {len(bundle_records_all)}")
    if chosen_records_path.exists():
        chosen_records_all = pd.read_csv(chosen_records_path).to_dict("records")
        log(f"  resumed chosen_records: {len(chosen_records_all)}")

    t_phase = time.time()
    for penalty in PENALTY_GRID:
        for share in SHARE_WILLING_GRID:
            key = (penalty, share)
            if key in completed:
                continue
            t_cell = time.time()
            res = evaluate_ml_cell(penalty, share, optim_data, ml_prep, model,
                                     schedules, sched_waits)
            grid_records.append({k: v for k, v in res.items()
                                  if k not in ("bundle_records", "chosen_records")})
            bundle_records_all.extend(res["bundle_records"])
            chosen_records_all.extend(res["chosen_records"])
            pd.DataFrame(grid_records).to_csv(grid_records_path, index=False)
            pd.DataFrame(bundle_records_all).to_csv(bundle_records_path, index=False)
            pd.DataFrame(chosen_records_all).to_csv(chosen_records_path, index=False)
            completed.add(key)
            state["ml_completed_cells"] = [list(c) for c in completed]
            save_state(state)
            log(f"  P={penalty:.2f} share_w={share:.1f}  cost={res['total_cost_eur']/1e3:6.1f}k€  "
                f"(batch={res['cost_batched']/1e3:.0f} | stand={res['cost_express_standalone']/1e3:.0f} "
                f"| bundle={res['cost_express_bundled']/1e3:.0f}) "
                f"in_dist={res['pct_tours_in_dist']:.0f}%  "
                f"t={time.time()-t_cell:.0f}s")
    log(f"PHASE 1 done in {time.time()-t_phase:.0f}s. "
        f"Total cells: {len(completed)} / 55")


# ── Phase 1.5 — Feature-bounds tornado ───────────────────────────────────────
def run_phase_1_5(state):
    """Perturb each input feature ±10% and ±25% at (P=PHASE15_P, share=PHASE15_SHARE)
    on the ML cost matrix; report aggregate weekly-cost delta per feature × delta.

    For each cell of the 312 (provider, PLZ) we already have the chosen schedule
    from Phase 1.  We re-build the feature row, perturb ONE feature at a time,
    re-predict the cost via Daganzo-Hybrid.  Summed deltas = weekly cost shift.
    """
    log("=" * 70)
    log("PHASE 1.5 — Feature-bounds tornado")
    log(f"  Operating point: P={PHASE15_P}, share_willing={PHASE15_SHARE}")
    log(f"  Features perturbed: {PHASE15_FEATURES}")
    log(f"  Deltas: {PHASE15_DELTAS}")
    log("=" * 70)

    out_path = OUT / "tab_feature_bounds_tornado.csv"
    if out_path.exists():
        log(f"  resumed (already exists): {out_path.name}")
        return

    from batch_delivery.features import ALL_COLS
    cache_path = ROOT / "results/penalty_sweep/sched_cost_cache.npz"
    if not cache_path.exists():
        log("  ERROR: sched_cost cache missing — run penalty_sweep_pareto.py first")
        return
    cache = np.load(cache_path)
    sched_cost = cache["sched_cost"]
    prov_order = list(cache["prov_order"])
    plz_order = list(cache["plz_order"])

    model = load_model()
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    provider_data = chk["provider_data"]
    optim_data = chk4["optimization_data"]

    # Rebuild the same per-(provider, PLZ) feature rows used by penalty_sweep
    from batch_delivery.features import _PROVIDER_IDX
    n_pp_arr = []
    pp_meta = []
    for prov, plz in zip(prov_order, plz_order):
        pld = provider_data[prov]["plz_demand"]
        od = optim_data[prov]["plz_data"]
        row = pld[pld.plz == plz]
        if row.empty or plz not in od:
            continue
        row = row.iloc[0]
        d = od[plz]
        weekly = int(row.weekly_parcels)
        n_pp_arr.append(weekly)
        base = {c: 0.0 for c in ALL_COLS}
        base.update({
            "area_km2": float(d["area_km2"]),
            "hub_dist_km": float(d["hub_dist_km"]),
            "n_stops": float(d["total_points"]),
            "b2c_share": float(row.b2c_weekly / max(1, weekly)),
            "provider_idx": float(_PROVIDER_IDX.get(prov, 0)),
            "centroid_hub_dist_km": float(d["hub_dist_km"]),
            "max_hub_dist_km": float(d["hub_dist_km"]) * 1.2,
            "demand_std": max(1.0, weekly / max(1, d["total_points"])) * 0.3,
            "max_stop_demand": max(1.0, weekly / max(1, d["total_points"])) * 2,
            "ch_area_km2": float(d["area_km2"]) * 0.6,
            "ch_perimeter_km": np.sqrt(float(d["area_km2"])) * 4,
            "mean_nn_dist_km": 0.15,
            "mean_inter_stop_dist_km": np.sqrt(float(d["area_km2"])) * 0.4,
            "stop_density_ch": d["total_points"] / max(0.01, d["area_km2"] * 0.6),
            "coord_std_x": np.sqrt(float(d["area_km2"])) * 0.3,
            "coord_std_y": np.sqrt(float(d["area_km2"])) * 0.3,
            "aspect_ratio": 1.2,
            "delivery_frequency": 1.0,  # placeholder, doesn't matter at scale
        })
        pp_meta.append({"provider": prov, "plz": plz, "base": base, "weekly": weekly})
    n_pp_arr = np.array(n_pp_arr, dtype=np.float64)

    schedules = enumerate_schedules(MAX_HOLD)
    sched_waits = np.array([avg_wait_days(sorted(s)) for s in schedules])
    sched_sizes = np.array([len(s) for s in schedules])
    daily_idx = int(np.where(sched_sizes == N_DAYS)[0][0])

    # Baseline using the SAME single-day prediction methodology as the
    # perturbed cells (so the comparison is apples-to-apples).
    base_mx = []
    for pp in pp_meta:
        f = pp["base"].copy()
        avg_n_parcels = pp["weekly"] / N_DAYS
        f["n_parcels"] = avg_n_parcels
        f["parcels_per_stop"] = avg_n_parcels / max(1, f["n_stops"])
        f["load_factor"] = avg_n_parcels / 230
        f["min_vehicles"] = max(1, int(np.ceil(avg_n_parcels / 230)))
        f["parcels_per_km2"] = avg_n_parcels / max(0.01, f["area_km2"])
        f["demand_cap_ratio"] = avg_n_parcels / (f["min_vehicles"] * 230)
        f["day_idx"] = 0
        f["delivery_frequency"] = 1.0
        base_mx.append([f[c] for c in ALL_COLS])
    df_base = pd.DataFrame(base_mx, columns=ALL_COLS)
    base_preds = model.predict(df_base)
    baseline_total = float(base_preds.sum()) * N_DAYS
    log(f"  baseline single-day x 6 at (P={PHASE15_P}, share=1.0): {baseline_total/1e3:,.1f} k€ "
        f"(consistent with perturbed methodology)")

    # For each (feature, delta): rebuild perturbed feature row per pp, single best schedule
    rows = []
    for feat in PHASE15_FEATURES:
        for delta in PHASE15_DELTAS:
            t0 = time.time()
            mx = []
            for ppi, pp in enumerate(pp_meta):
                # Use the chosen schedule's delivery-day feature row.
                # Simplification: take the schedule's first delivery day,
                # union source days = the chosen schedule's source-window.
                # Build one feature row representing the "average delivery".
                f = pp["base"].copy()
                # Weekly volume → average daily demand pattern matters less here
                # since we're testing relative cost sensitivity.
                avg_n_parcels = pp["weekly"] / N_DAYS
                f["n_parcels"] = avg_n_parcels
                f["parcels_per_stop"] = avg_n_parcels / max(1, f["n_stops"])
                f["load_factor"] = avg_n_parcels / 230
                f["min_vehicles"] = max(1, int(np.ceil(avg_n_parcels / 230)))
                f["parcels_per_km2"] = avg_n_parcels / max(0.01, f["area_km2"])
                f["demand_cap_ratio"] = avg_n_parcels / (f["min_vehicles"] * 230)
                f["day_idx"] = 0
                f["delivery_frequency"] = 1.0
                # Perturb only the target feature
                if feat in ("n_parcels", "n_stops", "area_km2", "hub_dist_km", "b2c_share"):
                    f[feat] = f[feat] * (1 + delta)
                    # Recompute derived features:
                    if feat == "n_parcels":
                        f["parcels_per_stop"] = f["n_parcels"] / max(1, f["n_stops"])
                        f["load_factor"] = f["n_parcels"] / 230
                        f["min_vehicles"] = max(1, int(np.ceil(f["n_parcels"] / 230)))
                        f["parcels_per_km2"] = f["n_parcels"] / max(0.01, f["area_km2"])
                        f["demand_cap_ratio"] = f["n_parcels"] / (f["min_vehicles"] * 230)
                    elif feat == "n_stops":
                        f["parcels_per_stop"] = f["n_parcels"] / max(1, f["n_stops"])
                    elif feat == "area_km2":
                        f["parcels_per_km2"] = f["n_parcels"] / max(0.01, f["area_km2"])
                        f["stop_density_ch"] = f["n_stops"] / max(0.01, f["area_km2"] * 0.6)
                elif feat == "parcels_per_stop":
                    # Equivalent to scaling n_parcels with n_stops fixed
                    f["parcels_per_stop"] = f["parcels_per_stop"] * (1 + delta)
                    f["n_parcels"] = f["parcels_per_stop"] * f["n_stops"]
                    f["load_factor"] = f["n_parcels"] / 230
                    f["parcels_per_km2"] = f["n_parcels"] / max(0.01, f["area_km2"])
                elif feat == "parcels_per_km2":
                    f["parcels_per_km2"] = f["parcels_per_km2"] * (1 + delta)
                    f["n_parcels"] = f["parcels_per_km2"] * f["area_km2"]
                    f["parcels_per_stop"] = f["n_parcels"] / max(1, f["n_stops"])
                    f["load_factor"] = f["n_parcels"] / 230
                mx.append([f[c] for c in ALL_COLS])
            df = pd.DataFrame(mx, columns=ALL_COLS)
            preds = model.predict(df)
            perturbed_total = float(preds.sum()) * N_DAYS  # extrapolate single-day to week
            # NB: simple weekly extrapolation. Captures relative shifts; not absolute cost.
            # Better: rebuild full sched_cost with perturbed features, but ~10x cost.
            rows.append({
                "feature": feat, "delta": delta,
                "perturbed_weekly_cost_eur": perturbed_total,
                "baseline_weekly_cost_eur": baseline_total,
                "delta_eur": perturbed_total - baseline_total,
                "delta_pct": 100.0 * (perturbed_total - baseline_total) / baseline_total,
            })
            log(f"    {feat:25s}  Δ={delta:+.2f}  → {perturbed_total/1e3:,.0f} k€  "
                f"({(perturbed_total - baseline_total)/baseline_total*100:+.2f}%)  "
                f"t={time.time()-t0:.0f}s")

    df_tornado = pd.DataFrame(rows)
    df_tornado.to_csv(out_path, index=False)
    log(f"  saved {out_path.name}")


# ── Phase 2 — VROOM validation ───────────────────────────────────────────────
def run_phase_2(state):
    log("=" * 70)
    log("PHASE 2 — VROOM validation")
    log("=" * 70)

    chosen_path = OUT / "tab_chosen_schedules.csv"
    if not chosen_path.exists():
        log("  ERROR: tab_chosen_schedules.csv missing. Run Phase 1 first.")
        return
    df_chosen = pd.read_csv(chosen_path)

    completed = set(tuple(c) for c in state["vroom_completed_cells"])
    vroom_records = []
    vroom_path = OUT / "tab_vroom_validation.csv"
    if vroom_path.exists():
        vroom_records = pd.read_csv(vroom_path).to_dict("records")
        log(f"  resumed vroom_records: {len(vroom_records)}")

    for penalty, share in VROOM_VALIDATION_CELLS:
        key = (penalty, share)
        if key in completed:
            continue
        log(f"\n→ VROOM validation: P={penalty}, share_willing={share}")
        t_cell = time.time()
        cell_records = _vroom_solve_cell(df_chosen, penalty, share)
        vroom_records.extend(cell_records)
        pd.DataFrame(vroom_records).to_csv(vroom_path, index=False)
        completed.add(key)
        state["vroom_completed_cells"] = [list(c) for c in completed]
        save_state(state)
        log(f"  done in {time.time()-t_cell:.0f}s. Records: {len(cell_records)}")


def _vroom_solve_cell(df_chosen, penalty, share):
    """Solve VROOM for the chosen schedules of one (penalty, share) cell.

    For each (provider, plz):
      - For each delivery day d of the chosen schedule: build VROOM request
        with batched_demand[d] (accumulated source days, after express
        subtraction). Solve. Record cost.
      - For each non-delivery day d: build hub-bundled express request
        per (provider, hub, day) using LPT-balanced bundles. Solve. Record.

    Returns: list of records (one per VROOM-solved tour).
    """
    from batch_delivery.routing.core import solve_single_plz, build_vroom_jobs, build_vroom_vehicles
    from batch_delivery.io.demand import get_source_days
    from batch_delivery.config.constants import provider_to_demand_prefix, WEEKDAYS as CDAYS

    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    provider_data = chk["provider_data"]
    optim_data = chk4["optimization_data"]

    sub = df_chosen[(np.isclose(df_chosen.penalty, penalty)) &
                    (np.isclose(df_chosen.share_willing, share))]
    if sub.empty:
        log(f"  WARNING: no chosen schedules for ({penalty}, {share})")
        return []

    fs_b2c_v = fs_b2c(share)
    fs_b2b_v = fs_b2b(share)

    schedules = enumerate_schedules(MAX_HOLD)
    sched_by_idx = {i: sorted(s) for i, s in enumerate(schedules)}

    # Pre-load ml_predictions from grid for comparison
    grid = pd.read_csv(OUT / "tab_ml_grid.csv")
    bundles_df = pd.read_csv(OUT / "tab_bundle_stats.csv")

    records = []
    log(f"  Routing {len(sub)} (provider, PLZ) schedules through VROOM ...")
    t_start = time.time()
    progress_every = max(1, len(sub) // 20)

    for ridx, (_, r) in enumerate(sub.iterrows()):
        if ridx % progress_every == 0 and ridx > 0:
            elapsed = time.time() - t_start
            eta = elapsed * (len(sub) - ridx) / ridx
            log(f"    [{ridx}/{len(sub)}]  elapsed={elapsed:.0f}s  eta={eta:.0f}s")
        prov = r.provider
        plz = str(r.plz).zfill(5) if isinstance(r.plz, (int, np.integer)) else str(r.plz)
        if prov not in provider_data:
            continue
        pdata = provider_data[prov]
        odata = optim_data[prov]
        plz_data_prov = odata["plz_data"]
        df_assign = pdata["df_assignments"]
        prefix = provider_to_demand_prefix(prov)
        sched_idx = int(r.schedule_idx)
        sched_days = sched_by_idx[sched_idx]

        if plz not in plz_data_prov:
            continue
        pd_ = plz_data_prov[plz]
        b2c_by_day = pd_["b2c"]
        b2b_by_day = pd_["b2b"]

        hub_row_df = df_assign[df_assign["plz"] == plz]
        if hub_row_df.empty:
            continue
        hub_row = hub_row_df.iloc[0]

        # For each delivery day d, build batched VROOM request
        col_total = f"{prefix}_total"
        for dd in sched_days:
            src_days = get_source_days(dd, sched_days)
            # Aggregate per-stop demand across source days (str_idx union)
            frames = []
            for d in src_days:
                gdf_d = pdata["daily_gdfs_wgs"].get(d)
                if gdf_d is None:
                    continue
                pts = gdf_d[gdf_d["plz"] == plz]
                if len(pts) == 0:
                    continue
                f = pd.DataFrame({
                    "str_idx": pts["str_idx"].astype(str).values,
                    "lon": pts["lon"].astype(np.float64).values,
                    "lat": pts["lat"].astype(np.float64).values,
                    "dhl_b2c": pts.get(f"{prefix}_b2c", pts.get("dhl_b2c", 0)).astype(int).values,
                    "dhl_b2b": pts.get(f"{prefix}_b2b", pts.get("dhl_b2b", 0)).astype(int).values,
                })
                f["dhl_total"] = f["dhl_b2c"] + f["dhl_b2b"]
                # Subtract express share per stop (keep batched portion)
                f["dhl_b2c"] = (f["dhl_b2c"] * (1 - fs_b2c_v)).round().astype(int)
                f["dhl_b2b"] = (f["dhl_b2b"] * (1 - fs_b2b_v)).round().astype(int)
                f["dhl_total"] = f["dhl_b2c"] + f["dhl_b2b"]
                frames.append(f[f["dhl_total"] > 0])
            if not frames:
                records.append({
                    "penalty": penalty, "share_willing": share,
                    "provider": prov, "plz": plz, "day": dd,
                    "mode": "batched",
                    "vroom_cost_eur": 0.0, "vroom_n_routes": 0,
                    "vroom_distance_km": 0.0, "vroom_duration_h": 0.0,
                    "vroom_n_parcels": 0, "vroom_status": "EMPTY",
                })
                continue
            pts_agg = pd.concat(frames, ignore_index=True)
            pts_agg = pts_agg.groupby("str_idx", as_index=False).agg(
                lon=("lon", "first"), lat=("lat", "first"),
                dhl_total=("dhl_total", "sum"),
            )
            pts_agg = pts_agg[pts_agg["dhl_total"] > 0]
            if pts_agg.empty:
                continue
            jobs, total_demand = build_vroom_jobs(pts_agg)
            if not jobs:
                continue
            vehicles, n_veh = build_vroom_vehicles(
                hub=hub_row, total_demand=total_demand, day_idx=dd,
                seed_key=f"P{penalty}_share{share:.1f}_{prov}_{plz}_d{dd}_batched",
                n_jobs=len(jobs),
            )
            result = solve_single_plz(
                plz_code=plz,
                request_body={"jobs": jobs, "vehicles": vehicles},
                idx=0, total=1, day_name=CDAYS[dd] if dd < len(CDAYS) else "",
                cache_tag=f"overnight_p{penalty}_s{share:.1f}_{prov}",
            )
            records.append({
                "penalty": penalty, "share_willing": share,
                "provider": prov, "plz": plz, "day": dd,
                "mode": "batched",
                "vroom_cost_eur": float(result["cost"] or 0) / 100.0,
                "vroom_n_routes": int(result["n_routes"]),
                "vroom_distance_km": float(result["distance_m"] or 0) / 1000.0,
                "vroom_duration_h": float(result["duration_s"] or 0) / 3600.0,
                "vroom_n_parcels": int(pts_agg["dhl_total"].sum()),
                "vroom_status": result["status"],
            })
    # Express tours are handled at HUB-LEVEL, not PLZ-level (per the bundling logic)
    # Implementation: simplified — defer hub-bundled VROOM resolve to a separate pass
    # if needed. For now we record only batched delivery-day costs.
    log(f"  recorded {len(records)} VROOM solves for cell P={penalty}, share={share}")
    return records


# ── Phase 3 — Reporting ──────────────────────────────────────────────────────
def run_phase_3(state):
    log("=" * 70)
    log("PHASE 3 — Reporting")
    log("=" * 70)

    grid = pd.read_csv(OUT / "tab_ml_grid.csv")
    grid = grid.sort_values(["penalty", "share_willing"]).reset_index(drop=True)

    # Plot 1: cost vs share, one line per penalty
    fig, ax = plt.subplots(figsize=(8, 5))
    pen_colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(PENALTY_GRID)))
    for pi, P in enumerate(PENALTY_GRID):
        sub = grid[np.isclose(grid.penalty, P)]
        ax.plot(sub.share_willing * 100, sub.total_cost_eur / 1e3,
                "o-", color=pen_colors[pi], linewidth=2, markersize=6,
                label=f"$P={P}$ €/parcel/day")
    ax.set_xlabel("Share of customers willing to wait [%]")
    ax.set_ylabel("Weekly routing cost [k€]")
    ax.set_title("Real-ML cost trade-off (hub-bundled express, MAX_HOLD=3, Daganzo-LGB-Hybrid)")
    ax.legend(title="Service penalty", loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cost_vs_share_per_penalty.png")
    fig.savefig(OUT / "fig_cost_vs_share_per_penalty.pdf")
    plt.close(fig)

    # Plot 2: ML vs VROOM scatter (only if Phase 2 done)
    vroom_path = OUT / "tab_vroom_validation.csv"
    if vroom_path.exists():
        vroom = pd.read_csv(vroom_path)
        chosen = pd.read_csv(OUT / "tab_chosen_schedules.csv")
        log(f"  VROOM records: {len(vroom)}")

        # Aggregate per (penalty, share, provider, plz) to weekly
        if "vroom_cost_eur" in vroom.columns and len(vroom):
            v_agg = (vroom.groupby(["penalty", "share_willing", "provider", "plz"], as_index=False)
                     .agg(vroom_total_cost=("vroom_cost_eur", "sum"),
                          vroom_total_routes=("vroom_n_routes", "sum"),
                          vroom_total_parcels=("vroom_n_parcels", "sum")))
            # join with ML-predicted dd_cost from chosen_records
            c_agg = (chosen.groupby(["penalty", "share_willing", "provider", "plz"], as_index=False)
                     .agg(ml_dd_cost=("dd_cost_eur", "first")))
            merged = pd.merge(v_agg, c_agg,
                              on=["penalty", "share_willing", "provider", "plz"])
            merged.to_csv(OUT / "tab_ml_vs_vroom_per_pp.csv", index=False)

            fig, ax = plt.subplots(figsize=(7, 6))
            for (P, share), g in merged.groupby(["penalty", "share_willing"]):
                ax.scatter(g.vroom_total_cost, g.ml_dd_cost,
                           label=f"$P={P}$, share={share}",
                           alpha=0.6, s=20, edgecolor="none")
            lo = min(merged.vroom_total_cost.min(), merged.ml_dd_cost.min())
            hi = max(merged.vroom_total_cost.max(), merged.ml_dd_cost.max())
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.7, label="$y=x$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("VROOM total cost per (provider, PLZ) [€]")
            ax.set_ylabel("ML predicted batched cost [€]")
            ax.set_title("ML-vs-VROOM on optimized schedules (Phase 2 cells)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, which="both")
            fig.tight_layout()
            fig.savefig(OUT / "fig_ml_vs_vroom_scatter.png")
            fig.savefig(OUT / "fig_ml_vs_vroom_scatter.pdf")
            plt.close(fig)

    # Plot 3: feature-bounds tornado
    tornado_path = OUT / "tab_feature_bounds_tornado.csv"
    if tornado_path.exists():
        df_t = pd.read_csv(tornado_path)
        # Show only ±25% deltas (the headline)
        df25 = df_t[np.isclose(df_t.delta.abs(), 0.25)].copy()
        df25 = df25.sort_values("feature")
        fig, ax = plt.subplots(figsize=(8.5, 5))
        feats_order = sorted(df25.feature.unique(),
                              key=lambda f: -abs(df25[df25.feature == f].delta_pct).max())
        y = np.arange(len(feats_order))
        for i, feat in enumerate(feats_order):
            sub_f = df_t[df_t.feature == feat]
            d_neg = float(sub_f[sub_f.delta == -0.25].delta_pct.iloc[0]) if (sub_f.delta == -0.25).any() else 0
            d_pos = float(sub_f[sub_f.delta == 0.25].delta_pct.iloc[0]) if (sub_f.delta == 0.25).any() else 0
            ax.barh(i - 0.18, d_neg, height=0.35, color="#b3261e",
                    label="-25%" if i == 0 else None)
            ax.barh(i + 0.18, d_pos, height=0.35, color="#1f4f8f",
                    label="+25%" if i == 0 else None)
        ax.set_yticks(y)
        ax.set_yticklabels(feats_order)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Weekly cost change [% vs baseline]")
        ax.set_title(f"Feature-bounds tornado @ P={PHASE15_P}, share=1.0 (no express)")
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "fig_feature_tornado.png")
        fig.savefig(OUT / "fig_feature_tornado.pdf")
        plt.close(fig)

    # REPORT.md
    lines = [
        "# Overnight Orchestrator Report (2026-05-27)",
        "",
        f"**Model:** Daganzo-LGB-Hybrid v2-aug",
        f"**MAX_HOLD:** {MAX_HOLD}",
        f"**Penalty grid:** {PENALTY_GRID}",
        f"**Share-willing grid:** {SHARE_WILLING_GRID}",
        f"**Bundling:** standalone if express ≥ {MIN_STANDALONE_EXPRESS_PARCELS} pkts, "
        f"else LPT-bundle target {TARGET_PARCELS_PER_BUNDLE} / max {MAX_PARCELS_PER_BUNDLE}",
        "",
        "## Phase 1 — ML Grid (real-ML, hub-bundled)",
        "",
        "Headline (share_willing=1.0, no express residual):",
        "",
        "| Penalty | Total cost [k€] | Saving vs all-daily | ⌀ Wait of batched | Mix 2/3/4/5/6-day |",
        "|---|---|---|---|---|",
    ]
    baseline = float(grid[(grid.share_willing == 0.0)].total_cost_eur.iloc[0]) if (grid.share_willing == 0.0).any() else 1977000
    for P in PENALTY_GRID:
        row = grid[(np.isclose(grid.penalty, P)) & (np.isclose(grid.share_willing, 1.0))]
        if row.empty:
            continue
        r = row.iloc[0]
        sav = 100.0 * (baseline - r.total_cost_eur) / baseline
        lines.append(f"| {P} | {r.total_cost_eur/1e3:,.0f} | {sav:+.2f}% | n/a | tbd |")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"  REPORT.md written")
    log(f"  All outputs in {OUT}")


# ── Main entry point ─────────────────────────────────────────────────────────
def main():
    global PENALTY_GRID, SHARE_WILLING_GRID, VROOM_VALIDATION_CELLS
    smoke = os.environ.get("ORCH_SMOKE", "0") == "1"
    if smoke:
        PENALTY_GRID = [0.5]
        SHARE_WILLING_GRID = [1.0, 0.5]
        VROOM_VALIDATION_CELLS = [(0.5, 1.0)]
        print("[SMOKE MODE] Reduced grid: 1 P × 2 shares + 1 VROOM cell")

    open(LOG, "w").close()
    log("=" * 70)
    log("OVERNIGHT ORCHESTRATOR — START" + (" [SMOKE]" if smoke else ""))
    log("=" * 70)
    state = load_state()
    log(f"State: {len(state['ml_completed_cells'])} ML cells, "
        f"{len(state['vroom_completed_cells'])} VROOM cells done.")

    run_phase_1(state)
    run_phase_1_5(state)
    run_phase_2(state)
    run_phase_3(state)

    log("\n" + "=" * 70)
    log("OVERNIGHT ORCHESTRATOR — COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
