"""Balanced overnight orchestrator — same grid as the unbalanced run, but
with `balance_fleet_per_hub_ml` applied after the cost-optimal schedule
selection.  Saves both initial (cost-optimal) and balanced schedules so
the impact of fleet balancing can be quantified per cell.

Grid:
  max_hold = 3
  penalty  ∈ {0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0}        — 8 values
  share    ∈ {0.0, 0.1, 0.2, ..., 1.0}                          — 11 values
  ⇒ 88 cells × 7 providers × ~20 s ≈ 3 h

Outputs (results/overnight_2026_05_27_balanced/):
  tab_ml_grid.csv                  same shape as unbalanced grid + balanced cost columns
  tab_chosen_schedules.csv         per (P, share, provider, plz) with BOTH unbalanced &
                                    balanced schedule
  tab_fleet_per_hub.csv            per (P, share, provider, hub, day) fleet load
                                    BEFORE and AFTER balancing
  tab_balancing_summary.csv        per cell: cost delta, imbalance reduction, swaps
  state.json
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
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from batch_delivery.features import ALL_COLS, _PROVIDER_IDX  # noqa: E402
from batch_delivery.optimization.core import (  # noqa: E402
    build_cost_matrices_ml,
    balance_fleet_per_hub_ml,
    optimize_cd_ml,
    _daily_fleet_per_hub,
    _fleet_imbalance,
)

OUT = ROOT / "results" / "overnight_2026_05_29_path2"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "orchestrator.log"
STATE_FILE = OUT / "state.json"

PENALTY_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]
SHARE_WILLING_GRID = [round(x, 1) for x in np.linspace(0.0, 1.0, 11)]
MAX_HOLD = 3
PROVIDERS = ["DHL", "Amazon", "DPD", "FedEx", "GLS", "Hermes", "UPS"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
N_DAYS = 6
FLEET_COST_BUDGET_PCT = 5.0   # max +5% cost for balancing (paper revision 2026-05-27)


# Smooth power-law model: t^(1/(1+k)) for B2B, t^(1+k) for B2C, aggregate solved numerically.
B2B_GLOBAL_SHARE = 0.2170
B2B_ADVANTAGE = 2.0
_B2C_SHARE = 1.0 - B2B_GLOBAL_SHARE
_EXP_LO = 1.0 / (1.0 + B2B_ADVANTAGE)
_EXP_HI = 1.0 + B2B_ADVANTAGE

def _solve_t(share_willing):
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
    return {"completed_cells": [], "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def save_state(state):
    state["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_FILE)


def enumerate_schedules():
    out = []
    for k in range(1, N_DAYS + 1):
        for combo in combinations(range(N_DAYS), k):
            days = sorted(combo)
            ok = True
            for i in range(len(days)):
                gap = (days[(i + 1) % len(days)] - days[i]) % N_DAYS
                if gap == 0:
                    gap = N_DAYS
                if gap > MAX_HOLD:
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


def load_model():
    from train_daganzo_hybrid import DaganzoLGBHybrid, _LGBIdentityWrap  # noqa
    import __main__
    __main__._LGBIdentityWrap = _LGBIdentityWrap
    # 2026-05-27 paper revision: median-calibrated alpha=1.343 model
    with open(ROOT / "results/sweep_v3_mergefix/daganzo_hybrid_v3aug_median.pkl", "rb") as f:
        d = pickle.load(f)
    if d.get("kind") == "DaganzoLGBHybrid":
        return DaganzoLGBHybrid(model=d["model"], combo_cols=d["combo_cols"],
                                 alpha=d["alpha"])
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


def evaluate_cell(P, share, provider_data, optim_data, ml_prep, model,
                   schedules, sched_waits):
    """Process one (P, share) cell — initial argmin → balance_fleet_per_hub_ml.
    Returns chosen records, balancing summary, and fleet records per provider.
    """
    fs_b2c_v = fs_b2c(share)
    fs_b2b_v = fs_b2b(share)

    cell_chosen = []
    cell_fleet = []
    cell_summary = []

    for prov in PROVIDERS:
        if prov not in optim_data or prov not in ml_prep:
            continue
        odata = optim_data[prov]
        prep = ml_prep[prov]
        plz_keys = odata["plz_keys"]
        plz_data = odata["plz_data"]
        plz_hub_arr = odata["plz_hub_arr"]
        hub_plz_list = odata["hub_plz_list"]

        # ── 1. Build cost matrices with current operating point
        matrices = build_cost_matrices_ml(
            plz_keys, plz_data, schedules, model, prov,
            prep["plz_day_coords"], prep["hub_coords_by_plz"],
            fast_share_b2c=fs_b2c_v, fast_share_b2b=fs_b2b_v,
        )
        cost_3d = matrices["cost_3d"]
        sched_active = matrices["sched_active"]
        veh_3d = matrices["veh_3d"]
        dd_cost = (cost_3d * sched_active[None, :, :]).sum(axis=2)
        total_cost_mx = cost_3d.sum(axis=2)

        # ── 2. Initial cost-optimal selection (under penalty)
        weekly_pkts = np.array([
            sum(plz_data[pc]["b2c"].values()) + sum(plz_data[pc]["b2b"].values())
            for pc in plz_keys
        ], dtype=np.float64)
        # FIX 2026-05-27: wait penalty uses PER-PLZ local willing fraction.
        plz_b2c_share = matrices.get("plz_b2c_share", None)
        if plz_b2c_share is not None:
            local_willing = (plz_b2c_share * (1.0 - fs_b2c_v)
                              + (1.0 - plz_b2c_share) * (1.0 - fs_b2b_v))
        else:
            local_willing = np.full(len(plz_keys), share)
        obj = (total_cost_mx
               + P * local_willing[:, None] * weekly_pkts[:, None] * sched_waits[None, :])
        # Tie-breaker: among schedules within 0.5% of the cell minimum, pick the
        # largest (daily as natural baseline at share=0 where all schedules ≈ tied).
        sched_size_arr = np.array([len(s) for s in schedules], dtype=np.float64)
        obj_min = obj.min(axis=1, keepdims=True)
        near_tied = obj <= obj_min * 1.005
        score = np.where(near_tied, sched_size_arr[None, :], -np.inf)
        chosen_init = score.argmax(axis=1).astype(np.int64)
        # At share=0 no parcel is willing to wait → enforce daily as baseline.
        if share == 0.0:
            daily_si = next(i for i, s in enumerate(schedules) if len(s) == N_DAYS)
            chosen_init = np.full(len(plz_keys), daily_si, dtype=chosen_init.dtype)

        # ── 2b. PATH 2 (2026-05-29): bundled refinement of the per-PLZ argmin via
        # optimize_cd_ml on the hub-bundled cost with the service penalty baked in.
        # Warm-started from the argmin so this converges in a few rounds. Finds the
        # non-separable hub-bundling synergies that the per-PLZ proxy misses by
        # construction.
        penalty_mx = (P * local_willing[:, None] * weekly_pkts[:, None]
                      * sched_waits[None, :])
        if share > 0.0:
            mat_pen = dict(matrices)
            mat_pen["dd_cost_mx"] = matrices["dd_cost_mx"] + penalty_mx
            cd = optimize_cd_ml(
                plz_keys, plz_hub_arr, hub_plz_list, mat_pen, schedules,
                fixed_assignment=chosen_init.astype(np.intp),
                max_rounds=8, shuffle_plz=True, seed=42,
                pair_polish=True, pair_polish_rounds=3, pair_polish_max_pairs=300,
                n_restarts=2,
            )
            chosen_init = cd["chosen"].astype(np.int64)

        # init_cost = bundled total (dd + hub-bundled express) of the refined init.
        # Use balance_fleet_per_hub_ml(max_swaps=0) which computes exactly that.
        bal0 = balance_fleet_per_hub_ml(
            {"chosen": chosen_init, "best_cost": 0.0},
            plz_keys, plz_hub_arr, hub_plz_list,
            matrices, schedules,
            cost_budget_pct=FLEET_COST_BUDGET_PCT, max_swaps=0,
        )
        init_cost = float(bal0["initial_total_cost"])

        # Fleet BEFORE balancing
        fleet_before = _daily_fleet_per_hub(
            chosen_init, plz_hub_arr, hub_plz_list, veh_3d, schedules,
        )
        imbalance_before = float(_fleet_imbalance(fleet_before))

        # ── 3. Apply ML fleet balancing (frequency-preserving). Init already has
        # the bundling synergies, so balancing only redistributes WHICH days are
        # served to flatten the per-depot vehicle peak at unchanged service.
        sa_result = {"chosen": chosen_init, "best_cost": init_cost}
        try:
            bal = balance_fleet_per_hub_ml(
                sa_result, plz_keys, plz_hub_arr, hub_plz_list,
                matrices, schedules,
                cost_budget_pct=FLEET_COST_BUDGET_PCT,
                penalty_mx=penalty_mx,
                preserve_frequency=True,
            )
            chosen_bal = bal["chosen"]
            bal_cost = float(bal["cost"])
            imbalance_after = float(bal["imbalance_after"])
            n_swaps = int(bal["swaps_made"])
        except Exception as e:
            log(f"    WARN {prov} balance failed: {e}")
            chosen_bal = chosen_init.copy()
            bal_cost = init_cost
            imbalance_after = imbalance_before
            n_swaps = 0

        fleet_after = _daily_fleet_per_hub(
            chosen_bal, plz_hub_arr, hub_plz_list, veh_3d, schedules,
        )

        # ── 4. Record cell-level summary
        cell_summary.append({
            "penalty": P, "share_willing": share, "provider": prov,
            "n_plz": len(plz_keys),
            "init_cost_eur": init_cost,
            "balanced_cost_eur": bal_cost,
            "cost_delta_eur": bal_cost - init_cost,
            "cost_delta_pct": 100.0 * (bal_cost - init_cost) / max(1, init_cost),
            "imbalance_before": imbalance_before,
            "imbalance_after": imbalance_after,
            "imbalance_reduction_pct": (100.0 *
                (imbalance_before - imbalance_after) / max(1, imbalance_before)),
            "max_fleet_before": float(fleet_before.max()),
            "max_fleet_after": float(fleet_after.max()),
            "total_routes_before": float(fleet_before.sum()),
            "total_routes_after": float(fleet_after.sum()),
            "swaps_made": n_swaps,
        })

        # ── 5. Per-(PLZ) records (both unbalanced + balanced schedule)
        for pi, pc in enumerate(plz_keys):
            si_i = int(chosen_init[pi])
            si_b = int(chosen_bal[pi])
            cell_chosen.append({
                "penalty": P, "share_willing": share,
                "provider": prov, "plz": pc,
                "weekly_parcels": int(weekly_pkts[pi]),
                "schedule_idx_init": si_i,
                "schedule_idx_balanced": si_b,
                "schedule_size_init": len(schedules[si_i]),
                "schedule_size_balanced": len(schedules[si_b]),
                "weekdays_init": ",".join(WEEKDAYS[d] for d in sorted(schedules[si_i])),
                "weekdays_balanced": ",".join(WEEKDAYS[d] for d in sorted(schedules[si_b])),
                "avg_wait_d_init": float(sched_waits[si_i]),
                "avg_wait_d_balanced": float(sched_waits[si_b]),
                "dd_cost_init": float(dd_cost[pi, si_i]),
                "dd_cost_balanced": float(dd_cost[pi, si_b]),
                "veh_init": float(veh_3d[pi, si_i].sum()),
                "veh_balanced": float(veh_3d[pi, si_b].sum()),
            })

        # ── 6. Per-hub × day fleet records
        for hi, h_plzs in enumerate(hub_plz_list):
            if len(h_plzs) == 0:
                continue
            hub_name = prep["hub_name_by_plz"].get(plz_keys[int(h_plzs[0])], f"hub_{hi}")
            for d in range(N_DAYS):
                cell_fleet.append({
                    "penalty": P, "share_willing": share,
                    "provider": prov, "hub": hub_name, "day": d,
                    "fleet_before": float(fleet_before[hi, d]),
                    "fleet_after": float(fleet_after[hi, d]),
                })

    return cell_chosen, cell_summary, cell_fleet


def main():
    smoke = os.environ.get("ORCH_SMOKE", "0") == "1"
    global PENALTY_GRID, SHARE_WILLING_GRID
    if smoke:
        PENALTY_GRID = [0.5]
        SHARE_WILLING_GRID = [1.0, 0.5]
        print("[SMOKE] Reduced grid: 1 P × 2 shares = 2 cells")

    open(LOG, "w").close()
    log("=" * 70)
    log("BALANCED OVERNIGHT ORCHESTRATOR — START" + (" [SMOKE]" if smoke else ""))
    log("=" * 70)
    log(f"  Penalty grid: {PENALTY_GRID}")
    log(f"  Share-willing grid: {SHARE_WILLING_GRID}")
    log(f"  Fleet-balance cost budget: {FLEET_COST_BUDGET_PCT}%")

    state = load_state()
    completed = set(tuple(c) for c in state["completed_cells"])

    log("\n[1] Loading checkpoints + model ...")
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    provider_data = chk["provider_data"]
    optim_data = chk4["optimization_data"]
    model = load_model()
    ml_prep = build_ml_prep(provider_data)
    schedules = enumerate_schedules()
    sched_waits = np.array([avg_wait_days(sorted(s)) for s in schedules])
    log(f"    {len(schedules)} valid schedules @ MAX_HOLD={MAX_HOLD}")

    # Init / load tables
    chosen_path = OUT / "tab_chosen_schedules.csv"
    summary_path = OUT / "tab_balancing_summary.csv"
    fleet_path = OUT / "tab_fleet_per_hub.csv"
    chosen_rows = []
    summary_rows = []
    fleet_rows = []
    if chosen_path.exists():
        chosen_rows = pd.read_csv(chosen_path).to_dict("records")
    if summary_path.exists():
        summary_rows = pd.read_csv(summary_path).to_dict("records")
    if fleet_path.exists():
        fleet_rows = pd.read_csv(fleet_path).to_dict("records")
    log(f"    resumed: {len(completed)} cells done, "
        f"{len(chosen_rows)} chosen rows, {len(summary_rows)} summary rows")

    log(f"\n[2] Sweeping {len(PENALTY_GRID) * len(SHARE_WILLING_GRID)} cells ...")
    t_phase = time.time()
    for P in PENALTY_GRID:
        for share in SHARE_WILLING_GRID:
            key = (P, share)
            if key in completed:
                continue
            t_cell = time.time()
            cell_chosen, cell_summary, cell_fleet = evaluate_cell(
                P, share, provider_data, optim_data, ml_prep, model,
                schedules, sched_waits,
            )
            chosen_rows.extend(cell_chosen)
            summary_rows.extend(cell_summary)
            fleet_rows.extend(cell_fleet)
            pd.DataFrame(chosen_rows).to_csv(chosen_path, index=False)
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            pd.DataFrame(fleet_rows).to_csv(fleet_path, index=False)
            completed.add(key)
            state["completed_cells"] = [list(c) for c in completed]
            save_state(state)

            # Aggregate cell-level numbers
            total_init = sum(r["init_cost_eur"] for r in cell_summary)
            total_bal = sum(r["balanced_cost_eur"] for r in cell_summary)
            total_swaps = sum(r["swaps_made"] for r in cell_summary)
            max_fleet_before = sum(r["max_fleet_before"] for r in cell_summary)
            max_fleet_after = sum(r["max_fleet_after"] for r in cell_summary)
            log(f"  P={P:.2f} share_w={share:.1f}  "
                f"init={total_init/1e3:6.1f} bal={total_bal/1e3:6.1f}kEUR "
                f"(d={total_bal-total_init:+.0f}EUR)  "
                f"max_fleet {max_fleet_before:.0f}->{max_fleet_after:.0f}  "
                f"swaps={total_swaps:3d}  t={time.time()-t_cell:.0f}s")

    log(f"\n[3] DONE in {time.time()-t_phase:.0f}s. Total: {len(completed)} cells.")
    log(f"  Outputs in {OUT}")


if __name__ == "__main__":
    main()
