"""VROOM validation of Path-2 balanced schedules at 4 operating points:

  (P = 0.0, theta = 1.0)   cost-optimal, full batching
  (P = 0.0, theta = 0.3)   cost-optimal, low willingness
  (P = 0.5, theta = 1.0)   sweet-spot, full batching
  (P = 0.5, theta = 0.3)   sweet-spot, low willingness

Same primitives as paper_vroom_validate_balanced.py but redirected to the
Path-2 chosen-schedules CSV and a clean output folder.

Resumable. Run overnight: each VROOM solve takes ~5-30 s, total ~4-8 h.
"""
from __future__ import annotations
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import overnight_orchestrator_2026_05_27 as orch  # noqa: E402

from batch_delivery.routing.core import (  # noqa: E402
    solve_single_plz, build_vroom_jobs, build_vroom_vehicles,
)
from batch_delivery.io.demand import get_source_days  # noqa: E402
from batch_delivery.config.constants import (  # noqa: E402
    provider_to_demand_prefix, WEEKDAYS as CDAYS,
)

PATH2 = ROOT / "results" / "overnight_2026_05_29_path2"
OUT = ROOT / "results" / "paper_results_final" / "07_validation"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINT = OUT / "tab_vroom_path2.csv"

# Per-provider individual sweet-spot (chord-distance knee from per-prov Pareto).
# All validated at theta = 1.0 (top-scenario, full batching).
# Each provider is ALSO validated at P = 0 as the cost-aggressive baseline.
PROVIDER_KNEES = {
    "Amazon": 0.25,
    "DHL":    0.25,
    "DPD":    0.75,
    "FedEx":  0.50,
    "GLS":    0.75,
    "Hermes": 0.50,
    "UPS":    0.50,
}
VALIDATION_KEYS: list[tuple[float, float, str]] = []
for prov, knee in PROVIDER_KNEES.items():
    VALIDATION_KEYS.append((0.0, 1.0, prov))   # cost-aggressive (P=0)
    VALIDATION_KEYS.append((knee, 1.0, prov))  # individual sweet-spot

N_DAYS = orch.N_DAYS
MAX_HOLD = orch.MAX_HOLD
PROVIDERS = orch.PROVIDERS


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_done_keys() -> set:
    if not CHECKPOINT.exists():
        return set()
    df = pd.read_csv(CHECKPOINT)
    if not len(df):
        return set()
    df["plz"] = df.plz.astype(str)
    return set(zip(df.penalty, df.share_willing, df.provider, df.plz))


def append_rows(rows: list) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(CHECKPOINT, mode="a", index=False,
              header=not CHECKPOINT.exists())


def solve_plz_batched(prov, plz, sched_days, pdata, plz_data_prov,
                      hub_row, prefix, P, share, fs_b2c_v, fs_b2b_v):
    """Solve VROOM for each delivery day of a (provider, plz) cell."""
    schedules_per_plz = [None] * N_DAYS
    for d in sorted(sched_days):
        schedules_per_plz[d] = sched_days
    rows = []
    for dd in sorted(sched_days):
        src_days = get_source_days(dd, sched_days)
        # parcels-of-day for this delivery day (batched portion only)
        b2c = sum(plz_data_prov.get("b2c", {}).get(d, 0) for d in src_days)
        b2b = sum(plz_data_prov.get("b2b", {}).get(d, 0) for d in src_days)
        b2c_batched = int(round(b2c * (1.0 - fs_b2c_v)))
        b2b_batched = int(round(b2b * (1.0 - fs_b2b_v)))
        n_total = b2c_batched + b2b_batched
        if n_total <= 0:
            continue
        try:
            res = solve_single_plz(
                prov, plz, dd, b2c_batched, b2b_batched,
                pdata, plz_data_prov, hub_row, prefix,
                seed_key=f"P{P}_share{share:.1f}_{prov}_{plz}_d{dd}_batched",
            )
            rows.append({
                "penalty": P, "share_willing": share,
                "provider": prov, "plz": plz, "day": dd,
                "n_parcels": n_total,
                "vroom_cost_eur": float(res.get("cost_eur", 0.0)),
                "vroom_n_routes": int(res.get("n_routes", 0)),
                "vroom_distance_km": float(res.get("distance_km", 0.0)),
                "vroom_status": res.get("status", "?"),
            })
        except Exception as e:  # noqa: BLE001
            rows.append({
                "penalty": P, "share_willing": share,
                "provider": prov, "plz": plz, "day": dd,
                "n_parcels": n_total,
                "vroom_cost_eur": 0.0, "vroom_n_routes": 0,
                "vroom_distance_km": 0.0,
                "vroom_status": f"ERR:{type(e).__name__}:{e}",
            })
    return rows


def main() -> None:
    smoke = os.environ.get("VROOM_VAL_SMOKE", "0") == "1"
    log(f"VROOM validation Path 2 — smoke={smoke}")
    log(f"Validation keys: {len(VALIDATION_KEYS)} (provider, P) combos at theta=1")
    for k in VALIDATION_KEYS:
        log(f"  P={k[0]:g} theta={k[1]:g} provider={k[2]}")

    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    pdata = chk["provider_data"]
    odata = chk4["optimization_data"]

    chosen = pd.read_csv(PATH2 / "tab_chosen_schedules.csv")
    chosen["plz"] = chosen.plz.astype(str).str.zfill(5)
    # Optional: use system-smoothed schedules if available
    sm_path = PATH2 / "_tab_chosen_with_system_smoothing.csv"
    if sm_path.exists():
        sm = pd.read_csv(sm_path); sm["plz"] = sm.plz.astype(str).str.zfill(5)
        sm = sm.rename(columns={"schedule_idx_system_smoothed":
                                 "schedule_idx_validate"})
        chosen = chosen.merge(
            sm[["penalty", "share_willing", "provider", "plz",
                 "schedule_idx_validate"]],
            on=["penalty", "share_willing", "provider", "plz"], how="left")
        chosen["schedule_idx_validate"] = (
            chosen.schedule_idx_validate.fillna(chosen.schedule_idx_balanced)
            .astype(int))
        log("Using system_smoothed schedules where available, balanced otherwise")
    else:
        chosen["schedule_idx_validate"] = chosen.schedule_idx_balanced
        log("Using balanced (per-hub) schedules")

    schedules = orch.enumerate_schedules(MAX_HOLD)
    sched_by_idx = {i: sorted(s) for i, s in enumerate(schedules)}

    done = load_done_keys()
    log(f"Already done: {len(done)} (provider, plz) cells")

    # Group keys by (P, share) so we build fs / cell only once each
    cells_to_provs: dict[tuple[float, float], list[str]] = {}
    for P, share, prov in VALIDATION_KEYS:
        cells_to_provs.setdefault((float(P), float(share)), []).append(prov)

    t0 = time.time()
    for (P, share), provs in cells_to_provs.items():
        log(f"\n=== Cell (P={P}, share={share}) — providers: {provs} ===")
        cell = chosen[(np.isclose(chosen.penalty, P)) &
                      (np.isclose(chosen.share_willing, share))]
        fs_b2c_v = orch.fs_b2c(share); fs_b2b_v = orch.fs_b2b(share)

        for prov in provs:
            if smoke and prov != "DHL":
                continue
            sub = cell[cell.provider == prov]
            if not len(sub):
                continue
            pkts = pdata.get(prov, {}); odat = odata.get(prov, {})
            if not pkts or not odat:
                continue
            plz_data = odat.get("plz_data", {})
            df_assign = pkts.get("df_assignments")
            prefix = provider_to_demand_prefix(prov)
            t_prov = time.time(); n_solved = 0
            plz_list = sub.plz.tolist()
            if smoke:
                plz_list = plz_list[:3]

            for plz in plz_list:
                key = (float(P), float(share), prov, plz)
                if key in done:
                    continue
                row = sub[sub.plz == plz].iloc[0]
                sched_idx = int(row.schedule_idx_validate)
                sched_days = sched_by_idx[sched_idx]
                pdat = plz_data.get(plz, {})
                if not pdat:
                    continue
                hub_row = df_assign[df_assign["plz"] == plz]
                if not len(hub_row):
                    continue
                hub_row = hub_row.iloc[0]
                rows = solve_plz_batched(
                    prov, plz, sched_days, pkts, pdat, hub_row, prefix,
                    P, share, fs_b2c_v, fs_b2b_v,
                )
                if rows:
                    append_rows(rows)
                    done.add(key)
                    n_solved += 1
                    if n_solved % 25 == 0:
                        log(f"  {prov} {n_solved}/{len(plz_list)} ... "
                            f"elapsed {(time.time()-t_prov)/60:.1f}min")
            log(f"  {prov} DONE: {n_solved} PLZ in {(time.time()-t_prov)/60:.1f}min")
    log(f"\nALL DONE in {(time.time()-t0)/60:.1f}min")
    log(f"Output: {CHECKPOINT}")


if __name__ == "__main__":
    main()
