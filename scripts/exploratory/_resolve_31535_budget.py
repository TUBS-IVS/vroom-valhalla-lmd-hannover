"""One-off: re-solve the 3 BUDGET_EXCEEDED days (share=0.5, DHL, 31535, days
3/4/5) with a much larger per-job budget, then patch tab_vroom_balanced.csv so
the out-of-sample validation set is fully solved (consistent dataset).

31535 is a ~357 km2 rural DHL cell (~880 stops). At the default 1200 s per-job
budget three of its six daily solves ran out of time; days 0-2 succeeded. We
keep the real 6-day schedule (so the source-day logic is unchanged), re-solve
the whole cell with a 3 h budget / 1 h HTTP timeout (days 0-2 hit cache), and
patch only days 3/4/5 back into the checkpoint.
"""
from __future__ import annotations
import shutil
import sys
import time
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Generous budget for the one big rural instance (patched BEFORE importing the
# runner so its solve calls read the larger values).
import batch_delivery.routing.core as rcore  # noqa: E402
rcore.PER_JOB_BUDGET_S = 10800   # 3 h wall-clock cap (incl. retries)
rcore.HTTP_TIMEOUT = 3600        # 1 h per single VROOM POST

import paper_vroom_validate_balanced as V  # noqa: E402
from batch_delivery.config.constants import provider_to_demand_prefix  # noqa: E402

PENALTY, SHARE, PROV, PLZ = 0.4, 0.5, "DHL", "31535"
TARGET_DAYS = {3, 4, 5}
CSV = V.CHECKPOINT


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    log(f"re-solving {PROV}/{PLZ} @ P={PENALTY}, share={SHARE}, "
        f"days={sorted(TARGET_DAYS)} (budget={rcore.PER_JOB_BUDGET_S}s)")
    chosen = pd.read_csv(V.BAL / "tab_chosen_schedules.csv")
    chosen["plz"] = chosen.plz.astype(str).str.zfill(5)
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    provider_data = chk["provider_data"]
    optim_data = chk4["optimization_data"]
    schedules = V.orch.enumerate_schedules(V.MAX_HOLD)
    sched_by_idx = {i: sorted(s) for i, s in enumerate(schedules)}

    fs_b2c_v = V.orch.fs_b2c(SHARE)
    fs_b2b_v = V.orch.fs_b2b(SHARE)
    r = chosen[(np.isclose(chosen.penalty, PENALTY)) &
               (np.isclose(chosen.share_willing, SHARE)) &
               (chosen.provider == PROV) & (chosen.plz == PLZ)].iloc[0]
    sched_days = sched_by_idx[int(r.schedule_idx_balanced)]
    log(f"schedule_days={sched_days} (size={len(sched_days)})")

    pdata = provider_data[PROV]
    plz_data_prov = optim_data[PROV]["plz_data"]
    hub_row = pdata["df_assignments"]
    hub_row = hub_row[hub_row["plz"] == PLZ].iloc[0]
    prefix = provider_to_demand_prefix(PROV)

    t0 = time.time()
    rows = V.solve_plz_batched(PROV, PLZ, sched_days, pdata, plz_data_prov[PLZ],
                               hub_row, prefix, PENALTY, SHARE, fs_b2c_v, fs_b2b_v)
    log(f"cell solved in {time.time() - t0:.0f}s")
    for row in sorted(rows, key=lambda x: x["day"]):
        log(f"  day {row['day']}: {row['vroom_status']:>14}  "
            f"cost={row['vroom_cost_eur']:.2f}  routes={row['vroom_n_routes']}  "
            f"parcels={row['vroom_n_parcels']}")

    new = {row["day"]: row for row in rows if row["day"] in TARGET_DAYS}
    bad = [d for d, row in new.items()
           if row["vroom_status"] not in ("OK", "CACHED")]
    if bad:
        log(f"WARNING: target days still not solved: {bad} -- NOT patching CSV")
        return

    df = pd.read_csv(CSV)
    backup = CSV.with_name(CSV.stem + "_prebudgetfix.csv")
    shutil.copy(CSV, backup)
    log(f"backup -> {backup.name}")
    base = ((np.isclose(df.penalty, PENALTY)) &
            (np.isclose(df.share_willing, SHARE)) &
            (df.provider == PROV) &
            (df.plz.astype(str).str.zfill(5) == PLZ))
    for d, row in new.items():
        m = base & (df.day == d)
        assert m.sum() == 1, f"expected 1 row for day {d}, got {m.sum()}"
        for col in ("vroom_cost_eur", "vroom_n_routes", "vroom_distance_km",
                    "vroom_n_parcels", "vroom_status"):
            df.loc[m, col] = row[col]
    df.to_csv(CSV, index=False)
    n_bad = int((df.vroom_status != "OK").sum())
    log(f"patched {len(new)} rows; remaining non-OK rows in file: {n_bad}")

    log("re-running assemble() for consistent diagnostics + figure ...")
    V.assemble()


if __name__ == "__main__":
    main()
