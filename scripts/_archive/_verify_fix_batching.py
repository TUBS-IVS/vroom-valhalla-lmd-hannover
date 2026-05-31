"""Verify the aggregate-rounding fix end-to-end on a few BATCHING share=0.5 cells.

Routes selected DPD/Amazon share=0.5 cells (schedule size < 6 = has held days)
through the FIXED solve_plz_batched and compares VROOM weekly cost to the ML
dd_cost_balanced. Expectation: the ~12% over-prediction collapses to ~2-3%.

Does NOT write to the main checkpoint.
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import paper_vroom_validate_balanced as V
import overnight_orchestrator_2026_05_27 as orch
from batch_delivery.config.constants import provider_to_demand_prefix

SH, P = 0.5, 0.4
fs_b2c, fs_b2b = orch.fs_b2c(SH), orch.fs_b2b(SH)
schedules = orch.enumerate_schedules(orch.MAX_HOLD)
sched_by_idx = {i: sorted(s) for i, s in enumerate(schedules)}

chosen = pd.read_csv(ROOT / "results/overnight_2026_05_27_balanced/tab_chosen_schedules.csv")
chosen["plz"] = chosen.plz.astype(str).str.zfill(5)
chosen = chosen[(np.isclose(chosen.penalty, P)) & (np.isclose(chosen.share_willing, SH))]

chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
provider_data, optim = chk["provider_data"], chk4["optimization_data"]

# Pick a few batching cells (size 3-4) from DPD and Amazon
picks = []
for prov in ["DPD", "Amazon"]:
    sub = chosen[(chosen.provider == prov) & (chosen.schedule_size_balanced.isin([3, 4]))]
    picks += list(sub.head(3).itertuples())

print(f"Verifying {len(picks)} batching cells (FIXED harness) ...\n")
print(f"{'prov':7s} {'plz':6s} sz  {'ML_dd':>9s} {'VROOM':>9s}  {'err%':>7s}")
errs = []
for r in picks:
    prov, plz = r.provider, str(r.plz).zfill(5)
    pdata, odata = provider_data[prov], optim[prov]
    if plz not in odata["plz_data"]:
        continue
    hub_df = pdata["df_assignments"][pdata["df_assignments"]["plz"] == plz]
    if hub_df.empty:
        continue
    sched_days = sched_by_idx[int(r.schedule_idx_balanced)]
    rows = V.solve_plz_batched(prov, plz, sched_days, pdata, odata["plz_data"][plz],
                               hub_df.iloc[0], provider_to_demand_prefix(prov),
                               P, SH, fs_b2c, fs_b2b)
    vcost = sum(x["vroom_cost_eur"] for x in rows)
    ml = float(r.dd_cost_balanced)
    err = 100.0 * (ml - vcost) / max(1.0, vcost)
    errs.append(err)
    print(f"{prov:7s} {plz:6s} {len(sched_days):2d}  {ml:9.1f} {vcost:9.1f}  {err:+7.2f}")

if errs:
    print(f"\nmean |err| = {np.mean(np.abs(errs)):.2f}%   mean bias = {np.mean(errs):+.2f}%")
    print("(was ~+12% before fix; expect ~2-3% now if fix works)")
