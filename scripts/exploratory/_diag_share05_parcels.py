"""Intensive consistency audit of the VROOM validation harness vs the optimized
production scenario, BEFORE fixing the held-day express rounding and re-running.

Checks:
  A. Cell coverage (all 312 (prov,plz) per share present? any drops?).
  B. share=1.0 fidelity: VROOM-delivered weekly parcels == production weekly_parcels
     (no missing/extra parcels; schedule indexing + b2c/b2b columns correct).
  C. Regional total: sum of VROOM parcels vs sum of optimized weekly_parcels.
  D. share=0.5: VROOM vs ML-expected batched parcels (the known rounding gap),
     broken down by schedule size, to confirm rounding is the ONLY discrepancy.
  E. Any zero/negative/absurd parcel cells.
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import overnight_orchestrator_2026_05_27 as orch
from batch_delivery.io.demand import get_source_days

schedules = orch.enumerate_schedules(orch.MAX_HOLD)
sched_by_idx = {i: sorted(s) for i, s in enumerate(schedules)}

chosen_all = pd.read_csv(ROOT / "results/overnight_2026_05_27_balanced/tab_chosen_schedules.csv")
chosen_all["plz"] = chosen_all.plz.astype(str).str.zfill(5)

vr_all = pd.read_csv(ROOT / "results/paper_final_2026_05_28/07_validation/tab_vroom_balanced.csv")
vr_all["plz"] = vr_all.plz.astype(str).str.zfill(5)

chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
optim = chk4["optimization_data"]


def expected_batched(prov, plz, sched_idx, fs_b2c, fs_b2b):
    pd_ = optim[prov]["plz_data"].get(plz)
    if pd_ is None:
        return None
    sched = sched_by_idx[int(sched_idx)]; sset = set(sched)
    exp = 0.0
    for dd in sched:
        for d in get_source_days(dd, sched):
            b2c = pd_["b2c"].get(d, 0); b2b = pd_["b2b"].get(d, 0)
            if d in sset:
                exp += b2c + b2b
            else:
                exp += (b2c - round(b2c * fs_b2c)) + (b2b - round(b2b * fs_b2b))
    return exp


for SH in [1.0, 0.5]:
    fs_b2c = orch.fs_b2c(SH); fs_b2b = orch.fs_b2b(SH)
    chosen = chosen_all[(np.isclose(chosen_all.penalty, 0.4)) & (np.isclose(chosen_all.share_willing, SH))]
    vr = vr_all[(np.isclose(vr_all.penalty, 0.4)) & (np.isclose(vr_all.share_willing, SH))]
    vparc = vr.groupby(["provider", "plz"]).vroom_n_parcels.sum()
    done = set(vparc.index)
    print(f"\n########## share={SH}  (fs_b2c={fs_b2c:.3f}, fs_b2b={fs_b2b:.3f}) ##########")
    print(f"A. coverage: chosen cells={len(chosen)}  VROOM-done cells={len(done)}")

    rows = []
    for _, r in chosen.iterrows():
        key = (r.provider, r.plz)
        if key not in done:
            continue
        exp = expected_batched(r.provider, r.plz, r.schedule_idx_balanced, fs_b2c, fs_b2b)
        if exp is None:
            continue
        vp = float(vparc[key])
        rows.append({"provider": r.provider, "plz": r.plz,
                     "weekly_parcels": float(r.weekly_parcels),
                     "sched_size": len(sched_by_idx[int(r.schedule_idx_balanced)]),
                     "ml_expected": exp, "vroom": vp,
                     "ratio_vs_expected": vp / max(1.0, exp),
                     "ratio_vs_weekly": vp / max(1.0, float(r.weekly_parcels))})
    df = pd.DataFrame(rows)
    if SH == 1.0:
        print(f"B. share=1.0 fidelity — VROOM vs production weekly_parcels:")
        print(f"   mean ratio={df.ratio_vs_weekly.mean():.4f}  median={df.ratio_vs_weekly.median():.4f}  "
              f"min={df.ratio_vs_weekly.min():.3f}  max={df.ratio_vs_weekly.max():.3f}")
        off = df[(df.ratio_vs_weekly < 0.98) | (df.ratio_vs_weekly > 1.02)]
        print(f"   cells off by >2%: {len(off)}")
        if len(off):
            print(off[["provider","plz","sched_size","weekly_parcels","vroom","ratio_vs_weekly"]].head(15).to_string(index=False))
        print(f"C. regional totals: optimized weekly={df.weekly_parcels.sum():,.0f}  "
              f"VROOM delivered={df.vroom.sum():,.0f}  ratio={df.vroom.sum()/df.weekly_parcels.sum():.4f}")
    print(f"D. VROOM vs ML-expected batched ratio by schedule size:")
    for sz, g in df.groupby("sched_size"):
        print(f"   size={sz}  n={len(g):3d}  ratio={g.ratio_vs_expected.mean():.3f}")
    bad = df[(df.vroom <= 0) | (df.ml_expected <= 0)]
    print(f"E. zero/absurd cells: {len(bad)}")
