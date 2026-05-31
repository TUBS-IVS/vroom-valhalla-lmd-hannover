"""Diagnostic: how often does the non-delivery-day (non-willing) residual form a
standalone per-PLZ tour, and what is the bundling potential?

Compares two cost accountings for the express residual on non-delivery days:
  PER-PLZ (unbundled): sum of cost_3d over non-delivery days  ← what the
       decomposition/break-even scripts used (and what init_cost selection uses)
  HUB-BUNDLED (production): _hub_express_day_ml merges all non-delivering PLZ of a
       hub into ONE tour per (hub, day)  ← what bal_cost actually reports

Reports, across the willingness grid (at P=0 and P=0.5):
  - count of standalone per-PLZ residual tours + how many are small (<230, <100 pkts)
  - per-PLZ express total vs hub-bundled total  → bundling saving
  - resulting total-cost saving% (unbundled vs bundled) → how much break-even understated
"""
from __future__ import annotations
import importlib.util, pickle, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OPT = ROOT / "results" / "paper_final_2026_05_28" / "05_optimization"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "orch_bal", ROOT / "scripts" / "overnight_orchestrator_balanced.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    from batch_delivery.optimization.core import _hub_express_day_ml
    mod = load_module()
    N_DAYS = mod.N_DAYS
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    optim_data = chk4["optimization_data"]
    model = mod.load_model()
    ml_prep = mod.build_ml_prep(chk["provider_data"])
    schedules = mod.enumerate_schedules()
    daily_si = next(i for i, s in enumerate(schedules) if len(s) == N_DAYS)

    ch = pd.read_csv(OPT / "tab_chosen_schedules_full.csv")
    ch["plz"] = ch.plz.astype(str).str.zfill(5)
    SHARE_GRID = sorted(ch.share_willing.unique())

    rows = []
    for P in [0.0, 0.5]:
        for share in SHARE_GRID:
            fs_b2c_v, fs_b2b_v = mod.fs_b2c(share), mod.fs_b2b(share)
            tot_dd = perplz_expr = hub_expr = baseline = 0.0
            n_standalone = n_small230 = n_small100 = 0
            n_hub_days_bundled = n_hub_days_active = 0
            for prov in mod.PROVIDERS:
                if prov not in optim_data or prov not in ml_prep:
                    continue
                odata = optim_data[prov]; prep = ml_prep[prov]
                plz_keys = odata["plz_keys"]; hub_plz_list = odata["hub_plz_list"]
                mat = mod.build_cost_matrices_ml(
                    plz_keys, odata["plz_data"], schedules, model, prov,
                    prep["plz_day_coords"], prep["hub_coords_by_plz"],
                    fast_share_b2c=fs_b2c_v, fast_share_b2b=fs_b2b_v,
                )
                cost_3d = mat["cost_3d"]; sa = mat["sched_active"]
                dd_cost_mx = mat["dd_cost_mx"]; raw_express = mat["raw_express"]
                # chosen balanced schedule per cell
                sub = ch[(ch.penalty == P) & (ch.share_willing == share) & (ch.provider == prov)]
                idx_by = {r.plz: int(r.schedule_idx_balanced) for r in sub.itertuples()}
                chosen = np.array([idx_by.get(str(pc).zfill(5), daily_si) for pc in plz_keys])

                tot_dd += float(dd_cost_mx[np.arange(len(plz_keys)), chosen].sum())
                baseline += float(dd_cost_mx[:, daily_si].sum())
                # per-PLZ express (non-delivery days of chosen schedule)
                for pi in range(len(plz_keys)):
                    nd = ~sa[chosen[pi]]
                    for d in range(N_DAYS):
                        if nd[d] and raw_express[pi, d] > 0:
                            perplz_expr += float(cost_3d[pi, chosen[pi], d])
                            n_standalone += 1
                            if raw_express[pi, d] < 230: n_small230 += 1
                            if raw_express[pi, d] < 100: n_small100 += 1
                # hub-bundled express
                cache = {}
                for hi in range(len(hub_plz_list)):
                    for d in range(N_DAYS):
                        v = _hub_express_day_ml(hi, d, chosen, hub_plz_list, schedules,
                                                raw_express, mat["expr_stops"], mat, cache)
                        if v > 0:
                            hub_expr += v
                            n_hub_days_active += 1
                            h_ps = hub_plz_list[hi]
                            contrib = int(((~sa[chosen[h_ps], d]) & (raw_express[h_ps, d] > 0)).sum())
                            if contrib > 1:
                                n_hub_days_bundled += 1
            tot_unb = tot_dd + perplz_expr
            tot_bun = tot_dd + hub_expr
            rows.append({
                "P": P, "share": share,
                "n_standalone_residual_tours": n_standalone,
                "n_small_<230": n_small230, "n_small_<100": n_small100,
                "n_hubday_active": n_hub_days_active,
                "n_hubday_multiPLZ_bundled": n_hub_days_bundled,
                "perplz_express_k": perplz_expr / 1e3,
                "hub_bundled_express_k": hub_expr / 1e3,
                "bundling_saving_k": (perplz_expr - hub_expr) / 1e3,
                "saving_unbundled_pct": 100 * (baseline - tot_unb) / baseline,
                "saving_bundled_pct": 100 * (baseline - tot_bun) / baseline,
            })
        print(f"P={P} done")

    df = pd.DataFrame(rows)
    df.to_csv(OPT / "tab_bundling_potential.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    for P in [0.0, 0.5]:
        print(f"\n=== P={P} : per-PLZ (unbundled) vs hub-bundled express ===")
        sub = df[df.P == P]
        print(sub[["share", "n_standalone_residual_tours", "n_small_<230",
                   "n_hubday_multiPLZ_bundled", "perplz_express_k", "hub_bundled_express_k",
                   "bundling_saving_k", "saving_unbundled_pct", "saving_bundled_pct"]]
              .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
