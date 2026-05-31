"""Add P=0.4 as a first-class penalty to the balanced run (it was the geometric
Pareto knee but missing from the 8-point grid). Runs evaluate_cell at P=0.4 across
all shares — identical to the orchestrator — and appends the rows to:
  - BAL source: tab_chosen_schedules.csv, tab_balancing_summary.csv, tab_fleet_per_hub.csv
  - paper_final derived: tab_chosen_schedules_full.csv, tab_optimization_full_grid.csv
Idempotent: skips if P=0.4 already present.
"""
from __future__ import annotations
import importlib.util, pickle, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
BAL = ROOT / "results" / "overnight_2026_05_29_path2"
PF = ROOT / "results" / "paper_final_2026_05_30" / "05_optimization"
P_NEW = 0.4


def load_module():
    spec = importlib.util.spec_from_file_location("ob", ROOT / "scripts" / "overnight_orchestrator_balanced.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main():
    mod = load_module()
    chk = pickle.load(open(ROOT / "results/checkpoints/01_demand.pkl", "rb"))
    chk4 = pickle.load(open(ROOT / "results/checkpoints/04_optim_prep.pkl", "rb"))
    pdata = chk["provider_data"]; odata = chk4["optimization_data"]
    model = mod.load_model(); mlp = mod.build_ml_prep(pdata)
    sched = mod.enumerate_schedules()
    sw = np.array([mod.avg_wait_days(sorted(s)) for s in sched])

    shares = sorted(pd.read_csv(BAL / "tab_chosen_schedules.csv").share_willing.unique())
    chosen_rows, summ_rows, fleet_rows = [], [], []
    for share in shares:
        cc, cs, cf = mod.evaluate_cell(P_NEW, share, pdata, odata, mlp, model, sched, sw)
        chosen_rows += cc; summ_rows += cs; fleet_rows += cf
        print(f"  P=0.4 share={share:.1f}: {len(cc)} cells")
    new_chosen = pd.DataFrame(chosen_rows)
    new_summ = pd.DataFrame(summ_rows)
    new_fleet = pd.DataFrame(fleet_rows)

    # ── append to BAL source (idempotent) ───────────────────────────────
    for fname, newdf in [("tab_chosen_schedules.csv", new_chosen),
                          ("tab_balancing_summary.csv", new_summ),
                          ("tab_fleet_per_hub.csv", new_fleet)]:
        p = BAL / fname
        old = pd.read_csv(p)
        old = old[old.penalty != P_NEW]                 # drop any prior 0.4
        comb = pd.concat([old, newdf[old.columns]], ignore_index=True)
        comb.to_csv(p, index=False)
        print(f"  appended {len(newdf)} rows -> {fname} (total {len(comb)})")

    # ── append to paper_final tab_chosen_schedules_full.csv ──────────────
    full = pd.read_csv(PF / "tab_chosen_schedules_full.csv")
    full = full[full.penalty != P_NEW]
    full = pd.concat([full, new_chosen[full.columns]], ignore_index=True)
    full.to_csv(PF / "tab_chosen_schedules_full.csv", index=False)
    print(f"  tab_chosen_schedules_full.csv -> {len(full)} rows")

    # ── compute + append grid row (v2 formula) ───────────────────────────
    grid = pd.read_csv(PF / "tab_optimization_full_grid.csv")
    baseline = float(grid[grid.share_willing == 0.0].bal_cost_eur.max())
    g_summ = new_summ.groupby("share_willing", as_index=False).agg(
        init_cost_eur=("init_cost_eur", "sum"), bal_cost_eur=("balanced_cost_eur", "sum"),
        max_fleet_before=("max_fleet_before", "sum"), max_fleet_after=("max_fleet_after", "sum"))
    wait = (new_chosen.groupby("share_willing")
            .apply(lambda d: (d.avg_wait_d_balanced * d.weekly_parcels).sum() / d.weekly_parcels.sum())
            .rename("wait_bal").reset_index())
    g = g_summ.merge(wait, on="share_willing")
    g["penalty"] = P_NEW
    g["saving_bal_pct"] = 100 * (baseline - g.bal_cost_eur) / baseline
    g["fleet_red_pct"] = 100 * (g.max_fleet_before - g.max_fleet_after) / g.max_fleet_before.clip(lower=1)
    g["delta_pct"] = 100 * (g.bal_cost_eur - g.init_cost_eur) / g.init_cost_eur.clip(lower=1)
    g = g[["init_cost_eur", "bal_cost_eur", "penalty", "share_willing", "delta_pct",
           "fleet_red_pct", "saving_bal_pct", "wait_bal"]]
    grid = grid[grid.penalty != P_NEW]
    grid = pd.concat([grid, g], ignore_index=True).sort_values(["penalty", "share_willing"])
    grid.to_csv(PF / "tab_optimization_full_grid.csv", index=False)

    s1 = g[np.isclose(g.share_willing, 1.0)].iloc[0]
    print(f"\n  P=0.4 grid @ share=100%: saving={s1.saving_bal_pct:.2f}%  wait={s1.wait_bal:.3f}d")
    print(f"  baseline={baseline/1e3:.1f}k€  (sanity: should be ~1909.7)")
    print("Done — P=0.4 is now a first-class penalty.")


if __name__ == "__main__":
    main()
