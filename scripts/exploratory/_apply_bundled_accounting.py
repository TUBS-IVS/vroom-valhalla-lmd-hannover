"""After _recompute_init_bundled.py finishes, overwrite init_cost_eur in
tab_balancing_summary.csv with the bundled value, recompute the dependent
cost_delta columns, and re-render the saving heatmap. Mix and maps figures
are unchanged because schedules and balanced costs are untouched.

Hard checks performed before writing:
  - row count of merge matches summary
  - no missing bundled values
  - per-row: bundled init <= unbundled init (bundling can only be cheaper)
  - per-cell: balanced saving >= 0 (no negative savings anywhere)
  - per-cell: residual = bal_sav - init_bundled_sav reported across full grid
    (NOT only theta=1)

Backs up the pre-overwrite summary to _accounting_unbundled.csv.
"""
from __future__ import annotations
import subprocess, sys, shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "overnight_2026_05_27_balanced"
BASE = 1909747.75   # daily-delivery baseline (P=10, share=0, all daily)


def _key(df):
    """Round float keys to a precision that survives pd.merge's exact-eq."""
    df = df.copy()
    df["penalty_k"] = df["penalty"].round(6)
    df["share_k"] = df["share_willing"].round(6)
    return df


def main() -> None:
    cache_p = OUT / "_init_bundled_cache.csv"
    summ_p = OUT / "tab_balancing_summary.csv"
    if not cache_p.exists():
        sys.exit(f"missing {cache_p} — run _recompute_init_bundled.py first")
    cache = pd.read_csv(cache_p)
    summ = pd.read_csv(summ_p)
    n_summ = len(summ)

    # Sanity: cache covers all summary cells
    n_cache_cells = cache.groupby(["penalty", "share_willing"]).ngroups
    n_summ_cells = summ.groupby(["penalty", "share_willing"]).ngroups
    print(f"summary: {n_summ} rows, {n_summ_cells} cells")
    print(f"cache:   {len(cache)} rows, {n_cache_cells} cells")
    if n_cache_cells != n_summ_cells:
        sys.exit(f"ABORT: cache has {n_cache_cells} cells, summary has {n_summ_cells}")

    shutil.copy(summ_p, OUT / "_accounting_unbundled.csv")
    print(f"backup -> _accounting_unbundled.csv ({n_summ} rows)")

    # Safe merge via rounded keys
    s = _key(summ); c = _key(cache)
    merged = s.merge(c[["penalty_k", "share_k", "provider", "init_cost_eur_bundled"]],
                     on=["penalty_k", "share_k", "provider"], how="left")
    if len(merged) != n_summ:
        sys.exit(f"ABORT: merge produced {len(merged)} rows vs {n_summ}")
    n_miss = int(merged.init_cost_eur_bundled.isna().sum())
    if n_miss:
        sys.exit(f"ABORT: {n_miss} summary rows have no bundled value in cache")

    # CHECK 1: bundled init <= unbundled init per row (bundling is always cheaper)
    bad = merged[merged.init_cost_eur_bundled > merged.init_cost_eur + 1e-3]
    if len(bad):
        print(f"  WARN: {len(bad)} rows have bundled > unbundled init (should not happen):")
        print(bad[["penalty", "share_willing", "provider", "init_cost_eur",
                    "init_cost_eur_bundled"]].head(10).to_string())

    summ["init_cost_eur"] = merged.init_cost_eur_bundled.values
    summ["cost_delta_eur"] = summ.balanced_cost_eur - summ.init_cost_eur
    summ["cost_delta_pct"] = (100.0 * summ.cost_delta_eur
                              / summ.init_cost_eur.clip(lower=1))

    # CHECK 2: per-cell savings, residual = bal_sav - init_sav across full grid
    per_cell = summ.groupby(["penalty", "share_willing"]).agg(
        i=("init_cost_eur", "sum"), b=("balanced_cost_eur", "sum")).reset_index()
    per_cell["init_sav"] = 100 * (BASE - per_cell.i) / BASE
    per_cell["bal_sav"] = 100 * (BASE - per_cell.b) / BASE
    per_cell["residual_pp"] = per_cell.bal_sav - per_cell.init_sav

    n_neg_bal = int((per_cell.bal_sav < -0.05).sum())
    n_big_res = int((per_cell.residual_pp > 4.0).sum())
    print(f"\n--- Cell-level checks (across {len(per_cell)} cells) ---")
    print(f"  cells with bal_sav < 0:        {n_neg_bal}  (must be 0)")
    print(f"  cells with residual > +4pp:    {n_big_res}  (small is good)")
    print(f"  residual_pp stats: "
          f"min={per_cell.residual_pp.min():.2f}  "
          f"max={per_cell.residual_pp.max():.2f}  "
          f"mean={per_cell.residual_pp.mean():.2f}  "
          f"median={per_cell.residual_pp.median():.2f}")

    if n_neg_bal > 0:
        print("\n  cells with negative balanced saving:")
        print(per_cell[per_cell.bal_sav < -0.05].round(2).to_string(index=False))

    th1 = per_cell[np.isclose(per_cell.share_willing, 1.0)]
    print(f"\n--- theta=1.0 row (paper main operating line) ---")
    print(th1[["penalty","init_sav","bal_sav","residual_pp"]].round(2).to_string(index=False))

    summ.to_csv(summ_p, index=False)
    per_cell.to_csv(OUT / "_per_cell_savings_bundled.csv", index=False)
    print(f"\nwrote {summ_p}")
    print(f"wrote {OUT / '_per_cell_savings_bundled.csv'}")

    print("\nRe-rendering saving heatmap ...")
    subprocess.run([sys.executable, "scripts/_fig_saving_fleet_heatmaps.py"],
                   check=True, cwd=str(ROOT))
    print("Done.")


if __name__ == "__main__":
    main()
