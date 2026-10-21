"""Regenerate fig_PF1 (Pareto + 0.4 sweet-spot star), fig_PF2 (heatmap), and
re-derive tab_optimization_full_grid.csv + tab_chosen_schedules_full.csv from the
BAL source (which now includes P=0.4) — by calling v2.fig_optimization directly.
Does NOT run v2.main(), so the manually-curated README.md is left untouched.

DEPRECATED (2026-08 revision). Stale entry point: it recomputes totals
WITHOUT the pool term and predates the universal tour rule, the two cost
lenses and the operator polish, so its numbers are not comparable with the
current results. Use scripts/revision/61_grid_run_v2.py for the grid and
scripts/revision/70_figs_tables_v2.py for figures and tables.
"""
from __future__ import annotations
import importlib.util, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# --- DEPRECATED ENTRY POINT (2026-08 revision) -----------------------------
import warnings as _deprecation_warnings

_deprecation_warnings.warn(
    "paper_final_regen_grid_pf.py is a STALE entry point: it recomputes totals WITHOUT the pool "
    "term and predates the universal tour rule, the two cost lenses and the "
    "operator polish. Its numbers are NOT comparable with the 2026-08 "
    "revision. Use scripts/revision/61_grid_run_v2.py for the grid and "
    "scripts/revision/70_figs_tables_v2.py for figures and tables.",
    DeprecationWarning,
    stacklevel=2,
)
# ---------------------------------------------------------------------------

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))


def main():
    spec = importlib.util.spec_from_file_location("pfv2", ROOT / "scripts" / "paper_final_v2.py")
    v2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v2)
    s, sched, fleet, agg, baseline = v2.load_data()
    print(f"  penalties in agg: {sorted(agg.penalty.unique())}")
    v2.fig_optimization(agg, sched, baseline)
    op = agg[(agg.penalty == 0.4) & (agg.share_willing == 1.0)]
    if len(op):
        r = op.iloc[0]
        print(f"  P=0.4 share=100%: saving_bal={r.saving_bal_pct:.2f}%  wait_bal={r.wait_bal:.3f}d")
    print("Done — PF1/PF2 + grid + chosen_schedules_full regenerated with P=0.4.")


if __name__ == "__main__":
    main()
