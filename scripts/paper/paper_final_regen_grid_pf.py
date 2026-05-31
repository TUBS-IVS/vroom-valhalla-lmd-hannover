"""Regenerate fig_PF1 (Pareto + 0.4 sweet-spot star), fig_PF2 (heatmap), and
re-derive tab_optimization_full_grid.csv + tab_chosen_schedules_full.csv from the
BAL source (which now includes P=0.4) — by calling v2.fig_optimization directly.
Does NOT run v2.main(), so the manually-curated README.md is left untouched.
"""
from __future__ import annotations
import importlib.util, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
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
