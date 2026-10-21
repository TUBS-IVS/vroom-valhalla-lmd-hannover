"""Schedule-size distribution across (P, share willing) for BOTH the
cost-optimal (init, no fleet sweep) and the fleet-balanced (post hub
optimization) output, stacked in one figure so the fleet-balancing shift
(pure-cost 2d/wk -> flatter 3-4d/wk) is directly visible.

Default output: results/paper_final_2026_05_28/05_optimization/
    fig_SM_init_vs_balanced.{png,pdf}

Task 19 W1b (v6 regeneration)
-----------------------------
v6 status B: ``schedule_size_init``/``schedule_size_balanced`` are exactly
``scripts/revision/74_v2_to_legacy_tables.py``'s ``tab_chosen_schedules.csv``
columns (``init`` = routing-optimal / stage 1, ``balanced`` =
operator-polished / stage 2 -- FREQUENCY-FREE at theta>0, Kompendium
§40.14, so this is two independent distributions, matching what this
script already plots). The per-panel cell count (``ax.set_ylim``) is read
from the data instead of the pre-revision hardcoded 312 -- it happens to
still be 312 on v6 (7 providers x their own PLZ set), but deriving it keeps
the axis correct if the cell universe ever changes again.
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
import _v6_provenance as V  # noqa: E402

SCRIPT = "_fig_schedule_mix_init_vs_balanced.py"
DEFAULT_REV = ROOT / "results" / "overnight_2026_05_27_balanced"
DEFAULT_OUT = ROOT / "results" / "paper_final_2026_05_28" / "05_optimization"

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
SCHED_COLOR = {2: "#9b2226", 3: "#bb3e03", 4: "#ee9b00", 5: "#94d2bd", 6: "#005f73"}
NCOL = 4
EXCLUDE_P = {0.4}   # leave out the fine-grid sweet-spot P for now


def _panel(ax, sub, col, P, n_cells, show_ylabel, show_xlabel):
    mix = sub.groupby(["share_willing", col]).size().unstack(fill_value=0)
    shares = mix.index.values * 100
    bottom = np.zeros(len(shares))
    for sz in sorted(mix.columns):
        ax.fill_between(shares, bottom, bottom + mix[sz].values,
                        color=SCHED_COLOR.get(int(sz), "gray"), alpha=0.88)
        bottom += mix[sz].values
    ax.set_title(f"P = {P:g} €/p/d", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, n_cells)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("Cells" if show_ylabel else "")
    ax.set_xlabel("Share willing [%]" if show_xlabel else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    V.add_v6_args(ap, default_rev=DEFAULT_REV, default_out=DEFAULT_OUT,
                 rev_help="directory holding tab_chosen_schedules.csv -- "
                          "the legacy-adapted 74_ <out>/run/ dir for v6")
    args = ap.parse_args(argv)
    rev = Path(args.rev_dir)
    OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    sched_path = rev / "tab_chosen_schedules.csv"
    sched = pd.read_csv(sched_path)
    V.require_columns(sched, ["penalty", "share_willing",
                              "schedule_size_init", "schedule_size_balanced"],
                      source=str(sched_path))
    P_VALUES = [p for p in sorted(sched.penalty.unique())
                if not any(np.isclose(p, e) for e in EXCLUDE_P)]
    nP = len(P_VALUES)
    nrow = int(np.ceil(nP / NCOL))
    # Total (provider, plz) cells at one (P, theta) point -- data-driven,
    # not the pre-revision hardcoded 312 (Task 19 W1b).
    n_cells = int(sched[np.isclose(sched.penalty, P_VALUES[0])
                       & np.isclose(sched.share_willing, 0.0)].shape[0])

    fig = plt.figure(figsize=(20, 16))
    subfigs = fig.subfigures(2, 1, hspace=0.07)
    blocks = [("schedule_size_init", "Cost-optimal (init, no fleet sweep)"),
              ("schedule_size_balanced", "Fleet-balanced (after hub optimization)")]

    for sf, (col, title) in zip(subfigs, blocks):
        axes = sf.subplots(nrow, NCOL)
        axes = np.atleast_2d(axes)
        for i, P in enumerate(P_VALUES):
            r, c = i // NCOL, i % NCOL
            _panel(axes[r, c], sched[sched.penalty == P], col, P, n_cells,
                  show_ylabel=(c == 0), show_xlabel=(r == nrow - 1))
        for j in range(nP, nrow * NCOL):           # hide unused axes
            axes[j // NCOL, j % NCOL].axis("off")
        sf.suptitle(title, fontsize=13, fontweight="bold")

    handles = [Patch(color=SCHED_COLOR[s], label=f"{s}d/wk", alpha=0.88)
               for s in (2, 3, 4, 5, 6)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=11,
               bbox_to_anchor=(0.5, 0.005), frameon=False)
    V.footer(fig, plan=V.PLAN_BOTH, script=SCRIPT,
             source="tab_chosen_schedules.csv", y=-0.035)
    written = V.savefig_pinned(fig, OUT, "fig_SM_init_vs_balanced")
    plt.close(fig)
    print(f"saved {written[0]}  (P={[f'{p:g}' for p in P_VALUES]}, "
          f"n_cells={n_cells})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
