"""Single-row schedule-mix figure across all eight service-penalty levels,
Stage-3 revision of scripts/figures/fig_SM_mix_8P.py.

Path fixes only -- the plotted quantity (`schedule_size_init`, the
cost-optimal per-cell delivery-day count before any balancing/smoothing)
is unchanged, because frequency-preserving balancing and system smoothing
leave the per-cell delivery-day count unchanged. This invariant is no
longer just asserted narratively: it is checked in code against Stage 3
(`_tab_chosen_with_system_smoothing.csv`, `schedule_size_system_smoothed`)
before any plotting happens.

Output: results/revision_2026_07/figures/fig4_SM_mix_pct_8P.{png,pdf}

Input/output root: ``C.OUT_DIR``, overridable with the ``REV_DIR``
environment variable (default ``results/revision_2026_07`` -- this script
reproduces the submitted revision figure when run with no environment set).
``scripts/revision/70_figs_tables_v2.py`` sets ``REV_DIR`` to the v5-schema
grid.  NOTE: this builder reads the 2026-07 STAGE-3 schema
(``tab_costs_smoothed.csv`` etc.); pointing ``REV_DIR`` at a v5-schema grid
gives it no inputs -- ``70_`` renders the v5 figures itself.

DEPRECATED (2026-08 revision): superseded by scripts/revision/61_grid_run_v2.py,
67_validate_vroom_v2.py, 70_figs_tables_v2.py and 73_tables_ops_v2.py.
"""
from __future__ import annotations
import sys

# --- DEPRECATED ENTRY POINT (2026-08 revision) -----------------------------
import warnings as _deprecation_warnings

_deprecation_warnings.warn(
    "32_fig4_mix.py is a STALE entry point: it recomputes totals WITHOUT the pool "
    "term and predates the universal tour rule, the two cost lenses and the "
    "operator polish. Its numbers are NOT comparable with the 2026-08 "
    "revision. Use scripts/revision/61_grid_run_v2.py for the grid, "
    "scripts/revision/67_validate_vroom_v2.py for VROOM validation, "
    "scripts/revision/70_figs_tables_v2.py for figures and tables, and "
    "scripts/revision/73_tables_ops_v2.py for the v2 ops/knee/value-of-"
    "stage-2 tables.",
    DeprecationWarning,
    stacklevel=2,
)
# ---------------------------------------------------------------------------

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _stage3_common as C  # noqa: E402

BAL = C.RUN_DIR
OUT = C.OUT_DIR / "figures"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})
FREQ_COLOR = {2: "#1d3557", 3: "#2a9d8f", 4: "#e9c46a",
              5: "#f4a261", 6: "#e76f51"}


def _stack(ax, sub, col):
    agg = sub.groupby(["share_willing", col]).size().reset_index(name="cnt")
    piv = (agg.pivot(index="share_willing", columns=col, values="cnt")
           .fillna(0).sort_index())
    piv = piv.div(piv.sum(axis=1), axis=0) * 100
    x = piv.index.values * 100
    bottom = np.zeros(len(piv))
    for sz in (2, 3, 4, 5, 6):
        if sz not in piv.columns:
            continue
        h = piv[sz].values
        ax.fill_between(x, bottom, bottom + h, color=FREQ_COLOR[sz],
                         alpha=0.92)
        bottom += h
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.2)


def main():
    sched = pd.read_csv(BAL / "tab_chosen_schedules.csv")
    sched["plz"] = sched.plz.astype(str)

    # ---- Stage-invariance assert: frequency preserved init -> system-smoothed ----
    sm = pd.read_csv(BAL / "_tab_chosen_with_system_smoothing.csv")
    sm["plz"] = sm.plz.astype(str)
    j = sched.merge(sm, on=["penalty", "share_willing", "provider", "plz"])
    same = j.schedule_size_init == j.schedule_size_system_smoothed
    # Whether the stages preserve a cell's delivery FREQUENCY is a
    # property of the GRID, not of this code: the 2026-07 pipeline
    # re-times days only, a v5/v6 stage 2 is frequency-free by design.
    # C.FREQ_INVARIANT declares which grid this is (default: the 2026-07
    # one, so the gate is unchanged); when it is not declared the
    # violation is REPORTED loudly instead of silently passing, because
    # the fig-4 caption may then not claim invariance.
    if C.FREQ_INVARIANT:
        assert same.all(), \
            "frequency not preserved across stages -- fig4 caption claim invalid"
        print(f"Stage-invariance OK: schedule_size_init == schedule_size_system_smoothed "
              f"for all {len(j)} (penalty, share_willing, provider, plz) rows")
    else:
        n_diff = int((~same).sum())
        print("Stage-invariance NOT DECLARED (REV_FREQ_INVARIANT=0): "
              f"{n_diff} of {len(j)} (penalty, share_willing, provider, "
              "plz) rows change delivery frequency between the plotted "
              "(init) plan and the final plan. This figure shows the "
              "init plan; its caption must NOT claim that the later "
              "stages preserve delivery frequency.")

    P_VALUES = sorted(sched.penalty.unique())
    P_VALUES = [p for p in P_VALUES if not np.isclose(p, 0.4)]
    n = len(P_VALUES)
    print(f"Rendering {n} penalty levels: {P_VALUES}")

    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(2.75 * ncols, 2.3 * nrows),
                              sharey=True, sharex=True)
    for idx, P in enumerate(P_VALUES):
        r_, c_ = divmod(idx, ncols)
        ax = axes[r_, c_]
        _stack(ax, sched[np.isclose(sched.penalty, P)],
                "schedule_size_init")
        ax.set_title(rf"Service penalty $P = {P:g}$ €/p/d",
                      fontsize=9.5)
    for idx in range(n, nrows * ncols):
        r_, c_ = divmod(idx, ncols)
        axes[r_, c_].set_visible(False)

    fig.tight_layout(rect=[0.05, 0.16, 1, 1], pad=0.4,
                      w_pad=0.3, h_pad=0.6)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bb_tl = axes[0, 0].get_position()
    bb_br = axes[-1, -1].get_position()
    cx = (bb_tl.x0 + bb_br.x1) / 2.0
    cy = (bb_br.y0 + bb_tl.y1) / 2.0
    tb = axes[-1, 0].get_tightbbox(r).transformed(inv)
    xlab_y = tb.y0 - 0.030
    fig.text(cx, xlab_y, r"Willingness-to-wait share $\theta$ [%]",
             ha="center", va="top", fontsize=11)
    lb = axes[0, 0].get_tightbbox(r).transformed(inv)
    fig.text(lb.x0 - 0.004, cy, "Share of postal-code areas [%]",
             rotation=90, ha="right", va="center", fontsize=10)
    handles = [Patch(facecolor=FREQ_COLOR[s], label=f"{s} day/wk")
               for s in (2, 3, 4, 5, 6)]
    fig.legend(handles=handles, title="Delivery days per week",
                loc="upper center", ncol=5,
                frameon=True, framealpha=0.9, edgecolor="0.8",
                fontsize=9, title_fontsize=9,
                bbox_to_anchor=(cx, xlab_y - 0.055),
                handlelength=1.4, columnspacing=1.3, borderpad=0.4)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig4_SM_mix_pct_8P.{ext}",
                     bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'fig4_SM_mix_pct_8P.png'}")
    print(f"saved {OUT/'fig4_SM_mix_pct_8P.pdf'}")


if __name__ == "__main__":
    main()
