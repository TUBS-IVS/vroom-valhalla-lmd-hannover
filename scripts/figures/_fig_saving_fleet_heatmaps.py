"""Three compact heatmaps over the (P, theta) grid for the paper:
  (a) cost saving [%] of the cost-optimal (init) plan vs daily baseline,
  (b) cost saving [%] of the fleet-balanced plan,
  (c) peak-fleet reduction [%] achieved by fleet balancing.

Same agg logic as paper_final_v2.load_data (reads tab_balancing_summary.csv +
tab_chosen_schedules.csv). Paper style (serif / dejavuserif mathtext), sized to
sit on the text width of a Transportation Research Procedia single column.

Default output: results/paper_final_2026_05_28/05_optimization/
    fig_saving_fleet_heatmaps.{png,pdf}

Task 19 W1b (v6 regeneration)
-----------------------------
v6 status B for panels (a)/(b): ``init``/``balanced``-suffixed columns are
exactly ``scripts/revision/74_v2_to_legacy_tables.py``'s
``tab_balancing_summary.csv`` schema (routing-optimal stage 1 / operator-
polished stage 2). Panel (c) is DROPPED on v6: ``max_fleet_before`` is a
74_ ``NO_SOURCE`` column (all-NaN) -- the v5/v6 fleet table is written at
the FINAL plan only, so there is no per-hub-day stage-1 fleet to take a
peak over without the partition-aware stage-1 refleeting
``scripts/revision/75_fig_fleet_week_classes.py`` implements, which is a
C-status port out of this wave's scope (see
``_fig_fleet_impact_vs_baseline.py``, same root cause). Baseline is this
grid's own theta=0 row -- read fresh from the SAME table, not a separate
constant.
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
import _v6_provenance as V  # noqa: E402

SCRIPT = "_fig_saving_fleet_heatmaps.py"
DEFAULT_REV = ROOT / "results" / "overnight_2026_05_29_path2"
DEFAULT_OUT = ROOT / "results" / "paper_final_2026_05_30" / "05_optimization"

rcParams.update({
    "font.family": "serif", "font.size": 9,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 5, "ytick.labelsize": 5,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})


def load_agg(rev: Path):
    s_path = rev / "tab_balancing_summary.csv"
    s = pd.read_csv(s_path)
    V.require_columns(s, ["penalty", "share_willing", "init_cost_eur",
                          "balanced_cost_eur", "max_fleet_before",
                          "max_fleet_after"], source=str(s_path))
    have_fleet_before = bool(s.max_fleet_before.notna().any())

    aggs = dict(init_cost_eur=("init_cost_eur", "sum"),
               bal_cost_eur=("balanced_cost_eur", "sum"))
    if have_fleet_before:
        aggs["max_fleet_before"] = ("max_fleet_before", "sum")
        aggs["max_fleet_after"] = ("max_fleet_after", "sum")
    agg = s.groupby(["penalty", "share_willing"], as_index=False).agg(**aggs)
    baseline = float(agg[agg.share_willing == 0.0].bal_cost_eur.max())
    agg["saving_init_pct"] = 100 * (baseline - agg.init_cost_eur) / baseline
    agg["saving_bal_pct"] = 100 * (baseline - agg.bal_cost_eur) / baseline
    if have_fleet_before:
        agg["fleet_red_pct"] = (
            100 * (agg.max_fleet_before - agg.max_fleet_after)
            / agg.max_fleet_before.clip(lower=1))
    else:
        print(f"{s_path}: max_fleet_before is entirely NaN (74_ NO_SOURCE "
              "-- no v5/v6 stage-1 per-hub-day fleet); dropping panel (c) "
              "peak-fleet reduction, keeping (a)/(b) cost-saving only")
    return agg, baseline, have_fleet_before


def _heat(ax, mat, cmap, vmin, vmax, title):
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([f"{x*100:.0f}" for x in mat.columns])
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([f"{p:g}" for p in mat.index])
    ax.set_xlabel(r"Willingness-to-wait share $\theta$ [%]")
    ax.set_title(title, fontsize=7.5)
    thr = vmin + (vmax - vmin) * 0.55
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="white" if v < thr else "black", fontsize=4.0)
    return im


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    V.add_v6_args(ap, default_rev=DEFAULT_REV, default_out=DEFAULT_OUT,
                 rev_help="legacy-adapted run/ directory (74_ <out>/run) "
                          "for v6, or the original path2 dir")
    args = ap.parse_args(argv)
    rev = Path(args.rev_dir)
    OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    agg, baseline, have_fleet = load_agg(rev)
    agg = agg[~np.isclose(agg.penalty, 0.4)]   # 0.4 is the fine-sweep sweet spot, shown separately
    sav_init = agg.pivot(index="penalty", columns="share_willing", values="saving_init_pct")
    sav_bal = agg.pivot(index="penalty", columns="share_willing", values="saving_bal_pct")

    s_vmin = min(sav_init.values.min(), sav_bal.values.min())
    s_vmax = max(sav_init.values.max(), sav_bal.values.max())

    ncols = 3 if have_fleet else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7.4 if have_fleet else 5.2, 2.5))
    im_a = _heat(axes[0], sav_init, "viridis", s_vmin, s_vmax, "(a) Cost saving [%], cost-optimal")
    _heat(axes[1], sav_bal, "viridis", s_vmin, s_vmax, "(b) Cost saving [%], fleet-balanced")
    axes[0].set_ylabel(r"Service penalty $P$ [€/p/d]")
    cb1 = fig.colorbar(im_a, ax=axes[1], fraction=0.046, pad=0.03)
    cb1.set_label("Cost saving [%]", fontsize=8)
    cb1.ax.tick_params(labelsize=7)

    fleet_max_str = ""
    if have_fleet:
        fleet = agg.pivot(index="penalty", columns="share_willing",
                          values="fleet_red_pct")
        f_vmax = float(np.ceil(fleet.values.max() / 5) * 5)
        im_c = _heat(axes[2], fleet, "magma", 0.0, f_vmax,
                    "(c) Peak-fleet reduction [%]")
        cb2 = fig.colorbar(im_c, ax=axes[2], fraction=0.046, pad=0.03)
        cb2.set_label("Peak-fleet reduction [%]", fontsize=8)
        cb2.ax.tick_params(labelsize=7)
        fleet_max_str = (f"; fleet-red max {fleet.values.max():.1f}% at "
                         f"P={fleet.max(axis=1).idxmax():g}")

    fig.tight_layout(w_pad=0.6, rect=[0, 0.08, 1, 1])
    stems = "tab_balancing_summary.csv" + (
        "" if have_fleet else " (panel c dropped -- max_fleet_before "
        "NO_SOURCE on v5/v6)")
    V.footer(fig, plan=V.PLAN_BOTH, script=SCRIPT, source=stems, y=0.0)
    written = V.savefig_pinned(fig, OUT, "fig_saving_fleet_heatmaps")
    plt.close(fig)
    print(f"saved {written[0]} (baseline={baseline/1e3:.0f} k€/wk; "
          f"saving range [{s_vmin:.1f},{s_vmax:.1f}]%{fleet_max_str})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
