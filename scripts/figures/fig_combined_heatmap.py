"""Six-panel combined heatmap over the $(P, \\theta)$ grid for the EWGT
paper. Top row holds the cost-side metrics (saving of the cost-optimal
selection, saving of the fleet-balanced pipeline output, and the
parcels-weighted mean customer waiting time). Bottom row holds the
fleet-side metrics (peak fleet reduction, weekly CV reduction, and total
weekly fleet reduction). All saving and reduction columns are reported
relative to the daily-delivery baseline at $\\theta = 0$; the total-fleet
panel uses a diverging colour map centred at zero because some operating
points raise the weekly vehicle stock instead of lowering it.

Same TRPro styling as fig_SM_mix_pct_8P (serif, dejavuserif mathtext,
size 11, ticks 9, line width and savefig.dpi=300).

Default output: results/EWGT_Results/fig_grid_heatmap_6.{png,pdf}

Task 19 W1b (v6 regeneration)
-----------------------------
v6 status A: every panel is directly derivable from the v6-native grid
tables (``tab_costs_v2.csv`` / ``tab_wait_v2.csv`` /
``tab_fleet_per_hub_v2.csv``) with column renames -- no
``74_v2_to_legacy_tables.py`` adapter needed.  ``--rev-dir`` selects the
data source by SCHEMA DETECTION: a directory carrying ``tab_costs_v2.csv``
at its top level is read natively (v6 two-plan columns
``cost_stage1_eur``/``cost_stage2_eur``, baseline computed from the grid's
OWN theta=0 rows -- never the stale ``BASE_TOTAL = 1,909,747.75 EUR``
2026-07/path2 constant, which prices a different, unheaded baseline); a
directory carrying ``tab_balancing_summary.csv`` instead falls back to the
original path2 columns and logic, UNCHANGED, so a pre-revision directory
(if one still existed) would render exactly as before. ``--rev-dir``'s
default is the script's original hardcoded path2 directory (unchanged);
the v6 regeneration run passes ``--rev-dir results/revision_2026_08_v6``
explicitly. ``--out-dir`` replaces the hardcoded ``OUT`` the same way.

Both plans shown: panels (a)/(d)/(e)/(f) top-row-cost and peak/CV/total
fleet compare the routing-optimal (stage 1) selection against the
operator-polished (stage 2) full-pipeline output; (c) wait is the
operator-polished (stage 2) plan only ("full pipeline output" per its own
title). Provenance footer and pinned PDF/PNG metadata follow
``scripts/figures/_v6_provenance.py``.
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts" / "revision"))
import _v6_provenance as V  # noqa: E402
import _figs_tables_v2 as H  # noqa: E402

DEFAULT_REV = ROOT / "results" / "overnight_2026_05_29_path2"
DEFAULT_OUT = ROOT / "results" / "EWGT_Results"
SCRIPT = "fig_combined_heatmap.py"
N_DAYS = 6

rcParams.update({
    "font.family": "serif", "font.size": 13,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 13, "axes.titlesize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

BASE_TOTAL_PATH2_STALE = 1909747.75  # 2026-07/path2 only -- NEVER use on v6


def heat(ax, mat, cmap, title, vmin=None, vmax=None, fmt="{:.1f}",
          norm=None, cbar_label="[%]", text_color=None,
          invert_thr=False):
    if norm is None:
        if vmin is None:
            vmin = float(np.nanmin(mat.values))
        if vmax is None:
            vmax = float(np.nanmax(mat.values))
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap,
                        vmin=vmin, vmax=vmax)
        thr = vmin + (vmax - vmin) * (0.55 if not invert_thr else 0.65)
        if text_color is not None:
            def color_for(v):
                return text_color
        elif invert_thr:
            # light-low, dark-high colormap (e.g. YlOrRd):
            # white text on dark high values, black text on light low values
            def color_for(v):
                return "white" if v > thr else "black"
        else:
            def color_for(v):
                return "white" if v < thr else "black"
    else:
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap, norm=norm)
        thr_lo = norm.vmin + (norm.vmax - norm.vmin) * 0.30
        thr_hi = norm.vmin + (norm.vmax - norm.vmin) * 0.70
        if text_color is not None:
            def color_for(v):
                return text_color
        else:
            def color_for(v):
                return "white" if (v < thr_lo or v > thr_hi) else "black"
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([f"{x*100:.0f}" for x in mat.columns], fontsize=10)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([f"{p:g}" for p in mat.index], fontsize=10)
    ax.set_title(title, fontsize=12.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color=color_for(v), fontsize=11)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=11)
    cb.ax.tick_params(labelsize=10)
    return im


def _fleet_reduction_panels(sys_day: pd.DataFrame, fleet_col: str) -> pd.DataFrame:
    """Peak/CV/total-fleet reduction per (P, theta) vs. the theta=0 profile
    of the SAME ``fleet_col``, Mon-Sat. Shared by both the path2 and the
    v6-native codepath -- only which column (and which rows) feed it
    differs."""
    base_day = (sys_day[np.isclose(sys_day.share_willing, 0.0)]
                .groupby("day")[fleet_col].mean())
    base = np.array([base_day.loc[d] for d in range(N_DAYS)])
    base_peak, base_total = float(base.max()), float(base.sum())
    base_cv = float(base.std() / base.mean())
    print(f"Baseline Mo-Sa: peak={base_peak:.0f} total={base_total:.0f} "
          f"cv={base_cv:.3f}")

    rows = []
    for (P, sh), g in sys_day.groupby(["penalty", "share_willing"]):
        g = g.sort_values("day")
        v = g[fleet_col].values
        peak = float(v.max()) if len(v) else float("nan")
        total = float(v.sum()) if len(v) else float("nan")
        cv = float(v.std() / v.mean()) if v.mean() > 0 else 0.0
        # A perfectly flat baseline (base_cv == 0) has no variance to
        # express a % reduction against -- same "no defined change" rule
        # already used above for `cv` itself, not a silent data fallback.
        cv_red = 100 * (base_cv - cv) / base_cv if base_cv > 0 else 0.0
        rows.append(dict(penalty=P, share_willing=sh,
                         peak_red=100 * (base_peak - peak) / base_peak,
                         cv_red=cv_red,
                         total_red=100 * (base_total - total) / base_total))
    return pd.DataFrame(rows)


def _load_path2(rev: Path) -> tuple[pd.DataFrame, ...]:
    """ORIGINAL (pre-revision) codepath, UNCHANGED: path2's own schema and
    the stale ``BASE_TOTAL_PATH2_STALE`` EUR constant -- kept only so a
    directory carrying the old files still renders exactly as before."""
    s = pd.read_csv(rev / "tab_balancing_summary.csv")
    s = s[~np.isclose(s.penalty, 0.4)].copy()
    f = pd.read_csv(rev / "tab_fleet_per_hub.csv")
    f = f[~np.isclose(f.penalty, 0.4)].copy()
    sched = pd.read_csv(rev / "tab_chosen_schedules.csv")
    sched["plz"] = sched.plz.astype(str)
    sched = sched[~np.isclose(sched.penalty, 0.4)].copy()

    ag = s.groupby(["penalty", "share_willing"], as_index=False).agg(
        init=("init_cost_eur", "sum"), bal=("balanced_cost_eur", "sum"))
    ag["init_sav"] = 100 * (BASE_TOTAL_PATH2_STALE - ag.init) / BASE_TOTAL_PATH2_STALE
    ag["bal_sav"] = 100 * (BASE_TOTAL_PATH2_STALE - ag.bal) / BASE_TOTAL_PATH2_STALE
    pivI = ag.pivot(index="penalty", columns="share_willing", values="init_sav")
    pivB = ag.pivot(index="penalty", columns="share_willing", values="bal_sav")

    sched["wait_x_par"] = sched.avg_wait_d_balanced * sched.weekly_parcels
    wg = sched.groupby(["penalty", "share_willing"], as_index=False).agg(
        num=("wait_x_par", "sum"), den=("weekly_parcels", "sum"))
    wg["avg_wait_d"] = wg.num / wg.den
    pivW = wg.pivot(index="penalty", columns="share_willing", values="avg_wait_d")

    sys_day = (f.groupby(["penalty", "share_willing", "day"], as_index=False)
               .agg(fa=("fleet_after", "sum")))
    cells = _fleet_reduction_panels(sys_day, "fa")
    pivPK = cells.pivot(index="penalty", columns="share_willing", values="peak_red")
    pivCV = cells.pivot(index="penalty", columns="share_willing", values="cv_red")
    pivTOT = cells.pivot(index="penalty", columns="share_willing", values="total_red")
    meta = dict(plan=V.PLAN_BOTH, source="tab_balancing_summary.csv + "
                "tab_fleet_per_hub.csv + tab_chosen_schedules.csv (path2)")
    return pivI, pivB, pivW, pivPK, pivCV, pivTOT, meta


def _load_v6_native(rev: Path) -> tuple[pd.DataFrame, ...]:
    """v6 status A: derived straight from the native grid tables, no 74_
    adapter. Baseline is THIS grid's own theta=0 row (never the stale
    path2/2026-07 constant -- 74_'s own docstring rule)."""
    costs = pd.read_csv(rev / "tab_costs_v2.csv")
    wait = pd.read_csv(rev / "tab_wait_v2.csv")
    fleet = pd.read_csv(rev / "tab_fleet_per_hub_v2.csv")
    V.require_columns(costs, ["penalty", "share_willing", "cost_stage1_eur",
                              "cost_stage2_eur"], source="tab_costs_v2.csv")
    V.require_columns(wait, ["penalty", "share_willing", "wait_num_willing",
                             "total_parcels"], source="tab_wait_v2.csv")
    V.require_columns(fleet, ["penalty", "share_willing", "day", "fleet"],
                      source="tab_fleet_per_hub_v2.csv")
    H.check_grid_integrity(costs, wait, label="fig_combined_heatmap v6-native")

    base = H.baseline(costs)  # base["routing_eur"]: this grid's OWN theta=0
    ag = costs.groupby(["penalty", "share_willing"], as_index=False).agg(
        init=("cost_stage1_eur", "sum"), bal=("cost_stage2_eur", "sum"))
    ag["init_sav"] = 100 * (base["routing_eur"] - ag.init) / base["routing_eur"]
    ag["bal_sav"] = 100 * (base["routing_eur"] - ag.bal) / base["routing_eur"]
    pivI = ag.pivot(index="penalty", columns="share_willing", values="init_sav")
    pivB = ag.pivot(index="penalty", columns="share_willing", values="bal_sav")

    # Operator-polished (stage 2 / balanced) plan's wait -- "full pipeline
    # output" per the panel's own title; wait_num_willing (no _stage1
    # suffix) IS the stage-2 numerator (_figs_tables_v2.py convention).
    wg = wait.groupby(["penalty", "share_willing"], as_index=False).agg(
        num=("wait_num_willing", "sum"), den=("total_parcels", "sum"))
    wg["avg_wait_d"] = wg.num / wg.den
    pivW = wg.pivot(index="penalty", columns="share_willing", values="avg_wait_d")

    # Fleet table is written at the FINAL (stage-2/balanced) plan only; at
    # theta=0 stage1==stage2 by construction (nothing is willing to move),
    # so the theta=0 slice of THIS SAME column is a valid "before" baseline
    # for every (P, theta) cell -- no 74_/75_ stage-1 refleeting needed.
    sys_day = (fleet.groupby(["penalty", "share_willing", "day"],
                            as_index=False).fleet.sum())
    cells = _fleet_reduction_panels(sys_day, "fleet")
    pivPK = cells.pivot(index="penalty", columns="share_willing", values="peak_red")
    pivCV = cells.pivot(index="penalty", columns="share_willing", values="cv_red")
    pivTOT = cells.pivot(index="penalty", columns="share_willing", values="total_red")
    meta = dict(plan=V.PLAN_BOTH, source="tab_costs_v2.csv + tab_wait_v2.csv "
                "+ tab_fleet_per_hub_v2.csv (v6-native)")
    return pivI, pivB, pivW, pivPK, pivCV, pivTOT, meta


def render(pivI, pivB, pivW, pivPK, pivCV, pivTOT, out_dir: Path,
          meta: dict) -> None:
    # ---- figure ----
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.0))

    # Top row
    v_cost_max = float(np.ceil(max(pivI.values.max(),
                                     pivB.values.max()) / 5) * 5)
    heat(axes[0, 0], pivI, "viridis",
          "(a) Cost saving, cost-optimal selection\n"
          "(before fleet balancing)",
          vmin=0, vmax=v_cost_max, cbar_label="Saving [%]")
    heat(axes[0, 1], pivB, "viridis",
          "(b) Cost saving, full pipeline\n"
          "(after fleet balancing and smoothing)",
          vmin=0, vmax=v_cost_max, cbar_label="Saving [%]")
    w_max = float(np.ceil(pivW.values.max() * 10) / 10)
    heat(axes[0, 2], pivW, "YlOrRd",
          "(c) Mean additional customer wait per parcel\n"
          "(full pipeline output)",
          vmin=0, vmax=w_max, fmt="{:.2f}", cbar_label="Wait [d]",
          invert_thr=True)

    # Bottom row
    pk_max = float(np.ceil(pivPK.values.max() / 5) * 5)
    pk_min = float(np.floor(min(0, pivPK.values.min()) / 5) * 5)
    heat(axes[1, 0], pivPK, "magma",
          "(d) Peak-fleet reduction\n"
          "(full pipeline output)",
          vmin=pk_min, vmax=pk_max, cbar_label="Reduction [%]")
    cv_max = float(np.ceil(pivCV.values.max() / 10) * 10)
    heat(axes[1, 1], pivCV, "magma",
          "(e) Mo--Sa coefficient of variation reduction\n"
          "(full pipeline output)",
          vmin=0, vmax=cv_max, cbar_label="Reduction [%]")
    # Total fleet: diverging on signed change (positive = more fleet,
    # negative = less fleet). Red = increase (bad), blue = reduction (good).
    pivCHG = -pivTOT  # flip sign so positive = increase
    tot_abs = max(abs(pivCHG.values.min()), abs(pivCHG.values.max()))
    tot_abs = float(np.ceil(tot_abs))
    norm_tot = TwoSlopeNorm(vmin=-tot_abs, vcenter=0.0, vmax=tot_abs)
    heat(axes[1, 2], pivCHG, "RdBu_r",
          "(f) Total weekly fleet change [%]\n"
          "(full pipeline output)",
          norm=norm_tot, cbar_label="Change [%]")

    # outer axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Willingness-to-wait share $\theta$ [%]",
                       fontsize=12.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Service penalty $P$ [€/p/d]", fontsize=12.5)

    fig.tight_layout(pad=0.6, w_pad=1.4, h_pad=1.8,
                      rect=[0, 0.05, 1, 1])
    fig.text(0.5, 0.03,
             "All cost savings, fleet reductions and the wait metric are "
             r"reported relative to the daily-delivery baseline at $\theta=0$.",
             ha="center", va="bottom", fontsize=11)
    V.footer(fig, plan=meta["plan"], script=SCRIPT, source=meta["source"],
             y=0.005)

    written = V.savefig_pinned(fig, out_dir, "fig_grid_heatmap_6")
    plt.close(fig)
    print(f"saved {written[0]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    V.add_v6_args(ap, default_rev=DEFAULT_REV, default_out=DEFAULT_OUT,
                 rev_help="grid directory: a v6-native dir (carries "
                          "tab_costs_v2.csv) or the original path2 dir "
                          "(carries tab_balancing_summary.csv)")
    args = ap.parse_args(argv)
    rev = Path(args.rev_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (rev / "tab_costs_v2.csv").exists():
        print(f"v6-native schema detected under {rev}")
        pivI, pivB, pivW, pivPK, pivCV, pivTOT, meta = _load_v6_native(rev)
    elif (rev / "tab_balancing_summary.csv").exists():
        print(f"path2 schema detected under {rev}")
        pivI, pivB, pivW, pivPK, pivCV, pivTOT, meta = _load_path2(rev)
    else:
        raise SystemExit(
            f"{rev}: neither tab_costs_v2.csv (v6-native) nor "
            "tab_balancing_summary.csv (path2) found -- refusing to guess")
    render(pivI, pivB, pivW, pivPK, pivCV, pivTOT, out_dir, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
