"""Render the sweet-spot 2-panel and saving 3-panel figures in WIDER aspect
ratios for the EWGT paper, flat into results/EWGT_Results/ by default.

Task 19 W1b (v6 regeneration)
-----------------------------
v6 status B: same two source tables and the same two corrections as
``_fig_sweetspot_path2.py`` and ``_fig_saving_fleet_heatmaps.py`` (both are
also in this wave's B set):

* the baseline is this grid's OWN theta=0 total (74_'s
  ``legacy_manifest.json``), never the stale ``BASE = 1,909,747.75`` EUR
  2026-07/path2 constant;
* the sweet-spot ``P`` is the chord-distance knee of THIS run's own system
  curve (``_v6_provenance.chord_knee``), not a hardcoded ``SWEET_P = 0.5``;
* panel (c) "Peak-fleet reduction" is DROPPED -- ``max_fleet_before`` is a
  74_ ``NO_SOURCE`` (all-NaN) column on v5/v6, same root cause as
  ``_fig_saving_fleet_heatmaps.py``.
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

SCRIPT = "fig_wide.py"
DEFAULT_REV = ROOT / "results" / "overnight_2026_05_29_path2"
DEFAULT_OUT = ROOT / "results" / "EWGT_Results"

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(rev: Path):
    s_path, c_path = rev / "tab_balancing_summary.csv", rev / "tab_chosen_schedules.csv"
    s = pd.read_csv(s_path)
    c = pd.read_csv(c_path)
    c["plz"] = c.plz.astype(str)
    V.require_columns(s, ["penalty", "share_willing", "init_cost_eur",
                          "balanced_cost_eur", "max_fleet_before",
                          "max_fleet_after"], source=str(s_path))
    V.require_columns(c, ["penalty", "share_willing", "avg_wait_d_init",
                          "weekly_parcels"], source=str(c_path))
    return s, c


def pareto_at_theta1(s, c, base_total: float):
    th = 1.0
    s1 = s[np.isclose(s.share_willing, th)]
    c1 = c[np.isclose(c.share_willing, th)]
    rows = []
    for P in sorted(s1.penalty.unique()):
        gp = s1[np.isclose(s1.penalty, P)]
        cp = c1[np.isclose(c1.penalty, P)]
        cost = gp.init_cost_eur.sum()
        wait = (cp.avg_wait_d_init * cp.weekly_parcels).sum() / cp.weekly_parcels.sum()
        rows.append({"penalty": P, "wait_d": wait,
                      "sav_pct": 100 * (base_total - cost) / base_total})
    return pd.DataFrame(rows).sort_values("penalty").reset_index(drop=True)


# ─── Sweet-spot 2-panel (WIDER) ────────────────────────────────────────
def fig_sweetspot_wide(s, c, out_dir: Path, base_total: float):
    par = pareto_at_theta1(s, c, base_total)
    assert len(par) >= 3, f"only {len(par)} penalty levels -- need >=3"
    knee_idx = V.chord_knee(par.wait_d.values, par.sav_pct.values)
    sweet_p = float(par.penalty.iloc[knee_idx])
    knee_lo = float(par.penalty.iloc[max(0, knee_idx - 1)])
    knee_hi = float(par.penalty.iloc[min(len(par) - 1, knee_idx + 1)])

    sweet = par[np.isclose(par.penalty, sweet_p)].iloc[0]
    max_sav = par.sav_pct.max(); max_wait = par.wait_d.max()
    par["sav_n"] = par.sav_pct / max_sav * 100
    par["wait_n"] = par.wait_d / max_wait * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Panel A
    ax1.axvspan(par[np.isclose(par.penalty, knee_lo)].wait_d.iloc[0],
                 par[np.isclose(par.penalty, knee_hi)].wait_d.iloc[0],
                 color="orange", alpha=0.18,
                 label=rf"Knee region $P \in [{knee_lo:g}, {knee_hi:g}]$")
    ax1.plot(par.wait_d, par.sav_pct, "o-", color="#1d3557",
              linewidth=1.8, markersize=7)
    ax1.scatter([sweet.wait_d], [sweet.sav_pct], marker="*",
                 s=420, color="gold", edgecolor="black",
                 linewidth=1.2, zorder=10,
                 label=rf"Geometric knee $P = {sweet_p:g}$")
    for _, r in par.iterrows():
        ax1.annotate(rf"${r.penalty:g}$", (r.wait_d, r.sav_pct),
                      xytext=(6, 4), textcoords="offset points",
                      fontsize=8, color="#555")
    ax1.set_xlabel(r"Average wait [d] ($\theta = 1$)")
    ax1.set_ylabel("Cost saving vs daily baseline [%]")
    ax1.set_title("(a) Cost-service Pareto frontier")
    ax1.grid(alpha=0.3); ax1.legend(loc="lower right")

    # Panel B
    ax2.plot(par.penalty, par.sav_n, "o-", color="#1d3557",
              linewidth=1.8, markersize=7, label="% of max saving")
    ax2.plot(par.penalty, par.wait_n, "s--", color="#e76f51",
              linewidth=1.8, markersize=7, label="% of max wait")
    ax2.axvspan(knee_lo, knee_hi, color="orange", alpha=0.18)
    ax2.axvline(sweet_p, color="#e63946", linestyle=":", linewidth=1)
    ax2.fill_between(par.penalty, par.sav_n, par.wait_n,
                      color="#2a9d8f", alpha=0.13)
    ax2.set_xlabel(r"Service penalty $P$ [€/p/d]")
    ax2.set_ylabel("Fraction of maximum [%]")
    ax2.set_title("(b) Diminishing returns")
    ax2.legend(loc="upper right"); ax2.grid(alpha=0.3)
    ax2.set_ylim(-5, 105)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    V.footer(fig, plan=V.PLAN1, script=SCRIPT,
             source="tab_balancing_summary.csv + tab_chosen_schedules.csv")
    written = V.savefig_pinned(fig, out_dir, "fig_PF3_sweetspot")
    plt.close(fig)
    print(f"  saved {written[0]} (11.5\" wide; knee P={sweet_p:g})")


# ─── Saving heatmaps (WIDER) ────────────────────────────────────────────
def fig_saving_heatmaps_wide(s, out_dir: Path, base_total: float):
    have_fleet = bool(s.max_fleet_before.notna().any())
    aggs = dict(init=("init_cost_eur", "sum"), bal=("balanced_cost_eur", "sum"))
    if have_fleet:
        aggs["peak_b"] = ("max_fleet_before", "sum")
        aggs["peak_a"] = ("max_fleet_after", "sum")
    else:
        print("  max_fleet_before is entirely NaN (74_ NO_SOURCE -- no "
              "v5/v6 stage-1 per-hub-day fleet); dropping panel (c) "
              "peak-fleet reduction, keeping (a)/(b) cost-saving only")
    ag = s.groupby(["penalty", "share_willing"], as_index=False).agg(**aggs)
    ag = ag[~np.isclose(ag.penalty, 0.4)]
    ag["init_sav"] = 100 * (base_total - ag.init) / base_total
    ag["bal_sav"] = 100 * (base_total - ag.bal) / base_total

    pivI = ag.pivot(index="penalty", columns="share_willing", values="init_sav")
    pivB = ag.pivot(index="penalty", columns="share_willing", values="bal_sav")
    sV_min = min(pivI.values.min(), pivB.values.min())
    sV_max = max(pivI.values.max(), pivB.values.max())

    ncols = 3 if have_fleet else 2
    fig, axes = plt.subplots(1, ncols, figsize=(13.5 if have_fleet else 9.2, 4.4))

    def heat(ax, mat, cmap, vmin, vmax, title):
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f"{x*100:.0f}" for x in mat.columns], fontsize=8)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"{p:g}" for p in mat.index], fontsize=8)
        ax.set_xlabel(r"$\theta$ [%]", fontsize=10)
        ax.set_title(title, fontsize=11)
        thr = vmin + (vmax - vmin) * 0.55
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isnan(v): continue
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v < thr else "black", fontsize=6.5)
        return im

    im_a = heat(axes[0], pivI, "viridis", sV_min, sV_max,
                 "(a) Cost saving, cost-optimal [%]")
    axes[0].set_ylabel(r"$P$ [€/p/d]", fontsize=10)
    heat(axes[1], pivB, "viridis", sV_min, sV_max,
          "(b) Cost saving, fleet-balanced [%]")
    cb1 = fig.colorbar(im_a, ax=axes[1], fraction=0.046, pad=0.03)
    cb1.set_label("[%]", fontsize=9); cb1.ax.tick_params(labelsize=8)
    if have_fleet:
        ag["peak_red"] = (100 * (ag.peak_b - ag.peak_a)
                          / ag.peak_b.clip(lower=1))
        pivP = ag.pivot(index="penalty", columns="share_willing",
                        values="peak_red")
        pP_max = float(np.ceil(pivP.values.max() / 5) * 5)
        im_c = heat(axes[2], pivP, "magma", 0, pP_max,
                     "(c) Peak-fleet reduction [%]")
        cb2 = fig.colorbar(im_c, ax=axes[2], fraction=0.046, pad=0.03)
        cb2.set_label("[%]", fontsize=9); cb2.ax.tick_params(labelsize=8)
    fig.tight_layout(w_pad=0.8, rect=[0, 0.08, 1, 1])
    stems = "tab_balancing_summary.csv" + (
        "" if have_fleet else " (panel c dropped -- max_fleet_before "
        "NO_SOURCE on v5/v6)")
    V.footer(fig, plan=V.PLAN_BOTH, script=SCRIPT, source=stems, y=0.0)
    written = V.savefig_pinned(fig, out_dir, "fig_PF2_saving_fleet_heatmaps")
    plt.close(fig)
    print(f"  saved {written[0]} (wide)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    V.add_v6_args(ap, default_rev=DEFAULT_REV, default_out=DEFAULT_OUT,
                 rev_help="legacy-adapted run/ directory (74_ <out>/run) "
                          "for v6, or the original path2 dir")
    args = ap.parse_args(argv)
    rev = Path(args.rev_dir)
    out_root = Path(args.out_dir)
    # fig_PF3_sweetspot / fig_PF2_saving_fleet_heatmaps are ALSO the stems
    # _fig_sweetspot_path2.py / _fig_saving_fleet_heatmaps.py write --
    # historically never a collision (this script's default is the
    # separate EWGT_Results/ dir). An explicit --out-dir (the v6
    # regeneration run, sharing one paper_ewgt_2026/ root across scripts)
    # would silently overwrite one pair with the other; route this
    # script's own outputs into a `wide/` subfolder instead so both
    # regenerated products survive (Task 19 W1b; see _STATUS.md).
    out_dir = (out_root if args.out_dir == str(DEFAULT_OUT)
              else out_root / "wide")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_total = V.base_total_with_path2_fallback(rev)
    s, c = load(rev)
    print(f"=== Wider figures ({out_dir}); baseline={base_total:,.2f} EUR/wk ===")
    fig_sweetspot_wide(s, c, out_dir, base_total)
    fig_saving_heatmaps_wide(s, out_dir, base_total)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
