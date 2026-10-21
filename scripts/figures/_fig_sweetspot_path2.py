"""Reproduce the three sweet-spot figures from yesterday with Path-2 data:

  1) Two-panel: (A) Pareto frontier with sweet-spot star + knee region,
                (B) Diminishing returns — saving captured % vs wait incurred %.
  2) Single-panel scatter with viridis colormap encoding P.
  3) Pareto efficiency curve: saving kept - wait paid as function of P.

The sweet-spot is the chord-distance knee of the system-level (P sweep)
Pareto frontier -- recomputed every run, never hardcoded (Task 19 W1b; see
below).

Default outputs to results/paper_final_2026_05_30/05_optimization/:
  fig_PF3_sweetspot.{png,pdf}
  fig_pareto_viridis.{png,pdf}
  fig_pareto_efficiency.{png,pdf}

Task 19 W1b (v6 regeneration)
-----------------------------
v6 status B: ``init``-suffixed columns are the routing-optimal (stage 1)
plan -- this script's own docstring says "Path-2 init", so per the brief's
plan convention it stays on stage 1. Source:
``scripts/revision/74_v2_to_legacy_tables.py``'s ``tab_balancing_summary.csv``
/ ``tab_chosen_schedules.csv``.

Two corrections, both required because the ORIGINAL numbers were baked in
rather than derived from the loaded curve:

* ``BASE = 1,909,747.75`` EUR (the stale 2026-07/path2 denominator) ->
  this grid's OWN baseline, from 74_'s ``legacy_manifest.json`` (a v6
  saving must never be taken against a different grid's total).
* ``SWEET_P = 0.5`` (hardcoded) -> ``_v6_provenance.chord_knee`` on THIS
  run's own system curve. On v6 this still lands at P=0.5, but the value
  is now a consequence of the data, not an assumption about it.

The old ``$P \\geq 5 \\to$ daily (0%)`` annotation is FALSE on v6
(Kompendium §40.23b: v6's operator polish keeps a little pooling all the
way to P=10 -- saving is 0.12%/0.06% at P=5/10, not exactly zero, and 1-2
of 312 cells still batch); replaced with the actual minimum saving among
the P>=5 points, read from the curve.
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
from matplotlib import rcParams, cm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
import _v6_provenance as V  # noqa: E402

SCRIPT = "_fig_sweetspot_path2.py"
DEFAULT_REV = ROOT / "results" / "overnight_2026_05_29_path2"
DEFAULT_OUT = ROOT / "results" / "paper_final_2026_05_30"

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 11, "axes.titlesize": 12, "legend.fontsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "savefig.bbox": "tight", "savefig.dpi": 220, "pdf.fonttype": 42,
})


def load_pareto(rev: Path, base_total: float):
    summ_path, chosen_path = rev / "tab_balancing_summary.csv", rev / "tab_chosen_schedules.csv"
    summ = pd.read_csv(summ_path)
    chosen = pd.read_csv(chosen_path)
    V.require_columns(summ, ["penalty", "share_willing", "init_cost_eur"],
                      source=str(summ_path))
    V.require_columns(chosen, ["penalty", "share_willing", "avg_wait_d_init",
                               "weekly_parcels", "schedule_size_init"],
                      source=str(chosen_path))
    th = 1.0
    cs = chosen[np.isclose(chosen.share_willing, th)]
    ss = summ[np.isclose(summ.share_willing, th)]
    rows = []
    for P in sorted(cs.penalty.unique()):
        g = cs[np.isclose(cs.penalty, P)]
        gs = ss[np.isclose(ss.penalty, P)]
        wait = (g.avg_wait_d_init * g.weekly_parcels).sum() / g.weekly_parcels.sum()
        cost = gs.init_cost_eur.sum()
        sav = 100 * (base_total - cost) / base_total
        rows.append({"penalty": P, "wait_d": wait, "sav_pct": sav,
                      "mean_freq": g.schedule_size_init.mean(),
                      "n_batched": int((g.schedule_size_init < 6).sum())})
    return pd.DataFrame(rows).sort_values("penalty").reset_index(drop=True)


# ─── Figure 1: 2-panel sweet-spot ──────────────────────────────────────
def fig_sweetspot_2panel(pareto: pd.DataFrame, out_dir: Path, sweet_p: float,
                         knee_lo: float, knee_hi: float) -> None:
    sweet_row = pareto[np.isclose(pareto.penalty, sweet_p)].iloc[0]
    max_sav = pareto.sav_pct.max()
    max_wait = pareto.wait_d.max()
    pct_sav = 100 * sweet_row.sav_pct / max_sav
    pct_wait = 100 * sweet_row.wait_d / max_wait

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    fig.suptitle(rf"Service-penalty sweet-spot — why $P = {sweet_p:g}$ €/parcel/day",
                  fontsize=14, y=1.02)

    # Panel A: Pareto frontier
    ax1.axvspan(pareto[np.isclose(pareto.penalty, knee_lo)].wait_d.iloc[0],
                 pareto[np.isclose(pareto.penalty, knee_hi)].wait_d.iloc[0],
                 color="orange", alpha=0.18, label=f"Knee region $P\\in[{knee_lo:g},{knee_hi:g}]$")
    ax1.plot(pareto.wait_d, pareto.sav_pct, "o-", color="#1d3557",
              linewidth=2.2, markersize=7)
    ax1.scatter([sweet_row.wait_d], [sweet_row.sav_pct], marker="*",
                 s=540, color="#e63946", edgecolor="black", linewidth=1.4,
                 zorder=10, label=f"Geometric knee  $P={sweet_p:g}$")
    # Annotate cost-optimal and sweet-spot
    max_row = pareto.iloc[pareto.sav_pct.idxmax()]
    ax1.annotate(rf"$P\to 0$ (cost-optimal)" "\n"
                  f"{max_row.sav_pct:.1f}% @ {max_row.wait_d:.2f} d",
                  (max_row.wait_d, max_row.sav_pct),
                  xytext=(-15, -20), textcoords="offset points",
                  fontsize=10, color="#1d3557", ha="right")
    ax1.annotate(f"$P = {sweet_p:g}$ (sweet-spot)\n"
                  f"{sweet_row.sav_pct:.1f}% saving  ·  {sweet_row.wait_d:.2f} d wait\n"
                  f"= {pct_sav:.0f}% of max saving for {pct_wait:.0f}% of max wait",
                  (sweet_row.wait_d, sweet_row.sav_pct),
                  xytext=(60, -50), textcoords="offset points",
                  fontsize=9.5, color="#1d3557",
                  arrowprops=dict(arrowstyle="-", color="#e63946", lw=1),
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                             edgecolor="#e63946"))
    # Annotate the high-P tail with the ACTUAL minimum saving there. The
    # pre-revision claim "$P >= 5$ -> daily (0%)" is FALSE on v6 (Kompendium
    # §40.23b): the operator polish keeps a sliver of pooling all the way
    # to P=10, so saving never hits exactly zero.
    tail = pareto[pareto.penalty >= 5]
    if len(tail):
        tail_min = float(tail.sav_pct.min())
        ax1.text(0.02, max_sav * 0.07,
                  rf"$P \geq 5$: saving $\leq$ {tail_min:.2g}% "
                  "(not exactly daily)",
                  fontsize=10, color="gray")
    ax1.set_xlabel("Average customer wait  [days]")
    ax1.set_ylabel("Weekly cost saving vs daily baseline  [%]")
    ax1.set_title(r"A · Cost-service Pareto frontier", fontsize=12, loc="left")
    ax1.grid(alpha=0.3); ax1.legend(loc="lower right")
    ax1.set_xlim(-0.02, max_wait * 1.05); ax1.set_ylim(-1, max_sav + 2)

    # Panel B: Diminishing returns — % of max saving captured + % of max wait
    pareto["sav_pct_of_max"] = 100 * pareto.sav_pct / max_sav
    pareto["wait_pct_of_max"] = 100 * pareto.wait_d / max_wait
    ax2.plot(pareto.penalty, pareto.sav_pct_of_max, "o-", color="#1d3557",
              linewidth=2.2, markersize=7, label="% of max saving captured")
    ax2.plot(pareto.penalty, pareto.wait_pct_of_max, "s--", color="#e76f51",
              linewidth=2.2, markersize=7, label="% of max wait incurred")
    ax2.axvspan(knee_lo, knee_hi, color="orange", alpha=0.18)
    ax2.axvline(sweet_p, color="#e63946", linestyle=":", linewidth=1.2)
    ax2.text(sweet_p, -8, f"$P = {sweet_p:g}$", color="#e63946",
              ha="center", fontsize=10, fontweight="bold")
    gap = sweet_row.sav_pct / max_sav * 100 - sweet_row.wait_d / max_wait * 100
    ax2.annotate(f"gap = {gap:.0f} pp\n(saving kept − wait paid)",
                 (sweet_p, (pct_sav + pct_wait) / 2),
                 xytext=(0.7, 50), textcoords="data",
                 fontsize=9.5, color="#1d3557",
                 arrowprops=dict(arrowstyle="<->", color="#1d3557", lw=1.2))
    ax2.fill_between(pareto.penalty, pareto.sav_pct_of_max,
                      pareto.wait_pct_of_max, color="#2a9d8f", alpha=0.12)
    ax2.set_xlabel(r"Service penalty $P$  [€ / parcel / day]  = shadow price of waiting")
    ax2.set_ylabel("Fraction of maximum  [%]")
    ax2.set_title("B · Diminishing returns — the gap peaks in the knee region",
                   fontsize=12, loc="left")
    ax2.legend(loc="upper right"); ax2.grid(alpha=0.3)
    ax2.set_ylim(-5, 105); ax2.set_xlim(-0.1, pareto.penalty.max() * 1.05)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    V.footer(fig, plan=V.PLAN1, script=SCRIPT,
             source="tab_balancing_summary.csv + tab_chosen_schedules.csv")
    written = V.savefig_pinned(fig, out_dir, "fig_PF3_sweetspot")
    plt.close(fig)
    print(f"saved {written[0]}")


# ─── Figure 2: viridis scatter ─────────────────────────────────────────
def fig_pareto_viridis(pareto: pd.DataFrame, out_dir: Path, sweet_p: float,
                       knee_lo: float, knee_hi: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    # Smooth curve through points (use spline-ish via PCHIP)
    from scipy.interpolate import PchipInterpolator
    x_fine = np.linspace(pareto.wait_d.min(), pareto.wait_d.max(), 200)
    try:
        pchip = PchipInterpolator(pareto.wait_d.values, pareto.sav_pct.values)
        y_fine = pchip(x_fine)
        ax.plot(x_fine, y_fine, "-", color="#888888", linewidth=1.2, alpha=0.6,
                 zorder=2)
    except Exception:
        ax.plot(pareto.wait_d, pareto.sav_pct, "-", color="#888888", linewidth=1.2, alpha=0.6,
                 zorder=2)
    # Penalty as colormap (log scale, viridis_r so 0 is dark)
    eps = 0.005
    P_plot = pareto.penalty.copy()
    P_plot[P_plot == 0] = eps  # avoid log(0)
    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=max(eps, P_plot.min()), vmax=P_plot.max())
    sc = ax.scatter(pareto.wait_d, pareto.sav_pct, c=P_plot,
                     cmap="viridis_r", norm=norm, s=130,
                     edgecolor="black", linewidth=0.5, zorder=5)
    sweet_row = pareto[np.isclose(pareto.penalty, sweet_p)].iloc[0]
    ax.scatter([sweet_row.wait_d], [sweet_row.sav_pct], marker="o",
                s=400, facecolor="none", edgecolor="#e63946", linewidth=2.5,
                zorder=11)
    knee_lo_row = pareto[np.isclose(pareto.penalty, knee_lo)].iloc[0]
    knee_hi_row = pareto[np.isclose(pareto.penalty, knee_hi)].iloc[0]
    ax.axvspan(knee_lo_row.wait_d, knee_hi_row.wait_d, color="orange", alpha=0.18)
    ax.set_xlabel("Average customer wait  [days]")
    ax.set_ylabel("Weekly cost saving vs daily baseline  [%]")
    ax.grid(alpha=0.3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"Service penalty $P$  [€ / parcel / day]")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    V.footer(fig, plan=V.PLAN1, script=SCRIPT,
             source="tab_balancing_summary.csv + tab_chosen_schedules.csv")
    written = V.savefig_pinned(fig, out_dir, "fig_pareto_viridis")
    plt.close(fig)
    print(f"saved {written[0]}")


# ─── Figure 3: Pareto efficiency (saving kept - wait paid) ────────────
def fig_pareto_efficiency(pareto: pd.DataFrame, out_dir: Path, sweet_p: float,
                          knee_lo: float, knee_hi: float) -> None:
    max_sav = pareto.sav_pct.max()
    max_wait = pareto.wait_d.max()
    pareto = pareto.copy()
    pareto["efficiency"] = (100 * pareto.sav_pct / max_sav
                             - 100 * pareto.wait_d / max_wait)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(pareto.penalty, pareto.efficiency, "o-", color="#1d3557",
             linewidth=2, markersize=8)
    sweet_eff = pareto[np.isclose(pareto.penalty, sweet_p)].iloc[0].efficiency
    ax.axvspan(knee_lo, knee_hi, color="orange", alpha=0.18)
    ax.scatter([sweet_p], [sweet_eff], marker="o", s=320,
                facecolor="none", edgecolor="#e63946", linewidth=2.5, zorder=10)
    ax.set_xlabel(r"Service penalty $P$  [€ / parcel / day]")
    ax.set_ylabel("Pareto efficiency:  saving kept − wait paid  [pp]")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, pareto.penalty.max() * 1.05)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    V.footer(fig, plan=V.PLAN1, script=SCRIPT,
             source="tab_balancing_summary.csv + tab_chosen_schedules.csv")
    written = V.savefig_pinned(fig, out_dir, "fig_pareto_efficiency")
    plt.close(fig)
    print(f"saved {written[0]}  (peak at "
          f"P={pareto.penalty[pareto.efficiency.idxmax()]:g})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    V.add_v6_args(ap, default_rev=DEFAULT_REV, default_out=DEFAULT_OUT,
                 rev_help="legacy-adapted run/ directory (74_ <out>/run) "
                          "for v6, or the original path2 dir")
    args = ap.parse_args(argv)
    rev = Path(args.rev_dir)
    out_root = Path(args.out_dir)
    # The historical DEFAULT keeps its own OUT_BASE / OUT_BASE/05_optimization
    # split, unchanged. An EXPLICIT --out-dir (the actual v6 regeneration
    # run) writes flat -- matching every other script in this wave and the
    # shared results/revision_2026_08_analyses_v6/paper_ewgt_2026/_STATUS.md
    # convention (Task 19 W1b).
    if args.out_dir == str(DEFAULT_OUT):
        out_base, out_dir = out_root, out_root / "05_optimization"
    else:
        out_base = out_dir = out_root
    out_base.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_total = V.base_total_with_path2_fallback(rev)
    pareto = load_pareto(rev, base_total)
    assert len(pareto) >= 3, (
        f"{rev}: only {len(pareto)} penalty levels -- chord_knee needs >=3")
    knee_idx = V.chord_knee(pareto.wait_d.values, pareto.sav_pct.values)
    sweet_p = float(pareto.penalty.iloc[knee_idx])
    knee_lo = float(pareto.penalty.iloc[max(0, knee_idx - 1)])
    knee_hi = float(pareto.penalty.iloc[min(len(pareto) - 1, knee_idx + 1)])

    pareto.to_csv(out_base / "_pareto_path2_theta1.csv", index=False)
    print(f"Loaded {len(pareto)} Pareto points (theta=1); baseline="
          f"{base_total:,.2f} EUR/wk; chord-distance knee P={sweet_p:g} "
          f"(neighbours {knee_lo:g}/{knee_hi:g})")
    print(pareto[["penalty", "wait_d", "sav_pct", "mean_freq"]].round(3).to_string(index=False))
    fig_sweetspot_2panel(pareto, out_dir, sweet_p, knee_lo, knee_hi)
    fig_pareto_viridis(pareto, out_dir, sweet_p, knee_lo, knee_hi)
    fig_pareto_efficiency(pareto, out_dir, sweet_p, knee_lo, knee_hi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
