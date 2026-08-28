"""Mechanism figure: why the optimiser keeps (and re-adds) daily plans.

Answers two questions the result figures raise but do not explain, entirely
from the **v6** grid tables:

  (a)  why mean delivery days per week are U-shaped in theta -- a minimum
       near theta = 0.5, a rise towards theta = 0.9, then a jump down at
       theta = 1;
  (b)  what drives that shape: the express pooled tour gets more expensive
       per vehicle-day as the non-willing remainder thins out (b1), the
       penalty mass the routing plan pays grows with theta (b2), and the
       saving only jumps once the express component disappears at
       theta = 1 (b3);
  (c)  why daily plans become MORE frequent as P grows: the per-cell
       routing saving per parcel of maximal bundling is small for a large
       share of cells, so a two-day plan's penalty (~ P x 1 waiting day
       per parcel) overtakes it.

**Which plan each quantity belongs to** -- checked against the tables, not
assumed:

* ``cost_stage1_eur`` / ``penalty_before_eur`` are the **routing-optimal
  plan (stage 1)**.  ``saving_pct`` and the penalty mass in (b2)/(b3) are
  therefore stage-1 quantities, and ``saving_pct`` is gated against the
  grid's own ``routing_saving_plan1_pct``.
* ``dd_cost_eur + express_cost_eur + pool_cost_eur == routing_total_eur ==
  cost_stage2_eur`` exactly (verified for all 88 points), and
  ``fleet == dd_single_veh + dd_pool_veh + express_veh`` with
  ``sum(fleet) == vehicle_days``.  The cost decomposition and the vehicle
  counts are therefore the **operator-polished plan (stage 2)**, and every
  express quantity in (b1)/(b3) says so.

Panel (a) draws ``mean_days_plan*_provmean`` (the mean over the seven LSPs),
which is the column the Kompendium quotes (§40.21: 2.38 / 3.24 / 4.17 /
4.71 / 5.04 at theta = 1); the parcel-weighted ``mean_days_plan*`` is
carried in the CSV alongside it (they differ by <= 0.06 d).

Panel (c) uses the **v6** per-cell costs (``tab_per_cell_costs_v2.csv``,
plan ``stage1``), not the v5 table the dashboard draft read.

Outputs (results/revision_2026_08_v6/):
  figures/supp_fig_mechanism_v2.{pdf,png}
  tables/tab_mechanism_theta_P_v2.csv        one row per (P, theta), 88 rows
  tables/tab_saving_per_parcel_hist_v2.csv   histogram of (c) + threshold rows

Gates (fail loud):
  G1  saving_pct == grid routing_saving_plan1_pct, every (P, theta), 1e-6 pp
  G2  express vehicles and express cost are exactly 0 at theta = 1 (and at
      theta = 0), and strictly positive in between
  G3  the (0, 0) baseline is all-daily (mean days == 6) and its routing cost
      equals the grid's own baseline
  G4  the histogram's cumulative share at a penalty threshold equals the
      share computed directly from the per-cell values
  G5  the per-cell universe is complete (312 cells) at both endpoints
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "presentation"))
sys.path.insert(0, str(ROOT / "scripts" / "revision"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402

import _style as S  # noqa: E402  -- the ONLY colour source
from _figs_tables_v2 import PLAN1, PLAN2  # noqa: E402

REV = ROOT / "results" / "revision_2026_08_v6"
COSTS_CSV = REV / "tab_costs_v2.csv"
FLEET_CSV = REV / "tab_fleet_per_hub_v2.csv"
GRID_CSV = REV / "tables" / "tab_grid_full_v2.csv"
CELLS_CSV = REV / "tables" / "tab_per_cell_costs_v2.csv"
FIG_DIR = REV / "figures"
TAB_DIR = REV / "tables"

STEM = "supp_fig_mechanism_v2"
TAB_MECH = "tab_mechanism_theta_P_v2.csv"
TAB_HIST = "tab_saving_per_parcel_hist_v2.csv"

P_CURVES = (0.0, 0.25, 0.5, 1.0)     # panel (a)
P_EXPRESS = (0.0, 0.25)              # panel (b1)
P_FOCUS = 0.25                       # panels (b2), (b3)
HIST_THRESHOLDS = (0.25, 0.5, 1.0)   # panel (c): P x 1 waiting day per parcel
HIST_BIN = 0.05                      # so 0.25 / 0.5 / 1.0 are exact bin edges
N_CELLS = 312                        # the v6 (provider, plz) universe
N_PROVIDERS = 7

#: P is always encoded by the paper's service-penalty ramp (``CMAP_PENALTY``);
#: the plan is always encoded by line style / marker, never by hue.
_P_SHADE = {0.0: 0.05, 0.25: 0.33, 0.5: 0.58, 1.0: 0.80}
QTY_SAVING = S.FREQ[6]     # stage-1 routing saving
QTY_EXPRESS = S.FREQ[2]    # stage-2 express component
FOOT_KW = dict(ha="center", va="top", fontsize=7.4, color=S.INK_SOFT)

#: Matplotlib stamps a /CreationDate into every PDF (and a timestamp into
#: every PNG), so two renders of identical content get different md5s. 70_'s
#: manifest and gate G7 are md5-based (Kompendium §38.8); this mirrors 70_'s
#: own suppression so a re-render of unchanged inputs is byte-identical.
_PDF_META = {"CreationDate": None}
_PNG_META = {"Software": None}


def p_color(P: float):
    return matplotlib.colormaps[S.CMAP_PENALTY](_P_SHADE[P])


# ─────────────────────────────────────────────────────────────────────────
# data (pure functions: frames in, frames out -- no plotting, no file I/O)
# ─────────────────────────────────────────────────────────────────────────
def regular_eur_per_vehicle_day(costs: pd.DataFrame, fleet: pd.DataFrame) -> float:
    """Weekly routing cost per vehicle-day of the daily baseline (P=0, th=0).

    The reference line of panel (b1): what a REGULAR delivery tour costs per
    vehicle and day when every cell is served daily.  Stage 1 and stage 2
    coincide at the baseline, so the value is plan-independent.
    """
    c = costs[(costs.penalty == 0) & (costs.share_willing == 0)]
    f = fleet[(fleet.penalty == 0) & (fleet.share_willing == 0)]
    assert len(c) == N_PROVIDERS, f"G3: {len(c)} baseline cost rows"
    vd = float(f.fleet.sum())
    assert vd > 0, "G3: baseline has no vehicle-days"
    return float(c.cost_stage1_eur.sum()) / vd


def mechanism_table(costs: pd.DataFrame, fleet: pd.DataFrame,
                    grid: pd.DataFrame) -> pd.DataFrame:
    """One row per (P, theta) with everything panels (a), (b1)-(b3) draw.

    Pure data-shaping.  Columns:

    ``mean_days_plan1``/``2``          provider mean, the Kompendium column
    ``mean_days_plan1_pw``/``2_pw``    parcel-weighted counterpart
    ``express_eur_per_vd``             stage-2 express cost / express vehicle-days
    ``express_share_pct``              stage-2 express cost / stage-2 routing cost
    ``penalty_keur``                   stage-1 penalty mass (k EUR/week)
    ``penalty_plan2_keur``             stage-2 penalty mass (k EUR/week)
    ``saving_pct``                     stage-1 routing saving vs the baseline
    ``saving_plan2_pct``               stage-2 routing saving vs the baseline

    ``express_eur_per_vd`` is NaN exactly where there are no express
    vehicles (theta in {0, 1}); that is asserted, never patched (G2).
    """
    base = costs[(costs.penalty == 0) & (costs.share_willing == 0)]
    rt_base = float(base.cost_stage1_eur.sum())
    g0 = grid[(grid.penalty == 0) & (grid.share_willing == 0)]
    assert len(g0) == 1, "G3: no unique baseline row in the grid table"
    assert abs(rt_base - float(g0.routing_cost_plan1_eur.iloc[0])) < 1e-6, \
        "G3: baseline routing cost differs between cost and grid tables"
    assert float(g0.mean_days_plan1_provmean.iloc[0]) == 6.0, \
        "G3: baseline is not all-daily"

    rows = []
    for _, g in grid.sort_values(["penalty", "share_willing"]).iterrows():
        P, th = float(g.penalty), float(g.share_willing)
        c = costs[np.isclose(costs.penalty, P) & np.isclose(costs.share_willing, th)]
        f = fleet[np.isclose(fleet.penalty, P) & np.isclose(fleet.share_willing, th)]
        assert len(c) == N_PROVIDERS, f"G: {len(c)} cost rows at P={P}, theta={th}"
        ev = float(f.express_veh.sum())
        ec = float(c.express_cost_eur.sum())
        endpoint = th in (0.0, 1.0)
        assert (ev == 0 and ec == 0) if endpoint else (ev > 0 and ec > 0), (
            f"G2: express vehicles/cost {ev}/{ec} at P={P}, theta={th} -- "
            "expected exactly zero at theta in {0, 1} and positive in between")
        saving = 100.0 * (rt_base - float(c.cost_stage1_eur.sum())) / rt_base
        assert abs(saving - float(g.routing_saving_plan1_pct)) < 1e-6, (
            f"G1: recomputed stage-1 saving {saving} != grid "
            f"{float(g.routing_saving_plan1_pct)} at P={P}, theta={th}")
        rows.append(dict(
            penalty=P, share_willing=th,
            mean_days_plan1=float(g.mean_days_plan1_provmean),
            mean_days_plan2=float(g.mean_days_plan2_provmean),
            mean_days_plan1_pw=float(g.mean_days_plan1),
            mean_days_plan2_pw=float(g.mean_days_plan2),
            express_veh_days=ev, express_cost_eur=ec,
            express_eur_per_vd=(ec / ev if ev > 0 else np.nan),
            express_share_pct=100.0 * ec / float(c.routing_total_eur.sum()),
            penalty_keur=float(c.penalty_before_eur.sum()) / 1000.0,
            penalty_plan2_keur=float(c.penalty_eur.sum()) / 1000.0,
            saving_pct=saving,
            saving_plan2_pct=float(g.routing_saving_plan2_pct),
            wait_d_plan1=float(g.wait_d_plan1), wait_d_plan2=float(g.wait_d_plan2),
        ))
    return pd.DataFrame(rows)


def saving_per_parcel(cells: pd.DataFrame) -> pd.DataFrame:
    """Per-cell routing saving per parcel of maximal bundling, (P=0, theta=1).

    Pure data-shaping.  Both endpoints are read on the **routing-optimal
    plan (stage 1)**, so the number is "what bundling this cell as hard as
    the router wants would save per parcel and week" -- the quantity a
    two-day plan's penalty (~ P x 1 waiting day per parcel) competes with.
    The parcel count is the baseline's, so a plan that moves parcels
    between days cannot move the denominator.
    """
    d = cells[cells.plan == "stage1"]
    base = d[(d.penalty == 0) & (d.share_willing == 0)].set_index(["provider", "plz"])
    full = d[(d.penalty == 0) & np.isclose(d.share_willing, 1.0)].set_index(["provider", "plz"])
    assert len(base) == N_CELLS and len(full) == N_CELLS, \
        f"G5: {len(base)}/{len(full)} cells, expected {N_CELLS}"
    assert base.index.sort_values().equals(full.index.sort_values()), \
        "G5: the two endpoints do not cover the same cells"
    j = full.join(base, rsuffix="_base")
    assert (j.cell_parcels_week_base > 0).all(), "G5: a baseline cell has no parcels"
    out = j[["cell_cost_eur", "cell_parcels_week_base"]].copy()
    out["saving_eur_per_parcel"] = (
        (j.cell_cost_eur_base - j.cell_cost_eur) / j.cell_parcels_week_base)
    return out.reset_index()[["provider", "plz", "cell_parcels_week_base",
                              "saving_eur_per_parcel"]]


def hist_table(sav: pd.DataFrame, bin_width: float = HIST_BIN,
               thresholds=HIST_THRESHOLDS) -> pd.DataFrame:
    """Histogram of the per-cell saving per parcel, with the penalty edges.

    Pure data-shaping.  Bins are half-open ``[left, right)`` of width
    ``bin_width``, aligned to zero so every threshold is an exact edge; the
    last bin is closed so the maximum is counted.  ``cum_share_pct`` at the
    edge ``P`` is therefore literally "share of cells whose saving per parcel
    is below the penalty of a two-day plan at that P", which G4 checks
    against the share computed straight from the values.
    """
    v = sav.saving_eur_per_parcel.to_numpy(float)
    assert np.isfinite(v).all(), "G: non-finite per-cell saving"
    lo = np.floor(v.min() / bin_width) * bin_width
    hi = np.ceil(v.max() / bin_width) * bin_width + bin_width
    edges = np.round(np.arange(lo, hi + 0.5 * bin_width, bin_width), 10)
    counts, _ = np.histogram(v, bins=edges)
    assert counts.sum() == len(v), "G: histogram lost cells"
    tab = pd.DataFrame(dict(bin_left=edges[:-1], bin_right=edges[1:],
                            n_cells=counts))
    tab["share_pct"] = 100.0 * tab.n_cells / len(v)
    tab["cum_share_pct"] = tab.share_pct.cumsum()
    tab["penalty_threshold"] = np.where(
        np.isin(np.round(tab.bin_right, 10), np.round(thresholds, 10)),
        tab.bin_right, np.nan)
    for P in thresholds:
        row = tab[np.isclose(tab.bin_right, P)]
        assert len(row) == 1, f"G4: {P} is not a bin edge"
        direct = 100.0 * float((v < P).mean())
        assert abs(float(row.cum_share_pct.iloc[0]) - direct) < 1e-9, (
            f"G4: cumulative share {float(row.cum_share_pct.iloc[0])} != direct "
            f"{direct} at the P={P} threshold")
    return tab


# ─────────────────────────────────────────────────────────────────────────
# figure
# ─────────────────────────────────────────────────────────────────────────
def draw(mech: pd.DataFrame, hist: pd.DataFrame, sav: pd.DataFrame,
         regular_eur_vd: float) -> plt.Figure:
    S.apply("paper")
    fig = plt.figure(figsize=(7.0, 6.8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.12, 1.0], hspace=0.52,
                           wspace=0.40, figure=fig,
                           left=0.085, right=0.985, top=0.96, bottom=0.20)

    # (a) mean delivery days over theta -----------------------------------
    ax = fig.add_subplot(gs[0, :2])
    for P in P_CURVES:
        d = mech[np.isclose(mech.penalty, P)].sort_values("share_willing")
        col = p_color(P)
        ax.plot(d.share_willing * 100, d.mean_days_plan1, "-o", color=col,
                ms=2.6, lw=1.3, label=rf"$P = {P:g}$")
        ax.plot(d.share_willing * 100, d.mean_days_plan2, "--", color=col,
                lw=1.1, alpha=0.85)
    ax.axvspan(40, 60, color=S.INK_SOFT, alpha=0.07, lw=0, zorder=0)
    ax.annotate(r"minimum near $\theta \approx 0.5$", xy=(50, 4.70),
                xytext=(21, 3.30), fontsize=7, color=S.INK_SOFT,
                arrowprops=dict(arrowstyle="-", color=S.INK_SOFT, lw=0.7))
    ax.annotate("thin express remainder,\npenalty at its peak", xy=(90, 5.90),
                xytext=(46, 6.28), fontsize=7, color=S.INK_SOFT,
                arrowprops=dict(arrowstyle="-", color=S.INK_SOFT, lw=0.7))
    ax.annotate("no express tours left", xy=(99, 2.15), xytext=(63, 2.90),
                fontsize=7, color=S.INK_SOFT,
                arrowprops=dict(arrowstyle="-", color=S.INK_SOFT, lw=0.7))
    ax.set_xlabel(r"Willing customers $\theta$ [%]")
    ax.set_ylabel("Mean delivery days per week", fontsize=9.5)
    ax.set_xlim(0, 103)
    ax.set_ylim(1.7, 6.75)
    ax.set_yticks([2, 3, 4, 5, 6])
    ax.grid(axis="y", color=S.GRID, lw=0.5, alpha=0.8, zorder=0)
    ax.set_title("(a) Delivery days the optimiser chooses", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="lower left",
              columnspacing=1.0, handlelength=1.6, borderaxespad=0.3)

    # (c) per-cell saving per parcel vs the penalty thresholds -------------
    axc = fig.add_subplot(gs[0, 2])
    axc.bar(hist.bin_left, hist.n_cells, width=HIST_BIN, align="edge",
            color=QTY_EXPRESS, edgecolor="white", lw=0.3, zorder=2)
    top = float(hist.n_cells.max())
    axc.set_ylim(0, top * 1.45)
    for i, P in enumerate(HIST_THRESHOLDS):
        share = float(hist[np.isclose(hist.bin_right, P)].cum_share_pct.iloc[0])
        axc.axvline(P, color=p_color(P), lw=1.1, ls="--", zorder=3)
        axc.text(0.97, 0.97 - 0.085 * i, f"$P = {P:g}$: {share:.0f} %",
                 transform=axc.transAxes, color=p_color(P), fontsize=7,
                 ha="right", va="top")
    axc.text(0.97, 0.97 - 0.085 * len(HIST_THRESHOLDS) - 0.02,
             "of cells save less than\na two-day plan's penalty",
             transform=axc.transAxes, fontsize=6.6, color=S.INK_SOFT,
             ha="right", va="top")
    axc.set_xlabel("Routing saving per parcel\nat maximal bundling [€]", fontsize=8.5)
    axc.set_ylabel("Cells", fontsize=9.5)
    axc.set_title("(c) Why daily plans grow with $P$", fontsize=9.5)
    axc.grid(axis="y", color=S.GRID, lw=0.5, alpha=0.8, zorder=0)

    # (b1) express cost per vehicle-day -----------------------------------
    ax1 = fig.add_subplot(gs[1, 0])
    for P in P_EXPRESS:
        d = mech[np.isclose(mech.penalty, P)].sort_values("share_willing")
        ax1.plot(d.share_willing * 100, d.express_eur_per_vd, "-o",
                 color=p_color(P), ms=2.6, lw=1.3, label=rf"$P = {P:g}$")
    ax1.axhline(regular_eur_vd, color=S.INK_SOFT, ls=":", lw=1.0)
    ax1.text(3, regular_eur_vd + 8, f"regular tour {regular_eur_vd:.0f} €",
             fontsize=6.6, color=S.INK_SOFT)
    ax1.set_title("(b1) Express € per vehicle-day", fontsize=9.5)
    ax1.set_xlabel(r"$\theta$ [%]")
    ax1.set_ylabel("€ per vehicle-day", fontsize=8.5)
    ax1.grid(axis="y", color=S.GRID, lw=0.5, alpha=0.8, zorder=0)
    ax1.legend(frameon=False, fontsize=7.4, loc="upper left", handlelength=1.6,
               borderaxespad=0.3)

    # (b2) penalty mass of the routing plan --------------------------------
    ax2 = fig.add_subplot(gs[1, 1])
    d = mech[np.isclose(mech.penalty, P_FOCUS)].sort_values("share_willing")
    ax2.plot(d.share_willing * 100, d.penalty_keur, "-o", color=p_color(P_FOCUS),
             ms=2.6, lw=1.3)
    ax2.set_title(rf"(b2) Penalty paid, $P = {P_FOCUS:g}$", fontsize=9.5)
    ax2.set_xlabel(r"$\theta$ [%]")
    ax2.set_ylabel("k€ per week", fontsize=8.5)
    ax2.grid(axis="y", color=S.GRID, lw=0.5, alpha=0.8, zorder=0)
    ax2.text(0.04, 0.96, "more parcels wait,\nso each dropped\nday costs more",
             transform=ax2.transAxes, fontsize=6.6, color=S.INK_SOFT, va="top")

    # (b3) saving vs express share ----------------------------------------
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(d.share_willing * 100, d.saving_pct, "-s", color=QTY_SAVING,
             ms=2.6, lw=1.3)
    ax3.plot(d.share_willing * 100, d.express_share_pct, "-o", color=QTY_EXPRESS,
             ms=2.6, lw=1.3)
    ax3.set_ylim(-1.2, 22.5)
    ax3.text(0.03, 0.99, "express cost share\n(stage 2)", transform=ax3.transAxes,
             color=QTY_EXPRESS, fontsize=6.8, va="top")
    ax3.text(0.30, 0.14, "routing saving\n(stage 1)", transform=ax3.transAxes,
             color=QTY_SAVING, fontsize=6.8, va="top")
    ax3.set_title(rf"(b3) Saving vs express, $P = {P_FOCUS:g}$", fontsize=9.5)
    ax3.set_xlabel(r"$\theta$ [%]")
    ax3.set_ylabel("%", fontsize=8.5)
    ax3.grid(axis="y", color=S.GRID, lw=0.5, alpha=0.8, zorder=0)

    # Kept to five short lines: the saved bounding box is "tight", so one long
    # line would widen the figure past the paper's \linewidth.
    fig.text(0.5, 0.125,
             f"Grid v6. In (a) the {PLAN1} is solid, the {PLAN2} dashed;\n"
             "the curve is the provider mean of the LSPs' mean delivery days.\n"
             "Express quantities in (b1)/(b3) come from the stage-2 cost "
             "decomposition; the penalty mass in (b2) and the saving in "
             "(b3)/(c) are stage 1.\n"
             "Express tour = parcels of non-willing customers on days without a "
             "regular tour, bundled at the depot; in (b3) the saving jumps at "
             r"$\theta = 1$," "\n"
             "where that component disappears. Panel (c): all "
             f"{N_CELLS} cells at $P = 0$; a two-day plan's penalty is "
             r"$\approx P \times 1$ waiting day per parcel.",
             **FOOT_KW)
    return fig


def save(fig, stem: str) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for ext, meta in (("pdf", _PDF_META), ("png", _PNG_META)):
        p = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", metadata=meta)
        print(f"  saved {p.relative_to(ROOT)}")
        out.append(p)
    plt.close(fig)
    return out


def write(tab: pd.DataFrame, name: str) -> Path:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    p = TAB_DIR / name
    tab.to_csv(p, index=False)
    print(f"  wrote {p.relative_to(ROOT)}  ({len(tab)} rows)")
    return p


def main() -> None:
    costs = pd.read_csv(COSTS_CSV)
    fleet = pd.read_csv(FLEET_CSV)
    grid = pd.read_csv(GRID_CSV)
    cells = pd.read_csv(CELLS_CSV, dtype={"plz": str})

    mech = mechanism_table(costs, fleet, grid)
    write(mech, TAB_MECH)
    sav = saving_per_parcel(cells)
    hist = hist_table(sav)
    write(hist, TAB_HIST)

    reg = regular_eur_per_vehicle_day(costs, fleet)
    print(f"  regular delivery tour: {reg:.1f} € per vehicle-day (P=0, theta=0)")
    d = mech[np.isclose(mech.penalty, P_FOCUS)].sort_values("share_willing")
    lo = d[np.isclose(d.share_willing, 0.1)].express_eur_per_vd.iloc[0]
    hi = d[np.isclose(d.share_willing, 0.9)].express_eur_per_vd.iloc[0]
    print(f"  express tour at P={P_FOCUS:g}: {lo:.1f} € per vehicle-day at "
          f"theta=0.1 -> {hi:.1f} € at theta=0.9")
    print(f"  per-cell saving per parcel: {sav.saving_eur_per_parcel.min():.3f} "
          f".. {sav.saving_eur_per_parcel.max():.3f} €, "
          + ", ".join(
              f"{float(hist[np.isclose(hist.bin_right, P)].cum_share_pct.iloc[0]):.1f} % "
              f"below P={P:g}" for P in HIST_THRESHOLDS))

    save(draw(mech, hist, sav, reg), STEM)


if __name__ == "__main__":
    main()
