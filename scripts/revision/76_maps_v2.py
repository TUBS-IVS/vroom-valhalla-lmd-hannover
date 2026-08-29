"""Paper-style spatial figures from the v6 grid, operator-polished plan.

Re-renders four presentation figures (40_act4_maps.py fig41/fig44,
70_act7_savings_structure.py fig71/fig75) from the **v6 two-plan grid**
instead of the legacy Stage-3 tables, on the **operator-polished plan
(stage 2)** throughout, in the paper's rcParams (``_style.apply("paper")``).

  supp_map_freq_theta_v2     delivery frequency per area, P = 0.25, theta panels
  supp_map_saving_P_v2       routing-lens cost saving per area, theta = 1, P panels
  supp_map_wait_theta_v2     added wait of a held parcel per area, P = 0.25, theta panels
  supp_penalty_raumtyp_v2    euro-weighted saving vs P by settlement type, theta = 1

Source: ``tables/tab_per_cell_costs_v2.csv`` (72_, plan = "balanced" == the
runner's stage 2), whose per-cell routing cost sums back to
``tab_costs_v2.cost_stage2_eur`` by construction (72_ gate 1).  The
baseline cell cost is the (P, theta) = (0, 0) row.

Two rules from the Kompendium that shape what is drawn:

* §38.2(a) -- a per-area *cost* decomposition is exact only at theta = 1
  (express component hub-bundled, zero at theta = 1).  The saving map and
  the penalty curve are therefore theta = 1 only; the frequency and wait
  maps are not cost decompositions and may vary theta.
* 72_ -- the operator lens is hub-, not cell-attributable.  Per-area
  savings are ROUTING-LENS costs of the operator-polished plan and every
  panel says so; the system value in a panel title is the same lens/plan
  (``routing_saving_plan2_pct`` of the grid table, asserted equal to the
  cell sum).

Frequency per area = parcel-weighted median of the LSPs' chosen schedule
sizes (an actual size, never a .5 median).  Wait per area = parcel-weighted
mean of the chosen schedules' average wait, i.e. the wait of a parcel that
IS held (theta = 1: every parcel; theta < 1: the willing share).

Gates (fail loud): baseline rows plan-invariant and all-daily; per-P cell
sum == grid routing saving (1e-6 pp); every polygon maps to a model unit;
settlement type present for every unit; theta = 0 panel all-daily.

Outputs: results/revision_2026_08_v6/figures/*.{pdf,png} and one CSV per
figure in results/revision_2026_08_v6/tables/.
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
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

import _data as D  # noqa: E402  geometry + raumtyp loaders only
import _plots as PL  # noqa: E402
import _style as S  # noqa: E402  -- the ONLY colour source
from _figs_tables_v2 import PLAN1, PLAN2, LENS_ROUTING  # noqa: E402

REV = ROOT / "results" / "revision_2026_08_v6"
CELLS_CSV = REV / "tables" / "tab_per_cell_costs_v2.csv"
GRID_CSV = REV / "tables" / "tab_grid_full_v2.csv"
FIG_DIR = REV / "figures"
TAB_DIR = REV / "tables"

PLAN = "balanced"            # 72_'s name for the runner's stage-2 plan
P_REF = 0.25
THETA_PANELS = (0.0, 0.3, 0.6, 1.0)
P_PANELS = (0.0, 0.25, 0.5, 0.75, 1.0, 2.0)
FREQ_SIZES = (2, 3, 4, 5, 6)
# A per-cell pool share is a parcel-proportional ATTRIBUTION (72_), so a cell
# can come out a few tenths of a percent below zero without any real loss.
# Only a material negative (< -NOISE_PP pp) earns the diverging ramp;
# attribution noise is clipped at zero and disclosed in the note.
NOISE_PP = 0.5
RT_ORDER = ("urban", "suburban", "rural")
RT_MARKER = {"urban": "o", "suburban": "s", "rural": "^"}
DECL_PLAN = f"{PLAN2}"
DECL_COST = f"{LENS_ROUTING} · {PLAN2}"
FOOT_KW = dict(ha="center", va="top", fontsize=7.6, color=S.INK_SOFT)


# ─────────────────────────────────────────────────────────────────────────
# data
# ─────────────────────────────────────────────────────────────────────────
def load_cells() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(plan-2 cells for every (P, theta), baseline cells) with zero-padded plz."""
    d = pd.read_csv(CELLS_CSV, dtype={"plz": str})
    d["plz"] = d["plz"].str.zfill(5)
    base = d[(d.penalty == 0) & (d.share_willing == 0)]
    b1 = base[base.plan == "stage1"].set_index(["provider", "plz"]).sort_index()
    b2 = base[base.plan == PLAN].set_index(["provider", "plz"]).sort_index()
    assert b1.index.equals(b2.index) and np.allclose(b1.cell_cost_eur, b2.cell_cost_eur), \
        "G: baseline cells differ between plans"
    assert (b2.mean_days == 6).all(), "G: baseline is not all-daily"
    cells = d[d.plan == PLAN].copy()
    cells.attrs["plan"] = PLAN
    return cells, b2.reset_index()[["provider", "plz", "cell_cost_eur",
                                     "cell_parcels_week"]].rename(
        columns={"cell_cost_eur": "base_cost_eur", "cell_parcels_week": "base_parcels"})


def load_stage1_cells() -> pd.DataFrame:
    """The routing-optimal (stage-1) cells, for the plan-contrast variants."""
    d = pd.read_csv(CELLS_CSV, dtype={"plz": str})
    d["plz"] = d["plz"].str.zfill(5)
    c = d[d.plan == "stage1"].copy()
    c.attrs["plan"] = "stage1"
    return c


def at(cells: pd.DataFrame, P: float, th: float) -> pd.DataFrame:
    sub = cells[np.isclose(cells.penalty, P) & np.isclose(cells.share_willing, th)]
    assert len(sub) == 312, f"G: {len(sub)} cells at P={P}, theta={th}"
    return sub


def wmedian(values: np.ndarray, weights: np.ndarray) -> float:
    o = np.argsort(values)
    cw = np.cumsum(weights[o])
    return float(values[o][np.searchsorted(cw, 0.5 * cw[-1])])


def view(units) -> "pd.DataFrame":
    g = D.load_plz_geometry()
    ids = {str(x).zfill(5) for x in units}
    v = D.clip_to_scope(g, ids).copy()
    v["unit"] = np.where(v.cluster_id.isin(ids), v.cluster_id, v.plz)
    bad = v[~v.unit.isin(ids)]
    assert bad.empty, f"G: polygons without model unit: {sorted(bad.plz.unique())}"
    return v


def paint(ax, v, vals: pd.DataFrame, col: str, **kw):
    m = v.merge(vals, on="unit", how="left")
    assert m[col].notna().all(), f"G: {int(m[col].isna().sum())} polygons without {col}"
    PL.choropleth(ax, m, col, style="paper", **kw)


#: Matplotlib stamps a /CreationDate into every PDF, so two renders of
#: identical content get different md5s. 70_'s manifest and gate G7 are
#: md5-based (Kompendium §38.8), so this mirrors 70_'s own suppression
#: exactly: a re-render of unchanged inputs must be byte-identical, not
#: merely content-identical.
_PDF_META = {"CreationDate": None}
_PNG_META = {"Software": None}


def save(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext, meta in (("pdf", _PDF_META), ("png", _PNG_META)):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", bbox_inches="tight", metadata=meta)
        print(f"  saved {(FIG_DIR / f'{stem}.{ext}').relative_to(ROOT)}")
    plt.close(fig)


def write(tab: pd.DataFrame, name: str) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_csv(TAB_DIR / name, index=False)
    print(f"  wrote {(TAB_DIR / name).relative_to(ROOT)}")


# ─────────────────────────────────────────────────────────────────────────
# figures
# ─────────────────────────────────────────────────────────────────────────
def freq_table(cells: pd.DataFrame, P: float, thetas) -> pd.DataFrame:
    """Per-area parcel-weighted median delivery frequency, one row per (theta, plz).

    Pure data-shaping (no plotting, no file I/O): the frequency an area is
    painted with is the parcel-weighted median of its LSPs' chosen schedule
    sizes -- an actual size, never a .5 median (``wmedian``).  ``at()``
    enforces the full cell universe per (P, theta); the caller picks the
    thetas.  Gate: every theta=0 row must be all-daily (freq == 6).
    """
    rows = []
    for th in thetas:
        sub = at(cells, P, th)
        for plz, g in sub.groupby("plz"):
            rows.append(dict(penalty=P, share_willing=th, unit=plz,
                             n_lsp=len(g),
                             freq=wmedian(g.mean_days.to_numpy(),
                                          g.cell_parcels_week.to_numpy())))
    tab = pd.DataFrame(rows)
    assert (tab[tab.share_willing == 0].freq == 6).all(), "G: theta=0 not all-daily"
    return tab


def fig_freq(cells, v, P: float = P_REF) -> None:
    plan = cells.attrs.get("plan", PLAN)
    decl = PLAN1 if plan == "stage1" else PLAN2
    tag = (("" if np.isclose(P, P_REF) else f"_P{P:g}".replace(".", ""))
           + ("_routing" if plan == "stage1" else ""))
    tab = freq_table(cells, P, THETA_PANELS)
    write(tab, f"tab_map_freq_theta{tag}_v2.csv")

    S.apply("paper")
    cmap = ListedColormap([S.FREQ[s] for s in FREQ_SIZES])
    norm = BoundaryNorm([s - 0.5 for s in FREQ_SIZES] + [FREQ_SIZES[-1] + 0.5], cmap.N)
    n = len(THETA_PANELS)
    fig, axes = plt.subplots(1, n, figsize=(7.0, 2.35))
    for ax, th in zip(axes, THETA_PANELS):
        paint(ax, v, tab[np.isclose(tab.share_willing, th)][["unit", "freq"]],
              "freq", cmap=cmap, norm=norm, title=rf"$\theta = {th * 100:.0f}\,\%$")
    handles = [Patch(facecolor=S.FREQ[s], label=f"{s} d/wk") for s in FREQ_SIZES]
    fig.legend(handles=handles, title="Delivery frequency (parcel-weighted median over LSPs)",
               title_fontsize=8, fontsize=8, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=1.4,
               columnspacing=1.4)
    fig.text(0.5, -0.12, rf"{decl}; $P = {P:g}$ €/p/d. "
             "Merged postal-code clusters carry their cluster's value.", **FOOT_KW)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, f"supp_map_freq_theta{tag}_v2")


def saving_table(cells: pd.DataFrame, base: pd.DataFrame, p_values,
                  grid: pd.DataFrame) -> pd.DataFrame:
    """Per-area cost saving at theta=1, one row per (P, unit).

    Pure data-shaping (no plotting, no file I/O): euro-weighted saving over
    the LSPs serving an area, gated per P against the grid's own
    ``routing_saving_plan2_pct`` (1e-6 pp, 72_'s identity).
    """
    rows = []
    for P in p_values:
        sub = at(cells, P, 1.0).merge(base, on=["provider", "plz"])
        g = sub.groupby("plz", as_index=False).agg(
            base_eur=("base_cost_eur", "sum"), plan_eur=("cell_cost_eur", "sum"),
            n_lsp=("provider", "size"))
        g["saving_pct"] = (1 - g.plan_eur / g.base_eur) * 100
        sys_cells = (1 - sub.cell_cost_eur.sum() / sub.base_cost_eur.sum()) * 100
        sys_grid = float(grid[np.isclose(grid.penalty, P) & np.isclose(grid.share_willing, 1.0)]
                         .routing_saving_plan2_pct.iloc[0])
        assert abs(sys_cells - sys_grid) < 1e-6, f"G: cell sum {sys_cells} != grid {sys_grid} at P={P}"
        g.insert(0, "penalty", P)
        g["system_saving_pct"] = sys_grid
        rows.append(g)
    return pd.concat(rows).rename(columns={"plz": "unit"})


def saving_draw_range(tab: pd.DataFrame, noise_pp: float = NOISE_PP):
    """The clip-to-0 / diverging-scale decision for the saving choropleth.

    A per-cell pool-share attribution can read a few tenths of a percent
    below zero without any real loss (72_).  Only a material negative
    (< -``noise_pp`` pp) earns the diverging ramp; otherwise the noise is
    clipped to zero and disclosed in a footnote.  Pure decision logic (no
    matplotlib objects): returns ``(vmin, vmax, diverging, note, drawn)``
    where ``drawn`` is ``tab.saving_pct`` clipped at ``vmin``.
    """
    vmax = float(np.ceil(tab.saving_pct.max() / 5) * 5)
    vmin_raw = float(tab.saving_pct.min())
    if vmin_raw < -noise_pp:
        vmin = float(np.floor(vmin_raw / 5) * 5)
        diverging, note = True, ""
    else:
        vmin = 0.0
        diverging = False
        neg = tab[tab.saving_pct < 0]
        note = ("" if neg.empty else
                f" {len(neg)} area(s) at {vmin_raw:.1f} % (pool attribution) shown as 0.")
    drawn = tab.saving_pct.clip(lower=vmin)
    return vmin, vmax, diverging, note, drawn


def fig_saving(cells, base, v, grid) -> None:
    tab = saving_table(cells, base, P_PANELS, grid)
    write(tab, "tab_map_saving_P_v2.csv")
    print(f"  per-area saving range {tab.saving_pct.min():.1f} .. {tab.saving_pct.max():.1f} %")

    vmin, vmax, diverging, note, drawn = saving_draw_range(tab)
    tab["saving_pct_drawn"] = drawn
    if diverging:
        norm, cmap = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax), S.CMAP_CHANGE
    else:
        norm, cmap = Normalize(vmin=vmin, vmax=vmax), S.CMAP_SAVING_MAP

    S.apply("paper")
    ncol, nrow = 3, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 4.3), layout="constrained")
    for ax, P in zip(axes.ravel(), P_PANELS):
        sub = tab[np.isclose(tab.penalty, P)]
        paint(ax, v, sub[["unit", "saving_pct_drawn"]], "saving_pct_drawn", cmap=cmap,
              norm=norm, title=rf"$P = {P:g}$ €/p/d" "\n"
                               f"system {sub.system_saving_pct.iloc[0]:.1f} %")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    cb.set_label("Cost saving per area [%]", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.text(0.5, -0.02, rf"{DECL_COST}; $\theta = 1$. Area saving = euro-weighted over the "
             "LSPs serving it;\npanel titles give the system value of the same plan and lens."
             + note, **FOOT_KW)
    save(fig, "supp_map_saving_P_v2")


def fig_wait(cells, v) -> None:
    rows = []
    for th in THETA_PANELS:
        sub = at(cells, P_REF, th)
        g = sub.groupby("plz", as_index=False).apply(
            lambda x: pd.Series(dict(
                wait_d=float(np.average(x.wait_days, weights=x.cell_parcels_week)),
                n_lsp=len(x))), include_groups=False)
        g.insert(0, "share_willing", th)
        g.insert(0, "penalty", P_REF)
        rows.append(g)
    tab = pd.concat(rows).rename(columns={"plz": "unit"})
    assert (tab[tab.share_willing == 0].wait_d == 0).all(), "G: theta=0 has wait"
    write(tab, "tab_map_wait_theta_v2.csv")

    S.apply("paper")
    vmax = float(np.ceil(tab.wait_d.max() * 10) / 10)
    norm = Normalize(vmin=0.0, vmax=vmax)
    n = len(THETA_PANELS)
    fig, axes = plt.subplots(1, n, figsize=(7.0, 2.35), layout="constrained")
    for ax, th in zip(axes, THETA_PANELS):
        paint(ax, v, tab[np.isclose(tab.share_willing, th)][["unit", "wait_d"]],
              "wait_d", cmap=S.CMAP_WAIT, norm=norm, title=rf"$\theta = {th * 100:.0f}\,\%$")
    sm = plt.cm.ScalarMappable(cmap=S.CMAP_WAIT, norm=norm)
    cb = fig.colorbar(sm, ax=list(axes), fraction=0.03, pad=0.02)
    cb.set_label("Added wait of a held parcel [d]", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.text(0.5, -0.04, rf"{DECL_PLAN}; $P = {P_REF:g}$ €/p/d. Parcel-weighted over the LSPs "
             r"serving the area; held = the willing share $\theta$ (all parcels at $\theta = 1$).",
             **FOOT_KW)
    save(fig, "supp_map_wait_theta_v2")


def penalty_raumtyp_table(cells: pd.DataFrame, base: pd.DataFrame,
                           grid: pd.DataFrame, raumtyp: pd.DataFrame) -> pd.DataFrame:
    """Euro-weighted cost saving by settlement type (+ a system row), theta=1.

    Pure data-shaping (no plotting, no file I/O). ``raumtyp`` is a
    (plz, raumtyp_3) lookup, already zero-padded; every cell must resolve to
    one (fail loud otherwise).  Gated per P against the grid's own
    ``routing_saving_plan2_pct`` (1e-6 pp, 72_'s identity), which becomes
    the appended "system" row.
    """
    pens = sorted(cells.penalty.unique())
    rows = []
    for P in pens:
        sub = (at(cells, P, 1.0).merge(base, on=["provider", "plz"])
               .merge(raumtyp, on="plz", how="left"))
        assert sub.raumtyp_3.notna().all(), f"G: settlement type missing at P={P}"
        g = sub.groupby("raumtyp_3", as_index=False).agg(
            base_eur=("base_cost_eur", "sum"), plan_eur=("cell_cost_eur", "sum"),
            n_cells=("plz", "size"))
        g["saving_pct"] = (1 - g.plan_eur / g.base_eur) * 100
        sysv = float(grid[np.isclose(grid.penalty, P) & np.isclose(grid.share_willing, 1.0)]
                     .routing_saving_plan2_pct.iloc[0])
        assert abs((1 - sub.cell_cost_eur.sum() / sub.base_cost_eur.sum()) * 100 - sysv) < 1e-6, (
            f"G: cell sum {(1 - sub.cell_cost_eur.sum() / sub.base_cost_eur.sum()) * 100} "
            f"!= grid {sysv} at P={P}")
        g = pd.concat([g, pd.DataFrame([dict(raumtyp_3="system", base_eur=sub.base_cost_eur.sum(),
                                             plan_eur=sub.cell_cost_eur.sum(), n_cells=len(sub),
                                             saving_pct=sysv)])])
        g.insert(0, "penalty", P)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def fig_penalty(cells, base, grid) -> None:
    rt = D.load_raumtyp()[["plz", "raumtyp_3"]]
    rt["plz"] = rt["plz"].str.zfill(5)
    pens = sorted(cells.penalty.unique())
    tab = penalty_raumtyp_table(cells, base, grid, rt)
    write(tab, "tab_penalty_raumtyp_v2.csv")

    S.apply("paper")
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    for r in RT_ORDER:
        g = tab[tab.raumtyp_3 == r].sort_values("penalty")
        ax.plot(g.penalty, g.saving_pct, ls="-", marker=RT_MARKER[r], ms=4.2, lw=1.5,
                color=S.RAUMTYP[r], label=r.capitalize(), zorder=3)
    g = tab[tab.raumtyp_3 == "system"].sort_values("penalty")
    ax.plot(g.penalty, g.saving_pct, ls="--", marker="D", ms=3.4, lw=1.3, color=S.INK,
            label="System", zorder=4)
    ax.set_xscale("symlog", linthresh=0.25)
    ax.set_xticks(pens)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(r"Service penalty $P$ [€/parcel/day]")
    ax.set_ylabel("Cost saving [%]")
    ax.axhline(0, color=S.GRID, lw=0.8, zorder=1)
    ax.grid(axis="y", color=S.GRID, lw=0.5, alpha=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=9, handlelength=2.4)
    # v6 never reaches exactly zero (the polish keeps a little pooling), so
    # report the residual instead of a zero-crossing that does not exist.
    tail = tab[(tab.raumtyp_3 != "system") & (tab.penalty >= 5)]
    fig.text(0.5, -0.03, rf"{DECL_COST}; $\theta = 1$. Euro-weighted saving within each "
             rf"settlement type (plz_raumtyp.csv). From $P = 5$ €/p/d on, every type keeps "
             rf"$\leq {tail.saving_pct.max():.1f}$ % saving.", **FOOT_KW)
    fig.tight_layout()
    save(fig, "supp_penalty_raumtyp_v2")


def main() -> None:
    cells, base = load_cells()
    grid = pd.read_csv(GRID_CSV)
    v = view(base.plz.unique())
    print(f"{len(v)} polygons, {base.plz.nunique()} model units, {len(base)} cells")
    fig_freq(cells, v)
    fig_freq(cells, v, P=0.0)
    fig_freq(load_stage1_cells(), v, P=0.0)
    fig_saving(cells, base, v, grid)
    fig_wait(cells, v)
    fig_penalty(cells, base, grid)


if __name__ == "__main__":
    main()
