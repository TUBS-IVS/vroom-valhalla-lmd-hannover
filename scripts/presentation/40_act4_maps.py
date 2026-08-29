"""Act 4 -- spatial results, Stage-3 basis.

Figures
  fig41_map_freq_by_theta   median delivery frequency per area, panels over theta
  fig42_map_freq_by_P       median delivery frequency per area, panels over P
  fig43_map_freq_provider   delivery frequency per area, one panel per provider
  fig44_map_wait            mean added customer wait per area
  fig45_map_efficiency      VROOM cost/parcel, km/parcel and load factor per area

Two things every map here has to get right, and which the retired
_fig_maps_per_share_P04.py got wrong:

1. Optimisation runs on *merged PLZ clusters*, not single postal codes. Each
   polygon must therefore be painted with the value of the cluster it belongs
   to, resolved through `_unit()` below. All 48 model units resolve to at least
   one polygon (audited: 36 cluster heads + 12 standalone areas).
2. The penalty a panel is drawn from must match the penalty its title claims.
   The retired script selected P=0.5 while titling itself P=0.4. Here the value
   flows from a single constant into both selection and title.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

import _data as D
import _plots as P
import _style as S

ACT = "4 - Spatial"
# The plan these maps are drawn on. `scripts/revision/76_maps_v2.py` draws the
# paper's spatial supplement from the same grid on the same plan under the
# same aggregation rule (compendium 40.23b), and `_gate_freq` / `_gate_wait`
# below check these panels against its tables cell by cell -- so the deck's
# maps and the paper's supplement agree by construction, not by eye.
PLAN = D.CHOSEN_PLAN_DEFAULT
STAGE3 = D.plan_stamp(PLAN)

# The efficient operating point carried in the paper text.
P_REF = 0.25
THETA_REF = 1.0

THETA_PANELS = [0.0, 0.3, 0.6, 1.0]
P_PANELS = [0.0, 0.25, 0.5, 1.0, 2.0, 10.0]

def _freq_cmap():
    cmap = ListedColormap([D.FREQ_COLOR[s] for s in D.FREQ_SIZES])
    norm = BoundaryNorm([s - 0.5 for s in D.FREQ_SIZES] + [D.FREQ_SIZES[-1] + 0.5],
                        cmap.N)
    return cmap, norm


def _view(model_ids):
    """Polygons of the modelled area, with a `unit` column giving the model
    unit (merged cluster head, or the area itself when unmerged) that each
    polygon's value must be read from."""
    g = D.load_plz_geometry()
    ids = {str(x).zfill(5) for x in model_ids}
    v = D.clip_to_scope(g, ids).copy()
    v["unit"] = np.where(v.cluster_id.isin(ids), v.cluster_id, v.plz)
    unresolved = v[~v.unit.isin(ids)]
    assert unresolved.empty, (
        f"{len(unresolved)} polygon(s) cannot be mapped to a model unit: "
        f"{sorted(unresolved.plz.unique())}")
    return v


def _paint(ax, view, values: pd.DataFrame, col: str, *, cmap, norm,
           title: str, style: str):
    """Merge a per-unit value frame onto the polygons and draw."""
    m = view.merge(values.rename(columns={"unit": "unit"}), on="unit",
                   how="left")
    n_missing = int(m[col].isna().sum())
    if n_missing:
        print(f"    [{title}] {n_missing} polygon(s) without a value")
    P.choropleth(ax, m, col, cmap=cmap, norm=norm, title=title, style=style)
    return m


def _wmedian(values, weights) -> float:
    """Parcel-weighted median -- always one of the values, never a half.

    A model unit is served by up to seven LSPs at different frequencies, and
    the plain median of an even count lands between two classes: on v6's
    operator plan 84 of 4 224 unit-cells come out at x.5, a delivery
    frequency the model cannot produce. Weighting by that LSP's parcels in
    the unit both removes the artefact and answers the question the map is
    asking -- how often is a parcel HERE delivered -- and is the rule
    76_maps_v2.py uses for the paper's version of this figure.
    """
    v = np.asarray(values)
    w = np.asarray(weights, dtype=float)
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])])


def _per_unit(sub, value_col, out_col, how) -> pd.DataFrame:
    """Aggregate one (P, theta) slice from cell choices to model units."""
    rows = []
    for unit, g in sub.groupby("plz"):
        w = g.weekly_parcels.to_numpy(dtype=float)
        v = g[value_col].to_numpy()
        rows.append({"unit": unit,
                     out_col: (_wmedian(v, w) if how == "wmedian"
                               else float(np.average(v, weights=w)))})
    return pd.DataFrame(rows)


def _median_freq(sched, penalty, theta) -> pd.DataFrame:
    sub = sched[np.isclose(sched.penalty, penalty)
                & np.isclose(sched.share_willing, theta)]
    assert len(sub), f"no schedule rows for P={penalty}, theta={theta}"
    if "weekly_parcels" not in sub.columns:            # legacy grid
        return (sub.groupby("plz", as_index=False)
                .agg(freq=("schedule_size_system_smoothed", "median"))
                .rename(columns={"plz": "unit"}))
    out = _per_unit(sub, "schedule_size_system_smoothed", "freq", "wmedian")
    bad = sorted(set(out.freq) - set(float(x) for x in D.FREQ_SIZES))
    assert not bad, (
        f"painted frequencies {bad} are not admissible delivery frequencies "
        f"{D.FREQ_SIZES} -- the aggregation invented a class")
    return out


def _mean_wait(sched, penalty, theta) -> pd.DataFrame:
    sub = sched[np.isclose(sched.penalty, penalty)
                & np.isclose(sched.share_willing, theta)]
    assert len(sub), f"no schedule rows for P={penalty}, theta={theta}"
    if "weekly_parcels" not in sub.columns:            # legacy grid
        return (sub.groupby("plz", as_index=False)
                .agg(wait=("avg_wait_d_system_smoothed", "mean"))
                .rename(columns={"plz": "unit"}))
    return _per_unit(sub, "avg_wait_d_system_smoothed", "wait", "wmean")


def _gate(frames, ref_name, col, ref_col, thetas, penalty) -> None:
    """Check painted panels against 76_maps_v2.py's own table for this grid.

    76_ writes the paper's version of these maps from `tab_per_cell_costs_v2`
    under the same rule and the same plan; the deck's version reads
    `_tab_chosen_v2` instead. Two independent paths from the same grid to the
    same numbers -- so if they agree cell by cell, the port is right, and if
    they do not, one of them is on the wrong plan or the wrong grid.
    """
    ref_path = D.REV / "tables" / ref_name
    if D.SCHEMA != D.SCHEMA_V2 or PLAN != D.PLAN_BALANCED:
        print(f"  [gate] {ref_name} covers the operator plan on a v2 grid "
              f"only; not applicable here")
        return
    if not ref_path.exists():
        raise FileNotFoundError(
            f"{ref_path} is missing; run scripts/revision/76_maps_v2.py so "
            f"these panels can be checked against the paper's own version")
    ref = pd.read_csv(ref_path, dtype={"unit": str})
    D.prov.record(ref_path)
    n = 0
    for th, vals in zip(thetas, frames):
        r = (ref[np.isclose(ref.penalty, penalty)
                 & np.isclose(ref.share_willing, th)][["unit", ref_col]]
             .rename(columns={ref_col: "_ref"}))
        if not len(r):
            continue
        m = vals.merge(r, on="unit", how="outer", indicator=True)
        assert (m._merge == "both").all(), (
            f"{ref_name} and this panel cover different units at theta={th}")
        assert np.allclose(m[col], m._ref, atol=1e-9), (
            f"panel theta={th} disagrees with {ref_name} by up to "
            f"{float((m[col] - m._ref).abs().max()):.3g} -- one of the "
            f"two is on the wrong plan or the wrong grid")
        n += len(m)
    print(f"  [gate] {n} unit values reproduce {ref_name} "
          f"({D.GRID_PLAN_LABEL[PLAN]})")


# ---------------------------------------------------------------- fig41
def fig41_map_freq_by_theta():
    sched = D.load_chosen_stage3(PLAN)
    view = _view(sched.plz.unique())
    cmap, norm = _freq_cmap()
    frames = [_median_freq(sched, P_REF, th) for th in THETA_PANELS]
    _gate(frames, "tab_map_freq_theta_v2.csv", "freq", "freq",
          THETA_PANELS, P_REF)
    dom = frames[THETA_PANELS.index(1.0)].freq.mode().iloc[0]
    assert dom == 3, (
        f"the dominant delivery frequency at P={P_REF:g}, theta=1 is "
        f"{dom:g} d/wk; compendium 40.23b has 3 d/wk on this grid and plan")
    print(f"  dominant frequency at P={P_REF:g}, theta=1: {dom:.0f} day/wk")

    for style in S.styles():
        S.apply(style)
        n = len(THETA_PANELS)
        fig, axes = plt.subplots(
            1, n, figsize=S.figsize(style, (3.6 * n, 4.8), (4.4 * n, 5.6)))
        for ax, th, vals in zip(np.atleast_1d(axes), THETA_PANELS, frames):
            _paint(ax, view, vals, "freq",
                   cmap=cmap, norm=norm,
                   title=rf"$\theta = {th * 100:.0f}\%$", style=style)
        handles = [Patch(facecolor=D.FREQ_COLOR[s], label=f"{s} day/wk")
                   for s in D.FREQ_SIZES]
        fig.legend(handles=handles, title="Median delivery frequency",
                   loc="lower center", ncol=len(handles), frameon=True,
                   framealpha=0.9, edgecolor="0.8", bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(rf"Delivery frequency per area at $P = {P_REF:g}$ €/p/d, "
                     r"by willingness-to-wait share $\theta$")
        fig.tight_layout(rect=[0, 0.07, 1, 0.95])
        S.save(fig, "fig41_map_freq_by_theta", style, S.TIER_A)

    D.prov.write("fig41_map_freq_by_theta",
                 title="Delivery-frequency map over theta",
                 tier=S.TIER_A, act=ACT, basis=STAGE3,
                 claim=f"At P={P_REF:g} consolidation spreads outward from the "
                       f"periphery as theta rises: at theta=0 every area stays "
                       f"at 6 day/wk, at theta=1 the modal area is at "
                       f"{dom:.0f} day/wk and the rural fringe drops to the "
                       f"2-3 day/wk classes.",
                 caveats=f"Frequencies are the {D.GRID_PLAN_LABEL[PLAN]}'s. "
                         f"They are not interchangeable with the routing "
                         f"plan's: v6's stage 2 is frequency-free at "
                         f"theta > 0 (compendium 40.14), so a frequency map "
                         f"has to name its plan. Per unit the value is the "
                         f"parcel-weighted median over the LSPs serving it, "
                         f"checked against 76_maps_v2.py's own table; values "
                         f"are per merged PLZ cluster, so every member "
                         f"polygon carries its cluster's value.")


# ---------------------------------------------------------------- fig42
def fig42_map_freq_by_P():
    sched = D.load_chosen_stage3(PLAN)
    view = _view(sched.plz.unique())
    cmap, norm = _freq_cmap()
    pens = [p for p in P_PANELS if (np.isclose(sched.penalty.values[:, None],
                                               p).any())]

    for style in S.styles():
        S.apply(style)
        ncols = 3
        nrows = (len(pens) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=S.figsize(style, (3.6 * ncols, 4.4 * nrows),
                              (4.4 * ncols, 4.8 * nrows)))
        axes = np.atleast_2d(axes)
        for idx, pen in enumerate(pens):
            r, c = divmod(idx, ncols)
            _paint(axes[r, c], view, _median_freq(sched, pen, THETA_REF),
                   "freq", cmap=cmap, norm=norm,
                   title=rf"$P = {pen:g}$ €/p/d", style=style)
        for idx in range(len(pens), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)
        handles = [Patch(facecolor=D.FREQ_COLOR[s], label=f"{s} day/wk")
                   for s in D.FREQ_SIZES]
        fig.legend(handles=handles, title="Median delivery frequency",
                   loc="lower center", ncol=len(handles), frameon=True,
                   framealpha=0.9, edgecolor="0.8", bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(r"Delivery frequency per area at $\theta = 100\%$, "
                     r"by service penalty $P$")
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        S.save(fig, "fig42_map_freq_by_P", style, S.TIER_A)

    D.prov.write("fig42_map_freq_by_P",
                 title="Delivery-frequency map over the service penalty",
                 tier=S.TIER_A, act=ACT, basis=STAGE3,
                 claim="The service penalty is the policy lever that undoes "
                       "consolidation: raising P from 0 to 10 €/parcel/day "
                       "returns essentially every area to daily delivery.",
                 caveats=f"{D.GRID_PLAN_LABEL[PLAN].capitalize()}; "
                         f"parcel-weighted median over the LSPs serving each "
                         f"unit. Values are per merged PLZ cluster.")


# ---------------------------------------------------------------- fig43
def fig43_map_freq_provider():
    sched = D.load_chosen_stage3(PLAN)
    view = _view(sched.plz.unique())
    cmap, norm = _freq_cmap()
    sub = sched[np.isclose(sched.penalty, P_REF)
                & np.isclose(sched.share_willing, THETA_REF)]

    for style in S.styles():
        S.apply(style)
        ncols = 4
        nrows = 2
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=S.figsize(style, (3.2 * ncols, 4.0 * nrows),
                              (4.0 * ncols, 4.4 * nrows)))
        for idx, provider in enumerate(D.PROVIDERS):
            r, c = divmod(idx, ncols)
            vals = (sub[sub.provider == provider]
                    .groupby("plz", as_index=False)
                    .agg(freq=("schedule_size_system_smoothed", "median"))
                    .rename(columns={"plz": "unit"}))
            # one LSP per unit here, so the median is that LSP's own choice
            assert not (vals.freq % 1).any(), (
                f"{provider}: a single-LSP panel produced a half frequency")
            _paint(axes[r, c], view, vals, "freq", cmap=cmap, norm=norm,
                   title=provider, style=style)
        axes[1, 3].set_visible(False)
        handles = [Patch(facecolor=D.FREQ_COLOR[s], label=f"{s} day/wk")
                   for s in D.FREQ_SIZES]
        fig.legend(handles=handles, title="Delivery frequency",
                   loc="lower right", ncol=2, frameon=True, framealpha=0.9,
                   edgecolor="0.8", bbox_to_anchor=(0.97, 0.10))
        fig.suptitle(rf"Delivery frequency per provider at $P = {P_REF:g}$ €/p/d, "
                     rf"$\theta = {THETA_REF * 100:.0f}\%$")
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        S.save(fig, "fig43_map_freq_provider", style, S.TIER_B)

    D.prov.write("fig43_map_freq_provider",
                 title="Delivery frequency per provider, spatial",
                 tier=S.TIER_B, act=ACT, basis=STAGE3,
                 claim="Providers consolidate different parts of the region: "
                       "coverage differs (each serves 37-48 areas) and so does "
                       "the frequency each area is assigned.",
                 caveats=f"{D.GRID_PLAN_LABEL[PLAN].capitalize()}. Blank "
                         f"areas are outside that provider's service set, "
                         f"not zero-frequency.")


# ---------------------------------------------------------------- fig44
def fig44_map_wait():
    sched = D.load_chosen_stage3(PLAN)
    view = _view(sched.plz.unique())

    for style in S.styles():
        S.apply(style)
        n = len(THETA_PANELS)
        fig, axes = plt.subplots(
            1, n, figsize=S.figsize(style, (3.6 * n, 4.8), (4.4 * n, 5.6)))
        vmax = 0.0
        frames = []
        for th in THETA_PANELS:
            vals = _mean_wait(sched, P_REF, th)
            frames.append(vals)
            vmax = max(vmax, float(vals.wait.max()))
        _gate(frames, "tab_map_wait_theta_v2.csv", "wait", "wait_d",
              THETA_PANELS, P_REF)
        vmax = max(vmax, 1e-9)
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=0.0, vmax=vmax)
        for ax, th, vals in zip(np.atleast_1d(axes), THETA_PANELS, frames):
            _paint(ax, view, vals, "wait", cmap=S.CMAP_WAIT, norm=norm,
                   title=rf"$\theta = {th * 100:.0f}\%$", style=style)
        sm = plt.cm.ScalarMappable(cmap=S.CMAP_WAIT, norm=norm)
        cb = fig.colorbar(sm, ax=list(np.atleast_1d(axes)), fraction=0.025,
                          pad=0.02)
        cb.set_label("Mean added wait [d]")
        fig.suptitle(rf"Mean additional customer wait per area at "
                     rf"$P = {P_REF:g}$ €/p/d")
        S.save(fig, "fig44_map_wait", style, S.TIER_A)

    D.prov.write("fig44_map_wait", title="Added-wait map over theta",
                 tier=S.TIER_A, act=ACT, basis=STAGE3,
                 claim=f"Service degradation is spatially concentrated: the "
                       f"areas that consolidate carry up to {vmax:.2f} d of "
                       f"added wait while the dense core stays near zero.",
                 caveats=f"{D.GRID_PLAN_LABEL[PLAN].capitalize()}. This is the "
                         f"wait of a parcel that IS held: the parcel-weighted "
                         f"mean over the LSPs serving each area, so an LSP "
                         f"keeping daily delivery there pulls it down. At "
                         f"theta < 1 only the willing share waits at all, so "
                         f"the system average is lower again (fig32).")


# ---------------------------------------------------------------- fig45
def fig45_map_efficiency():
    # PLAN and THETA_REF are filters, not decoration: the v2 validation solves
    # (P = 0.25, theta = 1) once per plan, so summing per cell without the
    # plan filter would add each area's balanced and stage-1 tours together
    # and roughly double every cost and distance on the map.
    v = D.load_vroom(plan=PLAN, theta=THETA_REF)
    sub = v[np.isclose(v.penalty, P_REF)]
    assert len(sub), f"no VROOM rows at P={P_REF}, theta={THETA_REF}"


    per = (sub.groupby("plz", as_index=False)
           .agg(cost=("vroom_cost_eur", "sum"),
                km=("vroom_distance_km", "sum"),
                routes=("vroom_n_routes", "sum"),
                parcels=("vroom_n_parcels", "sum")))
    per["eur_per_parcel"] = per.cost / per.parcels
    per["km_per_parcel"] = per.km / per.parcels
    per["parcels_per_route"] = per.parcels / per.routes
    per = per.rename(columns={"plz": "unit"})
    view = _view(per.unit.unique())

    panels = [
        ("eur_per_parcel", "Cost per parcel [€]", "viridis_r", "{:.2f}"),
        ("km_per_parcel", "Distance per parcel [km]", "plasma_r", "{:.3f}"),
        ("parcels_per_route", "Parcels per route", "cividis", "{:.0f}"),
    ]
    print("  " + "; ".join(
        f"{c}: {per[c].min():.3f}-{per[c].max():.3f}" for c, *_ in panels))

    for style in S.styles():
        S.apply(style)
        fig, axes = plt.subplots(
            1, 3, figsize=S.figsize(style, (12.0, 5.0), (16.0, 6.0)))
        from matplotlib.colors import Normalize
        for ax, (col, label, cmap, _fmt) in zip(axes, panels):
            norm = Normalize(vmin=float(per[col].min()),
                             vmax=float(per[col].max()))
            _paint(ax, view, per[["unit", col]], col, cmap=cmap, norm=norm,
                   title=label, style=style)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle(rf"Operational efficiency per area, real VROOM routes at "
                     rf"$P = {P_REF:g}$ €/p/d, $\theta = 100\%$")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        S.save(fig, "fig45_map_efficiency", style, S.TIER_A)

    D.prov.write("fig45_map_efficiency",
                 title="Per-area operational efficiency from real routes",
                 tier=S.TIER_A, act=ACT,
                 basis=f"VROOM validation of {D.VAL.parent.name} "
                       f"({D.GRID_PLAN_LABEL[PLAN]}), theta=1",
                 claim=f"Cost per parcel spans "
                       f"{per.eur_per_parcel.min():.2f}-"
                       f"{per.eur_per_parcel.max():.2f} € and distance per "
                       f"parcel {per.km_per_parcel.min():.3f}-"
                       f"{per.km_per_parcel.max():.3f} km across the region, "
                       f"so the same schedule policy has very different "
                       f"operational consequences by location.",
                 caveats="These are absolute efficiency levels on the Stage-3 "
                         "schedules, not savings: no per-area baseline route "
                         "solve exists on this demand allocation.")


def main():
    for fn in (fig41_map_freq_by_theta, fig42_map_freq_by_P,
               fig43_map_freq_provider, fig44_map_wait, fig45_map_efficiency):
        print(f"\n=== {fn.__name__} ===")
        fn()


if __name__ == "__main__":
    main()
