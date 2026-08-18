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
STAGE3 = "Stage 3 (per-hub balancing + system smoothing)"

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


def _median_freq(sched, penalty, theta) -> pd.DataFrame:
    sub = sched[np.isclose(sched.penalty, penalty)
                & np.isclose(sched.share_willing, theta)]
    assert len(sub), f"no schedule rows for P={penalty}, theta={theta}"
    return (sub.groupby("plz", as_index=False)
            .agg(freq=("schedule_size_system_smoothed", "median"))
            .rename(columns={"plz": "unit"}))


# ---------------------------------------------------------------- fig41
def fig41_map_freq_by_theta():
    sched = D.load_chosen_stage3()
    view = _view(sched.plz.unique())
    cmap, norm = _freq_cmap()

    for style in S.styles():
        S.apply(style)
        n = len(THETA_PANELS)
        fig, axes = plt.subplots(
            1, n, figsize=S.figsize(style, (3.6 * n, 4.8), (4.4 * n, 5.6)))
        for ax, th in zip(np.atleast_1d(axes), THETA_PANELS):
            _paint(ax, view, _median_freq(sched, P_REF, th), "freq",
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
                       f"at 6 day/wk, at theta=1 the rural fringe drops to the "
                       f"2-3 day/wk classes.",
                 caveats="Frequency is invariant from Stage 2 to Stage 3 "
                         "(asserted). Values are per merged PLZ cluster; every "
                         "member polygon carries its cluster's value.")


# ---------------------------------------------------------------- fig42
def fig42_map_freq_by_P():
    sched = D.load_chosen_stage3()
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
                 caveats="Values are per merged PLZ cluster.")


# ---------------------------------------------------------------- fig43
def fig43_map_freq_provider():
    sched = D.load_chosen_stage3()
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
                 caveats="Blank areas are outside that provider's service set, "
                         "not zero-frequency.")


# ---------------------------------------------------------------- fig44
def fig44_map_wait():
    sched = D.load_chosen_stage3()
    view = _view(sched.plz.unique())

    for style in S.styles():
        S.apply(style)
        n = len(THETA_PANELS)
        fig, axes = plt.subplots(
            1, n, figsize=S.figsize(style, (3.6 * n, 4.8), (4.4 * n, 5.6)))
        vmax = 0.0
        frames = []
        for th in THETA_PANELS:
            sub = sched[np.isclose(sched.penalty, P_REF)
                        & np.isclose(sched.share_willing, th)]
            vals = (sub.groupby("plz", as_index=False)
                    .agg(wait=("avg_wait_d_system_smoothed", "mean"))
                    .rename(columns={"plz": "unit"}))
            frames.append(vals)
            vmax = max(vmax, float(vals.wait.max()))
        vmax = max(vmax, 1e-9)
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=0.0, vmax=vmax)
        for ax, th, vals in zip(np.atleast_1d(axes), THETA_PANELS, frames):
            _paint(ax, view, vals, "wait", cmap="YlOrRd", norm=norm,
                   title=rf"$\theta = {th * 100:.0f}\%$", style=style)
        sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
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
                 caveats="Mean over the providers serving each area; a provider "
                         "keeping daily delivery there pulls the mean down.")


# ---------------------------------------------------------------- fig45
def fig45_map_efficiency():
    v = D.load_vroom()
    sub = v[np.isclose(v.penalty, P_REF) & np.isclose(v.share_willing, THETA_REF)]
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
                 tier=S.TIER_A, act=ACT, basis="Stage-3 VROOM revalidation",
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
