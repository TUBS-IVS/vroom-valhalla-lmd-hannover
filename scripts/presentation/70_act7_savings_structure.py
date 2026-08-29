"""Act 7 -- where consolidation pays off, per area. theta = 1.

Built on `00_recompute_per_plz_costs.py`'s per-area table: the chosen plan's
cost and the daily-delivery reference cost for every model unit, reconciled
cell by cell against the provider aggregates by that script's own gates.

On a v2 grid that table comes from 72_'s `tab_per_cell_costs_v2.csv` and a
euro is the cell's full ROUTING cost under the realistic-tour rule -- its own
tour plus its parcel-proportional share of every pooled tour it rides on --
at the OPERATOR-polished plan (compendium 40.23b, the same plan and lens
`scripts/revision/76_maps_v2.py` draws the paper's spatial supplement on).
The operator lens is not offered per area: it is hub-, not cell-attributable.
`_gate_system()` below checks the euro-weighted system saving of every
penalty against the grid's own table before any figure is drawn.

Figures
  fig71_map_saving        saving per area, panels over the service penalty
  fig72_raumtyp           saving by settlement type, distribution and totals
  fig73_threshold_demand  saving against weekly demand -- when batching pays
  fig74_regime_map        saving over (demand density, hub distance)
  fig75_breakeven         saving against P by settlement type
  fig76_provider_raumtyp  provider x settlement-type saving matrix
  fig77_drivers           which structural features explain the saving

One trap this act has to keep straight. The unweighted mean saving *per area*
is not the system saving: small peripheral units save proportionally far more
than the dense units that carry most of the parcels. Every distribution here
is labelled per-area, and every total is computed from absolute euros, never
by averaging percentages.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

import _data as D
import _plots as P
import _style as S

ACT = "7 - Where it pays off"
PLAN = D.CHOSEN_PLAN_DEFAULT
BASIS = (f"{D.plan_stamp(PLAN)}: per-area cost against that grid's own "
         f"daily-delivery reference, routing lens, theta=1 (the express "
         f"component is exactly 0 there, so the decomposition is exact)")

P_REF = 0.25
P_PANELS = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0]
RT_ORDER = ["urban", "suburban", "rural"]
RT_COLOR = D.RT_COLOR   # paper fig 6 settlement palette, via _data


_GATED = []


def _load():
    d = D.add_raumtyp(D.load_per_plz(PLAN))
    assert d.raumtyp_3.notna().all(), "settlement type missing for some areas"
    if not _GATED:
        _gate_system(d)
        _GATED.append(True)
    return d


def _gate_system(d) -> None:
    """The euro-weighted system saving must be the grid's, per penalty.

    This is the one number every panel title on fig71 and every line on fig75
    is built from, and it is the number a wrong plan or a wrong grid moves
    first: v6's routing-optimal plan saves 22.6 % at P = 0 where its
    operator-polished plan saves 20.0 %, and both look entirely plausible on
    a map. So the whole series is checked against
    `tables/tab_map_saving_P_v2.csv`, which 76_maps_v2.py derives
    independently from the same grid for the paper's supplement.
    """
    got = _system_saving(d).set_index("penalty").saving_pct
    ref_path = D.REV / "tables" / "tab_map_saving_P_v2.csv"
    if D.SCHEMA != D.SCHEMA_V2 or PLAN != D.PLAN_BALANCED:
        print(f"  [gate] {ref_path.name} covers the operator plan of a v2 "
              f"grid only; system saving not cross-checked")
        return
    if not ref_path.exists():
        raise FileNotFoundError(
            f"{ref_path} is missing; run scripts/revision/76_maps_v2.py so "
            f"this act's system savings can be checked against the paper's "
            f"own version")
    ref = pd.read_csv(ref_path)
    D.prov.record(ref_path)
    want = (ref.groupby("penalty").system_saving_pct.first())
    common = sorted(set(want.index) & set(got.index))
    assert common, "no penalty is present in both tables"
    for pen in common:
        assert abs(got[pen] - want[pen]) < 1e-6, (
            f"system saving at P={pen:g} is {got[pen]:.4f} % here and "
            f"{want[pen]:.4f} % in {ref_path.name} -- one of the two is on "
            f"the wrong plan or the wrong grid")
    print(f"  [gate] system saving at P="
          + ", ".join(f"{pen:g}: {got[pen]:.1f}%" for pen in common)
          + f" reproduces {ref_path.name} ({D.GRID_PLAN_LABEL[PLAN]})")


def _system_saving(d: pd.DataFrame) -> pd.DataFrame:
    """Euro-weighted system saving per penalty -- the honest aggregate."""
    g = (d.groupby("penalty", as_index=False)
         .agg(base=("dd_cost_baseline_eur", "sum"),
              s3=("dd_cost_stage3_eur", "sum")))
    g["saving_pct"] = (1 - g.s3 / g.base) * 100
    return g


def _view(units):
    g = D.load_plz_geometry()
    ids = {str(x).zfill(5) for x in units}
    v = D.clip_to_scope(g, ids).copy()
    v["unit"] = np.where(v.cluster_id.isin(ids), v.cluster_id, v.plz)
    return v


# ---------------------------------------------------------------- fig71
def fig71_map_saving():
    d = _load()
    view = _view(d.plz.unique())
    pens = [p for p in P_PANELS if np.isclose(d.penalty.values[:, None], p).any()]

    # Area-level saving aggregated over providers in euros, not by averaging
    # percentages, so a unit served by many providers is weighted correctly.
    agg = (d.groupby(["penalty", "plz"], as_index=False)
           .agg(base=("dd_cost_baseline_eur", "sum"),
                s3=("dd_cost_stage3_eur", "sum")))
    agg["saving_pct"] = (1 - agg.s3 / agg.base) * 100
    agg = agg.rename(columns={"plz": "unit"})
    vmax = float(np.ceil(agg.saving_pct.max() / 5) * 5)
    vmin = float(min(0.0, np.floor(agg.saving_pct.min() / 5) * 5))
    norm = (TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) if vmin < 0
            else Normalize(vmin=0, vmax=vmax))
    # Signed saving diverges around zero; unsigned is plain magnitude.
    # Same convention as the paper: signed quantities diverge around
    # zero on RdBu_r, unsigned magnitude uses the saving ramp.
    cmap = S.CMAP_CHANGE if vmin < 0 else S.CMAP_SAVING
    print(f"  per-area saving range {agg.saving_pct.min():.1f}% .. "
          f"{agg.saving_pct.max():.1f}%")

    sysg = _system_saving(d)
    for style in S.styles():
        S.apply(style)
        ncols = 3
        nrows = (len(pens) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, layout="constrained",
            figsize=S.figsize(style, (3.2 * ncols, 3.4 * nrows),
                              (3.8 * ncols, 3.6 * nrows)))
        axes = np.atleast_2d(axes)
        for idx, pen in enumerate(pens):
            r, c = divmod(idx, ncols)
            sub = agg[np.isclose(agg.penalty, pen)][["unit", "saving_pct"]]
            m = view.merge(sub, on="unit", how="left")
            sysv = float(sysg[np.isclose(sysg.penalty, pen)].saving_pct.iloc[0])
            P.choropleth(axes[r, c], m, "saving_pct", cmap=cmap, norm=norm,
                         title=rf"$P = {pen:g}$" "\n"
                               rf"system {sysv:.1f}%",
                         style=style)
        for idx in range(len(pens), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.022, pad=0.02)
        cb.set_label("Cost saving per area [%]")
        fig.suptitle(r"Where consolidation pays off, $\theta = 100\%$")
        S.save(fig, "fig71_map_saving", style, S.TIER_A)

    D.prov.write("fig71_map_saving", title="Per-area cost saving over the penalty",
                 tier=S.TIER_A, act=ACT, basis=BASIS,
                 claim=f"Savings are strongly peripheral: per-area savings reach "
                       f"{agg.saving_pct.max():.1f}% on the rural fringe while "
                       f"the dense core stays in single digits, and the pattern "
                       f"is erased as P rises.",
                 caveats="Panel titles give the euro-weighted system saving; the "
                         "colours are per-area percentages and average higher "
                         "because small units save proportionally more.")


# ---------------------------------------------------------------- fig72
def fig72_raumtyp():
    d = _load()
    at = d[np.isclose(d.penalty, P_REF)]

    dist = [at[at.raumtyp_3 == rt].saving_pct.values for rt in RT_ORDER]
    tot = (at.groupby("raumtyp_3", as_index=False)
           .agg(base=("dd_cost_baseline_eur", "sum"),
                s3=("dd_cost_stage3_eur", "sum"),
                parcels=("weekly_parcels", "sum"), n=("saving_pct", "size")))
    tot["saving_pct"] = (1 - tot.s3 / tot.base) * 100
    tot["saved_eur"] = tot.base - tot.s3
    tot = tot.set_index("raumtyp_3").reindex(RT_ORDER).reset_index()
    print(tot[["raumtyp_3", "n", "saving_pct", "saved_eur", "parcels"]]
          .round(1).to_string(index=False))

    for style in S.styles():
        S.apply(style)
        fig, axes = plt.subplots(1, 2,
                                 figsize=S.figsize(style, (11.5, 4.6),
                                                   (15.0, 5.8)))
        bp = axes[0].boxplot(dist, patch_artist=True, widths=0.55,
                             medianprops=dict(color="#111111", linewidth=2))
        for patch, rt in zip(bp["boxes"], RT_ORDER):
            patch.set_facecolor(RT_COLOR[rt])
            patch.set_alpha(0.85)
        axes[0].set_xticks(range(1, len(RT_ORDER) + 1))
        axes[0].set_xticklabels([rt.capitalize() for rt in RT_ORDER])
        axes[0].set_ylabel("Cost saving per area [%]")
        axes[0].set_title("(a) Distribution across areas")
        axes[0].grid(alpha=0.25, axis="y")

        x = np.arange(len(tot))
        b = axes[1].bar(x, tot.saved_eur / 1000, 0.55,
                        color=[RT_COLOR[rt] for rt in tot.raumtyp_3])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([rt.capitalize() for rt in tot.raumtyp_3])
        axes[1].set_ylabel("Absolute saving [1000 €/week]")
        axes[1].set_title("(b) Where the money is")
        axes[1].grid(alpha=0.25, axis="y")
        P.bar_labels(axes[1], b, (tot.saved_eur / 1000).values, "{:.0f}",
                     style=style)
        for xi, row in zip(x, tot.itertuples()):
            axes[1].annotate(f"{row.saving_pct:.1f}%", xy=(xi, 0),
                             xytext=(0, 14), ha="center",
                             textcoords="offset points",
                             fontsize=9 if style == "paper" else 13,
                             color="white", fontweight="bold")
        fig.suptitle(rf"Saving by settlement type at $P = {P_REF:g}$ €/p/d, "
                     rf"$\theta = 100\%$")
        P.footnote(fig, "Panel (a) is the unweighted per-area distribution; "
                        "panel (b) is euro-weighted, with the share of that "
                        "type's own baseline printed inside each bar. Rural "
                        "areas lead on both counts — the largest relative "
                        "saving and the largest absolute sum.", style)
        fig.tight_layout(rect=[0, 0.06, 1, 0.93])
        S.save(fig, "fig72_raumtyp", style, S.TIER_A)

    D.prov.write("fig72_raumtyp", title="Saving by settlement type",
                 tier=S.TIER_A, act=ACT, basis=BASIS,
                 claim="; ".join(
                     f"{r.raumtyp_3} {r.saving_pct:.1f}% "
                     f"({r.saved_eur / 1000:.0f}k €/wk)"
                     for r in tot.itertuples())
                     + f" at P={P_REF:g}. Rural areas lead on both the relative "
                       f"saving and the absolute sum, because the region has "
                       f"more rural model units than urban ones at comparable "
                       f"total parcel volume.",
                 caveats="Settlement type from plz_raumtyp.csv, joined on the "
                         "model unit's head postal code.")


# ---------------------------------------------------------------- fig73
def fig73_threshold_demand():
    d = _load()
    at = d[np.isclose(d.penalty, P_REF)].copy()
    at["parcels_per_day"] = at.weekly_parcels / D.N_DAYS

    for style in S.styles():
        S.apply(style)
        fig, axes = plt.subplots(1, 2,
                                 figsize=S.figsize(style, (11.5, 4.6),
                                                   (15.0, 5.8)))
        for rt in RT_ORDER:
            g = at[at.raumtyp_3 == rt]
            axes[0].scatter(g.parcels_per_day, g.saving_pct,
                            s=S.scale(style, 22, 1.9), color=RT_COLOR[rt],
                            alpha=0.8, edgecolor="none",
                            label=rt.capitalize())
            axes[1].scatter(g.demand_per_area, g.saving_pct,
                            s=S.scale(style, 22, 1.9), color=RT_COLOR[rt],
                            alpha=0.8, edgecolor="none",
                            label=rt.capitalize())
        for ax, col, xlab, logx in (
                (axes[0], "parcels_per_day", "Parcels per day and area", True),
                (axes[1], "demand_per_area",
                 "Weekly parcels per km$^2$", True)):
            x = at[col].values
            y = at.saving_pct.values
            ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
            lx = np.log10(x[ok])
            cf = np.polyfit(lx, y[ok], 1)
            xs = np.linspace(lx.min(), lx.max(), 50)
            ax.plot(10 ** xs, np.polyval(cf, xs), "--", color="#111111",
                    linewidth=S.scale(style, 1.5, 1.4))
            r = float(np.corrcoef(lx, y[ok])[0, 1])
            ax.set_xscale("log")
            ax.set_xlabel(xlab)
            ax.set_ylabel("Cost saving per area [%]")
            ax.grid(alpha=0.25)
            ax.set_title(f"r = {r:.2f} against log₁₀ demand")
            print(f"  {col}: r={r:.3f}")
        axes[0].legend(framealpha=0.9)
        fig.suptitle(rf"Consolidation pays where demand is thin "
                     rf"($P = {P_REF:g}$ €/p/d, $\theta = 100\%$)")
        fig.tight_layout(rect=[0, 0.02, 1, 0.92])
        S.save(fig, "fig73_threshold_demand", style, S.TIER_A)

    lo = at.nsmallest(10, "demand_per_area").saving_pct.mean()
    hi = at.nlargest(10, "demand_per_area").saving_pct.mean()
    D.prov.write("fig73_threshold_demand",
                 title="Saving against demand density",
                 tier=S.TIER_A, act=ACT, basis=BASIS,
                 claim=f"Demand density is the dominant driver: the ten "
                       f"sparsest areas average {lo:.1f}% saving, the ten "
                       f"densest {hi:.1f}%. Batching buys little where tours "
                       f"are already dense.")


# ---------------------------------------------------------------- fig74
def fig74_regime_map():
    d = _load()
    at = d[np.isclose(d.penalty, P_REF)].copy()

    nb = 5
    at["dens_bin"] = pd.qcut(at.demand_per_area, nb, duplicates="drop")
    at["hub_bin"] = pd.qcut(at.hub_dist_km, nb, duplicates="drop")
    piv = (at.pivot_table(index="hub_bin", columns="dens_bin",
                          values="saving_pct", aggfunc="mean",
                          observed=True))
    cnt = (at.pivot_table(index="hub_bin", columns="dens_bin",
                          values="saving_pct", aggfunc="size",
                          observed=True))

    for style in S.styles():
        S.apply(style)
        fig, ax = plt.subplots(figsize=S.figsize(style, (7.6, 5.6), (11.0, 6.4)))
        data = piv.values.astype(float)
        norm = Normalize(vmin=float(np.nanmin(data)), vmax=float(np.nanmax(data)))
        im = ax.imshow(data, cmap=S.CMAP_SAVING, norm=norm, aspect="auto",
                       origin="lower")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.isfinite(data[i, j]):
                    continue
                n = int(cnt.values[i, j]) if np.isfinite(cnt.values[i, j]) else 0
                ax.text(j, i, f"{data[i, j]:.0f}%\nn={n}", ha="center",
                        va="center",
                        fontsize=8 if style == "paper" else 12,
                        color="white" if norm(data[i, j]) > 0.6 else "#111111")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{iv.left:.0f}–{iv.right:.0f}"
                            for iv in piv.columns],
                           rotation=30 if style == "slides" else 0)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{iv.left:.1f}–{iv.right:.1f}"
                            for iv in piv.index])
        ax.set_xlabel("Weekly parcels per km$^2$ (quintile)")
        ax.set_ylabel("Hub distance [km] (quintile)")
        ax.set_title(rf"Saving regime at $P = {P_REF:g}$ €/p/d, "
                     rf"$\theta = 100\%$")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("Mean saving per area [%]")
        fig.tight_layout()
        S.save(fig, "fig74_regime_map", style, S.TIER_B)

    D.prov.write("fig74_regime_map",
                 title="Saving regime over density and hub distance",
                 tier=S.TIER_B, act=ACT, basis=BASIS,
                 claim=f"The operating regime that benefits most is low density "
                       f"combined with long hub access; mean saving spans "
                       f"{np.nanmin(piv.values):.0f}% to "
                       f"{np.nanmax(piv.values):.0f}% across the quintile grid.",
                 caveats="Quintile cells hold few observations each (n printed "
                         "per cell); read the gradient, not single cells.")


# ---------------------------------------------------------------- fig75
def fig75_breakeven():
    d = _load()
    per = (d.groupby(["penalty", "raumtyp_3"], as_index=False)
           .agg(base=("dd_cost_baseline_eur", "sum"),
                s3=("dd_cost_stage3_eur", "sum")))
    per["saving_pct"] = (1 - per.s3 / per.base) * 100
    sysg = _system_saving(d)

    for style in S.styles():
        S.apply(style)
        fig, ax = plt.subplots(figsize=S.figsize(style, (7.6, 4.8), (11.0, 5.8)))
        for rt in RT_ORDER:
            g = per[per.raumtyp_3 == rt].sort_values("penalty")
            ax.plot(g.penalty, g.saving_pct, "-o", color=RT_COLOR[rt],
                    markersize=S.scale(style, 5, 1.5),
                    linewidth=S.scale(style, 1.6, 1.7), label=rt.capitalize())
        ax.plot(sysg.penalty, sysg.saving_pct, "--s", color="#111111",
                markersize=S.scale(style, 4.5, 1.5),
                linewidth=S.scale(style, 1.5, 1.6), label="System")
        ax.set_xscale("symlog", linthresh=0.25)
        ax.set_xticks(sorted(d.penalty.unique()))
        ax.get_xaxis().set_major_formatter(
            plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel(r"Service penalty $P$ [€/parcel/day]")
        ax.set_ylabel("Cost saving [%]")
        ax.set_title(r"How far the service penalty can rise, $\theta = 100\%$")
        ax.axhline(0, color="#888888", linewidth=1)
        ax.grid(alpha=0.25)
        ax.legend(framealpha=0.9)
        # Which penalty flattens the curve is a measured property of the
        # grid, not a constant: on v6's operator plan a small saving
        # survives past P = 5, so the sentence is built from the data.
        flat = per[per.saving_pct < 1.0].groupby("raumtyp_3").penalty.min()
        flat_p = float(flat.max()) if len(flat) == len(RT_ORDER) else None
        P.footnote(fig, "Euro-weighted saving within each settlement type. "
                        + (f"Every type is below one percent from "
                           f"P = {flat_p:g} €/p/d upward: above that the "
                           f"penalty absorbs essentially the whole routing "
                           f"gain." if flat_p is not None else
                           "No penalty in this grid drives every settlement "
                           "type below one percent."),
                   style)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        S.save(fig, "fig75_breakeven", style, S.TIER_A)

    zero = per[(per.saving_pct < 0.05)].groupby("raumtyp_3").penalty.min()
    D.prov.write("fig75_breakeven", title="Saving against the service penalty",
                 tier=S.TIER_A, act=ACT, basis=BASIS,
                 claim="Consolidation collapses at a finite penalty: saving "
                       "first falls below 0.05 % at P = "
                       + ", ".join(f"{rt} {zero.get(rt, float('nan')):g}"
                                   for rt in RT_ORDER)
                       + " €/parcel/day, so the policy question is where the "
                         "compensation offered to customers sits below that.",
                 caveats="P is a modelled compensation per parcel per day of "
                         "delay, not an observed price.")


# ---------------------------------------------------------------- fig76
def fig76_provider_raumtyp():
    d = _load()
    at = d[np.isclose(d.penalty, P_REF)]
    per = (at.groupby(["provider", "raumtyp_3"], as_index=False)
           .agg(base=("dd_cost_baseline_eur", "sum"),
                s3=("dd_cost_stage3_eur", "sum")))
    per["saving_pct"] = (1 - per.s3 / per.base) * 100
    piv = (per.pivot(index="provider", columns="raumtyp_3", values="saving_pct")
           .reindex(D.PROVIDERS)[RT_ORDER])

    for style in S.styles():
        S.apply(style)
        fig, ax = plt.subplots(figsize=S.figsize(style, (6.8, 5.2), (9.5, 6.0)))
        data = piv.values.astype(float)
        norm = Normalize(vmin=float(np.nanmin(data)), vmax=float(np.nanmax(data)))
        im = ax.imshow(data, cmap=S.CMAP_SAVING, norm=norm, aspect="auto")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.isfinite(data[i, j]):
                    continue
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                        fontsize=9 if style == "paper" else 13,
                        color="white" if norm(data[i, j]) > 0.6 else "#111111")
        ax.set_xticks(range(len(RT_ORDER)))
        ax.set_xticklabels([rt.capitalize() for rt in RT_ORDER])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_title(rf"Saving [%] by provider and settlement type"
                     "\n"
                     rf"($P = {P_REF:g}$ €/p/d, $\theta = 100\%$)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("Cost saving [%]")
        fig.tight_layout()
        S.save(fig, "fig76_provider_raumtyp", style, S.TIER_B)

    D.prov.write("fig76_provider_raumtyp",
                 title="Provider x settlement-type saving matrix",
                 tier=S.TIER_B, act=ACT, basis=BASIS,
                 claim=f"Every provider gains most in rural areas, but the "
                       f"level differs sharply: the matrix spans "
                       f"{np.nanmin(data):.1f}% to {np.nanmax(data):.1f}%, "
                       f"reflecting different hub networks and demand profiles.",
                 caveats="Euro-weighted within each provider-type block; blank "
                         "cells mean that provider serves no area of that type.")


# ---------------------------------------------------------------- fig77
def fig77_drivers():
    d = _load()
    at = d[np.isclose(d.penalty, P_REF)].copy()
    feats = {
        "demand_per_area": "Parcels per km²",
        "weekly_parcels": "Weekly parcels",
        "n_stops_per_day": "Stops per day",
        "hub_dist_km": "Hub distance [km]",
        "area_km2": "Area [km²]",
        "b2c_share": "B2C share",
    }
    rows = []
    for col, label in feats.items():
        x = at[col].values.astype(float)
        y = at.saving_pct.values.astype(float)
        ok = np.isfinite(x) & np.isfinite(y)
        pear = float(np.corrcoef(x[ok], y[ok])[0, 1])
        rx = pd.Series(x[ok]).rank().values
        ry = pd.Series(y[ok]).rank().values
        spear = float(np.corrcoef(rx, ry)[0, 1])
        rows.append(dict(feature=label, pearson=pear, spearman=spear))
    corr = pd.DataFrame(rows).sort_values("spearman")
    print(corr.round(3).to_string(index=False))

    for style in S.styles():
        S.apply(style)
        fig, ax = plt.subplots(figsize=S.figsize(style, (7.4, 4.6), (10.5, 5.6)))
        y = np.arange(len(corr))
        colors = [D.PALETTE["accent"] if v < 0 else D.PALETTE["accent2"]
                  for v in corr.spearman]
        ax.barh(y, corr.spearman, 0.6, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(corr.feature)
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_xlabel("Spearman rank correlation with per-area saving")
        ax.set_title(rf"What drives the saving ($P = {P_REF:g}$ €/p/d)")
        ax.grid(alpha=0.25, axis="x")
        for yi, v in zip(y, corr.spearman):
            ax.text(v + (0.02 if v >= 0 else -0.02), yi, f"{v:+.2f}",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=9 if style == "paper" else 13)
        ax.set_xlim(-1.05, 1.05)
        P.footnote(fig, "Rank correlation over the 48 model units x 7 providers "
                        "at one operating point; a correlation is not a causal "
                        "effect and the features are themselves correlated.",
                   style)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        S.save(fig, "fig77_drivers", style, S.TIER_B)

    top = corr.iloc[0]
    hub = corr[corr.feature.str.startswith("Hub distance")].iloc[0]
    D.prov.write("fig77_drivers", title="Structural drivers of the saving",
                 tier=S.TIER_B, act=ACT, basis=BASIS,
                 claim=f"Two structural signals dominate and they point in "
                       f"opposite directions: demand volume suppresses the "
                       f"saving ({top.feature} {top.spearman:+.2f}) while hub "
                       f"distance raises it ({hub.spearman:+.2f}). Thin, "
                       f"far-flung areas are where consolidation earns its keep.",
                 caveats="Correlational only; the structural features are "
                         "mutually correlated, so these are not independent "
                         "effects.")


def main():
    for fn in (fig71_map_saving, fig72_raumtyp, fig73_threshold_demand,
               fig74_regime_map, fig75_breakeven, fig76_provider_raumtyp,
               fig77_drivers):
        print(f"\n=== {fn.__name__} ===")
        fn()


if __name__ == "__main__":
    main()
