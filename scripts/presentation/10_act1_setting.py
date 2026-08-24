"""Act 1 -- the setting: who delivers what, where, at what cost.

Figures
  fig11_lsp_volumes    weekly parcel volume and unit cost per provider
  fig12_map_demand     demand density across the region
  fig13_map_raumtyp    settlement-type classification of the study area
  fig14_headline       the headline numbers of the study

All of Act 1 is structural: demand, geography and the unbatched baseline. None
of it depends on the optimisation stage, so these figures are stable across the
Stage-2/Stage-3 revision. The baseline total is asserted against the pinned
1 909 747.75 € in load_baseline_per_provider().
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

import _data as D
import _plots as P
import _style as S

ACT = "1 - Setting"
BASIS = "Structural inputs (HAGRID demand, PLZ geodata, unbatched baseline)"

RT_ORDER = ["urban", "suburban", "rural"]
RT_COLOR = D.RT_COLOR   # paper fig 6 settlement palette, via _data


def _view(units):
    g = D.load_plz_geometry()
    ids = {str(x).zfill(5) for x in units}
    v = D.clip_to_scope(g, ids).copy()
    v["unit"] = np.where(v.cluster_id.isin(ids), v.cluster_id, v.plz)
    return v


# ---------------------------------------------------------------- fig11
def fig11_lsp_volumes():
    b = D.load_baseline_per_provider().set_index("provider").reindex(D.PROVIDERS)
    b = b.reset_index()
    total = b.weekly_parcels.sum()
    print(f"  {total:,} parcels/week across {len(b)} providers; "
          f"baseline {b.dd_cost.sum():,.0f} €/week")
    print(b[["provider", "weekly_parcels", "n_plz",
             "cost_per_1000_parcels"]].to_string(index=False))

    for style in S.styles():
        S.apply(style)
        fig, axes = plt.subplots(1, 2,
                                 figsize=S.figsize(style, (12.0, 4.6),
                                                   (15.0, 5.8)))
        x = np.arange(len(b))
        colors = [D.PROVIDER_COLOR[p] for p in b.provider]
        bars = axes[0].bar(x, b.weekly_parcels / 1000, 0.6, color=colors,
                           edgecolor="#333333", linewidth=0.6)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(b.provider,
                               rotation=30 if style == "slides" else 0)
        axes[0].set_ylabel("Parcels per week [1000]")
        axes[0].set_title("(a) Weekly parcel volume")
        axes[0].grid(alpha=0.25, axis="y")
        P.bar_labels(axes[0], bars, (b.weekly_parcels / 1000).values, "{:.0f}",
                     style=style)
        for xi, row in zip(x, b.itertuples()):
            axes[0].annotate(f"{row.n_plz} areas", xy=(xi, 0), xytext=(0, 8),
                             textcoords="offset points", ha="center",
                             fontsize=8 if style == "paper" else 11,
                             color="white", rotation=90)

        bars2 = axes[1].bar(x, b.cost_per_1000_parcels, 0.6, color=colors,
                            edgecolor="#333333", linewidth=0.6)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(b.provider,
                               rotation=30 if style == "slides" else 0)
        axes[1].set_ylabel("Baseline cost per 1000 parcels [€]")
        axes[1].set_title("(b) Unit cost, daily delivery")
        axes[1].grid(alpha=0.25, axis="y")
        P.bar_labels(axes[1], bars2, b.cost_per_1000_parcels.values, "{:.0f}",
                     style=style)

        tot_str = f"{total:,}".replace(",", " ")
        cost_str = f"{b.dd_cost.sum():,.0f}".replace(",", " ")
        P.footnote(fig, f"Region Hannover, {tot_str} parcels per week, "
                        f"{cost_str} € weekly routing cost under daily "
                        f"delivery. Demand from the HAGRID model.", style)
        fig.tight_layout(rect=[0, 0.07, 1, 1])
        S.save(fig, "fig11_lsp_volumes", style, S.TIER_A)

    # Space as thousands separator, applied to the numbers only: a blanket
    # comma replacement over the whole sentence would eat its punctuation.
    tot_num = f"{total:,}".replace(",", " ")
    cost_num = f"{b.dd_cost.sum():,.0f}".replace(",", " ")
    dhl_share = (b.loc[b.provider == "DHL"].weekly_parcels.iloc[0]
                 / total * 100)
    D.prov.write("fig11_lsp_volumes", title="Provider volumes and unit costs",
                 tier=S.TIER_A, act=ACT, basis=BASIS,
                 claim=f"Seven providers move {tot_num} parcels a week through "
                       f"the region at {cost_num} € of routing cost. DHL alone "
                       f"carries {dhl_share:.0f}% of volume at the lowest unit "
                       f"cost ({b.cost_per_1000_parcels.min():.0f} €/1000), "
                       f"while the smallest networks pay up to "
                       f"{b.cost_per_1000_parcels.max():.0f} €/1000 — the "
                       f"structural reason consolidation gains differ by "
                       f"provider.")


# ---------------------------------------------------------------- fig12
def fig12_map_demand():
    d = D.load_per_plz()
    at = d[np.isclose(d.penalty, 0.25)]
    per = (at.groupby("plz", as_index=False)
           .agg(parcels=("weekly_parcels", "sum"),
                area=("area_km2", "first")))
    per["density"] = per.parcels / per.area
    per = per.rename(columns={"plz": "unit"})
    view = _view(per.unit.unique())
    print(f"  demand density {per.density.min():.0f}–{per.density.max():.0f} "
          f"parcels/km²/week over {len(per)} units")

    for style in S.styles():
        S.apply(style)
        fig, axes = plt.subplots(1, 2,
                                 figsize=S.figsize(style, (9.6, 5.0), (13.0, 6.0)))
        for ax, col, label, cmap in (
                (axes[0], "parcels", "Weekly parcels per area", S.CMAP_DEMAND),
                (axes[1], "density", "Weekly parcels per km²", "Purples")):
            m = view.merge(per[["unit", col]], on="unit", how="left")
            norm = LogNorm(vmin=max(1.0, float(per[col].min())),
                           vmax=float(per[col].max()))
            P.choropleth(ax, m, col, cmap=cmap, norm=norm, title=label,
                         style=style)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle("Parcel demand across the Region Hannover study area")
        P.footnote(fig, "Summed over all seven providers, per model unit "
                        "(merged postal-code cluster). Log colour scale.", style)
        fig.tight_layout(rect=[0, 0.05, 1, 0.94])
        S.save(fig, "fig12_map_demand", style, S.TIER_A)

    D.prov.write("fig12_map_demand", title="Demand and demand density map",
                 tier=S.TIER_A, act=ACT, basis=BASIS,
                 claim=f"Demand density spans {per.density.min():.0f} to "
                       f"{per.density.max():.0f} parcels per km² per week — a "
                       f"factor of {per.density.max() / per.density.min():.0f} "
                       f"across one metropolitan region. This spread is what a "
                       f"single region-wide delivery policy has to cover.",
                 caveats="Log colour scale; values are per merged cluster, so "
                         "member polygons of one cluster share a value.")


# ---------------------------------------------------------------- fig13
def fig13_map_raumtyp():
    d = D.load_per_plz()
    units = d.plz.unique()
    rt = D.load_raumtyp()
    view = _view(units)
    m = view.merge(rt[["plz", "raumtyp_3"]], on="plz", how="left")

    codes = {rt_: i for i, rt_ in enumerate(RT_ORDER)}
    m["rt_code"] = m.raumtyp_3.map(codes)
    n_missing = int(m.rt_code.isna().sum())
    counts = m.raumtyp_3.value_counts().to_dict()
    print(f"  polygons by type: {counts}; {n_missing} unclassified")

    cmap = ListedColormap([RT_COLOR[rt_] for rt_ in RT_ORDER])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    for style in S.styles():
        S.apply(style)
        fig, ax = plt.subplots(figsize=S.figsize(style, (6.0, 6.0), (8.5, 6.4)))
        P.choropleth(ax, m, "rt_code", cmap=cmap, norm=norm,
                     title="Settlement type of the study area", style=style)
        handles = [Patch(facecolor=RT_COLOR[rt_], label=rt_.capitalize())
                   for rt_ in RT_ORDER]
        ax.legend(handles=handles, loc="upper right", framealpha=0.9,
                  title="Settlement type")
        # No cross-reference to the deck's internal act numbering here: the
        # footnote is baked into the PNG and the audience never sees that map.
        P.footnote(fig, "Settlement type per postal-code area, BBSR "
                        "classification (plz_raumtyp.csv).", style)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        S.save(fig, "fig13_map_raumtyp", style, S.TIER_B)

    D.prov.write("fig13_map_raumtyp", title="Settlement-type classification map",
                 tier=S.TIER_B, act=ACT, basis=BASIS,
                 claim="The study area is a dense core inside a wide suburban "
                       "and rural ring — the geography that makes a uniform "
                       "delivery-frequency policy inefficient.",
                 caveats=f"{n_missing} polygon(s) carry no classification and "
                         f"are drawn in the missing-data colour."
                 if n_missing else "")


# ---------------------------------------------------------------- fig14
def fig14_headline():
    b = D.load_baseline_per_provider()
    g = D.saving_grid()
    sv = D.load_savings_validation()
    f = D.load_fleet()

    best = g.loc[g.saving_pct.idxmax()]
    eff = g[np.isclose(g.penalty, 0.25) & np.isclose(g.share_willing, 1.0)].iloc[0]

    sys_day = (f.groupby(["penalty", "share_willing", "day"], as_index=False)
               .agg(s2=("fleet_stage2", "sum"), s3=("fleet_stage3", "sum")))
    ref = (sys_day[np.isclose(sys_day.share_willing, 0.0)]
           .groupby(["penalty", "day"], as_index=False).agg(v=("s2", "sum"))
           .groupby("day").v.mean())
    cell = sys_day[np.isclose(sys_day.penalty, 0.25)
                   & np.isclose(sys_day.share_willing, 1.0)].sort_values("day")
    peak_cut = (ref.max() - cell.s3.max()) / ref.max() * 100

    tiles = [
        (f"{b.weekly_parcels.sum() / 1e6:.2f} M", "parcels per week",
         D.PALETTE["baseline"]),
        (f"{b.dd_cost.sum() / 1e6:.2f} M €", "weekly baseline cost",
         D.PALETTE["baseline"]),
        (f"{eff.saving_pct:.1f} %", "cost saving at the\nefficient point",
         D.PALETTE["stage3"]),
        (f"{best.saving_pct:.1f} %", "cost saving at the\ncost-optimal point",
         D.PALETTE["stage2"]),
        (f"{peak_cut:.1f} %", "smaller peak fleet", D.PALETTE["accent2"]),
        (f"+{sv.conservatism_pp.min():.1f}…{sv.conservatism_pp.max():.1f} pp",
         "surrogate is conservative\nvs real routing", D.PALETTE["accent"]),
    ]
    print(f"  efficient {eff.saving_pct:.1f}%  best {best.saving_pct:.1f}%  "
          f"peak cut {peak_cut:.1f}%")

    for style in S.styles():
        S.apply(style)
        fig, axes = plt.subplots(2, 3,
                                 figsize=S.figsize(style, (11.0, 4.4),
                                                   (14.0, 6.2)))
        for ax, (big, small, color) in zip(axes.ravel(), tiles):
            ax.set_axis_off()
            ax.add_patch(plt.Rectangle((0.02, 0.06), 0.96, 0.88,
                                       transform=ax.transAxes, facecolor=color,
                                       alpha=0.14, edgecolor=color,
                                       linewidth=1.6))
            ax.text(0.5, 0.62, big, transform=ax.transAxes, ha="center",
                    va="center", color=color,
                    fontsize=20 if style == "paper" else 30, fontweight="bold")
            ax.text(0.5, 0.26, small, transform=ax.transAxes, ha="center",
                    va="center", color="#333333",
                    fontsize=9.5 if style == "paper" else 14)
        fig.suptitle("Time-based consolidation in Region Hannover — "
                     "the headline numbers")
        P.footnote(fig, r"Efficient point $P = 0.25$ €/p/d, cost-optimal point "
                        r"$P = 0$, both at $\theta = 100\%$. Savings against "
                        "the daily-delivery baseline, Stage 3.", style)
        fig.tight_layout(rect=[0, 0.06, 1, 0.92])
        S.save(fig, "fig14_headline", style, S.TIER_A)

    D.prov.write("fig14_headline", title="Headline results tile",
                 tier=S.TIER_A, act=ACT, basis="Stage 3 + VROOM revalidation",
                 claim=f"Consolidating {b.weekly_parcels.sum() / 1e6:.2f} M "
                       f"parcels a week cuts routing cost "
                       f"{eff.saving_pct:.1f}% at the efficient operating point "
                       f"and {best.saving_pct:.1f}% at the cost-optimal one, "
                       f"with {peak_cut:.1f}% less peak fleet, and real routing "
                       f"beats the prediction by "
                       f"{sv.conservatism_pp.min():.1f}-"
                       f"{sv.conservatism_pp.max():.1f} pp.",
                 caveats="All figures at theta = 100% willingness to wait, the "
                         "upper bound of the modelled range.")


def main():
    for fn in (fig11_lsp_volumes, fig12_map_demand, fig13_map_raumtyp,
               fig14_headline):
        print(f"\n=== {fn.__name__} ===")
        fn()


if __name__ == "__main__":
    main()
