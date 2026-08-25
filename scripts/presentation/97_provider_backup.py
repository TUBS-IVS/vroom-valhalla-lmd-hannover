"""Backup figures: the headline results split per provider, plus the bulge.

The four results figures in the talk are aggregated over all seven carriers.
Aggregation is what makes them readable, but it also hides that the carriers
behave very differently — DHL barely consolidates at all while GLS and DPD go
to two days a week across most of their network. These panels put the split
back in, one panel per provider, for the backup section.

    figP1_mix_by_provider      chosen delivery frequency vs adoption
    figP2_saving_by_provider   the P x theta saving grid, per carrier
    figP3_map_saving_provider  where each carrier's saving sits in space
    figP4_map_freq_provider    each carrier's chosen frequency in space

And four that unpack the bump at theta = 10 %, which survives even a punitive
penalty:

    figB1_who_consolidates     the size of the cells that batch — no outliers
    figB2_where_the_money_is   what the saving is NOT: dropped vehicles
    figB3_ptheta_collapse      the effective knob is the product P x theta
    figB4_size_vs_takt         cell size against chosen takt, measured

Slides rendering only (16:9, large type) — these are talk backup, not paper
figures.

Usage:
    python scripts/presentation/97_provider_backup.py [--only NAME]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap         # noqa: E402
from matplotlib.patches import Patch                               # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _data as D                                                  # noqa: E402
import _style as S                                                 # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
STYLE = "slides"
TIER = S.TIER_B

# The efficient operating point the talk carries.
P_REF, THETA_REF = 0.25, 1.0
# The corner the bump lives in.
P_BULGE, TH_BULGE = 10.0, 0.1

ORDER = ["DHL", "Amazon", "Hermes", "UPS", "DPD", "GLS", "FedEx"]


def _providers(df):
    """Provider order: biggest network first, so the eye starts at DHL."""
    have = [p for p in ORDER if p in set(df.provider)]
    return have + sorted(set(df.provider) - set(have))


def _grid(n, style, w=4.6, h=3.3, cols=4):
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols, h * rows))
    return fig, np.atleast_1d(axes).ravel(), rows


def _spare(axes, n):
    """The unused cell of the 4x2 grid — where legends and colourbars go.

    Seven carriers in an eight-cell grid leave exactly one free slot. Putting
    the key there instead of outside the figure stops it from sitting on top of
    the last panel, which is what `bbox_to_anchor` did.
    """
    ax = axes[n] if n < len(axes) else axes[-1]
    ax.set_axis_off()
    return ax


def _drop_p04(df):
    return df[~np.isclose(df.penalty, 0.4)].copy()


def _view(units):
    """Postal-code polygons restricted to the modelled area."""
    gdf = D.load_plz_geometry()
    view = D.clip_to_scope(gdf, units)
    ids = {str(u).zfill(5) for u in units}
    view["unit"] = view.cluster_id.where(view.cluster_id.isin(ids), view.plz)
    return view


# ═══════════════════════════════════════════════════════════════════════════
# per-provider splits of the aggregated results figures
# ═══════════════════════════════════════════════════════════════════════════
def figP1_mix_by_provider():
    """The frequency mix each carrier chooses, as adoption rises."""
    s = _drop_p04(D.load_chosen_stage3())
    at = s[np.isclose(s.penalty, P_REF)]
    provs = _providers(at)
    S.apply(STYLE)
    fig, axes, rows = _grid(len(provs), STYLE)
    for ax, prov in zip(axes, provs):
        sub = at[at.provider == prov]
        piv = (sub.groupby(["share_willing", "schedule_size_system_smoothed"])
                  .size().unstack(fill_value=0).sort_index())
        piv = piv.div(piv.sum(axis=1), axis=0) * 100
        x = piv.index.values * 100
        bottom = np.zeros(len(piv))
        for sz in D.FREQ_SIZES:
            if sz not in piv.columns:
                continue
            ax.fill_between(x, bottom, bottom + piv[sz].values,
                            color=D.FREQ_COLOR[sz], alpha=0.94)
            bottom += piv[sz].values
        ax.set_title(f"{prov}   ({sub.plz.nunique()} areas)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)
    for ax in axes[len(provs):]:
        ax.set_axis_off()
    fig.supxlabel("Willingness-to-wait share θ [%]")
    fig.supylabel("Share of that carrier's areas [%]")
    fig.suptitle(f"Chosen delivery frequency per carrier, "
                 f"P = {P_REF:g} €/parcel/day")
    handles = [Patch(facecolor=D.FREQ_COLOR[k], label=f"{k} day/wk")
               for k in D.FREQ_SIZES]
    _spare(axes, len(provs)).legend(handles=handles, loc="center",
                                    title="Delivery days per week",
                                    frameon=False, fontsize=15,
                                    title_fontsize=15)
    fig.tight_layout(rect=[0.02, 0.04, 1, 0.94])
    S.save(fig, "figP1_mix_by_provider", STYLE, TIER)


def figP2_saving_by_provider():
    """The saving grid each carrier sees, against its own baseline."""
    c = _drop_p04(D.load_costs())
    base = D.load_baseline_per_provider().set_index("provider").dd_cost
    c["saving_pct"] = 100 * (c.provider.map(base) - c.total_stage3_eur) \
        / c.provider.map(base)
    provs = _providers(c)
    S.apply(STYLE)
    fig, axes, rows = _grid(len(provs), STYLE, w=4.5, h=3.4)
    vmax = float(c.saving_pct.max())
    print(f"  per-carrier saving spans 0 – {vmax:.1f} %")
    im = None
    for ax, prov in zip(axes, provs):
        piv = (c[c.provider == prov]
               .pivot(index="penalty", columns="share_willing",
                      values="saving_pct").sort_index(ascending=False))
        im = ax.imshow(piv.values, cmap=S.CMAP_SAVING, vmin=0, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{v:.0%}".replace("%", "") for v in piv.columns],
                           fontsize=11)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{v:g}" for v in piv.index], fontsize=11)
        best = np.unravel_index(np.nanargmax(piv.values), piv.shape)
        ax.text(best[1], best[0], f"{piv.values[best]:.0f}", ha="center",
                va="center", fontsize=12, fontweight="bold", color="white")
        ax.set_title(f"{prov}   max {np.nanmax(piv.values):.1f} %")
    for ax in axes[len(provs):]:
        ax.set_axis_off()
    fig.supxlabel("Willingness-to-wait share θ [%]")
    fig.supylabel("Service penalty P [€/p/d]")
    fig.suptitle("Weekly cost saving per carrier, against that carrier's own "
                 "daily-delivery baseline")
    sp = _spare(axes, len(provs))
    fig.tight_layout(rect=[0.02, 0.04, 1, 0.94])
    cax = sp.inset_axes([0.10, 0.12, 0.13, 0.76])
    fig.colorbar(im, cax=cax, label="Saving [%]")
    S.save(fig, "figP2_saving_by_provider", STYLE, TIER)


def figP3_map_saving_provider():
    """Where each carrier's saving sits in space."""
    d = D.load_per_plz()
    at = d[np.isclose(d.penalty, P_REF)]
    view = _view(d.plz.unique())
    provs = _providers(at)
    S.apply(STYLE)
    fig, axes, rows = _grid(len(provs), STYLE, w=3.6, h=3.4)
    vmax = float(np.nanpercentile(at.saving_pct, 98))
    print(f"  colour scale 0 – {vmax:.0f} % (98th percentile)")
    im = None
    for ax, prov in zip(axes, provs):
        sub = (at[at.provider == prov][["plz", "saving_pct"]]
               .rename(columns={"plz": "unit"}))
        sub["unit"] = sub.unit.astype(str).str.zfill(5)
        m = view.merge(sub, on="unit", how="left")
        m.plot(ax=ax, column="saving_pct", cmap="Greens", vmin=0, vmax=vmax,
               edgecolor="white", linewidth=0.3,
               missing_kwds={"color": S.MISSING, "edgecolor": "white",
                             "linewidth": 0.3})
        med = float(np.nanmedian(sub.saving_pct))
        ax.set_title(f"{prov}\nmedian {med:.1f} %")
        ax.set_axis_off()
        ax.set_aspect("equal")
    for ax in axes[len(provs):]:
        ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap="Greens",
                               norm=plt.Normalize(vmin=0, vmax=vmax))
    sp = _spare(axes, len(provs))
    fig.suptitle(f"Where consolidation pays, per carrier · "
                 f"P = {P_REF:g} €/p/d, θ = {THETA_REF:.0%}")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.subplots_adjust(hspace=0.28)
    cax = sp.inset_axes([0.12, 0.14, 0.14, 0.72])
    fig.colorbar(sm, cax=cax, label="Cost saving per area [%]")
    S.save(fig, "figP3_map_saving_provider", STYLE, TIER)


def figP4_map_freq_provider():
    """Each carrier's chosen delivery frequency in space."""
    s = D.load_chosen_stage3()
    at = s[np.isclose(s.penalty, P_REF) & np.isclose(s.share_willing, THETA_REF)]
    view = _view(s.plz.unique())
    cmap = ListedColormap([D.FREQ_COLOR[k] for k in D.FREQ_SIZES])
    norm = BoundaryNorm([k - 0.5 for k in D.FREQ_SIZES]
                        + [D.FREQ_SIZES[-1] + 0.5], cmap.N)
    provs = _providers(at)
    S.apply(STYLE)
    fig, axes, rows = _grid(len(provs), STYLE, w=3.6, h=3.4)
    for ax, prov in zip(axes, provs):
        sub = (at[at.provider == prov]
               [["plz", "schedule_size_system_smoothed"]]
               .rename(columns={"plz": "unit",
                                "schedule_size_system_smoothed": "freq"}))
        sub["unit"] = sub.unit.astype(str).str.zfill(5)
        m = view.merge(sub, on="unit", how="left")
        m.plot(ax=ax, column="freq", cmap=cmap, norm=norm, edgecolor="white",
               linewidth=0.3,
               missing_kwds={"color": S.MISSING, "edgecolor": "white",
                             "linewidth": 0.3})
        ax.set_title(f"{prov}\nmedian {float(sub.freq.median()):.0f} day/wk")
        ax.set_axis_off()
        ax.set_aspect("equal")
    for ax in axes[len(provs):]:
        ax.set_axis_off()
    handles = [Patch(facecolor=D.FREQ_COLOR[k], label=f"{k} day/wk")
               for k in D.FREQ_SIZES]
    _spare(axes, len(provs)).legend(handles=handles, loc="center",
                                    title="Delivery frequency", frameon=False,
                                    fontsize=15, title_fontsize=15)
    fig.suptitle(f"Chosen delivery frequency per carrier · "
                 f"P = {P_REF:g} €/p/d, θ = {THETA_REF:.0%}")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.subplots_adjust(hspace=0.28)
    S.save(fig, "figP4_map_freq_provider", STYLE, TIER)


# ═══════════════════════════════════════════════════════════════════════════
# the bump at theta = 10 %
# ═══════════════════════════════════════════════════════════════════════════
def _bulge_frame():
    """The cells at (P = 10, theta = 0.1) with their baseline vehicle-days."""
    raw = pd.read_csv(D.RUN / "tab_chosen_schedules.csv")
    raw["plz"] = raw.plz.astype(str)
    cell = raw[np.isclose(raw.penalty, P_BULGE)
               & np.isclose(raw.share_willing, TH_BULGE)].copy()
    base = (raw[np.isclose(raw.penalty, 0.0) & np.isclose(raw.share_willing, 0.0)]
            [["provider", "plz", "veh_init", "dd_cost_init"]]
            .rename(columns={"veh_init": "veh_base",
                             "dd_cost_init": "dd_base"}))
    m = cell.merge(base, on=["provider", "plz"], how="left")
    m["consolidates"] = m.schedule_size_init < 6
    m["d_veh"] = m.veh_base - m.veh_init
    m["d_dd"] = m.dd_base - m.dd_cost_init
    return m


def figB1_who_consolidates():
    """The cells that batch are ordinary in size — there is no outlier tail."""
    m = _bulge_frame()
    yes = m[m.consolidates]
    no = m[~m.consolidates]
    print(f"  {len(yes)} consolidating cells, {yes.plz.nunique()} distinct "
          f"areas, {100*yes.weekly_parcels.sum()/m.weekly_parcels.sum():.1f} % "
          f"of regional volume; smallest {yes.weekly_parcels.min():.0f}/week")
    S.apply(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    bins = np.logspace(np.log10(500), np.log10(40000), 26)
    ax.hist([no.weekly_parcels, yes.weekly_parcels], bins=bins, stacked=True,
            color=[S.GRID, S.BRAND], edgecolor="white", linewidth=0.7,
            label=[f"stays daily  (n = {len(no)})",
                   f"consolidates  (n = {len(yes)})"])
    ax.set_xscale("log")
    ax.axvline(yes.weekly_parcels.min(), color=S.INK, ls="--", lw=2.0)
    top = ax.get_ylim()[1]
    ax.annotate(f"smallest consolidating cell\n"
                f"{yes.weekly_parcels.min():,.0f} parcels / week"
                .replace(",", " "),
                xy=(yes.weekly_parcels.min(), top * 0.40),
                xytext=(1250, top * 0.58), fontsize=14,
                arrowprops=dict(arrowstyle="->", color=S.INK, lw=1.6))
    ax.set_xlabel("Weekly parcels in the cell  (log scale)")
    ax.set_ylabel("Number of cells")
    ax.set_ylim(0, top * 1.22)
    ax.legend(loc="upper right", fontsize=13)
    ax.grid(alpha=0.25)

    ax = axes[1]
    q = [yes.weekly_parcels.quantile(v) for v in (0.0, .25, .5, .75, 1.0)]
    labels = ["min", "25 %", "median", "75 %", "max"]
    ax.barh(range(5), q, color=S.BRAND, alpha=0.9, height=0.62)
    for i, v in enumerate(q):
        ax.text(v * 1.03, i, f"{v:,.0f}".replace(",", " "), va="center",
                fontsize=15)
    ax.set_yticks(range(5))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(q) * 1.22)
    ax.set_xlabel("Weekly parcels per consolidating cell")
    ax.grid(alpha=0.25, axis="x")
    fig.suptitle(f"Who consolidates at P = {P_BULGE:g} €/p/d, "
                 f"θ = {TH_BULGE:.0%} — and how big they are")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    S.save(fig, "figB1_who_consolidates", STYLE, TIER)


def figB2_where_the_money_is():
    """The saving is not dropped vehicles — it is tour-days and distance."""
    m = _bulge_frame()
    yes = m[m.consolidates]
    veh_saved = float(yes.d_veh.sum())
    veh_eur = veh_saved * 189.15
    dd_eur = float(yes.d_dd.sum())
    costs = D.load_costs()
    tot = costs[np.isclose(costs.penalty, P_BULGE)
                & np.isclose(costs.share_willing, TH_BULGE)].total_stage3_eur.sum()
    sys_eur = D.BASE_TOTAL - tot
    print(f"  vehicle-days saved {veh_saved:.0f} of {m.veh_base.sum():.0f} "
          f"({veh_eur:,.0f} €) · per-cell dd delta {dd_eur:,.0f} € · "
          f"system saving {sys_eur:,.0f} € ({100*sys_eur/D.BASE_TOTAL:.1f} %)")
    S.apply(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))

    ax = axes[0]
    vals = [veh_eur, dd_eur - veh_eur]
    ax.bar([0], [vals[0]], color=S.BRAND, width=0.55,
           label="dropped vehicle-days")
    ax.bar([0], [vals[1]], bottom=[vals[0]], color=S.GRID, width=0.55,
           label="everything else (tour-days, distance)")
    ax.bar([1], [sys_eur], color=S.INK_SOFT, width=0.55,
           label="what the system actually saves")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["per-cell signal\n(unbundled)",
                        "system result\n(bundled)"])
    ax.set_ylabel("€ per week")
    ax.set_ylim(0, dd_eur * 1.18)
    ax.legend(loc="upper right", fontsize=13)
    ax.grid(alpha=0.25, axis="y")
    ax.annotate(f"only {veh_eur:,.0f} €".replace(",", " "),
                xy=(0.30, veh_eur), xytext=(0.62, dd_eur * 0.30),
                fontsize=14, fontweight="bold", color=S.BRAND,
                arrowprops=dict(arrowstyle="->", color=S.BRAND, lw=1.8))
    ax.text(0, dd_eur * 1.02, f"{dd_eur:,.0f} €".replace(",", " "),
            ha="center", fontsize=15, fontweight="bold")
    ax.text(1, sys_eur * 1.04, f"{sys_eur:,.0f} €".replace(",", " "),
            ha="center", fontsize=15, fontweight="bold")

    ax = axes[1]
    zero = int((yes.d_veh == 0).sum())
    ax.barh([0, 1], [zero, len(yes) - zero],
            color=[S.GRID, S.BRAND], height=0.55)
    for i, v in enumerate([zero, len(yes) - zero]):
        ax.text(v + 2, i, f"{v} cells", va="center", fontsize=15,
                fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["save no\nvehicle-day", "save at least\none"])
    ax.set_xlim(0, len(yes) * 1.2)
    ax.set_xlabel("Cells that chose to consolidate")
    ax.grid(alpha=0.25, axis="x")
    fig.suptitle("The saving is not dropped vehicles — "
                 f"P = {P_BULGE:g} €/p/d, θ = {TH_BULGE:.0%}")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    S.save(fig, "figB2_where_the_money_is", STYLE, TIER)


def figB3_ptheta_collapse():
    """Share consolidating collapses onto the product of penalty and adoption."""
    s = _drop_p04(D.load_chosen_stage3())
    col = "schedule_size_system_smoothed"
    g = (s.assign(c=(s[col] < 6).astype(float))
           .groupby(["penalty", "share_willing"], as_index=False).c.mean())
    g = g[g.share_willing > 0]
    g["c"] *= 100
    g["pt"] = g.penalty * g.share_willing
    rho_pt = g.c.corr(g.pt, method="spearman")
    rho_p = g.c.corr(g.penalty, method="spearman")
    rho_t = g.c.corr(g.share_willing, method="spearman")
    print(f"  Spearman: P·θ {rho_pt:.3f} · P {rho_p:.3f} · θ {rho_t:.3f}")
    S.apply(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6))
    for ax, xcol, lab, rho in ((axes[0], "share_willing",
                                "Willingness-to-wait share θ", rho_t),
                               (axes[1], "pt",
                                "Effective penalty  P · θ", rho_pt)):
        sc = ax.scatter(g[xcol] * (100 if xcol == "share_willing" else 1), g.c,
                        c=g.penalty, cmap=S.CMAP_PENALTY, s=170,
                        edgecolors="white", linewidths=1.3,
                        norm=matplotlib.colors.LogNorm(
                            vmin=max(0.2, g.penalty[g.penalty > 0].min()),
                            vmax=g.penalty.max()))
        ax.set_xlabel(lab)
        ax.set_ylabel("Cells choosing fewer than six delivery days [%]")
        ax.set_title(f"Spearman ρ = {rho:+.2f}")
        ax.grid(alpha=0.25)
        if xcol == "pt":
            ax.set_xscale("log")
    fig.colorbar(sc, ax=axes.tolist(), fraction=0.02, pad=0.02,
                 label="Service penalty P")
    fig.suptitle("θ alone explains almost nothing — the product with P does")
    fig.tight_layout(rect=[0, 0, 0.93, 0.93])
    S.save(fig, "figB3_ptheta_collapse", STYLE, TIER)


def figB4_size_vs_takt():
    """Cell size against the takt it chooses, at three effective penalties."""
    s = _drop_p04(D.load_chosen_stage3())
    vol = (D.load_per_plz()[["provider", "plz", "weekly_parcels"]]
           .drop_duplicates(["provider", "plz"]))
    s = s.merge(vol, on=["provider", "plz"], how="left")
    s["freq"] = s.schedule_size_system_smoothed
    picks = [(5.0, 0.1), (5.0, 0.2), (5.0, 0.3)]
    S.apply(STYLE)
    fig, axes = plt.subplots(1, len(picks), figsize=(5.2 * len(picks), 5.4),
                             sharey=True)
    for ax, (pen, th) in zip(np.atleast_1d(axes), picks):
        sub = s[np.isclose(s.penalty, pen) & np.isclose(s.share_willing, th)]
        data, labels, colours = [], [], []
        for k in D.FREQ_SIZES:
            v = sub[sub.freq == k].weekly_parcels.dropna()
            if len(v) == 0:
                continue
            data.append(v.values)
            labels.append(f"{k} d\n(n={len(v)})")
            colours.append(D.FREQ_COLOR[k])
        bp = ax.boxplot(data, patch_artist=True, widths=0.62,
                        medianprops=dict(color=S.INK, lw=2.2),
                        flierprops=dict(marker="o", markersize=4,
                                        markerfacecolor=S.GRID,
                                        markeredgecolor="none"))
        for patch, col in zip(bp["boxes"], colours):
            patch.set_facecolor(col)
            patch.set_alpha(0.85)
            patch.set_edgecolor("white")
        ax.set_xticklabels(labels, fontsize=13)
        ax.set_yscale("log")
        ax.set_title(f"P = {pen:g},  θ = {th:.0%}   (P·θ = {pen*th:g})")
        ax.grid(alpha=0.25, axis="y")
    np.atleast_1d(axes)[0].set_ylabel("Weekly parcels in the cell (log)")
    fig.suptitle("Small cells consolidate, large cells stay daily — and the "
                 "threshold moves with P · θ")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    S.save(fig, "figB4_size_vs_takt", STYLE, TIER)


FIGURES = {f.__name__: f for f in (
    figP1_mix_by_provider, figP2_saving_by_provider, figP3_map_saving_provider,
    figP4_map_freq_provider, figB1_who_consolidates, figB2_where_the_money_is,
    figB3_ptheta_collapse, figB4_size_vs_takt)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="figure names to build")
    a = ap.parse_args()
    todo = ([FIGURES[n] for n in a.only] if a.only else list(FIGURES.values()))
    for fn in todo:
        print(f"\n=== {fn.__name__} ===")
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
