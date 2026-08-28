"""Act 3 -- core optimisation results, Stage-3 basis.

Figures
  fig31_saving_grid      cost saving % over the full (P, theta) grid
  fig32_wait_grid        mean additional customer wait over the grid
  fig33_fleet_grid       peak-fleet and Mo-Sa CV reduction over the grid
  fig34_pareto           cost saving vs added wait, all 88 cells + front
  fig35_schedule_mix     share of postal-code areas by delivery frequency

The paper's combined six-panel fig5 is not re-implemented here: a correct
Stage-3 render already exists and is adopted verbatim by 95_adopt_paper_figs.py.
Six panels are unreadable projected, so the slide deck gets the panels split
across fig31/32/33 instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

import _data as D
import _plots as P
import _style as S

ACT = "3 - Core results"
STAGE3 = "Stage 3 (per-hub balancing + system smoothing), theta grid complete"


def _drop_p04(df):
    """P=0.4 exists only in the older balanced run, not in the Stage-3 grid."""
    return df[~np.isclose(df.penalty, 0.4)].copy()


# ---------------------------------------------------------------- fig31
def fig31_saving_grid():
    g = _drop_p04(D.saving_grid())
    piv = g.pivot(index="penalty", columns="share_willing", values="saving_pct")

    peak = piv.stack().idxmax()
    print(f"  max saving {piv.stack().max():.1f}% at (P={peak[0]}, theta={peak[1]})")

    for style in S.styles():
        S.apply(style)
        fig, ax = plt.subplots(figsize=S.figsize(style, (7.2, 5.4), (11.0, 6.2)))
        P.heat(ax, piv, S.CMAP_SAVING,
               "Weekly cost saving against daily-delivery baseline",
               vmin=0, vmax=float(np.ceil(piv.values.max() / 5) * 5),
               cbar_label="Saving [%]", style=style)
        P.grid_labels(ax, style)
        base_str = f"{D.BASE_TOTAL:,.0f}".replace(",", " ")
        P.footnote(fig, f"Baseline = {base_str} € per week, daily delivery. "
                        f"{STAGE3}.", style)
        fig.tight_layout()
        S.save(fig, "fig31_saving_grid", style, S.TIER_A)

    D.prov.write("fig31_saving_grid", title="Cost-saving grid over (P, theta)",
                 tier=S.TIER_A, act=ACT, basis=STAGE3,
                 claim=f"Saving peaks at {piv.stack().max():.1f}% at "
                       f"P={peak[0]}, theta={peak[1]}; the theta=0 column is "
                       f"identically 0% because no consolidation is admissible "
                       f"when nobody will wait.")


# ---------------------------------------------------------------- fig32
def fig32_wait_grid():
    w = _drop_p04(D.load_wait())
    piv = w.pivot(index="penalty", columns="share_willing",
                  values="avg_wait_d_stage3")
    print(f"  max wait {piv.values.max():.2f} d; wait at (0.25,1) = "
          f"{piv.loc[0.25, 1.0]:.2f} d")

    for style in S.styles():
        S.apply(style)
        fig, ax = plt.subplots(figsize=S.figsize(style, (7.2, 5.4), (11.0, 6.2)))
        P.heat(ax, piv, S.CMAP_WAIT,
               "Mean additional customer wait per parcel",
               vmin=0, vmax=float(np.ceil(piv.values.max() * 10) / 10),
               fmt="{:.2f}", cbar_label="Wait [d]", invert_thr=True,
               style=style)
        P.grid_labels(ax, style)
        P.footnote(fig, "Parcel-weighted mean over all postal-code areas and "
                        f"providers. {STAGE3}.", style)
        fig.tight_layout()
        S.save(fig, "fig32_wait_grid", style, S.TIER_A)

    D.prov.write("fig32_wait_grid", title="Added-wait grid over (P, theta)",
                 tier=S.TIER_A, act=ACT, basis=STAGE3,
                 claim=f"Mean added wait stays at or below "
                       f"{piv.values.max():.2f} d across the whole grid; "
                       f"{piv.loc[0.25, 1.0]:.2f} d at the P=0.25 operating point.")


# ---------------------------------------------------------------- fig33
def fig33_fleet_grid():
    f = _drop_p04(D.load_fleet())
    sys_day = (f.groupby(["penalty", "share_willing", "day"], as_index=False)
               .agg(fb=("fleet_stage2", "sum"), fa=("fleet_stage3", "sum")))

    # Reference: the theta=0 column is the unconsolidated system (no PLZ can
    # batch), averaged over P because P has no effect when nothing may wait.
    base_day = (sys_day[np.isclose(sys_day.share_willing, 0.0)]
                .groupby("day").fb.mean())
    base = np.array([base_day.loc[d] for d in range(D.N_DAYS)])
    base_peak, base_total = float(base.max()), float(base.sum())
    # Every CV reduction on this figure is relative to THIS grid's own
    # baseline, so the baseline is measured here and stated in the caption
    # rather than asserted against another grid's value. The submission's was
    # 0.135; the revision's partition-aware fleet count gives 0.139
    # (compendium 40.18), and the rule that follows from that is: never quote
    # a CV across grids, and always print the one the figure used.
    base_cv = float(base.std() / base.mean())
    print(f"  baseline fleet: peak={base_peak:.0f} total={base_total:.0f} "
          f"cv={base_cv:.3f}  [{D.REV.name}]")
    assert 0.05 < base_cv < 0.30, (
        f"baseline fleet CV {base_cv:.3f} is outside anything this system has "
        f"produced (0.135 submitted, 0.139 in the revision) -- the fleet "
        f"reference is broken, do not trust this figure")

    rows = []
    for (pen, sh), g in sys_day.groupby(["penalty", "share_willing"]):
        fa = g.sort_values("day").fa.values
        cv = float(fa.std() / fa.mean()) if fa.mean() > 0 else 0.0
        rows.append(dict(
            penalty=pen, share_willing=sh,
            peak_red=100 * (base_peak - fa.max()) / base_peak,
            cv_red=100 * (base_cv - cv) / base_cv,
            total_chg=100 * (fa.sum() - base_total) / base_total))
    import pandas as pd
    cells = pd.DataFrame(rows)
    pk = cells.pivot(index="penalty", columns="share_willing", values="peak_red")
    cv = cells.pivot(index="penalty", columns="share_willing", values="cv_red")
    tc = cells.pivot(index="penalty", columns="share_willing", values="total_chg")

    print(f"  peak reduction at (0.5,1) = {pk.loc[0.5, 1.0]:.1f}%")
    print(f"  CV reduction max = {cv.stack().max():.1f}% at {cv.stack().idxmax()}")
    print(f"  CV reduction at (0.5,1) = {cv.loc[0.5, 1.0]:.1f}%")

    for style in S.styles():
        S.apply(style)
        fig, axes = plt.subplots(1, 3,
                                 figsize=S.figsize(style, (16.0, 4.6),
                                                   (19.0, 6.0)))
        P.heat(axes[0], pk, S.CMAP_FLEET, "(a) Peak-fleet reduction",
               vmin=float(np.floor(min(0, pk.values.min()) / 5) * 5),
               vmax=float(np.ceil(pk.values.max() / 5) * 5),
               cbar_label="Reduction [%]", style=style)
        P.heat(axes[1], cv, S.CMAP_FLEET,
               "(b) Mon–Sat fleet CV reduction",
               vmin=0, vmax=float(np.ceil(cv.values.max() / 10) * 10),
               cbar_label="Reduction [%]", style=style)
        lim = float(np.ceil(max(abs(tc.values.min()), abs(tc.values.max()))))
        P.heat(axes[2], tc, S.CMAP_CHANGE, "(c) Total weekly fleet change",
               norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
               cbar_label="Change [%]", style=style)
        for ax in axes:
            ax.set_xlabel(r"$\theta$ [%]")
        axes[0].set_ylabel(r"Service penalty $P$ [€/p/d]")
        P.footnote(fig, "Reference is the unconsolidated system at "
                        rf"$\theta=0$: peak {base_peak:.0f} vehicles, "
                        f"CV {base_cv:.3f}.", style)
        fig.tight_layout(rect=[0, 0.04, 1, 1], w_pad=1.6)
        S.save(fig, "fig33_fleet_grid", style, S.TIER_A)

    D.prov.write("fig33_fleet_grid", title="Fleet peak / CV / total over the grid",
                 tier=S.TIER_A, act=ACT, basis=STAGE3,
                 claim=f"Peak fleet falls {pk.loc[0.5, 1.0]:.1f}% at "
                       f"(P=0.5, theta=1); Mon-Sat CV falls up to "
                       f"{cv.stack().max():.1f}% across the grid. Total weekly "
                       f"fleet changes between {tc.values.min():.1f}% and "
                       f"{tc.values.max():+.1f}%.",
                 caveats=f"The theta=0 reference is a Stage-2 quantity; "
                         f"the baseline CV this figure measured on "
                         f"{D.REV.name} is {base_cv:.3f} and is printed in "
                         f"the caption -- a CV is never quoted across grids.")


# ---------------------------------------------------------------- fig34
def fig34_pareto():
    g = _drop_p04(D.saving_grid())
    w = _drop_p04(D.load_wait())
    m = g.merge(w, on=["penalty", "share_willing"])
    m = m[m.share_willing > 0].copy()  # theta=0 is the degenerate origin

    star = D.load_pstar()
    front = P.pareto_front(-m.saving_pct.values, m.avg_wait_d_stage3.values)
    m["on_front"] = front
    fr = m[m.on_front].sort_values("saving_pct")
    print(f"  {int(front.sum())} of {len(m)} cells on the efficient front")

    for style in S.styles():
        S.apply(style)
        fig, ax = plt.subplots(figsize=S.figsize(style, (7.4, 5.4), (11.0, 6.2)))
        pens = sorted(m.penalty.unique())
        cmap = plt.get_cmap(S.CMAP_PENALTY)
        _lv = (lambda k: cmap(0.05 + 0.90 * k / max(1, len(pens) - 1)))
        for k, pen in enumerate(pens):
            sub = m[np.isclose(m.penalty, pen)].sort_values("share_willing")
            ax.plot(sub.avg_wait_d_stage3, sub.saving_pct, "-o",
                    color=_lv(k), markersize=S.scale(style, 4.0, 1.5),
                    linewidth=S.scale(style, 1.2, 1.8),
                    label=rf"$P={pen:g}$", alpha=0.9, zorder=2)
        ax.plot(fr.avg_wait_d_stage3, fr.saving_pct, "--",
                color="#111111", linewidth=S.scale(style, 1.6, 1.6),
                label="Efficient front", zorder=3)

        ax.set_xlabel("Mean additional customer wait [d]")
        ax.set_ylabel("Weekly cost saving [%]")
        ax.set_title("Cost saving against service degradation")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=None, framealpha=0.9,
                  loc="lower right" if style == "paper" else "best")
        P.footnote(fig, rf"All 88 grid cells with $\theta>0$. "
                        rf"Per-provider knee points $P^*$: "
                        + ", ".join(f"{r.provider} {r.P_star:g}"
                                    for r in star.itertuples()), style)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        S.save(fig, "fig34_pareto", style, S.TIER_A)

    D.prov.write("fig34_pareto", title="Cost/wait Pareto over all grid cells",
                 tier=S.TIER_A, act=ACT, basis=STAGE3,
                 claim=f"{int(front.sum())} of {len(m)} admissible cells lie on "
                       f"the efficient front; savings up to "
                       f"{m.saving_pct.max():.1f}% are reachable at "
                       f"{m.loc[m.saving_pct.idxmax()].avg_wait_d_stage3:.2f} d "
                       f"mean added wait.")


# ---------------------------------------------------------------- fig35
def fig35_schedule_mix():
    sched = D.load_chosen_stage3()
    sched = sched[~np.isclose(sched.penalty, 0.4)].copy()
    col = "schedule_size_system_smoothed"
    pens = sorted(sched.penalty.unique())

    for style in S.styles():
        S.apply(style)
        ncols = 4
        nrows = (len(pens) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, sharex=True, sharey=True,
            figsize=S.figsize(style, (2.85 * ncols, 2.35 * nrows),
                              (3.9 * ncols, 2.9 * nrows)))
        axes = np.atleast_2d(axes)
        for idx, pen in enumerate(pens):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            sub = sched[np.isclose(sched.penalty, pen)]
            agg = (sub.groupby(["share_willing", col]).size()
                   .reset_index(name="n"))
            piv = (agg.pivot(index="share_willing", columns=col, values="n")
                   .fillna(0).sort_index())
            piv = piv.div(piv.sum(axis=1), axis=0) * 100
            x = piv.index.values * 100
            bottom = np.zeros(len(piv))
            for sz in D.FREQ_SIZES:
                if sz not in piv.columns:
                    continue
                h = piv[sz].values
                ax.fill_between(x, bottom, bottom + h,
                                color=D.FREQ_COLOR[sz], alpha=0.93)
                bottom += h
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.2)
            ax.set_title(rf"$P = {pen:g}$ €/p/d")
        for idx in range(len(pens), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)

        from matplotlib import rcParams as rcp
        sizes = sorted(set(sched[col].unique()))
        handles = [Patch(facecolor=D.FREQ_COLOR[s], label=f"{s} day/wk")
                   for s in sizes]
        # Place the shared labels and the legend off the measured axes boxes,
        # exactly as the paper's fig 4 does. A fixed supxlabel position works on
        # the tall paper canvas but lands on top of the legend at 16:9.
        fig.tight_layout(rect=[0.05, 0.16, 1, 1], pad=0.4, w_pad=0.3, h_pad=0.6)
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        cx = (axes[0, 0].get_position().x0 + axes[-1, -1].get_position().x1) / 2
        cy = (axes[-1, -1].get_position().y0 + axes[0, 0].get_position().y1) / 2
        xlab_y = axes[-1, 0].get_tightbbox(rend).transformed(inv).y0 - 0.030
        fig.text(cx, xlab_y, r"Willingness-to-wait share $\theta$ [%]",
                 ha="center", va="top", fontsize=rcp["axes.labelsize"])
        lb = axes[0, 0].get_tightbbox(rend).transformed(inv)
        fig.text(lb.x0 - 0.004, cy, "Share of postal-code areas [%]",
                 rotation=90, ha="right", va="center",
                 fontsize=rcp["axes.labelsize"])
        fig.legend(handles=handles, title="Delivery days per week",
                   loc="upper center", ncol=len(handles), frameon=True,
                   framealpha=0.9, edgecolor="0.8",
                   bbox_to_anchor=(cx, xlab_y - 0.055),
                   handlelength=1.4, columnspacing=1.3, borderpad=0.4)
        S.save(fig, "fig35_schedule_mix", style, S.TIER_A)

    D.prov.write("fig35_schedule_mix",
                 title="Delivery-frequency mix by penalty and theta",
                 tier=S.TIER_A, act=ACT, basis=STAGE3,
                 claim="Consolidation is driven jointly by willingness to wait "
                       "and the service penalty: at P=0 the 2-3 day/wk classes "
                       "dominate as theta grows, while from P=5 upward almost "
                       "every area stays at 6 day/wk.",
                 caveats="Delivery frequency is invariant from Stage 2 to "
                         "Stage 3 (asserted in load_chosen_stage3); smoothing "
                         "reassigns weekdays, not their count.")



# ---------------------------------------------------------------- fig36
def fig36_pstar_knees():
    """Per-provider knee point on the cost/wait trade-off."""
    star = D.load_pstar().sort_values("saving_pct", ascending=False)
    print(star.round(3).to_string(index=False))

    for style in S.styles():
        S.apply(style)
        fig, axes = plt.subplots(1, 2,
                                 figsize=S.figsize(style, (12.0, 4.6),
                                                   (15.0, 5.8)))
        x = np.arange(len(star))
        colors = [D.PROVIDER_COLOR[p] for p in star.provider]
        b = axes[0].bar(x, star.saving_pct, 0.6, color=colors,
                        edgecolor="#333333", linewidth=0.6)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(star.provider,
                                rotation=30 if style == "slides" else 0)
        axes[0].set_ylabel("Cost saving at the knee [%]")
        axes[0].set_title("(a) Saving each provider can reach")
        axes[0].grid(alpha=0.25, axis="y")
        P.bar_labels(axes[0], b, star.saving_pct.values, "{:.1f}", style=style)

        for r in star.itertuples():
            axes[1].scatter(r.P_star, r.wait_d,
                            s=S.scale(style, 110, 2.0),
                            color=D.PROVIDER_COLOR[r.provider],
                            edgecolor="#333333", linewidth=0.8, zorder=3)
            axes[1].annotate(r.provider, xy=(r.P_star, r.wait_d),
                             xytext=(6, 6), textcoords="offset points",
                             fontsize=9 if style == "paper" else 13)
        axes[1].set_xlabel(r"Knee penalty $P^*$ [€/parcel/day]")
        axes[1].set_ylabel("Added wait at the knee [d]")
        axes[1].set_title("(b) Where each provider's knee sits")
        axes[1].grid(alpha=0.25)
        axes[1].set_xlim(0, float(star.P_star.max()) * 1.25)
        axes[1].set_ylim(0, float(star.wait_d.max()) * 1.25)

        P.footnote(fig, "The knee is the maximum-curvature point of that "
                        r"provider's cost/wait trade-off at $\theta = 100\%$. "
                        "Providers differ by a factor of "
                        f"{star.saving_pct.max() / star.saving_pct.min():.0f} "
                        "in reachable saving.", style)
        fig.tight_layout(rect=[0, 0.07, 1, 1])
        S.save(fig, "fig36_pstar_knees", style, S.TIER_A)

    D.prov.write("fig36_pstar_knees",
                 title="Per-provider knee point on the cost/wait trade-off",
                 tier=S.TIER_A, act=ACT, basis=STAGE3,
                 claim=f"There is no single right operating point: knee "
                       f"penalties range {star.P_star.min():g}-"
                       f"{star.P_star.max():g} €/parcel/day and the saving "
                       f"reachable there ranges "
                       f"{star.saving_pct.min():.1f}% ({star.iloc[-1].provider}) "
                       f"to {star.saving_pct.max():.1f}% "
                       f"({star.iloc[0].provider}). Consolidation policy has to "
                       f"be set per provider network.",
                 caveats="Knee located by maximum chord distance on the "
                         "per-provider trade-off curve; a heuristic, not an "
                         "optimality criterion.")


def main():
    for fn in (fig31_saving_grid, fig32_wait_grid, fig33_fleet_grid,
               fig34_pareto, fig35_schedule_mix, fig36_pstar_knees):
        print(f"\n=== {fn.__name__} ===")
        fn()


if __name__ == "__main__":
    main()
