"""The four results figures, reproduced in full for one carrier at a time.

`97_provider_backup.py` compares all seven carriers in one small-multiple grid,
at a single operating point. This does the other thing: it gives each carrier
the whole figure it would have had if the study had covered only that carrier —
same panels, same scales, same colours as the aggregated original, so a slide
can be laid straight beside its aggregated twin.

    figQ1_mix_<carrier>        the eight-penalty frequency-mix figure
    figQ2_saving_<carrier>     the annotated P x theta saving grid
    figQ3_freqmap_<carrier>    delivery frequency in space, across adoption
    figQ4_savingmap_<carrier>  where the saving sits, across penalty levels

Seven carriers times four figures is 28 files, all slides rendering.

Usage:
    python scripts/presentation/98_carrier_full.py [--only FIG] [--carrier NAME]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap         # noqa: E402
from matplotlib.patches import Patch                               # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _data as D                                                  # noqa: E402
import _style as S                                                 # noqa: E402

STYLE, TIER = "slides", S.TIER_B
P_REF = 0.25
THETA_PANELS = [0.0, 0.3, 0.6, 1.0]              # as in 40_act4_maps.py
P_MAP_PANELS = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0]
ORDER = ["DHL", "Amazon", "Hermes", "UPS", "DPD", "GLS", "FedEx"]


def _carriers(df, pick=None):
    have = [p for p in ORDER if p in set(df.provider)]
    have += sorted(set(df.provider) - set(have))
    return [p for p in have if pick is None or p in pick]


def _drop_p04(df):
    return df[~np.isclose(df.penalty, 0.4)].copy()


def _view(units):
    gdf = D.load_plz_geometry()
    view = D.clip_to_scope(gdf, units)
    ids = {str(u).zfill(5) for u in units}
    view["unit"] = view.cluster_id.where(view.cluster_id.isin(ids), view.plz)
    return view


def _paint(ax, view, values, col, *, cmap, norm=None, vmin=None, vmax=None,
           title=""):
    m = view.merge(values, on="unit", how="left")
    kw = dict(cmap=cmap, edgecolor="white", linewidth=0.3,
              missing_kwds={"color": S.MISSING, "edgecolor": "white",
                            "linewidth": 0.3})
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"], kw["vmax"] = vmin, vmax
    m.plot(ax=ax, column=col, **kw)
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_aspect("equal")


def _freq_cmap():
    cmap = ListedColormap([D.FREQ_COLOR[k] for k in D.FREQ_SIZES])
    norm = BoundaryNorm([k - 0.5 for k in D.FREQ_SIZES]
                        + [D.FREQ_SIZES[-1] + 0.5], cmap.N)
    return cmap, norm


# ── Q1 · the frequency mix, all eight penalties ─────────────────────────────
def figQ1_mix_per_carrier(pick=None):
    s = _drop_p04(D.load_chosen_stage3())
    pens = sorted(s.penalty.unique())
    col = "schedule_size_system_smoothed"
    for prov in _carriers(s, pick):
        at = s[s.provider == prov]
        S.apply(STYLE)
        fig, axes = plt.subplots(2, 4, figsize=(17.2, 6.6))
        axes = axes.ravel()
        for ax, pen in zip(axes, pens):
            sub = at[np.isclose(at.penalty, pen)]
            piv = (sub.groupby(["share_willing", col]).size()
                      .unstack(fill_value=0).sort_index())
            piv = piv.div(piv.sum(axis=1), axis=0) * 100
            x = piv.index.values * 100
            bottom = np.zeros(len(piv))
            for sz in D.FREQ_SIZES:
                if sz not in piv.columns:
                    continue
                ax.fill_between(x, bottom, bottom + piv[sz].values,
                                color=D.FREQ_COLOR[sz], alpha=0.94)
                bottom += piv[sz].values
            ax.set_title(f"P = {pen:g} €/p/d")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.grid(alpha=0.25)
        for ax in axes[len(pens):]:
            ax.set_axis_off()
        handles = [Patch(facecolor=D.FREQ_COLOR[k], label=f"{k} day/wk")
                   for k in D.FREQ_SIZES]
        fig.legend(handles=handles, loc="lower center", ncol=5,
                   title="Delivery days per week", bbox_to_anchor=(0.5, 0.005))
        fig.supxlabel("Willingness-to-wait share θ [%]", y=0.165)
        fig.supylabel(f"Share of {prov} areas [%]")
        fig.suptitle(f"{prov} — chosen delivery frequency "
                     f"({at.plz.nunique()} areas)")
        fig.tight_layout(rect=[0.02, 0.21, 1, 0.94])
        S.save(fig, f"figQ1_mix_{prov}", STYLE, TIER)


# ── Q2 · the saving grid, annotated ─────────────────────────────────────────
def figQ2_saving_per_carrier(pick=None):
    c = _drop_p04(D.load_costs())
    base = D.load_baseline_per_provider().set_index("provider").dd_cost
    c["saving_pct"] = 100 * (c.provider.map(base) - c.total_stage3_eur) \
        / c.provider.map(base)
    for prov in _carriers(c, pick):
        piv = (c[c.provider == prov]
               .pivot(index="penalty", columns="share_willing",
                      values="saving_pct").sort_index(ascending=False))
        top = float(np.nanmax(piv.values))
        S.apply(STYLE)
        fig, ax = plt.subplots(figsize=(14.0, 6.4))
        im = ax.imshow(piv.values, cmap=S.CMAP_SAVING, vmin=0, vmax=top,
                       aspect="auto")
        for r in range(piv.shape[0]):
            for k in range(piv.shape[1]):
                v = piv.values[r, k]
                ax.text(k, r, f"{v:.1f}", ha="center", va="center",
                        fontsize=13,
                        color="white" if v < 0.55 * top else "black")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{v:.0%}".replace("%", "") for v in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{v:g}" for v in piv.index])
        ax.set_xlabel("Willingness-to-wait share θ [%]")
        ax.set_ylabel("Service penalty P [€/p/d]")
        best = np.unravel_index(np.nanargmax(piv.values), piv.shape)
        ax.set_title(f"{prov} — weekly cost saving against its own "
                     f"daily-delivery baseline\n"
                     f"best {top:.1f} % at P = {piv.index[best[0]]:g}, "
                     f"θ = {piv.columns[best[1]]:.0%}")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Saving [%]")
        fig.tight_layout()
        S.save(fig, f"figQ2_saving_{prov}", STYLE, TIER)


# ── Q3 · frequency in space, across adoption ────────────────────────────────
def figQ3_freqmap_per_carrier(pick=None):
    s = D.load_chosen_stage3()
    view = _view(s.plz.unique())
    cmap, norm = _freq_cmap()
    for prov in _carriers(s, pick):
        at = s[s.provider == prov]
        S.apply(STYLE)
        fig, axes = plt.subplots(1, len(THETA_PANELS), figsize=(16.4, 5.4))
        for ax, th in zip(np.atleast_1d(axes), THETA_PANELS):
            sub = (at[np.isclose(at.penalty, P_REF)
                      & np.isclose(at.share_willing, th)]
                   [["plz", "schedule_size_system_smoothed"]]
                   .rename(columns={"plz": "unit",
                                    "schedule_size_system_smoothed": "freq"}))
            sub["unit"] = sub.unit.astype(str).str.zfill(5)
            med = float(sub.freq.median()) if len(sub) else float("nan")
            _paint(ax, view, sub, "freq", cmap=cmap, norm=norm,
                   title=f"θ = {th:.0%}\nmedian {med:.0f} day/wk")
        handles = [Patch(facecolor=D.FREQ_COLOR[k], label=f"{k} day/wk")
                   for k in D.FREQ_SIZES]
        fig.legend(handles=handles, loc="lower center", ncol=5,
                   title="Median delivery frequency",
                   bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(f"{prov} — delivery frequency per area at "
                     f"P = {P_REF:g} €/p/d, by adoption θ")
        fig.tight_layout(rect=[0, 0.13, 1, 0.93])
        S.save(fig, f"figQ3_freqmap_{prov}", STYLE, TIER)


# ── Q4 · saving in space, across penalty ────────────────────────────────────
def figQ4_savingmap_per_carrier(pick=None):
    d = D.load_per_plz()
    view = _view(d.plz.unique())
    base = D.load_baseline_per_provider().set_index("provider").dd_cost
    costs = D.load_costs()
    for prov in _carriers(d, pick):
        at = d[d.provider == prov]
        vmax = float(np.nanpercentile(at.saving_pct, 98))
        S.apply(STYLE)
        fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.6))
        axes = axes.ravel()
        for ax, pen in zip(axes, P_MAP_PANELS):
            sub = (at[np.isclose(at.penalty, pen)][["plz", "saving_pct"]]
                   .rename(columns={"plz": "unit"}))
            sub["unit"] = sub.unit.astype(str).str.zfill(5)
            row = costs[np.isclose(costs.penalty, pen)
                        & np.isclose(costs.share_willing, 1.0)
                        & (costs.provider == prov)]
            net = (100 * (base[prov] - float(row.total_stage3_eur.iloc[0]))
                   / base[prov]) if len(row) else float("nan")
            _paint(ax, view, sub, "saving_pct", cmap="Greens", vmin=0,
                   vmax=vmax, title=f"P = {pen:g}\nnetwork {net:.1f} %")
        for ax in axes[len(P_MAP_PANELS):]:
            ax.set_axis_off()
        sm = plt.cm.ScalarMappable(cmap="Greens",
                                   norm=plt.Normalize(vmin=0, vmax=vmax))
        fig.colorbar(sm, ax=axes.tolist(), fraction=0.02, pad=0.02,
                     label="Cost saving per area [%]")
        fig.suptitle(f"{prov} — where consolidation pays, θ = 100 %")
        fig.tight_layout(rect=[0, 0, 0.93, 0.93])
        fig.subplots_adjust(hspace=0.30)
        S.save(fig, f"figQ4_savingmap_{prov}", STYLE, TIER)


FIGURES = {f.__name__: f for f in (
    figQ1_mix_per_carrier, figQ2_saving_per_carrier,
    figQ3_freqmap_per_carrier, figQ4_savingmap_per_carrier)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="figure families to build")
    ap.add_argument("--carrier", nargs="*", help="limit to these carriers")
    a = ap.parse_args()
    todo = [FIGURES[n] for n in a.only] if a.only else list(FIGURES.values())
    for fn in todo:
        print(f"\n=== {fn.__name__} ===")
        fn(a.carrier)
    return 0




# ── B5 · why 10 % beats 20 %, in three bars ─────────────────────────────────
def figB5_prize_and_bill(pick=None):
    """What the region saves, and what it gives up to keep waiting short.

    The fee is never added to anybody's cost -- the reported total is routing
    money only. It steers which schedule wins, nothing else. So the grey part
    of each bar is not a bill: it is real routing saving the model declines in
    order to keep the wait down. As participation rises the same fee rate
    applies to more waiting parcels, the steering hardens, and the model
    retreats to shorter waits and thinner savings.
    """
    c = D.load_costs()
    tot = (c.groupby(["penalty", "share_willing"], as_index=False)
             .total_stage3_eur.sum()
             .merge(D.load_wait(), on=["penalty", "share_willing"]))
    tot["saved"] = D.BASE_TOTAL - tot.total_stage3_eur
    free = tot[np.isclose(tot.penalty, 0.0)].set_index("share_willing").saved
    thetas = [0.1, 0.2, 0.3]
    prize = [float(free.loc[t]) for t in thetas]
    kept = [float(tot[np.isclose(tot.penalty, 10.0)
                      & np.isclose(tot.share_willing, t)].saved.iloc[0])
            for t in thetas]
    cost = [p - k for p, k in zip(prize, kept)]
    sched = D.load_chosen_stage3()
    col = "schedule_size_system_smoothed"
    share = [100 * float((sched[np.isclose(sched.penalty, 10.0)
                                & np.isclose(sched.share_willing, t)][col]
                          < 6).mean()) for t in thetas]
    print(f"  prize {[f'{v:,.0f}' for v in prize]}")
    print(f"  kept  {[f'{v:,.0f}' for v in kept]}")
    print(f"  given up  {[f'{v:,.0f}' for v in cost]} — "
          f"×{cost[1] / cost[0]:.2f} from 10 % to 20 %, while the prize grows "
          f"×{prize[1] / prize[0]:.2f}")

    S.apply(STYLE)
    fig, ax = plt.subplots(figsize=(14.0, 6.4))
    x = np.arange(len(thetas))
    ax.bar(x, kept, 0.52, color="#00B050",
           label="what the region actually saves")
    ax.bar(x, cost, 0.52, bottom=kept, color=S.GRID,
           label="saving it gives up to keep the wait short")
    for i, (p, k) in enumerate(zip(prize, kept)):
        ax.text(i, p + 4000, f"{p:,.0f} €".replace(",", " "), ha="center",
                fontsize=16, fontweight="bold")
        if k > 6000:
            ax.text(i, k / 2, f"{k:,.0f} €".replace(",", " "), ha="center",
                    va="center", fontsize=15, color="white", fontweight="bold")
        else:
            ax.text(i, k + 6000, f"{k:,.0f} €".replace(",", " "), ha="center",
                    fontsize=15, color="#00B050", fontweight="bold")
    # the retreat is the point, so the wait belongs on the axis, not floating
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.0%} join in" + chr(10)
                        + f"{v:.1f} % of areas still bundle"
                        for t, v in zip(thetas, share)])
    ax.set_ylabel("€ per week")
    ax.set_ylim(0, max(prize) * 1.22)
    ax.legend(loc="upper left", fontsize=15)
    ax.grid(alpha=0.25, axis="y")
    ax.annotate("", xy=(2.32, prize[2]), xytext=(2.32, prize[0]),
                arrowprops=dict(arrowstyle="<->", color=S.INK, lw=2.0))
    ax.text(2.40, (prize[0] + prize[2]) / 2,
            f"what could be saved if\nwaiting counted for nothing\n"
            f"— grows only ×{prize[2] / prize[0]:.2f}", fontsize=14,
            va="center")
    ax.set_title("The more people wait, the harder the fee steers — and the "
                 "less is left\nWhole region, per week, at the harshest fee "
                 "(P = 10 €/p/d). The fee itself is never paid by anyone.")
    fig.tight_layout()
    S.save(fig, "figB5_prize_and_bill", STYLE, TIER)


FIGURES["figB5_prize_and_bill"] = figB5_prize_and_bill


if __name__ == "__main__":
    raise SystemExit(main())
