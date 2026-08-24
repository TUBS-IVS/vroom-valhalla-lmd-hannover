"""Act 5 -- operational and managerial consequences, Stage-3 basis.

Figures
  fig51_fleet_smoothing   weekly fleet profile Mon-Sat, Stage 2 vs Stage 3
  fig52_fleet_per_provider per-provider weekly fleet profile
  fig53_co2               vehicle-km and CO2 across the validated points
  fig54_load_factor       parcels per route and cost per parcel per provider
  fig55_pattern_clock     which weekdays the chosen schedules actually use

fig55 is the one figure here that is genuinely *new* at Stage 3 rather than a
re-render: system smoothing preserves how many days each area is served but
reassigns which ones, so the weekday distribution is a Stage-3 quantity and
differs from the Stage-2 choice it was derived from. The figure shows both.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import _data as D
import _plots as P
import _style as S

ACT = "5 - Operational"
STAGE3 = "Stage 3 (per-hub balancing + system smoothing)"

P_REF = 0.25
THETA_REF = 1.0


def _cell(df, penalty=P_REF, theta=THETA_REF):
    sub = df[np.isclose(df.penalty, penalty) & np.isclose(df.share_willing, theta)]
    assert len(sub), f"no rows at P={penalty}, theta={theta}"
    return sub


# ---------------------------------------------------------------- fig51
def fig51_fleet_smoothing():
    f = D.load_fleet()
    sub = _cell(f)
    day = (sub.groupby("day", as_index=False)
           .agg(s2=("fleet_stage2", "sum"), s3=("fleet_stage3", "sum")))

    ref = f[np.isclose(f.share_willing, 0.0)]
    ref_day = (ref.groupby(["day"], as_index=False)
               .agg(base=("fleet_stage2", "mean")))
    # theta=0 has one row per (P, provider, hub, day); sum providers/hubs then
    # average over P, matching the reference used in fig33.
    ref_day = (ref.groupby(["penalty", "day"], as_index=False)
               .agg(v=("fleet_stage2", "sum"))
               .groupby("day", as_index=False).agg(base=("v", "mean")))
    day = day.merge(ref_day, on="day")

    cv = lambda a: float(np.std(a) / np.mean(a))
    print(f"  baseline peak {day.base.max():.0f} cv {cv(day.base):.3f}")
    print(f"  stage2   peak {day.s2.max():.0f} cv {cv(day.s2):.3f}")
    print(f"  stage3   peak {day.s3.max():.0f} cv {cv(day.s3):.3f}")

    for style in S.styles():
        S.apply(style)
        fig, ax = plt.subplots(figsize=S.figsize(style, (7.6, 4.6), (11.5, 5.8)))
        x = np.arange(D.N_DAYS)
        w = 0.27
        b1 = ax.bar(x - w, day.base, w, label="Daily delivery (baseline)",
                    color=D.PALETTE["baseline"])
        b2 = ax.bar(x, day.s2, w, label="Stage 2 (per-hub balancing)",
                    color=D.PALETTE["stage2"])
        b3 = ax.bar(x + w, day.s3, w, label="Stage 3 (+ system smoothing)",
                    color=D.PALETTE["stage3"])
        ax.set_xticks(x)
        ax.set_xticklabels(D.WEEKDAYS)
        ax.set_ylabel("Vehicles required")
        ax.set_title(rf"Weekly fleet profile at $P = {P_REF:g}$ €/p/d, "
                     rf"$\theta = {THETA_REF * 100:.0f}\%$")
        ax.grid(alpha=0.25, axis="y")
        ax.legend(loc="lower right", framealpha=0.9)
        P.footnote(fig, f"Coefficient of variation across Mon–Sat: "
                        f"{cv(day.base):.3f} baseline → {cv(day.s2):.3f} "
                        f"Stage 2 → {cv(day.s3):.3f} Stage 3.", style)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        S.save(fig, "fig51_fleet_smoothing", style, S.TIER_A)

    D.prov.write("fig51_fleet_smoothing",
                 title="Weekly fleet profile, baseline vs stage 2 vs stage 3",
                 tier=S.TIER_A, act=ACT, basis=STAGE3,
                 claim=f"System smoothing flattens the weekly fleet requirement "
                       f"without changing any delivery frequency: CV falls "
                       f"{cv(day.base):.3f} → {cv(day.s2):.3f} → "
                       f"{cv(day.s3):.3f} and the peak from "
                       f"{day.base.max():.0f} to {day.s3.max():.0f} vehicles.")


# ---------------------------------------------------------------- fig52
def fig52_fleet_per_provider():
    f = D.load_fleet()
    sub = _cell(f)
    per = (sub.groupby(["provider", "day"], as_index=False)
           .agg(s2=("fleet_stage2", "sum"), s3=("fleet_stage3", "sum")))

    for style in S.styles():
        S.apply(style)
        ncols, nrows = 4, 2
        fig, axes = plt.subplots(
            nrows, ncols, sharex=True,
            figsize=S.figsize(style, (3.1 * ncols, 2.5 * nrows),
                              (4.0 * ncols, 3.0 * nrows)))
        for idx, provider in enumerate(D.PROVIDERS):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            g = per[per.provider == provider].sort_values("day")
            x = np.arange(D.N_DAYS)
            ax.bar(x - 0.2, g.s2, 0.4, color=D.PALETTE["stage2"],
                   label="Stage 2")
            ax.bar(x + 0.2, g.s3, 0.4, color=D.PALETTE["stage3"],
                   label="Stage 3")
            ax.set_xticks(x)
            ax.set_xticklabels(D.WEEKDAYS, rotation=45 if style == "slides" else 0)
            ax.set_title(provider)
            ax.grid(alpha=0.2, axis="y")
            if idx == 0:
                ax.set_ylabel("Vehicles")
        axes[1, 3].axis("off")
        h, lb = axes[0, 0].get_legend_handles_labels()
        axes[1, 3].legend(h, lb, loc="center", frameon=True, framealpha=0.9)
        fig.suptitle(rf"Weekly fleet per provider at $P = {P_REF:g}$ €/p/d, "
                     rf"$\theta = {THETA_REF * 100:.0f}\%$")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        S.save(fig, "fig52_fleet_per_provider", style, S.TIER_B)

    D.prov.write("fig52_fleet_per_provider",
                 title="Per-provider weekly fleet profile",
                 tier=S.TIER_B, act=ACT, basis=STAGE3,
                 claim="Smoothing acts within each provider's own hub network; "
                       "no vehicles or parcels are shared between providers.")


# ---------------------------------------------------------------- fig53
def fig53_co2():
    v = D.load_vroom()
    per = (v.groupby(["penalty"], as_index=False)
           .agg(km=("vroom_distance_km", "sum"),
                routes=("vroom_n_routes", "sum"),
                parcels=("vroom_n_parcels", "sum"),
                cost=("vroom_cost_eur", "sum")))
    per["co2_t"] = per.km * D.CO2_KG_PER_KM / 1000.0
    per["km_per_parcel"] = per.km / per.parcels

    # The highest-penalty validated cell is the least-consolidated one and acts
    # as the in-sample reference; there is no VROOM baseline solve on this
    # demand allocation, so all deltas are stated relative to it, not to daily
    # delivery.
    ref = per[np.isclose(per.penalty, per.penalty.max())].iloc[0]
    per["km_vs_ref_pct"] = (per.km / ref.km - 1.0) * 100.0
    print(per[["penalty", "km", "co2_t", "km_per_parcel",
               "km_vs_ref_pct"]].round(3).to_string(index=False))

    for style in S.styles():
        S.apply(style)
        fig, axes = plt.subplots(1, 2,
                                 figsize=S.figsize(style, (11.0, 4.4),
                                                   (14.0, 5.6)))
        x = np.arange(len(per))
        bars = axes[0].bar(x, per.km / 1000.0, 0.6, color=D.PALETTE["stage3"])
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"{p:g}" for p in per.penalty])
        axes[0].set_xlabel(r"Service penalty $P$ [€/p/d]")
        axes[0].set_ylabel("Weekly vehicle distance [1000 km]")
        axes[0].set_title("(a) Vehicle-kilometres")
        axes[0].grid(alpha=0.25, axis="y")
        P.bar_labels(axes[0], bars, (per.km / 1000.0).values, "{:.1f}",
                     style=style)

        bars2 = axes[1].bar(x, per.co2_t, 0.6, color=D.PALETTE["accent2"])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"{p:g}" for p in per.penalty])
        axes[1].set_xlabel(r"Service penalty $P$ [€/p/d]")
        axes[1].set_ylabel("Weekly CO$_2$ [t]")
        axes[1].set_title("(b) Tailpipe CO$_2$")
        axes[1].grid(alpha=0.25, axis="y")
        P.bar_labels(axes[1], bars2, per.co2_t.values, "{:.1f}", style=style)

        P.footnote(fig, f"Real VROOM routes on the Stage-3 schedules, "
                        rf"$\theta = 100\%$. CO$_2$ at "
                        f"{D.CO2_KG_PER_KM:g} kg per vehicle-km (external "
                        f"assumption, not a model output). Distances are "
                        f"absolute weekly totals for the four validated "
                        f"operating points.", style)
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        S.save(fig, "fig53_co2", style, S.TIER_A)

    D.prov.write("fig53_co2", title="Vehicle-km and CO2 at the validated points",
                 tier=S.TIER_A, act=ACT, basis="Stage-3 VROOM revalidation",
                 claim=f"Consolidation cuts distance as well as cost: weekly "
                       f"vehicle-km fall from {ref.km / 1000:.1f}k at "
                       f"P={ref.penalty:g} to {per.km.min() / 1000:.1f}k at "
                       f"P={per.loc[per.km.idxmin()].penalty:g}, i.e. "
                       f"{abs(per.km_vs_ref_pct.min()):.1f}% less distance and "
                       f"{ref.co2_t - per.co2_t.min():.1f} t less CO2 per week.",
                 caveats="Deltas are against the least-consolidated validated "
                         "cell (P=0.75), not against daily delivery: no VROOM "
                         "baseline solve exists on this demand allocation. The "
                         "CO2 factor is an external assumption. "
                         + (D.vroom_partial_note(v) + " (DHL/30855 at P=0), so "
                            "the P=0 distance is marginally understated."
                            if D.vroom_partial_note(v) else ""))


# ---------------------------------------------------------------- fig54
def fig54_load_factor():
    v = D.load_vroom()
    per = (v.groupby(["penalty", "provider"], as_index=False)
           .agg(km=("vroom_distance_km", "sum"),
                routes=("vroom_n_routes", "sum"),
                parcels=("vroom_n_parcels", "sum"),
                cost=("vroom_cost_eur", "sum")))
    per["parcels_per_route"] = per.parcels / per.routes
    per["eur_per_parcel"] = per.cost / per.parcels

    for style in S.styles():
        S.apply(style)
        fig, axes = plt.subplots(1, 2,
                                 figsize=S.figsize(style, (12.0, 4.6),
                                                   (15.0, 5.8)))
        pens = sorted(per.penalty.unique())
        x = np.arange(len(D.PROVIDERS))
        w = 0.8 / len(pens)
        cmap = plt.get_cmap(S.CMAP_PENALTY)
        _lv = (lambda k: cmap(0.05 + 0.90 * k / max(1, len(pens) - 1)))
        for k, pen in enumerate(pens):
            g = (per[np.isclose(per.penalty, pen)]
                 .set_index("provider").reindex(D.PROVIDERS))
            off = (k - (len(pens) - 1) / 2) * w
            axes[0].bar(x + off, g.parcels_per_route, w, color=_lv(k),
                        label=rf"$P={pen:g}$")
            axes[1].bar(x + off, g.eur_per_parcel, w, color=_lv(k),
                        label=rf"$P={pen:g}$")
        for ax, ylab, title in (
                (axes[0], "Parcels per route", "(a) Vehicle load factor"),
                (axes[1], "Cost per parcel [€]", "(b) Unit cost")):
            ax.set_xticks(x)
            ax.set_xticklabels(D.PROVIDERS,
                               rotation=30 if style == "slides" else 0)
            ax.set_ylabel(ylab)
            ax.set_title(title)
            ax.grid(alpha=0.25, axis="y")
        axes[0].legend(ncol=2, framealpha=0.9, fontsize=None)
        P.footnote(fig, r"Real VROOM routes on the Stage-3 schedules, "
                        r"$\theta = 100\%$.", style)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        S.save(fig, "fig54_load_factor", style, S.TIER_B)

    lo = per[np.isclose(per.penalty, 0.0)]
    D.prov.write("fig54_load_factor",
                 title="Load factor and unit cost per provider",
                 tier=S.TIER_B, act=ACT, basis="Stage-3 VROOM revalidation",
                 claim=f"Consolidation raises vehicle utilisation: at P=0 "
                       f"parcels per route span "
                       f"{lo.parcels_per_route.min():.0f}-"
                       f"{lo.parcels_per_route.max():.0f} across providers, at "
                       f"unit costs of {lo.eur_per_parcel.min():.2f}-"
                       f"{lo.eur_per_parcel.max():.2f} € per parcel.")


# ---------------------------------------------------------------- fig55
def fig55_pattern_clock():
    s2 = pd.read_csv(D.RUN / "tab_chosen_schedules.csv", dtype={"plz": str})
    D.prov.record(D.RUN / "tab_chosen_schedules.csv")
    s3 = D.load_chosen_stage3()

    # The weekday columns hold day *names* ("Mon,Tue,Sat"), not day indices.
    idx_of = {name: i for i, name in enumerate(D.WEEKDAYS)}

    def _weekday_share(df, col, penalty, theta):
        sub = df[np.isclose(df.penalty, penalty)
                 & np.isclose(df.share_willing, theta)]
        assert len(sub), f"no rows at P={penalty}, theta={theta}"
        counts = np.zeros(D.N_DAYS)
        for spec in sub[col].astype(str):
            for token in spec.split(","):
                token = token.strip()
                if not token:
                    continue
                assert token in idx_of, (
                    f"unparsable weekday {token!r} in column {col!r}; "
                    f"expected one of {D.WEEKDAYS}")
                counts[idx_of[token]] += 1
        share = counts / len(sub) * 100.0
        assert share.sum() > 0, f"column {col!r} parsed to an all-zero profile"
        return share

    col2 = "weekdays_balanced" if "weekdays_balanced" in s2.columns else None
    if col2 is None:
        cands = [c for c in s2.columns if "weekday" in c]
        assert cands, f"no weekday column in stage-2 table: {list(s2.columns)}"
        col2 = cands[0]
    share2 = _weekday_share(s2, col2, P_REF, THETA_REF)
    share3 = _weekday_share(s3, "weekdays_system_smoothed", P_REF, THETA_REF)
    shifted = float(np.abs(share3 - share2).sum() / 2)
    print(f"  stage2 weekday share: {np.round(share2, 1)}")
    print(f"  stage3 weekday share: {np.round(share3, 1)}")
    print(f"  net reassignment: {shifted:.1f} percentage points of area-days")

    for style in S.styles():
        S.apply(style)
        fig, ax = plt.subplots(figsize=S.figsize(style, (7.6, 4.4), (11.5, 5.6)))
        x = np.arange(D.N_DAYS)
        ax.bar(x - 0.2, share2, 0.4, color=D.PALETTE["stage2"],
               label="Stage 2 (cost-optimal per hub)")
        ax.bar(x + 0.2, share3, 0.4, color=D.PALETTE["stage3"],
               label="Stage 3 (after system smoothing)")
        ax.set_xticks(x)
        ax.set_xticklabels(D.WEEKDAYS)
        ax.set_ylabel("Areas served on that weekday [%]")
        ax.set_title(rf"Which weekdays the schedules use, $P = {P_REF:g}$ €/p/d, "
                     rf"$\theta = {THETA_REF * 100:.0f}\%$")
        ax.grid(alpha=0.25, axis="y")
        ax.legend(framealpha=0.9, loc="lower right")
        P.footnote(fig, f"Smoothing reassigns weekdays without changing any "
                        f"area's delivery frequency: "
                        f"{shifted:.1f} pp of area-days move.", style)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        S.save(fig, "fig55_pattern_clock", style, S.TIER_A)

    D.prov.write("fig55_pattern_clock",
                 title="Weekday usage, stage 2 vs stage 3",
                 tier=S.TIER_A, act=ACT, basis=STAGE3,
                 claim=f"System smoothing works purely by moving delivery days "
                       f"between weekdays: {shifted:.1f} pp of area-days are "
                       f"reassigned while every area's frequency is unchanged. "
                       f"This is what buys the fleet-CV reduction at zero cost "
                       f"in service frequency.",
                 caveats="Weekday shares sum to more than 100% because an area "
                         "is served on several days per week.")


def main():
    for fn in (fig51_fleet_smoothing, fig52_fleet_per_provider, fig53_co2,
               fig54_load_factor, fig55_pattern_clock):
        print(f"\n=== {fn.__name__} ===")
        fn()


if __name__ == "__main__":
    main()
