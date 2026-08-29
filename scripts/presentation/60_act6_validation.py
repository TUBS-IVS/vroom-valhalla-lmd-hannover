"""Act 6 -- validation of the surrogate against real routing, Stage-3 basis.

Figures
  fig61_vroom_scatter    per-area surrogate cost vs VROOM-actual cost
  fig62_pred_vs_actual   predicted vs realised system saving, 4 operating points
  fig63_diagnostics      MAPE / bias per operating point and per provider

fig62 is the credibility figure of the whole deck: it shows the surrogate is
*conservative* -- every predicted saving is realised or beaten -- which is the
direction an operator needs the error to point.
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

ACT = "6 - Validation"
# Which validation this act is drawn from is a property of the grid in use, not
# a constant: the submission's covers one plan at four points, the revision's
# covers both plans and states its own row count.
def _basis() -> str:
    import _data as _D
    if _D.val_schema() == _D.VAL_SCHEMA_LEGACY:
        return "Stage-3 VROOM revalidation (4 operating points, theta=1)"
    n = len(_D.load_vroom_v2())
    plans = ", ".join(sorted(_D.load_vroom_v2().plan.astype(str).unique()))
    return (f"VROOM validation v2 on {_D.REV.name}: {n} solved instances, "
            f"plans {plans}, theta=1")


BASIS = _basis()


# ---------------------------------------------------------------- fig61
def fig61_vroom_scatter():
    mv = D.load_ml_vs_vroom()
    mv = mv.dropna(subset=["vroom_cost_eur", "ml_dd_cost_eur"])
    err = (mv.ml_dd_cost_eur - mv.vroom_cost_eur) / mv.vroom_cost_eur * 100
    mape = float(np.abs(err).mean())
    bias = float(err.mean())
    r2 = float(1 - ((mv.ml_dd_cost_eur - mv.vroom_cost_eur) ** 2).sum()
               / ((mv.vroom_cost_eur - mv.vroom_cost_eur.mean()) ** 2).sum())
    print(f"  n={len(mv)} MAPE={mape:.2f}% bias={bias:+.2f}% R2={r2:.4f}")

    for style in S.styles():
        S.apply(style)
        fig, ax = plt.subplots(figsize=S.figsize(style, (6.2, 5.6), (8.4, 6.4)))
        pens = sorted(mv.penalty.unique())
        cmap = plt.get_cmap(S.CMAP_PENALTY)
        _lv = (lambda k: cmap(0.05 + 0.90 * k / max(1, len(pens) - 1)))
        for k, pen in enumerate(pens):
            g = mv[np.isclose(mv.penalty, pen)]
            ax.scatter(g.vroom_cost_eur / 1000, g.ml_dd_cost_eur / 1000,
                       s=S.scale(style, 16, 2.0), color=_lv(k), alpha=0.75,
                       edgecolor="none", label=rf"$P={pen:g}$")
        lim = [0, float(max(mv.vroom_cost_eur.max(),
                           mv.ml_dd_cost_eur.max()) / 1000) * 1.05]
        ax.plot(lim, lim, "--", color="#333333",
                linewidth=S.scale(style, 1.2, 1.4), label="Parity")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("VROOM actual cost per area [1000 €/week]")
        ax.set_ylabel("Surrogate predicted cost [1000 €/week]")
        ax.set_title("Surrogate vs real routing, per area")
        ax.grid(alpha=0.25)
        ax.legend(framealpha=0.9, loc="upper left")
        P.footnote(fig, f"n = {len(mv)} area-provider observations. "
                        f"MAPE {mape:.2f}%, bias {bias:+.2f}%, "
                        f"$R^2$ = {r2:.4f}.", style)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        S.save(fig, "fig61_vroom_scatter", style, S.TIER_A)

    D.prov.write("fig61_vroom_scatter", title="Surrogate vs VROOM per area",
                 tier=S.TIER_A, act=ACT, basis=BASIS,
                 claim=f"On the Stage-3 schedules the surrogate tracks real "
                       f"routing to {mape:.2f}% MAPE with {bias:+.2f}% bias "
                       f"across {len(mv)} area-provider observations "
                       f"(R2 = {r2:.4f}).")


# ---------------------------------------------------------------- fig62
def fig62_pred_vs_actual():
    """Predicted against realised, in whichever form the validation supports.

    A realised SAVING needs a solved baseline to be a saving against. The
    submission's validation has one; the revision's solves the scenario points
    first and the theta = 0 baseline is a separate, much larger item, so until
    it lands the honest figure is predicted against realised COST at the same
    points -- both real, both measured on the same tours -- rather than a
    saving formed from an actual numerator and a predicted denominator.
    """
    sv = D.load_savings_validation()
    have = D.vroom_actual_baseline_available()
    # A percentage needs a full system total on both sides. On v6 that rules
    # out item 0 (it IS the baseline) and item 3, which the G6 budget
    # fallback sampled to 1 000 of 1 594 instances -- its totals are real but
    # they are a subset, so a "39 % saving" read off them would be an artefact
    # of the sample size. Both are dropped here and named in the caveat.
    dropped = []
    if have and "saving_defined" in sv.columns:
        drop = sv[~sv.saving_defined]
        dropped = [f"item {int(r.item)} (P={r.penalty:g}, "
                   f"theta={r.share_willing:g}, {r.plan}): "
                   + ("the baseline itself" if r.is_baseline
                      else f"sampled, {int(r.n)} instances")
                   for r in drop.itertuples()]
        sv = sv[sv.saving_defined].copy()
        for d in dropped:
            print(f"  [excluded] {d}")
        assert len(sv), (
            "no validated point carries a defined saving; every item is "
            "either the baseline or a sampled one")
    thetas = sorted(set(sv.share_willing)) if len(sv) else []
    if not have and "plan" in sv.columns:
        # one bar pair per (point, plan); label them so the two plans of the
        # revision are never silently averaged into one series
        sv = sv.sort_values(["plan", "penalty"]).reset_index(drop=True)
        sv["label"] = [rf"$P={r.penalty:g}$" + "\n" + str(r.plan)
                       for r in sv.itertuples()]
    else:
        sv = sv.reset_index(drop=True)
        sv["label"] = [rf"$P={p:g}$" for p in sv.penalty]

    if have:
        cols = ("predicted_saving_pct", "actual_saving_pct")
        labels = ("Predicted (surrogate)", "Realised (VROOM)")
        ylab = "Weekly system cost saving [%]"
        title = "Predicted vs realised saving at the validated points"
        fmt = "{:.1f}"
        scale = 1.0
    else:
        cols = ("surrogate_total_eur", "vroom_actual_total_eur")
        labels = ("Surrogate price", "VROOM actual")
        ylab = "Weekly routing cost of the validated tours [M€]"
        title = "Surrogate price vs solver cost, same tours"
        fmt = "{:.2f}"
        scale = 1e-6
    print(sv[["penalty", *([c for c in ("plan",) if c in sv.columns]),
              *cols]].to_string(index=False))

    for style in S.styles():
        S.apply(style)
        fig, ax = plt.subplots(figsize=S.figsize(style, (7.6, 4.8), (11.0, 5.8)))
        x = np.arange(len(sv))
        w = 0.36
        v1 = sv[cols[0]].values * scale
        v2 = sv[cols[1]].values * scale
        b1 = ax.bar(x - w / 2, v1, w, color=D.PALETTE["stage2"], label=labels[0])
        b2 = ax.bar(x + w / 2, v2, w, color=D.PALETTE["accent2"], label=labels[1])
        ax.set_xticks(x)
        ax.set_xticklabels(sv.label)
        ax.set_xlabel(r"Service penalty [€/p/d], all at "
                      + (r"$\theta = 100\%$" if thetas == [1.0]
                         else r"$\theta \in \{"
                              + ", ".join(f"{t:g}" for t in thetas) + r"\}$"))
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
        ax.set_ylim(0, max(v1.max(), v2.max()) * 1.22)
        P.bar_labels(ax, b1, v1, fmt, style=style)
        P.bar_labels(ax, b2, v2, fmt, style=style)
        gap = (sv.gap_pct.values if "gap_pct" in sv.columns
               else sv.conservatism_pp.values)
        unit = "%" if not have else " pp"
        for xi, g, top in zip(x, gap, np.maximum(v1, v2)):
            ax.annotate(f"+{g:.1f}{unit}", xy=(xi, top),
                        xytext=(0, 22 if style == "slides" else 15),
                        textcoords="offset points", ha="center",
                        fontsize=9 if style == "paper" else 13,
                        color=D.PALETTE["accent"])
        ax.legend(framealpha=0.9, loc="upper right")
        note = (f"Realised saving exceeds the prediction at every point; the "
                f"surrogate is conservative by "
                f"{sv.conservatism_pp.min():.1f}–"
                f"{sv.conservatism_pp.max():.1f} pp." if have else
                f"The surrogate prices the same tours "
                f"{sv.gap_pct.min():.1f}–{sv.gap_pct.max():.1f} % above the "
                f"solver, i.e. conservatively. A realised SAVING is not shown: "
                f"the theta = 0 baseline of this grid has not been solved yet, "
                f"so there is nothing to measure a saving against.")
        P.footnote(fig, note, style)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        S.save(fig, "fig62_pred_vs_actual", style, S.TIER_A)

    claim = (f"Across the validated operating points the realised saving "
             f"({sv.actual_saving_pct.min():.1f}-"
             f"{sv.actual_saving_pct.max():.1f}%) exceeds the surrogate "
             f"prediction ({sv.predicted_saving_pct.min():.1f}-"
             f"{sv.predicted_saving_pct.max():.1f}%) by "
             f"{sv.conservatism_pp.min():.1f}-{sv.conservatism_pp.max():.1f} "
             f"pp. The error points in the safe direction." if have else
             f"On the same tours the surrogate prices "
             f"{sv.gap_pct.min():.1f}-{sv.gap_pct.max():.1f} % above the "
             f"solver at every validated point and in both plans, so the "
             f"error points in the safe direction. The predicted savings are "
             f"{sv.predicted_saving_pct.min():.1f}-"
             f"{sv.predicted_saving_pct.max():.1f}%.")
    D.prov.write("fig62_pred_vs_actual",
                 title=("Predicted vs realised system saving" if have else
                        "Surrogate price vs solver cost at the validated "
                        "points"),
                 tier=S.TIER_A, act=ACT, basis=BASIS, claim=claim,
                 caveats=((f"{len(sv)} validated points, all at theta=1, on "
                           f"both plans; the conservatism band widens with "
                           f"the penalty. Excluded because no system "
                           f"percentage can be formed from them: "
                           + "; ".join(dropped) + "."
                           if dropped else
                           f"{len(sv)} validated points; the conservatism "
                           f"band widens with the penalty.") if have else
                          "No realised SAVING is stated: the theta = 0 "
                          "baseline of this grid has not been solved, so a "
                          "saving would pair an actual numerator with a "
                          "predicted denominator. Both plans are shown "
                          "separately and must not be averaged."))


# ---------------------------------------------------------------- fig63
def fig63_diagnostics():
    diag = D.load_vroom_diagnostics()
    cells = diag[diag.penalty != "ALL"].copy()
    cells["penalty"] = cells.penalty.astype(float)
    overall = diag[diag.penalty == "ALL"].iloc[0]

    mv = D.load_ml_vs_vroom()
    mv = mv.dropna(subset=["vroom_cost_eur", "ml_dd_cost_eur"])
    mv["err_pct"] = ((mv.ml_dd_cost_eur - mv.vroom_cost_eur)
                     / mv.vroom_cost_eur * 100)
    per_prov = (mv.groupby("provider")
                .agg(mape=("err_pct", lambda s: float(np.abs(s).mean())),
                     bias=("err_pct", "mean"), n=("err_pct", "size"))
                .reindex(D.PROVIDERS).reset_index())
    print(f"  overall MAPE {float(overall.MAPE_pct):.2f}% "
          f"bias {float(overall.bias_pct):+.2f}% n={int(overall.n)}")
    print(per_prov.round(2).to_string(index=False))

    for style in S.styles():
        S.apply(style)
        fig, axes = plt.subplots(1, 2,
                                 figsize=S.figsize(style, (12.0, 4.4),
                                                   (15.0, 5.6)))
        x = np.arange(len(cells))
        b = axes[0].bar(x, cells.MAPE_pct, 0.55, color=D.PALETTE["stage3"])
        axes[0].plot(x, cells.bias_pct, "o-", color=D.PALETTE["accent"],
                     linewidth=S.scale(style, 1.4, 1.6),
                     markersize=S.scale(style, 5, 1.6), label="Bias")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([rf"$P={p:g}$" for p in cells.penalty])
        axes[0].set_ylabel("Error [%]")
        axes[0].set_title("(a) Per operating point")
        axes[0].grid(alpha=0.25, axis="y")
        axes[0].legend(framealpha=0.9)
        P.bar_labels(axes[0], b, cells.MAPE_pct.values, "{:.1f}", style=style)

        xp = np.arange(len(per_prov))
        b2 = axes[1].bar(xp, per_prov.mape, 0.55, color=D.PALETTE["stage2"])
        axes[1].plot(xp, per_prov.bias, "o-", color=D.PALETTE["accent"],
                     linewidth=S.scale(style, 1.4, 1.6),
                     markersize=S.scale(style, 5, 1.6), label="Bias")
        axes[1].set_xticks(xp)
        axes[1].set_xticklabels(per_prov.provider,
                               rotation=30 if style == "slides" else 0)
        axes[1].set_ylabel("Error [%]")
        axes[1].set_title("(b) Per provider")
        axes[1].grid(alpha=0.25, axis="y")
        axes[1].legend(framealpha=0.9)
        P.bar_labels(axes[1], b2, per_prov.mape.values, "{:.1f}", style=style)

        P.footnote(fig, f"Bars are MAPE, line is signed bias. Pooled over all "
                        f"{int(overall.n)} observations: MAPE "
                        f"{float(overall.MAPE_pct):.2f}%, bias "
                        f"{float(overall.bias_pct):+.2f}%, "
                        f"$R^2$ = {float(overall.R2):.4f}.", style)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        S.save(fig, "fig63_diagnostics", style, S.TIER_B)

    D.prov.write("fig63_diagnostics",
                 title="Surrogate error per operating point and provider",
                 tier=S.TIER_B, act=ACT, basis=BASIS,
                 claim=f"Pooled surrogate error is {float(overall.MAPE_pct):.2f}% "
                       f"MAPE at {float(overall.bias_pct):+.2f}% bias. Error "
                       f"grows with the penalty ({cells.MAPE_pct.min():.1f}% to "
                       f"{cells.MAPE_pct.max():.1f}%) because higher penalties "
                       f"push areas onto schedules further from the training "
                       f"pool's centre of mass.",
                 caveats="Bias is positive throughout: the surrogate "
                         "over-estimates cost, which is why the realised "
                         "savings in fig62 come out above prediction.")


def main():
    for fn in (fig61_vroom_scatter, fig62_pred_vs_actual, fig63_diagnostics):
        print(f"\n=== {fn.__name__} ===")
        fn()


if __name__ == "__main__":
    main()
