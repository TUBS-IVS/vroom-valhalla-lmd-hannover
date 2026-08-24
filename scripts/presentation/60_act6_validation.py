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
BASIS = "Stage-3 VROOM revalidation (4 operating points, theta=1)"


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
    sv = D.load_savings_validation()
    print(sv[["penalty", "predicted_saving_pct", "actual_saving_pct",
              "conservatism_pp"]].to_string(index=False))

    for style in S.styles():
        S.apply(style)
        fig, ax = plt.subplots(figsize=S.figsize(style, (7.6, 4.8), (11.0, 5.8)))
        x = np.arange(len(sv))
        w = 0.36
        b1 = ax.bar(x - w / 2, sv.predicted_saving_pct, w,
                    color=D.PALETTE["stage2"], label="Predicted (surrogate)")
        b2 = ax.bar(x + w / 2, sv.actual_saving_pct, w,
                    color=D.PALETTE["accent2"], label="Realised (VROOM)")
        ax.set_xticks(x)
        ax.set_xticklabels([rf"$P={p:g}$" for p in sv.penalty])
        ax.set_xlabel(r"Service penalty [€/p/d], all at $\theta = 100\%$")
        ax.set_ylabel("Weekly system cost saving [%]")
        ax.set_title("Predicted vs realised saving at the validated points")
        ax.grid(alpha=0.25, axis="y")
        ax.set_ylim(0, max(sv.actual_saving_pct.max(),
                           sv.predicted_saving_pct.max()) * 1.22)
        P.bar_labels(ax, b1, sv.predicted_saving_pct.values, "{:.1f}",
                     style=style)
        P.bar_labels(ax, b2, sv.actual_saving_pct.values, "{:.1f}",
                     style=style)
        for xi, row in zip(x, sv.itertuples()):
            ax.annotate(f"+{row.conservatism_pp:.1f} pp",
                        xy=(xi, max(row.predicted_saving_pct,
                                    row.actual_saving_pct)),
                        xytext=(0, 22 if style == "slides" else 15),
                        textcoords="offset points", ha="center",
                        fontsize=9 if style == "paper" else 13,
                        color=D.PALETTE["accent"])
        ax.legend(framealpha=0.9, loc="upper right")
        P.footnote(fig, f"Realised saving exceeds the prediction at every "
                        f"point; the surrogate is conservative by "
                        f"{sv.conservatism_pp.min():.1f}–"
                        f"{sv.conservatism_pp.max():.1f} pp.", style)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        S.save(fig, "fig62_pred_vs_actual", style, S.TIER_A)

    D.prov.write("fig62_pred_vs_actual",
                 title="Predicted vs realised system saving",
                 tier=S.TIER_A, act=ACT, basis=BASIS,
                 claim=f"Across four validated operating points the realised "
                       f"saving ({sv.actual_saving_pct.min():.1f}-"
                       f"{sv.actual_saving_pct.max():.1f}%) exceeds the "
                       f"surrogate prediction ({sv.predicted_saving_pct.min():.1f}"
                       f"-{sv.predicted_saving_pct.max():.1f}%) by "
                       f"{sv.conservatism_pp.min():.1f}-"
                       f"{sv.conservatism_pp.max():.1f} pp. The error points in "
                       f"the safe direction.",
                 caveats="Four points only, all at theta=1; the conservatism "
                         "band widens with the penalty.")


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
