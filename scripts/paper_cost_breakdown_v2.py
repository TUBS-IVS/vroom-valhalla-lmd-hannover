"""Re-plot the cost breakdown with HONEST labels and schedule-size breakdown.

Issue with v1: "Batched routing" included cells that chose DAILY (6d/wk)
schedules — those aren't really batched, they're just normal daily delivery.
At share=0% the optimizer picks daily for 82% of cells, so the "batched"
fraction is misleadingly large.

v2 separates:
  1. Daily-routing cost      (cells with chosen schedule size = 6)
  2. Batched-routing cost    (cells with chosen schedule size 1..5)
  3. Express-standalone      (≥150 pcs/PLZ/day on non-delivery days)
  4. Express-hub-bundled     (LPT bin-packing, smaller residuals)

Plus a second panel showing the schedule-size mix across share.
"""
from __future__ import annotations
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "overnight_2026_05_27"
OUT = BASE

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

PENALTY_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]
C_DAILY     = "#264653"
C_BATCHED   = "#1f4f8f"
C_STAND     = "#e76f51"
C_BUNDLED   = "#f4a261"
C_TOTAL     = "#0b1f3a"


def load_data():
    g = pd.read_csv(BASE / "tab_ml_grid.csv").sort_values(
        ["penalty", "share_willing"]).reset_index(drop=True)
    chosen = pd.read_csv(BASE / "tab_chosen_schedules.csv")
    # Split cost_batched into daily-routing (size=6) and true-batched (size<6) by
    # the chosen schedules' cell-level dd_cost_eur sums.
    daily = chosen[chosen.schedule_size == 6].groupby(
        ["penalty", "share_willing"], as_index=False).agg(
        cost_daily=("dd_cost_eur", "sum"),
        n_daily_cells=("plz", "count"))
    nondaily = chosen[chosen.schedule_size < 6].groupby(
        ["penalty", "share_willing"], as_index=False).agg(
        cost_true_batched=("dd_cost_eur", "sum"),
        n_batched_cells=("plz", "count"))
    g = g.merge(daily, on=["penalty", "share_willing"], how="left")
    g = g.merge(nondaily, on=["penalty", "share_willing"], how="left")
    g["cost_daily"] = g.cost_daily.fillna(0.0)
    g["cost_true_batched"] = g.cost_true_batched.fillna(0.0)
    g["n_daily_cells"] = g.n_daily_cells.fillna(0).astype(int)
    g["n_batched_cells"] = g.n_batched_cells.fillna(0).astype(int)

    # k€ versions
    g["daily_kE"] = g.cost_daily / 1e3
    g["batched_kE"] = g.cost_true_batched / 1e3
    g["stand_kE"] = g.cost_express_standalone / 1e3
    g["bundle_kE"] = g.cost_express_bundled / 1e3
    g["total_kE"] = g.total_cost_eur / 1e3
    g["share_pct"] = g.share_willing * 100
    return g, chosen


def stacked_panel(ax, sub, title, ymax):
    x = sub.share_pct.values
    d = sub.daily_kE.values
    b = sub.batched_kE.values
    s = sub.stand_kE.values
    u = sub.bundle_kE.values
    ax.fill_between(x, 0, d, color=C_DAILY, alpha=0.88, label="Daily routing (6 d/wk)")
    ax.fill_between(x, d, d + b, color=C_BATCHED, alpha=0.88,
                     label="Batched routing (1–5 d/wk)")
    ax.fill_between(x, d + b, d + b + s, color=C_STAND, alpha=0.88,
                     label="Express — standalone")
    ax.fill_between(x, d + b + s, d + b + s + u, color=C_BUNDLED, alpha=0.88,
                     label="Express — hub-bundled")
    ax.plot(x, d + b + s + u, color=C_TOTAL, linewidth=1.6,
             marker="o", markersize=4)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-2, 102)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.3)


def main():
    g, chosen = load_data()
    print(f"  grid rows: {len(g)}")

    # ── Plot 1: single panel P=0.5 with correct labels
    sub = g[np.isclose(g.penalty, 0.5)].sort_values("share_pct")
    ymax = float((sub.daily_kE + sub.batched_kE + sub.stand_kE + sub.bundle_kE).max()) * 1.05
    fig, (ax, ax_n) = plt.subplots(2, 1, figsize=(11, 9), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.4]})
    stacked_panel(ax, sub, "Cost decomposition at $P=0.5$ €/parcel/day", ymax)
    # Annotate totals
    x = sub.share_pct.values
    tot = sub.daily_kE + sub.batched_kE + sub.stand_kE + sub.bundle_kE
    for xi, yt in zip(x, tot):
        ax.text(xi, yt + 18, f"{yt:.0f}", ha="center", fontsize=8.5)
    ax.set_ylabel("Weekly cost [k€]")
    ax.legend(loc="lower left", framealpha=0.95)

    # Bottom: schedule-size mix as stacked area (count of cells)
    sched_mix = chosen[chosen.penalty == 0.5].groupby(
        ["share_willing", "schedule_size"]).size().unstack(fill_value=0).sort_index()
    sched_mix["share_pct"] = sched_mix.index * 100
    palette = {2: "#9b2226", 3: "#bb3e03", 4: "#ee9b00", 5: "#94d2bd", 6: "#005f73"}
    bot = np.zeros(len(sched_mix))
    for sz in sorted(sched_mix.columns.drop("share_pct")):
        vals = sched_mix[sz].values
        ax_n.bar(sched_mix.share_pct, vals, bottom=bot,
                  color=palette.get(int(sz), "gray"),
                  edgecolor="white", width=7,
                  label=f"{int(sz)} d/wk")
        bot += vals
    ax_n.set_xlabel("Share of customers willing to wait [%]")
    ax_n.set_ylabel("Cells by chosen schedule")
    ax_n.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax_n.legend(title="Schedule size", loc="upper right", ncol=5, fontsize=8,
                 bbox_to_anchor=(1.0, 1.18))
    ax_n.grid(axis="y", alpha=0.3)
    fig.suptitle("Service-quality dial at P = 0.5 — how the 312 cells split between routing modes",
                  fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "fig_breakdown_v2_p050.png")
    fig.savefig(OUT / "fig_breakdown_v2_p050.pdf")
    plt.close(fig)
    print("  fig_breakdown_v2_p050")

    # ── Plot 2: 2x4 grid with correct labels
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    ymax_all = float((g.daily_kE + g.batched_kE + g.stand_kE + g.bundle_kE).max()) * 1.04
    for pi, P in enumerate(PENALTY_GRID):
        ax = axes[pi // 4, pi % 4]
        sub = g[np.isclose(g.penalty, P)].sort_values("share_pct")
        stacked_panel(ax, sub, f"$P={P}$ €/p/d", ymax_all)
        if pi // 4 == 1:
            ax.set_xlabel("Share willing [%]")
        if pi % 4 == 0:
            ax.set_ylabel("Weekly cost [k€]")
    handles = [Patch(color=C_DAILY,   alpha=0.88, label="Daily routing (chosen sched = 6 d/wk)"),
                Patch(color=C_BATCHED, alpha=0.88, label="Batched routing (chosen sched = 1–5 d/wk)"),
                Patch(color=C_STAND,   alpha=0.88, label="Express — standalone tour (≥150 pcs/PLZ/day)"),
                Patch(color=C_BUNDLED, alpha=0.88, label="Express — hub-bundled (LPT bin-packing)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11,
                bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.suptitle("Cost decomposition across the (P × share) grid — separated by chosen schedule size",
                  fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "fig_breakdown_v2_grid.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_breakdown_v2_grid.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_breakdown_v2_grid")

    # ── Diagnostic table: cost split at P=0.5
    print("\nAt P=0.5:")
    cols = ["share_willing", "daily_kE", "batched_kE", "stand_kE", "bundle_kE",
             "total_kE", "n_daily_cells", "n_batched_cells"]
    print(g[g.penalty == 0.5][cols].round(1).to_string(index=False))

    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
