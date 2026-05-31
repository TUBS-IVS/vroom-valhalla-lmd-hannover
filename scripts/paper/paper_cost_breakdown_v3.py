"""Simplified cost breakdown — only TWO categories:

  1. Batched tours          — cost from cells with chosen schedule_size 1..5
                              (consolidates ≥2 source days into one tour)
  2. Non-batched tours      — daily tours + same-day residual tours
                              (chosen size = 6, plus residual on non-delivery
                              days of smaller schedules; both deliver 1 day's
                              demand — i.e. "normal" tours)

B2C / B2B are no longer treated asymmetrically (the previous
fs_b2b = (1−share)·0.5 artefact is dropped for plotting; for paper-ready
numbers the orchestrator should be re-run with fs_b2b = fs_b2c = 1−share).

Outputs (results/overnight_2026_05_27/):
  fig_breakdown_v3_p050.{png,pdf}     — single panel at P=0.5
  fig_breakdown_v3_grid.{png,pdf}     — 2x4 grid over all penalties
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
    "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

PENALTY_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]
C_NORMAL  = "#264653"      # non-batched (daily + residual)
C_BATCHED = "#e9c46a"      # batched (multi-day consolidated)
C_TOTAL   = "#0b1f3a"


def load_data():
    g = pd.read_csv(BASE / "tab_ml_grid.csv").sort_values(
        ["penalty", "share_willing"]).reset_index(drop=True)
    chosen = pd.read_csv(BASE / "tab_chosen_schedules.csv")

    # Per (P, share) split: BATCHED only when chosen schedule size <6,
    # NON-BATCHED = daily-chosen cells' cost + all express residual.
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

    # Final two categories
    g["cost_non_batched"] = (g.cost_daily
                              + g.cost_express_standalone
                              + g.cost_express_bundled)
    g["cost_batched"] = g.cost_true_batched

    g["non_batched_kE"] = g.cost_non_batched / 1e3
    g["batched_kE"] = g.cost_batched / 1e3
    g["total_kE"] = g.total_cost_eur / 1e3
    g["batched_share_pct"] = 100 * g.cost_batched / g.total_cost_eur.clip(lower=1)
    g["share_pct"] = g.share_willing * 100
    return g, chosen


def stacked_panel(ax, sub, title, ymax, annotate=False):
    x = sub.share_pct.values
    nb = sub.non_batched_kE.values
    b = sub.batched_kE.values
    ax.fill_between(x, 0, nb, color=C_NORMAL, alpha=0.88,
                     label="Non-batched tours (daily + same-day residual)")
    ax.fill_between(x, nb, nb + b, color=C_BATCHED, alpha=0.92,
                     label="Batched tours (≥2 source days consolidated)")
    ax.plot(x, nb + b, color=C_TOTAL, linewidth=1.6,
             marker="o", markersize=4)
    if annotate:
        for xi, yt in zip(x, nb + b):
            ax.text(xi, yt + 18, f"{yt:.0f}", ha="center", fontsize=8.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-2, 102)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.3)


def main():
    g, chosen = load_data()
    print(f"  grid rows: {len(g)}")

    # ── Plot 1: single panel P=0.5 with secondary axis = batched %
    sub = g[np.isclose(g.penalty, 0.5)].sort_values("share_pct")
    ymax = float((sub.non_batched_kE + sub.batched_kE).max()) * 1.05

    fig, (ax, ax_pct) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1.5]})
    stacked_panel(ax, sub, "Cost decomposition at $P=0.5$ €/parcel/day",
                   ymax, annotate=True)
    ax.set_ylabel("Weekly cost [k€]")
    ax.legend(loc="upper right", framealpha=0.95)

    # Bottom: batched-share percentage
    ax_pct.bar(sub.share_pct, sub.batched_share_pct, width=7,
                color=C_BATCHED, edgecolor="black")
    ax_pct.set_xlabel("Share of customers willing to wait [%]")
    ax_pct.set_ylabel("Batched cost / Total [%]")
    ax_pct.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax_pct.grid(axis="y", alpha=0.3)
    for xi, p in zip(sub.share_pct, sub.batched_share_pct):
        ax_pct.text(xi, p + 1, f"{p:.0f}%", ha="center", fontsize=8.5)
    ax_pct.set_ylim(0, max(60, sub.batched_share_pct.max() * 1.15))

    fig.suptitle("How much of the routing cost is genuine multi-day batching?",
                  fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "fig_breakdown_v3_p050.png")
    fig.savefig(OUT / "fig_breakdown_v3_p050.pdf")
    plt.close(fig)
    print("  fig_breakdown_v3_p050")

    # ── Plot 2: 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    ymax_all = float((g.non_batched_kE + g.batched_kE).max()) * 1.04
    for pi, P in enumerate(PENALTY_GRID):
        ax = axes[pi // 4, pi % 4]
        sub = g[np.isclose(g.penalty, P)].sort_values("share_pct")
        stacked_panel(ax, sub, f"$P={P}$ €/p/d", ymax_all)
        if pi // 4 == 1:
            ax.set_xlabel("Share willing [%]")
        if pi % 4 == 0:
            ax.set_ylabel("Weekly cost [k€]")
    handles = [Patch(color=C_NORMAL, alpha=0.88,
                       label="Non-batched tours  (daily delivery + same-day residual)"),
                Patch(color=C_BATCHED, alpha=0.92,
                       label="Batched tours  (≥2 source days consolidated)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11,
                bbox_to_anchor=(0.5, -0.03), frameon=False)
    fig.suptitle("Cost decomposition: batched vs. non-batched tours across the (P × share) grid",
                  fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "fig_breakdown_v3_grid.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_breakdown_v3_grid.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_breakdown_v3_grid")

    # ── Headline at P=0.5
    print("\nP=0.5  (batched vs non-batched):")
    cols = ["share_willing", "non_batched_kE", "batched_kE", "total_kE",
             "batched_share_pct", "n_batched_cells", "n_daily_cells"]
    print(g[g.penalty == 0.5][cols].round(1).to_string(index=False))

    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
