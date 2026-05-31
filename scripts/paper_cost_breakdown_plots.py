"""Fancy paper-grade cost-breakdown plots across (penalty, share_willing).

Visualises the three cost components that the share_willing dial controls:
  1. cost_batched               — the weekly batched-schedule routing cost
                                  (willing customers, multi-PLZ batched tours)
  2. cost_express_standalone    — standalone express tours per PLZ × day
                                  (≥150 pcs/PLZ/day)
  3. cost_express_bundled       — hub-bundled express via LPT bin-packing
                                  (<150 pcs/PLZ/day, ≤1000 pcs/bundle)

Outputs (results/overnight_2026_05_27/):
  fig_breakdown_stacked_p050.{png,pdf}          — stacked area at P = 0.5
  fig_breakdown_stacked_grid.{png,pdf}          — 2x4 stacked grid for all P
  fig_breakdown_tour_counts.{png,pdf}           — standalone vs bundled tour counts
  fig_breakdown_express_share_per_P.{png,pdf}   — express / total ratio across (P, share)
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
C_BATCHED   = "#1f4f8f"
C_STAND     = "#e76f51"
C_BUNDLED   = "#f4a261"
C_TOTAL     = "#264653"
PROV_VIRIDIS = plt.cm.viridis(np.linspace(0.15, 0.9, len(PENALTY_GRID)))


def load_grid():
    g = pd.read_csv(BASE / "tab_ml_grid.csv")
    g = g.sort_values(["penalty", "share_willing"]).reset_index(drop=True)
    g["share_pct"] = g.share_willing * 100
    g["cost_express_total"] = g.cost_express_standalone + g.cost_express_bundled
    g["batched_kE"] = g.cost_batched / 1e3
    g["stand_kE"] = g.cost_express_standalone / 1e3
    g["bundle_kE"] = g.cost_express_bundled / 1e3
    g["total_kE"] = g.total_cost_eur / 1e3
    g["express_share_pct"] = 100 * g.cost_express_total / g.total_cost_eur.clip(lower=1)
    return g


# ────────────────────────────────────────────────────────────────────────
def plot_stacked_p050(grid):
    """Single-panel stacked composition at P=0.5 with annotations."""
    g = grid[np.isclose(grid.penalty, 0.5)].sort_values("share_pct")
    x = g.share_pct.values
    y_batched = g.batched_kE.values
    y_stand = g.stand_kE.values
    y_bundle = g.bundle_kE.values

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.fill_between(x, 0, y_batched, color=C_BATCHED, alpha=0.88,
                     label="Batched routing (willing customers)")
    ax.fill_between(x, y_batched, y_batched + y_stand,
                     color=C_STAND, alpha=0.88,
                     label="Express — standalone (≥150 pcs/PLZ/day)")
    ax.fill_between(x, y_batched + y_stand, y_batched + y_stand + y_bundle,
                     color=C_BUNDLED, alpha=0.88,
                     label="Express — hub-bundled (LPT bin-packing)")
    ax.plot(x, y_batched + y_stand + y_bundle, color=C_TOTAL,
             linewidth=2.0, marker="o", markersize=5, label="Total cost")

    # Annotate total on each bar
    for xi, ytot, yst, ybu in zip(x, y_batched + y_stand + y_bundle, y_stand, y_bundle):
        ax.text(xi, ytot + 12, f"{ytot:.0f}", ha="center", fontsize=8.5,
                  color="#222")

    ax.set_xlabel("Share of customers willing to wait [%]")
    ax.set_ylabel("Weekly cost [k€]")
    ax.set_title("Cost breakdown by service component at $P=0.5$ €/parcel/day\n"
                  "(Daganzo-LGB-Hybrid surrogate, 312 (provider, PLZ) cells, hub-bundled express)")
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / "fig_breakdown_stacked_p050.png")
    fig.savefig(OUT / "fig_breakdown_stacked_p050.pdf")
    plt.close(fig)
    print("  fig_breakdown_stacked_p050")


# ────────────────────────────────────────────────────────────────────────
def plot_stacked_grid(grid):
    """2x4 grid: stacked breakdown for every P."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    ymax = float((grid.batched_kE + grid.stand_kE + grid.bundle_kE).max()) * 1.04
    for pi, P in enumerate(PENALTY_GRID):
        ax = axes[pi // 4, pi % 4]
        g = grid[np.isclose(grid.penalty, P)].sort_values("share_pct")
        x = g.share_pct.values
        b = g.batched_kE.values
        s = g.stand_kE.values
        u = g.bundle_kE.values
        ax.fill_between(x, 0, b, color=C_BATCHED, alpha=0.88, label="Batched")
        ax.fill_between(x, b, b + s, color=C_STAND, alpha=0.88, label="Stand-alone exp.")
        ax.fill_between(x, b + s, b + s + u, color=C_BUNDLED, alpha=0.88, label="Hub-bundled exp.")
        ax.plot(x, b + s + u, color=C_TOTAL, linewidth=1.4, marker="o", markersize=3)
        ax.set_title(f"$P={P}$ €/p/d", fontsize=11)
        ax.set_xlim(-2, 102)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.3)
        if pi // 4 == 1:
            ax.set_xlabel("Share willing [%]")
        if pi % 4 == 0:
            ax.set_ylabel("Weekly cost [k€]")
    # Single legend
    handles = [Patch(color=C_BATCHED, alpha=0.88, label="Batched routing (willing fraction)"),
                Patch(color=C_STAND, alpha=0.88, label="Express — standalone tours"),
                Patch(color=C_BUNDLED, alpha=0.88, label="Express — hub-bundled tours")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11,
                bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle("Cost breakdown across the (penalty × share-willing) grid",
                  fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "fig_breakdown_stacked_grid.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_breakdown_stacked_grid.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  fig_breakdown_stacked_grid")


# ────────────────────────────────────────────────────────────────────────
def plot_tour_counts(grid):
    """Bar-chart at P=0.5: #standalone vs #bundled tours per share."""
    g = grid[np.isclose(grid.penalty, 0.5)].sort_values("share_pct")
    x = g.share_pct.values
    n_st = g.n_standalone_tours.values
    n_bu = g.n_bundled_tours.values
    n_ev = g.n_express_events.values

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    w = 4
    ax.bar(x - w/2, n_st, w, color=C_STAND, label="Standalone tours", edgecolor="black")
    ax.bar(x + w/2, n_bu, w, color=C_BUNDLED, label="Hub-bundled tours", edgecolor="black")
    ax.set_xlabel("Share willing [%]")
    ax.set_ylabel("Number of express tours per week")
    ax.set_title("Where does residual demand land? — Standalone vs Hub-bundled  (P = 0.5)")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for xi, st, bu in zip(x, n_st, n_bu):
        if st > 0:
            ax.text(xi - w/2, st + 3, f"{int(st)}", ha="center", fontsize=8)
        if bu > 0:
            ax.text(xi + w/2, bu + 3, f"{int(bu)}", ha="center", fontsize=8)

    ax = axes[1]
    ax.plot(x, n_ev, "o-", color=C_TOTAL, linewidth=2, markersize=7,
             label="Total express events (PLZ × day)")
    ax.fill_between(x, 0, n_ev, color=C_TOTAL, alpha=0.18)
    ax.set_xlabel("Share willing [%]")
    ax.set_ylabel("Express events per week")
    ax.set_title("Total daily-express coverage across the share grid  (P = 0.5)")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for xi, ev in zip(x, n_ev):
        if ev > 0:
            ax.text(xi, ev + 4, f"{int(ev)}", ha="center", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(OUT / "fig_breakdown_tour_counts.png")
    fig.savefig(OUT / "fig_breakdown_tour_counts.pdf")
    plt.close(fig)
    print("  fig_breakdown_tour_counts")


# ────────────────────────────────────────────────────────────────────────
def plot_express_share_per_P(grid):
    """Line plot: express cost share of total across (P, share)."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for pi, P in enumerate(PENALTY_GRID):
        g = grid[np.isclose(grid.penalty, P)].sort_values("share_pct")
        ax.plot(g.share_pct, g.express_share_pct, "o-",
                 color=PROV_VIRIDIS[pi], linewidth=2, markersize=6,
                 label=f"$P={P}$ €/p/d")
    ax.set_xlabel("Share willing to wait [%]")
    ax.set_ylabel("Express cost / Total cost [%]")
    ax.set_title("How much of total cost is the express service?")
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.legend(title="Service penalty", loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_breakdown_express_share_per_P.png")
    fig.savefig(OUT / "fig_breakdown_express_share_per_P.pdf")
    plt.close(fig)
    print("  fig_breakdown_express_share_per_P")


def main():
    print("=" * 60)
    print("Cost breakdown plots")
    print("=" * 60)
    grid = load_grid()
    print(f"  grid rows: {len(grid)}  (P × share)")
    plot_stacked_p050(grid)
    plot_stacked_grid(grid)
    plot_tour_counts(grid)
    plot_express_share_per_P(grid)
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
