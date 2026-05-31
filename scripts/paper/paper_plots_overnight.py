"""Paper-ready plots from the overnight orchestrator outputs.

Generates five publication-style figures, matching the layout the user
sketched:

  1. fig01_heatmap_penalty_share.{png,pdf}
        Cost grid:  rows = share_willing, cols = penalty
        Numeric annotations per cell.

  2. fig02_schedule_mix_vs_penalty.{png,pdf}
        Stacked bars:  delivery-frequency mix per penalty, at share=1.0
        (full willingness, no express residual).
        Uses the 20-point fine penalty grid from results/penalty_sweep/.

  3. fig03_pareto_cost_wait.{png,pdf}
        Service-cost Pareto frontier across penalties at share=1.0.
        Operating point P=0.5 highlighted with a gold star.

  4. fig04_schedule_mix_vs_share.{png,pdf}
        Stacked area:  delivery-frequency mix shifts with share_willing
        at P=0.5 (overnight data).

  5. fig05_provider_cost_vs_share.{png,pdf}
        Per-LSP cost trajectory at window=3, P=0.5.

Inputs:
  results/overnight_2026_05_27/tab_ml_grid.csv
  results/overnight_2026_05_27/tab_chosen_schedules.csv
  results/penalty_sweep/tab_penalty_pareto.csv  (fine 20-P grid)
"""
from __future__ import annotations
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
OVERNIGHT = ROOT / "results" / "overnight_2026_05_27"
PSWEEP = ROOT / "results" / "penalty_sweep"
OUT = OVERNIGHT
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 12,
    "axes.labelsize": 13, "axes.titlesize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "axes.titleweight": "regular",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

# Consistent delivery-frequency palette (1..6 days/wk)
FREQ_COLOR = {
    1: "#9d2226",   # rare
    2: "#1d3557",   # navy
    3: "#2a9d8f",   # teal
    4: "#e9c46a",   # yellow
    5: "#f4a261",   # orange
    6: "#e76f51",   # red-orange
}
OPERATING_P = 0.5
WEEKLY_BASELINE_K_EUR = 1977.0   # all-daily cost (from prior analysis)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _enumerate_schedules(max_hold=3):
    from itertools import combinations
    out = []
    n = 6
    for k in range(1, n + 1):
        for combo in combinations(range(n), k):
            days = sorted(combo)
            ok = True
            for i in range(len(days)):
                gap = (days[(i + 1) % len(days)] - days[i]) % n
                if gap == 0:
                    gap = n
                if gap > max_hold:
                    ok = False
                    break
            if ok:
                out.append(frozenset(days))
    return out


def _avg_wait_days(s, n=6):
    if not s:
        return 0.0
    ds = sorted(s)
    total = 0.0
    for di in range(n):
        next_dd = min(((d - di) % n, d) for d in ds)[1]
        total += (next_dd - di) % n
    return total / n


# ---------------------------------------------------------------------------
# Plot 1: heatmap
# ---------------------------------------------------------------------------
def plot_heatmap(grid):
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    pen = sorted(grid.penalty.unique())
    shares = sorted(grid.share_willing.unique())
    M = np.zeros((len(shares), len(pen)))
    for i, s in enumerate(shares):
        for j, p in enumerate(pen):
            cell = grid[(np.isclose(grid.penalty, p)) & (np.isclose(grid.share_willing, s))]
            if len(cell):
                M[i, j] = cell.total_cost_eur.iloc[0] / 1e3
    im = ax.imshow(M, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(pen)))
    ax.set_xticklabels([f"{p:g}" for p in pen])
    ax.set_yticks(range(len(shares)))
    ax.set_yticklabels([f"{int(s*100)}%" for s in shares])
    ax.set_xlabel("Service penalty $P$ [€/parcel/day]")
    ax.set_ylabel("Share of customers willing to wait")
    ax.set_title("Total weekly routing cost [k€]")
    for i in range(len(shares)):
        for j in range(len(pen)):
            v = M[i, j]
            color = "white" if (v - M.min()) / max(1, M.max() - M.min()) < 0.55 else "black"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color=color, fontsize=10)
    cbar = plt.colorbar(im, ax=ax, label="Weekly cost [k€]")
    fig.tight_layout()
    fig.savefig(OUT / "fig01_heatmap_penalty_share.png")
    fig.savefig(OUT / "fig01_heatmap_penalty_share.pdf")
    plt.close(fig)
    print("  fig01_heatmap_penalty_share")


# ---------------------------------------------------------------------------
# Plot 2: stacked bars — schedule mix vs penalty (from fine penalty_sweep grid)
# ---------------------------------------------------------------------------
def plot_mix_vs_penalty():
    df = pd.read_csv(PSWEEP / "tab_penalty_pareto.csv")
    df = df.sort_values("penalty").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    for sz in [1, 2, 3, 4, 5, 6]:
        col = f"mix_{sz}day"
        if col not in df.columns:
            continue
        h = df[col].values
        ax.bar(x, h, bottom=bottom, color=FREQ_COLOR[sz],
               label=f"{sz} day/wk", edgecolor="white", linewidth=0.4, width=0.85)
        bottom += h
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:g}" for p in df.penalty], rotation=45, ha="right")
    ax.set_xlabel("Service penalty $P$ [€/parcel/day]")
    ax.set_ylabel("Count of (provider, PLZ) cells")
    ax.set_title("Delivery-frequency mix shifts with the service penalty")
    ax.legend(title="Delivery days per week", bbox_to_anchor=(1.02, 1.0),
              loc="upper left", frameon=True)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig02_schedule_mix_vs_penalty.png")
    fig.savefig(OUT / "fig02_schedule_mix_vs_penalty.pdf")
    plt.close(fig)
    print("  fig02_schedule_mix_vs_penalty")


# ---------------------------------------------------------------------------
# Plot 3: Pareto cost vs wait
# ---------------------------------------------------------------------------
def plot_pareto(grid):
    # Compute avg_wait per cell using tab_chosen_schedules
    chosen = pd.read_csv(OVERNIGHT / "tab_chosen_schedules.csv")
    # weighted average wait per cell by weekly_parcels
    agg = (chosen.groupby(["penalty", "share_willing"], as_index=False)
           .apply(lambda g: pd.Series({
               "weighted_wait": (g.avg_wait_d * g.weekly_parcels).sum()
                                / max(1, g.weekly_parcels.sum())
           })).reset_index(drop=True))
    grid = grid.merge(agg, on=["penalty", "share_willing"], how="left")

    # Filter to share=1.0 (no express, pure batched optimum)
    s1 = grid[np.isclose(grid.share_willing, 1.0)].sort_values("penalty")

    # Also bring in the fine 20-point penalty sweep for richer Pareto
    psweep = pd.read_csv(PSWEEP / "tab_penalty_pareto.csv").sort_values("penalty")

    fig, ax = plt.subplots(figsize=(9, 6))
    # Fine penalty_sweep line
    ax.plot(psweep.avg_wait_days, psweep.total_cost_eur / 1e3, "o-",
            color="#1f4f8f", linewidth=2.2, markersize=6, alpha=0.9)
    # Annotate the headline penalty points
    annotate_at = [0.025, 0.075, 0.125, 0.2, 0.3, 0.5, 0.75, 1.5, 3, 5]
    for _, r in psweep.iterrows():
        if r.penalty in annotate_at:
            ax.annotate(f"$P={r.penalty:g}$",
                        (r.avg_wait_days, r.total_cost_eur / 1e3),
                        xytext=(7, 6), textcoords="offset points", fontsize=10)
    # Operating point P=0.5 star
    op = psweep[np.isclose(psweep.penalty, 0.5)]
    if len(op):
        ax.scatter(op.avg_wait_days, op.total_cost_eur / 1e3,
                    marker="*", s=440, c="gold", edgecolor="black", zorder=10,
                    label=f"Operating point $P={OPERATING_P}$ €/parcel/day")
        ax.legend(loc="upper right")
    ax.set_xlabel("Average customer wait [days]")
    ax.set_ylabel("Weekly routing cost [k€]")
    ax.set_title("Service-cost Pareto frontier: penalty sensitivity\n"
                  "(Daganzo-LGB-Hybrid surrogate, 312 (provider, PLZ) cells)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig03_pareto_cost_wait.png")
    fig.savefig(OUT / "fig03_pareto_cost_wait.pdf")
    plt.close(fig)
    print("  fig03_pareto_cost_wait")


# ---------------------------------------------------------------------------
# Plot 4: stacked area — delivery-frequency mix shifts with share_willing
# ---------------------------------------------------------------------------
def plot_mix_vs_share():
    chosen = pd.read_csv(OVERNIGHT / "tab_chosen_schedules.csv")
    sub = chosen[np.isclose(chosen.penalty, OPERATING_P)].copy()
    # aggregate per (share_willing, schedule_size) → count
    agg = (sub.groupby(["share_willing", "schedule_size"], as_index=False)
           .size())
    pivot = agg.pivot(index="share_willing", columns="schedule_size",
                       values="size").fillna(0).sort_index()
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100   # convert to %

    fig, ax = plt.subplots(figsize=(9, 5))
    x = pivot.index.values * 100
    bottom = np.zeros(len(pivot))
    sizes_ordered = [s for s in (2, 3, 4, 5, 6) if s in pivot.columns]
    for sz in sizes_ordered:
        h = pivot[sz].values
        ax.fill_between(x, bottom, bottom + h,
                         color=FREQ_COLOR[sz], alpha=0.92, label=f"{sz} day/wk")
        bottom += h
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Share of customers willing to wait [%]")
    ax.set_ylabel("Delivery-frequency mix [%]")
    ax.set_title(f"How willingness-to-wait reshapes the delivery-frequency mix "
                  f"(operating point $P={OPERATING_P}$ €/parcel/day)")
    ax.legend(title="Delivery days/week", loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "fig04_schedule_mix_vs_share.png")
    fig.savefig(OUT / "fig04_schedule_mix_vs_share.pdf")
    plt.close(fig)
    print("  fig04_schedule_mix_vs_share")


# ---------------------------------------------------------------------------
# Plot 5: per-LSP cost trajectory
# ---------------------------------------------------------------------------
def plot_provider(grid):
    chosen = pd.read_csv(OVERNIGHT / "tab_chosen_schedules.csv")
    sub = chosen[np.isclose(chosen.penalty, OPERATING_P)].copy()
    agg = (sub.groupby(["share_willing", "provider"], as_index=False)
           .agg(cost=("dd_cost_eur", "sum")))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    palette = {
        "Amazon": "#1f77b4", "DHL": "#ff7f0e", "DPD": "#2ca02c",
        "FedEx": "#d62728", "GLS": "#9467bd", "Hermes": "#8c564b",
        "UPS": "#e377c2",
    }
    for prov, g in agg.groupby("provider"):
        g = g.sort_values("share_willing")
        ax.plot(g.share_willing * 100, g.cost / 1e3, "o-",
                color=palette.get(prov, "grey"), linewidth=2, markersize=5,
                label=prov)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of customers willing to wait [%]")
    ax.set_ylabel("Weekly routing cost [k€]")
    ax.set_title(f"Cost trajectory by logistics provider "
                  f"(window = 3 days, $P = {OPERATING_P}$)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig05_provider_cost_vs_share.png")
    fig.savefig(OUT / "fig05_provider_cost_vs_share.pdf")
    plt.close(fig)
    print("  fig05_provider_cost_vs_share")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading overnight grid ...")
    grid = pd.read_csv(OVERNIGHT / "tab_ml_grid.csv")
    print(f"  {len(grid)} cells")
    print()
    print("Generating paper plots:")
    plot_heatmap(grid)
    plot_mix_vs_penalty()
    plot_pareto(grid)
    plot_mix_vs_share()
    plot_provider(grid)
    print()
    print(f"Done. Outputs in {OUT}")


if __name__ == "__main__":
    main()
