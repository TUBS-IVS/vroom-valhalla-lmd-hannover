"""Three companion plots for the share×penalty grid that the paper needs:

  fig_share_cost_abs.{png,pdf}        — Weekly routing cost [k€]   (the original)
  fig_share_cost_saving_pct.{png,pdf} — Saving vs daily baseline [%]
  fig_share_avg_wait.{png,pdf}        — Effective customer-weighted wait [days]

The wait is averaged across ALL customers, weighted by parcels per cell, with
share_willing % waiting the schedule's avg_wait_d and (1-share_willing) % going
via hub-bundled express (wait = 0 d).

Baseline for the saving plot: the all-daily total cost at high P (P=5 or 10),
share=1, taken from results/penalty_sweep/tab_penalty_pareto.csv (1977.1 k€).
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

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "overnight_2026_05_27"
OUT = BASE
PSWEEP = ROOT / "results" / "penalty_sweep"

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

PENALTY_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]


def baseline_daily_cost():
    psweep = pd.read_csv(PSWEEP / "tab_penalty_pareto.csv")
    return float(psweep[psweep.penalty >= 5.0].total_cost_eur.iloc[0])


def main():
    grid = pd.read_csv(BASE / "tab_ml_grid.csv")
    grid = grid.sort_values(["penalty", "share_willing"]).reset_index(drop=True)
    chosen = pd.read_csv(BASE / "tab_chosen_schedules.csv")

    # Per-(P, share) effective customer-weighted wait (share fraction waits the
    # chosen schedule's avg_wait_d, the rest goes express → wait 0)
    wait_rows = []
    for (P, sh), g in chosen.groupby(["penalty", "share_willing"]):
        tot_pkts = float(g.weekly_parcels.sum())
        waiting_pkts = sh * tot_pkts
        weighted_wait_batched = float((g.avg_wait_d * g.weekly_parcels).sum())
        effective_wait = (sh * weighted_wait_batched) / max(1.0, tot_pkts)
        wait_rows.append({"penalty": float(P), "share_willing": float(sh),
                          "effective_wait_days": float(effective_wait),
                          "weighted_wait_batched_d": float(
                              weighted_wait_batched / max(1.0, tot_pkts))})
    wait_df = pd.DataFrame(wait_rows)

    base = baseline_daily_cost()
    print(f"Baseline (all-daily, high P) = {base/1e3:.1f} k€")

    grid = grid.merge(wait_df, on=["penalty", "share_willing"], how="left")
    grid["saving_pct"] = 100.0 * (base - grid.total_cost_eur) / base
    grid.to_csv(BASE / "tab_ml_grid_with_wait_and_saving.csv", index=False)

    # ── Plot 1: absolute cost
    pen_colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(PENALTY_GRID)))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for pi, P in enumerate(PENALTY_GRID):
        sub = grid[np.isclose(grid.penalty, P)].sort_values("share_willing")
        ax.plot(sub.share_willing * 100, sub.total_cost_eur / 1e3, "o-",
                 color=pen_colors[pi], linewidth=2, markersize=6,
                 label=f"$P={P}$ €/p/d")
    ax.axhline(base / 1e3, color="black", linestyle="--", linewidth=1,
                label=f"Daily baseline = {base/1e3:.0f} k€")
    ax.set_xlabel("Share of customers willing to wait [%]")
    ax.set_ylabel("Weekly routing cost [k€]")
    ax.set_title("Cost trade-off across (P, share)  — Daganzo-LGB-Hybrid, hub-bundled express, MAX_HOLD=3")
    ax.legend(title="Service penalty", loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_share_cost_abs.png")
    fig.savefig(OUT / "fig_share_cost_abs.pdf")
    plt.close(fig)
    print("  fig_share_cost_abs")

    # ── Plot 2: saving %
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for pi, P in enumerate(PENALTY_GRID):
        sub = grid[np.isclose(grid.penalty, P)].sort_values("share_willing")
        ax.plot(sub.share_willing * 100, sub.saving_pct, "o-",
                 color=pen_colors[pi], linewidth=2, markersize=6,
                 label=f"$P={P}$ €/p/d")
    ax.axhline(0, color="black", linestyle="--", linewidth=1,
                label=f"Daily baseline (0%)")
    ax.set_xlabel("Share of customers willing to wait [%]")
    ax.set_ylabel("Cost saving vs daily baseline [%]")
    ax.set_title("% Saving across (P, share)  — Daganzo-LGB-Hybrid surrogate")
    ax.legend(title="Service penalty", loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_share_cost_saving_pct.png")
    fig.savefig(OUT / "fig_share_cost_saving_pct.pdf")
    plt.close(fig)
    print("  fig_share_cost_saving_pct")

    # ── Plot 3: effective customer wait
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for pi, P in enumerate(PENALTY_GRID):
        sub = grid[np.isclose(grid.penalty, P)].sort_values("share_willing")
        ax.plot(sub.share_willing * 100, sub.effective_wait_days, "o-",
                 color=pen_colors[pi], linewidth=2, markersize=6,
                 label=f"$P={P}$ €/p/d")
    ax.set_xlabel("Share of customers willing to wait [%]")
    ax.set_ylabel("Effective customer-weighted wait [days]")
    ax.set_title("Service-quality trade-off  —  effective wait = share × schedule wait, express = 0d")
    ax.legend(title="Service penalty", loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_share_avg_wait.png")
    fig.savefig(OUT / "fig_share_avg_wait.pdf")
    plt.close(fig)
    print("  fig_share_avg_wait")

    # Bonus: print headline numbers per penalty at share=1.0
    print("\n  At share_willing = 100%:")
    for P in PENALTY_GRID:
        sub = grid[np.isclose(grid.penalty, P) & np.isclose(grid.share_willing, 1.0)]
        if len(sub):
            r = sub.iloc[0]
            print(f"    P={P:5.2f}  cost={r.total_cost_eur/1e3:7.1f} k€  "
                  f"saving={r.saving_pct:6.2f}%  wait={r.effective_wait_days:5.3f} d")

    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
