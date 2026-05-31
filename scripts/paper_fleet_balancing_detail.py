"""Detailed fleet-balancing impact analysis.

After running overnight_orchestrator_balanced.py we have, per cell,
both the unbalanced (cost-optimal) and the hub-balanced schedules
plus the per-(provider, hub, day) fleet load before/after.

This script answers the user's question: *what does hub balancing buy
us, and where?*

  fig_FB5_imbalance_distribution.{png,pdf}    — boxplot of imbalance Δ per cell
  fig_FB6_peak_day_heatmap.{png,pdf}          — peak-day fleet vs penalty/share
  fig_FB7_avg_peak_per_hub.{png,pdf}          — bar chart per (provider, hub)
                                                  showing peak fleet shrink
  fig_FB8_swap_distribution.{png,pdf}          — histogram of # swaps per cell
  fig_FB9_imbalance_reduction_vs_n_plz.{png,pdf}
  tab_fleet_balancing_summary.csv
  tab_per_hub_summary.csv

Inputs (from results/overnight_2026_05_27_balanced/):
  tab_balancing_summary.csv
  tab_chosen_schedules.csv
  tab_fleet_per_hub.csv
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
BAL = ROOT / "results" / "overnight_2026_05_27_balanced"
OUT = BAL

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
OPERATING_P = 0.5
OPERATING_SHARE = 1.0


def main():
    summary = pd.read_csv(BAL / "tab_balancing_summary.csv")
    fleet = pd.read_csv(BAL / "tab_fleet_per_hub.csv")
    print(f"  summary rows: {len(summary)}")
    print(f"  fleet rows:   {len(fleet)}")

    # ── Plot FB5: imbalance reduction per cell (boxplot per provider)
    fig, ax = plt.subplots(figsize=(9, 5))
    provs = sorted(summary.provider.unique())
    data = [summary[summary.provider == p].imbalance_reduction_pct.values for p in provs]
    bp = ax.boxplot(data, labels=provs, patch_artist=True)
    for patch, color in zip(bp["boxes"], plt.cm.viridis(np.linspace(0.15, 0.9, len(provs)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Imbalance reduction [%]")
    ax.set_title("Fleet-imbalance reduction per LSP (across all 88 cells)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_FB5_imbalance_distribution.png")
    fig.savefig(OUT / "fig_FB5_imbalance_distribution.pdf")
    plt.close(fig)
    print("  fig_FB5_imbalance_distribution")

    # ── Plot FB6: peak fleet after balancing per cell heatmap
    agg = summary.groupby(["penalty", "share_willing"], as_index=False).agg(
        max_fleet_before=("max_fleet_before", "sum"),
        max_fleet_after=("max_fleet_after", "sum"),
        cost_delta=("cost_delta_eur", "sum"),
    )
    pen = sorted(agg.penalty.unique())
    shares = sorted(agg.share_willing.unique())
    M_before = np.zeros((len(shares), len(pen)))
    M_after = np.zeros_like(M_before)
    for i, s in enumerate(shares):
        for j, p in enumerate(pen):
            r = agg[(np.isclose(agg.penalty, p)) & (np.isclose(agg.share_willing, s))]
            if len(r):
                M_before[i, j] = r.max_fleet_before.iloc[0]
                M_after[i, j] = r.max_fleet_after.iloc[0]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    vmin = min(M_before.min(), M_after.min())
    vmax = max(M_before.max(), M_after.max())
    for ax, M, title in [(axA, M_before, "BEFORE balancing"),
                          (axB, M_after, "AFTER balancing")]:
        im = ax.imshow(M, aspect="auto", cmap="viridis_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pen)))
        ax.set_xticklabels([f"{p:g}" for p in pen])
        ax.set_yticks(range(len(shares)))
        ax.set_yticklabels([f"{int(s*100)}%" for s in shares])
        ax.set_xlabel("Service penalty $P$")
        ax.set_title(f"Peak weekly fleet — {title}")
        for i in range(len(shares)):
            for j in range(len(pen)):
                v = M[i, j]
                color = "white" if (v - vmin) / max(1, vmax - vmin) > 0.55 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color=color, fontsize=8)
    axA.set_ylabel("Share willing")
    plt.colorbar(im, ax=[axA, axB], label="Peak fleet (vehicles)",
                  shrink=0.8, pad=0.02)
    fig.suptitle("Peak fleet across the (P, share) grid — balancing flattens hub demand",
                  fontsize=13, y=1.02)
    fig.savefig(OUT / "fig_FB6_peak_day_heatmap.png")
    fig.savefig(OUT / "fig_FB6_peak_day_heatmap.pdf")
    plt.close(fig)
    print("  fig_FB6_peak_day_heatmap")

    # ── Plot FB7: average peak-fleet per (provider, hub) bar
    sub_fleet = fleet[(np.isclose(fleet.penalty, OPERATING_P)) &
                       (np.isclose(fleet.share_willing, OPERATING_SHARE))]
    hub_peak = sub_fleet.groupby(["provider", "hub"], as_index=False).agg(
        peak_before=("fleet_before", "max"),
        peak_after=("fleet_after", "max"))
    hub_peak["delta"] = hub_peak.peak_before - hub_peak.peak_after
    hub_peak = hub_peak.sort_values("delta", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(hub_peak))
    width = 0.38
    ax.bar(x - width/2, hub_peak.peak_before, width, color="#e76f51",
            label="Peak before", edgecolor="black")
    ax.bar(x + width/2, hub_peak.peak_after, width, color="#1f4f8f",
            label="Peak after", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.provider}\n{r.hub[:18]}"
                          for _, r in hub_peak.iterrows()],
                          rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Peak weekly fleet (vehicles)")
    ax.set_title(f"Top-20 hubs by fleet reduction (operating point $P={OPERATING_P}$)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_FB7_avg_peak_per_hub.png")
    fig.savefig(OUT / "fig_FB7_avg_peak_per_hub.pdf")
    plt.close(fig)
    print("  fig_FB7_avg_peak_per_hub")

    # ── Plot FB8: swap distribution
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(summary.swaps_made, bins=30, color="#1f4f8f",
             edgecolor="white", alpha=0.85)
    ax.set_xlabel("Number of fleet-balancing swaps per cell")
    ax.set_ylabel("Count of (P, share, provider) cells")
    ax.set_title(f"Distribution of swaps made — median {summary.swaps_made.median():.0f}, "
                  f"max {summary.swaps_made.max()}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_FB8_swap_distribution.png")
    fig.savefig(OUT / "fig_FB8_swap_distribution.pdf")
    plt.close(fig)
    print("  fig_FB8_swap_distribution")

    # ── Plot FB9: imbalance reduction vs n_plz
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for prov, g in summary.groupby("provider"):
        ax.scatter(g.n_plz, g.imbalance_reduction_pct, s=22, alpha=0.6, label=prov)
    ax.set_xlabel("Number of PLZ at the LSP")
    ax.set_ylabel("Imbalance reduction [%]")
    ax.set_title("Fleet-balancing impact scales with LSP coverage")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_FB9_imbalance_reduction_vs_n_plz.png")
    fig.savefig(OUT / "fig_FB9_imbalance_reduction_vs_n_plz.pdf")
    plt.close(fig)
    print("  fig_FB9_imbalance_reduction_vs_n_plz")

    # ── Summary tables
    overall = summary.agg(
        n_cells=("provider", "count"),
        mean_imbalance_reduction_pct=("imbalance_reduction_pct", "mean"),
        median_imbalance_reduction_pct=("imbalance_reduction_pct", "median"),
        mean_cost_delta_pct=("cost_delta_pct", "mean"),
        max_fleet_before_total=("max_fleet_before", "sum"),
        max_fleet_after_total=("max_fleet_after", "sum"),
        total_swaps=("swaps_made", "sum"),
    )
    overall_df = pd.DataFrame([overall.iloc[0].to_dict()
                                 for col in [None]]) if hasattr(overall, "iloc") else overall
    # Per-provider summary
    per_prov = summary.groupby("provider").agg(
        n_cells=("provider", "count"),
        mean_imbalance_reduction_pct=("imbalance_reduction_pct", "mean"),
        mean_cost_delta_pct=("cost_delta_pct", "mean"),
        sum_peak_before=("max_fleet_before", "sum"),
        sum_peak_after=("max_fleet_after", "sum"),
        sum_swaps=("swaps_made", "sum"),
    ).reset_index()
    per_prov["peak_reduction_pct"] = (
        100 * (per_prov.sum_peak_before - per_prov.sum_peak_after)
        / per_prov.sum_peak_before.clip(lower=1)
    )
    per_prov.to_csv(OUT / "tab_per_provider_fleet_summary.csv", index=False)
    print("\nPer-provider summary:")
    print(per_prov.round(2).to_string(index=False))

    # Per-hub summary (operating point)
    if not sub_fleet.empty:
        hub_summary = sub_fleet.groupby(["provider", "hub"]).agg(
            peak_before=("fleet_before", "max"),
            peak_after=("fleet_after", "max"),
        ).reset_index()
        hub_summary["peak_reduction"] = hub_summary.peak_before - hub_summary.peak_after
        hub_summary["peak_reduction_pct"] = (
            100 * hub_summary.peak_reduction / hub_summary.peak_before.clip(lower=1)
        )
        hub_summary.to_csv(OUT / "tab_per_hub_summary.csv", index=False)
        print(f"  per-hub summary: {len(hub_summary)} (provider, hub) rows")

    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
