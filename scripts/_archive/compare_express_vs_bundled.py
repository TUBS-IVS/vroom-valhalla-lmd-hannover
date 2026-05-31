"""Compare three express-handling approaches for the cost-vs-willingness plot.

  A) Linear blend  (results/willingness_p050/tab_grid.csv)
      — DEPRECATED: weighted average between fast=0 and fast=1 corner points.
        Wrong because mixed demand isn't simply additive.

  B) Per-PLZ express (results/express_aware/tab_cost_vs_share.csv)
      — Realistic UPPER BOUND: each (PLZ, non-delivery-day) gets its own
        VROOM-mini-tour. Even 5 parcels triggers tour-startup cost.

  C) Hub-bundled express (results/willingness_hub_bundled_daganzo/tab_grid.csv)
      — REALISTIC: small-volume PLZs at the same hub on the same day are
        pooled (LPT-balanced) into shared tours. Standalone if ≥150 pkts.

This script overlays all three on a single plot for the paper.
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "compare_express_handling"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})


def main():
    expr_aware = pd.read_csv(ROOT / "results/express_aware/tab_cost_vs_share.csv")
    hb = pd.read_csv(ROOT / "results/willingness_hub_bundled_daganzo/tab_grid.csv")
    blend = pd.read_csv(ROOT / "results/willingness_p050/tab_grid.csv")

    # Keep window=3 (matches abstract MAX_HOLDING=3 = postponement 3 days)
    hb3 = hb[hb.window == 3].sort_values("share_willing").reset_index(drop=True)
    blend3 = blend[blend.window == 3].sort_values("share_willing").reset_index(drop=True)

    # express_aware uses share_fast directly; convert to share_willing
    expr_aware = expr_aware.sort_values("share_willing").reset_index(drop=True)

    print("=" * 70)
    print("Comparison @ window=3 days")
    print("=" * 70)
    print(f"{'share':>8} {'A linear':>10} {'B per-PLZ':>11} {'C bundled':>11} {'C-B':>8}")
    for share in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        a = blend3[np.isclose(blend3.share_willing, share)].total_cost_eur
        b_row = expr_aware[np.isclose(expr_aware.share_willing, share)]
        if not len(b_row):
            # express_aware has narrower grid; find closest
            b_row = expr_aware.iloc[(expr_aware.share_willing - share).abs().argmin():]
            b_row = b_row.iloc[:1]
        b = b_row.total_cost_eur
        c = hb3[np.isclose(hb3.share_willing, share)].total_cost_eur
        a_v = float(a.iloc[0])/1e3 if len(a) else float('nan')
        b_v = float(b.iloc[0])/1e3 if len(b) else float('nan')
        c_v = float(c.iloc[0])/1e3 if len(c) else float('nan')
        diff = (c_v - b_v)
        print(f"{share*100:>6.0f}% {a_v:>10.1f} {b_v:>11.1f} {c_v:>11.1f} {diff:>+8.1f}")

    # Plot all three lines at window=3
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(blend3.share_willing * 100, blend3.total_cost_eur / 1e3,
            "x--", color="#999999", linewidth=1.4, markersize=7,
            label="(A) Linear blend — DEPRECATED", alpha=0.85)
    ax.plot(expr_aware.share_willing * 100, expr_aware.total_cost_eur / 1e3,
            "s-", color="#b3261e", linewidth=2, markersize=6,
            label="(B) Per-PLZ express tours — upper bound")
    ax.plot(hb3.share_willing * 100, hb3.total_cost_eur / 1e3,
            "o-", color="#1f4f8f", linewidth=2.2, markersize=7,
            label="(C) Hub-bundled express — operationally realistic")
    # Anchor lines for reference
    daily_baseline = 1977.1
    ax.axhline(daily_baseline, color="black", linestyle=":",
                linewidth=1, alpha=0.5)
    ax.text(2, daily_baseline + 5, "All-daily baseline (1,977 k€)",
            fontsize=8, color="black", alpha=0.7)
    ax.set_xlabel("Share of customers willing to wait [%]")
    ax.set_ylabel("Weekly routing cost [k€]")
    ax.set_title("Express-residual handling matters: per-PLZ vs hub-bundled vs blend\n"
                  "Postponement window = 3 days, operating point $P=0.5$ €/parcel/day")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_three_way_comparison.png")
    fig.savefig(OUT / "fig_three_way_comparison.pdf")
    plt.close(fig)
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
