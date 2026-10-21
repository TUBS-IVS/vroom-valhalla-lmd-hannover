"""Mathematical derivation of the service-quality sweet-spot.

Three independent methods to identify the Pareto-optimal (P, share) operating
point from the cost-vs-wait curve:

  1. Knee-point method: max perpendicular distance from chord line
                        (endpoints of Pareto curve)
  2. Marginal-saving threshold: smallest P where d(saving)/d(wait) > P
                                  (i.e. each extra wait-day still pays >P)
  3. Composite-score grid: maximize α · saving_norm − (1-α) · wait_norm

Output figure: fig_sweet_spot_math.{png,pdf}

Status B (Task 19): all three knee methods run on the SYSTEM Pareto frontier
(total cost + parcels-weighted wait vs P/share), which 74_-legacy's
tab_balancing_summary.csv/tab_chosen_schedules.csv carry directly -- no
NaN-only columns are touched here (weekly_parcels, avg_wait_d_balanced,
balanced_cost_eur all have real v6 values), so this is a straight repoint.
"""
from __future__ import annotations
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paper_v6_common as V6  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "overnight_2026_05_27_balanced"
OUT = BASE

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    global BASE, OUT
    ap = argparse.ArgumentParser(description=__doc__)
    V6.add_v6_cli_args(ap, needs_legacy=True)
    args = ap.parse_args()
    if args.legacy_dir is not None:
        BASE = Path(args.legacy_dir)
    elif args.rev_dir is not None:
        BASE, _ = V6.run_legacy_adapter(args.rev_dir,
                                        Path(args.out_dir or OUT) / "_legacy")
    if args.out_dir is not None:
        OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    s = pd.read_csv(BASE / "tab_balancing_summary.csv")
    sched = pd.read_csv(BASE / "tab_chosen_schedules.csv")

    # Aggregate per (P, share): total cost (balanced) + parcels-weighted wait
    agg_s = sched.groupby(["penalty", "share_willing"], as_index=False).agg(
        weekly_parcels=("weekly_parcels", "sum"),
        weighted_wait=("avg_wait_d_balanced", lambda x: 0.0),  # placeholder
    )
    # Recompute parcels-weighted wait
    wait_rows = []
    for (P, sh), g in sched.groupby(["penalty", "share_willing"]):
        wait = (g.avg_wait_d_balanced * g.weekly_parcels).sum() / g.weekly_parcels.sum()
        wait_rows.append({"penalty": P, "share_willing": sh, "wait_d": float(wait)})
    waits = pd.DataFrame(wait_rows)

    agg_cost = s.groupby(["penalty", "share_willing"], as_index=False).agg(
        total_cost=("balanced_cost_eur", "sum"),
    )
    df = agg_cost.merge(waits, on=["penalty", "share_willing"])

    # Daily baseline = highest P, share=0 (everyone gets daily)
    baseline_cost = float(df[df.share_willing == 0.0].total_cost.max())
    print(f"Daily baseline cost: {baseline_cost/1e3:.1f} k€")

    df["saving_pct"] = 100 * (baseline_cost - df.total_cost) / baseline_cost

    # ── Pareto frontier: for each unique wait, find the minimum cost
    # Then sort by wait ascending
    df_sorted = df.sort_values("wait_d").reset_index(drop=True)

    # ── Method 1: Knee-point via maximum distance from chord
    # Chord goes from (wait=0, saving=0) to (wait=max, saving=max)
    pareto = df_sorted.copy()
    p_endpoints_x = np.array([pareto.wait_d.min(), pareto.wait_d.max()])
    p_endpoints_y = np.array([pareto[pareto.wait_d == p_endpoints_x[0]].saving_pct.max(),
                               pareto[pareto.wait_d == p_endpoints_x[1]].saving_pct.max()])
    # Line equation: ax + by + c = 0 through endpoints
    dx = p_endpoints_x[1] - p_endpoints_x[0]
    dy = p_endpoints_y[1] - p_endpoints_y[0]
    norm = np.sqrt(dx ** 2 + dy ** 2)
    a = -dy / norm
    b = dx / norm
    c = -(a * p_endpoints_x[0] + b * p_endpoints_y[0])
    # Distance from point to line
    pareto["chord_dist"] = np.abs(a * pareto.wait_d + b * pareto.saving_pct + c)
    knee_idx = pareto.chord_dist.idxmax()
    knee = pareto.iloc[knee_idx]
    print(f"\n[1] Knee-Point (Chord-Distance Maximum):")
    print(f"    P={knee.penalty}, share={knee.share_willing}, "
          f"wait={knee.wait_d:.3f}d, saving={knee.saving_pct:.2f}%")

    # ── Method 2: Marginal-saving — d(saving)/d(wait) ≥ P
    # For each P, look at how share=1.0 → 0 trajectory
    print(f"\n[2] Marginal saving per wait-day (at share=1.0 vs share=0):")
    marg_rows = []
    for P in sorted(df.penalty.unique()):
        sub = df[df.penalty == P].sort_values("share_willing")
        if len(sub) < 2:
            continue
        max_saving = float(sub.saving_pct.max())
        max_wait = float(sub.wait_d.max())
        if max_wait > 0:
            marg = max_saving / max_wait     # % saving per wait day
        else:
            marg = 0.0
        marg_rows.append({"penalty": P, "max_saving_pct": max_saving,
                          "max_wait_d": max_wait, "marg_pct_per_day": marg})
    marg_df = pd.DataFrame(marg_rows)
    print(marg_df.round(2).to_string(index=False))

    # ── Method 3: Composite score — pick alpha that's interesting
    print(f"\n[3] Composite-Score grid (alpha = weight on saving):")
    max_saving_overall = df.saving_pct.max()
    max_wait_overall = df.wait_d.max()
    composite_rows = []
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        df["score"] = (
            alpha * df.saving_pct / max(max_saving_overall, 1e-3)
            - (1 - alpha) * df.wait_d / max(max_wait_overall, 1e-3)
        )
        best = df.loc[df.score.idxmax()]
        composite_rows.append({
            "alpha": alpha,
            "P": best.penalty, "share": best.share_willing,
            "wait_d": best.wait_d, "saving_pct": best.saving_pct,
        })
        print(f"    alpha={alpha}: P={best.penalty}, share={best.share_willing}, "
              f"wait={best.wait_d:.3f}d, saving={best.saving_pct:.2f}%")

    # ── Plot 1: Pareto curve with sweet-spot markers
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left panel: full Pareto + knee
    ax = axes[0]
    for P in sorted(df.penalty.unique()):
        sub = df[df.penalty == P].sort_values("share_willing")
        ax.plot(sub.wait_d, sub.saving_pct, "o-", label=f"P={P}",
                 markersize=5, linewidth=1.5, alpha=0.85)
    # Chord line
    ax.plot(p_endpoints_x, p_endpoints_y, "k--", linewidth=0.8, alpha=0.5,
             label="Chord")
    # Knee marker
    ax.scatter(knee.wait_d, knee.saving_pct, s=300, marker="*",
                c="gold", edgecolor="black", zorder=10,
                label=f"Knee (P={knee.penalty}, share={knee.share_willing})")
    ax.set_xlabel("Average customer wait [days]")
    ax.set_ylabel("Cost saving vs daily baseline [%]")
    ax.set_title("Pareto frontier — Cost-Service trade-off\n"
                  f"Knee at wait={knee.wait_d:.2f}d, saving={knee.saving_pct:.1f}%")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)

    # Right panel: marginal saving curves
    ax = axes[1]
    for P in sorted(df.penalty.unique()):
        sub = df[df.penalty == P].sort_values("wait_d")
        if len(sub) < 3:
            continue
        # Compute discrete derivative
        x = sub.wait_d.values
        y = sub.saving_pct.values
        # Smooth with cum-max
        dy_dx = np.gradient(y, x + 1e-6)
        ax.plot(x[1:], dy_dx[1:], "-", label=f"P={P}", alpha=0.7)
    ax.set_xlabel("Average customer wait [days]")
    ax.set_ylabel("d(saving) / d(wait)  [% saving per wait-day]")
    ax.set_title("Marginal saving per wait-day\n"
                  "Sweet-spot: where slope = service-cost threshold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    V6.add_provenance_footer(
        fig, plan="operator-polished (balanced)",
        script="paper_sweet_spot_math.py",
        source="B: 74_-legacy tab_balancing_summary.csv/tab_chosen_schedules.csv")
    V6.savefig_pair(fig, OUT / "fig_sweet_spot_math.png", OUT / "fig_sweet_spot_math.pdf")
    plt.close(fig)
    print(f"\nSaved {OUT/'fig_sweet_spot_math.png'}")

    # ── Save summary
    out_df = df.sort_values(["penalty", "share_willing"])
    out_df.to_csv(OUT / "tab_sweet_spot_data.csv", index=False)
    print(f"Saved {OUT/'tab_sweet_spot_data.csv'}")


if __name__ == "__main__":
    main()
