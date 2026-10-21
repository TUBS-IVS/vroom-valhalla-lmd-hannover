"""Clean, rigorous sweet-spot (Pareto-knee) derivation for the service penalty P.

Replaces the cluttered dual-axis cost/wait plot with ONE two-panel figure:

  Panel A  Pareto frontier (cost-saving vs avg wait), knee region shaded,
           P=0.5 operating point starred, endpoints labelled.
  Panel B  Diminishing-returns view: % of max saving CAPTURED vs % of max wait
           INCURRED, both as function of P. Their vertical gap == the Kneedle
           distance; it peaks across the knee region P in [0.30, 0.50].

Also writes tab_sweetspot_knee.csv with the per-P knee analysis and the
shadow-price check (marginal EUR/parcel-day == P, the optimisation's KKT cond).

Output: 05_optimization/fig_PF3_sweetspot.{png,pdf}, tab_sweetspot_knee.csv

DEPRECATED (2026-08 revision). Stale entry point: it recomputes totals
WITHOUT the pool term and predates the universal tour rule, the two cost
lenses and the operator polish, so its numbers are not comparable with the
current results. Use scripts/revision/61_grid_run_v2.py for the grid and
scripts/revision/70_figs_tables_v2.py for figures and tables.

Status B (Task 19): the original ``tab_penalty_finegrid_production.csv``
input is produced by a separate, out-of-scope-for-this-wave C-port script
(``paper_final_finegrid_production.py``) that re-runs the grid at a finer P
resolution -- not something this wave re-runs ("nothing heavy"). Instead
the same three columns (penalty, saving_pct, avg_wait) come from 74_-legacy's
tab_costs_smoothed.csv/tab_wait_smoothed.csv at v6's own (coarser, 8-point)
P grid via ``_paper_v6_common.build_penalty_series`` -- real v6 numbers at a
lower resolution, never an invented finer one. BASELINE_WEEKLY_EUR/
WEEKLY_PARCELS (the shadow-price cross-check's fixed constants) are
recomputed from the same v6 source rather than left at their pre-revision
values when --legacy-dir is given.
"""
from __future__ import annotations
import argparse
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# --- DEPRECATED ENTRY POINT (2026-08 revision) -----------------------------
import warnings as _deprecation_warnings

_deprecation_warnings.warn(
    "paper_final_sweetspot.py is a STALE entry point: it recomputes totals WITHOUT the pool "
    "term and predates the universal tour rule, the two cost lenses and the "
    "operator polish. Its numbers are NOT comparable with the 2026-08 "
    "revision. Use scripts/revision/61_grid_run_v2.py for the grid and "
    "scripts/revision/70_figs_tables_v2.py for figures and tables.",
    DeprecationWarning,
    stacklevel=2,
)
# ---------------------------------------------------------------------------

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paper_v6_common as V6  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "paper_final_2026_05_30" / "05_optimization"
rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 12.5,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Global constants for the shadow-price (EUR/parcel-day) cross-check
BASELINE_WEEKLY_EUR = 1_909_700.0     # daily-baseline weekly cost
WEEKLY_PARCELS = 1_263_130.0          # total parcels per week
KNEE_LO, KNEE_HI = 0.30, 0.50         # Kneedle knee region (>=95% max curvature)
P_STAR = 0.40                         # geometric Pareto-knee operating point
# v6's own P grid is the coarser {0, .25, .5, .75, 1, 2, 5, 10} (no finer
# sweep re-run this wave -- see module docstring); KNEE_LO/P_STAR are
# snapped to the nearest available grid points in v6 mode, set in main().


def main():
    global OUT, BASELINE_WEEKLY_EUR, WEEKLY_PARCELS, KNEE_LO, KNEE_HI, P_STAR
    ap = argparse.ArgumentParser(description=__doc__)
    V6.add_v6_cli_args(ap, needs_legacy=True)
    args = ap.parse_args()
    legacy_run = legacy_rev = None
    if args.legacy_dir is not None:
        legacy_run = Path(args.legacy_dir)
        legacy_rev = legacy_run.parent / "rev"
    elif args.rev_dir is not None:
        legacy_run, legacy_rev = V6.run_legacy_adapter(
            args.rev_dir, Path(args.out_dir or OUT) / "_legacy")
    if args.out_dir is not None:
        OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    if legacy_rev is not None:
        d = V6.build_penalty_series(legacy_rev, share=1.0)
        # Recompute the shadow-price cross-check's fixed constants from the
        # same v6 source instead of leaving the pre-revision numbers in
        # place: BASELINE_WEEKLY_EUR is theta=0's system total (any P);
        # WEEKLY_PARCELS is the demand total, which does not depend on P/theta.
        costs = pd.read_csv(legacy_rev / "tab_costs_smoothed.csv")
        BASELINE_WEEKLY_EUR = float(
            costs[np.isclose(costs.share_willing, 0.0)]
            .groupby("penalty").total_stage3_eur.sum().iloc[0])
        chosen = pd.read_csv(legacy_run / "tab_chosen_schedules.csv")
        one_pt = chosen[np.isclose(chosen.penalty, chosen.penalty.iloc[0])
                        & np.isclose(chosen.share_willing, chosen.share_willing.iloc[0])]
        WEEKLY_PARCELS = float(one_pt.weekly_parcels.sum())
        # v6 has no P=0.30/0.40 (only the standard 8-point grid) -- snap the
        # knee-region/operating-point markers to the nearest grid points
        # rather than crash on an exact-match lookup that can never succeed.
        KNEE_LO, KNEE_HI, P_STAR = 0.25, 0.5, 0.5
        src_note = ("B: 74_-legacy tab_costs_smoothed.csv/tab_wait_smoothed.csv "
                   "(v6's own 8-point P grid, not a finer re-run)")
    else:
        d = pd.read_csv(OUT / "tab_penalty_finegrid_production.csv")
        src_note = "tab_penalty_finegrid_production.csv (historical path)"
    d = d.sort_values("penalty")
    P = d.penalty.values
    S = d.saving_pct.values
    W = d.avg_wait.values
    Smax, Wmax = S.max(), W.max()

    # ── Kneedle knee: normalise, distance above the chord (0,0)->(1,1)
    o = np.argsort(W)
    Wn = (W[o] - W[o].min()) / (W[o].max() - W[o].min())
    Sn = (S[o] - S[o].min()) / (S[o].max() - S[o].min())
    dist = (Sn - Wn) / np.sqrt(2.0)
    knee_P = P[o][np.argmax(dist)]

    # ── marginal exchange rate dS/dW and shadow-price check (EUR/parcel-day)
    rows = []
    for i in range(len(d)):
        if i < len(d) - 1:
            dS = S[i] - S[i + 1]
            dW = W[i] - W[i + 1]
            rate_pp_day = dS / dW if abs(dW) > 1e-9 else np.nan
            # 1 pp saving = BASELINE_WEEKLY_EUR/100; per avg-day over all parcels
            eur_per_parcel_day = (rate_pp_day * BASELINE_WEEKLY_EUR / 100.0) / WEEKLY_PARCELS
        else:
            rate_pp_day = eur_per_parcel_day = np.nan
        rows.append({
            "penalty": P[i], "saving_pct": S[i], "avg_wait_d": W[i],
            "pct_of_max_saving": 100 * S[i] / Smax,
            "pct_of_max_wait": 100 * W[i] / Wmax,
            "marginal_pp_saving_per_wait_day": rate_pp_day,
            "implied_shadow_price_eur_per_parcel_day": eur_per_parcel_day,
        })
    kdf = pd.DataFrame(rows)
    kdf.to_csv(OUT / "tab_sweetspot_knee.csv", index=False)

    s_star = float(S[np.isclose(P, P_STAR)][0])
    w_star = float(W[np.isclose(P, P_STAR)][0])
    pct_save = 100 * s_star / Smax
    pct_wait = 100 * w_star / Wmax

    # ════════════════════════════════════════════════════════════════════
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))

    # ── Panel A: Pareto frontier ────────────────────────────────────────
    axA.axvspan(W[np.isclose(P, KNEE_HI)][0], W[np.isclose(P, KNEE_LO)][0],
                color="#ffe8cc", alpha=0.7, zorder=0, label=f"Knee region  P∈[{KNEE_LO:g},{KNEE_HI:g}]")
    axA.plot(W, S, "-", color="#1d3557", lw=2, zorder=2)
    axA.scatter(W, S, s=28, color="#1d3557", zorder=3)

    # star the operating point
    axA.scatter([w_star], [s_star], s=320, marker="*", color="#e63946",
                edgecolor="black", lw=0.8, zorder=5)
    axA.annotate(f"P = {P_STAR:g}  (sweet-spot)\n{s_star:.1f}% saving  ·  {w_star:.2f} d wait\n"
                 f"= {pct_save:.0f}% of max saving for {pct_wait:.0f}% of max wait",
                 xy=(w_star, s_star), xytext=(w_star + 0.18, s_star - 4.5),
                 fontsize=10, ha="left",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#e63946", lw=1.0),
                 arrowprops=dict(arrowstyle="->", color="#e63946", lw=1.2))

    # endpoints
    axA.annotate(f"P -> 0  (cost-optimal)\n{Smax:.1f}% @ {Wmax:.2f} d",
                 xy=(W[np.argmin(P)], S[np.argmin(P)]),
                 xytext=(W[np.argmin(P)] - 0.02, S[np.argmin(P)] + 0.4),
                 fontsize=9.5, ha="right", va="bottom", color="#264653")
    # "P >= 5 -> daily (0%)" is a false invariant on v6: the operator polish
    # keeps a small residual pooling saving past P=5 (compendium 40.23b), so
    # the annotation states the grid's OWN largest P>=5 saving instead of
    # asserting an exact zero it does not reach.
    s_tail = S[P >= 5].max() if (P >= 5).any() else S.min()
    axA.annotate(f"P >= 5  ->  <= {s_tail:.1f}% (near-daily, not exactly 0)",
                 xy=(0.0, 0.0), xytext=(0.06, 1.6),
                 fontsize=9.5, color="#6c757d")

    # geometric knee marker
    wk = W[np.isclose(P, knee_P)][0]; sk = S[np.isclose(P, knee_P)][0]
    axA.scatter([wk], [sk], s=90, marker="D", facecolor="none",
                edgecolor="#2a9d8f", lw=1.8, zorder=4,
                label=f"Geometric knee  P={knee_P:g}")

    axA.set_xlabel("Average customer wait  [days]")
    axA.set_ylabel("Weekly cost saving vs daily baseline  [%]")
    axA.set_title("A · Cost–service Pareto frontier")
    axA.set_xlim(-0.02, 1.02); axA.set_ylim(-1, 25)
    axA.grid(alpha=0.25)
    axA.legend(loc="lower right", fontsize=9, frameon=True)

    # ── Panel B: diminishing returns (saving captured vs wait incurred) ──
    m = P <= 1.55
    axB.axvspan(KNEE_LO, KNEE_HI, color="#ffe8cc", alpha=0.7, zorder=0)
    axB.plot(P[m], 100 * S[m] / Smax, "o-", color="#1d3557", lw=2,
             label="% of max saving captured")
    axB.plot(P[m], 100 * W[m] / Wmax, "s--", color="#e76f51", lw=2,
             label="% of max wait incurred")
    axB.fill_between(P[m], 100 * W[m] / Wmax, 100 * S[m] / Smax,
                     color="#a8dadc", alpha=0.35, zorder=1)
    axB.axvline(P_STAR, color="#e63946", lw=1.4, ls=":")
    axB.annotate(f"P = {P_STAR:g}", xy=(P_STAR, 5), xytext=(P_STAR + 0.03, 5),
                 color="#e63946", fontsize=10, fontweight="bold")
    axB.annotate("", xy=(P_STAR, pct_save), xytext=(P_STAR, pct_wait),
                 arrowprops=dict(arrowstyle="<->", color="#1d3557", lw=1.3))
    axB.annotate(f"gap = {pct_save - pct_wait:.0f} pp\n(saving kept ≫ wait paid)",
                 xy=(P_STAR, (pct_save + pct_wait) / 2),
                 xytext=(P_STAR + 0.06, (pct_save + pct_wait) / 2 + 3), fontsize=9.5, color="#1d3557")

    axB.set_xlabel("Service penalty P  [€ / parcel / day]  =  shadow price of waiting")
    axB.set_ylabel("Fraction of maximum  [%]")
    axB.set_title("B · Diminishing returns — the gap peaks in the knee region")
    axB.set_xlim(-0.03, 1.55); axB.set_ylim(0, 103)
    axB.grid(alpha=0.25)
    axB.legend(loc="upper right", fontsize=9.5, frameon=True)

    fig.suptitle(f"Service-penalty sweet-spot — why P = {P_STAR:g} €/parcel/day",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    V6.add_provenance_footer(fig, plan="operator-polished (balanced), routing lens",
                             script="paper_final_sweetspot.py", source=src_note)
    V6.savefig_pair(fig, OUT / "fig_PF3_sweetspot.png", OUT / "fig_PF3_sweetspot.pdf")
    plt.close(fig)

    print(f"  Kneedle knee  P={knee_P:g}  (region [{KNEE_LO},{KNEE_HI}])")
    print(f"  Operating point P=0.50 -> {s_star:.1f}% saving, {w_star:.2f}d wait "
          f"({100*s_star/Smax:.0f}% of max saving, {100*w_star/Wmax:.0f}% of max wait)")
    print("  Shadow-price check (implied EUR/parcel-day should ~= P):")
    for pp in (0.3, 0.5, 1.0):
        r = kdf[np.isclose(kdf.penalty, pp)]
        if len(r):
            print(f"    P={pp:>4}  -> {r.implied_shadow_price_eur_per_parcel_day.iloc[0]:.2f} €/parcel-day")
    print("  [OK] fig_PF3_sweetspot + tab_sweetspot_knee.csv")


if __name__ == "__main__":
    main()
