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
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_final_2026_05_28" / "05_optimization"
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


def main():
    d = pd.read_csv(OUT / "tab_penalty_finegrid_production.csv").sort_values("penalty")
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
    axA.annotate(f"P → 0  (cost-optimal)\n{Smax:.1f}% @ {Wmax:.2f} d",
                 xy=(W[np.argmin(P)], S[np.argmin(P)]),
                 xytext=(W[np.argmin(P)] - 0.02, S[np.argmin(P)] + 0.4),
                 fontsize=9.5, ha="right", va="bottom", color="#264653")
    axA.annotate("P ≥ 5  →  daily (0%)", xy=(0.0, 0.0), xytext=(0.06, 1.6),
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
    fig.savefig(OUT / "fig_PF3_sweetspot.png")
    fig.savefig(OUT / "fig_PF3_sweetspot.pdf")
    plt.close(fig)

    print(f"  Kneedle knee  P={knee_P:g}  (region [{KNEE_LO},{KNEE_HI}])")
    print(f"  Operating point P=0.50 -> {s_star:.1f}% saving, {w_star:.2f}d wait "
          f"({100*s_star/Smax:.0f}% of max saving, {100*w_star/Wmax:.0f}% of max wait)")
    print("  Shadow-price check (implied EUR/parcel-day should ~= P):")
    for pp in (0.3, 0.5, 1.0):
        r = kdf[np.isclose(kdf.penalty, pp)]
        if len(r):
            print(f"    P={pp:>4}  -> {r.implied_shadow_price_eur_per_parcel_day.iloc[0]:.2f} €/parcel-day")
    print("  ✓ fig_PF3_sweetspot + tab_sweetspot_knee.csv")


if __name__ == "__main__":
    main()
