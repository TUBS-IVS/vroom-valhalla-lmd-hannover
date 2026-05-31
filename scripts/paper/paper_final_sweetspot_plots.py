"""Clean, annotation-free PAPER figures around the service-penalty sweet-spot.

No in-figure explanations (captions carry those). Four standalone figures:

  fig_PF3b_pareto_clean   Pareto frontier (wait vs saving), points coloured by P,
                          knee region shaded, geometric knee ringed.
  fig_PF4_knee_curve      Pareto efficiency  (S/Smax − W/Wmax)  vs P — single hump,
                          peak = geometric knee at P≈0.4.
  fig_PF5_shadow_price    Implied marginal €/parcel-day (central diff) vs nominal P,
                          with y=x reference — confirms P is the shadow price.
  fig_PF6_provider_pareto Per-LSP Pareto frontiers at share=100% — heterogeneity.
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
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "paper_final_2026_05_30" / "05_optimization"
rcParams.update({
    "font.family": "serif", "font.size": 12,
    "axes.labelsize": 13, "axes.titlesize": 13.5,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
PROV_COLOR = {"DHL": "#d62828", "Amazon": "#003049", "DPD": "#f77f00",
              "FedEx": "#5a189a", "GLS": "#2a9d8f", "Hermes": "#9d4edd", "UPS": "#7d5a50"}
KNEE_LO, KNEE_HI, P_KNEE = 0.30, 0.50, 0.40
BASELINE_WEEKLY_EUR, WEEKLY_PARCELS = 1_909_700.0, 1_263_130.0


def _frontier():
    d = pd.read_csv(OUT / "tab_penalty_finegrid_production.csv").sort_values("penalty")
    return d.penalty.values, d.saving_pct.values, d.avg_wait.values


def fig_pareto_clean(P, S, W):
    Smax, Wmax = S.max(), W.max()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.axvspan(W[np.isclose(P, KNEE_HI)][0], W[np.isclose(P, KNEE_LO)][0],
               color="#ffe8cc", alpha=0.6, zorder=0)
    ax.plot(W, S, "-", color="#adb5bd", lw=1.4, zorder=1)
    sc = ax.scatter(W, S, c=P, cmap="viridis_r", s=60, zorder=3,
                    norm=matplotlib.colors.LogNorm(vmin=0.02, vmax=10))
    # ring the geometric knee
    wk, sk = W[np.isclose(P, P_KNEE)][0], S[np.isclose(P, P_KNEE)][0]
    ax.scatter([wk], [sk], s=300, marker="o", facecolor="none",
               edgecolor="#e63946", lw=2.2, zorder=4)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Service penalty P  [€ / parcel / day]")
    cb.set_ticks([0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10])
    cb.set_ticklabels(["0.05", "0.1", "0.25", "0.5", "1", "2", "5", "10"])
    ax.set_xlabel("Average customer wait  [days]")
    ax.set_ylabel("Weekly cost saving vs daily baseline  [%]")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-1, 25); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_PF3b_pareto_clean.png"); fig.savefig(OUT / "fig_PF3b_pareto_clean.pdf")
    plt.close(fig); print("  [OK] fig_PF3b_pareto_clean")


def fig_knee_curve(P, S, W):
    gap = 100 * S / S.max() - 100 * W / W.max()
    m = P <= 2.05
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axvspan(KNEE_LO, KNEE_HI, color="#ffe8cc", alpha=0.6, zorder=0)
    ax.plot(P[m], gap[m], "o-", color="#1d3557", lw=2, ms=5, zorder=2)
    pk = np.argmax(gap)
    ax.scatter([P[pk]], [gap[pk]], s=240, marker="o", facecolor="none",
               edgecolor="#e63946", lw=2.2, zorder=3)
    ax.set_xlabel("Service penalty P  [€ / parcel / day]")
    ax.set_ylabel("Pareto efficiency:  saving kept − wait paid  [pp]")
    ax.set_xlim(-0.04, 2.05); ax.set_ylim(0, 45); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_PF4_knee_curve.png"); fig.savefig(OUT / "fig_PF4_knee_curve.pdf")
    plt.close(fig); print("  [OK] fig_PF4_knee_curve")


def fig_shadow_price(P, S, W):
    # central-difference marginal saving per wait-day -> €/parcel-day, at each interior P
    Pp, rate = [], []
    for i in range(1, len(P) - 1):
        dS = S[i - 1] - S[i + 1]; dW = W[i - 1] - W[i + 1]
        if abs(dW) < 1e-9:
            continue
        eur = (dS / dW) * BASELINE_WEEKLY_EUR / 100.0 / WEEKLY_PARCELS
        Pp.append(P[i]); rate.append(eur)
    Pp, rate = np.array(Pp), np.array(rate)
    fig, ax = plt.subplots(figsize=(6.8, 6))
    lim = [0.02, 2.2]
    ax.plot(lim, lim, "--", color="#adb5bd", lw=1.4, zorder=1)
    ax.scatter(Pp, rate, s=70, color="#2a9d8f", edgecolor="black", lw=0.5, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("Nominal service penalty P  [€ / parcel / day]")
    ax.set_ylabel("Implied marginal saving  [€ / parcel / day]")
    ax.set_aspect("equal"); ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig_PF5_shadow_price.png"); fig.savefig(OUT / "fig_PF5_shadow_price.pdf")
    plt.close(fig); print("  [OK] fig_PF5_shadow_price")


def fig_provider_pareto():
    ch = pd.read_csv(OUT / "tab_chosen_schedules_full.csv")
    s100 = ch[ch.share_willing == ch.share_willing.max()]
    Pmax = s100.penalty.max()
    daily = s100[s100.penalty == Pmax].groupby("provider").dd_cost_balanced.sum()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for prov, col in PROV_COLOR.items():
        sub = s100[s100.provider == prov]
        if not len(sub):
            continue
        base = daily.get(prov, np.nan)
        xs, ys = [], []
        for P in sorted(sub.penalty.unique()):
            g = sub[sub.penalty == P]
            ys.append(100 * (base - g.dd_cost_balanced.sum()) / base)
            xs.append((g.avg_wait_d_balanced * g.weekly_parcels).sum() / g.weekly_parcels.sum())
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], "-o", color=col,
                lw=1.8, ms=4.5, label=prov)
    ax.set_xlabel("Average customer wait  [days]")
    ax.set_ylabel("Weekly cost saving vs daily baseline  [%]")
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(-1, 30); ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=10, ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_PF6_provider_pareto.png"); fig.savefig(OUT / "fig_PF6_provider_pareto.pdf")
    plt.close(fig); print("  [OK] fig_PF6_provider_pareto")


def main():
    P, S, W = _frontier()
    fig_pareto_clean(P, S, W)
    fig_knee_curve(P, S, W)
    fig_shadow_price(P, S, W)
    fig_provider_pareto()
    print("Done — 4 clean paper figures written.")


if __name__ == "__main__":
    main()
