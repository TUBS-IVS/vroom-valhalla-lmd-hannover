"""Reproduce the three sweet-spot figures from yesterday with Path-2 data:

  1) Two-panel: (A) Pareto frontier with sweet-spot star + knee region,
                (B) Diminishing returns — saving captured % vs wait incurred %.
  2) Single-panel scatter with viridis colormap encoding P.
  3) Pareto efficiency curve: saving kept - wait paid as function of P.

The new sweet-spot is P = 0.5 (knee from chord-distance method on the
8-point Path-2 Pareto frontier).

Outputs to results/paper_final_2026_05_30/05_optimization/:
  fig_PF3_sweetspot.{png,pdf}
  fig_pareto_viridis.{png,pdf}
  fig_pareto_efficiency.{png,pdf}
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams, cm

ROOT = Path(__file__).resolve().parents[1]
PATH2 = ROOT / "results" / "overnight_2026_05_29_path2"
OUT_BASE = ROOT / "results" / "paper_final_2026_05_30"
OUT = OUT_BASE / "05_optimization"
OUT.mkdir(parents=True, exist_ok=True)
BASE = 1909747.75

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 11, "axes.titlesize": 12, "legend.fontsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "savefig.bbox": "tight", "savefig.dpi": 220, "pdf.fonttype": 42,
})

SWEET_P = 0.5
KNEE_LO, KNEE_HI = 0.25, 0.75


def load_pareto():
    summ = pd.read_csv(PATH2 / "tab_balancing_summary.csv")
    chosen = pd.read_csv(PATH2 / "tab_chosen_schedules.csv")
    th = 1.0
    cs = chosen[np.isclose(chosen.share_willing, th)]
    ss = summ[np.isclose(summ.share_willing, th)]
    rows = []
    for P in sorted(cs.penalty.unique()):
        g = cs[np.isclose(cs.penalty, P)]
        gs = ss[np.isclose(ss.penalty, P)]
        wait = (g.avg_wait_d_init * g.weekly_parcels).sum() / g.weekly_parcels.sum()
        cost = gs.init_cost_eur.sum()
        sav = 100 * (BASE - cost) / BASE
        rows.append({"penalty": P, "wait_d": wait, "sav_pct": sav,
                      "mean_freq": g.schedule_size_init.mean(),
                      "n_batched": int((g.schedule_size_init < 6).sum())})
    return pd.DataFrame(rows).sort_values("penalty").reset_index(drop=True)


# ─── Figure 1: 2-panel sweet-spot ──────────────────────────────────────
def fig_sweetspot_2panel(pareto: pd.DataFrame) -> None:
    sweet_row = pareto[np.isclose(pareto.penalty, SWEET_P)].iloc[0]
    max_sav = pareto.sav_pct.max()
    max_wait = pareto.wait_d.max()
    pct_sav = 100 * sweet_row.sav_pct / max_sav
    pct_wait = 100 * sweet_row.wait_d / max_wait

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    fig.suptitle(rf"Service-penalty sweet-spot — why $P = {SWEET_P:g}$ €/parcel/day",
                  fontsize=14, y=1.02)

    # Panel A: Pareto frontier
    ax1.axvspan(pareto[np.isclose(pareto.penalty, KNEE_LO)].wait_d.iloc[0],
                 pareto[np.isclose(pareto.penalty, KNEE_HI)].wait_d.iloc[0],
                 color="orange", alpha=0.18, label=f"Knee region $P\\in[{KNEE_LO:g},{KNEE_HI:g}]$")
    ax1.plot(pareto.wait_d, pareto.sav_pct, "o-", color="#1d3557",
              linewidth=2.2, markersize=7)
    ax1.scatter([sweet_row.wait_d], [sweet_row.sav_pct], marker="*",
                 s=540, color="#e63946", edgecolor="black", linewidth=1.4,
                 zorder=10, label=f"Geometric knee  $P={SWEET_P:g}$")
    # Annotate cost-optimal and sweet-spot
    max_row = pareto.iloc[pareto.sav_pct.idxmax()]
    ax1.annotate(rf"$P\to 0$ (cost-optimal)" "\n"
                  f"{max_row.sav_pct:.1f}% @ {max_row.wait_d:.2f} d",
                  (max_row.wait_d, max_row.sav_pct),
                  xytext=(-15, -20), textcoords="offset points",
                  fontsize=10, color="#1d3557", ha="right")
    ax1.annotate(f"$P = {SWEET_P:g}$ (sweet-spot)\n"
                  f"{sweet_row.sav_pct:.1f}% saving  ·  {sweet_row.wait_d:.2f} d wait\n"
                  f"= {pct_sav:.0f}% of max saving for {pct_wait:.0f}% of max wait",
                  (sweet_row.wait_d, sweet_row.sav_pct),
                  xytext=(60, -50), textcoords="offset points",
                  fontsize=9.5, color="#1d3557",
                  arrowprops=dict(arrowstyle="-", color="#e63946", lw=1),
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                             edgecolor="#e63946"))
    # Annotate P>=2 region
    daily_rows = pareto[pareto.sav_pct < 0.5]
    if len(daily_rows):
        first_daily = daily_rows.iloc[0]
        ax1.text(0.02, 1.5, rf"$P \geq 5 \to$ daily (0%)",
                  fontsize=10, color="gray")
    ax1.set_xlabel("Average customer wait  [days]")
    ax1.set_ylabel("Weekly cost saving vs daily baseline  [%]")
    ax1.set_title(r"A · Cost-service Pareto frontier", fontsize=12, loc="left")
    ax1.grid(alpha=0.3); ax1.legend(loc="lower right")
    ax1.set_xlim(-0.02, max_wait * 1.05); ax1.set_ylim(-1, max_sav + 2)

    # Panel B: Diminishing returns — % of max saving captured + % of max wait
    pareto["sav_pct_of_max"] = 100 * pareto.sav_pct / max_sav
    pareto["wait_pct_of_max"] = 100 * pareto.wait_d / max_wait
    ax2.plot(pareto.penalty, pareto.sav_pct_of_max, "o-", color="#1d3557",
              linewidth=2.2, markersize=7, label="% of max saving captured")
    ax2.plot(pareto.penalty, pareto.wait_pct_of_max, "s--", color="#e76f51",
              linewidth=2.2, markersize=7, label="% of max wait incurred")
    ax2.axvspan(KNEE_LO, KNEE_HI, color="orange", alpha=0.18)
    ax2.axvline(SWEET_P, color="#e63946", linestyle=":", linewidth=1.2)
    ax2.text(SWEET_P, -8, f"$P = {SWEET_P:g}$", color="#e63946",
              ha="center", fontsize=10, fontweight="bold")
    gap = sweet_row.sav_pct / max_sav * 100 - sweet_row.wait_d / max_wait * 100
    ax2.annotate(f"gap = {gap:.0f} pp\n(saving kept − wait paid)",
                 (SWEET_P, (pct_sav + pct_wait) / 2),
                 xytext=(0.7, 50), textcoords="data",
                 fontsize=9.5, color="#1d3557",
                 arrowprops=dict(arrowstyle="<->", color="#1d3557", lw=1.2))
    ax2.fill_between(pareto.penalty, pareto.sav_pct_of_max,
                      pareto.wait_pct_of_max, color="#2a9d8f", alpha=0.12)
    ax2.set_xlabel(r"Service penalty $P$  [€ / parcel / day]  = shadow price of waiting")
    ax2.set_ylabel("Fraction of maximum  [%]")
    ax2.set_title("B · Diminishing returns — the gap peaks in the knee region",
                   fontsize=12, loc="left")
    ax2.legend(loc="upper right"); ax2.grid(alpha=0.3)
    ax2.set_ylim(-5, 105); ax2.set_xlim(-0.1, pareto.penalty.max() * 1.05)

    fig.tight_layout()
    fig.savefig(OUT / "fig_PF3_sweetspot.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_PF3_sweetspot.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved fig_PF3_sweetspot")


# ─── Figure 2: viridis scatter ─────────────────────────────────────────
def fig_pareto_viridis(pareto: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    # Smooth curve through points (use spline-ish via PCHIP)
    from scipy.interpolate import PchipInterpolator
    x_fine = np.linspace(pareto.wait_d.min(), pareto.wait_d.max(), 200)
    try:
        pchip = PchipInterpolator(pareto.wait_d.values, pareto.sav_pct.values)
        y_fine = pchip(x_fine)
        ax.plot(x_fine, y_fine, "-", color="#888888", linewidth=1.2, alpha=0.6,
                 zorder=2)
    except Exception:
        ax.plot(pareto.wait_d, pareto.sav_pct, "-", color="#888888", linewidth=1.2, alpha=0.6,
                 zorder=2)
    # Penalty as colormap (log scale, viridis_r so 0 is dark)
    eps = 0.005
    P_plot = pareto.penalty.copy()
    P_plot[P_plot == 0] = eps  # avoid log(0)
    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=max(eps, P_plot.min()), vmax=P_plot.max())
    sc = ax.scatter(pareto.wait_d, pareto.sav_pct, c=P_plot,
                     cmap="viridis_r", norm=norm, s=130,
                     edgecolor="black", linewidth=0.5, zorder=5)
    sweet_row = pareto[np.isclose(pareto.penalty, SWEET_P)].iloc[0]
    ax.scatter([sweet_row.wait_d], [sweet_row.sav_pct], marker="o",
                s=400, facecolor="none", edgecolor="#e63946", linewidth=2.5,
                zorder=11)
    knee_lo_row = pareto[np.isclose(pareto.penalty, KNEE_LO)].iloc[0]
    knee_hi_row = pareto[np.isclose(pareto.penalty, KNEE_HI)].iloc[0]
    ax.axvspan(knee_lo_row.wait_d, knee_hi_row.wait_d, color="orange", alpha=0.18)
    ax.set_xlabel("Average customer wait  [days]")
    ax.set_ylabel("Weekly cost saving vs daily baseline  [%]")
    ax.grid(alpha=0.3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"Service penalty $P$  [€ / parcel / day]")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pareto_viridis.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_pareto_viridis.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved fig_pareto_viridis")


# ─── Figure 3: Pareto efficiency (saving kept - wait paid) ────────────
def fig_pareto_efficiency(pareto: pd.DataFrame) -> None:
    max_sav = pareto.sav_pct.max()
    max_wait = pareto.wait_d.max()
    pareto = pareto.copy()
    pareto["efficiency"] = (100 * pareto.sav_pct / max_sav
                             - 100 * pareto.wait_d / max_wait)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(pareto.penalty, pareto.efficiency, "o-", color="#1d3557",
             linewidth=2, markersize=8)
    sweet_eff = pareto[np.isclose(pareto.penalty, SWEET_P)].iloc[0].efficiency
    ax.axvspan(KNEE_LO, KNEE_HI, color="orange", alpha=0.18)
    ax.scatter([SWEET_P], [sweet_eff], marker="o", s=320,
                facecolor="none", edgecolor="#e63946", linewidth=2.5, zorder=10)
    ax.set_xlabel(r"Service penalty $P$  [€ / parcel / day]")
    ax.set_ylabel("Pareto efficiency:  saving kept − wait paid  [pp]")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, pareto.penalty.max() * 1.05)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pareto_efficiency.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_pareto_efficiency.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved fig_pareto_efficiency  (peak at P={pareto.penalty[pareto.efficiency.idxmax()]:g})")


def main():
    pareto = load_pareto()
    pareto.to_csv(OUT_BASE / "_pareto_path2_theta1.csv", index=False)
    print(f"Loaded {len(pareto)} Pareto points (theta=1)")
    print(pareto[["penalty", "wait_d", "sav_pct", "mean_freq"]].round(3).to_string(index=False))
    fig_sweetspot_2panel(pareto)
    fig_pareto_viridis(pareto)
    fig_pareto_efficiency(pareto)


if __name__ == "__main__":
    main()
