"""Render the sweet-spot 2-panel and saving 3-panel figures in WIDER aspect
ratios for the EWGT paper, flat into results/EWGT_Results/.
"""
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
PATH2 = ROOT / "results" / "overnight_2026_05_29_path2"
OUT = ROOT / "results" / "EWGT_Results"
OUT.mkdir(parents=True, exist_ok=True)
BASE = 1909747.75
SWEET_P = 0.5
KNEE_LO, KNEE_HI = 0.25, 0.75

rcParams.update({
    "font.family": "serif", "font.size": 11,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load():
    s = pd.read_csv(PATH2 / "tab_balancing_summary.csv")
    c = pd.read_csv(PATH2 / "tab_chosen_schedules.csv"); c["plz"] = c.plz.astype(str)
    return s, c


def pareto_at_theta1(s, c):
    th = 1.0
    s1 = s[np.isclose(s.share_willing, th)]
    c1 = c[np.isclose(c.share_willing, th)]
    rows = []
    for P in sorted(s1.penalty.unique()):
        gp = s1[np.isclose(s1.penalty, P)]
        cp = c1[np.isclose(c1.penalty, P)]
        cost = gp.init_cost_eur.sum()
        wait = (cp.avg_wait_d_init * cp.weekly_parcels).sum() / cp.weekly_parcels.sum()
        rows.append({"penalty": P, "wait_d": wait,
                      "sav_pct": 100 * (BASE - cost) / BASE})
    return pd.DataFrame(rows).sort_values("penalty").reset_index(drop=True)


# ─── Sweet-spot 2-panel (WIDER) ────────────────────────────────────────
def fig_sweetspot_wide(s, c):
    par = pareto_at_theta1(s, c)
    sweet = par[np.isclose(par.penalty, SWEET_P)].iloc[0]
    max_sav = par.sav_pct.max(); max_wait = par.wait_d.max()
    par["sav_n"] = par.sav_pct / max_sav * 100
    par["wait_n"] = par.wait_d / max_wait * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Panel A
    ax1.axvspan(par[np.isclose(par.penalty, KNEE_LO)].wait_d.iloc[0],
                 par[np.isclose(par.penalty, KNEE_HI)].wait_d.iloc[0],
                 color="orange", alpha=0.18,
                 label=rf"Knee region $P \in [{KNEE_LO:g}, {KNEE_HI:g}]$")
    ax1.plot(par.wait_d, par.sav_pct, "o-", color="#1d3557",
              linewidth=1.8, markersize=7)
    ax1.scatter([sweet.wait_d], [sweet.sav_pct], marker="*",
                 s=420, color="gold", edgecolor="black",
                 linewidth=1.2, zorder=10,
                 label=rf"Geometric knee $P = {SWEET_P:g}$")
    for _, r in par.iterrows():
        ax1.annotate(rf"${r.penalty:g}$", (r.wait_d, r.sav_pct),
                      xytext=(6, 4), textcoords="offset points",
                      fontsize=8, color="#555")
    ax1.set_xlabel(r"Average wait [d] ($\theta = 1$)")
    ax1.set_ylabel("Cost saving vs daily baseline [%]")
    ax1.set_title("(a) Cost-service Pareto frontier")
    ax1.grid(alpha=0.3); ax1.legend(loc="lower right")

    # Panel B
    ax2.plot(par.penalty, par.sav_n, "o-", color="#1d3557",
              linewidth=1.8, markersize=7, label="% of max saving")
    ax2.plot(par.penalty, par.wait_n, "s--", color="#e76f51",
              linewidth=1.8, markersize=7, label="% of max wait")
    ax2.axvspan(KNEE_LO, KNEE_HI, color="orange", alpha=0.18)
    ax2.axvline(SWEET_P, color="#e63946", linestyle=":", linewidth=1)
    ax2.fill_between(par.penalty, par.sav_n, par.wait_n,
                      color="#2a9d8f", alpha=0.13)
    ax2.set_xlabel(r"Service penalty $P$ [€/p/d]")
    ax2.set_ylabel("Fraction of maximum [%]")
    ax2.set_title("(b) Diminishing returns")
    ax2.legend(loc="upper right"); ax2.grid(alpha=0.3)
    ax2.set_ylim(-5, 105)

    fig.tight_layout()
    fig.savefig(OUT / "fig_PF3_sweetspot.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_PF3_sweetspot.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved fig_PF3_sweetspot (11.5\" wide)")


# ─── Saving 3-panel heatmaps (WIDER) ───────────────────────────────────
def fig_saving_heatmaps_wide(s):
    ag = s.groupby(["penalty", "share_willing"], as_index=False).agg(
        init=("init_cost_eur", "sum"),
        bal=("balanced_cost_eur", "sum"),
        peak_b=("max_fleet_before", "sum"),
        peak_a=("max_fleet_after", "sum"))
    ag = ag[~np.isclose(ag.penalty, 0.4)]
    ag["init_sav"] = 100 * (BASE - ag.init) / BASE
    ag["bal_sav"]  = 100 * (BASE - ag.bal)  / BASE
    ag["peak_red"] = 100 * (ag.peak_b - ag.peak_a) / ag.peak_b.clip(lower=1)

    pivI = ag.pivot(index="penalty", columns="share_willing", values="init_sav")
    pivB = ag.pivot(index="penalty", columns="share_willing", values="bal_sav")
    pivP = ag.pivot(index="penalty", columns="share_willing", values="peak_red")
    sV_min = min(pivI.values.min(), pivB.values.min())
    sV_max = max(pivI.values.max(), pivB.values.max())
    pP_max = float(np.ceil(pivP.values.max() / 5) * 5)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    def heat(ax, mat, cmap, vmin, vmax, title):
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f"{x*100:.0f}" for x in mat.columns], fontsize=8)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f"{p:g}" for p in mat.index], fontsize=8)
        ax.set_xlabel(r"$\theta$ [%]", fontsize=10)
        ax.set_title(title, fontsize=11)
        thr = vmin + (vmax - vmin) * 0.55
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isnan(v): continue
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v < thr else "black", fontsize=6.5)
        return im

    im_a = heat(axes[0], pivI, "viridis", sV_min, sV_max,
                 "(a) Cost saving, cost-optimal [%]")
    axes[0].set_ylabel(r"$P$ [€/p/d]", fontsize=10)
    heat(axes[1], pivB, "viridis", sV_min, sV_max,
          "(b) Cost saving, fleet-balanced [%]")
    im_c = heat(axes[2], pivP, "magma", 0, pP_max,
                 "(c) Peak-fleet reduction [%]")
    cb1 = fig.colorbar(im_a, ax=axes[1], fraction=0.046, pad=0.03)
    cb1.set_label("[%]", fontsize=9); cb1.ax.tick_params(labelsize=8)
    cb2 = fig.colorbar(im_c, ax=axes[2], fraction=0.046, pad=0.03)
    cb2.set_label("[%]", fontsize=9); cb2.ax.tick_params(labelsize=8)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / "fig_PF2_saving_fleet_heatmaps.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_PF2_saving_fleet_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved fig_PF2_saving_fleet_heatmaps (13.5\" wide)")


def main():
    s, c = load()
    print("=== Wider figures for EWGT_Results/ ===")
    fig_sweetspot_wide(s, c)
    fig_saving_heatmaps_wide(s)
    print("\nDone.")


if __name__ == "__main__":
    main()
