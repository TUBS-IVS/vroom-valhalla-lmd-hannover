"""Per-provider Pareto frontier at theta=1.0 from Path-2 data. Each provider
has its own daily baseline and its own cost-vs-wait curve — providers with
lower demand density consolidate more aggressively, providers with high demand
density (DHL) can barely batch.

Outputs:
  results/paper_final_2026_05_30/05_optimization/fig_PF6_provider_pareto.{png,pdf}
  results/paper_final_2026_05_30/05_optimization/_per_provider_pareto.csv
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
OUT = ROOT / "results" / "paper_final_2026_05_30" / "05_optimization"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 10,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "savefig.bbox": "tight", "savefig.dpi": 220, "pdf.fonttype": 42,
})

PROV_COLORS = {
    "Amazon":  "#1d3557",
    "DHL":     "#e63946",
    "DPD":     "#2a9d8f",
    "FedEx":   "#e76f51",
    "GLS":     "#f4a261",
    "Hermes":  "#264653",
    "UPS":     "#9d4edd",
}


def main():
    summ = pd.read_csv(PATH2 / "tab_balancing_summary.csv")
    chosen = pd.read_csv(PATH2 / "tab_chosen_schedules.csv")

    # Provider baselines
    base_per_prov = (summ[np.isclose(summ.share_willing, 0.0)]
                     .groupby("provider").balanced_cost_eur.mean())
    print("Daily baseline per provider [k€/wk]:")
    for p, c in base_per_prov.items():
        print(f"  {p:<7} {c/1000:7.1f}")

    # Theta = 1 row
    th = 1.0
    s1 = summ[np.isclose(summ.share_willing, th)]
    c1 = chosen[np.isclose(chosen.share_willing, th)]

    rows = []
    for prov in sorted(s1.provider.unique()):
        base = float(base_per_prov[prov])
        for P in sorted(s1.penalty.unique()):
            gp = s1[(s1.provider == prov) & np.isclose(s1.penalty, P)]
            cp = c1[(c1.provider == prov) & np.isclose(c1.penalty, P)]
            if len(gp) == 0 or len(cp) == 0:
                continue
            cost = gp.init_cost_eur.iloc[0]
            sav = 100 * (base - cost) / base
            wait = float((cp.avg_wait_d_init * cp.weekly_parcels).sum()
                          / cp.weekly_parcels.sum())
            mean_freq = float(cp.schedule_size_init.mean())
            rows.append({"provider": prov, "penalty": P, "wait_d": wait,
                          "sav_pct": sav, "mean_freq": mean_freq,
                          "baseline_k": base / 1000})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "_per_provider_pareto.csv", index=False)

    # ─── Plot: per-provider Pareto ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for prov in sorted(df.provider.unique()):
        d = df[df.provider == prov].sort_values("penalty")
        ax.plot(d.wait_d, d.sav_pct, "o-", color=PROV_COLORS[prov],
                 linewidth=2, markersize=7,
                 label=f"{prov:<7} (Baseline {d.baseline_k.iloc[0]:.0f}k€/Wo)")
        # Mark P=0.5 sweet-spot for each
        sweet = d[np.isclose(d.penalty, 0.5)]
        if len(sweet):
            ax.scatter([sweet.wait_d.iloc[0]], [sweet.sav_pct.iloc[0]],
                        marker="*", s=300, color=PROV_COLORS[prov],
                        edgecolor="black", linewidth=1.0, zorder=5)
    ax.set_xlabel("Average customer wait [days] (parcels-weighted, $\\theta = 1$)")
    ax.set_ylabel("Cost saving vs provider's daily baseline [%]")
    ax.set_title("Per-provider Pareto frontiers — sweet-spot P = 0.5 marked",
                  fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_xlim(-0.02)
    ax.set_ylim(-1)
    fig.tight_layout()
    fig.savefig(OUT / "fig_PF6_provider_pareto.png", bbox_inches="tight")
    fig.savefig(OUT / "fig_PF6_provider_pareto.pdf", bbox_inches="tight")
    plt.close(fig)

    # ─── Summary table ──────────────────────────────────────────────────
    sweet_per_prov = df[np.isclose(df.penalty, 0.5)][[
        "provider", "wait_d", "sav_pct", "mean_freq", "baseline_k"]]
    max_sav_per_prov = df.loc[df.groupby("provider").sav_pct.idxmax(),
                                ["provider", "penalty", "wait_d", "sav_pct"]]
    print("\n=== Per-provider analysis at sweet-spot P=0.5, theta=1 ===")
    print(sweet_per_prov.round(2).to_string(index=False))
    print("\n=== Maximum saving (P=0) per provider ===")
    print(max_sav_per_prov.round(2).to_string(index=False))
    print(f"\nsaved {OUT/'fig_PF6_provider_pareto.png'}")


if __name__ == "__main__":
    main()
