"""Pareto frontier at theta=1.0 from the Path-2 init schedules. Plots
(cost saving %, parcels-weighted average wait [d]) for each available penalty
P, marks the sweet-spot (Pareto knee), and prints the table.

Outputs:
  results/overnight_2026_05_29_path2/_fig_pareto_path2.png
  results/overnight_2026_05_29_path2/_pareto_path2.csv
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
OUT = ROOT / "results" / "overnight_2026_05_29_path2"
BASE = 1909747.75

rcParams.update({
    "font.family": "serif", "font.size": 10,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "savefig.bbox": "tight", "savefig.dpi": 200, "pdf.fonttype": 42,
})


def main():
    summ = pd.read_csv(OUT / "tab_balancing_summary.csv")
    chosen = pd.read_csv(OUT / "tab_chosen_schedules.csv")

    # theta = 1 column only
    th = 1.0
    cs = chosen[np.isclose(chosen.share_willing, th)]
    ss = summ[np.isclose(summ.share_willing, th)]

    pareto = (cs.groupby("penalty")
              .apply(lambda g: pd.Series({
                  "wait_d": (g.avg_wait_d_init * g.weekly_parcels).sum()
                            / g.weekly_parcels.sum(),
                  "mean_freq": g.schedule_size_init.mean(),
                  "n_batched": int((g.schedule_size_init < 6).sum()),
              })).reset_index())
    cost = ss.groupby("penalty").init_cost_eur.sum().reset_index()
    pareto = pareto.merge(cost, on="penalty")
    pareto["init_sav_pct"] = 100 * (BASE - pareto.init_cost_eur) / BASE
    pareto = pareto.sort_values("penalty")

    # Knee identification: maximize curvature on the (wait, saving) curve
    if len(pareto) >= 4:
        x = pareto.wait_d.values
        y = pareto.init_sav_pct.values
        # Normalize and compute distance from line (max-saving point to zero)
        x_n = (x - x.min()) / (x.max() - x.min() + 1e-9)
        y_n = (y - y.min()) / (y.max() - y.min() + 1e-9)
        # Distance from the line connecting (max-wait, max-sav) to (0,0)
        # the knee is the point farthest from the chord (1,1)-(0,0)
        dists = np.abs(y_n - x_n)  # diagonal chord distance
        knee_idx = int(dists.argmax())
        knee = pareto.iloc[knee_idx]
    else:
        knee = pareto.iloc[len(pareto) // 2]

    print("=== Pareto frontier @ theta=1.0 (Path 2 init) ===")
    print(pareto[["penalty", "wait_d", "init_sav_pct",
                   "mean_freq", "n_batched"]].round(3).to_string(index=False))
    print(f"\nKnee identification: P={knee.penalty} "
          f"(wait={knee.wait_d:.3f}d, saving={knee.init_sav_pct:.2f}%)")
    pareto.to_csv(OUT / "_pareto_path2.csv", index=False)

    # ── Plot ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(pareto.wait_d, pareto.init_sav_pct, "o-", color="#2a9d8f",
            linewidth=2, markersize=9, label="Pfad-2 init (gebündelt + Penalty)")
    for _, r in pareto.iterrows():
        ax.annotate(f"$P={r.penalty:g}$",
                    (r.wait_d, r.init_sav_pct),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=8, color="#1d3557")
    ax.scatter([knee.wait_d], [knee.init_sav_pct], marker="*",
               s=480, color="gold", edgecolor="black", linewidth=1.2,
               zorder=10, label=f"Sweet-Spot $P={knee.penalty:g}$")
    ax.set_xlabel("Durchschnittliche Wartezeit [Tage] (Paket-gewichtet)")
    ax.set_ylabel("Cost Saving vs daily baseline [%]")
    ax.set_title(r"Pareto-Front bei $\theta = 1$ — Pfad-2-Optimierung",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_xlim(left=-0.02)
    ax.set_ylim(bottom=-0.5)
    fig.tight_layout()
    fig.savefig(OUT / "_fig_pareto_path2.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {OUT/'_fig_pareto_path2.png'}")


if __name__ == "__main__":
    main()
