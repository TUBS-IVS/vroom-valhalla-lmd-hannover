"""Pareto frontier at theta=1.0 from the Path-2 init schedules. Plots
(cost saving %, parcels-weighted average wait [d]) for each available penalty
P, marks the sweet-spot (Pareto knee), and prints the table.

Default outputs:
  results/overnight_2026_05_29_path2/_fig_pareto_path2.png
  results/overnight_2026_05_29_path2/_pareto_path2.csv

Task 19 W1b (v6 regeneration)
-----------------------------
v6 status B: ``init``-suffixed columns are the routing-optimal (stage 1)
plan -- this script's own title says "Path-2 init", so per the brief's plan
convention it stays on stage 1 (the routing optimum), not the default
operator plan. Source: ``scripts/revision/74_v2_to_legacy_tables.py``'s
``tab_balancing_summary.csv``/``tab_chosen_schedules.csv``.  The hardcoded
``BASE = 1,909,747.75`` EUR is the STALE 2026-07/path2 denominator -- 74_'s
own docstring rules a v6 saving must never be taken against it (the bundle
head re-prices the theta=0 baseline too); replaced with this grid's own
baseline from 74_'s ``legacy_manifest.json``.
"""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
import _v6_provenance as V  # noqa: E402

SCRIPT = "_fig_pareto_path2.py"
DEFAULT_REV = ROOT / "results" / "overnight_2026_05_29_path2"

rcParams.update({
    "font.family": "serif", "font.size": 10,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "savefig.bbox": "tight", "savefig.dpi": 200, "pdf.fonttype": 42,
})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    V.add_v6_args(ap, default_rev=DEFAULT_REV, default_out=DEFAULT_REV,
                 rev_help="legacy-adapted run/ directory (74_ <out>/run) "
                          "for v6, or the original path2 dir")
    args = ap.parse_args(argv)
    rev = Path(args.rev_dir)
    OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    summ_path, chosen_path = rev / "tab_balancing_summary.csv", rev / "tab_chosen_schedules.csv"
    summ = pd.read_csv(summ_path)
    chosen = pd.read_csv(chosen_path)
    V.require_columns(summ, ["penalty", "share_willing", "init_cost_eur"],
                      source=str(summ_path))
    V.require_columns(chosen, ["penalty", "share_willing", "avg_wait_d_init",
                               "weekly_parcels", "schedule_size_init"],
                      source=str(chosen_path))
    base_total = V.base_total_with_path2_fallback(rev)

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
    pareto["init_sav_pct"] = 100 * (base_total - pareto.init_cost_eur) / base_total
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
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    V.footer(fig, plan=V.PLAN1, script=SCRIPT,
             source="tab_balancing_summary.csv + tab_chosen_schedules.csv")
    written = V.savefig_pinned(fig, OUT, "_fig_pareto_path2")
    plt.close(fig)
    print(f"\nsaved {written[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
