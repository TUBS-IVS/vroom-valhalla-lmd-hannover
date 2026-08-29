"""Per-provider Pareto frontier at theta=1.0 from Path-2 data. Each provider
has its own daily baseline and its own cost-vs-wait curve — providers with
lower demand density consolidate more aggressively, providers with high demand
density (DHL) can barely batch.

Default outputs:
  results/paper_final_2026_05_30/05_optimization/fig_PF6_provider_pareto.{png,pdf}
  results/paper_final_2026_05_30/05_optimization/_per_provider_pareto.csv

Task 19 W1b (v6 regeneration)
-----------------------------
v6 status B: every column read (``init_cost_eur``, ``avg_wait_d_init``,
``schedule_size_init``, ``weekly_parcels``) is the routing-optimal (stage 1)
plan by this script's OWN design (it never reads a ``_balanced`` column),
so per the brief's plan convention it stays on stage 1, not the default
operator plan. Each provider's own theta=0 baseline is read fresh from the
SAME (legacy-adapted) ``tab_balancing_summary.csv`` -- no separate baseline
constant to go stale. Note: the fixed "$P=0.5$" marker below is this
script's own pre-existing illustrative reference point, not a per-provider
knee -- the actual per-LSP routing-lens P* varies (0.25/0.5/0.75) and is
audited in ``tab_pstar_knees_smoothed.csv``; see ``_STATUS.md``.
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

SCRIPT = "_fig_per_provider_pareto_path2.py"
DEFAULT_REV = ROOT / "results" / "overnight_2026_05_29_path2"
DEFAULT_OUT = ROOT / "results" / "paper_final_2026_05_30" / "05_optimization"

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    V.add_v6_args(ap, default_rev=DEFAULT_REV, default_out=DEFAULT_OUT,
                 rev_help="legacy-adapted run/ directory (74_ <out>/run) "
                          "for v6, or the original path2 dir")
    args = ap.parse_args(argv)
    rev = Path(args.rev_dir)
    OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    summ_path, chosen_path = rev / "tab_balancing_summary.csv", rev / "tab_chosen_schedules.csv"
    summ = pd.read_csv(summ_path)
    chosen = pd.read_csv(chosen_path)
    V.require_columns(summ, ["penalty", "share_willing", "provider",
                             "init_cost_eur", "balanced_cost_eur"],
                      source=str(summ_path))
    V.require_columns(chosen, ["penalty", "share_willing", "provider",
                               "avg_wait_d_init", "weekly_parcels",
                               "schedule_size_init"], source=str(chosen_path))

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
    # NOTE (Task 19 W1b): P=0.5 is a fixed illustrative reference point, not
    # each provider's own knee -- the routing-lens P* actually varies by
    # provider (0.25/0.5/0.75; tab_pstar_knees_smoothed.csv). Worded as
    # "reference point", not "sweet-spot", so the title cannot be read as
    # re-asserting a uniform per-provider optimum.
    ax.set_title("Per-provider Pareto frontiers — reference point P = 0.5 "
                "marked", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_xlim(-0.02)
    ax.set_ylim(-1)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    V.footer(fig, plan=V.PLAN1, script=SCRIPT,
             source="tab_balancing_summary.csv + tab_chosen_schedules.csv")
    written = V.savefig_pinned(fig, OUT, "fig_PF6_provider_pareto")
    plt.close(fig)

    # ─── Summary table ──────────────────────────────────────────────────
    sweet_per_prov = df[np.isclose(df.penalty, 0.5)][[
        "provider", "wait_d", "sav_pct", "mean_freq", "baseline_k"]]
    max_sav_per_prov = df.loc[df.groupby("provider").sav_pct.idxmax(),
                                ["provider", "penalty", "wait_d", "sav_pct"]]
    print("\n=== Per-provider analysis at reference point P=0.5, theta=1 ===")
    print(sweet_per_prov.round(2).to_string(index=False))
    print("\n=== Maximum saving (P=0) per provider ===")
    print(max_sav_per_prov.round(2).to_string(index=False))
    print(f"\nsaved {written[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
