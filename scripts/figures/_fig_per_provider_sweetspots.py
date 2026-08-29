"""Per-provider individual sweet-spot identification + LSP-type recommendation
framework.

For each LSP we:
  1. Build the (wait, saving) Pareto curve from P sweep at theta=1
  2. Identify the geometric knee (chord distance method) — that provider's
     individual cost-service trade-off optimum
  3. Categorise the LSP by its structural sensitivity to penalty:
     - "Cost-friendly" (LSPs where low P unlocks large savings — typical of
       low-density operations)
     - "Service-bound" (LSPs where daily delivery is near-optimal even at
       small P — typical of high-density / time-critical premium operators)

Default outputs:
  results/paper_final_2026_05_30/05_optimization/fig_per_provider_sweetspots.{png,pdf}
  results/paper_final_2026_05_30/05_optimization/_per_provider_knees.csv

Task 19 W1b (v6 regeneration)
-----------------------------
v6 status B: the curve (per P, per provider) is read from
``scripts/revision/74_v2_to_legacy_tables.py``'s ``tab_balancing_summary.csv``
/ ``tab_chosen_schedules.csv`` (``_init`` = routing-optimal / stage 1, this
script's only plan). The KNEE ITSELF, though, is switched from this
script's OWN ``chord_knee()`` re-derivation to a direct read of 74_'s
``tab_pstar_knees_smoothed.csv`` -- the audited per-LSP routing-lens P*
table (Kompendium §40.22), computed by the identical chord-distance formula
(``scripts/revision/_figs_tables_v2.py::lsp_knees``), so this figure's
knee marker can never drift from the number the paper actually cites. Only
``mean_freq``/``n_batched`` at the knee still come from this script's own
curve (not carried by the knees table).
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
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
import _v6_provenance as V  # noqa: E402

SCRIPT = "_fig_per_provider_sweetspots.py"
DEFAULT_REV = ROOT / "results" / "overnight_2026_05_29_path2"
DEFAULT_OUT = ROOT / "results" / "paper_final_2026_05_30" / "05_optimization"

rcParams.update({
    "font.family": "serif", "font.size": 9,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
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


def chord_knee(wait: np.ndarray, sav: np.ndarray) -> int:
    """Return index of the geometric knee (chord-distance method)."""
    if len(wait) < 3:
        return 0
    x_n = (wait - wait.min()) / (wait.max() - wait.min() + 1e-9)
    y_n = (sav - sav.min()) / (sav.max() - sav.min() + 1e-9)
    # Distance from the chord (0,0)-(1,1) for each point; sign-flipped to
    # find the knee of a concave saving-vs-wait curve.
    return int(np.argmax(y_n - x_n))


def _load_official_knees(rev: Path) -> pd.DataFrame | None:
    """74_'s audited routing-lens P* table (``<out>/rev/`` sibling of
    ``<out>/run/``), or None when it does not exist (default/path2 mode) --
    the caller then falls back to this script's own ``chord_knee()``."""
    path = rev.parent / "rev" / "tab_pstar_knees_smoothed.csv"
    if not path.exists():
        return None
    knees = pd.read_csv(path)
    V.require_columns(knees, ["provider", "P_star", "saving_pct", "wait_d"],
                      source=str(path))
    return knees


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
    official_knees = _load_official_knees(rev)
    print("knee source: " + ("tab_pstar_knees_smoothed.csv (audited)"
                             if official_knees is not None
                             else "this script's own chord_knee() (fallback"
                                  " -- no audited knees table found)"))

    base = (summ[np.isclose(summ.share_willing, 0.0)]
            .groupby("provider").balanced_cost_eur.mean())

    th = 1.0
    s1 = summ[np.isclose(summ.share_willing, th)]
    c1 = chosen[np.isclose(chosen.share_willing, th)]

    knees = []
    curves = {}
    for prov in sorted(s1.provider.unique()):
        b = float(base[prov])
        d = []
        for P in sorted(s1.penalty.unique()):
            gp = s1[(s1.provider == prov) & np.isclose(s1.penalty, P)]
            cp = c1[(c1.provider == prov) & np.isclose(c1.penalty, P)]
            if len(gp) == 0 or len(cp) == 0:
                continue
            cost = gp.init_cost_eur.iloc[0]
            sav = 100 * (b - cost) / b
            wait = float((cp.avg_wait_d_init * cp.weekly_parcels).sum()
                          / cp.weekly_parcels.sum())
            mean_freq = float(cp.schedule_size_init.mean())
            n_batched = int((cp.schedule_size_init < 6).sum())
            d.append({"penalty": P, "wait_d": wait, "sav_pct": sav,
                       "mean_freq": mean_freq, "n_batched": n_batched})
        cur = pd.DataFrame(d).sort_values("penalty").reset_index(drop=True)
        curves[prov] = cur
        if official_knees is not None:
            row = official_knees[official_knees.provider == prov]
            assert len(row) == 1, (
                f"{prov}: expected exactly one audited knee row, got "
                f"{len(row)}")
            P_star = float(row.P_star.iloc[0])
            match = cur[np.isclose(cur.penalty, P_star)]
            assert len(match), (
                f"{prov}: audited P*={P_star:g} is not one of this curve's "
                f"own penalty levels {sorted(cur.penalty)}")
            ki = int(match.index[0])
        else:
            ki = chord_knee(cur.wait_d.values, cur.sav_pct.values)
        k_row = cur.iloc[ki]
        max_sav = cur.sav_pct.max()
        # Penalty sensitivity slope: average decrease per Δlog(P)
        # negative -> consolidation crashes quickly as P rises -> "service-bound"
        # close to zero -> consolidation persists at higher P -> "cost-friendly"
        cur_low = cur[cur.penalty <= 0.5]
        if len(cur_low) >= 2:
            slope = (cur_low.sav_pct.diff().dropna()
                     / cur_low.penalty.diff().dropna()).mean()
        else:
            slope = 0.0
        knees.append({"provider": prov,
                       "baseline_k_per_wk": b / 1000,
                       "max_saving_pct": max_sav,
                       "knee_P": k_row.penalty,
                       "knee_wait_d": k_row.wait_d,
                       "knee_saving_pct": k_row.sav_pct,
                       "knee_mean_freq": k_row.mean_freq,
                       "knee_n_batched": k_row.n_batched,
                       "slope_per_P": slope})
    kdf = pd.DataFrame(knees).sort_values("max_saving_pct", ascending=False)
    kdf.to_csv(OUT / "_per_provider_knees.csv", index=False)

    # ─── Classification: Cost-friendly vs Service-bound ─────────────────
    # Threshold: max saving 15 %% — providers above are "cost-friendly",
    # below are "service-bound".
    kdf["category"] = np.where(kdf.max_saving_pct >= 15.0,
                                 "Cost-friendly", "Service-bound")
    kdf["recommendation_P"] = np.where(
        kdf.category == "Cost-friendly", 0.25, 0.5)

    # ─── Plot: 2-panel — Pareto per provider + knee summary ─────────────
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1], wspace=0.22)
    ax_par = fig.add_subplot(gs[0, 0])
    ax_kn  = fig.add_subplot(gs[0, 1])

    # Pareto fronts
    for prov in kdf.provider:
        cur = curves[prov]
        ax_par.plot(cur.wait_d, cur.sav_pct, "o-",
                     color=PROV_COLORS[prov], linewidth=2, markersize=6,
                     label=prov)
        # Mark knee
        krow = kdf[kdf.provider == prov].iloc[0]
        ax_par.scatter([krow.knee_wait_d], [krow.knee_saving_pct],
                        marker="*", s=370, color=PROV_COLORS[prov],
                        edgecolor="black", linewidth=1.1, zorder=10)
        ax_par.annotate(rf"$P={krow.knee_P:g}$",
                          (krow.knee_wait_d, krow.knee_saving_pct),
                          xytext=(7, 5), textcoords="offset points",
                          fontsize=7.5, color=PROV_COLORS[prov])
    ax_par.set_xlabel("Average customer wait [days]  ($\\theta = 1$)")
    ax_par.set_ylabel("Cost saving vs provider's daily baseline [%]")
    ax_par.set_title("Per-provider Pareto frontiers — geometric knee marked",
                      fontsize=10.5)
    ax_par.grid(alpha=0.3)
    ax_par.legend(loc="lower right", fontsize=8.5, ncol=2)
    ax_par.set_xlim(-0.02)
    ax_par.set_ylim(-1)

    # Knee summary table as horizontal bars
    y = np.arange(len(kdf))
    cf_mask = (kdf.category == "Cost-friendly").values
    bar_cols = [PROV_COLORS[p] for p in kdf.provider]
    ax_kn.barh(y, kdf.knee_saving_pct, color=bar_cols, edgecolor="black",
               linewidth=0.5, alpha=0.85)
    for i, (_, r) in enumerate(kdf.iterrows()):
        cat_tag = "Cost" if r.category == "Cost-friendly" else "Service"
        ax_kn.text(r.knee_saving_pct + 0.7, i,
                    f"{r.knee_saving_pct:.1f}% @ $P={r.knee_P:g}$  "
                    f"({cat_tag})  wait {r.knee_wait_d:.2f}d  "
                    f"freq {r.knee_mean_freq:.1f}d/wk",
                    va="center", fontsize=8)
    ax_kn.set_yticks(y); ax_kn.set_yticklabels(kdf.provider, fontsize=9)
    ax_kn.invert_yaxis()
    ax_kn.set_xlabel("Saving at individual knee  [%]")
    ax_kn.set_xlim(0, kdf.knee_saving_pct.max() * 1.8)
    ax_kn.set_title("Individual sweet-spots", fontsize=10.5)
    ax_kn.grid(axis="x", alpha=0.3)
    ax_kn.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Provider-individuelle Sweet-Spots — gleicher Optimierer, "
                  "verschiedene Kosten-Service-Trade-offs",
                  fontsize=11.5, y=1.02)
    V.footer(fig, plan=V.PLAN1, script=SCRIPT,
             source="tab_balancing_summary.csv + tab_chosen_schedules.csv"
                    + (" + tab_pstar_knees_smoothed.csv (knee)"
                       if official_knees is not None else ""),
             y=-0.06)
    written = V.savefig_pinned(fig, OUT, "fig_per_provider_sweetspots")
    plt.close(fig)

    print("=== Per-provider individual sweet-spots (theta=1) ===")
    print(kdf[["provider", "baseline_k_per_wk", "max_saving_pct",
                "knee_P", "knee_saving_pct", "knee_wait_d",
                "knee_mean_freq", "category", "recommendation_P"]]
          .round(2).to_string(index=False))
    print(f"\nsaved {written[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
