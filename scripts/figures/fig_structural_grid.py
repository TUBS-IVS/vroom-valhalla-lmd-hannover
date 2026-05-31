"""6-panel structural-results figure for the EWGT paper.

Top row:
 (a) iso-saving contour over (theta, hub-distance) at $P = 0.25$;
 (b) median per-cell saving by urban / suburban / rural region type at
     $P = 0.25$ as a function of willingness-to-wait share;
 (c) per-cell knee saving vs hub distance with binned median.
Bottom row:
 (d) per-cell knee saving vs postal-code area size;
 (e) per-cell knee saving vs parcels per drop-site;
 (f) system Pareto frontier (cost saving vs additional customer wait)
     over the full penalty sweep at $\\theta = 1$, with the
     price-sensitive operating point $P = 0.25$ highlighted.

Output: results/EWGT_Results/fig_structural_grid_6.{png,pdf}
"""
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy.stats import spearmanr
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[1]
PATH2 = ROOT / "results" / "overnight_2026_05_29_path2"
OUT = ROOT / "results" / "EWGT_Results"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 13,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 13, "axes.titlesize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

RAUMTYP_PAL = {
    "urban":    "#1d3557",
    "suburban": "#2a9d8f",
    "rural":    "#e76f51",
}
RAUMTYP_ORDER = ["rural", "suburban", "urban"]
P_FOCUS = 0.25


def lsp_optimal_P(bal, sched_full, theta_target=1.0):
    """Per-LSP chord-distance knee on the (saving, wait) Pareto at
    theta_target. Picks the operating point that maximises the orthogonal
    distance to the chord connecting the cost-minimum (high saving, high
    wait) and the wait-minimum (low saving, low wait) extremes -- i.e. the
    point that still keeps most of the saving while the wait has already
    been substantially reduced."""
    base_lsp = (bal[(np.isclose(bal.penalty, 0))
                     & (np.isclose(bal.share_willing, 0))]
                  [["provider", "balanced_cost_eur"]]
                  .rename(columns={"balanced_cost_eur": "base_cost"}))
    sub_cost = bal[np.isclose(bal.share_willing, theta_target)].merge(
        base_lsp, on="provider", how="left")
    sub_cost["saving_pct"] = (100 * (sub_cost.base_cost
                                      - sub_cost.balanced_cost_eur)
                                / sub_cost.base_cost.clip(lower=1))
    s = sched_full[np.isclose(sched_full.share_willing,
                                theta_target)].copy()
    s["wait_x_par"] = s.avg_wait_d_balanced * s.weekly_parcels
    wait_lsp = (s.groupby(["penalty", "provider"])
                   .apply(lambda g: g.wait_x_par.sum()
                                     / g.weekly_parcels.sum())
                   .reset_index(name="avg_wait"))
    m = sub_cost.merge(wait_lsp, on=["penalty", "provider"], how="left")
    m = m[~np.isclose(m.penalty, 0.4)]

    optimal = {}
    details = {}
    for lsp in sorted(m.provider.unique()):
        ms = (m[m.provider == lsp].sort_values("penalty")
                                     .reset_index(drop=True))
        if len(ms) < 3:
            continue
        sav = ms.saving_pct.values.astype(float)
        wait = ms.avg_wait.values.astype(float)
        # Normalise inside LSP range
        s_n = (sav - sav.min()) / (sav.max() - sav.min() + 1e-12)
        w_n = (wait - wait.min()) / (wait.max() - wait.min() + 1e-12)
        # Chord runs from (s_n=1, w_n=1) cost-extreme to (s_n=0, w_n=0)
        # service-extreme; signed distance = (s_n - w_n) / sqrt(2).
        d = (s_n - w_n) / np.sqrt(2)
        best_i = int(np.argmax(d))
        optimal[lsp] = float(ms.iloc[best_i].penalty)
        details[lsp] = dict(P=float(ms.iloc[best_i].penalty),
                             saving=float(sav[best_i]),
                             wait=float(wait[best_i]),
                             chord_dist=float(d[best_i]))
    return optimal, details


def main():
    sched = pd.read_csv(PATH2 / "tab_chosen_schedules.csv")
    sched["plz"] = sched.plz.astype(str).str.zfill(5)
    bal = pd.read_csv(PATH2 / "tab_balancing_summary.csv")
    sched_full = pd.read_csv(PATH2 / "tab_chosen_schedules.csv")

    # ---- compute LSP-specific optimal P (chord-distance knee) ----
    optimal_P, opt_details = lsp_optimal_P(bal, sched_full)
    print("\nLSP-specific optimal P (chord-distance knee, at theta=1):")
    for lsp, det in opt_details.items():
        print(f"  {lsp:>7}: P* = {det['P']:.2f}, "
              f"sav = {det['saving']:5.1f}%, "
              f"wait = {det['wait']:.2f} d, "
              f"chord = {det['chord_dist']:.3f}")

    # Per-cell baseline = dd_cost at (P=0, theta=0)
    base = (sched[(np.isclose(sched.penalty, 0.0))
                   & (np.isclose(sched.share_willing, 0.0))]
              [["provider", "plz", "dd_cost_init"]]
              .rename(columns={"dd_cost_init": "baseline_cost"}))

    # Build the LSP-specific selection: each LSP keeps only its
    # rows at its own optimal P; that defines the operating point used
    # for sub_feat throughout panels (b)-(f).
    sub_parts = []
    for lsp, P_opt in optimal_P.items():
        sub_p = sched[(sched.provider == lsp)
                       & (np.isclose(sched.penalty, P_opt))]
        sub_parts.append(sub_p)
    sub = pd.concat(sub_parts, ignore_index=True).merge(
        base, on=["provider", "plz"], how="left").copy()

    prov_dd = (sub.groupby(["penalty", "share_willing", "provider"])
                    .agg(sum_dd=("dd_cost_balanced", "sum"),
                         sum_par=("weekly_parcels", "sum"))
                    .reset_index())
    # Match the bal rows by (provider, penalty) using each LSP's optimum
    bal_parts = []
    for lsp, P_opt in optimal_P.items():
        bal_p = bal[(bal.provider == lsp)
                     & (np.isclose(bal.penalty, P_opt))]
        bal_parts.append(bal_p)
    prov_total = pd.concat(bal_parts, ignore_index=True)[
        ["penalty", "share_willing", "provider", "balanced_cost_eur"]]
    prov = prov_dd.merge(prov_total,
                          on=["penalty", "share_willing", "provider"],
                          how="left")
    prov["express_cost"] = (prov.balanced_cost_eur - prov.sum_dd
                              ).clip(lower=0)
    sub = sub.merge(prov[["penalty", "share_willing", "provider",
                            "express_cost", "sum_par"]],
                     on=["penalty", "share_willing", "provider"],
                     how="left")
    sub["cell_express"] = (sub.weekly_parcels / sub.sum_par.clip(lower=1)
                            * sub.express_cost)
    sub["cell_full_cost"] = sub.dd_cost_balanced + sub.cell_express
    sub["saving_pct"] = 100 * (sub.baseline_cost - sub.cell_full_cost
                                 ) / sub.baseline_cost.clip(lower=1)

    knee = pd.read_csv(OUT / "tab_PLZ_knee_with_features.csv",
                        dtype={"plz": str})
    if "raumtyp_3" not in knee.columns:
        rt = pd.read_csv(ROOT / "data/geodata/plz_raumtyp.csv",
                          dtype={"plz": str})[["plz", "raumtyp_3"]]
        knee = knee.merge(rt, on="plz", how="left")
    rt = pd.read_csv(ROOT / "data/geodata/plz_raumtyp.csv",
                      dtype={"plz": str})[["plz", "raumtyp_3"]]
    sub_feat = sub.merge(
        knee[["provider", "plz", "hub_dist_km", "area_km2",
              "parcels_per_stop"]],
        on=["provider", "plz"], how="left")
    sub_feat = sub_feat.merge(rt, on="plz", how="left")

    # ---------- figure ----------
    # New layout:
    # Top row: (a) system Pareto, (b) iso-saving, (c) raumtyp curves
    # Bottom row: (d) hub_dist, (e) area, (f) parcels-per-stop scatter
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.0))

    # ---- (a) system Pareto curves across multiple theta levels ----
    ax = axes[0, 0]
    # Aggregate per (P, theta): system cost saving + parcels-weighted wait
    ag_cost = (bal.groupby(["penalty", "share_willing"], as_index=False)
                   .agg(bal_cost=("balanced_cost_eur", "sum")))
    ag_cost["sav_pct"] = 100 * (1909747.75 - ag_cost.bal_cost) / 1909747.75
    sched_full = pd.read_csv(PATH2 / "tab_chosen_schedules.csv")
    sched_full["wait_x_par"] = (sched_full.avg_wait_d_balanced
                                  * sched_full.weekly_parcels)
    ag_wait = (sched_full.groupby(["penalty", "share_willing"],
                                     as_index=False)
                          .agg(num=("wait_x_par", "sum"),
                               den=("weekly_parcels", "sum")))
    ag_wait["avg_wait"] = ag_wait.num / ag_wait.den
    pf = ag_cost.merge(ag_wait[["penalty", "share_willing", "avg_wait"]],
                        on=["penalty", "share_willing"])
    pf = pf[~np.isclose(pf.penalty, 0.4)]
    # Lines coloured by willingness-to-wait (viridis, continuous)
    theta_levels = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    cmap_th = plt.cm.viridis(np.linspace(0.15, 0.9, len(theta_levels)))
    # Markers coloured by service penalty P (sequential, discretised so
    # each P value gets a clearly distinct colour)
    P_unique = np.array(sorted(pf.penalty.unique()))
    P_idx_map = {P: i for i, P in enumerate(P_unique)}
    cmap_P_arr = plt.cm.plasma(
        np.linspace(0.05, 0.95, len(P_unique)))
    bnds = np.arange(len(P_unique) + 1) - 0.5
    bnorm_P = mcolors.BoundaryNorm(bnds, ncolors=len(P_unique))
    cmap_P = mcolors.ListedColormap(cmap_P_arr)
    sm_P = plt.cm.ScalarMappable(cmap=cmap_P, norm=bnorm_P)
    sm_P.set_array([])
    for it, th in enumerate(theta_levels):
        sub_pf = pf[np.isclose(pf.share_willing, th)].sort_values(
            "penalty").reset_index(drop=True)
        ax.plot(sub_pf.avg_wait, sub_pf.sav_pct, "-",
                 color=cmap_th[it], linewidth=2.0,
                 label=rf"$\theta = {th:g}$", zorder=2)
        for _, row in sub_pf.iterrows():
            mc = cmap_P_arr[P_idx_map[row.penalty]]
            ax.scatter(row.avg_wait, row.sav_pct, s=40, color=mc,
                        edgecolor="black", linewidth=0.4, zorder=3)
    cb_p = fig.colorbar(sm_P, ax=ax, fraction=0.046, pad=0.03,
                          ticks=np.arange(len(P_unique)))
    cb_p.ax.set_yticklabels([f"{p:g}" for p in P_unique], fontsize=9)
    cb_p.set_label(r"Service penalty $P$ [€/p/d]", fontsize=10)
    ax.set_xlabel("Mean additional customer wait [d]")
    ax.set_ylabel(r"Weekly cost saving vs baseline [%]")
    ax.set_title(r"(a) System Pareto frontier"
                  + "\n(curves for $\\theta \\in [0.1, 1]$)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10, ncol=2,
                title="Willingness-to-wait share", title_fontsize=10)

    # ---- (b) saving vs theta, by carrier-type class at P=0.25 ----
    ax = axes[0, 1]
    CARRIER_CLASS = {
        "GLS": "Cost-aggressive", "DPD": "Cost-aggressive",
        "Hermes": "Hybrid", "FedEx": "Hybrid", "UPS": "Hybrid",
        "Amazon": "Service-bound", "DHL": "Service-bound",
    }
    CLASS_ORDER = ["Cost-aggressive", "Hybrid", "Service-bound"]
    CLASS_PAL = {
        "Cost-aggressive": "#2a9d8f",
        "Hybrid":          "#e9c46a",
        "Service-bound":   "#d62828",
    }
    df_bb = sub_feat.copy()
    df_bb["carrier_class"] = df_bb.provider.map(CARRIER_CLASS)
    for cls in CLASS_ORDER:
        sub_c = df_bb[df_bb.carrier_class == cls]
        if len(sub_c) == 0:
            continue
        g = (sub_c.groupby("share_willing").saving_pct
                    .agg(med="median",
                         lo=lambda s: s.quantile(0.25),
                         hi=lambda s: s.quantile(0.75))
                    .reset_index())
        ax.fill_between(g.share_willing * 100, g.lo, g.hi,
                         color=CLASS_PAL[cls], alpha=0.18, linewidth=0)
        ax.plot(g.share_willing * 100, g.med, "o-",
                 color=CLASS_PAL[cls], linewidth=2.0, markersize=6,
                 label=cls)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7,
                alpha=0.5)
    ax.set_xlabel(r"Willingness-to-wait share $\theta$ [%]")
    ax.set_ylabel("Cost saving per postal-code area [%]")
    ax.set_title(rf"(b) Saving by carrier type at operator-optimal $P^\star$"
                  + "\n(line = median, band = 25-75th percentile)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, title="Carrier type",
                title_fontsize=10)

    # ---- (c) saving vs theta, lines per raumtyp at P=0.25 ----
    ax = axes[0, 2]
    df_c = sub_feat.dropna(subset=["raumtyp_3"]).copy()
    for rt_name in RAUMTYP_ORDER:
        sub_rt = df_c[df_c.raumtyp_3 == rt_name]
        if len(sub_rt) == 0:
            continue
        g = (sub_rt.groupby("share_willing").saving_pct
                    .agg(med="median",
                         lo=lambda s: s.quantile(0.25),
                         hi=lambda s: s.quantile(0.75))
                    .reset_index())
        ax.fill_between(g.share_willing * 100, g.lo, g.hi,
                         color=RAUMTYP_PAL[rt_name], alpha=0.18,
                         linewidth=0)
        ax.plot(g.share_willing * 100, g.med, "o-",
                 color=RAUMTYP_PAL[rt_name], linewidth=2.0,
                 markersize=6, label=rt_name.capitalize())
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7,
                alpha=0.5)
    ax.set_xlabel(r"Willingness-to-wait share $\theta$ [%]")
    ax.set_ylabel("Cost saving per postal-code area [%]")
    ax.set_title(rf"(c) Saving by region type at operator-optimal $P^\star$"
                  + "\n(line = median, band = 25-75th percentile)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, title="Region type",
                title_fontsize=10)

    # ---- (d)-(f) bottom row: lines vs theta, bucketed by structural
    #     feature quartile, in the same style as (b)/(c)
    quartile_specs = [
        (axes[1, 0], "hub_dist_km", "Hub distance",
         "(d) Saving by hub distance", "km"),
        (axes[1, 1], "area_km2", "Postal-code area",
         "(e) Saving by postal-code area size", r"km$^2$"),
        (axes[1, 2], "parcels_per_stop",
         "Parcels per drop-site",
         "(f) Saving by parcels per drop-site",
         "pkts/site/wk"),
    ]
    cmap_q = plt.cm.YlOrBr(np.linspace(0.30, 0.85, 4))
    for ax_, feat, feat_label, title, unit in quartile_specs:
        df_q = sub_feat[sub_feat[feat].notna()].copy()
        qedges = df_q[feat].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
        q_labels = [f"Q1 ($\\leq${qedges[1]:.1f})",
                    f"Q2 ({qedges[1]:.1f}-{qedges[2]:.1f})",
                    f"Q3 ({qedges[2]:.1f}-{qedges[3]:.1f})",
                    f"Q4 (>{qedges[3]:.1f})"]
        df_q["q_bucket"] = pd.cut(df_q[feat], bins=qedges,
                                    labels=q_labels, include_lowest=True)
        for iq, ql in enumerate(q_labels):
            sub_q = df_q[df_q.q_bucket == ql]
            if len(sub_q) == 0:
                continue
            g = (sub_q.groupby("share_willing").saving_pct
                       .agg(med="median",
                            lo=lambda s: s.quantile(0.25),
                            hi=lambda s: s.quantile(0.75))
                       .reset_index())
            ax_.fill_between(g.share_willing * 100, g.lo, g.hi,
                              color=cmap_q[iq], alpha=0.18,
                              linewidth=0)
            ax_.plot(g.share_willing * 100, g.med, "o-",
                      color=cmap_q[iq], linewidth=2.0,
                      markersize=6, label=ql)
        ax_.axhline(0, color="black", linestyle="--",
                     linewidth=0.7, alpha=0.5)
        ax_.set_xlabel(r"Willingness-to-wait share $\theta$ [%]")
        ax_.set_ylabel("Cost saving per postal-code area [%]")
        ax_.set_title(rf"{title} at operator-optimal $P^\star$"
                  + "\n(line = median, band = 25-75th percentile)")
        ax_.grid(alpha=0.3)
        legend_loc = ("lower left" if feat == "parcels_per_stop"
                       else "lower right")
        ax_.legend(loc=legend_loc, fontsize=10,
                    title=f"{feat_label} [{unit}]",
                    title_fontsize=10)

    fig.tight_layout(pad=0.8, w_pad=1.6, h_pad=1.6)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_structural_grid_6.{ext}",
                     bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'fig_structural_grid_6.png'}")


if __name__ == "__main__":
    main()
