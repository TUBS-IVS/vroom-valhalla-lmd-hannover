"""6-panel structural-results figure, Stage-3 (system-smoothed) revision of
scripts/figures/fig_structural_grid.py.

Data-source substitutions relative to the frozen submission figure:
  - Per-provider costs: `balanced_cost_eur` (tab_balancing_summary.csv,
    Stage 2) -> `total_stage3_eur` (tab_costs_smoothed.csv, Stage 3).
  - Per-cell delivery-day cost: `dd_cost_balanced` -> `dd_cost_system_smoothed`
    (RUN_DIR/_tab_chosen_with_system_smoothing.csv).
  - Hub express allocated to PLZ proportional to weekly parcels: sourced
    from tab_express_smoothed.csv (summed to provider level), mirroring
    the original's `cell_express` allocation exactly.
  - Wait: `avg_wait_d_balanced` -> `avg_wait_d_system_smoothed`.
  - Per-LSP knee (operator-optimal P*): IMPORTED from Task 8's
    tab_pstar_knees_smoothed.csv -- NOT re-derived. The gate already
    confirmed all seven P* are unchanged from the submission
    (0.25/0.25/0.5/0.5/0.5/0.75/0.75).
  - Structural features (hub_dist_km, area_km2, parcels_per_stop) and
    raumtyp_3 are penalty/theta-independent PLZ facts; read directly from
    the 04_optim_prep.pkl checkpoint + data/geodata/plz_raumtyp.csv,
    exactly as scripts/figures/fig_plz_structural_correlation.py does (the
    actual generator of the structural-feature schema; the task brief's
    pointer to paper_final_sweetspot.py does not match -- that script has
    no PLZ-level join logic. Flagging per CLAUDE.md's discrepancy rule).

Output: results/revision_2026_07/figures/fig6_structural_grid_6_smoothed.{png,pdf}

Input/output root: ``C.OUT_DIR``, overridable with the ``REV_DIR``
environment variable (default ``results/revision_2026_07`` -- this script
reproduces the submitted revision figure when run with no environment set).
``scripts/revision/70_figs_tables_v2.py`` sets ``REV_DIR`` to the v5-schema
grid.  NOTE: this builder reads the 2026-07 STAGE-3 schema
(``tab_costs_smoothed.csv`` etc.); pointing ``REV_DIR`` at a v5-schema grid
gives it no inputs -- ``70_`` renders the v5 figures itself.

FROZEN (2026-08 revision): builds the accepted paper Fig. 6; see the banner
below. Its plotting code must not change.
"""
from __future__ import annotations
import sys

# --- FROZEN ACCEPTED-FIGURE BUILDER (2026-08 revision) ----------------------
import warnings as _frozen_notice

_frozen_notice.warn(
    "31_fig6_structural_smoothed.py is FROZEN: it builds the accepted paper "
    "Fig. 6 and its plotting code must not change. Do NOT run it against "
    "the 2026-07 tables and quote the result -- that schema has no pool "
    "term and predates the universal tour rule and the two cost lenses. "
    "Run it only through scripts/revision/74_v2_to_legacy_tables.py "
    "--render, which adapts a v5/v6 grid to the schema this file reads and "
    "sets REV_DIR / REV_RUN_DIR / REV_BASE_TOTAL / REV_BASELINE_CV "
    "accordingly. The two-lens designs live in 70_figs_tables_v2.py as "
    "supp_*.",
    UserWarning,
    stacklevel=2,
)
# ---------------------------------------------------------------------------

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import rcParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _stage3_common as C  # noqa: E402

PATH2 = C.RUN_DIR
OUT = C.OUT_DIR / "figures"
OUT.mkdir(parents=True, exist_ok=True)

rcParams.update({
    "font.family": "serif", "font.size": 13,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 13, "axes.titlesize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
})

# Matplotlib stamps a /CreationDate into every PDF and a timestamp into
# every PNG, so two renders of identical content get different md5s. This
# builder is invoked by 74_v2_to_legacy_tables.py's render() (in turn
# called from 70_figs_tables_v2.py), whose manifest and gate G7
# (71_sync_paper_figs.py) are md5-based, so both are suppressed here
# exactly as 70_figs_tables_v2.py suppresses them: a re-render of
# unchanged inputs must be byte-identical, not merely content-identical.
_PDF_META = {"CreationDate": None}
_PNG_META = {"Software": None}

RAUMTYP_PAL = {
    "urban":    "#1d3557",
    "suburban": "#2a9d8f",
    "rural":    "#e76f51",
}
RAUMTYP_ORDER = ["rural", "suburban", "urban"]


def load_pstar_knees() -> tuple[dict, dict]:
    """Import (never re-derive) the Task 8 per-LSP knee P* on Stage-3 costs."""
    df = pd.read_csv(C.OUT_DIR / "tab_pstar_knees_smoothed.csv")
    optimal = dict(zip(df.provider, df.P_star))
    details = {row.provider: dict(P=row.P_star, saving=row.saving_pct,
                                   wait=row.wait_d, chord_dist=row.chord_dist)
               for row in df.itertuples()}
    return optimal, details


def build_structural_features() -> pd.DataFrame:
    """Penalty/theta-independent PLZ structural facts, ported verbatim from
    scripts/figures/fig_plz_structural_correlation.py (weekly_parcels,
    n_stops_per_day, parcels_per_stop, hub_dist_km, area_km2)."""
    _, optim_data = C.load_checkpoints()
    rows = []
    for prov in C.PROVIDERS:
        od = optim_data.get(prov)
        if od is None:
            continue
        for plz, pdmeta in od["plz_data"].items():
            n_stops_per_day = float(pdmeta["n_stops_per_day"])
            weekly_parcels = float(sum(pdmeta["b2c"].values())
                                    + sum(pdmeta["b2b"].values()))
            stops_per_week = n_stops_per_day * C.N_DAYS
            pps = weekly_parcels / stops_per_week if stops_per_week > 0 else np.nan
            rows.append(dict(provider=prov, plz=str(plz).zfill(5),
                              hub_dist_km=float(pdmeta["hub_dist_km"]),
                              area_km2=float(pdmeta["area_km2"]),
                              n_stops_per_day=n_stops_per_day,
                              parcels_per_stop=pps))
    return pd.DataFrame(rows)


def build_sub_feat(optimal_P: dict) -> pd.DataFrame:
    """Per-(provider, plz, penalty=P*_provider, share_willing) Stage-3
    saving, mirroring fig_structural_grid.py's `sub`/`sub_feat` construction
    exactly, with Stage-3 cost/express/wait sources substituted in."""
    sched = pd.read_csv(PATH2 / "tab_chosen_schedules.csv")
    sched["plz"] = sched.plz.astype(str).str.zfill(5)
    smoothed = pd.read_csv(PATH2 / "_tab_chosen_with_system_smoothing.csv")
    smoothed["plz"] = smoothed.plz.astype(str).str.zfill(5)
    costs = pd.read_csv(C.OUT_DIR / "tab_costs_smoothed.csv")
    expr = pd.read_csv(C.OUT_DIR / "tab_express_smoothed.csv")

    # Per-cell baseline = dd_cost_init at (P=0, theta=0)
    base = (sched[(np.isclose(sched.penalty, 0.0))
                   & (np.isclose(sched.share_willing, 0.0))]
              [["provider", "plz", "dd_cost_init"]]
              .rename(columns={"dd_cost_init": "baseline_cost"}))

    # weekly_parcels + dd_cost_system_smoothed per (penalty, share, provider, plz)
    smoothed_m = smoothed.merge(
        sched[["penalty", "share_willing", "provider", "plz", "weekly_parcels"]],
        on=["penalty", "share_willing", "provider", "plz"], how="left")
    assert smoothed_m.weekly_parcels.isna().sum() == 0, (
        "unmatched rows joining weekly_parcels onto the system-smoothed table")

    # LSP-specific selection: each LSP keeps only its own knee P*, all theta.
    sub_parts = []
    for lsp, P_opt in optimal_P.items():
        sub_p = smoothed_m[(smoothed_m.provider == lsp)
                            & (np.isclose(smoothed_m.penalty, P_opt))]
        sub_parts.append(sub_p)
    sub = pd.concat(sub_parts, ignore_index=True).merge(
        base, on=["provider", "plz"], how="left").copy()

    prov_dd = (sub.groupby(["penalty", "share_willing", "provider"])
                   .agg(sum_dd=("dd_cost_system_smoothed", "sum"),
                        sum_par=("weekly_parcels", "sum"))
                   .reset_index())

    # Hub express (tab_express_smoothed.csv) summed to provider level, then
    # restricted to each LSP's own knee P* (all theta) -- same filter pattern
    # the original applies to `bal`.
    expr_prov = (expr.groupby(["penalty", "share_willing", "provider"],
                               as_index=False)
                      .agg(express_cost=("express_cost_stage3_eur", "sum")))
    expr_parts = []
    for lsp, P_opt in optimal_P.items():
        expr_p = expr_prov[(expr_prov.provider == lsp)
                            & (np.isclose(expr_prov.penalty, P_opt))]
        expr_parts.append(expr_p)
    prov_expr = pd.concat(expr_parts, ignore_index=True)

    prov = prov_dd.merge(prov_expr,
                          on=["penalty", "share_willing", "provider"],
                          how="left")
    sub = sub.merge(prov[["penalty", "share_willing", "provider",
                            "express_cost", "sum_par"]],
                     on=["penalty", "share_willing", "provider"],
                     how="left")
    sub["cell_express"] = (sub.weekly_parcels / sub.sum_par.clip(lower=1)
                            * sub.express_cost)
    sub["cell_full_cost"] = sub.dd_cost_system_smoothed + sub.cell_express
    sub["saving_pct"] = 100 * (sub.baseline_cost - sub.cell_full_cost
                                 ) / sub.baseline_cost.clip(lower=1)

    struct = build_structural_features()
    sub_feat = sub.merge(struct, on=["provider", "plz"], how="left")
    rt = pd.read_csv(C.ROOT / "data/geodata/plz_raumtyp.csv",
                      dtype={"plz": str})[["plz", "raumtyp_3"]]
    sub_feat = sub_feat.merge(rt, on="plz", how="left")
    return sub_feat


def main():
    optimal_P, opt_details = load_pstar_knees()
    print("\nLSP-specific optimal P (imported from Task 8, Stage-3 chord-distance "
          "knee, theta=1):")
    for lsp, det in opt_details.items():
        print(f"  {lsp:>7}: P* = {det['P']:.2f}, "
              f"sav = {det['saving']:5.1f}%, "
              f"wait = {det['wait']:.2f} d, "
              f"chord = {det['chord_dist']:.3f}")

    sub_feat = build_sub_feat(optimal_P)

    # ---- Panel (a): system Pareto frontier, Stage 3 ----
    costs = pd.read_csv(C.OUT_DIR / "tab_costs_smoothed.csv")
    waits = pd.read_csv(C.OUT_DIR / "tab_wait_smoothed.csv")
    ag_cost = (costs.groupby(["penalty", "share_willing"], as_index=False)
                     .agg(bal_cost=("total_stage3_eur", "sum")))
    ag_cost["sav_pct"] = 100 * (C.BASE_TOTAL - ag_cost.bal_cost) / C.BASE_TOTAL
    pf = ag_cost.merge(waits.rename(columns={"avg_wait_d_stage3": "avg_wait"}),
                        on=["penalty", "share_willing"], how="left")
    pf = pf[~np.isclose(pf.penalty, 0.4)]

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.0))

    ax = axes[0, 0]
    theta_levels = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    cmap_th = plt.cm.viridis(np.linspace(0.15, 0.9, len(theta_levels)))
    P_unique = np.array(sorted(pf.penalty.unique()))
    P_idx_map = {P: i for i, P in enumerate(P_unique)}
    cmap_P_arr = plt.cm.plasma(np.linspace(0.05, 0.95, len(P_unique)))
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

    # ---- Panel (b): saving vs theta, by carrier-type class at each LSP's own P* ----
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
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xlabel(r"Willingness-to-wait share $\theta$ [%]")
    ax.set_ylabel("Cost saving per postal-code area [%]")
    ax.set_title(rf"(b) Saving by carrier type at routing-lens $P^\star$"
                  + "\n(line = median, band = 25-75th percentile)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, title="Carrier type",
                title_fontsize=10)

    # ---- Panel (c): saving vs theta, lines per raumtyp at each LSP's own P* ----
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
    ax.axhline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xlabel(r"Willingness-to-wait share $\theta$ [%]")
    ax.set_ylabel("Cost saving per postal-code area [%]")
    ax.set_title(rf"(c) Saving by region type at routing-lens $P^\star$"
                  + "\n(line = median, band = 25-75th percentile)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, title="Region type",
                title_fontsize=10)

    # ---- Panels (d)-(f): lines vs theta, bucketed by structural feature quartile ----
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
        ax_.set_title(rf"{title} at routing-lens $P^\star$"
                  + "\n(line = median, band = 25-75th percentile)")
        ax_.grid(alpha=0.3)
        legend_loc = ("lower left" if feat == "parcels_per_stop"
                       else "lower right")
        ax_.legend(loc=legend_loc, fontsize=10,
                    title=f"{feat_label} [{unit}]",
                    title_fontsize=10)

    fig.tight_layout(pad=0.8, w_pad=1.6, h_pad=1.6)

    for ext, meta in (("png", _PNG_META), ("pdf", _PDF_META)):
        fig.savefig(OUT / f"fig6_structural_grid_6_smoothed.{ext}",
                     bbox_inches="tight", metadata=meta)
    plt.close(fig)
    print(f"saved {OUT/'fig6_structural_grid_6_smoothed.png'}")


if __name__ == "__main__":
    main()
