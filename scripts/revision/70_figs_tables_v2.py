"""Figures and tables for the v5-schema (two-plan, two-lens) revision grid.

Driver for the EWGT revision: it points ``REV_DIR`` at a v5-schema grid
(``results/revision_2026_08_v5`` by default, Task 11's head grid via
``--rev-dir``) and renders every paper figure and table from it.

Why this is not a thin wrapper around 30_/31_/32_/40_
-----------------------------------------------------
Those four builders read the 2026-07 **Stage-3** schema
(``tab_costs_smoothed.csv``, ``tab_express_smoothed.csv``,
``_tab_chosen_with_system_smoothing.csv``).  The v5 grid has a different
schema and, more importantly, a different *object*: it carries TWO plans per
(P, theta) instead of one.  Every panel therefore has to declare which plan
and which lens it shows, which is a change of content, not of path.  The
four builders keep their ``REV_DIR`` override so they still reproduce the
submitted revision figures byte-for-byte from the 2026-07 directory.

The two plans
-------------
* **plan 1 -- routing-optimal (stage 1).**  Coordinate descent on routing
  cost + service penalty.  This is the plan behind the submitted paper's
  headline saving.
* **plan 2 -- operator-polished (stage 2).**  Frequency-free OpCost polish,
  best-of-3 starts (Task 6f).  Stage 3 is off in v5.

The two lenses
--------------
* **routing lens** -- the paper's cost: 189.15 EUR/vehicle-day
  + 0.3864 EUR/km + 36 EUR/route-hour (VROOM ``per_hour`` default).
* **operator lens** -- ``variable + 6 x 189.15 x sum_h peak_h``: the driver
  is employed and the van owned for the whole week, so the fixed bill is paid
  once per **hub peak** vehicle.  The service penalty enters the stage-2
  objective but is *not* part of the reported operator cost.

Outputs (all under ``<rev>/``)
------------------------------
``figures/``
  ``fig4_freq_mix_two_plans.{png,pdf}``   frequency mix, both plans
  ``fig4b_mean_days.{png,pdf}``           mean delivery days, both plans
  ``fig5_grid_heatmap_v2.{png,pdf}``      headline 2x3 (see below)
  ``fig5b_offdiagonal_v2.{png,pdf}``      the other half of the 2x2
  ``fig6_structural_v2.{png,pdf}``        Pareto + knees + structure

``tables/``
  ``tab_headline_theta1_v2.csv|.tex``     2x2 headline at theta=1
  ``tab_grid_full_v2.csv``                the same columns for all 88 cells
  ``tab_pstar_knees_v2.csv|.tex``         per-LSP knee P*, both lenses
  ``tab_stage2_ablation_v2.csv|.tex``     range / solo / freqpres / v5
  ``tab_fleet_diagnostics_v2.csv|.tex``   gap to the flat-profile bound

Colours come from ``scripts/presentation/_style.py`` only; provider identity
is deliberately not a hue (Kompendium §38).  The frequency legend is
{2,3,4,5,6} -- a once-weekly pattern has a 6-day gap and is infeasible under
``MAX_HOLDING_DAYS = 3`` (§38.5).  ``tab_sensitivity_master_plz.csv`` is
never read (§38.3).

Usage
-----
    python scripts/revision/70_figs_tables_v2.py                    # v5
    python scripts/revision/70_figs_tables_v2.py --rev-dir results/revision_2026_08_head
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import rcParams  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts" / "presentation"))

import _figs_tables_v2 as H  # noqa: E402
import _style as S  # noqa: E402  -- the ONLY colour source

DEFAULT_REV = ROOT / "results" / "revision_2026_08_v5"
FREQ_CLASSES = (2, 3, 4, 5, 6)

RC = {
    "font.family": "serif", "font.size": 13,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 13, "axes.titlesize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300, "pdf.fonttype": 42,
    "figure.facecolor": "white", "savefig.facecolor": "white",
}


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
        print(f"  saved {(out_dir / f'{name}.{ext}').relative_to(ROOT)}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# heat-map primitive (ported from 30_fig5_heatmap_smoothed.py)
# ─────────────────────────────────────────────────────────────────────────
def heat(ax, mat, cmap, title, vmin=None, vmax=None, fmt="{:.1f}",
         norm=None, cbar_label="[%]", invert_thr=False):
    if norm is None:
        vmin = float(np.nanmin(mat.values)) if vmin is None else vmin
        vmax = float(np.nanmax(mat.values)) if vmax is None else vmax
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap,
                       vmin=vmin, vmax=vmax)
        thr = vmin + (vmax - vmin) * (0.65 if invert_thr else 0.55)
        if invert_thr:
            def color_for(v):
                return "white" if v > thr else "black"
        else:
            def color_for(v):
                return "white" if v < thr else "black"
    else:
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap, norm=norm)
        lo = norm.vmin + (norm.vmax - norm.vmin) * 0.30
        hi = norm.vmin + (norm.vmax - norm.vmin) * 0.70

        def color_for(v):
            return "white" if (v < lo or v > hi) else "black"
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([f"{x * 100:.0f}" for x in mat.columns], fontsize=9.5)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([f"{p:g}" for p in mat.index], fontsize=10)
    ax.set_title(title, fontsize=11.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color=color_for(v), fontsize=8.5)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=10)
    cb.ax.tick_params(labelsize=9)
    return im


def piv(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot(index="penalty", columns="share_willing", values=col)


def _signed_cmap(higher_is_better: bool) -> str:
    """Diverging map oriented so that RED always means "worse".

    ``S.CMAP_CHANGE`` is RdBu_r (red = high).  For a quantity where high is
    bad -- more peak vehicles, more vehicle-days -- that is already right;
    for a saving, where high is good, the map is flipped so a NEGATIVE
    saving reads red.  Without this rule the same colour would mean
    "operator wins" in one panel and "operator loses" in the next.
    """
    return S.CMAP_CHANGE[:-2] if higher_is_better else S.CMAP_CHANGE


def _sym_norm(mat) -> TwoSlopeNorm:
    a = float(np.ceil(max(abs(np.nanmin(mat.values)),
                          abs(np.nanmax(mat.values)))))
    a = max(a, 1.0)
    return TwoSlopeNorm(vmin=-a, vcenter=0.0, vmax=a)


# ─────────────────────────────────────────────────────────────────────────
# Fig. 5 -- headline 2x3, and the off-diagonal companion
# ─────────────────────────────────────────────────────────────────────────
def fig5(full: pd.DataFrame, base: dict, base_cv: float, out: Path) -> None:
    rcParams.update(RC)
    fig, ax = plt.subplots(2, 3, figsize=(19.0, 10.2))

    heat(ax[0, 0], piv(full, "routing_saving_plan1_pct"), S.CMAP_SAVING,
         "(a) Routing-cost saving\n"
         r"routing lens $\cdot$ routing-optimal plan (stage 1)",
         vmin=0, cbar_label="Routing saving [%]")
    heat(ax[0, 1], piv(full, "operator_saving_plan2_pct"), S.CMAP_SAVING,
         "(b) Operator-cost saving\n"
         r"operator lens $\cdot$ operator-polished plan (stage 2)",
         vmin=0, cbar_label="Operator saving [%]")
    heat(ax[0, 2], piv(full, "wait_d_plan2"), S.CMAP_WAIT,
         "(c) Mean additional wait per parcel\n"
         r"operator-polished plan (stage 2)",
         vmin=0, fmt="{:.2f}", cbar_label="Wait [d]", invert_thr=True)

    m = piv(full, "hub_peak_plan2_vs_base_pct")
    heat(ax[1, 0], m, _signed_cmap(higher_is_better=False),
         r"(d) $\sum_h$ hub peak vehicles vs baseline"
         "\n" r"operator-polished plan (stage 2)",
         norm=_sym_norm(m), cbar_label="Change [%]")
    # reversed fleet ramp: light = flat week = the operator's goal, matching
    # panel (c) where light = little wait
    heat(ax[1, 1], piv(full, "sys_cv"), S.fleet_cmap(reverse=True),
         "(e) Mon--Sat CV of the system fleet\n"
         rf"operator-polished plan (stage 2); baseline CV = {base_cv:.3f}",
         vmin=0, fmt="{:.3f}", cbar_label="CV [--]", invert_thr=True)
    m = piv(full, "vehicle_days_plan2_vs_base_pct")
    heat(ax[1, 2], m, _signed_cmap(higher_is_better=False),
         "(f) Weekly vehicle-days vs baseline\n"
         r"operator-polished plan (stage 2)",
         norm=_sym_norm(m), cbar_label="Change [%]")

    for a in ax[-1, :]:
        a.set_xlabel(r"Willingness-to-wait share $\theta$ [%]", fontsize=12)
    for a in ax[:, 0]:
        a.set_ylabel(r"Service penalty $P$ [€/p/d]", fontsize=12)
    fig.tight_layout(pad=0.6, w_pad=1.4, h_pad=1.9, rect=[0, 0.045, 1, 1])
    fig.text(0.5, 0.006,
             "Panels (a) and (b) are different cost lenses AND different "
             "plans and must not be read as one series. All quantities are "
             r"relative to the daily-delivery baseline at $\theta = 0$ "
             f"(routing {base['routing_eur']:,.0f} €/wk, operator "
             f"{base['operator_eur']:,.0f} €/wk, "
             r"$\sum_h$ peak " f"{base['sum_hub_peak']:.0f} vehicles). "
             + H.COST_MODEL_TEXT,
             ha="center", va="bottom", fontsize=10)
    save(fig, out, "fig5_grid_heatmap_v2")


def fig5b(full: pd.DataFrame, base: dict, out: Path) -> None:
    """The off-diagonal cells of the 2 plans x 2 lenses matrix."""
    rcParams.update(RC)
    fig, ax = plt.subplots(2, 2, figsize=(13.0, 10.2))

    m = piv(full, "operator_saving_plan1_pct")
    heat(ax[0, 0], m, _signed_cmap(higher_is_better=True),
         "(a) Operator-cost saving\n"
         r"operator lens $\cdot$ routing-optimal plan (stage 1)",
         norm=_sym_norm(m), cbar_label="Operator saving [%]")
    heat(ax[0, 1], piv(full, "routing_saving_plan2_pct"), S.CMAP_SAVING,
         "(b) Routing-cost saving\n"
         r"routing lens $\cdot$ operator-polished plan (stage 2)",
         vmin=0, cbar_label="Routing saving [%]")
    heat(ax[1, 0], piv(full, "wait_d_plan1"), S.CMAP_WAIT,
         "(c) Mean additional wait per parcel\n"
         r"routing-optimal plan (stage 1)",
         vmin=0, fmt="{:.2f}", cbar_label="Wait [d]", invert_thr=True)
    m = piv(full, "hub_peak_plan1_vs_base_pct")
    heat(ax[1, 1], m, _signed_cmap(higher_is_better=False),
         r"(d) $\sum_h$ hub peak vehicles vs baseline"
         "\n" r"routing-optimal plan (stage 1)",
         norm=_sym_norm(m), cbar_label="Change [%]")

    for a in ax[-1, :]:
        a.set_xlabel(r"Willingness-to-wait share $\theta$ [%]", fontsize=12)
    for a in ax[:, 0]:
        a.set_ylabel(r"Service penalty $P$ [€/p/d]", fontsize=12)
    fig.tight_layout(pad=0.6, w_pad=1.4, h_pad=1.9, rect=[0, 0.05, 1, 1])
    fig.text(0.5, 0.008,
             "Companion to Fig. 5: the two off-diagonal cells of the "
             "2 plans x 2 lenses matrix, plus the routing-optimal plan's "
             "wait and hub peaks. Panel (a) is negative where the "
             "routing-optimal two-day patterns multiply the hub peaks. "
             + H.COST_MODEL_TEXT,
             ha="center", va="bottom", fontsize=10)
    save(fig, out, "fig5b_offdiagonal_v2")


# ─────────────────────────────────────────────────────────────────────────
# Fig. 4 -- delivery-frequency mix, both plans
# ─────────────────────────────────────────────────────────────────────────
def _stack(ax, mix: pd.DataFrame, P: float) -> None:
    sub = mix[np.isclose(mix.penalty, P)]
    pv = (sub.pivot(index="share_willing", columns="size", values="pct")
          .fillna(0.0).sort_index())
    x = pv.index.values * 100
    bottom = np.zeros(len(pv))
    for sz in FREQ_CLASSES:
        if sz not in pv.columns:
            continue
        h = pv[sz].values
        ax.fill_between(x, bottom, bottom + h, color=S.FREQ[sz], alpha=0.92)
        bottom += h
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.2)


def fig4(mix1: pd.DataFrame, mix2: pd.DataFrame, out: Path) -> None:
    rcParams.update(RC)
    rcParams.update({"font.size": 11, "axes.titlesize": 11})
    P_values = sorted(mix1.penalty.unique())
    ncols = len(P_values)
    fig, axes = plt.subplots(2, ncols, figsize=(2.05 * ncols, 5.4),
                             sharex=True, sharey=True)
    for j, P in enumerate(P_values):
        _stack(axes[0, j], mix1, P)
        _stack(axes[1, j], mix2, P)
        axes[0, j].set_title(rf"$P = {P:g}$", fontsize=9.5)
    axes[0, 0].set_ylabel("Routing-optimal\n(stage 1)\n"
                          "Share of areas [%]", fontsize=9)
    axes[1, 0].set_ylabel("Operator-polished\n(stage 2)\n"
                          "Share of areas [%]", fontsize=9)
    for a in axes[1, :]:
        a.set_xlabel(r"$\theta$ [%]", fontsize=9.5)

    handles = [Patch(facecolor=S.FREQ[s], label=f"{s} day/wk")
               for s in FREQ_CLASSES]
    fig.legend(handles=handles, title="Delivery days per week",
               loc="lower center", ncol=len(FREQ_CLASSES), frameon=True,
               framealpha=0.9, edgecolor="0.8", fontsize=9,
               title_fontsize=9, handlelength=1.4, columnspacing=1.3,
               bbox_to_anchor=(0.5, -0.035))
    fig.suptitle("Delivery-frequency mix over the "
                 r"$(P, \theta)$ grid, both plans "
                 r"(service penalty $P$ in €/parcel/day)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0.045, 1, 0.95])
    save(fig, out, "fig4_freq_mix_two_plans")


def fig4b(full: pd.DataFrame, out: Path) -> None:
    """Mean delivery days per week, both plans -- the (0,1) 2.03 vs 2.37."""
    rcParams.update(RC)
    P_values = sorted(full.penalty.unique())
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    cmap = plt.get_cmap(S.CMAP_PENALTY)(np.linspace(0.05, 0.92,
                                                    len(P_values)))
    for k, (a, col, ttl) in enumerate([
            (ax[0], "mean_days_plan1", "(a) Routing-optimal plan (stage 1)"),
            (ax[1], "mean_days_plan2",
             "(b) Operator-polished plan (stage 2)")]):
        for i, P in enumerate(P_values):
            g = full[np.isclose(full.penalty, P)].sort_values("share_willing")
            a.plot(g.share_willing * 100, g[col], "o-", color=cmap[i],
                   markersize=4, linewidth=1.8, label=rf"$P = {P:g}$")
        a.set_title(ttl, fontsize=12)
        a.set_xlabel(r"Willingness-to-wait share $\theta$ [%]")
        a.grid(alpha=0.3)
        a.set_ylim(1.8, 6.15)
    ax[0].set_ylabel("Mean delivery days per week\n"
                     "(area-weighted)")
    ax[1].legend(loc="lower left", fontsize=9, ncol=2,
                 title=r"Service penalty [€/p/d]", title_fontsize=9)
    fig.suptitle("Weekly delivery frequency of the two plans "
                 r"(6 = daily; 1 day/week is infeasible at "
                 r"$\mathrm{MAX\_HOLDING\_DAYS}=3$)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, out, "fig4b_mean_days")


# ─────────────────────────────────────────────────────────────────────────
# Fig. 6 -- Pareto fronts, per-LSP knees in both lenses, structure
# ─────────────────────────────────────────────────────────────────────────
def _structural_facts() -> pd.DataFrame:
    """Penalty/theta-independent PLZ facts from 04_optim_prep.pkl (53 kB).

    Only the small optimisation-prep checkpoint is read -- no demand pickle,
    no cost matrices, no model.
    """
    import pickle
    with open(ROOT / "results" / "checkpoints" / "04_optim_prep.pkl",
              "rb") as fh:
        od = pickle.load(fh)["optimization_data"]
    rows = []
    for prov in H.PROVIDERS:
        d = od.get(prov)
        if d is None:
            continue
        # hub_plz_list holds INDICES into plz_keys, not PLZ codes
        keys = list(d["plz_keys"])
        cells_per_hub = {}
        for idxs in d["hub_plz_list"]:
            for i in np.asarray(idxs).ravel():
                cells_per_hub[str(keys[int(i)])] = int(len(idxs))
        for plz, meta in d["plz_data"].items():
            nspd = float(meta["n_stops_per_day"])
            wk = float(sum(meta["b2c"].values()) + sum(meta["b2b"].values()))
            rows.append(dict(
                provider=prov, plz=str(plz),
                hub_dist_km=float(meta["hub_dist_km"]),
                area_km2=float(meta["area_km2"]),
                weekly_parcels=wk,
                n_stops_per_day=nspd,
                parcels_per_stop=(wk / (nspd * H.N_DAYS)
                                  if nspd > 0 else np.nan),
                hub_n_cells=cells_per_hub.get(str(plz), np.nan)))
    return pd.DataFrame(rows)


def _quartile_lines(ax, df, feat, label, unit, cmap,
                    edges=None, labs=None):
    """Median weekly delivery days vs theta, bucketed by a structural feature.

    With ``edges``/``labs`` omitted the buckets are the feature's own
    quartiles.  Discrete features (hub size) pass explicit class edges,
    because their quartiles collapse onto repeated values.
    """
    d = df[df[feat].notna()].copy()
    if edges is None:
        edges = d[feat].quantile([0, .25, .5, .75, 1.]).values
        labs = [f"Q1 ($\\leq${edges[1]:.1f})",
                f"Q2 ({edges[1]:.1f}--{edges[2]:.1f})",
                f"Q3 ({edges[2]:.1f}--{edges[3]:.1f})",
                f"Q4 (>{edges[3]:.1f})"]
        assert len(set(np.round(edges, 9))) == len(edges), (
            f"{feat}: quartile edges are not unique ({edges}) -- pass "
            "explicit class edges for this discrete feature")
    d["q"] = pd.cut(d[feat], bins=edges, labels=labs, include_lowest=True)
    for i, ql in enumerate(labs):
        g = d[d.q == ql]
        if g.empty:
            continue
        agg = (g.groupby("share_willing")["size"]
               .agg(med="median", lo=lambda s: s.quantile(.25),
                    hi=lambda s: s.quantile(.75)).reset_index())
        ax.fill_between(agg.share_willing * 100, agg.lo, agg.hi,
                        color=cmap[i], alpha=0.18, linewidth=0)
        ax.plot(agg.share_willing * 100, agg.med, "o-", color=cmap[i],
                linewidth=2.0, markersize=5, label=ql)
    ax.set_xlabel(r"Willingness-to-wait share $\theta$ [%]")
    ax.set_ylabel("Delivery days per week\n(median, band = IQR)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, title=f"{label} [{unit}]",
              title_fontsize=9)


def fig6(full: pd.DataFrame, knees_r: pd.DataFrame, knees_o: pd.DataFrame,
         cell_freq2: pd.DataFrame, out: Path) -> None:
    rcParams.update(RC)
    fig, ax = plt.subplots(2, 3, figsize=(18.0, 10.4))

    theta_levels = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    P_unique = np.array(sorted(full.penalty.unique()))
    cmap_th = plt.get_cmap(S.CMAP_THETA)(np.linspace(0.15, 0.9,
                                                     len(theta_levels)))
    cmap_P = plt.get_cmap(S.CMAP_PENALTY)(np.linspace(0.05, 0.95,
                                                      len(P_unique)))
    P_idx = {P: i for i, P in enumerate(P_unique)}

    def pareto(a, sav_col, wait_col, title):
        for it, th in enumerate(theta_levels):
            s = full[np.isclose(full.share_willing, th)].sort_values("penalty")
            a.plot(s[wait_col], s[sav_col], "-", color=cmap_th[it],
                   linewidth=2.0, label=rf"$\theta = {th:g}$", zorder=2)
            for _, row in s.iterrows():
                a.scatter(row[wait_col], row[sav_col], s=38,
                          color=cmap_P[P_idx[row.penalty]],
                          edgecolor="black", linewidth=0.4, zorder=3)
        a.axhline(0, color="black", ls="--", lw=0.7, alpha=0.5)
        a.set_xlabel("Mean additional customer wait [d]")
        a.set_ylabel("Weekly cost saving vs baseline [%]")
        a.set_title(title, fontsize=11.5)
        a.grid(alpha=0.3)
        a.legend(loc="upper left", fontsize=9, ncol=2,
                 title="Willingness-to-wait share", title_fontsize=9)

    pareto(ax[0, 0], "routing_saving_plan1_pct", "wait_d_plan1",
           "(a) System Pareto front\n"
           r"routing lens $\cdot$ routing-optimal plan (stage 1)")
    pareto(ax[0, 1], "operator_saving_plan2_pct", "wait_d_plan2",
           "(b) System Pareto front\n"
           r"operator lens $\cdot$ operator-polished plan (stage 2)")
    sm = plt.cm.ScalarMappable(
        cmap=matplotlib.colors.ListedColormap(cmap_P),
        norm=matplotlib.colors.BoundaryNorm(
            np.arange(len(P_unique) + 1) - 0.5, ncolors=len(P_unique)))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax[0, 1], fraction=0.046, pad=0.03,
                      ticks=np.arange(len(P_unique)))
    cb.ax.set_yticklabels([f"{p:g}" for p in P_unique], fontsize=9)
    cb.set_label(r"Service penalty $P$ [€/p/d]", fontsize=10)

    # (c) per-LSP knee fronts, both lenses
    a = ax[0, 2]
    for kn, ls, mk, lab in [(knees_r, "-", "o", "routing lens / stage-1 plan"),
                            (knees_o, "--", "s",
                             "operator lens / stage-2 plan")]:
        for _, r in kn.iterrows():
            a.scatter(r.wait_d, r.saving_pct, s=70, marker=mk,
                      facecolor=S.CARRIER[r.carrier_class],
                      edgecolor="black", linewidth=0.6, zorder=3)
            a.annotate(r.provider, (r.wait_d, r.saving_pct),
                       textcoords="offset points", xytext=(6, 4),
                       fontsize=8)
    a.set_xlabel(r"Mean additional customer wait at $P^\star$ [d]")
    a.set_ylabel(r"Cost saving at $P^\star$ [%]")
    a.set_title(r"(c) Per-LSP knee $P^\star$, both lenses"
                "\ncircle = routing lens/stage 1, square = operator "
                "lens/stage 2", fontsize=11.5)
    a.grid(alpha=0.3)
    a.legend(handles=[Patch(facecolor=S.CARRIER[c], edgecolor="black",
                            label=c) for c in S.CARRIER_ORDER],
             loc="upper left", fontsize=9, title="Carrier type",
             title_fontsize=9)

    # (d)-(f) structural: delivery frequency of the operator plan
    cmap_q = plt.get_cmap(S.CMAP_QUARTILE)(np.linspace(0.30, 0.85, 4))
    _quartile_lines(ax[1, 0], cell_freq2, "hub_dist_km", "Hub distance",
                    "km", cmap_q)
    ax[1, 0].set_title("(d) Delivery frequency by hub distance\n"
                       r"operator-polished plan (stage 2), $P = 0$",
                       fontsize=11.5)
    _quartile_lines(ax[1, 1], cell_freq2, "parcels_per_stop",
                    "Parcels per drop-site", "pkts/site/wk", cmap_q)
    ax[1, 1].set_title("(e) Delivery frequency by parcels per drop-site\n"
                       r"operator-polished plan (stage 2), $P = 0$",
                       fontsize=11.5)
    _quartile_lines(ax[1, 2], cell_freq2, "hub_n_cells",
                    "Cells served by the hub", "areas", cmap_q,
                    edges=[0.5, 1.5, 4.5, 12.5, 1e9],
                    labs=["1 (single-cell hub)", "2--4", "5--12",
                          r"$\geq$13"])
    ax[1, 2].set_title("(f) Delivery frequency by hub size\n"
                       r"operator-polished plan (stage 2), $P = 0$",
                       fontsize=11.5)

    fig.tight_layout(pad=0.8, w_pad=1.6, h_pad=1.8, rect=[0, 0.035, 1, 1])
    fig.text(0.5, 0.005,
             "Panels (a) and (b) are different lenses AND different plans. "
             "Panel (f) is the mechanism behind the operator lens: a hub "
             "that serves a single area cannot rotate delivery days, so the "
             "operator polish drives it back to (near-)daily service. "
             + H.COST_MODEL_TEXT,
             ha="center", va="bottom", fontsize=10)
    save(fig, out, "fig6_structural_v2")


# ─────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────
HEADLINE_P = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]

HEADLINE_COLS = [
    "penalty", "share_willing",
    "routing_saving_plan1_pct", "routing_saving_plan2_pct",
    "operator_saving_plan1_pct", "operator_saving_plan2_pct",
    "sum_hub_peak_plan1", "sum_hub_peak_plan2",
    "hub_peak_plan1_vs_base_pct", "hub_peak_plan2_vs_base_pct",
    "wait_d_plan1", "wait_d_plan2",
    "mean_days_plan1", "mean_days_plan2",
]


def write_tables(full: pd.DataFrame, base: dict, knees_r, knees_o,
                 tables: Path, rev: Path) -> dict:
    tables.mkdir(parents=True, exist_ok=True)
    written = {}

    # -- 1. headline 2x2 at theta = 1 --------------------------------
    hl = full[np.isclose(full.share_willing, 1.0)
              & full.penalty.isin(HEADLINE_P)][HEADLINE_COLS].copy()
    p = tables / "tab_headline_theta1_v2.csv"
    hl.to_csv(p, index=False)
    written["headline"] = p
    pretty = hl.rename(columns={
        "penalty": r"$P$",
        "routing_saving_plan1_pct": "rout.\\ sav.\\ pl.1 [\\%]",
        "routing_saving_plan2_pct": "rout.\\ sav.\\ pl.2 [\\%]",
        "operator_saving_plan1_pct": "oper.\\ sav.\\ pl.1 [\\%]",
        "operator_saving_plan2_pct": "oper.\\ sav.\\ pl.2 [\\%]",
        "sum_hub_peak_plan1": "$\\sum_h$ peak pl.1",
        "sum_hub_peak_plan2": "$\\sum_h$ peak pl.2",
        "hub_peak_plan1_vs_base_pct": "peak pl.1 vs base [\\%]",
        "hub_peak_plan2_vs_base_pct": "peak pl.2 vs base [\\%]",
        "wait_d_plan1": "wait pl.1 [d]", "wait_d_plan2": "wait pl.2 [d]",
        "mean_days_plan1": "days/wk pl.1",
        "mean_days_plan2": "days/wk pl.2",
    }).drop(columns=["share_willing"])
    for c in (r"$\sum_h$ peak pl.1", r"$\sum_h$ peak pl.2"):
        pretty[c] = pretty[c].round(0).astype(int)
    H.to_latex(pretty, tables / "tab_headline_theta1_v2.tex",
               caption=(r"Full adoption ($\theta = 1$): both plans in both "
                        r"cost lenses. Plan~1 = routing-optimal (stage~1), "
                        r"plan~2 = operator-polished (stage~2). Savings are "
                        r"against the $\theta = 0$ daily-delivery baseline "
                        f"(routing {base['routing_eur']:,.0f}~\\euro/wk, "
                        f"operator {base['operator_eur']:,.0f}~\\euro/wk, "
                        r"$\sum_h$ peak "
                        f"{base['sum_hub_peak']:.0f} vehicles). "
                        + H.COST_MODEL_CAPTION),
               label="tab:headline_theta1")

    # -- 2. full grid ------------------------------------------------
    p = tables / "tab_grid_full_v2.csv"
    full.to_csv(p, index=False)
    written["grid"] = p

    # -- 3. per-LSP knees, both lenses -------------------------------
    kn = knees_r.merge(knees_o, on="provider", suffixes=("_routing",
                                                         "_operator"))
    kn = kn[["provider", "P_star_submission_routing",
             "class_submission_routing", "P_star_routing",
             "carrier_class_routing", "saving_pct_routing", "wait_d_routing",
             "P_star_operator", "carrier_class_operator",
             "saving_pct_operator", "wait_d_operator"]].rename(columns={
                 "P_star_submission_routing": "P_star_submission",
                 "class_submission_routing": "class_submission"})
    p = tables / "tab_pstar_knees_v2.csv"
    kn.to_csv(p, index=False)
    written["pstar"] = p
    kn_tex = kn.rename(columns={
        "provider": "LSP",
        "P_star_submission": r"$P^\star$ (submission)",
        "class_submission": "class (submission)",
        "P_star_routing": r"$P^\star$ rout./pl.1",
        "carrier_class_routing": "class rout./pl.1",
        "saving_pct_routing": r"sav.\ rout.\ [\%]",
        "wait_d_routing": r"wait rout.\ [d]",
        "P_star_operator": r"$P^\star$ oper./pl.2",
        "carrier_class_operator": "class oper./pl.2",
        "saving_pct_operator": r"sav.\ oper.\ [\%]",
        "wait_d_operator": r"wait oper.\ [d]"})
    H.to_latex(kn_tex, tables / "tab_pstar_knees_v2.tex",
               caption=(r"Per-LSP knee $P^\star$ on the (saving, wait) front "
                        r"at $\theta = 1$, recomputed in both lenses: "
                        r"routing lens on the routing-optimal plan, operator "
                        r"lens on the operator-polished plan. "
                        + H.COST_MODEL_CAPTION),
               label="tab:pstar_knees")

    # -- 4. fleet diagnostics ---------------------------------------
    diag = full[["penalty", "share_willing", "sum_hub_peak_plan2",
                 "flat_bound", "peak_gap", "peak_gap_eur", "sys_peak",
                 "sys_cv", "vehicle_days_plan2"]].copy()
    p = tables / "tab_fleet_diagnostics_v2.csv"
    diag.to_csv(p, index=False)
    written["diagnostics"] = p
    pts = diag[diag.apply(lambda r: (round(r.penalty, 3),
                                     round(r.share_willing, 3))
                          in ABLATION_POINTS, axis=1)]
    pts = pts.copy()
    # the CV lives at 0.0x -- a shared "%.2f" would print it as 0.03
    pts["sys_cv"] = pts.sys_cv.map(lambda v: f"{v:.3f}")
    pts = pts.rename(columns={
        "penalty": "$P$", "share_willing": r"$\theta$",
        "sum_hub_peak_plan2": r"$\sum_h$ peak",
        "flat_bound": "flat bound", "peak_gap": "gap [veh]",
        "peak_gap_eur": r"gap [\euro/wk]",
        "sys_peak": "system peak", "sys_cv": "Mo--Sa CV",
        "vehicle_days_plan2": "vehicle-days"})
    H.to_latex(pts, tables / "tab_fleet_diagnostics_v2.tex",
               caption=(r"Fleet diagnostics of the operator-polished plan: "
                        r"achieved $\sum_h$ hub peak against the "
                        r"flat-profile bound $\sum_h \lceil "
                        r"\mathrm{vehicle\ days}_h / 6 \rceil$; the gap is "
                        r"priced at 1\,134.90~\euro\ per peak vehicle and "
                        r"week."),
               label="tab:fleet_diagnostics")
    return written


ABLATION_POINTS = [(0.0, 1.0), (0.25, 1.0), (0.5, 1.0), (0.0, 0.6)]


def _ablation_row(variant: str, rev_dir: Path, base: dict, *,
                  routing_col: str, reconstruct: bool) -> pd.DataFrame:
    """One stage-2 variant at the four ablation points, or a reason why not."""
    costs_p = rev_dir / "tab_costs_v2.csv"
    if not costs_p.exists():
        return pd.DataFrame([dict(variant=variant, penalty=P,
                                  share_willing=th, status="DIR_MISSING")
                             for P, th in ABLATION_POINTS])
    costs = pd.read_csv(costs_p)
    fleet = pd.read_csv(rev_dir / "tab_fleet_per_hub_v2.csv")
    wait = pd.read_csv(rev_dir / "tab_wait_v2.csv")
    cover = costs.groupby(["penalty", "share_willing"]).provider.nunique()

    rows = []
    for P, th in ABLATION_POINTS:
        key = (P, th)
        n = int(cover.get(key, 0))
        if n != H.N_PROVIDERS:
            rows.append(dict(variant=variant, penalty=P, share_willing=th,
                             status=f"NOT_COVERED ({n}/7 providers)"))
            continue
        c = costs[np.isclose(costs.penalty, P)
                  & np.isclose(costs.share_willing, th)]
        f = fleet[np.isclose(fleet.penalty, P)
                  & np.isclose(fleet.share_willing, th)]
        w = wait[np.isclose(wait.penalty, P)
                 & np.isclose(wait.share_willing, th)]
        rout = float(c[routing_col].sum())
        hp = float(f.groupby(["provider", "hub"]).fleet.max().sum())
        vd = float(f.fleet.sum())
        if reconstruct:
            op = rout - H.FIXED_COST_EUR * vd + H.WEEK_FIXED_COST_EUR * hp
        else:
            op = float(c.operator_cost_eur.sum())
        rows.append(dict(
            variant=variant, penalty=P, share_willing=th, status="ok",
            operator_cost_eur=op,
            operator_saving_pct=100 * (base["operator_eur"] - op)
            / base["operator_eur"],
            sum_hub_peak=hp, vehicle_days=vd, routing_cost_eur=rout,
            routing_saving_pct=100 * (base["routing_eur"] - rout)
            / base["routing_eur"],
            wait_d=float(w.wait_num_willing.sum() / w.total_parcels.sum())))
    return pd.DataFrame(rows)


def write_ablation(rev: Path, base: dict, v5_costs: pd.DataFrame,
                   tables: Path) -> Path:
    """Stage-2 variants at four operating points.

    run 2  = range balancer + stage 3 (results/revision_2026_08) -- pre-6e
             schema, so the operator cost is RECONSTRUCTED from the fleet
             table (see H.reconstruct_operator_cost).
    v3     = operator-solo (results/revision_2026_08_v3)
    v4     = operator-freqpres (results/revision_2026_08_v4/_partial_...);
             the partial run only covers theta <= 0.4, so at the four
             operating points its plan is reported through v5's
             ``opcost_freqpres_start_eur`` -- the v4 plan's own OpCost,
             which v5 records as the ``v5 <= v4`` reference.
    v5     = frequency-free best-of-3 (this grid)
    """
    res = ROOT / "results"
    parts = [
        _ablation_row("run2 range balancer + stage 3",
                      res / "revision_2026_08", base,
                      routing_col="cost_stage3_eur", reconstruct=True),
        _ablation_row("v3 operator-solo", res / "revision_2026_08_v3", base,
                      routing_col="cost_stage2_eur", reconstruct=False),
        _ablation_row("v4 operator-freqpres (partial)",
                      res / "revision_2026_08_v4" / "_partial_5664cca", base,
                      routing_col="cost_stage2_eur", reconstruct=False),
        _ablation_row("v5 frequency-free best-of-3", rev, base,
                      routing_col="cost_stage2_eur", reconstruct=False),
    ]
    abl = pd.concat(parts, ignore_index=True)

    # v4's plan through v5's recorded OpCost (objective, penalty included)
    g = v5_costs.groupby(["penalty", "share_willing"], as_index=False).agg(
        opcost_v4=("opcost_freqpres_start_eur", "sum"),
        opcost_range=("opcost_range_start_eur", "sum"),
        opcost_v5=("operator_obj_eur", "sum"))
    ref = []
    for P, th in ABLATION_POINTS:
        r = g[np.isclose(g.penalty, P) & np.isclose(g.share_willing, th)]
        if r.empty:
            continue
        r = r.iloc[0]
        ref.append(dict(penalty=P, share_willing=th,
                        opcost_v4_plan_eur=float(r.opcost_v4),
                        opcost_range_plan_eur=float(r.opcost_range),
                        opcost_v5_plan_eur=float(r.opcost_v5),
                        v5_le_v4=bool(r.opcost_v5 <= r.opcost_v4 + 1e-6)))
    ref = pd.DataFrame(ref)
    ref.to_csv(tables / "tab_stage2_ablation_opcost_ref_v2.csv", index=False)

    p = tables / "tab_stage2_ablation_v2.csv"
    abl.to_csv(p, index=False)
    tex = (abl[abl.status == "ok"].drop(columns=["status"])
           .rename(columns={
               "variant": "stage-2 variant", "penalty": "$P$",
               "share_willing": r"$\theta$",
               "operator_cost_eur": r"oper.\ cost [\euro/wk]",
               "operator_saving_pct": r"oper.\ sav.\ [\%]",
               "sum_hub_peak": r"$\sum_h$ peak",
               "vehicle_days": "vehicle-days",
               "routing_cost_eur": r"rout.\ cost [\euro/wk]",
               "routing_saving_pct": r"rout.\ sav.\ [\%]",
               "wait_d": "wait [d]"}))
    H.to_latex(tex, tables / "tab_stage2_ablation_v2.tex",
               caption=(r"Stage-2 variants at four operating points "
                        r"($(P,\theta) = (0,1), (0.25,1), (0.5,1), (0,0.6)$). "
                        r"Run~2's operator cost is reconstructed from its "
                        r"fleet table (the pre-6e schema has no operator "
                        r"columns); stage~1 is bit-identical across all four "
                        r"variants by construction. "
                        + H.COST_MODEL_CAPTION),
               label="tab:stage2_ablation")
    print(abl.to_string(index=False))
    print("\nv4-plan OpCost reference (from v5's recorded start states):")
    print(ref.to_string(index=False))
    return p


# ─────────────────────────────────────────────────────────────────────────
# CO2 (§38.6) -- carried over unless a v5 gates report shows G1b diffs
# ─────────────────────────────────────────────────────────────────────────
def co2_decision(rev: Path) -> str:
    """Regenerate the CO2/vehicle-km table only if this grid can support it.

    The rule from the task brief is "regenerate only if this grid's gates
    report shows G1b diffs > 0".  Two facts decide it:

    1. whether this grid HAS a gates report with a G1b count at all, and
    2. whether this grid has VROOM route kilometres -- the CO2 table
       (Kompendium 38.6) is built from real routes at the four validated
       operating points, not from surrogate cost, so without a validation
       directory there is nothing to recompute it FROM.
    """
    gr = rev / "gates_report.md"
    val = rev / "validation"
    have_val = val.exists() and any(val.glob("*.csv"))
    if not gr.exists():
        gate = f"no gates report at {gr}, so no G1b diff is demonstrated"
    else:
        txt = gr.read_text(encoding="utf-8", errors="replace")
        gate = ("gates report present and mentions G1b"
                if "G1b" in txt else "gates report present, no G1b section")
    if not have_val:
        return (f"CARRY OVER unchanged. {gate}; and this grid has no "
                f"validation/ route data ({val}), so there are no VROOM "
                "kilometres to rebuild the table from -- the CO2 figures "
                "(Kompendium 38.6) stay the four validated points of the "
                "2026-07 revision until Task 12 re-validates.")
    return (f"REVIEW REQUIRED: {gate}, and this grid HAS validation route "
            f"data ({val}). Recompute the CO2/vehicle-km table from it "
            "before reusing the carried-over numbers.")


# ─────────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rev-dir", default=str(DEFAULT_REV),
                    help="v5-schema grid directory (default: %(default)s)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-tables", action="store_true")
    args = ap.parse_args(argv)

    rev = Path(args.rev_dir)
    if not rev.is_absolute():
        rev = (ROOT / rev).resolve()
    os.environ["REV_DIR"] = str(rev)      # the four builders honour this too
    figures, tables = rev / "figures", rev / "tables"

    print("=" * 74)
    print(f"v5 figures/tables  REV_DIR = {rev}")
    print("=" * 74)

    G = H.RevGrid(rev)
    base = H.baseline(G.costs)
    print(f"baseline (theta=0, identical for all {G.costs.penalty.nunique()} "
          f"P): routing {base['routing_eur']:,.2f} EUR/wk | operator "
          f"{base['operator_eur']:,.2f} EUR/wk | sum_h peak "
          f"{base['sum_hub_peak']:.0f} | vehicle-days "
          f"{base['vehicle_days']:.0f}")

    head = H.headline_rows(G.costs, G.wait, base, G.n_cells_per_provider())
    fl = H.fleet_aggregates(G.fleet)
    full = head.merge(fl, on=["penalty", "share_willing"], how="inner")
    assert len(full) == len(head), "fleet table does not cover every cell"
    assert np.allclose(full.sum_hub_peak_plan2, full.sum_hub_peak_fleet), (
        "sum_hub_peak in tab_costs_v2 disagrees with the fleet table -- "
        "the fleet CSV is not at the stage-2 plan")
    base_cv = float(full[np.isclose(full.share_willing, 0.0)].sys_cv.iloc[0])
    print(f"baseline Mon-Sat system-fleet CV = {base_cv:.4f}")

    for P, th in ABLATION_POINTS:
        r = full[np.isclose(full.penalty, P)
                 & np.isclose(full.share_willing, th)].iloc[0]
        print(f"  (P={P}, th={th}): routing {r.routing_saving_plan1_pct:6.2f} "
              f"| {r.routing_saving_plan2_pct:6.2f} %   operator "
              f"{r.operator_saving_plan1_pct:7.2f} | "
              f"{r.operator_saving_plan2_pct:6.2f} %   peak "
              f"{r.sum_hub_peak_plan1:.0f} -> {r.sum_hub_peak_plan2:.0f} "
              f"({r.hub_peak_plan2_vs_base_pct:+.1f} % vs base)   wait "
              f"{r.wait_d_plan1:.3f} -> {r.wait_d_plan2:.3f} d   days "
              f"{r.mean_days_plan1:.2f} -> {r.mean_days_plan2:.2f}   gap to "
              f"flat bound {r.peak_gap:.0f}")

    knees_r = H.lsp_knees(G.costs, G.wait, cost_col="cost_stage1_eur",
                          wait_num_col="wait_num_willing_stage1")
    knees_o = H.lsp_knees(G.costs, G.wait, cost_col="operator_cost_eur",
                          wait_num_col="wait_num_willing")
    print("\nper-LSP knee P* (theta=1):")
    print(f"{'LSP':>7} {'submission':>11} {'routing/pl1':>12} "
          f"{'operator/pl2':>13}  class(routing) -> class(operator)")
    for _, a in knees_r.iterrows():
        b = knees_o[knees_o.provider == a.provider].iloc[0]
        print(f"{a.provider:>7} {a.P_star_submission:>11.2f} "
              f"{a.P_star:>12.2f} {b.P_star:>13.2f}  "
              f"{a.carrier_class} -> {b.carrier_class}"
              f"{'   *CHANGED*' if a.carrier_class != b.carrier_class else ''}")

    if not args.no_figures:
        print("\n--- figures ---")
        sizes = H.schedule_sizes()
        mix1 = H.freq_mix(G.chosen, "schedule_idx_stage1", sizes)
        mix2 = H.freq_mix(G.chosen, "schedule_idx_balanced", sizes)
        fig5(full, base, base_cv, figures)
        fig5b(full, base, figures)
        fig4(mix1, mix2, figures)
        fig4b(full, figures)

        struct = _structural_facts()
        cf = G.chosen[np.isclose(G.chosen.penalty, 0.0)].copy()
        cf["plz"] = cf.plz.astype(str)
        cf["size"] = sizes[cf.schedule_idx_balanced.values.astype(int)]
        cf = cf.merge(struct, on=["provider", "plz"], how="left")
        assert cf.hub_dist_km.notna().all(), (
            "structural join left unmatched cells -- checkpoint and grid "
            "disagree on the (provider, plz) universe")
        fig6(full, knees_r, knees_o, cf, figures)

    if not args.no_tables:
        print("\n--- tables ---")
        write_tables(full, base, knees_r, knees_o, tables, rev)
        write_ablation(rev, base, G.costs, tables)
        print("\nwritten to " + str(tables.relative_to(ROOT)) + ":")
        for f in sorted(tables.iterdir()):
            print(f"  {f.name:<42} {f.stat().st_size:>9,d} B")

    print("\nCO2 table (Kompendium 38.6): " + co2_decision(rev))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
