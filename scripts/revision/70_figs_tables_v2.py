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
submitted revision figures from the 2026-07 directory (content-identical:
matplotlib stamps a ``/CreationDate`` into every PDF, so an md5 of a
re-render differs even when the drawing does not.  Figures produced by THIS
module suppress that stamp and are byte-identical across re-renders, which
is what the provenance manifest and gate G7 rely on).

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
The paper is ACCEPTED, so its figures keep the SUBMITTED layout and
only their numbers change.  Those three are rendered by the FROZEN
builders (30_/31_/32_) on tables ``74_v2_to_legacy_tables.py`` adapts
from this grid, and land in ``figures/`` under their canonical stems:

``figures/``
  ``fig4_SM_mix_pct_8P.{png,pdf}``          ACCEPTED  (32_)
  ``fig5_grid_heatmap_6_smoothed.{png,pdf}``ACCEPTED  (30_)
  ``fig6_structural_grid_6_smoothed.*``     ACCEPTED  (31_)

Everything this module draws itself is SUPPLEMENTARY -- the two-lens,
two-plan set -- and carries a ``supp_`` prefix so no sync can put it
in a main figure slot:

  ``manifest.json``                        provenance: rev dir, git
                                           HEAD, timestamp, md5 of
                                           every figure (accepted AND
                                           supplementary) and of the
                                           four grid CSVs.  ``71_``
                                           takes its source from here.
  ``supp_fig4_freq_mix_two_plans.*``       frequency mix, both plans
  ``supp_fig4b_mean_days.*``               mean delivery days
  ``supp_fig5_grid_heatmap_v2.*``          two-lens 2x3
  ``supp_fig5b_offdiagonal_v2.*``          the other half of the 2x2
  ``supp_fig6_structural_v2.*``            per-cell euro breakdown
  ``supp_fig6b_operator_lens_v2.*``        the operator lens per hub

``tables/``
  ``tab_headline_theta1_v2.csv|.tex``     2x2 headline at theta=1
  ``tab_grid_full_v2.csv``                the same columns for all 88 cells
  ``tab_pstar_knees_v2.csv|.tex``         per-LSP knee P*, both lenses
  ``tab_stage2_ablation_v2.csv|.tex``     range / solo / freqpres / v5
  ``tab_fleet_diagnostics_v2.csv|.tex``   gap to the flat-profile bound
  ``tab_per_cell_structural_v2.csv``      median + IQR per (feature, bucket,
                                          theta, plan), euro AND percent
  ``tab_head_usage_summary*_v2.csv|.tex`` head-priced share of the pooled
                                          cost, occurrence-weighted
  ``tab_grid_delta_v2.csv|.tex``          what the bundle head changed
                                          (``--compare-dir``)
  ``tab_co2_km_v2.csv|.tex``              vehicle-km and CO2 from THIS
                                          grid's validated VROOM routes

Per-cell euro costs come from ``72_per_cell_costs_v2.py``
(``<rev>/tables/tab_per_cell_costs_v2.csv``); their sum is re-gated against
the grid's own routing total, vehicle-days and hub peaks here, from the
written CSVs alone.  A missing per-cell table REFUSES (exit 2) unless
``--allow-missing-per-cell`` is passed -- fig 6 (b)-(f) has no substitute.

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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts" / "presentation"))

import _figs_tables_v2 as H  # noqa: E402
import _style as S  # noqa: E402  -- the ONLY colour source

DEFAULT_REV = ROOT / "results" / "revision_2026_08_v5"
#: the head-free grid a head-enabled render is compared against
DEFAULT_COMPARE = ROOT / "results" / "revision_2026_08_v5"
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


# Matplotlib stamps a /CreationDate into every PDF and a timestamp into
# every PNG, so two renders of identical content get different md5s. The
# manifest and gate G7 are md5-based, so both are suppressed here: a
# re-render of unchanged inputs must be byte-identical, not merely
# content-identical.
_PDF_META = {"CreationDate": None}
_PNG_META = {"Software": None}


def save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext, meta in (("png", _PNG_META), ("pdf", _PDF_META)):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", metadata=meta)
        # relative_to raises when the output tree is outside the repo (a
        # scratch render, another checkout): a path in a progress line is
        # never worth a traceback.
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"  saved {shown}")
        written.append(path)
    plt.close(fig)
    return written


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

    The flip is an explicit toggle on the base name, not a string slice:
    slicing ``"RdBu_r"[:-2]`` happens to work today and would silently
    return ``"Rd"`` if the style module ever named the map ``"RdBu"``.
    """
    base = S.CMAP_CHANGE
    reversed_ = base.endswith("_r")
    stem = base[:-2] if reversed_ else base
    want_reversed = reversed_ if not higher_is_better else not reversed_
    return stem + ("_r" if want_reversed else "")


def _sym_norm(*mats) -> TwoSlopeNorm:
    """Symmetric diverging norm shared by every matrix passed in.

    Fig. 5 and fig. 5b show the SAME quantity for the two plans; passing
    both matrices here gives them one scale, so the comparison the companion
    figure exists for is visual and not arithmetic.
    """
    a = max(max(abs(np.nanmin(m.values)), abs(np.nanmax(m.values)))
            for m in mats)
    a = max(float(np.ceil(a)), 1.0)
    return TwoSlopeNorm(vmin=-a, vcenter=0.0, vmax=a)


def panel(ax, full: pd.DataFrame, col: str, letter: str, what: str,
          cmap, **kw):
    """Draw one heat-map panel and DERIVE its plan/lens label from ``col``.

    The amendment's rule -- never mix a plan or a lens in one panel, always
    label it -- is enforced structurally here: the title's second line comes
    from ``H.declare(col)``, so changing the plotted column changes the
    label, and a column the mapping does not know about raises rather than
    rendering an unlabelled panel.  No panel title in this module hard-codes
    a plan or a lens.
    """
    title = f"({letter}) {what}\n" + H.declare(col).replace(
        "·", r"$\cdot$")
    return heat(ax, piv(full, col), cmap, title, **kw)


# ─────────────────────────────────────────────────────────────────────────
# Fig. 5 -- headline 2x3, and the off-diagonal companion
# ─────────────────────────────────────────────────────────────────────────
def _shared_norms(full: pd.DataFrame) -> dict:
    """One scale per quantity across fig. 5 and its companion fig. 5b.

    Both figures draw the wait and the hub-peak change, for the two plans.
    Auto-scaling each panel separately made 0.77 d and 0.97 d render as the
    same deep red; a shared norm makes the two plans comparable by eye.
    """
    return {
        "wait_max": float(np.ceil(max(full.wait_d_plan1.max(),
                                      full.wait_d_plan2.max()) * 10) / 10),
        "peak_norm": _sym_norm(piv(full, "hub_peak_plan1_vs_base_pct"),
                               piv(full, "hub_peak_plan2_vs_base_pct")),
    }


def _grid_axes(fig, ax) -> None:
    for a in ax[-1, :]:
        a.set_xlabel(r"Willingness-to-wait share $\theta$ [%]", fontsize=12)
    for a in ax[:, 0]:
        a.set_ylabel(r"Service penalty $P$ [€/p/d]", fontsize=12)


def _baseline_sentence(base: dict) -> str:
    return (r"All quantities are relative to the daily-delivery baseline at "
            r"$\theta = 0$ "
            f"(routing {base['routing_eur']:,.0f} €/wk, operator "
            f"{base['operator_eur']:,.0f} €/wk, "
            r"$\sum_h$ peak " f"{base['sum_hub_peak']:.0f} vehicles). ")


def fig5(full: pd.DataFrame, base: dict, base_cv: float, out: Path,
         norms: dict) -> list[Path]:
    rcParams.update(RC)
    fig, ax = plt.subplots(2, 3, figsize=(19.0, 10.2))

    panel(ax[0, 0], full, "routing_saving_plan1_pct", "a",
          "Routing-cost saving", S.CMAP_SAVING,
          vmin=0, cbar_label="Routing saving [%]")
    panel(ax[0, 1], full, "operator_saving_plan2_pct", "b",
          "Operator-cost saving", S.CMAP_SAVING,
          vmin=0, cbar_label="Operator saving [%]")
    panel(ax[0, 2], full, "wait_d_plan2", "c",
          "Mean additional wait per parcel", S.CMAP_WAIT,
          vmin=0, vmax=norms["wait_max"], fmt="{:.2f}",
          cbar_label="Wait [d]", invert_thr=True)

    panel(ax[1, 0], full, "hub_peak_plan2_vs_base_pct", "d",
          r"$\sum_h$ hub peak vehicles vs baseline",
          _signed_cmap(higher_is_better=False),
          norm=norms["peak_norm"], cbar_label="Change [%]")
    # reversed fleet ramp: light = flat week = the operator's goal, matching
    # panel (c) where light = little wait
    panel(ax[1, 1], full, "sys_cv", "e",
          f"Mon-Sat CV of the system fleet (baseline {base_cv:.3f})",
          S.fleet_cmap(reverse=True),
          vmin=0, fmt="{:.3f}", cbar_label="CV [-]", invert_thr=True)
    panel(ax[1, 2], full, "vehicle_days_plan2_vs_base_pct", "f",
          "Weekly vehicle-days vs baseline",
          _signed_cmap(higher_is_better=False),
          norm=_sym_norm(piv(full, "vehicle_days_plan2_vs_base_pct")),
          cbar_label="Change [%]")

    _grid_axes(fig, ax)
    fig.tight_layout(pad=0.6, w_pad=1.4, h_pad=1.9, rect=[0, 0.045, 1, 1])
    fig.text(0.5, 0.006,
             "Panels (a) and (b) are different cost lenses AND different "
             "plans and must not be read as one series. "
             + _baseline_sentence(base) + H.COST_MODEL_TEXT,
             ha="center", va="bottom", fontsize=10)
    return save(fig, out, "supp_fig5_grid_heatmap_v2")


def fig5b(full: pd.DataFrame, base: dict, out: Path,
          norms: dict) -> list[Path]:
    """The off-diagonal cells of the 2 plans x 2 lenses matrix."""
    rcParams.update(RC)
    fig, ax = plt.subplots(2, 2, figsize=(13.0, 10.2))

    panel(ax[0, 0], full, "operator_saving_plan1_pct", "a",
          "Operator-cost saving", _signed_cmap(higher_is_better=True),
          norm=_sym_norm(piv(full, "operator_saving_plan1_pct")),
          cbar_label="Operator saving [%]")
    panel(ax[0, 1], full, "routing_saving_plan2_pct", "b",
          "Routing-cost saving", S.CMAP_SAVING,
          vmin=0, cbar_label="Routing saving [%]")
    panel(ax[1, 0], full, "wait_d_plan1", "c",
          "Mean additional wait per parcel", S.CMAP_WAIT,
          vmin=0, vmax=norms["wait_max"], fmt="{:.2f}",
          cbar_label="Wait [d]", invert_thr=True)
    panel(ax[1, 1], full, "hub_peak_plan1_vs_base_pct", "d",
          r"$\sum_h$ hub peak vehicles vs baseline",
          _signed_cmap(higher_is_better=False),
          norm=norms["peak_norm"], cbar_label="Change [%]")

    _grid_axes(fig, ax)
    fig.tight_layout(pad=0.6, w_pad=1.4, h_pad=1.9, rect=[0, 0.05, 1, 1])
    fig.text(0.5, 0.008,
             "Companion to Fig. 5: the two off-diagonal cells of the "
             f"2 plans x 2 lenses matrix, plus the {H.PLAN_WORD[1]} "
             "plan's wait and hub peaks. Panels (c) and (d) share their "
             "colour "
             "scale with Fig. 5 (c) and (d), so the two plans are "
             "comparable by eye. Panel (a) is negative where the "
             "routing-optimal two-day patterns multiply the hub peaks. "
             + H.COST_MODEL_TEXT,
             ha="center", va="bottom", fontsize=10)
    return save(fig, out, "supp_fig5b_offdiagonal_v2")


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


def fig4(mix1: pd.DataFrame, mix2: pd.DataFrame,
         out: Path) -> list[Path]:
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
    axes[0, 0].set_ylabel(H.PLAN_ROW[1] + "\nShare of areas [%]",
                          fontsize=9)
    axes[1, 0].set_ylabel(H.PLAN_ROW[2] + "\nShare of areas [%]",
                          fontsize=9)
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
    return save(fig, out, "supp_fig4_freq_mix_two_plans")


def fig4b(full: pd.DataFrame, out: Path) -> list[Path]:
    """Mean delivery days per week, both plans -- the (0,1) 2.03 vs 2.37."""
    rcParams.update(RC)
    P_values = sorted(full.penalty.unique())
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    cmap = plt.get_cmap(S.CMAP_PENALTY)(np.linspace(0.05, 0.92,
                                                    len(P_values)))
    for letter, a, col in [("a", ax[0], "mean_days_plan1"),
                           ("b", ax[1], "mean_days_plan2")]:
        for i, P in enumerate(P_values):
            g = full[np.isclose(full.penalty, P)].sort_values("share_willing")
            a.plot(g.share_willing * 100, g[col], "o-", color=cmap[i],
                   markersize=4, linewidth=1.8, label=rf"$P = {P:g}$")
        # label derived from the plotted column, never hand-written
        a.set_title(f"({letter}) " + H.declare(col).capitalize(),
                    fontsize=12)
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
    return save(fig, out, "supp_fig4b_mean_days")


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
    df = pd.DataFrame(rows)
    # Region type (urban / suburban / rural) -- the submitted fig 6 (c)
    # split, joined at PLZ level (the grid's cell key).
    rt = H.region_types(ROOT)
    df = df.merge(rt, on="plz", how="left")
    assert df.raumtyp_3.notna().all(), (
        "cells without a region type: "
        f"{sorted(df[df.raumtyp_3.isna()].plz.unique())[:10]} -- "
        "data/geodata/plz_raumtyp.csv does not cover the grid's PLZ set")
    return df


def _quartile_lines(ax, df, feat, label, unit, cmap,
                    edges=None, labs=None) -> pd.DataFrame:
    """Median weekly delivery days vs theta, bucketed by a feature.

    With ``edges``/``labs`` omitted the buckets are the feature's own
    quartiles.  Discrete features (hub size) pass explicit class edges,
    because their quartiles collapse onto repeated values.

    Every legend entry carries the bucket's **n and its carrier**
    composition, and the composition is returned so the caption can
    state it.  Region Hannover has exactly one multi-depot network
    (DHL, 16 hubs); every other LSP runs a single depot.  A structural
    bucket can therefore be one carrier's depot network in disguise,
    and a panel that hides that invites a causal reading it cannot
    support.
    """
    d = df[df[feat].notna()].copy()
    if edges is None:
        edges = d[feat].quantile([0, .25, .5, .75, 1.]).values
        labs = [f"Q1 ($\\leq${edges[1]:.1f})",
                f"Q2 ({edges[1]:.1f}-{edges[2]:.1f})",
                f"Q3 ({edges[2]:.1f}-{edges[3]:.1f})",
                f"Q4 (>{edges[3]:.1f})"]
        assert len(set(np.round(edges, 9))) == len(edges), (
            f"{feat}: quartile edges are not unique ({edges}) -- pass "
            "explicit class edges for this discrete feature")
    d["q"] = pd.cut(d[feat], bins=edges, labels=labs, include_lowest=True)
    comp = []
    for i, ql in enumerate(labs):
        g = d[d.q == ql]
        if g.empty:
            continue
        provs = sorted(g.provider.unique())
        one = provs[0] if len(provs) == 1 else f"{len(provs)} LSPs"
        # a PLZ code is served by several LSPs, so the unit is the
        # (provider, plz) CELL -- plz.nunique() would undercount by ~6x
        n_cells = int(g.groupby(["provider", "plz"]).ngroups
                      if {"provider", "plz"} <= set(g.columns) else len(g))
        comp.append(dict(feature=feat, bucket=ql, n=n_cells,
                         providers=", ".join(provs),
                         single_carrier=len(provs) == 1))
        agg = (g.groupby("share_willing")["size"]
               .agg(med="median", lo=lambda s: s.quantile(.25),
                    hi=lambda s: s.quantile(.75)).reset_index())
        ax.fill_between(agg.share_willing * 100, agg.lo, agg.hi,
                        color=cmap[i], alpha=0.18, linewidth=0)
        ax.plot(agg.share_willing * 100, agg.med, "o-", color=cmap[i],
                linewidth=2.0, markersize=5,
                markeredgecolor="black", markeredgewidth=0.5,
                label=f"{ql}  n={n_cells}, {one}")
    ax.set_xlabel(r"Willingness-to-wait share $\theta$ [%]")
    ax.set_ylabel("Delivery days per week\n(median, band = IQR)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, title=f"{label} [{unit}]",
              title_fontsize=8.5)
    return pd.DataFrame(comp)


def _bucketise(df: pd.DataFrame, feat: str, edges=None, labs=None
               ) -> tuple[pd.DataFrame, list]:
    """Add a categorical ``bucket`` column; quartiles unless edges are given.

    Discrete features (hub size, region type) pass explicit classes, because
    their quartiles collapse onto repeated values.
    """
    d = df[df[feat].notna()].copy()
    if edges is None:
        edges = d[feat].quantile([0, .25, .5, .75, 1.]).values
        labs = [f"Q1 ($\\leq${edges[1]:.1f})",
                f"Q2 ({edges[1]:.1f}-{edges[2]:.1f})",
                f"Q3 ({edges[2]:.1f}-{edges[3]:.1f})",
                f"Q4 (>{edges[3]:.1f})"]
        assert len(set(np.round(edges, 9))) == len(edges), (
            f"{feat}: quartile edges are not unique ({edges}) -- pass "
            "explicit class edges for this discrete feature")
    d["bucket"] = pd.cut(d[feat], bins=edges, labels=labs,
                         include_lowest=True)
    return d, list(labs)


def _at_own_pstar(df: pd.DataFrame, knees: pd.DataFrame) -> pd.DataFrame:
    """Keep each LSP's rows at ITS OWN knee P* (all theta).

    The knee is lens-specific (Task 13 §4), so the caller passes the knee
    table of the lens the panel is drawn in; pairing a lens with the other
    lens's P* is exactly the mixture the amendment forbids.
    """
    parts = []
    for _, r in knees.iterrows():
        parts.append(df[(df.provider == r.provider)
                        & np.isclose(df.penalty, r.P_star)])
    out = pd.concat(parts, ignore_index=True)
    assert out.provider.nunique() == knees.provider.nunique(), (
        "an LSP has no rows at its own P* -- knee table and data disagree")
    return out


def _saving_panel(ax, df: pd.DataFrame, order: list, colours: dict,
                  value: str, ylabel: str, letter: str, what: str,
                  lens: str, legend_title: str, legend_loc="upper left",
                  band_plan: str = "stage1") -> pd.DataFrame:
    """Median *value* vs theta per bucket, BOTH plans, one lens.

    Line style carries the plan (solid = the plan named first in
    ``H.PLAN_OF_KEY``), colour carries the bucket, and the IQR band is drawn
    for ``band_plan`` only -- two overlapping bands would be unreadable.
    Every legend entry carries the bucket's n and its carrier composition:
    Region Hannover has exactly one multi-depot network, so a structural
    bucket can be one carrier's cells in disguise (Task 13 review I-1).
    """
    comp = H.bucket_composition(df, "bucket").set_index("bucket")
    for plan, style, lw in (("stage1", "-", 2.0), ("balanced", "--", 1.7)):
        d = df[df.plan == plan]
        if d.empty:
            continue
        agg = H.median_by_bucket(d, value, "bucket")
        for b in order:
            g = agg[agg.bucket == b].sort_values("share_willing")
            if g.empty:
                continue
            lab = None
            if plan == band_plan and b in comp.index:
                c = comp.loc[b]
                provs = str(c.providers).split(", ")
                who = provs[0] if c.single_carrier else f"{len(provs)} LSPs"
                lab = f"{b}  n={int(c.n)}, {who}"
            ax.plot(g.share_willing * 100, g.med, style, color=colours[b],
                    linewidth=lw, marker="o" if plan == band_plan else None,
                    markersize=4, markeredgecolor="black",
                    markeredgewidth=0.4, label=lab, zorder=3)
            if plan == band_plan:
                ax.fill_between(g.share_willing * 100, g.lo, g.hi,
                                color=colours[b], alpha=0.16, linewidth=0)
    ax.axhline(0, color="black", ls=":", lw=0.7, alpha=0.6)
    ax.set_xlabel(r"Willingness-to-wait share $\theta$ [%]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"({letter}) {what}\n{lens}" + r" $\cdot$ both plans, "
                 r"each LSP at its own $P^\star$", fontsize=11.5)
    ax.grid(alpha=0.3)
    leg = ax.legend(loc=legend_loc, fontsize=8, title=legend_title,
                    title_fontsize=8.5)
    ax.add_artist(leg)
    ax.legend(handles=[
        Line2D([], [], color="0.25", linestyle=s, linewidth=1.8,
               label=H.PLAN_WORD[H.PLAN_OF_KEY[p]])
        for p, s in (("stage1", "-"), ("balanced", "--"))],
        loc="lower left", fontsize=8, title="Plan", title_fontsize=8.5)
    return comp.reset_index()


def _pareto(ax, full, letter, sav_col, wait_col, theta_levels, cmap_th,
            cmap_P, P_idx):
    """One system Pareto front. Refuses a saving/wait pair of two plans."""
    assert H.plan_of(sav_col) == H.plan_of(wait_col), (
        f"Pareto panel ({letter}) mixes plans: {sav_col} is plan "
        f"{H.plan_of(sav_col)}, {wait_col} is plan {H.plan_of(wait_col)}")
    for it, th in enumerate(theta_levels):
        g = full[np.isclose(full.share_willing, th)].sort_values("penalty")
        ax.plot(g[wait_col], g[sav_col], "-", color=cmap_th[it],
                linewidth=2.0, label=rf"$\theta = {th:g}$", zorder=2)
        for _, row in g.iterrows():
            ax.scatter(row[wait_col], row[sav_col], s=38,
                       color=cmap_P[P_idx[row.penalty]],
                       edgecolor="black", linewidth=0.4, zorder=3)
    ax.axhline(0, color="black", ls="--", lw=0.7, alpha=0.5)
    ax.set_xlabel("Mean additional customer wait [d]")
    ax.set_ylabel("Weekly cost saving vs baseline [%]")
    ax.set_title(f"({letter}) System Pareto front\n"
                 + H.declare(sav_col).replace("·", r"$\cdot$"),
                 fontsize=11.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=2,
              title="Willingness-to-wait share", title_fontsize=9)


def fig6(full: pd.DataFrame, cells_r: pd.DataFrame, out: Path
         ) -> tuple[list[Path], pd.DataFrame]:
    """(a) system Pareto + (b)-(f) MEDIAN per-cell euro saving.

    This is the submitted figure's layout -- Pareto, carrier type, region
    type, hub distance, postal-code area, parcels per drop-site -- rebuilt
    on the v5/v6 grid from ``72_per_cell_costs_v2.py``'s per-cell costs.
    The submitted version approximated a cell's express cost by a
    parcel-proportional share of the PROVIDER's express total; the per-cell
    table splits each realised TOUR instead, and its sum is gated against
    the grid's own routing total, so nothing here is an approximation of
    the grid it summarises.

    ``area size`` is panel (e), as in the submission -- the hub-size split
    (which is DHL-only below its top class) lives in the operator-lens
    companion, where the mechanism it shows belongs.
    """
    rcParams.update(RC)
    fig, ax = plt.subplots(2, 3, figsize=(18.0, 10.4))

    theta_levels = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    P_unique = np.array(sorted(full.penalty.unique()))
    cmap_th = plt.get_cmap(S.CMAP_THETA)(np.linspace(0.15, 0.9,
                                                     len(theta_levels)))
    cmap_P = plt.get_cmap(S.CMAP_PENALTY)(np.linspace(0.05, 0.95,
                                                      len(P_unique)))
    P_idx = {P: i for i, P in enumerate(P_unique)}
    _pareto(ax[0, 0], full, "a", "routing_saving_plan1_pct", "wait_d_plan1",
            theta_levels, cmap_th, cmap_P, P_idx)
    sm = plt.cm.ScalarMappable(
        cmap=matplotlib.colors.ListedColormap(cmap_P),
        norm=matplotlib.colors.BoundaryNorm(
            np.arange(len(P_unique) + 1) - 0.5, ncolors=len(P_unique)))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax[0, 0], fraction=0.046, pad=0.03,
                      ticks=np.arange(len(P_unique)))
    cb.ax.set_yticklabels([f"{p:g}" for p in P_unique], fontsize=9)
    cb.set_label(r"Service penalty $P$ [€/p/d]", fontsize=10)

    ylab = "Median per-area saving\n[€/week]"
    cmap_q = plt.get_cmap(S.CMAP_QUARTILE)(np.linspace(0.45, 0.92, 4))
    comps = []

    # (b) carrier type -- an explicit class order, coloured by S.CARRIER
    d = cells_r.copy()
    d["bucket"] = pd.Categorical(d.carrier_class,
                                 categories=S.CARRIER_ORDER, ordered=True)
    comps.append(_saving_panel(
        ax[0, 1], d, S.CARRIER_ORDER, dict(S.CARRIER), "saving_eur", ylab,
        "b", "Saving by carrier type", H.LENS_ROUTING, "Carrier type")
        .assign(feature="carrier_class"))

    # (c) region type
    d = cells_r[cells_r.raumtyp_3.notna()].copy()
    d["bucket"] = pd.Categorical(d.raumtyp_3, categories=H.RAUMTYP_ORDER,
                                 ordered=True)
    comps.append(_saving_panel(
        ax[0, 2], d, H.RAUMTYP_ORDER, dict(S.RAUMTYP), "saving_eur", ylab,
        "c", "Saving by region type", H.LENS_ROUTING, "Region type")
        .assign(feature="raumtyp_3"))

    # (d)-(f) continuous structure, own quartiles
    for a_, letter, feat, title, legend_title in (
            (ax[1, 0], "d", "hub_dist_km", "Saving by hub distance",
             "Hub distance [km]"),
            (ax[1, 1], "e", "area_km2", "Saving by postal-code area",
             r"Area [km$^2$]"),
            (ax[1, 2], "f", "parcels_per_stop",
             "Saving by parcels per drop-site",
             "Parcels per drop-site visit [pkts/visit]")):
        d, order = _bucketise(cells_r, feat)
        comps.append(_saving_panel(
            a_, d, order, {b: cmap_q[i] for i, b in enumerate(order)},
            "saving_eur", ylab, letter, title, H.LENS_ROUTING,
            legend_title).assign(feature=feat))

    comp = pd.concat(comps, ignore_index=True)
    fig.tight_layout(pad=0.8, w_pad=1.6, h_pad=1.8, rect=[0, 0.045, 1, 1])
    fig.text(0.5, 0.004,
             "Panels (b)-(f): median over postal-code areas of the weekly "
             "cost saving of that area against its own daily-delivery "
             f"baseline, {H.LENS_ROUTING}; band = IQR of the "
             f"{H.PLAN_WORD[1]} plan. Each LSP is shown at its own knee "
             r"$P^\star$ of this lens. A cell's cost is its own tour plus a "
             "parcel-proportional share of every pooled tour it rides in; "
             "the per-area costs sum exactly to the grid's routing total "
             "(identity gate, 72_). " + H.COST_MODEL_TEXT,
             ha="center", va="bottom", fontsize=9, wrap=True)
    return save(fig, out, "supp_fig6_structural_v2"), comp


def fig6b(full: pd.DataFrame, knees_r: pd.DataFrame, knees_o: pd.DataFrame,
          hubs_o: pd.DataFrame, cell_freq2: pd.DataFrame, out: Path
          ) -> tuple[list[Path], pd.DataFrame]:
    """The operator lens: per HUB, not per cell -- plus the two lens panels.

    The weekly fixed bill is sized by a hub's PEAK day, so it is a property
    of the hub: what a cell "contributes" to it depends on when the rest of
    the hub peaks.  A per-cell operator saving is therefore not a
    well-defined quantity, and this figure does not draw one -- it shows the
    operator lens at the level where the number means something, and says so
    in its footer.
    """
    rcParams.update(RC)
    fig, ax = plt.subplots(2, 3, figsize=(18.0, 10.4))

    theta_levels = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    P_unique = np.array(sorted(full.penalty.unique()))
    cmap_th = plt.get_cmap(S.CMAP_THETA)(np.linspace(0.15, 0.9,
                                                     len(theta_levels)))
    cmap_P = plt.get_cmap(S.CMAP_PENALTY)(np.linspace(0.05, 0.95,
                                                      len(P_unique)))
    P_idx = {P: i for i, P in enumerate(P_unique)}
    _pareto(ax[0, 0], full, "a", "operator_saving_plan2_pct", "wait_d_plan2",
            theta_levels, cmap_th, cmap_P, P_idx)

    # (b) per-LSP knee P*, both lenses (marker = lens/plan, fill = class)
    a = ax[0, 1]
    marker_handles = []
    for kn, mk, cost_col in [(knees_r, "o", "routing_saving_plan1_pct"),
                             (knees_o, "s", "operator_saving_plan2_pct")]:
        for _, r in kn.iterrows():
            a.scatter(r.wait_d, r.saving_pct, s=70, marker=mk,
                      facecolor=S.CARRIER[r.carrier_class],
                      edgecolor="black", linewidth=0.6, zorder=3)
            a.annotate(r.provider, (r.wait_d, r.saving_pct),
                       textcoords="offset points", xytext=(6, 4), fontsize=8)
        marker_handles.append(Line2D(
            [], [], linestyle="none", marker=mk, markersize=8,
            markerfacecolor="white", markeredgecolor="black",
            label=H.declare(cost_col).replace("·", "/")))
    a.set_xlabel(r"Mean additional customer wait at $P^\star$ [d]")
    a.set_ylabel(r"Cost saving at $P^\star$ [%]")
    a.set_title(r"(b) Per-LSP knee $P^\star$, both lenses", fontsize=11.5)
    a.grid(alpha=0.3)
    leg1 = a.legend(handles=marker_handles, loc="lower right", fontsize=8,
                    title="Lens / plan", title_fontsize=8.5)
    a.add_artist(leg1)
    a.legend(handles=[Patch(facecolor=S.CARRIER[c], edgecolor="black",
                            label=c) for c in S.CARRIER_ORDER],
             loc="upper left", fontsize=9, title="Carrier type",
             title_fontsize=9)

    ylab = "Median per-hub saving\n[€/week]"
    cmap_q = plt.get_cmap(S.CMAP_QUARTILE)(np.linspace(0.45, 0.92, 4))
    comps = []

    # (c) per-hub operator saving by carrier type
    d = hubs_o.copy()
    d["bucket"] = pd.Categorical(d.carrier_class,
                                 categories=S.CARRIER_ORDER, ordered=True)
    comps.append(_saving_panel(
        ax[0, 2], d, S.CARRIER_ORDER, dict(S.CARRIER), "operator_saving_eur",
        ylab, "c", "Hub saving by carrier type", H.LENS_OPERATOR,
        "Carrier type", band_plan="balanced").assign(feature="carrier_class"))

    # (d) per-hub operator saving by hub size -- the DHL-only caveat panel
    hub_edges = [0.5, 1.5, 4.5, 12.5, 1e9]
    hub_labs = ["1 (single-cell hub)", "2-4", "5-12", r"$\geq$13"]
    d, order = _bucketise(hubs_o, "n_cells", edges=hub_edges, labs=hub_labs)
    comps.append(_saving_panel(
        ax[1, 0], d, order, {b: cmap_q[i] for i, b in enumerate(order)},
        "operator_saving_eur", ylab, "d", "Hub saving by hub size",
        H.LENS_OPERATOR, "Cells served by the hub",
        band_plan="balanced").assign(feature="hub_n_cells"))

    # (e) per-hub operator saving by the hub's dominant region type
    d = hubs_o[hubs_o.raumtyp_3.notna()].copy()
    d["bucket"] = pd.Categorical(d.raumtyp_3, categories=H.RAUMTYP_ORDER,
                                 ordered=True)
    comps.append(_saving_panel(
        ax[1, 1], d, H.RAUMTYP_ORDER, dict(S.RAUMTYP), "operator_saving_eur",
        ylab, "e", "Hub saving by region type", H.LENS_OPERATOR,
        "Region type (parcel-dominant)",
        band_plan="balanced").assign(feature="raumtyp_3"))

    # (f) the mechanism: delivery frequency by hub size (Kompendium 40.14)
    fcomp = _quartile_lines(ax[1, 2], cell_freq2, "hub_n_cells",
                            "Cells served by the hub", "areas", cmap_q,
                            edges=hub_edges, labs=hub_labs)
    single = fcomp[fcomp.single_carrier]
    carriers = sorted({c for cs in single.providers for c in cs.split(", ")})
    note = (f" ({carriers[0]} only below the top class)"
            if len(single) and len(carriers) == 1 else "")
    ax[1, 2].set_title(f"(f) Delivery frequency by hub size{note}\n"
                       + H.declare("mean_days_plan2") + r", $P = 0$",
                       fontsize=11.5)

    comp = pd.concat(comps + [fcomp.assign(panel="f")], ignore_index=True)
    if len(single) and len(carriers) == 1:
        who = carriers[0]
        confound = (
            f"Panels (d) and (f): the small-hub buckets hold {who} cells "
            f"only, because {who} runs the only multi-depot network in the "
            "case study (16 hubs; every other LSP has a single depot). Hub "
            "size and carrier identity are confounded there, so those two "
            f"panels are within-{who} statements, not general laws. ")
    else:
        confound = ""
    fig.tight_layout(pad=0.8, w_pad=1.6, h_pad=1.8, rect=[0, 0.045, 1, 1])
    fig.text(0.5, 0.004,
             f"The {H.LENS_OPERATOR} is HUB-attributable, not "
             "cell-attributable: the weekly fixed bill is sized by the hub's "
             "peak day, so what one area contributes to it depends on when "
             "the rest of the hub peaks. Panels (c)-(e) therefore aggregate "
             "the per-area attribution to the hub before taking a saving. "
             + confound + H.COST_MODEL_TEXT,
             ha="center", va="bottom", fontsize=9, wrap=True)
    return save(fig, out, "supp_fig6b_operator_lens_v2"), comp

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
                 tables: Path) -> dict:
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
                        f"{H.LENS_ROUTING} on the {H.PLAN_WORD[1]} plan, "
                        f"{H.LENS_OPERATOR} on the {H.PLAN_WORD[2]} plan. "
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
               caption=(f"Fleet diagnostics of the {H.PLAN_WORD[2]} "
                        r"plan: "
                        r"achieved $\sum_h$ hub peak against the "
                        r"flat-profile bound $\sum_h \lceil "
                        r"\mathrm{vehicle\ days}_h / 6 \rceil$; the gap is "
                        r"priced at 1\,134.90~\euro\ per peak vehicle and "
                        r"week."),
               label="tab:fleet_diagnostics")
    return written


ABLATION_POINTS = [(0.0, 1.0), (0.25, 1.0), (0.5, 1.0), (0.0, 0.6)]


# ─────────────────────────────────────────────────────────────────────────
# Lens frames: per CELL for the routing lens, per HUB for the operator lens
# ─────────────────────────────────────────────────────────────────────────
def _lens_frames(pc: pd.DataFrame, struct: pd.DataFrame,
                 knees_r: pd.DataFrame, knees_o: pd.DataFrame
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(cells at the routing P*, hubs at the operator P*)``.

    Each LSP is taken at the knee of the lens the frame is drawn in -- the
    knee is lens-dependent (Task 13 §4) and pairing a lens with the other
    lens's P* is the mixture the amendment forbids.  Carrier class comes
    from the SAME knee table as the P*, so class and operating point can
    never come from different lenses either.
    """
    pcs = H.per_cell_savings(pc)
    cells = pcs.merge(struct, on=["provider", "plz"], how="left")
    assert cells.hub_dist_km.notna().all(), (
        "per-cell/structural join left unmatched cells -- the checkpoint "
        "and the per-cell table disagree on the (provider, plz) universe")

    hubs = H.hub_lens(pc)
    # a hub's region type = the type its parcels are in (parcel-dominant),
    # so the label describes where the hub's work is, not where its
    # smallest cell is
    P0 = float(sorted(cells.penalty.unique())[0])
    rt = (cells[np.isclose(cells.share_willing, 0.0)
                & np.isclose(cells.penalty, P0)
                & (cells.plan == "stage1")]
          .groupby(["provider", "hub", "raumtyp_3"], as_index=False)
          .cell_parcels_week.sum()
          .sort_values("cell_parcels_week", ascending=False)
          .drop_duplicates(["provider", "hub"])[["provider", "hub",
                                                 "raumtyp_3"]])
    hubs = hubs.merge(rt, on=["provider", "hub"], how="left")
    assert hubs.raumtyp_3.notna().all(), "a hub has no region type"

    cls_r = dict(zip(knees_r.provider, knees_r.carrier_class))
    cls_o = dict(zip(knees_o.provider, knees_o.carrier_class))
    cells["carrier_class"] = cells.provider.map(cls_r)
    hubs["carrier_class"] = hubs.provider.map(cls_o)
    assert cells.carrier_class.notna().all()
    assert hubs.carrier_class.notna().all()
    return _at_own_pstar(cells, knees_r), _at_own_pstar(hubs, knees_o)


# ─────────────────────────────────────────────────────────────────────────
# Tables that need the per-cell table, the head usage or the validation
# ─────────────────────────────────────────────────────────────────────────
def write_per_cell_tables(cells_r: pd.DataFrame, hubs_o: pd.DataFrame,
                          tables: Path) -> dict:
    """The numbers behind fig 6 (b)-(f) and fig 6b (c)-(e), as CSV.

    Median AND the interquartile range, in euro AND in percent, per
    (feature, bucket, theta, plan) -- so a caption can quote a panel without
    re-deriving it, and so the percent version (which the submitted figure
    showed) is available next to the euro version this one plots.
    """
    tables.mkdir(parents=True, exist_ok=True)
    HUB_CLASSES = dict(edges=[0.5, 1.5, 4.5, 12.5, 1e9],
                       labs=["1", "2-4", "5-12", ">=13"])
    # (feature, how to bucket it) -- "q" = own quartiles, "hub" = the
    # discrete hub-size classes, None = the feature IS the class
    CELL_SPECS = [("carrier_class", None), ("raumtyp_3", None),
                  ("hub_dist_km", "q"), ("area_km2", "q"),
                  ("parcels_per_stop", "q")]
    HUB_SPECS = [("carrier_class", None), ("raumtyp_3", None),
                 ("n_cells", "hub")]
    out = []
    for level, frame, lens, values, specs in (
            ("cell", cells_r, H.LENS_ROUTING,
             ("saving_eur", "saving_pct"), CELL_SPECS),
            ("hub", hubs_o, H.LENS_OPERATOR,
             ("operator_saving_eur", "operator_saving_pct"), HUB_SPECS)):
        for feat, mode in specs:
            if mode is None:
                d = frame[frame[feat].notna()].copy()
                d["bucket"] = d[feat]
            else:
                d, _order = _bucketise(
                    frame, feat, **(HUB_CLASSES if mode == "hub" else {}))
            # the plan must be a key, not an average: median_by_bucket gets
            # one plan at a time
            for plan in sorted(d.plan.unique()):
                dp = d[d.plan == plan]
                for value in values:
                    agg = H.median_by_bucket(dp, value, "bucket")
                    out.append(agg.assign(level=level, lens=lens,
                                          feature=feat, plan=plan,
                                          value=value))
    tab = pd.concat(out, ignore_index=True)
    p = tables / "tab_per_cell_structural_v2.csv"
    tab.to_csv(p, index=False)
    print(f"  wrote {p.name} ({len(tab)} rows)")
    return {"per_cell_structural": p}


def write_head_usage(rev: Path, tables: Path) -> dict:
    """Head-priced share of pooled cost and of tours, occurrence-weighted."""
    hu = H.head_usage(rev)
    if hu is None:
        print("  head usage: this grid predates Task 11 "
              f"({rev / H.HEAD_USAGE_NAME} absent) -- no head, nothing to "
              "summarise")
        return {}
    written = {}
    for name, by in (("provider_kind", ("provider", "kind")),
                     ("kind", ("kind",)),
                     ("theta_kind", ("share_willing", "kind"))):
        s = H.head_usage_summary(hu, by=by)
        p = tables / f"tab_head_usage_summary_{name}_v2.csv"
        s.to_csv(p, index=False)
        written[f"head_usage_{name}"] = p
    overall = H.head_usage_summary(hu, by=("head_id",))
    p = tables / "tab_head_usage_summary_v2.csv"
    overall.to_csv(p, index=False)
    written["head_usage"] = p
    pk = H.head_usage_summary(hu, by=("provider", "kind"))
    show = pk[["provider", "kind", "pooled_cost_eur", "head_cost_eur",
               "head_cost_share", "n_groups", "n_multi_cell", "n_head",
               "head_share_of_multi_cell"]]
    print("\nhead-priced share of POOLED cost, occurrence-weighted "
          "(NaN = this provider/kind pooled nothing at all):")
    print(show.to_string(index=False))
    tex = show.rename(columns={
        "provider": "LSP", "kind": "tour kind",
        "pooled_cost_eur": r"pooled [\euro/wk]",
        "head_cost_eur": r"head-priced [\euro/wk]",
        "head_cost_share": "head share of cost",
        "n_groups": "tours", "n_multi_cell": "multi-cell tours",
        "n_head": "head-priced tours",
        "head_share_of_multi_cell": "head share of multi-cell tours"})
    H.to_latex(tex, tables / "tab_head_usage_summary_v2.tex",
               caption=("Certified-support bundle head: what actually priced "
                        "the realised pooled tours, summed over the whole "
                        "grid and divided once (never a mean of ratios). "
                        "A blank share means the LSP realised no pooled tour "
                        "of that kind at all."),
               label="tab:head_usage")
    return written


def write_delta(rev: Path, compare: Path, full: pd.DataFrame,
                costs: pd.DataFrame, tables: Path) -> dict:
    """v5 -> v6 delta at theta = 1: what the bundle head changed.

    Both grids' savings are taken against THEIR OWN theta = 0 baseline: the
    head prices the baseline's pooled small-cell tours too, so v6's baseline
    is not v5's and a saving computed across the two would be meaningless
    (Task 11 review addendum).
    """
    if compare is None:
        print("  no --compare-dir: the grid-to-grid delta table is skipped")
        return {}
    if not (compare / "tab_costs_v2.csv").exists():
        print(f"  --compare-dir {compare} has no tab_costs_v2.csv -- the "
              "delta table is skipped")
        return {}
    Gc = H.RevGrid(compare, require_chosen=True)
    base_c = H.baseline(Gc.costs)
    full_c = H.headline_rows(Gc.costs, Gc.wait, base_c,
                             Gc.n_cells_per_provider())
    a, b = compare.name.replace("revision_2026_08_", ""), \
        rev.name.replace("revision_2026_08_", "")
    d = H.grid_delta(full_c, full, Gc.costs, costs, a, b)
    p = tables / "tab_grid_delta_v2.csv"
    d.to_csv(p, index=False)
    print(f"\ngrid delta at theta=1  ({a} -> {b}):")
    cols = ["penalty", f"pooled_{a}_eur", f"pooled_{b}_eur",
            "pooled_delta_eur", f"head_share_{b}",
            "routing_saving_plan1_pct_delta_pp",
            "operator_saving_plan2_pct_delta_pp",
            "sum_hub_peak_plan2_delta"]
    print(d[cols].to_string(index=False))
    tex = d[cols].rename(columns={
        "penalty": "$P$",
        f"pooled_{a}_eur": rf"pooled {a} [\euro/wk]",
        f"pooled_{b}_eur": rf"pooled {b} [\euro/wk]",
        "pooled_delta_eur": r"$\Delta$ pooled [\euro/wk]",
        f"head_share_{b}": f"head share {b}",
        "routing_saving_plan1_pct_delta_pp": r"$\Delta$ rout.\ sav.\ [pp]",
        "operator_saving_plan2_pct_delta_pp": r"$\Delta$ oper.\ sav.\ [pp]",
        "sum_hub_peak_plan2_delta": r"$\Delta \sum_h$ peak"})
    H.to_latex(tex, tables / "tab_grid_delta_v2.tex",
               caption=(rf"What the certified-support bundle head changed at "
                        rf"$\theta = 1$: grid {a} (head-free) against grid "
                        rf"{b} (head installed). Savings are percentage-point "
                        r"differences, each taken against its OWN grid's "
                        r"$\theta = 0$ baseline -- the head prices the "
                        r"baseline's pooled tours too, so the two baselines "
                        r"differ. " + H.COST_MODEL_CAPTION),
               label="tab:grid_delta")
    return {"delta": p}


def write_co2(rev: Path, tables: Path) -> dict:
    """CO2 and vehicle-km from the grid's OWN validated route kilometres."""
    try:
        t = H.co2_table(rev)
    except H.Co2Unavailable as exc:
        print(f"\nCO2 table (Kompendium 38.6): REFUSED -- {exc}")
        return {}
    p = tables / "tab_co2_km_v2.csv"
    t.to_csv(p, index=False)
    print("\nCO2 / vehicle-km from validated VROOM routes "
          f"({H.CO2_KG_PER_KM} kg/vehicle-km, external assumption):")
    print(t[["plan", "penalty", "share_willing", "n_instances", "km_week",
             "co2_t_week", "vs_least_consolidated_pct",
             "n_flagged"]].to_string(index=False))
    tex = (t[["plan", "penalty", "km_week", "co2_t_week",
              "vs_least_consolidated_pct"]]
           .rename(columns={"plan": "plan", "penalty": "$P$",
                            "km_week": "km/week",
                            "co2_t_week": "CO$_2$ t/week",
                            "vs_least_consolidated_pct":
                                r"vs least consolidated [\%]"}))
    H.to_latex(tex, tables / "tab_co2_km_v2.tex",
               caption=(r"Vehicle kilometres and CO$_2$ at the validated "
                        r"operating points, from the actual VROOM routes of "
                        r"this grid. CO$_2$ factor "
                        f"{H.CO2_KG_PER_KM} kg/vehicle-km (external "
                        r"assumption, not a model result). The reference is "
                        r"the least consolidated \emph{validated} point of "
                        r"the same plan; there is no VROOM baseline solve of "
                        r"daily delivery on this allocation."),
               label="tab:co2_km")
    return {"co2": p}



def _ablation_row(variant: str, rev_dir: Path, base: dict, *,
                  routing_col: str, reconstruct: bool) -> pd.DataFrame:
    """One stage-2 variant at the four ablation points, or a reason why not.

    Every table this row needs is existence-checked up front, so a missing
    or half-written ablation directory reports ``DIR_MISSING`` next to the
    other variants instead of aborting the whole table with a traceback.
    """
    needed = ["tab_costs_v2.csv", "tab_fleet_per_hub_v2.csv",
              "tab_wait_v2.csv"]
    absent = [n for n in needed if not (rev_dir / n).exists()]
    if absent:
        return pd.DataFrame([dict(variant=variant, penalty=P,
                                  share_willing=th,
                                  status=f"DIR_MISSING ({', '.join(absent)})")
                             for P, th in ABLATION_POINTS])
    costs = pd.read_csv(rev_dir / "tab_costs_v2.csv")
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
        # Same cross-check the v5 path runs: the fleet CSV must describe the
        # plan whose cost we are reading. A pre-6e run (run 2) carries no
        # sum_hub_peak column, so there the pairing rests on the run's own
        # stage ordering -- see reconstruct_operator_cost's docstring.
        if "sum_hub_peak" in c.columns:
            assert abs(hp - float(c.sum_hub_peak.sum())) < 1e-6, (
                f"{variant} (P={P}, theta={th}): fleet-derived hub peak "
                f"{hp:.0f} != sum_hub_peak {c.sum_hub_peak.sum():.0f} -- "
                "the fleet CSV is not at the plan this row reports")
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
# CO2 (Kompendium 38.6) -- REGENERATED from this grid's own validated
# route kilometres by H.co2_table(); H.co2_decision() says whether that
# is possible here and REFUSES otherwise.  There is deliberately no
# carry-over path: the submitted table describes a different grid, a
# different plan and a different cost path (Task 13B).


# ─────────────────────────────────────────────────────────────────────────
# The ACCEPTED paper figures: the frozen builders, on tables 74_
# adapts from this grid.  Their layout is the submitted one; only the
# numbers change.  Everything this module renders itself is
# SUPPLEMENTARY (``supp_`` prefix) and 71_ never syncs it into a main
# figure slot.
# ─────────────────────────────────────────────────────────────────────────
def _render_accepted_figures(rev: Path, express_allocation: str
                             ) -> list[Path]:
    import importlib.util
    path = Path(__file__).resolve().parent / "74_v2_to_legacy_tables.py"
    spec = importlib.util.spec_from_file_location("_legacy74", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    print(chr(10) + "--- accepted paper figures (74_ adapter + the "
          "three frozen builders) ---")
    produced, meta = mod.build_and_render(
        rev, express_allocation=express_allocation, do_render=True)
    bt = meta["base_total_eur"]
    cv = meta["baseline_cv_exact"]
    print(f"  grid baseline {bt:,.2f} EUR/wk, Mo-Sa fleet CV "
          f"{cv:.4f}; {len(produced)} file(s) copied into figures/")
    return produced


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rev-dir", default=str(DEFAULT_REV),
                    help="v5-schema grid directory (default: %(default)s)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-tables", action="store_true")
    ap.add_argument("--compare-dir", default=str(DEFAULT_COMPARE),
                    help="the OTHER grid for the delta table (default: "
                         "%(default)s); skipped when it is the same "
                         "directory as --rev-dir or does not exist")
    ap.add_argument("--no-legacy", action="store_true",
                    help="skip the accepted paper figures (74_ + the three "
                         "frozen builders). The supplementary two-lens "
                         "figures are rendered either way")
    ap.add_argument("--express-allocation", default="per-tour",
                    choices=("per-tour", "dd-only"),
                    help="how 74_ hands a cell its express residual to "
                         "the frozen fig-6 builder; see 74_'s docstring")
    ap.add_argument("--allow-missing-per-cell", action="store_true",
                    help="downgrade a missing tab_per_cell_costs_v2.csv "
                         "from a refusal to a skip. fig 6 (b)-(f) and the "
                         "operator-lens companion are then NOT rendered -- "
                         "there is no substitute for them")
    args = ap.parse_args(argv)

    rev = Path(args.rev_dir)
    if not rev.is_absolute():
        rev = (ROOT / rev).resolve()
    # Exported so that anything this process spawns (and any later
    # in-process import of _stage3_common) sees the same grid.  It does NOT
    # reach 30_/31_/32_/40_: they are separate processes and this module
    # never invokes them -- run them with REV_DIR set in the shell instead.
    os.environ["REV_DIR"] = str(rev)
    figures, tables = rev / "figures", rev / "tables"
    compare = Path(args.compare_dir) if args.compare_dir else None
    if compare is not None:
        if not compare.is_absolute():
            compare = (ROOT / compare).resolve()
        if compare == rev:
            compare = None          # a grid is not its own reference

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

    # ── per-cell euro costs (72_): fig 6 (b)-(f) and the operator-lens
    #    companion have no substitute, so a missing table refuses by
    #    default and only an explicit flag downgrades it to a skip.
    struct_all = _structural_facts()
    cells_r = hubs_o = pc = None
    try:
        pc = H.load_per_cell(rev)
    except H.PerCellMissing as exc:
        if not args.allow_missing_per_cell:
            print(f"\nREFUSED -- {exc}")
            return 2
        print(f"\nWARNING: {exc}\n  --allow-missing-per-cell was passed: "
              "fig 6, fig 6b and the per-cell tables are SKIPPED.")
    if pc is not None:
        deltas = H.check_per_cell_against_grid(pc, G.costs)
        print(f"\nper-cell identity gate: {len(deltas)} (P, theta, provider, "
              "plan) group(s) re-checked from the written CSVs, worst "
              f"|delta| cost {deltas.cost_delta.abs().max():.3e} EUR / "
              f"vehicle-days {deltas.vd_delta.abs().max():.3e} / hub peak "
              f"{deltas.peak_delta.abs().max():.3e}")
        cells_r, hubs_o = _lens_frames(pc, struct_all, knees_r, knees_o)

    if not args.no_figures:
        print("\n--- figures ---")
        sizes = H.schedule_sizes()
        mix1 = H.freq_mix(G.chosen, "schedule_idx_stage1", sizes)
        mix2 = H.freq_mix(G.chosen, "schedule_idx_balanced", sizes)
        norms = _shared_norms(full)
        produced: list[Path] = []
        produced += fig5(full, base, base_cv, figures, norms)
        produced += fig5b(full, base, figures, norms)
        produced += fig4(mix1, mix2, figures)
        produced += fig4b(full, figures)

        cf = G.chosen[np.isclose(G.chosen.penalty, 0.0)].copy()
        cf["plz"] = cf.plz.astype(str)
        cf["size"] = sizes[cf.schedule_idx_balanced.values.astype(int)]
        cf = cf.merge(struct_all, on=["provider", "plz"], how="left")
        assert cf.hub_dist_km.notna().all(), (
            "structural join left unmatched cells -- checkpoint and grid "
            "disagree on the (provider, plz) universe")
        if cells_r is not None:
            paths, comp = fig6(full, cells_r, figures)
            produced += paths
            paths, comp_b = fig6b(full, knees_r, knees_o, hubs_o, cf,
                                  figures)
            produced += paths
            comp = pd.concat([comp.assign(figure="fig6"),
                              comp_b.assign(figure="fig6b")],
                             ignore_index=True)
            print("\nfig 6 / 6b bucket composition "
                  "(a single-carrier bucket cannot separate structure from "
                  "carrier identity):")
            print(comp.to_string(index=False))
            comp.to_csv(figures / "fig6_bucket_composition.csv", index=False)

        if not args.no_legacy:
            produced += _render_accepted_figures(
                rev, args.express_allocation)

        mpath = H.write_manifest(figures, rev, ROOT, produced)
        doc = H.read_manifest(mpath)
        print(f"\nwrote {mpath.relative_to(ROOT)}  "
              f"(rev_dir={Path(doc['rev_dir']).name}, "
              f"git={doc['git_head'][:12]}, {doc['rendered_utc']}, "
              f"{len(doc['figures'])} figure files)")
        stale = H.check_manifest_fresh(doc, figures)
        assert not stale, (
            "the render is already stale against its own inputs: "
            + "; ".join(stale))

    if not args.no_tables:
        print("\n--- tables ---")
        write_tables(full, base, knees_r, knees_o, tables)
        write_ablation(rev, base, G.costs, tables)
        if cells_r is not None:
            write_per_cell_tables(cells_r, hubs_o, tables)
        write_head_usage(rev, tables)
        write_delta(rev, compare, full, G.costs, tables)
        write_co2(rev, tables)
        print("\nwritten to " + str(tables.relative_to(ROOT)) + ":")
        for f in sorted(tables.iterdir()):
            print(f"  {f.name:<42} {f.stat().st_size:>9,d} B")

    print("\nCO2 table (Kompendium 38.6): " + H.co2_decision(rev))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
