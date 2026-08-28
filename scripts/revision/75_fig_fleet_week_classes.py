"""Supplementary figure: Mon-Sat vehicle profile per carrier class and plan.

Rows are the three carrier classes of the paper's routing-lens taxonomy
(Fig. 6b; read from ``tab_pstar_knees_v2.csv``, never hard-coded) plus the
seven-LSP system; columns are the validated operating points P in
{0, 0.25, 0.5, 0.75} at theta = 1.  Every panel shows three weekly profiles:

* **baseline** -- daily delivery, the (P, theta) = (0, 0) row of the fleet table;
* **routing-optimal plan (stage 1)** -- recomputed from ``schedule_idx_stage1``
  with the partition-aware fleet counter (vehicle counts are price-independent:
  ceil(parcels / capacity) per own tour or pooled group), because the grid
  runner only records the FINAL plan's per-hub-day fleet;
* **operator-polished plan (stage 2)** -- the recorded fleet table, gated
  against the same recomputation at ``schedule_idx_balanced``.

Vehicles are **indexed to the class's baseline weekly-mean fleet (= 100)** so
the four rows share one y-axis and the classes stay comparable; the absolute
fleet of an individual LSP cannot be read back from the figure or the table
(the point of aggregating to classes).  Dotted horizontals are the fleet the
operator has to keep -- the sum of the weekly per-depot maxima, the size that
the operator lens bills -- again as an index of the baseline value.

Colours come from ``scripts/presentation/_style.py`` only.  The three plan
colours were checked with the dataviz palette validator: normal-vision and
CVD separation pass for every adjacent pair; line style and marker shape are
the secondary encoding.

Outputs (results/revision_2026_08_v6/):
  figures/supp_fig7_fleet_week_classes.{pdf,png}
  tables/tab_fleet_week_classes_v2.csv   class x P x plan x day (index + abs.
                                          only for the class aggregates)

Gates (fail loud, Kompendium §38.2 spirit):
  G1  baseline profile identical for every (P, theta = 0) row of the fleet table
  G2  recomputed stage-2 profile == recorded fleet table, per provider and P
  G3  system-wide stage-2 profile == tab_grid_full_v2 sys_Mon..sys_Sat, per P
  G4  system-wide baseline profile == the (0, 0) sys_* row of the grid
  G5  every provider carries exactly one routing-lens class
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "revision"))
sys.path.insert(0, str(ROOT / "scripts" / "presentation"))
sys.path.insert(0, str(ROOT / "src"))

import _stage3_common as C  # noqa: E402
import _style as S  # noqa: E402  -- the ONLY colour source
from _figs_tables_v2 import PLAN1, PLAN2, WEEKDAYS  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REV = ROOT / "results" / "revision_2026_08_v6"
FLEET_CSV = REV / "tab_fleet_per_hub_v2.csv"
CHOSEN_CSV = REV / "_tab_chosen_v2.csv"
GRID_CSV = REV / "tables" / "tab_grid_full_v2.csv"
KNEES_CSV = REV / "tables" / "tab_pstar_knees_v2.csv"
FIG_OUT = REV / "figures" / "supp_fig7_fleet_week_classes"
TAB_OUT = REV / "tables" / "tab_fleet_week_classes_v2.csv"

THETA = 1.0
P_COLS = (0.0, 0.25, 0.5, 0.75)
DAY_LBL = ["Mo", "Tu", "We", "Th", "Fr", "Sa"]
SYSTEM_LABEL = "All seven LSPs"

# plan colours (values from _style.py; see module docstring for the check)
PLAN_COLOR = {"base": S.INK_SOFT, 1: S.FREQ[6], 2: S.FREQ[2]}
PLAN_LABEL = {"base": "baseline (daily delivery)", 1: PLAN1, 2: PLAN2}


# ─────────────────────────────────────────────────────────────────────────
# data
# ─────────────────────────────────────────────────────────────────────────
def _profile(df: pd.DataFrame) -> np.ndarray:
    """(6,) vehicles per weekday summed over the frame's hubs."""
    return df.groupby("day")["fleet"].sum().reindex(range(6)).to_numpy(float)


def _sum_hub_peak(df: pd.DataFrame) -> float:
    return float(df.groupby("hub")["fleet"].max().sum())


def load_classes() -> dict[str, str]:
    k = pd.read_csv(KNEES_CSV)
    cls = dict(zip(k["provider"], k["carrier_class_routing"]))
    assert set(cls) == set(C.PROVIDERS), f"G5: knee table providers {set(cls)}"
    assert set(cls.values()) <= set(S.CARRIER), f"G5: unknown class {set(cls.values())}"
    # rows ordered by ascending routing-lens P* (service-bound first)
    order = (k.groupby("carrier_class_routing")["P_star_routing"].mean()
             .sort_values().index.tolist())
    return cls, order


def recompute_profiles(fleet: pd.DataFrame, chosen: pd.DataFrame,
                       p_values: tuple[float, ...]) -> dict:
    """Per provider: baseline profile + per-P stage-1 / stage-2 profiles and
    sum-of-hub-peaks.  Cost matrices depend on theta only, so they are built
    once per provider and every P is evaluated on the same matrices."""
    from batch_delivery.optimization.balancing import _daily_fleet_per_hub
    from batch_delivery.optimization.core import build_cost_matrices_ml
    from batch_delivery.optimization.costs import (
        _hub_delivery_pool_vehicles, _hub_express_vehicles)

    provider_data, optim_data = C.load_checkpoints()
    model = C.load_model()
    ml_prep = C.build_ml_prep(provider_data)
    schedules = C.enumerate_schedules()

    out = {}
    for prov in C.PROVIDERS:
        od, prep = optim_data[prov], ml_prep[prov]
        m = build_cost_matrices_ml(
            od["plz_keys"], od["plz_data"], schedules, model, prov,
            prep["plz_day_coords"], prep["hub_coords_by_plz"],
            fast_share_b2c=C.fs_b2c(THETA), fast_share_b2b=C.fs_b2b(THETA))
        hpl = od["hub_plz_list"]
        pha = np.zeros(len(od["plz_keys"]), dtype=int)
        for hi, h in enumerate(hpl):
            pha[h] = hi
        ec, pc = {}, {}

        def pv(hi, d, chv, _m=m, _hpl=hpl, _ec=ec, _pc=pc):
            return (_hub_express_vehicles(hi, d, chv, _hpl, schedules,
                                          _m["raw_express"], _m, _ec)
                    + _hub_delivery_pool_vehicles(hi, d, chv, _hpl, schedules,
                                                  _m, _pc))

        fb = fleet[(fleet.share_willing == 0) & (fleet.provider == prov)]
        base_profiles = np.stack([_profile(g) for _, g in fb.groupby("penalty")])
        assert np.allclose(base_profiles, base_profiles[0]), f"G1: {prov}"
        base0 = fb[fb.penalty == 0]
        rec = {"base": _profile(base0), "base_peak": _sum_hub_peak(base0),
               "n_hubs": len(hpl), "P": {}}

        keys = [str(k) for k in od["plz_keys"]]
        for P in p_values:
            sub = chosen[np.isclose(chosen.penalty, P)
                         & np.isclose(chosen.share_willing, THETA)
                         & (chosen.provider == prov)].set_index("plz").loc[keys]
            f1 = _daily_fleet_per_hub(
                sub["schedule_idx_stage1"].to_numpy(int), pha, hpl,
                m["veh_3d"], schedules, pool_veh_fn=pv,
                sched_active=m["sched_active"])
            f2 = _daily_fleet_per_hub(
                sub["schedule_idx_balanced"].to_numpy(int), pha, hpl,
                m["veh_3d"], schedules, pool_veh_fn=pv,
                sched_active=m["sched_active"])
            ft = fleet[np.isclose(fleet.penalty, P)
                       & np.isclose(fleet.share_willing, THETA)
                       & (fleet.provider == prov)]
            assert np.allclose(f2.sum(axis=0), _profile(ft)), (
                f"G2: {prov} P={P}: recomputed stage-2 profile != fleet table")
            assert abs(f2.max(axis=1).sum() - _sum_hub_peak(ft)) < 1e-9, (
                f"G2: {prov} P={P}: sum of hub peaks differs")
            rec["P"][P] = {1: f1.sum(axis=0), 2: f2.sum(axis=0),
                           "peak1": float(f1.max(axis=1).sum()),
                           "peak2": float(f2.max(axis=1).sum())}
            print(f"  {prov:7s} P={P:<4g} base {rec['base'].astype(int).tolist()}"
                  f" | plan1 {f1.sum(axis=0).astype(int).tolist()}"
                  f" | plan2 {f2.sum(axis=0).astype(int).tolist()}"
                  f" | sum hub peaks {rec['base_peak']:.0f}/"
                  f"{rec['P'][P]['peak1']:.0f}/{rec['P'][P]['peak2']:.0f}",
                  flush=True)
        del m
        out[prov] = rec
    return out


def aggregate(per_prov: dict, classes: dict[str, str], order: list[str],
              grid: pd.DataFrame) -> pd.DataFrame:
    """Class-level (and system) rows; gates G3/G4 against the grid table."""
    groups = {c: [p for p in C.PROVIDERS if classes[p] == c] for c in order}
    groups[SYSTEM_LABEL] = list(C.PROVIDERS)
    rows = []
    for grp, provs in groups.items():
        base = sum(per_prov[p]["base"] for p in provs)
        base_peak = sum(per_prov[p]["base_peak"] for p in provs)
        ref = base.mean()   # index base: baseline weekly-mean vehicles/day
        n_hubs = sum(per_prov[p]["n_hubs"] for p in provs)
        for P in P_COLS:
            plans = {"base": (base, base_peak)}
            for pl in (1, 2):
                plans[pl] = (sum(per_prov[p]["P"][P][pl] for p in provs),
                             sum(per_prov[p]["P"][P][f"peak{pl}"] for p in provs))
            for pl, (prof, peak) in plans.items():
                rows.append(dict(
                    group=grp, n_lsp=len(provs), n_hubs=n_hubs, penalty=P,
                    share_willing=THETA,
                    plan={"base": "baseline", 1: "plan1_routing_optimal",
                          2: "plan2_operator_polished"}[pl],
                    index_base_veh_per_day=ref,
                    **{f"idx_{d}": 100.0 * prof[i] for i, d in enumerate(WEEKDAYS)},
                    idx_sum_hub_peak=100.0 * peak / ref,
                    cv=float(prof.std(ddof=0) / prof.mean()),
                    **({f"veh_{d}": prof[i] for i, d in enumerate(WEEKDAYS)}
                       | {"sum_hub_peak": peak}),
                ))
    tab = pd.DataFrame(rows)
    for d in WEEKDAYS:
        tab[f"idx_{d}"] = tab[f"idx_{d}"] / tab["index_base_veh_per_day"]

    sysr = tab[tab.group == SYSTEM_LABEL]
    g00 = grid[(grid.penalty == 0) & (grid.share_willing == 0)].iloc[0]
    b = sysr[(sysr.plan == "baseline")].iloc[0]
    assert all(abs(b[f"veh_{d}"] - g00[f"sys_{d}"]) < 1e-9 for d in WEEKDAYS), \
        "G4: system baseline profile != grid (0,0) sys_* row"
    for P in P_COLS:
        g = grid[np.isclose(grid.penalty, P) & np.isclose(grid.share_willing, THETA)].iloc[0]
        r = sysr[np.isclose(sysr.penalty, P) & (sysr.plan == "plan2_operator_polished")].iloc[0]
        assert all(abs(r[f"veh_{d}"] - g[f"sys_{d}"]) < 1e-9 for d in WEEKDAYS), \
            f"G3: system stage-2 profile != grid sys_* at P={P}"
        assert abs(r["sum_hub_peak"] - g["sum_hub_peak_plan2"]) < 1e-9, f"G3 peak P={P}"
    return tab


# ─────────────────────────────────────────────────────────────────────────
# figure
# ─────────────────────────────────────────────────────────────────────────
def draw(tab: pd.DataFrame, order: list[str]) -> plt.Figure:
    S.apply("paper")
    groups = order + [SYSTEM_LABEL]
    nrow, ncol = len(groups), len(P_COLS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 1.55 * nrow + 0.9),
                             sharex=True, sharey=True)
    x = np.arange(6)
    spec = {  # secondary encoding: dash + marker shape, not colour alone
        "baseline": dict(color=PLAN_COLOR["base"], ls="--", lw=1.1, marker="o",
                         ms=3.0, mfc="white", mew=0.9, zorder=2),
        "plan1_routing_optimal": dict(color=PLAN_COLOR[1], ls="-", lw=1.1,
                                      marker="s", ms=2.8, zorder=3),
        "plan2_operator_polished": dict(color=PLAN_COLOR[2], ls="-", lw=1.8,
                                        marker="o", ms=3.2, zorder=4),
    }
    ymax = float(np.ceil(tab[[f"idx_{d}" for d in WEEKDAYS] + ["idx_sum_hub_peak"]]
                         .to_numpy().max() / 10) * 10)
    for i, grp in enumerate(groups):
        for j, P in enumerate(P_COLS):
            ax = axes[i, j]
            sub = tab[(tab.group == grp) & np.isclose(tab.penalty, P)].set_index("plan")
            for plan, kw in spec.items():
                r = sub.loc[plan]
                ax.plot(x, [r[f"idx_{d}"] for d in WEEKDAYS], **kw)
                ax.axhline(r["idx_sum_hub_peak"], color=kw["color"], ls=":",
                           lw=0.8, alpha=0.9, zorder=1)
            pk = [sub.loc[p, "idx_sum_hub_peak"] for p in spec]
            cv = [sub.loc[p, "cv"] for p in spec]
            # bottom-left is always empty: no profile drops below index 40
            ax.text(0.04, 0.04,
                    "Σ peaks " + "→".join(f"{v:.0f}" for v in pk)
                    + "\nCV " + "→".join(f"{v:.2f}".lstrip("0") for v in cv),
                    transform=ax.transAxes, fontsize=6.6, color=S.INK,
                    ha="left", va="bottom", linespacing=1.25)
            ax.grid(axis="y", color=S.GRID, lw=0.5, alpha=0.8, zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels(DAY_LBL, fontsize=7)
            ax.tick_params(axis="x", pad=2)
            ax.tick_params(axis="y", labelsize=8)
            if i == 0:
                ax.set_title(rf"$P = {P:g}$ €/p/d", fontsize=10)
            if j == 0:
                n = int(sub["n_lsp"].iloc[0])
                lbl = grp if grp == SYSTEM_LABEL else f"{grp}\n({n} LSPs)"
                ax.set_ylabel(lbl, fontsize=9)
    axes[0, 0].set_ylim(0, ymax)
    fig.supylabel("Vehicles per weekday, index (class baseline weekly mean = 100)",
                  fontsize=9, x=0.005)
    handles = [Line2D([], [], **{k: v for k, v in spec[p].items() if k != "zorder"},
                      label=PLAN_LABEL[q])
               for p, q in (("baseline", "base"), ("plan1_routing_optimal", 1),
                            ("plan2_operator_polished", 2))]
    handles.append(Line2D([], [], color=S.INK_SOFT, ls=":", lw=0.9,
                          label="Σ peaks = fleet kept for the week "
                                "(Σ weekly depot maxima, index)"))
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, 1.0), handlelength=2.6,
               columnspacing=1.6)
    fig.text(0.5, -0.005,
             rf"$\theta = 1$ (full adoption). Carrier classes: routing-lens knee "
             r"taxonomy (Fig. 6b). Rows share one axis; the routing-optimal plan "
             r"spends the weekday freedom on cheap days, the operator polish "
             r"spends it on a flat week.",
             ha="center", va="top", fontsize=7.6, color=S.INK_SOFT, wrap=True)
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.93))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-table", action="store_true",
                    help="skip the recomputation and redraw from the CSV")
    a = ap.parse_args()

    classes, order = load_classes()
    if a.from_table:
        tab = pd.read_csv(TAB_OUT)
    else:
        fleet = pd.read_csv(FLEET_CSV)
        chosen = pd.read_csv(CHOSEN_CSV, dtype={"plz": str})
        grid = pd.read_csv(GRID_CSV)
        per_prov = recompute_profiles(fleet, chosen, P_COLS)
        tab = aggregate(per_prov, classes, order, grid)
        TAB_OUT.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(TAB_OUT, index=False)
        print(f"wrote {TAB_OUT.relative_to(ROOT)}")

    fig = draw(tab, order)
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    # Matplotlib stamps a /CreationDate into every PDF, so two renders of
    # identical content get different md5s. 70_'s manifest and gate G7 are
    # md5-based (Kompendium §38.8); suppressing it here mirrors 70_'s own
    # convention exactly, so a re-render of unchanged inputs is
    # byte-identical, not merely content-identical.
    meta = {"pdf": {"Title": "supp_fig7 fleet week by carrier class",
                    "CreationDate": None},
            "png": {"Software": None}}
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG_OUT}.{ext}", bbox_inches="tight", metadata=meta[ext])
        print(f"saved {FIG_OUT.relative_to(ROOT)}.{ext}")
    plt.close(fig)

    s = tab[tab.group == SYSTEM_LABEL]
    for P in P_COLS:
        r = s[np.isclose(s.penalty, P)].set_index("plan")
        print(f"system P={P:<4g} CV base/plan1/plan2 = "
              f"{r.loc['baseline','cv']:.3f}/{r.loc['plan1_routing_optimal','cv']:.3f}/"
              f"{r.loc['plan2_operator_polished','cv']:.3f}  kept fleet idx "
              f"{r.loc['baseline','idx_sum_hub_peak']:.0f}/"
              f"{r.loc['plan1_routing_optimal','idx_sum_hub_peak']:.0f}/"
              f"{r.loc['plan2_operator_polished','idx_sum_hub_peak']:.0f}")


if __name__ == "__main__":
    main()
