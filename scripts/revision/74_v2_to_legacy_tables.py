"""74: adapt a v5/v6 grid to the LEGACY schema the frozen builders read.

The paper is accepted, so its figures keep their SUBMITTED layout and only
their numbers change.  ``30_fig5_heatmap_smoothed.py``,
``31_fig6_structural_smoothed.py`` and ``32_fig4_mix.py`` ARE those figures;
their plotting code is not touched here.  What they read is the 2026-07
Stage-3 schema, which a v5/v6 grid does not have -- so this script writes
that schema FROM the v5/v6 tables plus ``72_per_cell_costs_v2.py``'s per-cell
costs, and the builders then render the accepted figures on the new numbers.

The mapping, one line per legacy name
-------------------------------------
=========================== =============================================
legacy name                 v5/v6 source
=========================== =============================================
``*_init`` / stage 1        the **routing-optimal** stage-1 plan
``*_balanced`` / stage 2    the **operator-polished** plan (stage 3 is OFF
                            in v5/v6, so stage 2 IS the final plan)
``*_system_smoothed`` /     the same operator plan, in the **routing lens**
``*_stage3``                (``cost_stage2_eur``)
wait / fleet / freq mix     the operator plan
per-PLZ euro                ``72_``'s per-cell costs, routing lens
``tab_pstar_knees``         the **routing-lens** knees (submission classes)
=========================== =============================================

Two roots, because the builders read from two: ``<out>/run`` replaces
``_stage3_common.RUN_DIR`` (env ``REV_RUN_DIR``) and ``<out>/rev`` replaces
``OUT_DIR`` (env ``REV_DIR``).  ``REV_BASE_TOTAL`` and ``REV_BASELINE_CV``
carry the grid's OWN baseline: the bundle head prices the theta = 0 pooled
tours too, so a v6 saving must never be taken against the 2026-07
denominator, and the fleet CV baseline is 0.139 here, not the 2026-07 0.135.

Express allocation
------------------
The submitted fig. 6 gives a cell ``dd_cost + weekly_parcels/sum * express``:
the express residual split over a provider's cells in proportion to their
parcels.  ``72_`` splits each realised express TOUR over its own members.
``--express-allocation``:

* ``per-tour`` (default) -- ``dd_cost_system_smoothed`` is written as
  ``cell_cost - provider-proportional express share``, so when the frozen
  builder adds that share back, its per-cell euro equals ``72_``'s exactly.
  This is the mapping the task brief asks for.
* ``dd-only`` -- ``dd_cost_system_smoothed`` is the cell's own + pooled
  delivery cost and the builder's provider-proportional allocation stands,
  i.e. the submitted figure's own rule on new numbers.

Both sum to the same provider total, and both are gated against it.

Columns with no v5/v6 source are written as **NaN**, never guessed, and are
listed on every run (:data:`NO_SOURCE`).  None of them is read by fig. 4/5/6.

Usage
-----
    python scripts/revision/74_v2_to_legacy_tables.py \\
        --rev-dir results/revision_2026_08_v6 --render

``--render`` runs the three frozen builders in-process with the environment
above and copies their figures into ``<rev>/figures/`` under their canonical
stems, where ``70_``'s manifest and ``71_``'s sync find them.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import _figs_tables_v2 as H  # noqa: E402

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
N_DAYS = 6

# ── the frozen schemas, column for column ────────────────────────────────
# Taken from the real 2026-07 / path2_2026_05_29 files.  Every one of these
# columns is written; ``assert_schema`` refuses a frame missing one or
# carrying an extra, so a builder can never meet a half-adapted table.
SCHEMA = {
    "tab_balancing_summary.csv": [
        "penalty", "share_willing", "provider", "n_plz", "init_cost_eur",
        "balanced_cost_eur", "cost_delta_eur", "cost_delta_pct",
        "imbalance_before", "imbalance_after", "imbalance_reduction_pct",
        "max_fleet_before", "max_fleet_after", "total_routes_before",
        "total_routes_after", "swaps_made"],
    "tab_chosen_schedules.csv": [
        "penalty", "share_willing", "provider", "plz", "weekly_parcels",
        "schedule_idx_init", "schedule_idx_balanced", "schedule_size_init",
        "schedule_size_balanced", "weekdays_init", "weekdays_balanced",
        "avg_wait_d_init", "avg_wait_d_balanced", "dd_cost_init",
        "dd_cost_balanced", "veh_init", "veh_balanced"],
    "_tab_chosen_with_system_smoothing.csv": [
        "penalty", "share_willing", "provider", "plz",
        "schedule_idx_system_smoothed", "schedule_size_system_smoothed",
        "weekdays_system_smoothed", "avg_wait_d_system_smoothed",
        "dd_cost_system_smoothed"],
    "tab_costs_smoothed.csv": [
        "penalty", "share_willing", "provider", "dd_cost_stage3_eur",
        "express_stage3_eur", "total_stage3_eur"],
    "tab_wait_fixed.csv": [
        "penalty", "share_willing", "wait_old", "wait_fixed",
        "willing_parcels", "total_parcels"],
    "tab_wait_smoothed.csv": [
        "penalty", "share_willing", "avg_wait_d_stage3"],
    "tab_fleet_per_hub_fixed.csv": [
        "penalty", "share_willing", "provider", "hub", "day", "dd_veh",
        "expr_veh_old", "expr_veh_fixed", "fleet_old", "fleet_fixed"],
    "tab_fleet_per_hub_smoothed.csv": [
        "penalty", "share_willing", "provider", "hub", "day",
        "fleet_stage2", "fleet_stage3"],
    "tab_express_smoothed.csv": [
        "penalty", "share_willing", "provider", "hub", "day",
        "express_cost_stage3_eur"],
    "tab_pstar_knees_smoothed.csv": [
        "provider", "P_star", "saving_pct", "wait_d", "chord_dist"],
    "tab_per_plz_costs_theta1.csv": [
        "penalty", "share_willing", "provider", "plz", "hub",
        "schedule_size_stage3", "schedule_days_stage3", "avg_wait_d_stage3",
        "dd_cost_stage3_eur", "dd_cost_baseline_eur", "saving_abs_eur",
        "saving_pct", "weekly_parcels", "b2c_share", "area_km2",
        "hub_dist_km", "n_stops_per_day", "demand_per_area"],
}

#: Which of the two roots each file belongs in.
IN_RUN_DIR = {"tab_balancing_summary.csv", "tab_chosen_schedules.csv",
              "_tab_chosen_with_system_smoothing.csv"}

#: Legacy columns with NO v5/v6 source.  Written as NaN, never guessed; the
#: reason is printed on every run and belongs in the task report.  None of
#: them is read by fig. 4, 5 or 6.
NO_SOURCE = {
    ("tab_balancing_summary.csv", "max_fleet_before"):
        "the v5/v6 fleet table is written at the FINAL plan only, so the "
        "stage-1 plan has no per-hub-day fleet to take a maximum over. Its "
        "TOTALS (sum_hub_peak_before, vehicle_days_before) are carried.",
    ("tab_fleet_per_hub_fixed.csv", "expr_veh_old"):
        "the pre-2026-08-25 express accounting (>= 1 vehicle per "
        "non-delivering cell) does not exist in the v5/v6 schema, which is "
        "partition-exact by construction.",
    ("tab_fleet_per_hub_fixed.csv", "fleet_old"):
        "same: there is no pre-fix fleet in a grid that never computed one.",
    ("tab_express_smoothed.csv", "day"):
        "72_'s per-cell express attribution is weekly, so the express cost "
        "resolves per (provider, hub) but not per DAY. 31_ sums this file "
        "to provider level, so no figure needs the day.",
}

CANONICAL_FIGURES = ("fig4_SM_mix_pct_8P", "fig5_grid_heatmap_6_smoothed",
                     "fig6_structural_grid_6_smoothed")
BUILDERS = ("32_fig4_mix.py", "30_fig5_heatmap_smoothed.py",
            "31_fig6_structural_smoothed.py")

#: the runner's tolerance semantics (61_grid_run_v2._tol): a tenth of a
#: cent absolute floor, 1e-9 relative for scale safety.
ABS_TOL = 1e-3
REL_TOL = 1e-9


def _tol(ref: float) -> float:
    return max(ABS_TOL, REL_TOL * abs(ref))


KEY = ["penalty", "share_willing", "provider"]


def provider_express_share(plan_df: pd.DataFrame) -> pd.Series:
    """The provider-proportional express share the frozen builder adds.

    31_ computes ``cell_express = weekly_parcels / sum(weekly_parcels) *
    sum(express_cost)`` per (P, theta, provider).  This is that expression,
    so the adapter can subtract exactly what the builder will add back.
    """
    tot = plan_df.groupby(KEY).express_share_eur.transform("sum")
    par = plan_df.groupby(KEY).cell_parcels_week.transform("sum")
    return tot * plan_df.cell_parcels_week / par.clip(lower=1.0)


def dd_cost_column(plan_df: pd.DataFrame, mode: str) -> pd.Series:
    """The legacy ``dd_cost_*`` column of one plan.

    ``per-tour``  -- ``cell_cost - provider-proportional express share``,
    so the builder's own allocation adds the share back and its per-cell
    euro equals 72_'s per-TOUR attribution exactly.
    ``dd-only``   -- the cell's own + pooled delivery cost, i.e. the
    submitted figure's own rule on the new numbers.

    Both sum, together with the express total, to the plan's routing total.
    """
    if mode == "per-tour":
        out = plan_df.cell_cost_eur - provider_express_share(plan_df)
    elif mode == "dd-only":
        out = plan_df.own_cost_eur + plan_df.pool_share_eur
    else:
        raise SystemExit(f"unknown express allocation {mode!r}")
    assert (out >= -1e-6).all(), (
        f"{mode}: a cell's delivery-day cost came out negative (min "
        f"{out.min():.3f} EUR) -- the express allocation cannot be "
        "reconstructed this way for this grid")
    return out


def assert_schema(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Refuse a frame that is not exactly the legacy schema, in order."""
    want = SCHEMA[name]
    have = list(df.columns)
    assert have == want, (
        f"{name}: column mismatch.\n  want {want}\n  have {have}\n"
        f"  missing {[c for c in want if c not in have]}\n"
        f"  extra   {[c for c in have if c not in want]}")
    return df


def _gate(name: str, got: float, want: float, label: str) -> float:
    delta = float(got) - float(want)
    assert abs(delta) <= _tol(want), (
        f"IDENTITY GATE FAILED [{name}] {label}: {got:.6f} != {want:.6f} "
        f"(delta {delta:.3e}, tol {_tol(want):.3e})")
    return delta


# ─────────────────────────────────────────────────────────────────────────
def grid_baseline(rev: Path) -> dict:
    """This grid's OWN theta = 0 baseline -- pure pandas, no import of C.

    Computed BEFORE ``_stage3_common`` is imported, because that module
    freezes ``BASE_TOTAL``, ``BASELINE_CV`` and both roots at import time and
    the frozen builders read all four from there.
    """
    costs = pd.read_csv(rev / "tab_costs_v2.csv")
    fleet = pd.read_csv(rev / "tab_fleet_per_hub_v2.csv")
    b0 = costs[np.isclose(costs.share_willing, 0.0)]
    assert len(b0), f"{rev}: no theta=0 rows -- the grid has no baseline"
    P0 = sorted(b0.penalty.unique())[0]
    base_total = float(b0[np.isclose(b0.penalty, P0)].cost_stage2_eur.sum())
    f0 = fleet[np.isclose(fleet.share_willing, 0.0)]
    day = f0.groupby(["penalty", "day"]).fleet.sum().groupby("day").mean()
    v = np.array([day.loc[d] for d in range(N_DAYS)])
    cv = float(v.std() / v.mean())
    return dict(base_total_eur=base_total, baseline_cv=round(cv, 3),
                baseline_cv_exact=cv, baseline_peak=float(v.max()),
                baseline_total=float(v.sum()))


def _set_env(out: Path, base: dict) -> None:
    """Point the frozen builders at the adapted tables and this grid's own
    baseline.  MUST run before ``_stage3_common`` is imported anywhere in
    this process -- that module freezes all four values at import time.
    """
    assert "_stage3_common" not in sys.modules, (
        "_stage3_common was already imported, so REV_DIR / REV_RUN_DIR / "
        "REV_BASE_TOTAL / REV_BASELINE_CV are frozen at their defaults and "
        "the frozen builders would read the wrong roots")
    os.environ["REV_RUN_DIR"] = str((out / "run").resolve())
    os.environ["REV_DIR"] = str((out / "rev").resolve())
    os.environ["REV_BASE_TOTAL"] = repr(float(base["base_total_eur"]))
    os.environ["REV_BASELINE_CV"] = repr(float(base["baseline_cv"]))
    # v5/v6 stage 2 is frequency-FREE, so the delivery-day count is NOT
    # preserved between the plotted (stage-1) plan and the final plan.
    # 32_'s invariance gate is therefore REPORTED, not asserted -- and the
    # fig-4 caption must not claim invariance any more.
    os.environ["REV_FREQ_INVARIANT"] = "0"


def _plz_facts(ckpt: Path) -> pd.DataFrame:
    """area, hub distance, stops and B2C share per (provider, plz)."""
    import pickle
    with open(ckpt / "04_optim_prep.pkl", "rb") as fh:
        od = pickle.load(fh)["optimization_data"]
    rows = []
    for prov, d in od.items():
        for plz, meta in d["plz_data"].items():
            b2c = float(sum(meta["b2c"].values()))
            b2b = float(sum(meta["b2b"].values()))
            rows.append(dict(provider=prov, plz=str(plz),
                             area_km2=float(meta["area_km2"]),
                             hub_dist_km=float(meta["hub_dist_km"]),
                             n_stops_per_day=float(meta["n_stops_per_day"]),
                             b2c_share=b2c / max(1.0, b2c + b2b)))
    return pd.DataFrame(rows)


def build(rev: Path, express_allocation: str) -> tuple[dict, dict]:
    """Build every legacy frame.  Returns ``(frames, meta)``."""
    import _stage3_common as C          # imported AFTER _set_env

    costs = pd.read_csv(rev / "tab_costs_v2.csv")
    wait = pd.read_csv(rev / "tab_wait_v2.csv")
    fleet = pd.read_csv(rev / "tab_fleet_per_hub_v2.csv")
    pc = H.load_per_cell(rev)
    H.check_per_cell_against_grid(pc, costs)

    schedules = C.enumerate_schedules()
    sched_days = [sorted(s) for s in schedules]
    sizes = np.array([len(s) for s in schedules])
    waits = np.array([C.avg_wait_days(sorted(s)) for s in schedules])
    weekdays = [",".join(WEEKDAYS[d] for d in ds) for ds in sched_days]
    daystr = ["".join(str(d) for d in ds) for ds in sched_days]

    key = KEY

    pcs = pc.copy()
    pcs["dd_cost"] = np.nan
    for plan in ("stage1", "balanced"):
        m = pcs.plan == plan
        pcs.loc[m, "dd_cost"] = dd_cost_column(
            pcs[m], express_allocation).values
    d1 = pcs[pcs.plan == "stage1"].set_index([*key, "plz"])
    d2 = pcs[pcs.plan == "balanced"].set_index([*key, "plz"])
    assert set(d1.index) == set(d2.index), "the two plans differ in cells"
    d2 = d2.reindex(d1.index)          # align; never trust the file order
    assert d2.cell_cost_eur.notna().all()

    frames: dict[str, pd.DataFrame] = {}
    gates: list[float] = []

    # ── tab_chosen_schedules.csv ────────────────────────────────────────
    idx = d1.index
    ch = pd.DataFrame({
        "penalty": idx.get_level_values(0),
        "share_willing": idx.get_level_values(1),
        "provider": idx.get_level_values(2),
        "plz": idx.get_level_values(3),
        "weekly_parcels": d1.cell_parcels_week.values,
        "schedule_idx_init": d1.schedule_idx.values.astype(int),
        "schedule_idx_balanced": d2.schedule_idx.values.astype(int),
    })
    ch["schedule_size_init"] = sizes[ch.schedule_idx_init.values]
    ch["schedule_size_balanced"] = sizes[ch.schedule_idx_balanced.values]
    ch["weekdays_init"] = [weekdays[i] for i in ch.schedule_idx_init]
    ch["weekdays_balanced"] = [weekdays[i] for i in ch.schedule_idx_balanced]
    ch["avg_wait_d_init"] = waits[ch.schedule_idx_init.values]
    ch["avg_wait_d_balanced"] = waits[ch.schedule_idx_balanced.values]
    ch["dd_cost_init"] = d1.dd_cost.values
    ch["dd_cost_balanced"] = d2.dd_cost.values
    ch["veh_init"] = d1.veh_days_share.values
    ch["veh_balanced"] = d2.veh_days_share.values
    frames["tab_chosen_schedules.csv"] = assert_schema(
        "tab_chosen_schedules.csv", ch[SCHEMA["tab_chosen_schedules.csv"]])

    # ── _tab_chosen_with_system_smoothing.csv ───────────────────────────
    sm = pd.DataFrame({
        "penalty": idx.get_level_values(0),
        "share_willing": idx.get_level_values(1),
        "provider": idx.get_level_values(2),
        "plz": idx.get_level_values(3),
        "schedule_idx_system_smoothed": d2.schedule_idx.values.astype(int),
    })
    sm["schedule_size_system_smoothed"] = sizes[
        sm.schedule_idx_system_smoothed.values]
    sm["weekdays_system_smoothed"] = [
        weekdays[i] for i in sm.schedule_idx_system_smoothed]
    sm["avg_wait_d_system_smoothed"] = waits[
        sm.schedule_idx_system_smoothed.values]
    sm["dd_cost_system_smoothed"] = d2.dd_cost.values
    frames["_tab_chosen_with_system_smoothing.csv"] = assert_schema(
        "_tab_chosen_with_system_smoothing.csv",
        sm[SCHEMA["_tab_chosen_with_system_smoothing.csv"]])

    # ── tab_express_smoothed.csv (per hub; the day is not resolvable) ───
    ex = (pc[pc.plan == "balanced"]
          .groupby([*key, "hub"], as_index=False)
          .express_share_eur.sum()
          .rename(columns={"express_share_eur": "express_cost_stage3_eur"}))
    ex["day"] = np.nan
    frames["tab_express_smoothed.csv"] = assert_schema(
        "tab_express_smoothed.csv", ex[SCHEMA["tab_express_smoothed.csv"]])

    # ── tab_costs_smoothed.csv ──────────────────────────────────────────
    cs = costs[[*key]].copy()
    cs["dd_cost_stage3_eur"] = (costs.dd_cost_eur + costs.pool_cost_eur).values
    cs["express_stage3_eur"] = costs.express_cost_eur.values
    cs["total_stage3_eur"] = costs.cost_stage2_eur.values
    frames["tab_costs_smoothed.csv"] = assert_schema(
        "tab_costs_smoothed.csv", cs[SCHEMA["tab_costs_smoothed.csv"]])
    for _, r in cs.iterrows():
        gates.append(_gate(
            "tab_costs_smoothed", r.dd_cost_stage3_eur + r.express_stage3_eur,
            r.total_stage3_eur,
            f"dd+express==total P={r.penalty} th={r.share_willing} "
            f"{r.provider}"))

    # ── gate: the per-cell delivery-day costs plus the plan's own express
    #    total ARE the grid's routing total, for BOTH plans ──────────────
    ref_of = {"stage1": "cost_stage1_eur", "balanced": "cost_stage2_eur"}
    for plan, dd in (("stage1", d1), ("balanced", d2)):
        g = dd.groupby(level=[0, 1, 2]).agg(dd=("dd_cost", "sum"),
                                            ex=("express_share_eur", "sum"))
        m = costs.set_index(key)[ref_of[plan]]
        for gkey, r in g.iterrows():
            gates.append(_gate("per-cell", r.dd + r.ex, float(m.loc[gkey]),
                               f"{plan} {gkey}"))

    # ── fleet ───────────────────────────────────────────────────────────
    fl = fleet[[*key, "hub", "day"]].copy()
    fl["dd_veh"] = (fleet.dd_single_veh + fleet.dd_pool_veh).values
    fl["expr_veh_old"] = np.nan
    fl["expr_veh_fixed"] = fleet.express_veh.values
    fl["fleet_old"] = np.nan
    fl["fleet_fixed"] = fleet.fleet.values
    frames["tab_fleet_per_hub_fixed.csv"] = assert_schema(
        "tab_fleet_per_hub_fixed.csv",
        fl[SCHEMA["tab_fleet_per_hub_fixed.csv"]])

    # Stage 3 is OFF in v5/v6, so the stage-2 plan IS the final plan and the
    # two legacy fleet columns coincide BY CONSTRUCTION -- not a fill.
    fs = fleet[[*key, "hub", "day"]].copy()
    fs["fleet_stage2"] = fleet.fleet.values
    fs["fleet_stage3"] = fleet.fleet.values
    frames["tab_fleet_per_hub_smoothed.csv"] = assert_schema(
        "tab_fleet_per_hub_smoothed.csv",
        fs[SCHEMA["tab_fleet_per_hub_smoothed.csv"]])

    cidx = costs.set_index(key)
    for gkey, g in fleet.groupby(key):
        c = cidx.loc[gkey]
        gates.append(_gate("fleet", g.fleet.sum(), float(c.vehicle_days),
                           f"vehicle-days {gkey}"))
        gates.append(_gate("fleet", g.groupby("hub").fleet.max().sum(),
                           float(c.sum_hub_peak), f"hub peaks {gkey}"))

    # ── wait ────────────────────────────────────────────────────────────
    wg = wait.groupby(["penalty", "share_willing"], as_index=False).agg(
        num_w=("wait_num_willing", "sum"), num_a=("wait_num_all", "sum"),
        willing_parcels=("willing_parcels", "sum"),
        total_parcels=("total_parcels", "sum"))
    frames["tab_wait_fixed.csv"] = assert_schema("tab_wait_fixed.csv",
        pd.DataFrame({
            "penalty": wg.penalty, "share_willing": wg.share_willing,
            "wait_old": wg.num_a / wg.total_parcels,
            "wait_fixed": wg.num_w / wg.total_parcels,
            "willing_parcels": wg.willing_parcels,
            "total_parcels": wg.total_parcels}))
    frames["tab_wait_smoothed.csv"] = assert_schema("tab_wait_smoothed.csv",
        pd.DataFrame({"penalty": wg.penalty,
                      "share_willing": wg.share_willing,
                      "avg_wait_d_stage3": wg.num_w / wg.total_parcels}))

    # ── tab_balancing_summary.csv ───────────────────────────────────────
    n_plz = (pc[pc.plan == "stage1"].groupby(key).plz.nunique()
             .rename("n_plz").reset_index())
    mx = (fleet.groupby(key, as_index=False).fleet.max()
          .rename(columns={"fleet": "max_fleet_after"}))
    bs = costs[[*key]].copy()
    bs["init_cost_eur"] = costs.cost_stage1_eur.values
    bs["balanced_cost_eur"] = costs.cost_stage2_eur.values
    bs["imbalance_before"] = costs.imbalance_before.values
    bs["imbalance_after"] = costs.imbalance_after.values
    bs["total_routes_before"] = costs.vehicle_days_before.values
    bs["total_routes_after"] = costs.vehicle_days.values
    bs["swaps_made"] = costs.swaps_balance.values
    bs = bs.merge(n_plz, on=key, how="left").merge(mx, on=key, how="left")
    assert bs.n_plz.notna().all() and bs.max_fleet_after.notna().all()
    bs["cost_delta_eur"] = bs.balanced_cost_eur - bs.init_cost_eur
    bs["cost_delta_pct"] = 100 * bs.cost_delta_eur / bs.init_cost_eur
    bs["imbalance_reduction_pct"] = np.where(
        bs.imbalance_before > 0,
        100 * (bs.imbalance_before - bs.imbalance_after)
        / bs.imbalance_before.replace(0, np.nan), 0.0)
    bs["max_fleet_before"] = np.nan
    frames["tab_balancing_summary.csv"] = assert_schema(
        "tab_balancing_summary.csv",
        bs[SCHEMA["tab_balancing_summary.csv"]])

    # ── P*: the ROUTING-lens knees (the submission's classes) ───────────
    kn = H.lsp_knees(costs, wait, cost_col="cost_stage1_eur",
                     wait_num_col="wait_num_willing_stage1")
    frames["tab_pstar_knees_smoothed.csv"] = assert_schema(
        "tab_pstar_knees_smoothed.csv",
        kn[SCHEMA["tab_pstar_knees_smoothed.csv"]])

    # ── tab_per_plz_costs_theta1.csv (the presentation deck's table) ────
    struct = _plz_facts(C.CKPT)
    p1 = pc[(pc.plan == "balanced")
            & np.isclose(pc.share_willing, 1.0)].copy()
    base = (pc[np.isclose(pc.share_willing, 0.0)]
            .groupby(["provider", "plz"], as_index=False).cell_cost_eur.mean()
            .rename(columns={"cell_cost_eur": "dd_cost_baseline_eur"}))
    p1 = p1.merge(base, on=["provider", "plz"], how="left").merge(
        struct, on=["provider", "plz"], how="left")
    assert p1.area_km2.notna().all(), "per-PLZ facts missing for some cell"
    p1["schedule_size_stage3"] = sizes[p1.schedule_idx.values]
    p1["schedule_days_stage3"] = [daystr[i] for i in p1.schedule_idx]
    p1["avg_wait_d_stage3"] = p1.wait_days
    p1["dd_cost_stage3_eur"] = p1.cell_cost_eur
    p1["saving_abs_eur"] = p1.dd_cost_baseline_eur - p1.dd_cost_stage3_eur
    p1["saving_pct"] = 100 * p1.saving_abs_eur / p1.dd_cost_baseline_eur
    p1["weekly_parcels"] = p1.cell_parcels_week
    p1["demand_per_area"] = p1.weekly_parcels / p1.area_km2
    frames["tab_per_plz_costs_theta1.csv"] = assert_schema(
        "tab_per_plz_costs_theta1.csv",
        p1[SCHEMA["tab_per_plz_costs_theta1.csv"]])

    meta = dict(rev_dir=str(rev), **grid_baseline(rev),
                express_allocation=express_allocation,
                head_id=(sorted(set(costs.head_id.astype(str)))[0]
                         if "head_id" in costs.columns else "none"),
                n_gates=len(gates),
                worst_gate_delta=float(np.abs(gates).max()),
                pstar=dict(zip(kn.provider, kn.P_star)),
                no_source=[{"file": k[0], "column": k[1], "why": why}
                           for k, why in NO_SOURCE.items()])
    return frames, meta


def write(frames: dict, out: Path) -> dict[str, Path]:
    written = {}
    (out / "run").mkdir(parents=True, exist_ok=True)
    (out / "rev").mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        sub = "run" if name in IN_RUN_DIR else "rev"
        path = out / sub / name
        df.to_csv(path, index=False)
        written[name] = path
    return written


def render(out: Path, rev: Path) -> list[Path]:
    """Run the three FROZEN builders on the adapted tables, unchanged."""
    produced = []
    for fname in BUILDERS:
        spec = importlib.util.spec_from_file_location(
            f"_frozen_{fname[:2]}", HERE / fname)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        print(f"\n--- {fname} (frozen builder, plotting code unchanged) ---",
              flush=True)
        spec.loader.exec_module(mod)
        mod.main()
    src_dir = out / "rev" / "figures"
    dst_dir = rev / "figures"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for stem in CANONICAL_FIGURES:
        for ext in ("png", "pdf"):
            src = src_dir / f"{stem}.{ext}"
            assert src.exists(), f"{src} was not produced"
            dst = dst_dir / f"{stem}.{ext}"
            shutil.copy2(src, dst)
            produced.append(dst)
    return produced


def build_and_render(rev: Path, out: Path | None = None,
                     express_allocation: str = "per-tour",
                     do_render: bool = True) -> tuple[list[Path], dict]:
    """Adapt *rev* and (optionally) re-render the accepted paper figures."""
    rev = Path(rev)
    out = Path(out) if out else rev / "legacy"
    # The environment must be complete BEFORE _stage3_common is imported
    # anywhere in this process: it freezes both roots and both baselines at
    # import time, and the frozen builders read them from there.
    _set_env(out, grid_baseline(rev))
    frames, meta = build(rev, express_allocation)
    written = write(frames, out)
    meta["files"] = {k: str(v) for k, v in written.items()}
    (out / "legacy_manifest.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str) + chr(10),
        encoding="utf-8")
    produced = render(out, rev) if do_render else []
    meta["figures"] = [str(p) for p in produced]
    return produced, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rev-dir", default="results/revision_2026_08_v6")
    ap.add_argument("--out", default=None,
                    help="legacy table root (default: <rev>/legacy)")
    ap.add_argument("--express-allocation",
                    choices=("per-tour", "dd-only"), default="per-tour")
    ap.add_argument("--render", action="store_true",
                    help="also run the three frozen builders and copy their "
                         "figures into <rev>/figures/")
    args = ap.parse_args(argv)

    rev = Path(args.rev_dir)
    if not rev.is_absolute():
        rev = (ROOT / rev).resolve()
    out = Path(args.out) if args.out else rev / "legacy"
    if not out.is_absolute():
        out = (ROOT / out).resolve()

    print("=" * 74)
    print(f"v2 -> legacy adapter   rev={rev}")
    print(f"                       out={out}")
    print(f"                       express-allocation="
          f"{args.express_allocation}")
    print("=" * 74)
    produced, meta = build_and_render(rev, out, args.express_allocation,
                                      do_render=args.render)
    print(f"\nwrote {len(SCHEMA)} legacy table(s) under {out}")
    print(f"  this grid's own baseline: {meta['base_total_eur']:,.2f} EUR/wk, "
          f"Mo-Sa fleet CV {meta['baseline_cv_exact']:.4f} "
          f"(declared {meta['baseline_cv']})")
    print(f"  head_id={meta['head_id']}   identity gates: {meta['n_gates']}, "
          f"worst |delta| {meta['worst_gate_delta']:.3e}")
    print(f"  routing-lens P*: {meta['pstar']}")
    print("\nlegacy columns with NO v5/v6 source "
          "(written as NaN, never guessed; none is read by fig 4/5/6):")
    for rec in meta["no_source"]:
        print(f"  {rec['file']}::{rec['column']}")
        print(f"      {rec['why']}")
    if produced:
        print(f"\ncopied the accepted paper figures into {rev / 'figures'}:")
        for p in produced:
            print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
