"""73: v2 ops/knee/value-of-stage-2 tables -- the v5/v6-schema replacement
for ``40_tables_smoothed.py`` and ``41_op_kpi_tables_smoothed.py``.

Those two builders read the 2026-07 Stage-3 schema (``tab_costs_smoothed.csv``,
``_tab_chosen_with_system_smoothing.csv``, one plan per (P, theta)). The
v5/v6 grid (``61_grid_run_v2.py``) carries TWO plans -- routing-optimal
(stage 1) and operator-polished (stage 2/"balanced") -- in TWO cost lenses,
so their tables have no v2 equivalent (results inventory gaps G1/G2). This
script reproduces the same three analyses on the new schema:

  ``tab_plz_knee_with_features_v2.csv``   (G1) per (provider, plz): the
      structural features 40_ read off the optimisation-prep checkpoint
      (hub distance, area, parcels per drop-site, region type from
      ``data/geodata/plz_raumtyp.csv``), plus the per-cell euro saving at
      EACH LENS'S OWN knee P* (``tab_pstar_knees_v2.csv``, Task 13): the
      ``_routing`` columns at ``P_star_routing`` on the stage-1 plan, the
      ``_operator`` columns at ``P_star_operator`` on the balanced plan --
      never a lens paired with the other lens's knee. Costs come from
      ``<rev>/tables/tab_per_cell_costs_v2.csv`` (``72_``, Task 13B). If that
      file is absent, or present but does not yet cover theta=1 (or the
      exact P* penalties) for the join, this table is SKIPPED with a loud,
      specific message -- never approximated with a different theta or a
      nearby P.

      IMPORTANT: every saving in this table is a ROUTING-cost saving (own
      tour + parcel-proportional share of every pooled tour the cell rides
      in -- ``tab_per_cell_costs_v2.csv``'s ``cell_cost_eur``). There is no
      such thing as a per-cell OPERATOR-lens (OpCost) saving: the weekly
      fixed bill is sized by a HUB'S peak day, so what one cell "contributes"
      to it depends on when the rest of the hub peaks (``_figs_tables_v2.py``
      ``hub_lens``/``fig6b`` -- "a per-cell operator saving is therefore not
      a well-defined quantity, and this figure does not draw one"). The
      ``_operator`` suffix here names the KNEE POINT this column is
      evaluated at (this LSP's operator-lens P*, on the operator-POLISHED
      plan) -- it is still a routing-cost number, not an OpCost one. A
      cell's true operator-lens contribution only exists at HUB granularity
      (``_figs_tables_v2.hub_lens``); this table stays at PLZ granularity by
      the brief's own request, so it cannot offer that number.

  ``tab_op_kpi_per_day_v2.csv`` /         (G2) per (item, P, theta, plan,
  ``tab_op_kpi_weekly_v2.csv``            provider[, day]): actual km,
      routes, hours (``vroom_duration_h`` -- new in v2; the Stage-3
      revalidation 41_ read had no duration field), cost, in BOTH lenses,
      from ``<rev>/validation/tab_vroom_v2.csv`` (``67_``). The weekly table
      adds predicted-vs-actual saving % against the provider's own THETA=0
      baseline (VROOM validation item 0) -- WHEN item 0 has been solved for
      that provider. Item 0 is treated as OPTIONAL throughout (another task
      is adding it to 67_'s census; as of this writing
      ``results/revision_2026_08_v5/validation/tab_vroom_v2.csv`` has items
      1 and 2 only): its absence never blocks the rest of the table, it just
      leaves the saving columns NaN with ``baseline_available = False``.

      PARTIAL rows (an unassigned-jobs VROOM solve) are FLAGGED (counted in
      ``n_partial``) and COUNTED in every total -- never dropped -- matching
      41_'s original "kept for consistency with the published totals"
      convention and 67_'s own "state both totals" (``_all`` includes
      PARTIAL, ``_clean`` restricts to OK/CACHED) pattern.

  ``tab_value_of_stage2_v2.csv``          per (P, theta): stage-1
      (routing-optimal) vs stage-2 (operator-polished) plan, in both
      lenses -- Delta-routing, Delta-operator, Delta-peak, Delta-wait,
      Delta-mean-days. ``Delta = plan2 - plan1`` throughout: a positive
      Delta-routing is what the operator polish COST in routing terms: a
      negative Delta-operator is what it BOUGHT. Needs only the grid's own
      ``tab_costs_v2.csv`` / ``tab_wait_v2.csv`` / ``_tab_chosen_v2.csv`` --
      no VROOM, no 72_.

Identity gates
--------------
  * per-cell vs grid: ``_figs_tables_v2.check_per_cell_against_grid`` is
    re-run here from the two written CSVs alone (Sigma per-cell cost ==
    the grid's own ``cost_stage1_eur`` / ``cost_stage2_eur`` per
    (P, theta, provider, plan)) whenever the per-cell table loads, even if
    the knee table itself has to be skipped for missing theta=1 coverage.
  * validation rows vs the op-KPI report: the per-day and weekly tables
    must not lose or duplicate a single row of ``tab_vroom_v2.csv`` --
    Sigma ``n_instances`` in each table equals ``len(tab_vroom_v2.csv)``,
    the two tables' instance counts agree per (item, P, theta, plan,
    provider), and Sigma ``vroom_cost_eur`` in the raw file equals Sigma
    ``routing_cost_actual_all_eur`` in the weekly table.
  * theta=0 stage-1/stage-2 no-op: ``tab_value_of_stage2_v2.csv`` asserts
    every Delta is ~0 at theta=0 (``61_``'s G-6f-1 invariant).

This script never imports ``67_validate_vroom_v2.py`` or
``72_per_cell_costs_v2.py`` (both under active development elsewhere as of
this writing) -- only their written CSV CONTRACTS are a dependency. The
operator-lens reconstruction (``variable = Sigma(cost - 189.15*n_routes)``,
``peak_h = max_d Sigma n_routes of hub h``, ``OpCost = Sigma variable +
1134.90*Sigma_h peak_h``) is documented in 67_'s module docstring and
reimplemented here (``_operator_lens``), not imported, to keep this script
decoupled from a module that is still being edited.

Usage
-----
    python scripts/revision/73_tables_ops_v2.py                     # v5
    python scripts/revision/73_tables_ops_v2.py --rev-dir results/revision_2026_08_v6
"""
from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _figs_tables_v2 as H  # noqa: E402  -- v2 aggregation conventions (read-only dep)

DEFAULT_REV = ROOT / "results" / "revision_2026_08_v5"
CHECKPOINT_PATH = ROOT / "results" / "checkpoints" / "04_optim_prep.pkl"
RAUMTYP_PATH = ROOT / "data" / "geodata" / "plz_raumtyp.csv"
PSTAR_NAME = "tab_pstar_knees_v2.csv"
VROOM_NAME = "tab_vroom_v2.csv"
CLEAN_STATUSES = ("OK", "CACHED")


class PerCellKneeInputIncomplete(RuntimeError):
    """72_'s per-cell table exists but does not cover theta=1 (or a needed
    P*) for the knee join yet -- it is still running or was stopped early.
    Never approximate with a different theta or a nearby P; wait for 72_."""


# ═════════════════════════════════════════════════════════════════════════
# Structural facts -- penalty/theta-independent, same fields 40_ read
# ═════════════════════════════════════════════════════════════════════════
def structural_facts(optimization_data: dict, raumtyp: pd.DataFrame) -> pd.DataFrame:
    """(provider, plz) -> hub_dist_km, area_km2, weekly_parcels,
    n_stops_per_day, parcels_per_stop, raumtyp_3.

    Reads exactly the fields ``40_tables_smoothed.write_plz_knee_with_features``
    read off ``od["plz_data"]`` (hub_dist_km, area_km2, n_stops_per_day,
    b2c/b2b weekly demand), joined against ``data/geodata/plz_raumtyp.csv``
    the same way. Grid-independent (no P, no theta, no model, no cost
    matrices) -- one static table for any v5/v6 grid, exactly like 70_'s own
    ``_structural_facts()`` reads the same checkpoint for the same reason.
    """
    rows = []
    for prov, od in optimization_data.items():
        plz_data = od["plz_data"]
        for pc, meta in plz_data.items():
            weekly_parcels = float(sum(meta["b2c"].values())
                                   + sum(meta["b2b"].values()))
            n_stops_per_day = float(meta["n_stops_per_day"])
            stops_per_week = n_stops_per_day * H.N_DAYS
            rows.append(dict(
                provider=str(prov), plz=str(pc).zfill(5),
                hub_dist_km=float(meta["hub_dist_km"]),
                area_km2=float(meta["area_km2"]),
                weekly_parcels=weekly_parcels,
                n_stops_per_day=n_stops_per_day,
                parcels_per_stop=(weekly_parcels / stops_per_week
                                  if stops_per_week > 0 else np.nan)))
    df = pd.DataFrame(rows)
    rt = raumtyp[["plz", "raumtyp_3"]].copy()
    rt["plz"] = rt["plz"].astype(str).str.zfill(5)
    out = df.merge(rt, on="plz", how="left")
    n_missing = int(out.raumtyp_3.isna().sum())
    if n_missing:
        print(f"  WARN: {n_missing} (provider, plz) cell(s) without raumtyp_3")
    return out


def load_structural_facts() -> pd.DataFrame:
    with open(CHECKPOINT_PATH, "rb") as fh:
        od = pickle.load(fh)["optimization_data"]
    rt = pd.read_csv(RAUMTYP_PATH, dtype={"plz": str})
    return structural_facts(od, rt)


# ═════════════════════════════════════════════════════════════════════════
# Table 1: tab_plz_knee_with_features_v2.csv
# ═════════════════════════════════════════════════════════════════════════
def _require_pstar_coverage(per_cell: pd.DataFrame, pstar: pd.DataFrame) -> None:
    """Fail loud if theta=1 rows at every provider's own P* are not yet in
    the per-cell table, instead of joining onto an empty/partial slice."""
    have = {(str(r.provider), str(r.plan), round(float(r.penalty), 6))
            for r in per_cell[np.isclose(per_cell.share_willing, 1.0)]
                       .itertuples()}
    missing = []
    for _, r in pstar.iterrows():
        for plan, pcol in (("stage1", "P_star_routing"),
                          ("balanced", "P_star_operator")):
            key = (str(r.provider), plan, round(float(r[pcol]), 6))
            if key not in have:
                missing.append(key)
    if missing:
        theta_present = sorted(float(x) for x in per_cell.share_willing.unique())
        raise PerCellKneeInputIncomplete(
            f"{len(missing)} (provider, plan, P*) combination(s) needed for "
            f"the knee join are not in tab_per_cell_costs_v2.csv at theta=1 "
            f"yet -- e.g. {missing[:5]}. theta values currently present in "
            f"the per-cell table: {theta_present}. 72_per_cell_costs_v2.py "
            "is still running or was stopped before reaching theta=1 -- "
            "let it finish (or re-run it) rather than approximating with a "
            "different theta or a nearby P.")


def compute_plz_knee_with_features(structural: pd.DataFrame,
                                   per_cell: pd.DataFrame,
                                   pstar: pd.DataFrame) -> pd.DataFrame:
    """Structural features + per-cell ROUTING-cost saving % at each lens's
    own knee P*.

    ``_routing`` columns: plan == "stage1" at ``P_star_routing``, theta = 1
    (the fixed ``theta_target`` every knee in ``tab_pstar_knees_v2.csv`` is
    computed at -- see ``_figs_tables_v2.lsp_knees``). ``_operator``
    columns: plan == "balanced" at ``P_star_operator``, theta = 1. Never a
    lens paired with the other lens's P* (Task 13's central rule).

    Both sets of columns are the cell's own routing cost (``cell_cost_eur``
    -- own tour + pooled share); ``_operator`` names the KNEE this column is
    evaluated at, not an OpCost quantity -- a per-cell OpCost saving is not
    well-defined (see the module docstring and ``_figs_tables_v2.hub_lens``).
    """
    _require_pstar_coverage(per_cell, pstar)
    sav = H.per_cell_savings(per_cell)

    def _at_pstar(plan: str, pcol: str, suffix: str) -> pd.DataFrame:
        parts = []
        for _, r in pstar.iterrows():
            sub = sav[(sav.provider == r.provider) & (sav.plan == plan)
                      & np.isclose(sav.share_willing, 1.0)
                      & np.isclose(sav.penalty, float(r[pcol]))]
            parts.append(sub)
        out = (pd.concat(parts, ignore_index=True) if parts
               else sav.iloc[0:0].copy())
        keep = ["provider", "plz", "cell_cost_eur", "baseline_cell_eur",
                "saving_eur", "saving_pct"]
        out = out[keep].drop_duplicates(["provider", "plz"])
        return out.rename(columns={c: f"{c}_{suffix}" for c in keep
                                   if c not in ("provider", "plz")})

    at_r = _at_pstar("stage1", "P_star_routing", "routing")
    at_o = _at_pstar("balanced", "P_star_operator", "operator")

    out = structural.merge(
        pstar[["provider", "P_star_routing", "carrier_class_routing",
              "P_star_operator", "carrier_class_operator"]],
        on="provider", how="left")
    out = out.merge(at_r, on=["provider", "plz"], how="left")
    out = out.merge(at_o, on=["provider", "plz"], how="left")
    for suffix in ("routing", "operator"):
        missing = out[f"saving_pct_{suffix}"].isna()
        assert not missing.any(), (
            f"{int(missing.sum())} (provider, plz) cell(s) have no per-cell "
            f"row at their own {suffix} P* -- the knee table and the "
            "per-cell table disagree on cell coverage")
    return out.sort_values(["provider", "plz"]).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════
# Operator-lens reconstruction (documented in 67_, reimplemented not
# imported -- see module docstring)
# ═════════════════════════════════════════════════════════════════════════
def _operator_lens(df: pd.DataFrame, cost_col: str, routes_col: str,
                   hub_key: str = "hub_name") -> dict:
    """variable / sum_hub_peak / OpCost from instance rows already filtered
    to one (item, P, theta, plan, provider) group.

    ``variable = Sigma(cost - 189.15 * n_routes)`` over rows with a priced
    cost; ``peak_h = max_d Sigma n_routes of hub h's instances`` (an
    unpriced/missing-cost row still needs its van, so it still counts
    toward the peak); ``OpCost = Sigma variable + 1134.90 * Sigma_h peak_h``.
    """
    if df is None or df.empty:
        return dict(variable_eur=0.0, sum_hub_peak=0.0, opcost_eur=0.0,
                   routing_eur=0.0, vehicle_days=0.0, n=0)
    d = df.copy()
    routes = pd.to_numeric(d[routes_col], errors="coerce").fillna(0.0)
    cost = pd.to_numeric(d[cost_col], errors="coerce")
    priced = cost.notna()
    variable = float((cost[priced] - H.FIXED_COST_EUR * routes[priced]).sum())
    per_hub_day = d.assign(_routes=routes).groupby([hub_key, "day"])["_routes"].sum()
    sum_hub_peak = float(per_hub_day.groupby(level=0).max().sum()) if len(per_hub_day) else 0.0
    return dict(
        variable_eur=variable, sum_hub_peak=sum_hub_peak,
        opcost_eur=variable + H.WEEK_FIXED_COST_EUR * sum_hub_peak,
        routing_eur=float(cost[priced].sum()),
        vehicle_days=float(routes.sum()), n=int(len(d)))


# ═════════════════════════════════════════════════════════════════════════
# Tables 2/3: tab_op_kpi_per_day_v2.csv / tab_op_kpi_weekly_v2.csv
# ═════════════════════════════════════════════════════════════════════════
def compute_op_kpi_per_day(vroom: pd.DataFrame) -> pd.DataFrame:
    """Per (item, P, theta, plan, provider, day): actual + predicted routing
    cost and the operator lens's VARIABLE component (the weekly fixed/peak
    term is not a per-day quantity -- see ``tab_op_kpi_weekly_v2.csv`` for
    the full OpCost). PARTIAL/other flagged rows are counted (``n_partial``,
    ``n_other_flagged``) and included in every sum, matching 41_'s original
    "kept for consistency with the published totals" convention.
    """
    d = vroom.copy()
    d["vroom_status"] = d["vroom_status"].astype(str)
    key = ["item", "penalty", "share_willing", "plan", "provider", "day"]
    rows = []
    for keys, g in d.groupby(key, sort=True):
        pred_cost = pd.to_numeric(g.predicted_cost_eur, errors="coerce")
        pred_routes = pd.to_numeric(g.predicted_n_routes, errors="coerce").fillna(0.0)
        act_cost = pd.to_numeric(g.vroom_cost_eur, errors="coerce")
        act_routes = pd.to_numeric(g.vroom_n_routes, errors="coerce").fillna(0.0)
        priced = act_cost.notna()
        row = dict(zip(key, keys))
        row.update(
            n_instances=int(len(g)),
            n_partial=int((g.vroom_status == "PARTIAL").sum()),
            n_other_flagged=int((~g.vroom_status.isin((*CLEAN_STATUSES, "PARTIAL"))).sum()),
            n_routes=float(act_routes.sum()),
            distance_km=float(pd.to_numeric(g.vroom_distance_km, errors="coerce").fillna(0.0).sum()),
            duration_h=float(pd.to_numeric(g.vroom_duration_h, errors="coerce").fillna(0.0).sum()),
            parcels=float(pd.to_numeric(g.vroom_n_parcels, errors="coerce").fillna(0.0).sum()),
            routing_cost_actual_eur=float(act_cost[priced].sum()),
            routing_cost_pred_eur=float(pred_cost.sum()),
            variable_cost_actual_eur=float((act_cost[priced] - H.FIXED_COST_EUR * act_routes[priced]).sum()),
            variable_cost_pred_eur=float((pred_cost - H.FIXED_COST_EUR * pred_routes).sum()),
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out["parcels_per_route"] = out.parcels / out.n_routes.replace(0, np.nan)
    out["km_per_route"] = out.distance_km / out.n_routes.replace(0, np.nan)
    out["km_per_parcel"] = out.distance_km / out.parcels.replace(0, np.nan)
    out["eur_per_parcel"] = out.routing_cost_actual_eur / out.parcels.replace(0, np.nan)
    return out.sort_values(key).reset_index(drop=True)


def compute_op_kpi_weekly(vroom: pd.DataFrame) -> pd.DataFrame:
    """Per (item, P, theta, plan, provider): both lenses, actual (all rows
    AND clean-only) vs predicted, plus predicted-vs-actual saving % against
    the provider's own item-0 baseline WHEN item 0 has been solved for that
    provider (``baseline_available``; item 0 is optional -- see module
    docstring). PARTIAL rows are always counted (``n_partial``) and never
    dropped from the ``_all`` totals; ``_clean`` states the OK/CACHED-only
    alternative, mirroring 67_'s own "_all"/"_clean" pair.
    """
    d = vroom.copy()
    d["vroom_status"] = d["vroom_status"].astype(str)
    hub_key = "hub_name" if "hub_name" in d.columns else "hub_idx"
    key = ["item", "penalty", "share_willing", "plan", "provider"]

    rows = []
    for keys, g in d.groupby(key, sort=True):
        g_clean = g[g.vroom_status.isin(CLEAN_STATUSES)]
        pred = _operator_lens(g, "predicted_cost_eur", "predicted_n_routes", hub_key)
        act_all = _operator_lens(g, "vroom_cost_eur", "vroom_n_routes", hub_key)
        act_clean = _operator_lens(g_clean, "vroom_cost_eur", "vroom_n_routes", hub_key)
        row = dict(zip(key, keys))
        row.update(
            n_instances=int(len(g)), n_clean=int(len(g_clean)),
            n_partial=int((g.vroom_status == "PARTIAL").sum()),
            n_other_flagged=int((~g.vroom_status.isin((*CLEAN_STATUSES, "PARTIAL"))).sum()),
            n_days_with_delivery=int(g.day.nunique()),
            n_routes_week=act_all["vehicle_days"],
            distance_km_week=float(pd.to_numeric(g.vroom_distance_km, errors="coerce").fillna(0.0).sum()),
            duration_h_week=float(pd.to_numeric(g.vroom_duration_h, errors="coerce").fillna(0.0).sum()),
            parcels_week=float(pd.to_numeric(g.vroom_n_parcels, errors="coerce").fillna(0.0).sum()),
            routing_cost_actual_all_eur=act_all["routing_eur"],
            routing_cost_actual_clean_eur=act_clean["routing_eur"],
            routing_cost_pred_eur=pred["routing_eur"],
            variable_cost_actual_all_eur=act_all["variable_eur"],
            variable_cost_actual_clean_eur=act_clean["variable_eur"],
            variable_cost_pred_eur=pred["variable_eur"],
            sum_hub_peak_actual_all=act_all["sum_hub_peak"],
            sum_hub_peak_actual_clean=act_clean["sum_hub_peak"],
            sum_hub_peak_pred=pred["sum_hub_peak"],
            opcost_actual_all_eur=act_all["opcost_eur"],
            opcost_actual_clean_eur=act_clean["opcost_eur"],
            opcost_pred_eur=pred["opcost_eur"],
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out["parcels_per_route"] = out.parcels_week / out.n_routes_week.replace(0, np.nan)
    out["km_per_route"] = out.distance_km_week / out.n_routes_week.replace(0, np.nan)
    out["km_per_parcel"] = out.distance_km_week / out.parcels_week.replace(0, np.nan)
    out["eur_per_parcel"] = out.routing_cost_actual_all_eur / out.parcels_week.replace(0, np.nan)
    out["routing_bias_pct"] = ((out.routing_cost_pred_eur - out.routing_cost_actual_all_eur)
                               / out.routing_cost_actual_all_eur.replace(0, np.nan) * 100.0)
    out["opcost_bias_pct"] = ((out.opcost_pred_eur - out.opcost_actual_all_eur)
                              / out.opcost_actual_all_eur.replace(0, np.nan) * 100.0)

    save_cols = ["pred_routing_save_pct", "act_routing_save_pct_all",
                "act_routing_save_pct_clean", "pred_opcost_save_pct",
                "act_opcost_save_pct_all", "act_opcost_save_pct_clean"]
    base_all = d[d.item == 0]
    if base_all.empty:
        for c in save_cols:
            out[c] = np.nan
        out["baseline_available"] = False
        return out.sort_values(key).reset_index(drop=True)

    base_map = {}
    for prov, gb in base_all.groupby("provider"):
        gb_clean = gb[gb.vroom_status.isin(CLEAN_STATUSES)]
        base_map[str(prov)] = dict(
            pred=_operator_lens(gb, "predicted_cost_eur", "predicted_n_routes", hub_key),
            act_all=_operator_lens(gb, "vroom_cost_eur", "vroom_n_routes", hub_key),
            act_clean=_operator_lens(gb_clean, "vroom_cost_eur", "vroom_n_routes", hub_key))

    def _sv(base_val: float, point_val: float) -> float:
        return (base_val - point_val) / base_val * 100.0 if base_val else np.nan

    extra = []
    for rec in out.to_dict("records"):
        b = base_map.get(str(rec["provider"]))
        avail = b is not None and int(rec["item"]) != 0
        if not avail:
            extra.append({**{c: np.nan for c in save_cols}, "baseline_available": False})
            continue
        extra.append(dict(
            pred_routing_save_pct=_sv(b["pred"]["routing_eur"], rec["routing_cost_pred_eur"]),
            act_routing_save_pct_all=_sv(b["act_all"]["routing_eur"], rec["routing_cost_actual_all_eur"]),
            act_routing_save_pct_clean=_sv(b["act_clean"]["routing_eur"], rec["routing_cost_actual_clean_eur"]),
            pred_opcost_save_pct=_sv(b["pred"]["opcost_eur"], rec["opcost_pred_eur"]),
            act_opcost_save_pct_all=_sv(b["act_all"]["opcost_eur"], rec["opcost_actual_all_eur"]),
            act_opcost_save_pct_clean=_sv(b["act_clean"]["opcost_eur"], rec["opcost_actual_clean_eur"]),
            baseline_available=True))
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    return out.sort_values(key).reset_index(drop=True)


def _gate_validation_rows(vroom: pd.DataFrame, day: pd.DataFrame,
                          week: pd.DataFrame) -> None:
    """Sigma validation rows == report totals: the op-KPI tables must not
    lose or duplicate a single row of ``tab_vroom_v2.csv``."""
    assert int(day.n_instances.sum()) == len(vroom), (
        f"per-day table covers {int(day.n_instances.sum())} instances, "
        f"tab_vroom_v2.csv has {len(vroom)} -- rows were lost or double-counted")
    assert int(week.n_instances.sum()) == len(vroom), (
        f"weekly table covers {int(week.n_instances.sum())} instances, "
        f"tab_vroom_v2.csv has {len(vroom)} rows")
    key = ["item", "penalty", "share_willing", "plan", "provider"]
    day_roll = day.groupby(key, as_index=False).n_instances.sum()
    chk = day_roll.merge(week[key + ["n_instances"]], on=key, how="outer",
                         suffixes=("_day", "_week"), indicator=True)
    assert (chk._merge == "both").all(), (
        "per-day and weekly tables disagree on which (item, P, theta, plan, "
        f"provider) groups exist:\n{chk[chk._merge != 'both'][key]}")
    bad = chk[chk.n_instances_day != chk.n_instances_week]
    assert bad.empty, (
        f"per-day sum and weekly n_instances disagree for {len(bad)} "
        f"group(s):\n{bad[key]}")
    total_cost_v = float(pd.to_numeric(vroom.vroom_cost_eur, errors="coerce").fillna(0.0).sum())
    total_cost_w = float(week.routing_cost_actual_all_eur.sum())
    assert abs(total_cost_v - total_cost_w) < 1e-6 * max(1.0, abs(total_cost_v)), (
        f"Sigma vroom_cost_eur in tab_vroom_v2.csv ({total_cost_v:.2f}) != "
        f"Sigma routing_cost_actual_all_eur in the weekly table "
        f"({total_cost_w:.2f})")


# ═════════════════════════════════════════════════════════════════════════
# Table 4: tab_value_of_stage2_v2.csv
# ═════════════════════════════════════════════════════════════════════════
def compute_value_of_stage2(costs: pd.DataFrame, wait: pd.DataFrame,
                            n_cells: pd.Series | None = None) -> pd.DataFrame:
    """Per (P, theta): stage-1 (routing-optimal) vs stage-2 (operator-
    polished) plan, both lenses. ``Delta = plan2 - plan1``: a positive
    Delta-routing is what the operator polish COST in routing terms; a
    negative Delta-operator is what it BOUGHT. Built from
    ``_figs_tables_v2.headline_rows`` -- the same two-plan frame
    ``70_``'s ``tab_grid_full_v2.csv`` is drawn from, not a reimplementation.
    """
    base = H.baseline(costs)
    full = H.headline_rows(costs, wait, base=base, n_cells=n_cells)
    out = full[["penalty", "share_willing",
               "routing_cost_plan1_eur", "routing_cost_plan2_eur",
               "operator_cost_plan1_eur", "operator_cost_plan2_eur",
               "sum_hub_peak_plan1", "sum_hub_peak_plan2",
               "wait_d_plan1", "wait_d_plan2",
               "mean_days_plan1", "mean_days_plan2",
               "routing_saving_plan1_pct", "routing_saving_plan2_pct",
               "operator_saving_plan1_pct", "operator_saving_plan2_pct"]].copy()
    out["delta_routing_eur"] = out.routing_cost_plan2_eur - out.routing_cost_plan1_eur
    out["delta_operator_eur"] = out.operator_cost_plan2_eur - out.operator_cost_plan1_eur
    out["delta_peak"] = out.sum_hub_peak_plan2 - out.sum_hub_peak_plan1
    out["delta_wait_d"] = out.wait_d_plan2 - out.wait_d_plan1
    out["delta_mean_days"] = out.mean_days_plan2 - out.mean_days_plan1
    out["delta_routing_saving_pp"] = (out.routing_saving_plan2_pct
                                      - out.routing_saving_plan1_pct)
    out["delta_operator_saving_pp"] = (out.operator_saving_plan2_pct
                                       - out.operator_saving_plan1_pct)
    theta0 = out[np.isclose(out.share_willing, 0.0)]
    assert np.allclose(theta0.delta_routing_eur, 0.0, atol=1e-6), (
        "theta=0 stage-1/stage-2 routing delta is nonzero -- the two plans "
        "are supposed to be identical at theta=0 (stage-2 no-op, G-6f-1)")
    assert np.allclose(theta0.delta_operator_eur, 0.0, atol=1e-6), (
        "theta=0 stage-1/stage-2 operator delta is nonzero -- same no-op "
        "invariant violated in the operator lens")
    return out.sort_values(["penalty", "share_willing"]).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════
def _print_weekly_summary(week: pd.DataFrame) -> None:
    if week.empty:
        return
    agg = (week.groupby(["item", "penalty", "share_willing", "plan"], as_index=False)
          .agg(n_instances=("n_instances", "sum"),
               n_partial=("n_partial", "sum"),
               routing_actual_eur=("routing_cost_actual_all_eur", "sum"),
               routing_pred_eur=("routing_cost_pred_eur", "sum"),
               opcost_actual_eur=("opcost_actual_all_eur", "sum"),
               opcost_pred_eur=("opcost_pred_eur", "sum")))
    agg["routing_bias_pct"] = ((agg.routing_pred_eur - agg.routing_actual_eur)
                               / agg.routing_actual_eur * 100)
    print(agg.round(2).to_string(index=False))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rev-dir", default=str(DEFAULT_REV))
    args = ap.parse_args(argv)
    rev_dir = Path(args.rev_dir)
    if not rev_dir.is_absolute():
        rev_dir = ROOT / rev_dir
    tables_dir = rev_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"73 v2 ops/knee/value-of-stage-2 tables on {rev_dir}")
    print("=" * 72)

    written: dict[str, Path] = {}

    # -- table 4: value of stage 2 (grid-only) -----------------------------
    print("\n[1/3] tab_value_of_stage2_v2.csv")
    grid = H.RevGrid(rev_dir)
    vs2 = compute_value_of_stage2(grid.costs, grid.wait,
                                  n_cells=grid.n_cells_per_provider())
    p = tables_dir / "tab_value_of_stage2_v2.csv"
    vs2.to_csv(p, index=False)
    written["value_of_stage2"] = p
    print(f"  wrote {p} ({len(vs2)} rows); theta=0 no-op gate OK")

    # -- table 1: PLZ knee with features (needs 72_ + pstar) ---------------
    print("\n[2/3] tab_plz_knee_with_features_v2.csv")
    try:
        per_cell = H.load_per_cell(rev_dir)
    except H.PerCellMissing as exc:
        print(f"  SKIPPED -- {exc}")
        per_cell = None

    if per_cell is not None:
        gate = H.check_per_cell_against_grid(per_cell, grid.costs)
        print(f"  per-cell identity gate OK over {len(gate)} "
              "(P, theta, provider, plan) group(s) present in the per-cell table")
        pstar_p = tables_dir / PSTAR_NAME
        if not pstar_p.exists():
            print(f"  SKIPPED -- {pstar_p} missing. Run "
                  f"scripts/revision/70_figs_tables_v2.py --rev-dir {rev_dir} "
                  "first (it writes the per-LSP knee P* table this join needs).")
        else:
            pstar = pd.read_csv(pstar_p)
            try:
                structural = load_structural_facts()
                knee_df = compute_plz_knee_with_features(structural, per_cell, pstar)
            except PerCellKneeInputIncomplete as exc:
                print(f"  SKIPPED -- {exc}")
            else:
                p = tables_dir / "tab_plz_knee_with_features_v2.csv"
                knee_df.to_csv(p, index=False)
                written["plz_knee"] = p
                print(f"  wrote {p} ({len(knee_df)} rows)")

    # -- tables 2/3: op-kpi per day / weekly (needs 67_'s validation) ------
    print("\n[3/3] tab_op_kpi_per_day_v2.csv / tab_op_kpi_weekly_v2.csv")
    vroom_p = rev_dir / "validation" / VROOM_NAME
    if not vroom_p.exists():
        print(f"  SKIPPED -- {vroom_p} does not exist. Run "
              f"scripts/revision/67_validate_vroom_v2.py --rev-dir {rev_dir} "
              "first.")
    else:
        vroom = pd.read_csv(vroom_p)
        day = compute_op_kpi_per_day(vroom)
        week = compute_op_kpi_weekly(vroom)
        _gate_validation_rows(vroom, day, week)
        p1 = tables_dir / "tab_op_kpi_per_day_v2.csv"
        day.to_csv(p1, index=False)
        p2 = tables_dir / "tab_op_kpi_weekly_v2.csv"
        week.to_csv(p2, index=False)
        written["op_kpi_per_day"] = p1
        written["op_kpi_weekly"] = p2
        print(f"  wrote {p1} ({len(day)} rows)")
        print(f"  wrote {p2} ({len(week)} rows); validation-rows gate OK "
              f"({len(vroom)} rows reconciled)")
        if not bool(week.baseline_available.any()):
            print("  NOTE: VROOM validation item 0 (theta=0 baseline) is "
                  "not in this grid's tab_vroom_v2.csv yet -- "
                  "predicted-vs-actual saving % columns are NaN "
                  "(baseline_available=False) until it is solved.")
        print()
        _print_weekly_summary(week)

    print("\n" + "=" * 72)
    print("done. Tables written:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    if len(written) < 4:
        print(f"  ({4 - len(written)} of 4 tables skipped -- see SKIPPED "
              "messages above)")
    return written


if __name__ == "__main__":
    main()
