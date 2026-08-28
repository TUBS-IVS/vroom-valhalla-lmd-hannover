"""Aggregation helpers for the v5-schema (two-plan) figures and tables.

The v5 grid (``61_grid_run_v2.py``, Task 6f) writes **two plans per
(P, theta)**:

* the **routing-optimal** stage-1 plan -- the coordinate-descent selection
  that minimises routing cost + service penalty.  Its columns carry the
  ``*_before`` / ``*_stage1`` suffix.
* the **operator-polished** stage-2 plan -- the frequency-free OpCost
  polish (best-of-3 starts).  Its columns carry the plain names.

Every quantity is reported in **two cost lenses**:

* **routing lens** -- the paper's cost, ``189.15 EUR/vehicle-day
  + 0.3864 EUR/km + 36 EUR/route-hour`` (VROOM ``per_hour`` default).
  Column ``cost_stage1_eur`` (plan 1) / ``cost_stage2_eur`` (plan 2).
* **operator lens** -- ``variable + 6 * 189.15 * sum_h peak_h``; the weekly
  fixed bill is charged once per *hub peak* vehicle because the van is owned
  and the driver employed for the whole week.  The service penalty enters the
  stage-2 *objective* but is NOT part of the reported operator cost.
  Column ``operator_cost_before_eur`` (plan 1) / ``operator_cost_eur``
  (plan 2).

Never mix a lens or a plan inside one panel or one table column.  Every
helper here names both explicitly (``*_plan1`` = routing-optimal,
``*_plan2`` = operator-polished).

The wait metric is the **parcel-weighted additional wait**:
``sum_provider wait_num_willing / sum_provider total_parcels`` -- the
numerator counts only the willing parcels (they are the only ones that
wait), the denominator counts ALL parcels, so the metric is the mean extra
wait per delivered parcel.  Both plans share the same denominator (asserted).

Nothing in here reads ``tab_sensitivity_master_plz.csv`` (Kompendium §38.3).
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── cost model constants (Kompendium §40.11 / §40.12) ────────────────────
FIXED_COST_EUR = 189.15          # EUR per vehicle and DAY (driver included)
WEEK_FIXED_COST_EUR = 6 * FIXED_COST_EUR   # 1134.90 EUR per peak vehicle/wk
N_DAYS = 6
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
PROVIDERS = ["DHL", "Amazon", "DPD", "FedEx", "GLS", "Hermes", "UPS"]
N_PROVIDERS = len(PROVIDERS)

COST_MODEL_CAPTION = (
    r"Cost model: 189.15 \euro/vehicle-day + 0.3864 \euro/km "
    r"+ 36 \euro/route-hour (VROOM \texttt{per\_hour} default)."
)
COST_MODEL_TEXT = ("Cost model: 189.15 EUR/vehicle-day + 0.3864 EUR/km "
                   "+ 36 EUR/route-hour (VROOM per_hour default).")

PLAN1 = "routing-optimal plan (stage 1)"
PLAN2 = "operator-polished plan (stage 2)"
PLAN_ROW = {1: "Routing-optimal\n(stage 1)", 2: "Operator-polished\n(stage 2)"}
PLAN_WORD = {1: "routing-optimal", 2: "operator-polished"}
LENS_ROUTING = "routing lens"
LENS_OPERATOR = "operator lens"


# ── plan / lens declaration, derived from the plotted column ─────────────
# The amendment's central constraint is "never mix a plan or a lens in one
# panel, and label every panel".  A hand-written panel title cannot enforce
# that: swapping the plotted column leaves the title untouched.  So every
# declaration is DERIVED from the column name, and a column this table does
# not know about raises instead of rendering an unlabelled panel.
_PLAN_SUFFIX = (("_plan1", 1), ("_plan2", 2), ("_stage1", 1), ("_before", 1))

# Columns whose plan is not in their name.  The fleet CSV is written at the
# FINAL plan (Task 6f column list), so every aggregate derived from it is
# plan 2.
PLAN_BY_COLUMN = {
    "sys_cv": 2, "sys_peak": 2, "sys_total": 2,
    "flat_bound": 2, "peak_gap": 2, "peak_gap_eur": 2,
    "sum_hub_peak_fleet": 2,
}
_LENS_PREFIX = (("routing_", LENS_ROUTING), ("operator_", LENS_OPERATOR),
                ("variable_", LENS_OPERATOR))


def plan_of(col: str) -> int:
    """1 = routing-optimal (stage 1), 2 = operator-polished (stage 2)."""
    if col in PLAN_BY_COLUMN:
        return PLAN_BY_COLUMN[col]
    for suffix, plan in _PLAN_SUFFIX:
        if suffix in col:
            return plan
    raise KeyError(
        f"cannot tell which plan {col!r} belongs to -- add it to "
        "PLAN_BY_COLUMN or give it a _plan1/_plan2/_stage1/_before suffix "
        "before plotting it, so its panel cannot be mislabelled")


def lens_of(col: str) -> str | None:
    """The cost lens of a monetary column, or None for a non-cost quantity."""
    for prefix, lens in _LENS_PREFIX:
        if col.startswith(prefix):
            return lens
    return None


def declare(col: str) -> str:
    """The panel's plan (and lens, if it is a cost) as one label line."""
    plan = PLAN1 if plan_of(col) == 1 else PLAN2
    lens = lens_of(col)
    return f"{lens} · {plan}" if lens else plan


# ─────────────────────────────────────────────────────────────────────────
# Loading + fail-loud structural gates
# ─────────────────────────────────────────────────────────────────────────
class RevGrid:
    """The four v5 CSVs of one grid directory, with the invariants checked.

    ``costs``/``wait`` are per (penalty, share_willing, provider);
    ``fleet`` is per (penalty, share_willing, provider, hub, day) at the
    FINAL (stage-2) plan; ``chosen`` is per (..., plz) schedule indices.
    """

    REQUIRED_COST_COLS = (
        "cost_stage1_eur", "cost_stage2_eur", "operator_cost_eur",
        "operator_cost_before_eur", "sum_hub_peak", "sum_hub_peak_before",
        "vehicle_days", "vehicle_days_before", "variable_cost_eur",
        "variable_before_eur", "penalty_eur", "penalty_before_eur",
    )
    REQUIRED_WAIT_COLS = (
        "wait_num_willing", "wait_num_all", "total_parcels",
        "willing_parcels", "wait_num_willing_stage1", "wait_num_all_stage1",
        "mean_days", "mean_days_stage1",
    )

    def __init__(self, rev_dir: Path, require_chosen: bool = True):
        self.dir = Path(rev_dir)
        self.costs = pd.read_csv(self.dir / "tab_costs_v2.csv")
        self.wait = pd.read_csv(self.dir / "tab_wait_v2.csv")
        self.fleet = pd.read_csv(self.dir / "tab_fleet_per_hub_v2.csv")
        self.chosen = (pd.read_csv(self.dir / "_tab_chosen_v2.csv")
                       if require_chosen else None)
        missing = [c for c in self.REQUIRED_COST_COLS
                   if c not in self.costs.columns]
        assert not missing, (
            f"{self.dir} is not a v5-schema grid -- tab_costs_v2.csv lacks "
            f"{missing}. Point --rev-dir at a Task-6f (or later) run.")
        missing = [c for c in self.REQUIRED_WAIT_COLS
                   if c not in self.wait.columns]
        assert not missing, (
            f"{self.dir}/tab_wait_v2.csv lacks {missing} -- pre-6f schema.")
        check_grid_integrity(self.costs, self.wait, str(self.dir))

    # -- convenience --------------------------------------------------
    @property
    def cells(self) -> list[tuple[float, float]]:
        return sorted({(float(p), float(t)) for p, t in
                       zip(self.costs.penalty, self.costs.share_willing)})

    def n_cells_per_provider(self) -> pd.Series:
        assert self.chosen is not None, "chosen table not loaded"
        c = self.chosen
        one = c[(np.isclose(c.penalty, c.penalty.iloc[0]))
                & (np.isclose(c.share_willing, c.share_willing.iloc[0]))]
        return one.groupby("provider").size().rename("n_cells")


def check_grid_integrity(costs: pd.DataFrame, wait: pd.DataFrame,
                         label: str = "grid") -> None:
    """Fail loud on the three invariants the paper numbers depend on."""
    # 1. seven providers in every (P, theta) cell
    n_prov = costs.groupby(["penalty", "share_willing"]).provider.nunique()
    bad = n_prov[n_prov != N_PROVIDERS]
    assert bad.empty, (
        f"{label}: {len(bad)} (P, theta) cell(s) do not carry "
        f"{N_PROVIDERS} providers -- e.g. {bad.head().to_dict()}")

    # 2. the theta=0 baseline is identical for every P, PER PROVIDER.
    #    Summing over providers first would let two providers drift in
    #    opposite directions and still pass -- and lsp_knees() takes a
    #    per-provider baseline, so that drift would silently move a knee.
    cols = ["cost_stage1_eur", "cost_stage2_eur", "operator_cost_eur",
            "sum_hub_peak", "vehicle_days", "variable_cost_eur"]
    b0 = costs[np.isclose(costs.share_willing, 0.0)]
    assert len(b0) > 0, f"{label}: no theta=0 rows -- no baseline"
    b = b0.groupby(["provider", "penalty"])[cols].sum()
    spread = (b.groupby("provider").max() - b.groupby("provider").min()).abs()
    assert (spread.to_numpy() < 1e-6).all(), (
        f"{label}: the theta=0 baseline is NOT identical across P "
        f"(max per-provider spread per column: "
        f"{spread.max().to_dict()}) -- the two-plan tables would be "
        "measured against a moving baseline")

    # 3. both plans share the wait denominator
    dn = wait.groupby(["penalty", "share_willing"]).total_parcels.sum()
    assert np.allclose(dn.values, dn.values[0]), (
        f"{label}: total_parcels varies across (P, theta) "
        f"({dn.min():.1f} .. {dn.max():.1f}) -- the two plans would be "
        "normalised by different denominators")


# ─────────────────────────────────────────────────────────────────────────
# Baseline
# ─────────────────────────────────────────────────────────────────────────
def baseline(costs: pd.DataFrame) -> dict[str, float]:
    """The theta=0 daily-delivery baseline (identical for every P)."""
    b = costs[np.isclose(costs.share_willing, 0.0)]
    P0 = sorted(b.penalty.unique())[0]
    b = b[np.isclose(b.penalty, P0)]
    out = {
        "routing_eur": float(b.cost_stage2_eur.sum()),
        "operator_eur": float(b.operator_cost_eur.sum()),
        "variable_eur": float(b.variable_cost_eur.sum()),
        "sum_hub_peak": float(b.sum_hub_peak.sum()),
        "vehicle_days": float(b.vehicle_days.sum()),
    }
    # the stage-1 plan at theta=0 is the same daily plan
    assert abs(float(b.cost_stage1_eur.sum()) - out["routing_eur"]) < 1e-6, (
        "theta=0: the stage-1 and stage-2 plans differ -- the baseline is "
        "supposed to be the pinned daily schedule for both")
    return out


# ─────────────────────────────────────────────────────────────────────────
# The two-plan x two-lens headline frame
# ─────────────────────────────────────────────────────────────────────────
def headline_rows(costs: pd.DataFrame, wait: pd.DataFrame,
                  base: dict[str, float] | None = None,
                  n_cells: pd.Series | None = None) -> pd.DataFrame:
    """Per (P, theta): both plans, both lenses, never mixed in one column.

    ``*_plan1`` = routing-optimal stage-1 plan, ``*_plan2`` =
    operator-polished stage-2 plan.  Savings are percentages against the
    theta=0 baseline **of the same lens**.
    """
    if base is None:
        base = baseline(costs)
    g = (costs.groupby(["penalty", "share_willing"], as_index=False)
         .agg(routing_cost_plan1_eur=("cost_stage1_eur", "sum"),
              routing_cost_plan2_eur=("cost_stage2_eur", "sum"),
              operator_cost_plan1_eur=("operator_cost_before_eur", "sum"),
              operator_cost_plan2_eur=("operator_cost_eur", "sum"),
              variable_plan1_eur=("variable_before_eur", "sum"),
              variable_plan2_eur=("variable_cost_eur", "sum"),
              penalty_plan1_eur=("penalty_before_eur", "sum"),
              penalty_plan2_eur=("penalty_eur", "sum"),
              sum_hub_peak_plan1=("sum_hub_peak_before", "sum"),
              sum_hub_peak_plan2=("sum_hub_peak", "sum"),
              vehicle_days_plan1=("vehicle_days_before", "sum"),
              vehicle_days_plan2=("vehicle_days", "sum")))

    wg = wait.copy()
    if n_cells is not None:
        wg = wg.merge(n_cells, on="provider", how="left")
        assert wg.n_cells.notna().all(), "n_cells missing for some provider"
    else:
        wg["n_cells"] = 1.0
    agg = (wg.groupby(["penalty", "share_willing"], as_index=False)
           .agg(num_w_plan1=("wait_num_willing_stage1", "sum"),
                num_w_plan2=("wait_num_willing", "sum"),
                num_all_plan1=("wait_num_all_stage1", "sum"),
                num_all_plan2=("wait_num_all", "sum"),
                total_parcels=("total_parcels", "sum"),
                willing_parcels=("willing_parcels", "sum")))
    md = (wg.assign(_m1=wg.mean_days_stage1 * wg.n_cells,
                    _m2=wg.mean_days * wg.n_cells)
          .groupby(["penalty", "share_willing"], as_index=False)
          .agg(_m1=("_m1", "sum"), _m2=("_m2", "sum"),
               _n=("n_cells", "sum"),
               mean_days_plan1_provmean=("mean_days_stage1", "mean"),
               mean_days_plan2_provmean=("mean_days", "mean")))
    md["mean_days_plan1"] = md._m1 / md._n
    md["mean_days_plan2"] = md._m2 / md._n
    md = md.drop(columns=["_m1", "_m2", "_n"])

    out = g.merge(agg, on=["penalty", "share_willing"]).merge(
        md, on=["penalty", "share_willing"])

    out["wait_d_plan1"] = out.num_w_plan1 / out.total_parcels
    out["wait_d_plan2"] = out.num_w_plan2 / out.total_parcels
    out = out.drop(columns=["num_w_plan1", "num_w_plan2"])

    for pl in ("plan1", "plan2"):
        out[f"routing_saving_{pl}_pct"] = (
            100 * (base["routing_eur"] - out[f"routing_cost_{pl}_eur"])
            / base["routing_eur"])
        out[f"operator_saving_{pl}_pct"] = (
            100 * (base["operator_eur"] - out[f"operator_cost_{pl}_eur"])
            / base["operator_eur"])
        out[f"hub_peak_{pl}_vs_base_pct"] = (
            100 * (out[f"sum_hub_peak_{pl}"] - base["sum_hub_peak"])
            / base["sum_hub_peak"])
        out[f"vehicle_days_{pl}_vs_base_pct"] = (
            100 * (out[f"vehicle_days_{pl}"] - base["vehicle_days"])
            / base["vehicle_days"])
    return out.sort_values(["penalty", "share_willing"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────
# Fleet: system profile, CV, hub peaks, flat-profile bound
# ─────────────────────────────────────────────────────────────────────────
def fleet_aggregates(fleet: pd.DataFrame) -> pd.DataFrame:
    """Per (P, theta) fleet aggregates of the FINAL (stage-2) plan.

    ``sys_cv`` is the Mon--Sat coefficient of variation of the SYSTEM fleet
    (all providers and hubs summed per weekday), ``flat_bound`` is
    ``sum_h ceil(vehicle_days_h / 6)`` -- the peak a perfectly flat weekly
    profile would need -- and ``peak_gap`` is how far the achieved
    ``sum_hub_peak`` sits above it.
    """
    day = (fleet.groupby(["penalty", "share_willing", "day"], as_index=False)
           .fleet.sum())
    rows = []
    for (P, th), gd in day.groupby(["penalty", "share_willing"]):
        v = gd.sort_values("day").fleet.values.astype(float)
        assert len(v) == N_DAYS, f"(P={P}, th={th}) has {len(v)} weekdays"
        rows.append(dict(penalty=P, share_willing=th,
                         sys_peak=float(v.max()),
                         sys_total=float(v.sum()),
                         sys_cv=float(v.std() / v.mean()),
                         **{f"sys_{WEEKDAYS[i]}": float(v[i])
                            for i in range(N_DAYS)}))
    prof = pd.DataFrame(rows)

    hub = fleet.groupby(["penalty", "share_willing", "provider", "hub"]).fleet
    peak = (hub.max().groupby(["penalty", "share_willing"]).sum()
            .rename("sum_hub_peak_fleet").reset_index())
    bound = (hub.sum().apply(lambda x: math.ceil(x / N_DAYS))
             .groupby(["penalty", "share_willing"]).sum()
             .rename("flat_bound").reset_index())
    out = prof.merge(peak, on=["penalty", "share_willing"]).merge(
        bound, on=["penalty", "share_willing"])
    out["peak_gap"] = out.sum_hub_peak_fleet - out.flat_bound
    out["peak_gap_eur"] = out.peak_gap * WEEK_FIXED_COST_EUR
    return out.sort_values(["penalty", "share_willing"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────
# Run-2 operator-cost reconstruction (pre-6e schema, no operator columns)
# ─────────────────────────────────────────────────────────────────────────
def reconstruct_operator_cost(costs: pd.DataFrame, fleet: pd.DataFrame,
                              routing_col: str = "cost_stage3_eur"
                              ) -> pd.DataFrame:
    """Operator cost of a pre-6e run from its routing cost + fleet table.

    ``operator = routing - 189.15 * vehicle_days + 1134.90 * sum_h peak_h``

    The routing cost already contains ``189.15`` for *every* vehicle-day;
    the operator pays that fixed bill once per week per **hub peak**
    vehicle, so the per-day charge is removed and the weekly one added back.
    Returns per (penalty, share_willing): ``routing_eur``,
    ``vehicle_days``, ``sum_hub_peak``, ``variable_eur``, ``operator_eur``.

    **Assumption, unverifiable inside a pre-6e run.** ``routing_col`` and the
    fleet CSV must describe the SAME plan.  For run 2 that means
    ``cost_stage3_eur`` against a fleet table written after stage 3 -- run 2's
    schema carries no ``vehicle_days`` / ``sum_hub_peak`` column, so there is
    nothing to cross-check it against in-file.  It is corroborated
    externally: the reconstruction reproduces the controller's independently
    computed ``run2_op`` column to 1e-6 at all four ablation points.  On a
    v5-schema run the caller CAN and does check the fleet-derived hub peak
    against ``sum_hub_peak``.
    """
    assert routing_col in costs.columns, f"missing {routing_col}"
    rout = (costs.groupby(["penalty", "share_willing"], as_index=False)[
        routing_col].sum().rename(columns={routing_col: "routing_eur"}))
    vd = (fleet.groupby(["penalty", "share_willing"], as_index=False)
          .fleet.sum().rename(columns={"fleet": "vehicle_days"}))
    hp = (fleet.groupby(["penalty", "share_willing", "provider", "hub"])
          .fleet.max().groupby(["penalty", "share_willing"]).sum()
          .rename("sum_hub_peak").reset_index())
    out = rout.merge(vd, on=["penalty", "share_willing"]).merge(
        hp, on=["penalty", "share_willing"])
    out["variable_eur"] = out.routing_eur - FIXED_COST_EUR * out.vehicle_days
    out["operator_eur"] = (out.variable_eur
                           + WEEK_FIXED_COST_EUR * out.sum_hub_peak)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Per-LSP knee P* on the (saving, wait) front
# ─────────────────────────────────────────────────────────────────────────
CARRIER_CLASS_BY_PSTAR = {0.25: "Service-bound", 0.5: "Hybrid",
                          0.75: "Cost-aggressive"}
SUBMISSION_P_STAR = {"Amazon": 0.25, "DHL": 0.25,
                     "FedEx": 0.5, "Hermes": 0.5, "UPS": 0.5,
                     "DPD": 0.75, "GLS": 0.75}


def lsp_knees(costs: pd.DataFrame, wait: pd.DataFrame, *, cost_col: str,
              wait_num_col: str, theta_target: float = 1.0) -> pd.DataFrame:
    """Chord-distance knee P* per LSP on its own (saving, wait) front.

    Same rule as ``21_pstar_knees_smoothed.py``: normalise saving and wait
    inside the LSP's own range at ``theta_target`` and take the point that
    maximises the signed chord distance ``(s_n - w_n)/sqrt(2)``.

    ``cost_col`` selects the lens AND the plan (``cost_stage1_eur`` /
    ``cost_stage2_eur`` for the routing lens, ``operator_cost_before_eur`` /
    ``operator_cost_eur`` for the operator lens); ``wait_num_col`` must be
    the wait numerator of the SAME plan.
    """
    base = (costs[np.isclose(costs.share_willing, 0.0)]
            .groupby("provider", as_index=False)[cost_col].mean()
            .rename(columns={cost_col: "base_cost"}))
    sub = costs[np.isclose(costs.share_willing, theta_target)].merge(
        base, on="provider", how="left")
    sub["saving_pct"] = 100 * (sub.base_cost - sub[cost_col]) / sub.base_cost

    w = wait[np.isclose(wait.share_willing, theta_target)].copy()
    w["wait_d"] = w[wait_num_col] / w.total_parcels
    m = sub.merge(w[["penalty", "provider", "wait_d"]],
                  on=["penalty", "provider"], how="left")
    assert m.wait_d.notna().all(), "unmatched (penalty, provider) wait rows"

    rows = []
    for lsp, ms in m.groupby("provider"):
        ms = ms.sort_values("penalty").reset_index(drop=True)
        assert len(ms) >= 3, f"{lsp}: only {len(ms)} penalty levels"
        sav = ms.saving_pct.values.astype(float)
        wt = ms.wait_d.values.astype(float)
        s_n = (sav - sav.min()) / (sav.max() - sav.min() + 1e-12)
        w_n = (wt - wt.min()) / (wt.max() - wt.min() + 1e-12)
        d = (s_n - w_n) / np.sqrt(2)
        i = int(np.argmax(d))
        rows.append(dict(provider=lsp, P_star=float(ms.penalty.iloc[i]),
                         saving_pct=float(sav[i]), wait_d=float(wt[i]),
                         chord_dist=float(d[i])))
    out = pd.DataFrame(rows)
    out["carrier_class"] = out.P_star.map(CARRIER_CLASS_BY_PSTAR).fillna(
        "unclassified")
    out["P_star_submission"] = out.provider.map(SUBMISSION_P_STAR)
    out["class_submission"] = out.P_star_submission.map(
        CARRIER_CLASS_BY_PSTAR)
    out["changed"] = ~np.isclose(out.P_star, out.P_star_submission)
    return out.sort_values("provider").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────
# Schedule sizes (frequency mix)
# ─────────────────────────────────────────────────────────────────────────
def schedule_sizes() -> np.ndarray:
    """Delivery-day count of each of the 39 feasible weekly patterns.

    Ordering is ``enumerate_valid_schedules()``; the grid asserts its own
    enumeration equals it, so ``schedule_idx_*`` indexes into this array.
    """
    from batch_delivery.optimization.schedules import enumerate_valid_schedules
    sched = enumerate_valid_schedules()
    assert len(sched) == 39, f"expected 39 patterns, got {len(sched)}"
    return np.array([len(s) for s in sched], dtype=int)


def freq_mix(chosen: pd.DataFrame, idx_col: str,
             sizes: np.ndarray | None = None) -> pd.DataFrame:
    """Share of postal-code areas [%] per delivery frequency, per (P, theta).

    ``MAX_HOLDING_DAYS = 3`` makes a once-weekly pattern infeasible (gap 6),
    so the frequency classes are {2, 3, 4, 5, 6} -- a "1 day/wk" class in a
    legend would be a bug (Kompendium §38.5).
    """
    if sizes is None:
        sizes = schedule_sizes()
    df = chosen[["penalty", "share_willing", idx_col]].copy()
    df["size"] = sizes[df[idx_col].values.astype(int)]
    assert df["size"].between(2, 6).all(), (
        "delivery frequency outside {2..6} -- MAX_HOLDING_DAYS invariant "
        "violated by the chosen schedules")
    cnt = (df.groupby(["penalty", "share_willing", "size"]).size()
           .rename("n").reset_index())
    tot = (cnt.groupby(["penalty", "share_willing"]).n.sum()
           .rename("tot").reset_index())
    out = cnt.merge(tot, on=["penalty", "share_willing"])
    out["pct"] = 100 * out.n / out.tot
    return out


# ─────────────────────────────────────────────────────────────────────────
# md5 helpers for the paper sync (G7)
# ─────────────────────────────────────────────────────────────────────────
def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_copy(src: Path, dst: Path) -> dict:
    """Compare an intended source against its destination copy.

    Returns ``{src, dst, src_md5, dst_md5, status}`` with ``status`` one of
    ``PASS`` (byte-identical), ``FAIL`` (present but different) or
    ``MISSING`` (destination absent).  Never raises on a missing
    destination -- the caller prints the table and exits non-zero.
    """
    src, dst = Path(src), Path(dst)
    src_md5 = md5_of(src) if src.exists() else None
    if src_md5 is None:
        return dict(src=str(src), dst=str(dst), src_md5=None, dst_md5=None,
                    status="SRC_MISSING")
    if not dst.exists():
        return dict(src=str(src), dst=str(dst), src_md5=src_md5,
                    dst_md5=None, status="MISSING")
    dst_md5 = md5_of(dst)
    return dict(src=str(src), dst=str(dst), src_md5=src_md5,
                dst_md5=dst_md5,
                status="PASS" if dst_md5 == src_md5 else "FAIL")


# ─────────────────────────────────────────────────────────────────────────
# Provenance manifest (G7): what was rendered, from which grid, when
# ─────────────────────────────────────────────────────────────────────────
GRID_CSVS = ("tab_costs_v2.csv", "tab_wait_v2.csv",
             "tab_fleet_per_hub_v2.csv", "_tab_chosen_v2.csv")
MANIFEST_NAME = "manifest.json"


def git_head(root: Path) -> str:
    """The current commit, or a marker -- never an exception."""
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return "unknown"
        head = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=30)
        return head + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:                                   # noqa: BLE001
        return "unknown"


def write_manifest(fig_dir: Path, rev_dir: Path, root: Path,
                   figures: list[Path]) -> Path:
    """Record what this render is, so ``71_`` cannot sync the wrong grid.

    Holds the rev dir, the git HEAD, an ISO timestamp, the md5 + mtime of
    every produced PNG/PDF, and the md5 + mtime of the grid CSVs the render
    claims to summarise.  ``71_`` refuses to copy anything that does not
    match, and refuses a render older than the grid it came from.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    grid = {}
    for name in GRID_CSVS:
        f = Path(rev_dir) / name
        if f.exists():
            grid[name] = dict(md5=md5_of(f), mtime=f.stat().st_mtime)
    figs = {}
    for f in sorted(set(Path(x) for x in figures)):
        if f.exists():
            figs[f.name] = dict(md5=md5_of(f), mtime=f.stat().st_mtime,
                                bytes=f.stat().st_size)
    doc = {
        "schema": 1,
        "generator": "scripts/revision/70_figs_tables_v2.py",
        "rev_dir": str(Path(rev_dir).resolve()),
        "git_head": git_head(root),
        "rendered_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "grid_csvs": grid,
        "figures": figs,
    }
    path = fig_dir / MANIFEST_NAME
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + chr(10),
                    encoding="utf-8")
    return path


def read_manifest(path: Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    assert doc.get("schema") == 1, (
        f"{path}: unknown manifest schema {doc.get('schema')!r}")
    for key in ("rev_dir", "git_head", "rendered_utc", "figures",
                "grid_csvs"):
        assert key in doc, f"{path}: manifest lacks {key!r}"
    return doc


def check_manifest_fresh(doc: dict, fig_dir: Path) -> list[str]:
    """Reasons this render must NOT be synced. Empty list = clean.

    Three ways a render goes stale, all of which G7 has to catch BEFORE a
    single byte is copied (Kompendium §38.8):

    1. a grid CSV changed after the render -- the figures no longer show
       what the tables say;
    2. a figure file on disk no longer matches the md5 the manifest
       recorded -- someone edited or replaced it;
    3. a figure is older than the newest grid CSV -- it was carried over
       from an earlier render rather than produced by this one.
    """
    reasons = []
    rev = Path(doc["rev_dir"])
    newest_csv = 0.0
    for name, rec in sorted(doc["grid_csvs"].items()):
        f = rev / name
        if not f.exists():
            reasons.append(f"grid table {name} is gone from {rev}")
            continue
        newest_csv = max(newest_csv, f.stat().st_mtime)
        if md5_of(f) != rec["md5"]:
            reasons.append(
                f"grid table {name} changed since the render "
                "-- re-run 70_ before syncing")
    for name, rec in sorted(doc["figures"].items()):
        f = Path(fig_dir) / name
        if not f.exists():
            reasons.append(f"figure {name} is missing from {fig_dir}")
            continue
        if md5_of(f) != rec["md5"]:
            reasons.append(f"figure {name} changed since the render "
                           "(md5 differs from the manifest)")
        elif newest_csv and f.stat().st_mtime < newest_csv:
            reasons.append(
                f"figure {name} is OLDER than the grid tables it claims to "
                "render -- re-run 70_ before syncing")
    return reasons


def find_manifests(results_root: Path) -> list[Path]:
    """Every render manifest under ``results/*/figures/``, newest first."""
    found = sorted(Path(results_root).glob(f"*/figures/{MANIFEST_NAME}"))
    return sorted(found, key=lambda f: f.stat().st_mtime, reverse=True)


# ─────────────────────────────────────────────────────────────────────────
# LaTeX
# ─────────────────────────────────────────────────────────────────────────
def to_latex(df: pd.DataFrame, path: Path, caption: str, label: str,
             float_format: str = "%.2f", resize: bool = True) -> None:
    """Write a booktabs table.

    ``escape=False``: the column names are already LaTeX (``$P$``,
    ``$\\sum_h$``), and pandas' escaper would turn them into literal
    ``\\$P\\$``.  Keep every cell value free of raw ``&``, ``%`` and ``_``.
    """
    for col in df.columns:
        assert "_" not in str(col) or "$" in str(col), (
            f"column name {col!r} carries a bare underscore and escape=False "
            "is on -- rename it to a display label before calling to_latex")
        if df[col].dtype == object:
            bad = df[col].astype(str).str.contains(r"[&%#]", regex=True)
            assert not bad.any(), (
                f"column {col!r} carries LaTeX-special characters and "
                "escape=False is on: " f"{df[col][bad].tolist()[:3]}")
    body = df.to_latex(index=False, escape=False, float_format=float_format,
                       longtable=False)
    if resize:
        body = ("\\resizebox{\\textwidth}{!}{%\n" + body.rstrip("\n")
                + "\n}\n")
    path.write_text(
        "% generated by scripts/revision/70_figs_tables_v2.py -- do not edit\n"
        f"% {COST_MODEL_TEXT}\n"
        "\\begin{table}[t]\n\\centering\n\\small\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"{body}"
        "\\end{table}\n", encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════
# Per-cell plan costs (Task 13B) -- written by 72_per_cell_costs_v2.py
# ═════════════════════════════════════════════════════════════════════════
PER_CELL_NAME = "tab_per_cell_costs_v2.csv"
PER_CELL_COLS = (
    "penalty", "share_willing", "provider", "plz", "plan", "hub",
    "schedule_idx", "own_cost_eur", "pool_share_eur", "express_share_eur",
    "cell_cost_eur", "cell_parcels_week", "mean_days", "wait_days",
    "veh_days_share", "peak_veh_share", "hub_peak_day",
)
PLAN_OF_KEY = {"stage1": 1, "balanced": 2}


class PerCellMissing(FileNotFoundError):
    """``72_`` has not been run on this grid yet."""


def load_per_cell(rev_dir: Path) -> pd.DataFrame:
    """The per-cell plan-cost table of one grid, schema-checked.

    Raises :class:`PerCellMissing` naming the command that produces it
    rather than a bare ``FileNotFoundError``: the euro panels of fig. 6 have
    no substitute, so the caller has to decide loudly.
    """
    path = Path(rev_dir) / "tables" / PER_CELL_NAME
    if not path.exists():
        raise PerCellMissing(
            f"{path} does not exist. Run\n"
            f"    python scripts/revision/72_per_cell_costs_v2.py "
            f"--rev-dir {rev_dir}\n"
            "first -- the per-PLZ euro panels have no substitute.")
    df = pd.read_csv(path, dtype={"plz": str})
    missing = [c for c in PER_CELL_COLS if c not in df.columns]
    assert not missing, f"{path} lacks {missing} -- re-run 72_"
    bad = set(df.plan.unique()) - set(PLAN_OF_KEY)
    assert not bad, f"{path}: unknown plan value(s) {bad}"
    return df


def per_cell_savings(pc: pd.DataFrame) -> pd.DataFrame:
    """Per-cell euro saving against the cell's own theta = 0 baseline.

    The baseline is the (P = 0, theta = 0) row, which the grid pins to the
    daily schedule for BOTH plans -- asserted here per cell, because a
    baseline that differed by plan would make the two plans' savings
    incomparable.  Returns *pc* with ``baseline_cell_eur``, ``saving_eur``
    and ``saving_pct`` added.
    """
    b = pc[np.isclose(pc.share_willing, 0.0)]
    assert len(b) > 0, "no theta=0 rows -- the per-cell table has no baseline"
    key = ["provider", "plz"]
    spread = (b.groupby(key).cell_cost_eur.max()
              - b.groupby(key).cell_cost_eur.min()).abs()
    assert (spread < 1e-6).all(), (
        "the theta=0 per-cell baseline is not identical across P and plan "
        f"(worst spread {spread.max():.6f} EUR at {spread.idxmax()}) -- "
        "savings would be measured against a moving baseline")
    base = (b.groupby(key, as_index=False).cell_cost_eur.mean()
            .rename(columns={"cell_cost_eur": "baseline_cell_eur"}))
    out = pc.merge(base, on=key, how="left")
    assert out.baseline_cell_eur.notna().all(), (
        "a cell has no theta=0 baseline row")
    assert (out.baseline_cell_eur > 0).all(), (
        "a cell has a zero/negative baseline cost -- a saving percentage "
        "would be meaningless")
    out["saving_eur"] = out.baseline_cell_eur - out.cell_cost_eur
    out["saving_pct"] = 100 * out.saving_eur / out.baseline_cell_eur
    return out


def hub_lens(pc: pd.DataFrame) -> pd.DataFrame:
    """The OPERATOR lens per hub, from the per-cell attribution.

    The weekly fixed bill is sized by a hub's PEAK day, so it is a property
    of the hub, not of any one cell: what a cell "contributes" to it depends
    on when the rest of the hub peaks.  The per-cell peak column is
    therefore only summed to the level where it means something --

        operator = sum cell_cost - 189.15 * sum veh_days
                                 + 1134.90 * sum peak_veh

    -- and the saving is taken against the same hub's theta = 0 operator
    cost.  Hub names are unique per provider, not globally, so the key is
    (provider, hub).
    """
    key = ["penalty", "share_willing", "provider", "hub", "plan"]
    g = pc.groupby(key, as_index=False).agg(
        routing_eur=("cell_cost_eur", "sum"),
        vehicle_days=("veh_days_share", "sum"),
        hub_peak=("peak_veh_share", "sum"),
        parcels_week=("cell_parcels_week", "sum"),
        n_cells=("plz", "nunique"))
    g["variable_eur"] = g.routing_eur - FIXED_COST_EUR * g.vehicle_days
    g["operator_eur"] = g.variable_eur + WEEK_FIXED_COST_EUR * g.hub_peak
    b = g[np.isclose(g.share_willing, 0.0)]
    base = (b.groupby(["provider", "hub"], as_index=False)
            .agg(baseline_operator_eur=("operator_eur", "mean"),
                 baseline_routing_eur=("routing_eur", "mean")))
    out = g.merge(base, on=["provider", "hub"], how="left")
    assert out.baseline_operator_eur.notna().all(), (
        "a hub has no theta=0 baseline row")
    out["operator_saving_eur"] = out.baseline_operator_eur - out.operator_eur
    out["operator_saving_pct"] = (100 * out.operator_saving_eur
                                  / out.baseline_operator_eur)
    out["routing_saving_eur"] = out.baseline_routing_eur - out.routing_eur
    return out


#: the runner's tolerance semantics (61_grid_run_v2._tol), so the gate
#: re-run from the written CSVs has exactly the window 72_ used in
#: memory: a tenth of a cent absolute floor, 1e-9 relative.
PER_CELL_ABS_TOL = 1e-3
PER_CELL_REL_TOL = 1e-9


def check_per_cell_against_grid(pc: pd.DataFrame, costs: pd.DataFrame,
                                rtol: float = PER_CELL_REL_TOL,
                                atol: float = PER_CELL_ABS_TOL
                                ) -> pd.DataFrame:
    """Re-run 72_'s identity gate from the two CSVs alone.

    ``72_`` asserts it while the matrices are in memory; this repeats it
    from the written files, so a figure can never be drawn from a per-cell
    table that does not add up to the grid it claims to decompose.  Returns
    the per (P, theta, provider, plan) deltas.
    """
    ref_col = {("stage1", "cost"): "cost_stage1_eur",
               ("stage1", "vd"): "vehicle_days_before",
               ("stage1", "peak"): "sum_hub_peak_before",
               ("balanced", "cost"): "cost_stage2_eur",
               ("balanced", "vd"): "vehicle_days",
               ("balanced", "peak"): "sum_hub_peak"}
    got = (pc.groupby(["penalty", "share_willing", "provider", "plan"],
                      as_index=False)
           .agg(cost=("cell_cost_eur", "sum"), vd=("veh_days_share", "sum"),
                peak=("peak_veh_share", "sum")))
    rows = []
    for _, r in got.iterrows():
        c = costs[np.isclose(costs.penalty, r.penalty)
                  & np.isclose(costs.share_willing, r.share_willing)
                  & (costs.provider == r.provider)]
        assert len(c) == 1, (
            f"({r.penalty}, {r.share_willing}, {r.provider}): "
            f"{len(c)} grid row(s)")
        c = c.iloc[0]
        d = dict(penalty=r.penalty, share_willing=r.share_willing,
                 provider=r.provider, plan=r.plan)
        for what in ("cost", "vd", "peak"):
            want = float(c[ref_col[(r.plan, what)]])
            d[f"{what}_delta"] = float(r[what]) - want
            d[f"{what}_ref"] = want
        rows.append(d)
    out = pd.DataFrame(rows)
    for what in ("cost", "vd", "peak"):
        w = np.maximum(atol,
                       rtol * np.abs(out[f"{what}_ref"].to_numpy()))
        bad = out[np.abs(out[f"{what}_delta"]) > w]
        assert bad.empty, (
            f"per-cell IDENTITY GATE failed on {what} for {len(bad)} "
            f"(P, theta, provider, plan) group(s), worst "
            f"{bad[f'{what}_delta'].abs().max():.6e} -- e.g. "
            f"{bad.iloc[0].to_dict()}")
    return out


# ---------------------------------------------------------------------
# Structural facts: the penalty/theta-independent PLZ features
# ---------------------------------------------------------------------
RAUMTYP_ORDER = ["urban", "suburban", "rural"]


def region_types(root: Path) -> pd.DataFrame:
    """``plz -> raumtyp_3`` (urban / suburban / rural).

    PLZ level, not cluster level: the grid's cell keys are PLZ codes and all
    of them are in ``data/geodata/plz_raumtyp.csv``; the cluster file is
    keyed on cluster heads and would leave a third of them unmatched.
    """
    p = Path(root) / "data" / "geodata" / "plz_raumtyp.csv"
    rt = pd.read_csv(p, dtype={"plz": str})[["plz", "raumtyp_3"]]
    bad = set(rt.raumtyp_3.unique()) - set(RAUMTYP_ORDER)
    assert not bad, f"{p}: unknown region type(s) {bad}"
    return rt


def median_by_bucket(df: pd.DataFrame, value: str, bucket: str,
                     by=("share_willing",)) -> pd.DataFrame:
    """Median + IQR + n of *value* per bucket and theta.

    ``n`` counts the (provider, plz) CELLS -- or (provider, hub) hubs -- in
    the bucket, not the rows: a PLZ code is served by several LSPs.
    """
    unit = ["provider", "plz"] if "plz" in df.columns else ["provider", "hub"]
    g = df.groupby([*by, bucket], observed=True)
    out = g[value].agg(med="median", lo=lambda s: s.quantile(0.25),
                       hi=lambda s: s.quantile(0.75)).reset_index()
    n = (df.groupby([*by, bucket], observed=True)[unit]
         .apply(lambda d: d.drop_duplicates().shape[0])
         .rename("n").reset_index())
    return out.merge(n, on=[*by, bucket], how="left")


def bucket_composition(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    """n and the carrier composition of every bucket of a split.

    Region Hannover has exactly one multi-depot network, so a structural
    bucket can be one carrier's depots in disguise; a legend that hides that
    invites a causal reading it cannot support (Task 13 review I-1).
    """
    unit = ["provider", "plz"] if "plz" in df.columns else ["provider", "hub"]
    rows = []
    for b, g in df.groupby(bucket, observed=True):
        provs = sorted(g.provider.unique())
        rows.append(dict(
            feature=bucket, bucket=b,
            n=int(g[unit].drop_duplicates().shape[0]),
            providers=", ".join(provs), single_carrier=len(provs) == 1))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Head usage (Task 11) -- occurrence-weighted, never a mean of ratios
# ---------------------------------------------------------------------
HEAD_USAGE_NAME = "tab_head_usage_v2.csv"


def head_usage(rev_dir: Path):
    """``tab_head_usage_v2.csv``, or None for a pre-Task-11 grid."""
    p = Path(rev_dir) / HEAD_USAGE_NAME
    if not p.exists():
        return None
    return pd.read_csv(p)


def _ratio(num, den):
    """``num/den`` with NaN -- not 0 -- where nothing pooled.

    A (P, theta, provider) that realised no pooled tour has no head share:
    reporting 0 would say "the head priced none of it", which is a different
    statement from "there was nothing to price" (Task 11 concern 3).
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.where(den > 0, num / np.where(den > 0, den, np.nan), np.nan)


def head_usage_summary(hu: pd.DataFrame, by=("provider", "kind")
                       ) -> pd.DataFrame:
    """Head-priced share of pooled COST and of TOURS, occurrence-weighted.

    Sums the numerators and denominators and divides once.  Averaging the
    per-row ``head_cost_share`` would weight a 200-EUR hub-day the same as a
    60 000-EUR one and would silently drop the NaN rows.
    """
    g = hu.groupby(list(by), as_index=False).agg(
        pooled_cost_eur=("pooled_cost_eur", "sum"),
        head_cost_eur=("head_cost_eur", "sum"),
        n_groups=("n_groups_priced", "sum"),
        n_multi_cell=("n_multi_cell_groups", "sum"),
        n_head=("n_head", "sum"),
        n_fallback_uncertified=("n_fallback_uncertified", "sum"),
        n_fallback_unsupported=("n_fallback_unsupported", "sum"),
        n_fallback_no_head=("n_fallback_no_head", "sum"))
    g["head_cost_share"] = _ratio(g.head_cost_eur, g.pooled_cost_eur)
    g["head_group_share"] = _ratio(g.n_head, g.n_groups)
    g["head_share_of_multi_cell"] = _ratio(g.n_head, g.n_multi_cell)
    return g


# ---------------------------------------------------------------------
# v5 -> v6: what the bundle head changed
# ---------------------------------------------------------------------
def pooled_cost_totals(costs: pd.DataFrame) -> pd.DataFrame:
    """Per (P, theta): the pooled (express + small-delivery) cost.

    Present in both schemas: ``express_cost_eur`` and ``pool_cost_eur`` are
    written by the runner at the FINAL (stage-2) plan in v5 and v6 alike.
    ``head_cost_eur`` only exists from Task 11 on; a head-free grid priced
    exactly none of its pooled cost with a head, so 0.0 there is the fact,
    not a fill value.
    """
    missing = [c for c in ("express_cost_eur", "pool_cost_eur")
               if c not in costs.columns]
    assert not missing, (
        f"tab_costs_v2.csv lacks {missing} -- every v5/v6 grid writes the "
        "express/pooled cost split; without it the pooled total cannot be "
        "formed")
    agg = dict(express_cost_eur=("express_cost_eur", "sum"),
               pool_cost_eur=("pool_cost_eur", "sum"))
    if "head_cost_eur" in costs.columns:
        agg["head_cost_eur"] = ("head_cost_eur", "sum")
    g = costs.groupby(["penalty", "share_willing"], as_index=False).agg(**agg)
    g["pooled_cost_eur"] = g.express_cost_eur + g.pool_cost_eur
    if "head_cost_eur" not in g.columns:
        g["head_cost_eur"] = 0.0
    g["head_cost_share"] = _ratio(g.head_cost_eur, g.pooled_cost_eur)
    return g


def grid_delta(full_a: pd.DataFrame, full_b: pd.DataFrame,
               costs_a: pd.DataFrame, costs_b: pd.DataFrame,
               label_a: str, label_b: str,
               theta: float = 1.0) -> pd.DataFrame:
    """What changed between two grids at one theta, per P.

    ``full_*`` are ``headline_rows`` frames, i.e. savings already taken
    against each grid's OWN baseline -- v6's head prices the theta = 0
    pooled tours too, so its baseline is not v5's and the two must never be
    mixed.  Savings are compared in PERCENTAGE POINTS, costs in euro.
    """
    keep = ["penalty", "routing_saving_plan1_pct", "routing_saving_plan2_pct",
            "operator_saving_plan1_pct", "operator_saving_plan2_pct",
            "sum_hub_peak_plan2", "wait_d_plan2", "mean_days_plan2"]
    a = full_a[np.isclose(full_a.share_willing, theta)][keep]
    b = full_b[np.isclose(full_b.share_willing, theta)][keep]
    m = a.merge(b, on="penalty", suffixes=("_a", "_b"))
    assert len(m) == len(a) == len(b), (
        f"the two grids do not cover the same penalties at theta={theta}")
    pa = pooled_cost_totals(costs_a)
    pb = pooled_cost_totals(costs_b)
    pa = pa[np.isclose(pa.share_willing, theta)]
    pb = pb[np.isclose(pb.share_willing, theta)]
    cols = ["penalty", "pooled_cost_eur", "head_cost_eur", "head_cost_share"]
    m = (m.merge(pa[cols], on="penalty")
         .merge(pb[cols], on="penalty", suffixes=("_a", "_b")))
    out = pd.DataFrame({"penalty": m.penalty})
    out[f"pooled_{label_a}_eur"] = m.pooled_cost_eur_a
    out[f"pooled_{label_b}_eur"] = m.pooled_cost_eur_b
    out["pooled_delta_eur"] = m.pooled_cost_eur_b - m.pooled_cost_eur_a
    out[f"head_share_{label_b}"] = m.head_cost_share_b
    for col in ("routing_saving_plan1_pct", "routing_saving_plan2_pct",
                "operator_saving_plan1_pct", "operator_saving_plan2_pct"):
        out[f"{col}_{label_a}"] = m[f"{col}_a"]
        out[f"{col}_{label_b}"] = m[f"{col}_b"]
        out[f"{col}_delta_pp"] = m[f"{col}_b"] - m[f"{col}_a"]
    out["sum_hub_peak_plan2_delta"] = (m.sum_hub_peak_plan2_b
                                       - m.sum_hub_peak_plan2_a)
    out["wait_d_plan2_delta"] = m.wait_d_plan2_b - m.wait_d_plan2_a
    return out.sort_values("penalty").reset_index(drop=True)


# ---------------------------------------------------------------------
# CO2 from validated route kilometres (Kompendium 38.6)
# ---------------------------------------------------------------------
#: kg CO2 per vehicle-km. EXTERNAL ASSUMPTION, not a model result -- the
#: same factor 38.6 states for the submitted table, kept so the regenerated
#: numbers are comparable with it.
CO2_KG_PER_KM = 0.25
VALIDATION_NAME = "tab_vroom_v2.csv"
#: the full list of instances the validation set out to solve.  A
#: kilometre total is only a weekly mileage if every one of them is in
#: the result -- a validation still running looks exactly like a
#: finished one otherwise, and understates the week.
VALIDATION_QUEUE_NAME = "instance_queue.csv"


class Co2Unavailable(RuntimeError):
    """No validated route kilometres for this grid -- refuse, never carry."""


def _assert_validation_complete(val_dir: Path, v: pd.DataFrame) -> None:
    """Refuse a validation whose solved points are not finished.

    ``tab_vroom_v2.csv`` is appended as instances are solved, so a run in
    progress is indistinguishable from a finished one by shape alone -- and
    summing its kilometres gives a weekly mileage that is simply too small
    (measured on the v6 grid mid-run: 123 685 km against 224 678 km for the
    same operating point on a complete validation, a 45 % understatement no
    other check would have caught). ``instance_queue.csv`` is the full list
    the validation set out to solve, so compare against it.

    Only points that ARE in the result are compared. A queued (plan, P,
    theta) with no solved instance at all is simply a point this validation
    never ran; it does not appear in the CO2 table and is not a defect --
    v5, for instance, queued two such points and completed the six it did
    run. A point that is PARTLY solved is the dangerous one.
    """
    q_path = Path(val_dir) / VALIDATION_QUEUE_NAME
    if not q_path.exists():
        return                      # nothing to compare against
    q = pd.read_csv(q_path)
    key = ["plan", "penalty", "share_willing"]
    if not set(key) <= set(q.columns):
        return
    got = v.groupby(key).size().rename("solved")
    want = q.groupby(key).size().rename("queued")
    cmp = pd.concat([want, got], axis=1)
    cmp = cmp[cmp.solved.notna()]        # only points this run produced
    short = cmp[cmp.solved < cmp.queued]
    if short.empty:
        return
    rows = "; ".join(
        f"{k}: {int(r.solved)}/{int(r.queued)}"
        for k, r in short.iterrows())
    raise Co2Unavailable(
        f"{val_dir / VALIDATION_NAME} is INCOMPLETE -- "
        f"{int(short.solved.sum()):,d} of {int(short.queued.sum()):,d} "
        f"queued instance(s) are solved in {len(short)} of {len(cmp)} "
        "validated point(s). A kilometre sum over a validation that is "
        "still running is not the plan's weekly mileage, it is a fraction "
        "of it, and nothing in the file says so. Short: "
        f"{rows}. Wait for the validation to finish, then re-run."
    )


def co2_table(rev_dir: Path, factor: float = CO2_KG_PER_KM) -> pd.DataFrame:
    """km/week and t CO2/week per validated (plan, P, theta), from VROOM.

    Built from ``<rev>/validation/tab_vroom_v2.csv`` -- **actual** solved
    route kilometres, never surrogate cost, because 38.6's quantity is
    kilometres.  ``vs_least_consolidated_pct`` is the change against the
    least consolidated validated point OF THE SAME PLAN (the highest P),
    which is 38.6's own reference: there is no VROOM baseline solve of daily
    delivery on this allocation, so a "vs daily baseline" column would be an
    invention.

    Raises :class:`Co2Unavailable` when the grid has no validation data.
    There is deliberately no fallback: the submitted table belongs to a
    different grid, a different plan and a different cost path, so carrying
    it over silently is exactly the failure 38.8 is about.
    """
    path = Path(rev_dir) / "validation" / VALIDATION_NAME
    if not path.exists():
        raise Co2Unavailable(
            f"{path} does not exist, so this grid has no validated route "
            "kilometres. 38.6 is built from real VROOM km at the validated "
            "operating points; the surrogate cost cannot substitute for it "
            "(euro, not km), and the submitted table describes a different "
            "grid and a different plan. Run Task 12's validation on this "
            "grid first (67_validate_vroom_v2.py) -- do NOT carry the old "
            "numbers over.")
    v = pd.read_csv(path)
    for col in ("penalty", "share_willing", "plan", "vroom_distance_km",
                "vroom_status"):
        assert col in v.columns, f"{path} lacks {col!r}"
    assert v.vroom_distance_km.notna().all(), (
        f"{path}: {int(v.vroom_distance_km.isna().sum())} row(s) have no "
        "distance -- an unsolved instance cannot be counted as 0 km")
    if "g6_selected" in v.columns:
        sel = v.g6_selected.astype(bool)
        assert sel.all(), (
            f"{path}: {int((~sel).sum())} of {len(v)} row(s) are not "
            "g6_selected -- this validation is a SAMPLE, so its kilometre "
            "sum is not the plan's weekly mileage")
    _assert_validation_complete(path.parent, v)

    status = v.vroom_status.astype(str).str.upper()
    flagged = ~status.isin(["OK", "CACHED"])
    g = (v.assign(_flag=flagged.astype(int))
         .groupby(["plan", "penalty", "share_willing"], as_index=False)
         .agg(n_instances=("vroom_distance_km", "size"),
              km_week=("vroom_distance_km", "sum"),
              n_routes=("vroom_n_routes", "sum"),
              n_flagged=("_flag", "sum")))
    g["co2_t_week"] = g.km_week * factor / 1000.0
    ref = (g.sort_values("penalty").groupby("plan")
           .agg(ref_km=("km_week", "last"), ref_P=("penalty", "last"))
           .reset_index())
    g = g.merge(ref, on="plan", how="left")
    g["vs_least_consolidated_pct"] = 100 * (g.km_week - g.ref_km) / g.ref_km
    g["co2_kg_per_km"] = factor
    return g.sort_values(["plan", "penalty"]).reset_index(drop=True)


def co2_decision(rev_dir: Path) -> str:
    """One line saying whether the CO2 table can be regenerated here."""
    try:
        t = co2_table(rev_dir)
    except Co2Unavailable as exc:
        return f"REFUSED -- {exc}"
    plans = ", ".join(sorted(t.plan.unique()))
    return (f"REGENERATED from {len(t)} validated point(s) "
            f"(plans: {plans}) at {CO2_KG_PER_KM} kg CO2/vehicle-km "
            "(external assumption, Kompendium 38.6).")
