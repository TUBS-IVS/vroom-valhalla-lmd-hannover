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
import math
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
LENS_ROUTING = "routing lens"
LENS_OPERATOR = "operator lens"


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
        "wait_num_willing_stage1", "mean_days", "mean_days_stage1",
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

    # 2. the theta=0 baseline is identical for every P
    b = (costs[np.isclose(costs.share_willing, 0.0)]
         .groupby("penalty")[["cost_stage1_eur", "cost_stage2_eur",
                              "operator_cost_eur", "sum_hub_peak",
                              "vehicle_days", "variable_cost_eur"]].sum())
    assert len(b) >= 1, f"{label}: no theta=0 rows -- no baseline"
    spread = (b.max() - b.min()).abs()
    assert (spread < 1e-6).all(), (
        f"{label}: the theta=0 baseline is NOT identical across P "
        f"(max spread per column: {spread.to_dict()}) -- "
        "the two-plan tables would be measured against a moving baseline")

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


def wait_metric(wait: pd.DataFrame, num_col: str = "wait_num_willing"
                ) -> pd.DataFrame:
    """Parcel-weighted additional wait per (P, theta) for one plan."""
    g = (wait.groupby(["penalty", "share_willing"], as_index=False)
         .agg(_num=(num_col, "sum"), _den=("total_parcels", "sum")))
    g["wait_d"] = g._num / g._den
    return g.drop(columns=["_num", "_den"])


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
