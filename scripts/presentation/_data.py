"""Stage-3 data access for the presentation figure set, with provenance.

Every loader here goes through `_read()`, which records the file it touched
together with that file's size and modification time. Figure scripts call
`prov.write(name, ...)` at the end, dropping a JSON sidecar into
`results/presentation_2026_08/_prov/`. 90_build_manifest.py assembles those
sidecars into MANIFEST.md, so the manifest's provenance table is generated from
what the code actually read rather than from a hand-maintained list.

All paths are resolved once, at import, and asserted to exist -- the failure
mode this whole module exists to prevent is a figure script silently reading a
stale pre-refactor path (the reason most of scripts/figures/*.py no longer
runs).

Two grid schemas
----------------
The 2026-08 revision replaced the grid. Where the submission-era run
(`results/revision_2026_07`, "legacy") carried ONE plan and ONE cost lens, the
revision grid (`results/revision_2026_08_v*`, "v2") carries **two plans**
(routing-optimal stage 1 and operator-polished stage 2) in **two lenses**
(routing euro and operator euro). `REV` is chosen by `$PRES_REV_DIR` and
defaults to the v2 grid; `SCHEMA` says which one was found, and every loader
below dispatches on it.

Three groups of tables deliberately do NOT follow `REV`, because the analysis
behind them was not re-run and pointing them at the revision directory would
silently mean something else:

* `load_alpha_sensitivity()` / `load_cd_restart_spread()` — model diagnostics,
  unchanged by the revision;
* the VROOM validation loaders — the revision's validation (Task 12) is a
  different schema and, at the time of writing, still being produced;
* `load_per_plz()` — the per-area euro decomposition, which only exists on the
  legacy grid. Its v2 replacement needs the per-cell plan-cost columns of
  Task 11; until those land, `per_plz_eur_available()` is False and callers
  must fall back to the frequency-based structural view AND say so.

All three read `REV_LEGACY` explicitly, so the split is visible in the code
rather than hidden in a default.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _style as _S

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "results" / "presentation_2026_08"
PROV_DIR = OUT_ROOT / "_prov"

# ── which grid the figures and decks read ──────────────────────────────────
# The submission-era grid. Frozen: three analyses below read it by name.
REV_LEGACY = ROOT / "results" / "revision_2026_07"
# The revision grid. Provisional until the v6 head grid replaces it; that is
# the whole reason PRES_REV_DIR exists, so Part B is an environment variable
# rather than an edit.
REV_V2_DEFAULT = ROOT / "results" / "revision_2026_08_v5"

SCHEMA_V2 = "v2"
SCHEMA_LEGACY = "legacy"


def _detect_schema(rev: Path) -> str:
    """v2 if the directory carries the two-plan tables, else legacy."""
    if (rev / "tab_costs_v2.csv").exists():
        return SCHEMA_V2
    if (rev / "tab_costs_smoothed.csv").exists():
        return SCHEMA_LEGACY
    raise FileNotFoundError(
        f"{rev} is neither a v2 grid (tab_costs_v2.csv) nor a legacy grid "
        f"(tab_costs_smoothed.csv)")


def set_rev_dir(path) -> Path:
    """Point the revision loaders at another grid directory.

    Called by the `--rev-dir` flag of the deck builders. Rebinds the module
    globals rather than threading a handle through every loader, because the
    figure scripts read `D.REV` directly; call it before the first load.
    """
    global REV, VAL, SCHEMA
    REV = Path(path).resolve()
    VAL = REV / "validation"
    SCHEMA = _detect_schema(REV)
    _CACHE.clear()
    return REV


REV = Path(os.environ.get("PRES_REV_DIR") or REV_V2_DEFAULT).resolve()
SCHEMA = _detect_schema(REV)
VAL = REV / "validation"
_CACHE: dict = {}
RUN = ROOT / "results" / "runs" / "path2_2026_05_29"
SUPP = ROOT / "results" / "supplementary"
GEO = ROOT / "data" / "geodata"
CKPT = ROOT / "results" / "checkpoints"
LEGACY_OUT = ROOT / "results" / "paper_outputs_2026_05_30"

PROVIDERS = ["DHL", "Amazon", "DPD", "FedEx", "GLS", "Hermes", "UPS"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
N_DAYS = 6

# Pinned baseline: weekly direct-delivery cost of the unbatched (daily) system.
# Asserted against tab_baseline_per_provider.csv in load_baseline_per_provider().
BASE_TOTAL = 1909747.75

# ── the revision's two plans and two cost lenses ────────────────────────────
# PLAN is *which weekly schedule* the number describes; LENS is *what a euro
# means*. They are independent, and the deck must never collapse them: panel
# (a) of the revision's fig 5 is (routing plan, routing lens) and panel (b) is
# (operator plan, operator lens), which is two changes at once.
PLAN_ROUTING = "routing"      # stage 1: the routing-cost optimum
PLAN_OPERATOR = "operator"    # stage 2: the operator-cost polish of stage 1
PLANS = (PLAN_ROUTING, PLAN_OPERATOR)
PLAN_LABEL = {
    PLAN_ROUTING: "routing-optimal plan (stage 1)",
    PLAN_OPERATOR: "operator-polished plan (stage 2)",
}

LENS_ROUTING = "routing"      # what the tours cost to drive
LENS_OPERATOR = "operator"    # what an operator with salaried drivers pays
LENSES = (LENS_ROUTING, LENS_OPERATOR)
LENS_LABEL = {
    LENS_ROUTING: "routing euro",
    LENS_OPERATOR: "operator euro",
}
# Compendium 40.11/40.12. FIXED_COST_EUR = 189.15 is per vehicle and DAY and
# already contains the driver; an operator staffs each hub for its weekly peak,
# so below the peak only the kilometres are a real outlay.
FIXED_COST_EUR = 189.15
COST_PER_KM_EUR = 0.3864
COST_PER_ROUTE_HOUR_EUR = 36.0          # VROOM per_hour default, see 40.12
WEEK_FIXED_EUR = 6 * FIXED_COST_EUR     # 1 134.90 per peak vehicle and hub
LENS_FORMULA = {
    LENS_ROUTING: "direct tours + express tours + pooled tours",
    LENS_OPERATOR: "variable km cost + 1 134.90 EUR per peak vehicle and hub",
}
COST_MODEL_SENTENCE = (
    "189.15 EUR per vehicle-day + 0.3864 EUR/km + 36 EUR per route-hour")

# Pinned v5 baselines (P = 0, theta = 0: every area served daily). Asserted
# against the grid in `baseline_v2()`, so a re-run that moves them fails the
# build instead of quietly restating every saving on the slides.
BASE_ROUTING_V2 = 1_909_432.42
BASE_OPERATOR_V2 = 2_109_742.27
BASE_HUB_PEAK_V2 = 1239
BASE_VEHICLE_DAYS_V2 = 6375
BASE_FLEET_CV_V2 = 0.139                # 40.18; 0.135 in the submission

# Cells for which real VROOM routes were re-solved on the Stage-3 schedules.
VROOM_CELLS = [(0.0, 1.0), (0.25, 1.0), (0.5, 1.0), (0.75, 1.0)]

# Fuel/emission factor for the CO2 figure. Van diesel, HBEFA-class urban
# delivery: 0.25 kg CO2 per vehicle-km. Stated in the figure caption because it
# is an external assumption, not a model output.
CO2_KG_PER_KM = 0.25

PALETTE = {
    "baseline": _S.STAGE["baseline"],
    "stage2": _S.STAGE["stage2"],
    "stage3": _S.STAGE["stage3"],
    "accent": "#e76f51",
    "accent2": "#2a9d8f",
    "warn": "#e9c46a",
}
# Settlement type and delivery frequency follow the paper's figures 6 and 4.
RT_ORDER = ["urban", "suburban", "rural"]
RT_COLOR = _S.RAUMTYP

# Delivery frequencies run 2..6, never 1: MAX_HOLDING_DAYS = 3 makes a
# once-weekly schedule inadmissible (its gap of 6 days exceeds the cap), and
# enumerate_schedules() accordingly yields sizes {2:3, 3:14, 4:15, 5:6, 6:1}.
# Listing a 1 day/wk class would invent a category the model cannot produce.
FREQ_SIZES = [2, 3, 4, 5, 6]
FREQ_COLOR = _S.FREQ

# Providers use the paper's PROV_COLOR, so a provider is the same colour in the
# deck as in figures 1 and 3 of the manuscript.
PROVIDER_COLOR = dict(_S.PROVIDER)
PROVIDER_FILL = _S.STAGE["stage2"]

# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
class Provenance:
    def __init__(self) -> None:
        self.sources: dict[str, dict] = {}

    def record(self, path: Path) -> Path:
        p = Path(path)
        # PRES_REV_DIR may point outside the repository -- a scratch copy of a
        # grid, a network share, a test fixture. Provenance is a record of what
        # was read, so an outside path is recorded absolutely rather than
        # raising; only a path inside the repo gets the short relative key.
        try:
            key = str(p.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            key = str(p.resolve()).replace("\\", "/")
        if key not in self.sources:
            st = p.stat()
            self.sources[key] = {
                "bytes": st.st_size,
                "mtime": _dt.datetime.fromtimestamp(st.st_mtime)
                .strftime("%Y-%m-%d %H:%M"),
            }
        return p

    def write(self, name: str, *, title: str, tier: str, act: str,
              basis: str, claim: str, caveats: str = "") -> Path:
        PROV_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": name,
            "title": title,
            "tier": tier,
            "act": act,
            "basis": basis,
            "claim": claim,
            "caveats": caveats,
            "script": str(Path(sys.argv[0]).resolve().relative_to(ROOT))
            .replace("\\", "/"),
            "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "git_commit": _git_head(),
            "sources": self.sources,
        }
        p = PROV_DIR / f"{name}.json"
        p.write_text(json.dumps(payload, indent=2))
        return p


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=15,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


prov = Provenance()


def _read(path: Path, **kw) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"stage-3 input missing: {p}\n"
            f"  (presentation figures must not fall back to pre-refactor paths)"
        )
    prov.record(p)
    return pd.read_csv(p, **kw)


# --------------------------------------------------------------------------
# v2 grid: raw tables
# --------------------------------------------------------------------------
def _v2_only(what: str) -> None:
    if SCHEMA != SCHEMA_V2:
        raise RuntimeError(
            f"{what} needs a v2 revision grid; REV is {REV} ({SCHEMA}). "
            f"Set PRES_REV_DIR (or pass --rev-dir) to a directory containing "
            f"tab_costs_v2.csv.")


def _cached(key: str, make):
    if key not in _CACHE:
        _CACHE[key] = make()
    return _CACHE[key]


def load_costs_v2() -> pd.DataFrame:
    """`tab_costs_v2.csv` verbatim: per (P, theta, provider), both plans.

    The grid's own convention (61_grid_run_v2.py): a plain column name is the
    STAGE-2 / final plan, the stage-1 value carries `_stage1` or lives in a
    `*_before` column. `cost_stage1_eur` and `routing_total_eur` are both pure
    routing euro -- neither carries the service penalty, which sits in
    `penalty_eur` / `penalty_before_eur` -- so the routing lens compares like
    with like across the two plans.
    """
    _v2_only("load_costs_v2()")
    return _cached("costs_v2", lambda: _read(REV / "tab_costs_v2.csv"))


def load_wait_v2() -> pd.DataFrame:
    """`tab_wait_v2.csv`: willing-weighted wait and mean delivery days."""
    _v2_only("load_wait_v2()")
    return _cached("wait_v2", lambda: _read(REV / "tab_wait_v2.csv"))


def load_fleet_v2() -> pd.DataFrame:
    """`tab_fleet_per_hub_v2.csv`: per hub and day, AT THE OPERATOR PLAN.

    There is no per-hub-day table for the routing-optimal plan: the grid keeps
    only its totals, in the `*_before` columns of `tab_costs_v2.csv`. Any
    figure that wants the stage-1 daily profile has to say so.
    """
    _v2_only("load_fleet_v2()")
    return _cached("fleet_v2", lambda: _read(REV / "tab_fleet_per_hub_v2.csv"))


def load_chosen_v2() -> pd.DataFrame:
    """`_tab_chosen_v2.csv`: the schedule index chosen per cell, both plans."""
    _v2_only("load_chosen_v2()")
    return _cached("chosen_v2", lambda: _read(
        REV / "_tab_chosen_v2.csv", dtype={"plz": str}))


def load_headline_v2() -> pd.DataFrame:
    """Task 13's theta = 1 headline: both plans, both lenses, in one row."""
    _v2_only("load_headline_v2()")
    return _cached("headline_v2", lambda: _read(
        REV / "tables" / "tab_headline_theta1_v2.csv"))


def load_grid_full_v2() -> pd.DataFrame:
    """Task 13's full (P, theta) grid: every quantity the heatmaps show."""
    _v2_only("load_grid_full_v2()")
    return _cached("grid_full_v2", lambda: _read(
        REV / "tables" / "tab_grid_full_v2.csv"))


def load_pstar_v2() -> pd.DataFrame:
    """Task 13's per-LSP knee P*, in BOTH lenses, against the submission."""
    _v2_only("load_pstar_v2()")
    return _cached("pstar_v2", lambda: _read(
        REV / "tables" / "tab_pstar_knees_v2.csv"))


def load_fleet_diagnostics_v2() -> pd.DataFrame:
    """Per (P, theta): hub peaks against the flat lower bound, and the CV."""
    _v2_only("load_fleet_diagnostics_v2()")
    return _cached("fleetdiag_v2", lambda: _read(
        REV / "tables" / "tab_fleet_diagnostics_v2.csv"))


def load_overview_v2() -> pd.DataFrame:
    """The both-plans overview peek: the series compendium 40.15 quotes."""
    _v2_only("load_overview_v2()")
    return _cached("overview_v2", lambda: _read(
        REV / "_peek" / "results_overview_v5.csv"))


def load_discount_v2() -> pd.DataFrame:
    """The discount scenario: the penalty paid out, not shadow-priced (40.17).

    Two variants per (P, theta, plan): `*_perday` pays P euro per parcel and
    waiting day, `*_flat` pays a flat 0.50 EUR per delayed willing parcel.
    `breakeven_flat_*` is what the operator could pay per delayed parcel and
    still break even.
    """
    _v2_only("load_discount_v2()")
    return _cached("discount_v2", lambda: _read(
        REV / "_peek" / "discount_scenarios.csv"))


# --------------------------------------------------------------------------
# v2 grid: derived views
# --------------------------------------------------------------------------
# Which column of tab_costs_v2 carries (plan, lens). Every one of these four
# excludes the service penalty, so the four cells of the table are comparable
# with each other and with the baseline.
_V2_COST_COL = {
    (PLAN_ROUTING, LENS_ROUTING): "cost_stage1_eur",
    (PLAN_OPERATOR, LENS_ROUTING): "routing_total_eur",
    (PLAN_ROUTING, LENS_OPERATOR): "operator_cost_before_eur",
    (PLAN_OPERATOR, LENS_OPERATOR): "operator_cost_eur",
}
_V2_PEAK_COL = {PLAN_ROUTING: "sum_hub_peak_before",
                PLAN_OPERATOR: "sum_hub_peak"}
_V2_VEHDAY_COL = {PLAN_ROUTING: "vehicle_days_before",
                  PLAN_OPERATOR: "vehicle_days"}
_V2_PENALTY_COL = {PLAN_ROUTING: "penalty_before_eur",
                   PLAN_OPERATOR: "penalty_eur"}


def _check(plan: str, lens: str | None = None) -> None:
    if plan not in PLANS:
        raise ValueError(f"unknown plan {plan!r}, expected one of {PLANS}")
    if lens is not None and lens not in LENSES:
        raise ValueError(f"unknown lens {lens!r}, expected one of {LENSES}")


def cost_column(plan: str, lens: str) -> str:
    """The tab_costs_v2 column holding (plan, lens) cost in euro."""
    _check(plan, lens)
    return _V2_COST_COL[(plan, lens)]


def baseline_v2() -> dict:
    """The daily-delivery reference of the v2 grid, read from the grid itself.

    (P = 0, theta = 0) is every area served every day, which is the baseline
    the whole paper measures against, and stage 2 is pinned to a no-op there
    (61_grid_run_v2 G-6f-1) -- so both plans coincide and the row is the
    baseline in both lenses. Asserted against the pinned values so a re-run
    that moves the reference stops the build.
    """
    def _make():
        c = load_costs_v2()
        b = c[np.isclose(c.penalty, 0.0) & np.isclose(c.share_willing, 0.0)]
        assert len(b) == len(PROVIDERS), (
            f"baseline row count {len(b)}, expected {len(PROVIDERS)}")
        out = dict(
            routing_eur=float(b.cost_stage1_eur.sum()),
            operator_eur=float(b.operator_cost_eur.sum()),
            variable_eur=float(b.variable_cost_eur.sum()),
            hub_peak=int(b.sum_hub_peak.sum()),
            vehicle_days=int(b.vehicle_days.sum()),
        )
        for key, pin, tol in (("routing_eur", BASE_ROUTING_V2, 1.0),
                              ("operator_eur", BASE_OPERATOR_V2, 1.0),
                              ("hub_peak", BASE_HUB_PEAK_V2, 0),
                              ("vehicle_days", BASE_VEHICLE_DAYS_V2, 0)):
            assert abs(out[key] - pin) <= tol, (
                f"v2 baseline {key} = {out[key]} but the deck is pinned to "
                f"{pin}; every saving on every slide would be restated")
        # theta = 0 must be untouched by stage 2, or "baseline" is ambiguous
        assert np.allclose(b.cost_stage1_eur, b.routing_total_eur), (
            "stage 2 moved the theta = 0 plan; the baseline is not unique")
        return out
    return _cached("baseline_v2", _make)


def baseline_eur(lens: str = LENS_ROUTING) -> float:
    """The daily-delivery baseline in one lens."""
    _check(PLAN_ROUTING, lens)
    return baseline_v2()["routing_eur" if lens == LENS_ROUTING
                         else "operator_eur"]


def saving_grid_v2(plan: str = PLAN_ROUTING,
                   lens: str = LENS_ROUTING) -> pd.DataFrame:
    """(P, theta) -> cost and saving % of one plan seen through one lens.

    Also carries the plan's peak fleet, weekly vehicle-days and the service
    penalty it would incur, so a caller never has to re-derive which `*_before`
    column belongs to which plan.
    """
    _check(plan, lens)
    c = load_costs_v2()
    col = _V2_COST_COL[(plan, lens)]
    g = (c.groupby(["penalty", "share_willing"], as_index=False)
         .agg(total_eur=(col, "sum"),
              hub_peak=(_V2_PEAK_COL[plan], "sum"),
              vehicle_days=(_V2_VEHDAY_COL[plan], "sum"),
              penalty_eur=(_V2_PENALTY_COL[plan], "sum")))
    base = baseline_eur(lens)
    g["saving_pct"] = (1.0 - g.total_eur / base) * 100.0
    g["hub_peak_vs_base_pct"] = (
        g.hub_peak / baseline_v2()["hub_peak"] - 1.0) * 100.0
    g["plan"], g["lens"] = plan, lens
    return g


def wait_grid_v2(plan: str = PLAN_ROUTING) -> pd.DataFrame:
    """(P, theta) -> parcel-weighted added wait [d] and mean delivery days.

    Only the willing fraction waits; standard parcels keep daily service. The
    system value is therefore sum(numerator) / sum(all parcels) over providers,
    which is how 50_'s corrected formula and compendium 40.15 both read it.
    """
    _check(plan)
    w = load_wait_v2()
    num = ("wait_num_willing_stage1" if plan == PLAN_ROUTING
           else "wait_num_willing")
    days = "mean_days_stage1" if plan == PLAN_ROUTING else "mean_days"
    g = (w.groupby(["penalty", "share_willing"], as_index=False)
         .agg(num=(num, "sum"), total_parcels=("total_parcels", "sum"),
              willing_parcels=("willing_parcels", "sum"),
              mean_days=(days, "mean")))
    g["wait_d"] = g.num / g.total_parcels
    g["plan"] = plan
    return g


def chosen_sizes_v2(plan: str = PLAN_ROUTING) -> pd.DataFrame:
    """Per cell: how many delivery days that plan gives it.

    The grid stores a schedule INDEX; the size comes from the same enumeration
    the optimiser used, so the admissible-frequency assert below is a real
    check on the holding-days invariant rather than a restatement of the file.
    """
    _check(plan)

    def _make():
        from batch_delivery.optimization.schedules import (
            enumerate_valid_schedules)
        sched = enumerate_valid_schedules()
        assert len(sched) == 39, (
            f"{len(sched)} weekly patterns, expected 39 for "
            f"MAX_HOLDING_DAYS = 3")
        size = np.array([len(s) for s in sched], dtype=np.int64)
        df = load_chosen_v2().copy()
        col = ("schedule_idx_stage1" if plan == PLAN_ROUTING
               else "schedule_idx_system_smoothed")
        df["schedule_size"] = size[df[col].to_numpy()]
        observed = set(int(x) for x in df.schedule_size.unique())
        assert observed <= set(FREQ_SIZES), (
            f"delivery frequencies {sorted(observed - set(FREQ_SIZES))} fall "
            f"outside the admissible set {FREQ_SIZES}")
        df["plan"] = plan
        return df
    return _cached(f"chosen_sizes_{plan}", _make)


def consolidating_share_v2(penalty: float, share_willing: float,
                           plan: str = PLAN_ROUTING) -> float:
    """% of cells that give up daily delivery at one operating point."""
    s = chosen_sizes_v2(plan)
    sub = s[np.isclose(s.penalty, penalty)
            & np.isclose(s.share_willing, share_willing)]
    assert len(sub), f"no grid cell at P = {penalty}, theta = {share_willing}"
    return float(100.0 * (sub.schedule_size < N_DAYS).mean())


def hub_day_profile_v2(hub_fragment: str, penalty: float, share_willing: float
                       ) -> tuple[str, list[float]]:
    """Mon-Sat fleet of one hub at the OPERATOR plan, matched by name."""
    f = load_fleet_v2()
    hits = sorted({h for h in f.hub.unique() if hub_fragment.lower()
                   in str(h).lower()})
    assert len(hits) == 1, (
        f"{hub_fragment!r} matches {len(hits)} hubs: {hits}")
    sub = f[(f.hub == hits[0]) & np.isclose(f.penalty, penalty)
            & np.isclose(f.share_willing, share_willing)].sort_values("day")
    assert len(sub) == N_DAYS, f"{hits[0]}: {len(sub)} day rows, expected 6"
    return hits[0], [float(x) for x in sub.fleet]


def per_plz_eur_available() -> bool:
    """True once the Task-11 runner writes per-cell plan costs into the grid.

    Until then a per-area EURO map cannot be drawn on the v2 grid at all, and
    the honest substitute is the frequency-based structural view -- which the
    slide has to label as such.
    """
    if SCHEMA != SCHEMA_V2:
        return False
    cols = set(load_chosen_v2().columns)
    return bool(cols & {"plan_cost_stage1_eur", "plan_cost_eur",
                        "cell_cost_stage1_eur", "cell_cost_eur"})


# --------------------------------------------------------------------------
# stage-3 core tables (schema-dispatching)
# --------------------------------------------------------------------------
def load_costs(plan: str = PLAN_ROUTING) -> pd.DataFrame:
    """Per (P, theta, provider) cost, under whichever schema REV carries.

    On the legacy grid this is `tab_costs_smoothed.csv` unchanged and `plan` is
    ignored (there was only one). On a v2 grid it is `tab_costs_v2.csv` with
    `total_stage3_eur` aliased to the requested plan's routing euro, so callers
    that only want a system total keep working; the dd / express / pool split
    is NOT aliased, because the grid only records it for the operator plan --
    use `load_cost_split()` and be explicit.
    """
    if SCHEMA == SCHEMA_LEGACY:
        return _read(REV / "tab_costs_smoothed.csv")
    _check(plan)
    c = load_costs_v2().copy()
    c["plan"] = plan
    c["total_stage3_eur"] = c[_V2_COST_COL[(plan, LENS_ROUTING)]]
    c["operator_eur"] = c[_V2_COST_COL[(plan, LENS_OPERATOR)]]
    return c


def load_cost_split() -> pd.DataFrame:
    """dd / express / pooled tour cost -- operator plan only, v2 grid only."""
    _v2_only("load_cost_split()")
    c = load_costs_v2()
    out = c[["penalty", "share_willing", "provider", "dd_cost_eur",
             "express_cost_eur", "pool_cost_eur", "routing_total_eur"]].copy()
    assert np.allclose(out.dd_cost_eur + out.express_cost_eur
                       + out.pool_cost_eur, out.routing_total_eur), (
        "the tour-cost split does not sum to the routing total")
    out["plan"] = PLAN_OPERATOR
    return out


def load_wait(plan: str = PLAN_ROUTING) -> pd.DataFrame:
    """Per (P, theta): system-average added wait in days.

    Legacy grid: the 2026-08-25 corrected table. The earlier metric weighted
    *every* parcel with its cell's schedule wait, but only the willing fraction
    actually waits -- standard parcels keep daily service. That overstated the
    wait at theta < 1 by up to a factor of 60 and is identical at theta = 1.
    `avg_wait_d_stage3` carries the corrected series under its old name,
    because every caller reads it by that name.

    v2 grid: the same quantity for the requested plan, under the same name.
    """
    if SCHEMA == SCHEMA_LEGACY:
        w = _read(REV / "tab_wait_fixed.csv")
        return w.assign(avg_wait_d_stage3=w.wait_fixed)
    g = wait_grid_v2(plan)
    return g.assign(avg_wait_d_stage3=g.wait_d)


def load_fleet() -> pd.DataFrame:
    """Per (P, theta, provider, hub, day) fleet size.

    Legacy grid: still the pre-fix table, and deliberately -- it is the only
    one carrying the Stage-2/Stage-3 split that the smoothing figure compares,
    and that figure is drawn at theta = 1 where the fleet correction is a
    verified no-op. For anything spanning the theta grid use
    `load_fleet_fixed()`.

    v2 grid: `tab_fleet_per_hub_v2.csv`, which exists for the OPERATOR plan
    only; `fleet_stage2` and `fleet_stage3` are both aliased to it, so a figure
    comparing the two on a v2 grid draws a flat line rather than a wrong one.
    """
    if SCHEMA == SCHEMA_LEGACY:
        return _read(REV / "tab_fleet_per_hub_smoothed.csv")
    f = load_fleet_v2().copy()
    f["fleet_stage2"] = f["fleet"]
    f["fleet_stage3"] = f["fleet"]
    f["plan"] = PLAN_OPERATOR
    return f


def load_fleet_fixed() -> pd.DataFrame:
    """Per (P, theta, provider, hub, day): corrected fleet size.

    Legacy grid, the 2026-08-25 fix: the express parcels of all non-delivering
    cells at a hub ride ONE pooled tour, which is how their cost is computed.
    The earlier metric charged at least one vehicle per cell and invented up to
    29 vehicles per hub-day at intermediate theta. A no-op at theta = 0 and
    theta = 1.

    v2 grid: the partition-exact count (spec v3 4.3) is what the grid writes,
    so this is `load_fleet()` with `fleet_fixed` aliased on.
    """
    if SCHEMA == SCHEMA_LEGACY:
        return _read(REV / "tab_fleet_per_hub_fixed.csv")
    f = load_fleet()
    return f.assign(fleet_fixed=f["fleet"])


def load_express() -> pd.DataFrame:
    """Express-tour cost.

    Legacy grid: per (P, theta, provider, hub, day). v2 grid: per
    (P, theta, provider) only -- `express_cost_eur` of `tab_costs_v2.csv`, at
    the operator plan. The shape differs; a caller that groups by hub or day
    must check `SCHEMA` first.
    """
    if SCHEMA == SCHEMA_LEGACY:
        return _read(REV / "tab_express_smoothed.csv")
    c = load_costs_v2()
    return c[["penalty", "share_willing", "provider",
              "express_cost_eur"]].assign(
        express_stage3_eur=c.express_cost_eur, plan=PLAN_OPERATOR)


def load_per_plz() -> pd.DataFrame:
    """Per-PLZ Stage-3 dd cost, baseline reference and structural features.

    Produced by 00_recompute_per_plz_costs.py on the LEGACY grid; theta = 1
    only, where the hub-bundled express component is exactly zero and a per-PLZ
    decomposition is therefore exact. Pinned to `REV_LEGACY`: the v2 grid does
    not carry per-cell euro at all until Task 11's plan-cost columns land (see
    `per_plz_eur_available()`), and silently serving the legacy numbers from a
    v2 REV would date the map without saying so.
    """
    df = _read(REV_LEGACY / "tab_per_plz_costs_theta1.csv", dtype={"plz": str})
    assert (df.share_willing == 1.0).all(), "per-PLZ table must be theta=1 only"
    return df


def load_pstar() -> pd.DataFrame:
    """Per-provider knee point P* on the cost/wait trade-off.

    v2 grid: Task 13's two-lens table, with `P_star` / `saving_pct` / `wait_d`
    aliased to the ROUTING lens so submission-era callers keep their meaning;
    the operator-lens columns sit beside them.
    """
    if SCHEMA == SCHEMA_LEGACY:
        return _read(REV / "tab_pstar_knees_smoothed.csv")
    k = load_pstar_v2().copy()
    return k.assign(P_star=k.P_star_routing, saving_pct=k.saving_pct_routing,
                    wait_d=k.wait_d_routing)


# These two are pinned to the legacy directory on purpose: neither analysis was
# re-run for the revision, so pointing them at REV would either fail loudly (no
# such file in a v2 grid) or, worse, find a same-named file that means
# something else. The pin is the documentation.
def load_alpha_sensitivity() -> pd.DataFrame:
    return _read(REV_LEGACY / "tab_alpha_sensitivity.csv")


def load_cd_restart_spread() -> pd.DataFrame:
    return _read(REV_LEGACY / "tab_cd_restart_spread.csv")


# --------------------------------------------------------------------------
# stage-3 VROOM validation
# --------------------------------------------------------------------------
def load_vroom() -> pd.DataFrame:
    """Per (P, theta=1, provider, plz, day): real VROOM routes on Stage-3
    schedules.

    Non-OK rows are KEPT. Two rows (DHL, PLZ 30855, days 0 and 3 at P = 0) come
    back PARTIAL: VROOM returned a solution that does not serve every job, so
    the recorded cost is real but understates that cell-day. The published
    validation total (1 457 294.20 € at P = 0, i.e. the 23.69 % realised saving)
    includes them, so dropping them would silently disagree with the paper --
    it lifts the P = 0 saving to 24.92 % and removes 2 058 km. They are flagged
    here instead, for figures to disclose.
    """
    df = _read(VAL / "tab_vroom_smoothed.csv", dtype={"plz": str})
    bad = df[df.vroom_status != "OK"]
    if len(bad):
        cells = ", ".join(
            f"{r.provider}/{r.plz} d{int(r.day)} at P={r.penalty:g}"
            for r in bad.itertuples())
        print(f"  [vroom] {len(bad)} row(s) not fully solved, kept for "
              f"consistency with the published totals: {cells}")
    return df


def vroom_partial_note(df: pd.DataFrame) -> str:
    """One-clause disclosure of incompletely solved routing cells, or ''."""
    bad = df[df.vroom_status != "OK"]
    if not len(bad):
        return ""
    return (f"{len(bad)} of {len(df)} routing cells returned a partial solve "
            f"and are included as recorded")


def load_ml_vs_vroom() -> pd.DataFrame:
    return _read(VAL / "tab_ml_vs_vroom_smoothed.csv", dtype={"plz": str})


def load_vroom_diagnostics() -> pd.DataFrame:
    return _read(VAL / "tab_diagnostics_smoothed.csv")


def load_savings_validation() -> pd.DataFrame:
    """Predicted vs VROOM-actual saving at the four validated operating points."""
    return _read(VAL / "tab_savings_pred_vs_actual_smoothed.csv")


# --------------------------------------------------------------------------
# stage-2 schedule choices (frequency is invariant across stage 2 -> 3)
# --------------------------------------------------------------------------
def load_chosen_stage3() -> pd.DataFrame:
    """Per-PLZ Stage-3 schedule choice (post system smoothing).

    Also runs the frequency-invariance check that the per-PLZ *frequency* maps
    rely on: system smoothing reassigns which weekdays are served but must not
    change how many. Any map keyed on delivery frequency is only defensible
    while this holds, so it is asserted rather than assumed.
    """
    s2 = _read(RUN / "tab_chosen_schedules.csv", dtype={"plz": str})
    s3 = _read(RUN / "_tab_chosen_with_system_smoothing.csv", dtype={"plz": str})
    j = s2.merge(s3, on=["penalty", "share_willing", "provider", "plz"],
                 suffixes=("_s2", "_s3"))
    assert (j.schedule_size_init == j.schedule_size_system_smoothed).all(), (
        "frequency NOT preserved from stage 2 to stage 3 -- every per-PLZ "
        "frequency map in this set would be mislabelled"
    )
    print(f"  [invariance] schedule_size_init == schedule_size_system_smoothed "
          f"for all {len(j)} rows")
    observed = set(int(x) for x in s3.schedule_size_system_smoothed.unique())
    assert observed <= set(FREQ_SIZES), (
        f"delivery frequencies {sorted(observed - set(FREQ_SIZES))} fall outside "
        f"the admissible set {FREQ_SIZES}; the holding-days invariant or the "
        f"schedule enumeration changed")
    return s3


def load_baseline_per_provider() -> pd.DataFrame:
    """Per-provider weekly baseline (daily-delivery) direct cost."""
    df = _read(LEGACY_OUT / "02_baseline" / "tab_baseline_per_provider.csv")
    tot = float(df.dd_cost.sum())
    assert abs(tot - BASE_TOTAL) < 1.0, (
        f"baseline total {tot} != pinned BASE_TOTAL {BASE_TOTAL}")
    return df


# --------------------------------------------------------------------------
# training pool / model diagnostics (unchanged by stage 3)
# --------------------------------------------------------------------------
def load_training_pool() -> pd.DataFrame:
    """The 2733-row sweep_v3_mergefix pool the production hybrid was fit on."""
    return _read(SUPP / "sweep_v3_mergefix" / "training_matrix.csv")


# --------------------------------------------------------------------------
# geodata
# --------------------------------------------------------------------------
def load_raumtyp() -> pd.DataFrame:
    return _read(GEO / "plz_raumtyp.csv", dtype={"plz": str})


def load_plz_geometry():
    """PLZ polygons from the demand checkpoint, with cluster ids attached.

    Optimisation happens on merged PLZ clusters, so a per-PLZ choropleth has to
    paint every member polygon of a cluster with that cluster's value. The
    cluster_id column returned here is what the map scripts merge on.
    """
    import pickle
    ck = CKPT / "01_demand.pkl"
    if not ck.exists():
        raise FileNotFoundError(f"demand checkpoint missing: {ck}")
    prov.record(ck)
    with open(ck, "rb") as f:
        gdf = pickle.load(f)["gdf_plz"].copy()
    gdf["plz"] = gdf["plz"].astype(str).str.zfill(5)

    cl = _read(GEO / "plz_clusters.csv", dtype={"cluster_id": str})
    cl["members"] = cl["member_plz_list"].str.split(",")
    long = cl.explode("members").rename(columns={"members": "plz"})
    long["plz"] = long["plz"].astype(str).str.strip().str.zfill(5)
    long["cluster_id"] = long["cluster_id"].astype(str).str.zfill(5)

    gdf = gdf.merge(long[["plz", "cluster_id"]], on="plz", how="left")
    gdf["cluster_id"] = gdf["cluster_id"].fillna(gdf["plz"])
    return gdf


def clip_to_scope(gdf, in_scope_ids):
    """Restrict a PLZ GeoDataFrame to the modelled area."""
    ids = {str(x).zfill(5) for x in in_scope_ids}
    return gdf[gdf.cluster_id.isin(ids) | gdf.plz.isin(ids)].copy()


# --------------------------------------------------------------------------
# derived helpers
# --------------------------------------------------------------------------
def saving_grid(costs: pd.DataFrame | None = None, *,
                plan: str = PLAN_ROUTING,
                lens: str = LENS_ROUTING) -> pd.DataFrame:
    """(P, theta) -> total cost and saving % against that grid's baseline.

    On a v2 grid this is `saving_grid_v2(plan, lens)`; `plan` and `lens` are
    ignored on the legacy grid, which has only one of each. The baseline is the
    one belonging to the grid in use -- 1 909 747.75 EUR on the legacy grid,
    1 909 432 / 2 109 742 EUR on v5 -- so a percentage is never formed from two
    different runs.
    """
    if SCHEMA == SCHEMA_V2 and costs is None:
        return saving_grid_v2(plan, lens)
    c = load_costs() if costs is None else costs
    g = (c.groupby(["penalty", "share_willing"], as_index=False)
         .agg(total_eur=("total_stage3_eur", "sum"),
              dd_eur=("dd_cost_stage3_eur", "sum"),
              express_eur=("express_stage3_eur", "sum")))
    g["saving_pct"] = (1.0 - g.total_eur / BASE_TOTAL) * 100.0
    return g


def fleet_totals(fleet: pd.DataFrame | None = None) -> pd.DataFrame:
    """(P, theta) -> system fleet peak / mean / spread, stage 2 vs stage 3."""
    # System totals come from the corrected table: they span the theta grid,
    # which is exactly where the pooled-express fix bites.
    f = load_fleet_fixed() if fleet is None else fleet
    col = "fleet_fixed" if "fleet_fixed" in f.columns else "fleet_stage3"
    per_day = (f.groupby(["penalty", "share_willing", "day"], as_index=False)
               .agg(s2=(col, "sum"), s3=(col, "sum")))
    out = (per_day.groupby(["penalty", "share_willing"], as_index=False)
           .agg(peak_s2=("s2", "max"), peak_s3=("s3", "max"),
                mean_s2=("s2", "mean"), mean_s3=("s3", "mean"),
                min_s2=("s2", "min"), min_s3=("s3", "min")))
    out["spread_s2"] = out.peak_s2 - out.min_s2
    out["spread_s3"] = out.peak_s3 - out.min_s3
    out["cv_s3_pct"] = np.where(
        out.mean_s3 > 0, out.spread_s3 / out.mean_s3 * 100.0, np.nan)
    return out


def add_raumtyp(df: pd.DataFrame) -> pd.DataFrame:
    """Attach raumtyp_3 / raumtyp_8 to a per-PLZ frame."""
    rt = load_raumtyp()
    keep = [c for c in ("plz", "raumtyp_3", "raumtyp_8", "raumtyp_8_name")
            if c in rt.columns]
    return df.merge(rt[keep], on="plz", how="left")
