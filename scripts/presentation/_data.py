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
* the VROOM validation loaders — the revision's validation is a run of its own
  and lags the grid it validates, so `VAL` follows `VAL_GRID_NAME` and not
  `REV`. It is guarded: a v2 validation must carry items 0-3 or the loaders
  refuse it, because a figure drawn from a half-written validation states a
  saving against a baseline that has not been solved.

Both read `REV_LEGACY` / `VAL` explicitly, so the split is visible in the code
rather than hidden in a default.

Which plan the figures read
---------------------------
A v2 grid holds TWO schedule choices per cell, and `load_chosen_stage3()` used
to read neither: it read `results/runs/path2_2026_05_29`, the pre-revision
run, unconditionally. That run's stage 2 was frequency-PRESERVING; v6's is
frequency-FREE (compendium 40.14), so the invariance the legacy loader
asserted is measurably false on v6 — 20.1 % of cell choices change delivery
frequency between stage 1 and stage 2. Every consumer therefore says which
plan it is on. The default for the act figures is the OPERATOR plan
(`balanced` == stage 2), the convention `scripts/revision/76_maps_v2.py` uses
for the paper's spatial supplement (40.23b), and `plan_stamp()` puts it in the
figure's provenance basis.
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
# The revision grid. `REV_V2_NAME` is the one line Part B moves from "v5" to
# "v6" once the head grid is finished -- everything else, including the decks'
# provisional tag, is derived from it, so switching grids is a one-token edit
# and not a hunt through the figure scripts. `$PRES_REV_DIR` overrides it
# entirely, which is how a grid outside `results/` is used.
REV_V2_NAME = "revision_2026_08_v6"
REV_V2_DEFAULT = ROOT / "results" / REV_V2_NAME

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


# The VROOM validation is a much longer run than the grid it validates, so the
# two are not in step. `VAL` therefore follows its OWN setting -- $PRES_VAL_DIR,
# else the last grid whose validation is FINISHED -- and never points at a
# directory that is still being written; every figure drawn from it states
# which grid it is on. v6's validation completed 2026-08-28 (items 0-3, see
# results/revision_2026_08_v6/validation/validation_report.md), so this name
# now names v6 -- and `require_validation_items()` makes that a checked fact
# rather than a hope: a v2 validation missing any of items 0-3 is refused.
VAL_GRID_NAME = "revision_2026_08_v6"

# item 0 = the theta = 0 daily baseline (without it no ACTUAL saving exists,
# only predicted-vs-actual totals), 1 = the operator-plan scenario points,
# 2 = the stage-1 plan at the same points, 3 = the theta < 1 point. A figure
# drawn from a subset of these is drawn from a validation that is still being
# written.
VAL_REQUIRED_ITEMS = (0, 1, 2, 3)


def _resolve_val() -> Path:
    env = os.environ.get("PRES_VAL_DIR")
    if env:
        return Path(env).resolve()
    return (ROOT / "results" / VAL_GRID_NAME / "validation").resolve()


def set_rev_dir(path) -> Path:
    """Point the revision loaders at another grid directory.

    Called by the `--rev-dir` flag of the deck builders. Rebinds the module
    globals rather than threading a handle through every loader, because the
    figure scripts read `D.REV` directly; call it before the first load.
    """
    global REV, SCHEMA
    REV = Path(path).resolve()
    SCHEMA = _detect_schema(REV)
    _CACHE.clear()
    return REV


def set_val_dir(path) -> Path:
    """Point the VROOM-validation loaders at a finished validation output."""
    global VAL
    VAL = Path(path).resolve()
    _CACHE.clear()
    return VAL


REV = Path(os.environ.get("PRES_REV_DIR") or REV_V2_DEFAULT).resolve()
SCHEMA = _detect_schema(REV)
VAL = _resolve_val()
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

# The grid's OWN name for each plan, as it appears in the `plan` column of
# tab_per_cell_costs_v2.csv and in 61_grid_run_v2.py's column suffixes. The
# two vocabularies are kept apart on purpose: `PLAN_ROUTING`/`PLAN_OPERATOR`
# are what a figure means, `stage1`/`balanced` are what the file says, and
# `grid_plan()` is the only place that maps one onto the other.
PLAN_STAGE1 = "stage1"
PLAN_BALANCED = "balanced"
GRID_PLANS = (PLAN_STAGE1, PLAN_BALANCED)
_PLAN_ALIAS = {
    PLAN_ROUTING: PLAN_STAGE1, PLAN_STAGE1: PLAN_STAGE1,
    PLAN_OPERATOR: PLAN_BALANCED, PLAN_BALANCED: PLAN_BALANCED,
}
# What the act figures read unless told otherwise. 76_maps_v2.py draws the
# paper's spatial supplement on the operator-polished plan (40.23b) and these
# figures are the deck's version of the same maps, so they agree by default
# rather than by coincidence.
CHOSEN_PLAN_DEFAULT = PLAN_BALANCED
GRID_PLAN_LABEL = {
    PLAN_STAGE1: "routing-optimal plan (stage 1)",
    PLAN_BALANCED: "operator-polished plan (stage 2)",
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
# The daily-delivery reference of each grid, pinned PER GRID: a saving is only
# ever formed against the baseline of the grid it was computed on. v5's is the
# compendium's (40.12); v6's is stated in
# results/revision_2026_08_v6/DEEP_DIVE_V6_PAPER_IMPACT.md 2, which also warns
# that v6 scenarios must NOT be normalised against v5's baseline -- they differ
# by 11 341.62 EUR, 0.594 %.
BASELINE_PINS = {
    "revision_2026_08_v5": dict(routing=1_909_432.42, operator=2_109_742.27,
                                hub_peak=1239, vehicle_days=6375, cv=0.139),
    "revision_2026_08_v6": dict(routing=1_898_090.80, operator=2_098_400.65,
                                hub_peak=1239, vehicle_days=6375, cv=0.138998),
}
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


def grid_plan(plan: str | None = None) -> str:
    """The grid's own name (`stage1` / `balanced`) for a plan.

    Accepts either vocabulary -- the figure's (`routing` / `operator`) or the
    file's -- so a caller never has to remember which side of the adapter it
    is on. Anything else is refused rather than silently defaulted, because a
    typo that fell through to the default would draw the wrong plan under the
    right caption.
    """
    if plan is None:
        return CHOSEN_PLAN_DEFAULT
    key = str(plan)
    if key not in _PLAN_ALIAS:
        raise ValueError(
            f"unknown plan {plan!r}; expected one of "
            f"{sorted(set(_PLAN_ALIAS))}")
    return _PLAN_ALIAS[key]


def plan_stamp(plan: str | None = None) -> str:
    """One clause naming the grid and the plan, for a figure's provenance.

    Every figure that reads a per-cell schedule choice puts this in its
    `basis`, so the manifest records WHICH of the v2 grid's two plans the
    picture is of -- the distinction the pre-revision loader erased.
    """
    if SCHEMA != SCHEMA_V2:
        return f"{REV.name} (legacy grid, single plan)"
    return f"{REV.name}, {GRID_PLAN_LABEL[grid_plan(plan)]}"


def _avg_wait_days(days) -> float:
    """Mean days a uniformly-arriving parcel waits under one weekly pattern.

    The same expression `scripts/revision/_stage3_common.avg_wait_days` uses,
    inlined so the presentation layer does not import the revision helpers
    (which carry their own env-driven RUN_DIR / OUT_DIR defaults).
    """
    if not days:
        return 0.0
    ds = sorted(days)
    total = 0.0
    for di in range(N_DAYS):
        nxt = min(((d - di) % N_DAYS, d) for d in ds)[1]
        total += (nxt - di) % N_DAYS
    return total / N_DAYS


def _schedule_lookup():
    """(size, weekday string, average wait) per schedule index.

    Derived from the enumeration the optimiser itself used, never read: the
    grid stores only an INDEX, so recomputing size and wait here makes the
    admissible-frequency and holding-days checks real checks rather than a
    restatement of the file. Cross-checked against v6's
    `tab_per_cell_costs_v2.csv` (`mean_days` / `wait_days`): exact.
    """
    def _make():
        from batch_delivery.optimization.schedules import (
            enumerate_valid_schedules)
        sched = enumerate_valid_schedules()
        assert len(sched) == 39, (
            f"{len(sched)} weekly patterns, expected 39 for "
            f"MAX_HOLDING_DAYS = 3")
        size = np.array([len(x) for x in sched], dtype=np.int64)
        days = np.array([",".join(WEEKDAYS[d] for d in sorted(x))
                         for x in sched], dtype=object)
        wait = np.array([_avg_wait_days(sorted(x)) for x in sched],
                        dtype=np.float64)
        return size, days, wait
    return _cached("schedule_lookup", _make)


_V2_CHOSEN_IDX_COL = {PLAN_STAGE1: "schedule_idx_stage1",
                      PLAN_BALANCED: "schedule_idx_balanced"}


def load_chosen_v2(plan: str | None = None) -> pd.DataFrame:
    """`_tab_chosen_v2.csv`: the schedule index chosen per cell, both plans.

    With `plan=None` this is the file verbatim -- three index columns side by
    side. With `plan="balanced"` (the act figures' default) or
    `plan="stage1"` it is that ONE plan, expanded into the column names the
    per-PLZ frequency figures read: `schedule_idx_system_smoothed`,
    `schedule_size_system_smoothed`, `weekdays_system_smoothed` and
    `avg_wait_d_system_smoothed`. The `_system_smoothed` spelling is kept
    because every caller reads it by that name; on v6 stage 3 is OFF, so
    `schedule_idx_balanced == schedule_idx_system_smoothed` by construction
    (61_grid_run_v2) and the name describes the final plan either way. Both
    plans' sizes, weekdays and waits ride along as `*_stage1` / `*_balanced`,
    so a figure contrasting the two never loads the table twice.

    `weekly_parcels` is attached when the grid ships per-cell costs, which is
    what the parcel-weighted median of 76_maps_v2 needs; without them the
    column is absent rather than guessed, so a caller that needs it fails on
    the KeyError instead of silently switching to an unweighted median.
    """
    _v2_only("load_chosen_v2()")
    raw = _cached("chosen_v2", lambda: _read(
        REV / "_tab_chosen_v2.csv", dtype={"plz": str}))
    if plan is None:
        return raw
    gp = grid_plan(plan)
    return _cached(f"chosen_v2_{gp}", lambda: _chosen_v2_plan(raw, gp))


def _chosen_v2_plan(raw: pd.DataFrame, gp: str) -> pd.DataFrame:
    need = ["penalty", "share_willing", "provider", "plz"] + [
        _V2_CHOSEN_IDX_COL[q] for q in GRID_PLANS]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise KeyError(
            f"{REV / '_tab_chosen_v2.csv'} lacks {missing}; the presentation "
            f"figures need the full _tab_chosen_v2 schema of "
            f"61_grid_run_v2.py. Point PRES_REV_DIR at a grid that runner "
            f"wrote.")
    size, days, wait = _schedule_lookup()
    df = raw.copy()
    df["plz"] = df.plz.astype(str).str.zfill(5)
    for q in GRID_PLANS:
        idx = df[_V2_CHOSEN_IDX_COL[q]].to_numpy()
        df[f"schedule_size_{q}"] = size[idx]
        df[f"weekdays_{q}"] = days[idx]
        df[f"avg_wait_d_{q}"] = wait[idx]
    idx = df[_V2_CHOSEN_IDX_COL[gp]].to_numpy()
    df["schedule_idx_system_smoothed"] = idx
    df["schedule_size_system_smoothed"] = size[idx]
    df["weekdays_system_smoothed"] = days[idx]
    df["avg_wait_d_system_smoothed"] = wait[idx]
    df["plan"] = gp
    observed = set(int(x) for x in df.schedule_size_system_smoothed.unique())
    assert observed <= set(FREQ_SIZES), (
        f"delivery frequencies {sorted(observed - set(FREQ_SIZES))} fall "
        f"outside the admissible set {FREQ_SIZES}; the holding-days invariant "
        f"or the schedule enumeration changed")
    # Frequency is NOT invariant across the v2 grid's two plans -- stage 2 is
    # frequency-free at theta > 0 (compendium 40.14). This is REPORTED, not
    # asserted: the legacy loader asserted the opposite and would abort here.
    moved = float((df.schedule_size_stage1
                   != df.schedule_size_balanced).mean()) * 100.0
    print(f"  [plan] {REV.name} / {GRID_PLAN_LABEL[gp]}; stage 2 changes the "
          f"delivery frequency of {moved:.1f} % of cell choices "
          f"(frequency-free stage 2, compendium 40.14)")
    cpath = per_cell_costs_path()
    if cpath is not None:
        cells = _read(cpath, dtype={"plz": str})
        cells["plz"] = cells.plz.astype(str).str.zfill(5)
        vol = (cells[cells.plan.astype(str) == gp]
               [["penalty", "share_willing", "provider", "plz",
                 "cell_parcels_week"]]
               .rename(columns={"cell_parcels_week": "weekly_parcels"}))
        before = len(df)
        df = df.merge(vol, on=["penalty", "share_willing", "provider", "plz"],
                      how="left")
        assert len(df) == before, (
            f"the per-cell parcel join changed the row count {before} -> "
            f"{len(df)}; the two tables are not on the same cell grain")
        assert df.weekly_parcels.notna().all(), (
            f"{int(df.weekly_parcels.isna().sum())} cell choice(s) have no "
            f"parcel count in {cpath.name}")
    return df


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


def _peek(stem: str) -> Path:
    """A `_peek/<stem>*.csv`, whose exact name carries the grid version."""
    hits = sorted((REV / "_peek").glob(stem + "*.csv"))
    hits = [h for h in hits if "partial" not in h.name.lower()] or hits
    if not hits:
        raise FileNotFoundError(
            "no " + stem + "*.csv under " + str(REV / "_peek"))
    return hits[-1]


def load_overview_v2() -> pd.DataFrame:
    """The both-plans overview peek: the series the deep dive quotes."""
    _v2_only("load_overview_v2()")
    return _cached("overview_v2", lambda: _read(_peek("results_overview")))


def load_discount_v2() -> pd.DataFrame:
    """The discount scenario: the penalty paid out, not shadow-priced (40.17).

    Two variants per (P, theta, plan): `*_perday` pays P euro per parcel and
    waiting day, `*_flat` pays a flat 0.50 EUR per delayed willing parcel.
    `breakeven_flat_*` is what the operator could pay per delayed parcel and
    still break even.
    """
    _v2_only("load_discount_v2()")
    return _cached("discount_v2", lambda: _read(_peek("discount_scenarios")))


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
        pins = BASELINE_PINS.get(REV.name)
        if pins is None:
            print(f"  [baseline] {REV.name} has no pinned reference; using "
                  f"the grid's own: routing {out['routing_eur']:.2f}, "
                  f"operator {out['operator_eur']:.2f}, peak {out['hub_peak']}")
        else:
            for key, pin, tol in (("routing_eur", pins["routing"], 1.0),
                                  ("operator_eur", pins["operator"], 1.0),
                                  ("hub_peak", pins["hub_peak"], 0),
                                  ("vehicle_days", pins["vehicle_days"], 0)):
                assert abs(out[key] - pin) <= tol, (
                    f"{REV.name} baseline {key} = {out[key]} but the deck is "
                    f"pinned to {pin}; every saving on every slide would be "
                    f"restated")
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


def per_cell_costs_path():
    """The grid's per-cell plan-cost table, if it has one."""
    if SCHEMA != SCHEMA_V2:
        return None
    p = REV / "tables" / "tab_per_cell_costs_v2.csv"
    return p if p.exists() else None


def per_plz_eur_available() -> bool:
    """True once the grid carries per-cell plan costs.

    Without them a per-area EURO map cannot be drawn on a v2 grid at all, and
    the honest substitute is the frequency-based structural view -- which the
    figure then has to label as such. v6 ships `tab_per_cell_costs_v2.csv`
    (72_); v5 did not.
    """
    if SCHEMA != SCHEMA_V2:
        return False
    if per_cell_costs_path() is not None:
        return True
    cols = set(load_chosen_v2().columns)
    return bool(cols & {"plan_cost_stage1_eur", "plan_cost_eur",
                        "cell_cost_stage1_eur", "cell_cost_eur"})


def load_per_cell_costs_v2(plan: str = PLAN_OPERATOR) -> pd.DataFrame:
    """Per (P, theta, provider, plz) plan cost, for one plan.

    `cell_cost_eur` is that cell's own tour plus its packet-proportional share
    of any pooled tour it rides on, so the column sums to the plan's routing
    total and a per-area map built on it is exact rather than allocated by
    hand. The grid stores both plans in one long table; `plan` picks one.
    """
    p = per_cell_costs_path()
    if p is None:
        raise FileNotFoundError(
            str(REV.name) + " has no per-cell plan costs; "
            "per_plz_eur_available() is False and the caller must fall back "
            "to the frequency view AND say so on the figure")
    _check(plan)
    df = _read(p, dtype={"plz": str})
    have = set(df.plan.astype(str).unique())
    want = "stage1" if plan == PLAN_ROUTING else "balanced"
    key = want if want in have else sorted(have)[0]
    return df[df.plan.astype(str) == key].copy()


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


# What 00_recompute_per_plz_costs.py writes on a v2 grid, next to the other
# v6 tables. The legacy grid keeps its own file at the top level.
PER_PLZ_V2_REL = ("tables", "tab_per_plz_costs_theta1_v2.csv")


def per_plz_v2_path() -> Path:
    return REV.joinpath(*PER_PLZ_V2_REL)


def load_per_plz(plan: str | None = None) -> pd.DataFrame:
    """Per-PLZ plan cost, daily-delivery reference and structural features.

    theta = 1 only, on both schemas: that is where the hub-bundled EXPRESS
    component is exactly zero, so a per-area decomposition is exact rather
    than allocated (compendium 38.2a). Both files are written by
    `00_recompute_per_plz_costs.py`.

    LEGACY grid: `tab_per_plz_costs_theta1.csv`, one plan, and a euro there is
    the direct-delivery cost -- the legacy grid had no pooled small-delivery
    tours to attribute.

    v2 grid: `tables/tab_per_plz_costs_theta1_v2.csv`, from 72_'s per-cell
    table, for ONE plan (default: the operator-polished one). A euro there is
    the cell's full ROUTING cost under the realistic-tour rule -- its own tour
    plus its parcel-proportional share of every pooled tour it rides on -- and
    the three parts are carried alongside as `own_cost_eur` /
    `pool_share_eur` / `express_share_eur`. The legacy names
    `dd_cost_stage3_eur` / `dd_cost_baseline_eur` are aliased onto the plan
    and baseline cost so the Act-7 figures read the same column names on both
    schemas; the "dd" in those names is historical, and every v6 figure says
    routing lens in its caption. The operator lens is NOT offered per area: it
    is hub-attributable, not cell-attributable (72_).
    """
    if SCHEMA != SCHEMA_V2:
        if plan is not None and grid_plan(plan) != CHOSEN_PLAN_DEFAULT:
            raise RuntimeError(
                f"the legacy grid {REV.name} carries ONE plan; it cannot "
                f"serve {plan!r}.")
        df = _read(REV_LEGACY / "tab_per_plz_costs_theta1.csv",
                   dtype={"plz": str})
        assert (df.share_willing == 1.0).all(), (
            "per-PLZ table must be theta=1 only")
        return df

    gp = grid_plan(plan)
    path = per_plz_v2_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run\n"
            f"  python scripts/presentation/00_recompute_per_plz_costs.py\n"
            f"to derive it from {REV.name}'s tables/tab_per_cell_costs_v2.csv; "
            f"the legacy 2026-07 per-area table is NOT a substitute -- it is a "
            f"different grid with a different baseline.")
    df = _read(path, dtype={"plz": str})
    have = set(df.plan.astype(str).unique())
    if gp not in have:
        raise KeyError(
            f"{path.name} carries plans {sorted(have)}, not {gp!r}; re-run "
            f"00_recompute_per_plz_costs.py")
    df = df[df.plan.astype(str) == gp].copy()
    assert (df.share_willing == 1.0).all(), (
        "per-PLZ table must be theta=1 only")
    df["dd_cost_stage3_eur"] = df.cell_cost_eur
    df["dd_cost_baseline_eur"] = df.cell_cost_baseline_eur
    return df


def hub_cell_counts() -> pd.DataFrame:
    """Per (provider, hub): how many cells that depot serves.

    Read from the per-PLZ table's hub column, which is a static INPUT of the
    model -- the hub assignment of `io/hubs.py`, not a grid result -- so it is
    the same on every grid and the legacy pin costs nothing. This is what makes
    the one-area-depot count on the slides a derived, asserted fact instead of
    a typed-in one.
    """
    def _make():
        d = _read(REV_LEGACY / "tab_per_plz_costs_theta1.csv",
                  dtype={"plz": str})
        g = (d[np.isclose(d.penalty, 0.0)]
             .groupby(["provider", "hub"], as_index=False)
             .agg(n_cells=("plz", "nunique")))
        assert len(g) and g.n_cells.min() >= 1, "empty hub/cell assignment"
        return g
    return _cached("hub_cells", _make)


def one_cell_hubs(provider: str = "DHL") -> tuple[int, int]:
    """(hubs serving exactly one cell, hubs in total) for one provider.

    DHL is the only multi-depot network in the case study; every other LSP has
    a single depot, so `one_cell_hubs("GLS")` is (0, 1) and the statement the
    slides make is a DHL statement.
    """
    g = hub_cell_counts()
    h = g[g.provider == provider]
    assert len(h), f"no hubs for {provider!r}"
    return int((h.n_cells == 1).sum()), int(len(h))


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
# A VROOM solve that came out of the cache is as real as one solved now -- the
# cache stores solutions, not estimates -- so "solved" is OK or CACHED. Only
# PARTIAL (VROOM returned a route plan that does not serve every job) and rows
# with unassigned jobs are incomplete.
VROOM_OK = ("OK", "CACHED")

VAL_SCHEMA_V2 = "v2"
VAL_SCHEMA_LEGACY = "legacy"


def val_schema() -> str:
    """Which validation output `REV` carries, if any."""
    if (VAL / "tab_vroom_v2.csv").exists():
        return VAL_SCHEMA_V2
    if (VAL / "tab_vroom_smoothed.csv").exists():
        return VAL_SCHEMA_LEGACY
    raise FileNotFoundError(
        f"no VROOM validation under {VAL}: neither tab_vroom_v2.csv nor "
        f"tab_vroom_smoothed.csv. Point PRES_REV_DIR at a grid whose "
        f"validation has been produced.")


def validation_items() -> set[int]:
    """Which items of the v2 validation queue this directory actually holds."""
    if val_schema() != VAL_SCHEMA_V2:
        return set()
    df = _cached("vroom_v2_raw", lambda: _read(VAL / "tab_vroom_v2.csv"))
    return {int(x) for x in df.item.unique()}


def require_validation_items() -> None:
    """Refuse a v2 validation that is still being written.

    The four items are solved as separate, hours-long runs, and the directory
    exists from the first of them. Item 0 in particular is the theta = 0 daily
    baseline: without it `load_savings_validation()` has no solved denominator
    and every "realised saving" on a slide would be an actual numerator over a
    predicted baseline. So the presence of all of `VAL_REQUIRED_ITEMS` is
    asserted here, once, at the single point every v2 validation loader goes
    through -- rather than discovered as a NaN three figures later.
    """
    if val_schema() != VAL_SCHEMA_V2:
        return
    have = validation_items()
    missing = sorted(set(VAL_REQUIRED_ITEMS) - have)
    if missing:
        raise RuntimeError(
            f"{VAL} is an INCOMPLETE v2 validation: it carries items "
            f"{sorted(have)}, missing {missing} of the required "
            f"{list(VAL_REQUIRED_ITEMS)} "
            f"(0 = theta-0 baseline, 1 = operator-plan points, "
            f"2 = stage-1 plan, 3 = theta < 1). Set PRES_VAL_DIR to a "
            f"finished validation, or move _data.VAL_GRID_NAME back to a grid "
            f"whose validation is complete -- a figure drawn from this "
            f"directory would state a saving against a baseline nobody "
            f"solved.")


def validation_sampled_items() -> set[int]:
    """Items whose instances were only PARTLY solved (spec G6 fallback).

    The validation enumerates every instance into `instance_queue.csv` and
    then solves what fits an eight-VROOM-hour budget; on v6 item 3 came in
    over budget and was cut to a stratified sample (1 000 of 1 594). A sampled
    item's cost TOTAL is therefore not a system total, and any percentage
    formed from it -- a saving, a share -- is meaningless however real each
    individual solve is.

    Measured, not read from prose: the queue's count per item against the
    solved table's. If the queue is missing the answer is "unknown", which is
    returned as every item being suspect rather than none.
    """
    def _make():
        solved = load_vroom_v2().groupby("item").size()
        q = VAL / "instance_queue.csv"
        if not q.exists():
            print(f"  [vroom] no instance_queue.csv under {VAL}; the "
                  f"completeness of every item is unknown")
            return set(int(i) for i in solved.index)
        planned = _read(q).groupby("item").size()
        out = {int(i) for i in solved.index
               if int(solved[i]) < int(planned.get(i, solved[i]))}
        if out:
            print("  [vroom] item(s) " + ", ".join(
                f"{i} ({int(solved[i])} of {int(planned[i])} instances)"
                for i in sorted(out))
                + " were sampled, not enumerated; no system total or "
                  "percentage may be formed from them")
        return out
    return _cached("vroom_sampled_items", _make)


def load_vroom_v2() -> pd.DataFrame:
    """`validation/tab_vroom_v2.csv` verbatim: one row per solved INSTANCE.

    An instance is a hub-day-plan-kind routing problem, not a cell: a
    `*_group` row is one tour serving several cells at once, which is the whole
    point of the bundled pricing. `members` holds the cells it covers.

    Guarded by `require_validation_items()`: every v2 validation read in this
    module funnels through here, so a half-written validation directory is
    refused once instead of producing a quietly wrong figure.
    """
    def _make():
        require_validation_items()
        return _read(VAL / "tab_vroom_v2.csv")
    return _cached("vroom_v2", _make)


def _vroom_v2_cells() -> pd.DataFrame:
    """The single-cell instances of the v2 validation, keyed like the old table.

    Only `*_single` instances map onto one cell. Group instances are a pooled
    tour over several cells and cannot be split per cell without inventing an
    allocation, so they are excluded here and reported by
    `vroom_group_note()`.
    """
    def _make():
        df = load_vroom_v2()
        single = df[df.instance_kind.astype(str).str.endswith("_single")].copy()
        single["plz"] = (single.members.astype(str)
                         .str.strip('[]"\' ').str.strip('"'))
        single["mode"] = np.where(
            single.instance_kind.astype(str).str.startswith("express"),
            "express", "batched")
        out = single.rename(columns={
            "predicted_cost_eur": "ml_dd_cost_system_smoothed_eur"})
        out["ml_dd_cost_eur"] = out.ml_dd_cost_system_smoothed_eur
        out["n_routes"] = out.vroom_n_routes
        out["n_parcels"] = out.vroom_n_parcels
        return out
    return _cached("vroom_v2_cells", _make)


def load_vroom(plan: str | None = None,
               theta: float | None = None) -> pd.DataFrame:
    """Per (P, theta, provider, plz, day): real VROOM routes on the plan.

    LEGACY grid: `tab_vroom_smoothed.csv`. Non-OK rows are KEPT. Two rows
    (DHL, PLZ 30855, days 0 and 3 at P = 0) come back PARTIAL: VROOM returned a
    solution that does not serve every job, so the recorded cost is real but
    understates that cell-day. The published validation total (1 457 294.20 EUR
    at P = 0, i.e. the 23.69 % realised saving) includes them, so dropping them
    would silently disagree with the paper -- it lifts the P = 0 saving to
    24.92 % and removes 2 058 km. They are flagged here instead.

    v2 grid: the single-cell instances of `tab_vroom_v2.csv`, with the legacy
    column names, plus `plan`.

    `plan` and `theta` are FILTERS, and any figure that aggregates per cell or
    per provider needs both. The v2 validation covers two plans and two
    thetas: (P = 0, theta = 1) and (P = 0.25, theta = 1) are each solved
    TWICE, once per plan (queue items 1 and 2), item 0 is the theta = 0
    baseline and item 3 sits at theta = 0.5. Grouping by penalty alone
    therefore sums a cell's balanced and stage-1 tours into one number and
    mixes three different thetas into a series labelled theta = 1. The legacy
    validation had one plan and one theta, so this could not happen there and
    the filters are no-ops on it -- which is why they default to off.
    """
    if val_schema() == VAL_SCHEMA_LEGACY:
        df = _read(VAL / "tab_vroom_smoothed.csv", dtype={"plz": str})
    else:
        df = _vroom_v2_cells()
        note = vroom_group_note()
        if note:
            print(f"  [vroom] {note}")
        if plan is not None:
            gp = grid_plan(plan)
            df = df[df.plan.astype(str) == gp]
            assert len(df), (
                f"the validation of {VAL.parent.name} has no {gp!r} rows; it "
                f"covers {sorted(_vroom_v2_cells().plan.astype(str).unique())}")
        if theta is not None:
            df = df[np.isclose(df.share_willing, theta)]
            assert len(df), (
                f"the validation of {VAL.parent.name} has no rows at "
                f"theta = {theta}")
    bad = df[~df.vroom_status.isin(VROOM_OK)]
    if len(bad):
        cells = ", ".join(
            f"{r.provider}/{r.plz} d{int(r.day)} at P={r.penalty:g}"
            for r in bad.itertuples())
        print(f"  [vroom] {len(bad)} row(s) not fully solved, kept for "
              f"consistency with the published totals: {cells}")
    return df


def vroom_group_note() -> str:
    """One clause on the pooled-tour instances a per-cell view cannot show."""
    if val_schema() != VAL_SCHEMA_V2:
        return ""
    df = load_vroom_v2()
    n = int((~df.instance_kind.astype(str).str.endswith("_single")).sum())
    if not n:
        return ""
    return (f"{n} of {len(df)} validated instances are pooled tours over "
            f"several cells and are excluded from per-cell views")


def vroom_partial_note(df: pd.DataFrame) -> str:
    """One-clause disclosure of incompletely solved routing cells, or ''."""
    bad = df[~df.vroom_status.isin(VROOM_OK)]
    if not len(bad):
        return ""
    return (f"{len(bad)} of {len(df)} routing cells returned a partial solve "
            f"and are included as recorded")


def load_ml_vs_vroom() -> pd.DataFrame:
    """Per (P, theta, provider, plz): surrogate price against the solver's."""
    if val_schema() == VAL_SCHEMA_LEGACY:
        return _read(VAL / "tab_ml_vs_vroom_smoothed.csv", dtype={"plz": str})
    c = _vroom_v2_cells()
    return (c.groupby(["penalty", "share_willing", "plan", "provider", "plz"],
                      as_index=False)
            .agg(vroom_cost_eur=("vroom_cost_eur", "sum"),
                 ml_dd_cost_eur=("ml_dd_cost_eur", "sum"),
                 n_routes=("n_routes", "sum"),
                 n_parcels=("n_parcels", "sum")))


def load_vroom_diagnostics() -> pd.DataFrame:
    """Per (P, theta): n, MAPE, bias and R2 of the surrogate against VROOM.

    On a v2 grid it is COMPUTED from the instance table rather than read: the
    v2 validation writes its diagnostics as a markdown report, and recomputing
    them here keeps the figure and the report from drifting. Flagged rows
    (PARTIAL, unassigned, jobs removed) are excluded from the error statistics,
    which is the rule `validation_report.md` states for itself.
    """
    if val_schema() == VAL_SCHEMA_LEGACY:
        return _read(VAL / "tab_diagnostics_smoothed.csv")

    df = load_vroom_v2()
    ok = df[df.vroom_status.isin(VROOM_OK) & (df.n_unassigned == 0)
            & (df.jobs_removed == 0)].copy()
    ok["err_pct"] = ((ok.predicted_cost_eur - ok.vroom_cost_eur)
                     / ok.vroom_cost_eur * 100.0)

    def _stats(g):
        a, b = g.predicted_cost_eur, g.vroom_cost_eur
        ss = float(((b - b.mean()) ** 2).sum())
        return pd.Series(dict(
            n=int(len(g)),
            MAPE_pct=float(g.err_pct.abs().mean()),
            bias_pct=float(g.err_pct.mean()),
            R2=float(1 - ((a - b) ** 2).sum() / ss) if ss else np.nan))

    per = (ok.groupby(["penalty", "share_willing"]).apply(
        _stats, include_groups=False).reset_index())
    allrow = _stats(ok).to_frame().T
    allrow.insert(0, "share_willing", "ALL")
    allrow.insert(0, "penalty", "ALL")
    per["penalty"] = per.penalty.astype(object)
    return pd.concat([per, allrow], ignore_index=True)


def vroom_actual_baseline_available() -> bool:
    """Is there a SOLVED theta = 0 baseline to state a realised saving against?

    The v2 validation solves the scenario points first; item 0, the daily
    baseline, is a separate and much larger item. Until it is solved an
    ACTUAL saving percentage does not exist -- only predicted-vs-actual TOTALS
    do -- and a figure that shows one anyway is comparing an actual numerator
    with a predicted denominator.
    """
    if val_schema() == VAL_SCHEMA_LEGACY:
        return True
    df = load_vroom_v2()
    return bool((df.item == 0).any())


def load_savings_validation() -> pd.DataFrame:
    """Predicted vs VROOM-actual at the validated operating points.

    LEGACY: the table as published, with `actual_saving_pct` and
    `conservatism_pp`.

    v2: predicted and actual TOTALS per (item, P, theta, plan), which are
    real, plus `actual_saving_pct` / `conservatism_pp` -- which are NaN
    whenever the theta = 0 baseline has not been solved, because a saving
    needs a baseline of the same kind. `actual_available` says which case a
    caller is in; `vroom_actual_baseline_available()` is the same fact as a
    scalar.

    `saving_defined` says whether a PERCENTAGE may be formed from that row at
    all: false for item 0 (it is the baseline) and for any item the G6 budget
    fallback sampled rather than enumerated (`sampled`), whose totals are a
    subset of the system and whose percentages would be arbitrary. The totals
    of a sampled row are still real, and a caller may show them as totals.
    """
    if val_schema() == VAL_SCHEMA_LEGACY:
        return _read(VAL / "tab_savings_pred_vs_actual_smoothed.csv")

    df = load_vroom_v2()
    g = (df.groupby(["item", "penalty", "share_willing", "plan"],
                    as_index=False)
         .agg(surrogate_total_eur=("predicted_cost_eur", "sum"),
              vroom_actual_total_eur=("vroom_cost_eur", "sum"),
              n=("vroom_cost_eur", "size")))
    sampled = validation_sampled_items()
    # A saving is a system quantity. It exists for a point only if that
    # point's instances were ALL solved, and it is not defined for item 0 --
    # item 0 IS the baseline, so its "saving against itself" is a tautology
    # that would sit on the figure looking like a measurement.
    g["sampled"] = g.item.isin(sampled)
    g["is_baseline"] = g.item == 0
    g["saving_defined"] = ~g.sampled & ~g.is_baseline
    have = vroom_actual_baseline_available()
    base = float(baseline_eur(LENS_ROUTING)) if SCHEMA == SCHEMA_V2 else np.nan
    g["base_total_eur"] = base
    g["predicted_saving_pct"] = (1 - g.surrogate_total_eur / base) * 100.0
    if have:
        act_base = float(df[df.item == 0].vroom_cost_eur.sum())
        g["actual_saving_pct"] = (
            1 - g.vroom_actual_total_eur / act_base) * 100.0
    else:
        g["actual_saving_pct"] = np.nan
    g["conservatism_pp"] = g.actual_saving_pct - g.predicted_saving_pct
    g["actual_available"] = have
    # what IS real without a solved baseline: how far the surrogate's price of
    # the same tours sits from the solver's
    g["gap_pct"] = (g.surrogate_total_eur / g.vroom_actual_total_eur - 1) * 100
    return g


# --------------------------------------------------------------------------
# per-cell schedule choices, schema-dispatching
# --------------------------------------------------------------------------
def load_chosen_stage3(plan: str | None = None) -> pd.DataFrame:
    """Per-PLZ schedule choice, in the column names the frequency maps read.

    v2 grid: `load_chosen_v2(plan)`, defaulting to the OPERATOR plan. The
    figure states which plan it is on -- `plan_stamp()`.

    Legacy grid: the 2026-05-29 Path-2 run, whose stage 2 was
    frequency-PRESERVING, so it also runs the frequency-invariance check the
    per-PLZ *frequency* maps used to rely on: system smoothing reassigns which
    weekdays are served but must not change how many.

    That invariance does NOT hold on a v2 grid -- v6's stage 2 is
    frequency-free at theta > 0 (compendium 40.14) and moves the delivery
    frequency of about a fifth of all cell choices -- which is exactly why
    this function had to stop reading the legacy run unconditionally: it was
    serving a pre-revision schedule choice, under a v6 caption, to fig35/36,
    fig41-44, fig55 and the 97_/98_ backup sets.
    """
    if SCHEMA == SCHEMA_V2:
        return load_chosen_v2(grid_plan(plan))
    if plan is not None and grid_plan(plan) != CHOSEN_PLAN_DEFAULT:
        raise RuntimeError(
            f"the legacy grid {REV.name} carries ONE plan; it cannot serve "
            f"{plan!r}. Point PRES_REV_DIR at a v2 grid.")
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
