"""Persist per-area delivery costs for the chosen schedules at theta = 1.

Why theta = 1 only, on both schemas: the EXPRESS (fast-lane) tour is bundled
per hub and day, so a per-area saving built on a cell with express > 0 would
mix a hub-bundled quantity into a per-area one. At share_willing = 1.0 the
express component is exactly 0.0 for every provider and every penalty level --
verified against the grid before anything is written -- so the cell cost is
the whole cost and a per-area decomposition is exact. This is compendium
38.2(a), and it is the same rule ``scripts/revision/76_maps_v2.py`` follows.

The baseline reference is daily delivery: the full-week schedule
(Monday-Saturday), which is the (P, theta) = (0, 0) plan of the grid itself.

TWO SCHEMAS
-----------
**v2 grid** (``results/revision_2026_08_v*``, the default) -- reads
``tables/tab_per_cell_costs_v2.csv``, written by
``scripts/revision/72_per_cell_costs_v2.py``, for BOTH plans. A euro there is
the cell's full ROUTING cost under the realistic-tour rule: its own tour
(``own_cost_eur``, zero for a cell that is pooled or express-only on every
instance) plus its parcel-proportional share of every pooled small-delivery
and express-partition group price it rides on (``pool_share_eur`` /
``express_share_eur``). That is bookkeeping on prices the grid already
computed, so nothing here re-prices anything and no surrogate is loaded --
this path runs in seconds where the legacy one rebuilt every cost matrix.
The operator lens is deliberately NOT offered per area: it is hub-, not
cell-attributable (72_).

Structural features (area, hub distance, stops per day, B2C share) come from
``results/checkpoints/04_optim_prep.pkl``, a static model INPUT that no grid
revision touches; the join onto the grid's 312 cells is asserted 1:1 and the
parcel counts are asserted equal to the grid's own ``cell_parcels_week``.

Output: ``<REV>/tables/tab_per_plz_costs_theta1_v2.csv``, both plans, with a
``plan`` column. ``_data.load_per_plz(plan)`` serves one plan from it.

**legacy grid** (``results/revision_2026_07``) -- unchanged: recomputes the
Stage-3 direct-delivery cost from the production surrogate and the stored
2026-05-29 schedule choices. Output:
``results/revision_2026_07/tab_per_plz_costs_theta1.csv``. Resumable, because
that path is minutes of matrix building per cell.

Hard gates, v2 path (all fail loud, none of them a warning):
  premise  express share == 0 for every theta = 1 row, both plans
  gate A   sum of per-cell plan cost == tab_costs_v2's routing total for that
           (P, provider, plan) -- cost_stage1_eur / routing_total_eur
  gate B   the baseline cells are plan-invariant, all-daily, and sum per
           provider to that provider's (0, 0) cost; the system total matches
           _data.BASELINE_PINS for this grid
  gate C   own + pool share + express share == cell cost, row by row
  gate D   39 admissible weekly patterns; observed frequencies subset {2..6}
  gate E   every cell resolves to structural features and the same parcel
           count the grid recorded

Hard gates, legacy path (unchanged):
  gate A   sum of per-PLZ Stage-3 dd cost == tab_costs_smoothed.dd_cost_stage3_eur
  gate B   sum of per-PLZ baseline dd cost == tab_baseline_per_provider.dd_cost
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _data as D  # noqa: E402

THETA = 1.0
PENALTIES = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]

# Gate tolerances. Gate A reconciles two float64 sums of the same prices, so
# it is tight; the baseline pin is a printed euro figure and gets a euro.
GATE_A_ATOL = 1e-6
GATE_B_ATOL = 1e-6
PIN_ATOL = 1.0


# ==========================================================================
# v2 grid
# ==========================================================================
def _structural() -> pd.DataFrame:
    """Per (provider, plz) model inputs: parcels, area, hub distance, stops.

    Read from `04_optim_prep.pkl`, the same checkpoint the optimiser itself
    was handed. These are INPUTS -- no grid revision changes them -- which is
    why this path can attach them to a v6 result without re-running anything.
    """
    ck = D.CKPT / "04_optim_prep.pkl"
    if not ck.exists():
        raise FileNotFoundError(
            f"{ck} is missing; regenerate it with "
            f"scripts/revision/00_regen_checkpoints.py")
    D.prov.record(ck)
    with open(ck, "rb") as f:
        optim = pickle.load(f)["optimization_data"]
    rows = []
    for prov, od in optim.items():
        for key in od["plz_keys"]:
            pdata = od["plz_data"][key]
            b2c = float(sum(pdata["b2c"].values()))
            b2b = float(sum(pdata["b2b"].values()))
            wk = b2c + b2b
            area = float(pdata["area_km2"])
            rows.append(dict(
                provider=prov, plz=str(key).zfill(5),
                weekly_parcels=wk,
                b2c_share=(b2c / wk) if wk > 0 else np.nan,
                area_km2=area,
                hub_dist_km=float(pdata["hub_dist_km"]),
                n_stops_per_day=float(pdata["n_stops_per_day"]),
                demand_per_area=(wk / area) if area > 0 else np.nan))
    df = pd.DataFrame(rows)
    assert df.duplicated(["provider", "plz"]).sum() == 0, (
        "the checkpoint has duplicate (provider, plz) cells")
    return df


def _run_v2() -> Path:
    out = D.per_plz_v2_path()
    src = D.per_cell_costs_path()
    if src is None:
        raise FileNotFoundError(
            f"{D.REV.name} has no tables/tab_per_cell_costs_v2.csv; run "
            f"scripts/revision/72_per_cell_costs_v2.py on it first")
    t0 = time.time()

    cells = pd.read_csv(src, dtype={"plz": str})
    D.prov.record(src)
    cells["plz"] = cells.plz.astype(str).str.zfill(5)
    plans = sorted(cells.plan.astype(str).unique())
    assert set(plans) == set(D.GRID_PLANS), (
        f"{src.name} carries plans {plans}, expected {list(D.GRID_PLANS)}")
    print(f"read {len(cells):,} per-cell rows, plans {plans}", flush=True)

    # ---- gate C: the three parts are the whole cost, row by row -----------
    parts = cells.own_cost_eur + cells.pool_share_eur + cells.express_share_eur
    assert np.allclose(parts, cells.cell_cost_eur, atol=GATE_A_ATOL), (
        "GATE C FAIL: own + pool share + express share != cell cost "
        f"(max |delta| {float((parts - cells.cell_cost_eur).abs().max()):.3e})")
    print("gate C OK: own + pool + express == cell cost for every row",
          flush=True)

    # ---- premise: express is exactly zero at theta = 1 --------------------
    at_theta = cells[np.isclose(cells.share_willing, THETA)]
    assert len(at_theta), f"the grid has no theta = {THETA} rows"
    assert (at_theta.express_share_eur == 0.0).all(), (
        f"express is NOT exactly 0 at theta = {THETA} "
        f"(max {float(at_theta.express_share_eur.abs().max()):.6g}) -- a "
        f"per-area decomposition would mix a hub-bundled quantity into a "
        f"per-area one")
    print(f"premise OK: express share == 0.0 for all {len(at_theta):,} "
          f"theta = {THETA} cell rows", flush=True)

    # ---- gate B: the baseline is plan-invariant, daily, and pinned --------
    base = cells[np.isclose(cells.penalty, 0.0)
                 & np.isclose(cells.share_willing, 0.0)]
    b = {q: base[base.plan == q].set_index(["provider", "plz"]).sort_index()
         for q in D.GRID_PLANS}
    assert b[D.PLAN_STAGE1].index.equals(b[D.PLAN_BALANCED].index), (
        "GATE B FAIL: the two plans cover different baseline cells")
    assert np.allclose(b[D.PLAN_STAGE1].cell_cost_eur,
                       b[D.PLAN_BALANCED].cell_cost_eur, atol=GATE_B_ATOL), (
        "GATE B FAIL: the baseline is not plan-invariant; theta = 0 is "
        "supposed to be a stage-2 no-op (61_grid_run_v2 G-6f-1)")
    assert (b[D.PLAN_BALANCED].mean_days == 6).all(), (
        "GATE B FAIL: the baseline is not all-daily")
    baseline = (b[D.PLAN_BALANCED].reset_index()
                [["provider", "plz", "cell_cost_eur"]]
                .rename(columns={"cell_cost_eur": "cell_cost_baseline_eur"}))
    pin = D.BASELINE_PINS.get(D.REV.name)
    total = float(baseline.cell_cost_baseline_eur.sum())
    if pin is None:
        print(f"gate B: {D.REV.name} has no pinned baseline; the grid's own "
              f"is {total:,.2f} EUR", flush=True)
    else:
        assert abs(total - pin["routing"]) <= PIN_ATOL, (
            f"GATE B FAIL: per-cell baseline sum {total:.2f} != the pinned "
            f"routing baseline {pin['routing']:.2f} of {D.REV.name}")
        print(f"gate B OK: per-cell baseline sums to {total:,.2f} EUR, the "
              f"pinned routing baseline of {D.REV.name}", flush=True)

    # ---- gate A: per-cell sums reproduce the grid's provider totals -------
    costs = pd.read_csv(D.REV / "tab_costs_v2.csv")
    D.prov.record(D.REV / "tab_costs_v2.csv")
    ref_col = {D.PLAN_STAGE1: "cost_stage1_eur",
               D.PLAN_BALANCED: "routing_total_eur"}
    at = cells[np.isclose(cells.share_willing, THETA)]
    for q in D.GRID_PLANS:
        got = (at[at.plan == q]
               .groupby(["penalty", "provider"], as_index=False)
               .cell_cost_eur.sum())
        want = costs[np.isclose(costs.share_willing, THETA)][
            ["penalty", "provider", ref_col[q]]]
        m = got.merge(want, on=["penalty", "provider"], how="outer",
                      indicator=True)
        assert (m._merge == "both").all(), (
            f"GATE A FAIL ({q}): {int((m._merge != 'both').sum())} "
            f"(P, provider) pairs are in one table and not the other")
        delta = float((m.cell_cost_eur - m[ref_col[q]]).abs().max())
        assert delta < GATE_A_ATOL, (
            f"GATE A FAIL ({q}): per-cell sum differs from "
            f"tab_costs_v2.{ref_col[q]} by up to {delta:.3e} EUR")
        print(f"gate A OK ({q}): {len(m)} (P, provider) totals reproduce "
              f"tab_costs_v2.{ref_col[q]} to {delta:.2e} EUR", flush=True)

    # ---- gate D: the schedule enumeration and admissible frequencies ------
    size, days, wait = D._schedule_lookup()
    idx = at.schedule_idx.to_numpy()
    observed = set(int(x) for x in size[idx])
    assert observed <= set(D.FREQ_SIZES), (
        f"GATE D FAIL: delivery frequencies "
        f"{sorted(observed - set(D.FREQ_SIZES))} fall outside the admissible "
        f"set {D.FREQ_SIZES}")
    assert np.allclose(size[idx], at.mean_days.to_numpy()), (
        "GATE D FAIL: the schedule enumeration's sizes disagree with the "
        "grid's own mean_days -- the enumeration or its order changed")
    assert np.allclose(wait[idx], at.wait_days.to_numpy()), (
        "GATE D FAIL: the schedule enumeration's waits disagree with the "
        "grid's own wait_days")
    print(f"gate D OK: 39 patterns, frequencies {sorted(observed)}, sizes and "
          f"waits reproduce the grid's own columns", flush=True)

    # ---- assemble --------------------------------------------------------
    struct = _structural()
    out_df = at.copy()
    out_df["schedule_size_stage3"] = size[idx]
    out_df["schedule_days_stage3"] = [
        "".join(str(d) for d in sorted(x))
        for x in _sorted_days(out_df.schedule_idx.to_numpy())]
    out_df["avg_wait_d_stage3"] = wait[idx]
    out_df = out_df.merge(baseline, on=["provider", "plz"], how="left")

    # ---- gate E: every cell has its structural inputs, and the same volume
    before = len(out_df)
    out_df = out_df.merge(struct, on=["provider", "plz"], how="left",
                          validate="many_to_one")
    assert len(out_df) == before, (
        f"GATE E FAIL: the structural join changed the row count "
        f"{before} -> {len(out_df)}")
    assert out_df.area_km2.notna().all(), (
        f"GATE E FAIL: {int(out_df.area_km2.isna().sum())} cell(s) have no "
        f"structural features in 04_optim_prep.pkl")
    assert out_df.cell_cost_baseline_eur.notna().all(), (
        f"GATE E FAIL: {int(out_df.cell_cost_baseline_eur.isna().sum())} "
        f"cell(s) have no baseline row")
    assert np.allclose(out_df.weekly_parcels, out_df.cell_parcels_week), (
        "GATE E FAIL: the checkpoint's parcel counts differ from the grid's "
        "cell_parcels_week -- the two are not describing the same cells")
    print(f"gate E OK: {len(out_df):,} rows joined 1:1 to structural inputs, "
          f"parcel counts identical", flush=True)

    out_df["saving_abs_eur"] = (out_df.cell_cost_baseline_eur
                               - out_df.cell_cost_eur)
    out_df["saving_pct"] = np.where(
        out_df.cell_cost_baseline_eur > 0,
        (1.0 - out_df.cell_cost_eur / out_df.cell_cost_baseline_eur) * 100.0,
        np.nan)

    keep = ["penalty", "share_willing", "provider", "plz", "plan", "hub",
            "head_id", "schedule_idx", "schedule_size_stage3",
            "schedule_days_stage3", "avg_wait_d_stage3",
            "own_cost_eur", "pool_share_eur", "express_share_eur",
            "cell_cost_eur", "cell_cost_baseline_eur", "saving_abs_eur",
            "saving_pct", "weekly_parcels", "b2c_share", "area_km2",
            "hub_dist_km", "n_stops_per_day", "demand_per_area"]
    out_df = out_df[keep].sort_values(
        ["plan", "penalty", "provider", "plz"]).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"\nALL GATES PASSED -- {out} "
          f"({len(out_df):,} rows, {len(plans)} plans, {time.time() - t0:.1f}s)")

    # the number Act 7 puts in its panel titles, restated here so a bad port
    # is visible in this script's own output and not only three figures later
    for q in D.GRID_PLANS:
        g = (out_df[out_df.plan == q].groupby("penalty", as_index=False)
             .agg(base=("cell_cost_baseline_eur", "sum"),
                  plan_eur=("cell_cost_eur", "sum")))
        g["saving_pct"] = (1 - g.plan_eur / g.base) * 100
        print(f"  system saving, {q}: "
              + " ".join(f"P={r.penalty:g} {r.saving_pct:.1f}%"
                         for r in g.itertuples()))
    return out


def _sorted_days(idx: np.ndarray):
    from batch_delivery.optimization.schedules import enumerate_valid_schedules
    sched = enumerate_valid_schedules()
    return [sorted(sched[int(i)]) for i in idx]


# ==========================================================================
# legacy grid (unchanged)
# ==========================================================================
def _legacy_paths():
    sys.path.insert(0, str(D.ROOT / "scripts" / "revision"))
    import _stage3_common as C  # noqa: E402
    return C


def _load_state(state_json: Path) -> list[list[float]]:
    if state_json.exists():
        return json.loads(state_json.read_text())["completed"]
    return []


def _save_state(state_json: Path, completed: list[list[float]]) -> None:
    tmp = state_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"completed": completed}, indent=1))
    tmp.replace(state_json)


def _prune_partial(out_csv: Path, completed: list[list[float]]) -> None:
    """Self-heal: drop rows of any cell not recorded as completed."""
    if not out_csv.exists():
        return
    df = pd.read_csv(out_csv, dtype={"plz": str})
    if len(df) == 0:
        return
    mask = np.zeros(len(df), dtype=bool)
    for P, s in completed:
        mask |= np.isclose(df.penalty, P) & np.isclose(df.share_willing, s)
    if not mask.all():
        print(f"[self-heal] dropping {int((~mask).sum())} row(s) from "
              f"partially-appended cell(s)", flush=True)
        df[mask].to_csv(out_csv, index=False)


def _full_week_index(schedules: list, n_days: int) -> int:
    """Index of the Monday-Saturday (daily delivery) schedule."""
    full = frozenset(range(n_days))
    hits = [i for i, s in enumerate(schedules) if s == full]
    assert len(hits) == 1, f"expected exactly one full-week schedule, got {hits}"
    return hits[0]


def _run_legacy() -> Path:
    C = _legacy_paths()
    sys.path.insert(0, str(C.ROOT / "src"))
    from batch_delivery.optimization.core import build_cost_matrices_ml

    out_csv = C.OUT_DIR / "tab_per_plz_costs_theta1.csv"
    state_json = C.OUT_DIR / "state_per_plz.json"
    baseline_csv = (C.ROOT / "results" / "paper_outputs_2026_05_30"
                    / "02_baseline" / "tab_baseline_per_provider.csv")
    gate_b_rtol = 1e-9

    provider_data, optim_data = C.load_checkpoints()
    model = C.load_model()
    ml_prep = C.build_ml_prep(provider_data)
    schedules = C.enumerate_schedules()
    assert len(schedules) == 39, f"expected 39 patterns, got {len(schedules)}"
    full_idx = _full_week_index(schedules, C.N_DAYS)
    sched_waits = np.array([C.avg_wait_days(sorted(s)) for s in schedules])

    smoothed = pd.read_csv(C.RUN_DIR / "_tab_chosen_with_system_smoothing.csv")
    smoothed["plz"] = smoothed.plz.astype(str)
    ref_cost = pd.read_csv(C.OUT_DIR / "tab_costs_smoothed.csv")
    ref_base = pd.read_csv(baseline_csv)

    expr_at_theta = ref_cost[np.isclose(ref_cost.share_willing, THETA)]
    assert (expr_at_theta.express_stage3_eur == 0.0).all(), (
        "express is not exactly 0 at theta=%s -- per-PLZ decomposition would "
        "mix a hub-bundled quantity into a per-PLZ one" % THETA
    )
    print(f"premise OK: express == 0.0 for all {len(expr_at_theta)} "
          f"(P, provider) rows at theta={THETA}", flush=True)

    cells = [(P, THETA) for P in PENALTIES]
    _cells_env = os.environ.get("PRES_CELLS")
    if _cells_env:
        wanted = {float(x.strip()) for x in _cells_env.split(";") if x.strip()}
        cells = [c for c in cells if any(np.isclose(c[0], w) for w in wanted)]
        print(f"[env guard] PRES_CELLS={_cells_env!r} -> {len(cells)} cell(s)",
              flush=True)

    completed = _load_state(state_json)
    _prune_partial(out_csv, completed)
    n_target = len(cells)
    todo = [c for c in cells if not any(
        np.isclose(c[0], w[0]) and np.isclose(c[1], w[1]) for w in completed
    )]
    if len(todo) < n_target:
        print(f"[resume] {n_target - len(todo)} cell(s) already done, "
              f"{len(todo)} remaining", flush=True)

    fs_b2c_v, fs_b2b_v = C.fs_b2c(THETA), C.fs_b2b(THETA)
    t0 = time.time()

    for i, (P, share) in enumerate(todo, 1):
        rows: list[dict] = []

        for prov in C.PROVIDERS:
            od, prep = optim_data[prov], ml_prep[prov]
            plz_keys = od["plz_keys"]

            m = build_cost_matrices_ml(
                plz_keys, od["plz_data"], schedules, model, prov,
                prep["plz_day_coords"], prep["hub_coords_by_plz"],
                fast_share_b2c=fs_b2c_v, fast_share_b2b=fs_b2b_v,
            )

            sub = smoothed[(np.isclose(smoothed.penalty, P))
                           & (np.isclose(smoothed.share_willing, share))
                           & (smoothed.provider == prov)].set_index("plz")
            chosen = np.array(
                [int(sub.loc[str(pc), "schedule_idx_system_smoothed"])
                 for pc in plz_keys], dtype=np.int64)

            dd_mx = (m["cost_3d"] * m["sched_active"][None, :, :]).sum(axis=2)
            dd_s3 = dd_mx[np.arange(len(plz_keys)), chosen]
            dd_base = dd_mx[:, full_idx]

            want = ref_cost[(np.isclose(ref_cost.penalty, P))
                            & (np.isclose(ref_cost.share_willing, share))
                            & (ref_cost.provider == prov)]
            assert len(want) == 1, f"no reference cost row for {P}/{share}/{prov}"
            want_dd = float(want.iloc[0].dd_cost_stage3_eur)
            got_dd = float(dd_s3.sum())
            assert abs(got_dd - want_dd) < GATE_A_ATOL, (
                f"GATE A FAIL P={P} {prov}: per-PLZ sum {got_dd!r} != "
                f"stored dd_cost_stage3_eur {want_dd!r} "
                f"(delta {got_dd - want_dd:.6e})"
            )

            want_base = float(
                ref_base[ref_base.provider == prov].iloc[0].dd_cost)
            got_base = float(dd_base.sum())
            assert np.isclose(got_base, want_base, rtol=gate_b_rtol), (
                f"GATE B FAIL {prov}: baseline sum {got_base!r} != "
                f"tab_baseline_per_provider dd_cost {want_base!r} "
                f"(rel {abs(got_base - want_base) / want_base:.3e})"
            )

            hub_name_by_plz = prep["hub_name_by_plz"]
            for j, pc in enumerate(plz_keys):
                pd_ = od["plz_data"][pc]
                b2c_wk = float(sum(pd_["b2c"].values()))
                b2b_wk = float(sum(pd_["b2b"].values()))
                wk = b2c_wk + b2b_wk
                sched = sorted(schedules[int(chosen[j])])
                rows.append(dict(
                    penalty=P, share_willing=share, provider=prov, plz=str(pc),
                    hub=hub_name_by_plz.get(pc, ""),
                    schedule_size_stage3=len(sched),
                    schedule_days_stage3="".join(str(d) for d in sched),
                    avg_wait_d_stage3=float(sched_waits[int(chosen[j])]),
                    dd_cost_stage3_eur=float(dd_s3[j]),
                    dd_cost_baseline_eur=float(dd_base[j]),
                    saving_abs_eur=float(dd_base[j] - dd_s3[j]),
                    saving_pct=float(
                        (1.0 - dd_s3[j] / dd_base[j]) * 100.0)
                    if dd_base[j] > 0 else np.nan,
                    weekly_parcels=wk,
                    b2c_share=(b2c_wk / wk) if wk > 0 else np.nan,
                    area_km2=float(pd_["area_km2"]),
                    hub_dist_km=float(pd_["hub_dist_km"]),
                    n_stops_per_day=float(pd_["n_stops_per_day"]),
                    demand_per_area=(wk / float(pd_["area_km2"]))
                    if pd_["area_km2"] > 0 else np.nan,
                ))

        pd.DataFrame(rows).to_csv(out_csv, mode="a",
                                  header=not out_csv.exists(), index=False)
        completed.append([float(P), float(share)])
        _save_state(state_json, completed)
        print(f"[{i:2d}/{len(todo)}] P={P:<5g} theta={share:<4g} gates A+B OK "
              f"({len(completed)}/{n_target} total, {time.time() - t0:.0f}s)",
              flush=True)

    if len(completed) >= n_target:
        print(f"ALL GATES PASSED -- {out_csv}")
    else:
        print(f"INCOMPLETE: {len(completed)}/{n_target} cells done; "
              f"rerun to resume")
    return out_csv


def main() -> None:
    if D.SCHEMA == D.SCHEMA_V2:
        print(f"v2 grid {D.REV.name}: per-area costs from "
              f"tables/tab_per_cell_costs_v2.csv, both plans, theta = {THETA}")
        _run_v2()
    else:
        print(f"legacy grid {D.REV.name}: recomputing per-PLZ Stage-3 costs "
              f"from the production surrogate")
        _run_legacy()


if __name__ == "__main__":
    main()
