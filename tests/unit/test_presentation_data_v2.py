"""The presentation layer's v2-schema adapter, on a synthetic grid.

`scripts/presentation/_data.py` serves the talk decks from either of two grid
schemas: the submission-era one (`revision_2026_07`, one plan and one cost
lens) or the revision's (`revision_2026_08_v*`, two plans and two lenses). The
deck builders read it through the same loader names either way, which is
exactly the kind of adapter that goes quietly wrong.

These tests build a five-row grid by hand, with numbers chosen so that every
saving comes out round, and check the three things a wrong adapter would get
wrong without failing:

* it picks the right column for each (plan, lens) pair -- swapping two of them
  would still produce plausible percentages;
* it forms each percentage against the baseline of the grid it is reading, not
  against the other grid's pinned baseline;
* it refuses to write over a deck that already exists.

No file in `results/` is read: the fixtures are written into `tmp_path`, so the
tests run without the real grid and cannot be satisfied by it.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
PRES = ROOT / "scripts" / "presentation"

# 189.15 EUR per vehicle-day, six days a week, per peak vehicle and hub.
WEEK_FIXED = 6 * 189.15

PROVIDERS = ["DHL", "Amazon", "DPD", "FedEx", "GLS", "Hermes", "UPS"]


# ── fixtures ───────────────────────────────────────────────────────────────
def _cost_row(P, th, prov, *, rout1, rout2, var1, var2, peak1, peak2,
              vd1=100.0, vd2=100.0, pen1=0.0, pen2=0.0):
    """One row of tab_costs_v2, in the grid's own column convention.

    Plain names are the STAGE-2 (operator-polished) plan; the stage-1 values
    carry `_stage1` or live in a `*_before` column. `cost_stage1_eur` and
    `routing_total_eur` are both pure routing euro -- no penalty in either.
    """
    return dict(
        penalty=P, share_willing=th, provider=prov,
        dd_cost_eur=rout2, express_cost_eur=0.0, pool_cost_eur=0.0,
        routing_total_eur=rout2, penalty_eur=pen2,
        cost_stage1_eur=rout1, cost_stage2_eur=rout2, cost_stage3_eur=rout2,
        vehicle_days=vd2, fixed_cost_eur=0.0, variable_cost_eur=var2,
        sum_hub_peak=peak2, week_fixed_cost_eur=WEEK_FIXED * peak2,
        operator_cost_eur=var2 + WEEK_FIXED * peak2,
        operator_obj_eur=var2 + WEEK_FIXED * peak2 + pen2,
        operator_cost_before_eur=var1 + WEEK_FIXED * peak1,
        variable_before_eur=var1, variable_after_eur=var2,
        sum_hub_peak_before=peak1, sum_hub_peak_after=peak2,
        vehicle_days_before=vd1, vehicle_days_after=vd2,
        penalty_before_eur=pen1,
    )


@pytest.fixture
def grid(tmp_path):
    """A minimal but complete v2 grid: baseline plus two operating points."""
    rev = tmp_path / "revision_synthetic_v2"
    (rev / "tables").mkdir(parents=True)
    (rev / "_peek").mkdir()

    rows = []
    # (P = 0, theta = 0) is the baseline: both plans coincide, by construction
    # and by the grid's own theta = 0 pin.
    for prov in PROVIDERS:
        rows.append(_cost_row(0.0, 0.0, prov, rout1=1000.0, rout2=1000.0,
                              var1=400.0, var2=400.0, peak1=10, peak2=10))
    # (P = 0, theta = 1): the routing optimum saves routing euro and RAISES
    # the hub peak; the operator polish gives some of that back and cuts the
    # peak. The four (plan, lens) cells are deliberately all different.
    for prov in PROVIDERS:
        rows.append(_cost_row(0.0, 1.0, prov, rout1=800.0, rout2=900.0,
                              var1=320.0, var2=360.0, peak1=14, peak2=8))
    # (P = 1, theta = 1): a penalty-bearing point, so a loader that mistakes
    # the penalty for cost shows up.
    for prov in PROVIDERS:
        rows.append(_cost_row(1.0, 1.0, prov, rout1=950.0, rout2=960.0,
                              var1=380.0, var2=384.0, peak1=11, peak2=9,
                              pen1=50.0, pen2=30.0))
    pd.DataFrame(rows).to_csv(rev / "tab_costs_v2.csv", index=False)

    wait = []
    for P, th, n1, n2, d1, d2 in ((0.0, 0.0, 0.0, 0.0, 6.0, 6.0),
                                  (0.0, 1.0, 1000.0, 700.0, 2.0, 2.4),
                                  (1.0, 1.0, 200.0, 250.0, 5.0, 4.8)):
        for prov in PROVIDERS:
            wait.append(dict(
                penalty=P, share_willing=th, provider=prov,
                wait_num_willing=n2, wait_num_all=n2, total_parcels=1000.0,
                willing_parcels=1000.0 * th, wait_num_willing_stage1=n1,
                wait_num_all_stage1=n1, mean_days=d2, mean_days_stage1=d1))
    pd.DataFrame(wait).to_csv(rev / "tab_wait_v2.csv", index=False)

    fleet = []
    for P, th in ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
        for prov in PROVIDERS:
            for day in range(6):
                fleet.append(dict(
                    penalty=P, share_willing=th, provider=prov,
                    hub=f"{prov} depot", day=day, dd_single_veh=1.0,
                    dd_pool_veh=0.0, express_veh=0.0, fleet=float(day + 1)))
    pd.DataFrame(fleet).to_csv(rev / "tab_fleet_per_hub_v2.csv", index=False)

    # 38 is the daily schedule in enumerate_valid_schedules(); 0 is a two-day
    # one. The routing optimum consolidates one of the two cells, the operator
    # polish puts it back on six days.
    chosen = []
    for P, th, s1, s2 in ((0.0, 0.0, 38, 38), (0.0, 1.0, 0, 38),
                          (1.0, 1.0, 0, 0)):
        for prov in PROVIDERS:
            for plz in ("30159", "30169"):
                idx1 = s1 if plz == "30159" else 38
                idx2 = s2 if plz == "30159" else 38
                chosen.append(dict(
                    penalty=P, share_willing=th, provider=prov, plz=plz,
                    schedule_idx_stage1=idx1, schedule_idx_balanced=idx2,
                    schedule_idx_system_smoothed=idx2))
    pd.DataFrame(chosen).to_csv(rev / "_tab_chosen_v2.csv", index=False)

    pd.DataFrame([dict(
        provider=p, P_star_submission=0.25, class_submission="Service-bound",
        P_star_routing=0.25, carrier_class_routing="Service-bound",
        saving_pct_routing=10.0, wait_d_routing=0.5,
        P_star_operator=0.5 if p == "Amazon" else 0.25,
        carrier_class_operator="Hybrid" if p == "Amazon" else "Service-bound",
        saving_pct_operator=12.0, wait_d_operator=0.3)
        for p in PROVIDERS]).to_csv(
            rev / "tables" / "tab_pstar_knees_v2.csv", index=False)
    return rev


@pytest.fixture
def D(grid, monkeypatch):
    """`_data`, imported fresh and pointed at the synthetic grid."""
    monkeypatch.syspath_prepend(str(PRES))
    monkeypatch.setenv("PRES_REV_DIR", str(grid))
    sys.modules.pop("_data", None)
    mod = importlib.import_module("_data")
    assert mod.SCHEMA == mod.SCHEMA_V2
    assert mod.REV == grid.resolve()
    # The synthetic grid gets its own entry in the per-grid pin table, so the
    # assert under test is the real one -- not disabled, just told what this
    # grid is supposed to say.
    monkeypatch.setitem(
        mod.BASELINE_PINS, grid.resolve().name,
        dict(routing=7000.0, operator=2800.0 + 7 * WEEK_FIXED * 10,
             hub_peak=70, vehicle_days=700, cv=0.0))
    mod._CACHE.clear()
    yield mod
    sys.modules.pop("_data", None)


# ── the baseline ───────────────────────────────────────────────────────────
def test_baseline_comes_from_the_grid_not_from_the_other_grid(D):
    b = D.baseline_v2()
    assert b["routing_eur"] == pytest.approx(7000.0)
    assert b["operator_eur"] == pytest.approx(2800.0 + 7 * WEEK_FIXED * 10)
    assert b["hub_peak"] == 70
    # the legacy pin must not leak into a v2 percentage
    assert D.baseline_eur(D.LENS_ROUTING) != D.BASE_TOTAL


def test_baseline_fails_loudly_when_the_grid_moves(D, monkeypatch):
    pins = dict(D.BASELINE_PINS[D.REV.name], routing=1.0)
    monkeypatch.setitem(D.BASELINE_PINS, D.REV.name, pins)
    D._CACHE.clear()
    with pytest.raises(AssertionError, match="baseline routing_eur"):
        D.baseline_v2()


def test_an_unpinned_grid_is_allowed_but_says_so(D, monkeypatch, capsys):
    """A grid with no recorded reference must not fail -- and must not be
    silent about being unchecked."""
    monkeypatch.delitem(D.BASELINE_PINS, D.REV.name)
    D._CACHE.clear()
    b = D.baseline_v2()
    assert b["routing_eur"] == pytest.approx(7000.0)
    assert "no pinned reference" in capsys.readouterr().out


# ── two plans x two lenses ─────────────────────────────────────────────────
def test_each_plan_lens_pair_reads_its_own_column(D):
    assert D.cost_column(D.PLAN_ROUTING, D.LENS_ROUTING) == "cost_stage1_eur"
    assert D.cost_column(D.PLAN_OPERATOR, D.LENS_ROUTING) == "routing_total_eur"
    assert (D.cost_column(D.PLAN_ROUTING, D.LENS_OPERATOR)
            == "operator_cost_before_eur")
    assert D.cost_column(D.PLAN_OPERATOR, D.LENS_OPERATOR) == "operator_cost_eur"


@pytest.mark.parametrize("plan,lens,want", [
    # routing lens: 7 x 800 vs 7 x 1000 = 20 %; 7 x 900 = 10 %
    ("routing", "routing", 20.0),
    ("operator", "routing", 10.0),
    # operator lens: variable + 1134.90 per peak vehicle and hub, against a
    # baseline of 7 x (400 + 10 x 1134.90) = 82 243 EUR
    ("routing", "operator", -37.96),
    ("operator", "operator", 19.66),
])
def test_saving_of_each_plan_in_each_lens(D, plan, lens, want):
    g = D.saving_grid_v2(plan, lens)
    row = g[(g.penalty == 0.0) & (g.share_willing == 1.0)].iloc[0]
    assert row.saving_pct == pytest.approx(want, abs=0.01)


def test_the_operator_lens_can_be_negative_where_routing_is_positive(D):
    """The revision's central finding, on a grid built to reproduce it."""
    rout = D.saving_grid_v2(D.PLAN_ROUTING, D.LENS_ROUTING)
    oper = D.saving_grid_v2(D.PLAN_ROUTING, D.LENS_OPERATOR)
    at = lambda g: g[(g.penalty == 0.0) & (g.share_willing == 1.0)].iloc[0]
    assert at(rout).saving_pct > 0 > at(oper).saving_pct


def test_peak_fleet_follows_the_plan_not_the_lens(D):
    for lens in D.LENSES:
        r = D.saving_grid_v2(D.PLAN_ROUTING, lens)
        o = D.saving_grid_v2(D.PLAN_OPERATOR, lens)
        at = lambda g: g[(g.penalty == 0.0) & (g.share_willing == 1.0)].iloc[0]
        assert at(r).hub_peak == 7 * 14
        assert at(o).hub_peak == 7 * 8


def test_the_service_penalty_is_not_counted_as_cost(D):
    """At P = 1 both plans carry a penalty; neither cost column may include it."""
    g = D.saving_grid_v2(D.PLAN_ROUTING, D.LENS_ROUTING)
    row = g[(g.penalty == 1.0)].iloc[0]
    assert row.total_eur == pytest.approx(7 * 950.0)
    assert row.penalty_eur == pytest.approx(7 * 50.0)


def test_an_unknown_plan_or_lens_is_refused(D):
    with pytest.raises(ValueError, match="unknown plan"):
        D.saving_grid_v2("stage7", D.LENS_ROUTING)
    with pytest.raises(ValueError, match="unknown lens"):
        D.saving_grid_v2(D.PLAN_ROUTING, "carbon")


# ── wait, frequency and the per-area fallback ──────────────────────────────
def test_wait_is_per_plan_and_weighted_over_all_parcels(D):
    r = D.wait_grid_v2(D.PLAN_ROUTING)
    o = D.wait_grid_v2(D.PLAN_OPERATOR)
    at = lambda g: g[(g.penalty == 0.0) & (g.share_willing == 1.0)].iloc[0]
    assert at(r).wait_d == pytest.approx(1.0)      # 7x1000 / 7x1000
    assert at(o).wait_d == pytest.approx(0.7)
    assert at(r).mean_days == pytest.approx(2.0)
    assert at(o).mean_days == pytest.approx(2.4)


def test_delivery_frequency_comes_from_the_schedule_enumeration(D):
    s1 = D.chosen_sizes_v2(D.PLAN_ROUTING)
    s2 = D.chosen_sizes_v2(D.PLAN_OPERATOR)
    at = lambda s: s[(s.penalty == 0.0) & (s.share_willing == 1.0)]
    assert set(at(s1).schedule_size) == {2, 6}
    assert set(at(s2).schedule_size) == {6}
    # half the cells consolidate under the routing optimum, none after the
    # operator polish -- the shape of the one-area-depot finding
    assert D.consolidating_share_v2(0.0, 1.0, D.PLAN_ROUTING) == 50.0
    assert D.consolidating_share_v2(0.0, 1.0, D.PLAN_OPERATOR) == 0.0


def test_per_area_euro_needs_the_per_cell_table(D, tmp_path):
    """Without per-cell costs a euro map cannot be drawn, and saying so is the
    whole point; with them the loader picks the requested plan."""
    assert D.per_cell_costs_path() is None
    assert D.per_plz_eur_available() is False
    with pytest.raises(FileNotFoundError, match="no per-cell plan costs"):
        D.load_per_cell_costs_v2()

    (D.REV / "tables").mkdir(exist_ok=True)
    pd.DataFrame([
        dict(penalty=0.0, share_willing=1.0, provider="DHL", plz="30159",
             plan=plan, cell_cost_eur=100.0 + i)
        for i, plan in enumerate(("stage1", "balanced"))
    ]).to_csv(D.REV / "tables" / "tab_per_cell_costs_v2.csv", index=False)
    D._CACHE.clear()
    assert D.per_plz_eur_available() is True
    assert float(D.load_per_cell_costs_v2(D.PLAN_ROUTING).cell_cost_eur.iloc[0]) == 100.0
    assert float(D.load_per_cell_costs_v2(D.PLAN_OPERATOR).cell_cost_eur.iloc[0]) == 101.0


def test_hub_day_profile_is_matched_by_name_and_must_be_unique(D):
    hub, prof = D.hub_day_profile_v2("DHL", 0.0, 1.0)
    assert hub == "DHL depot"
    assert prof == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    with pytest.raises(AssertionError, match="matches 7 hubs"):
        D.hub_day_profile_v2("depot", 0.0, 1.0)


# ── legacy loader names, served from the v2 tables ─────────────────────────
def test_legacy_loader_names_still_work_on_a_v2_grid(D):
    c = D.load_costs(D.PLAN_OPERATOR)
    assert "total_stage3_eur" in c.columns
    assert c.total_stage3_eur.equals(c.routing_total_eur)
    w = D.load_wait(D.PLAN_OPERATOR)
    assert "avg_wait_d_stage3" in w.columns
    f = D.load_fleet_fixed()
    assert {"fleet_fixed", "fleet_stage2", "fleet_stage3"} <= set(f.columns)
    e = D.load_express()
    assert "express_stage3_eur" in e.columns
    k = D.load_pstar()
    assert set(k.P_star) == {0.25} and "P_star_operator" in k.columns


def test_saving_grid_dispatches_to_the_v2_path(D):
    a = D.saving_grid()
    b = D.saving_grid_v2(D.PLAN_ROUTING, D.LENS_ROUTING)
    assert a.saving_pct.tolist() == b.saving_pct.tolist()


def test_the_validation_dir_is_chosen_separately_from_the_grid(D, tmp_path):
    """A validation run lags the grid it validates, so VAL has its own setting.

    This is what stops a figure being drawn from a validation directory that
    is still being written while its grid is already final.
    """
    assert D.VAL.parent != D.REV, (
        "VAL followed REV; a live validation output would be read as if it "
        "were finished")
    val = tmp_path / "someval"
    val.mkdir()
    (val / "tab_vroom_smoothed.csv").write_text("penalty\n0.0\n")
    D.set_val_dir(val)
    assert D.VAL == val.resolve() and D.val_schema() == D.VAL_SCHEMA_LEGACY
    assert D.REV != val, "set_val_dir must not move the cost grid"


def test_unchanged_analyses_stay_pinned_to_the_submission_grid(D):
    """These two must not follow PRES_REV_DIR, or they change meaning.

    `load_per_plz` used to be on this list and is deliberately no longer:
    v6 ships per-cell plan costs, so the per-area decomposition exists on the
    revision grid and pinning it to the submission's would date every Act-7
    map without saying so. Its own dispatch is tested below.
    """
    import inspect
    for fn in (D.load_alpha_sensitivity, D.load_cd_restart_spread):
        assert "REV_LEGACY" in inspect.getsource(fn), fn.__name__
    assert D.REV_LEGACY != D.REV


def test_v2_only_loaders_say_what_is_missing_on_a_legacy_grid(D, tmp_path):
    legacy = tmp_path / "revision_legacy"
    legacy.mkdir()
    (legacy / "tab_costs_smoothed.csv").write_text(
        "penalty,share_willing,provider,dd_cost_stage3_eur,"
        "express_stage3_eur,total_stage3_eur\n0.0,1.0,DHL,1.0,0.0,1.0\n")
    D.set_rev_dir(legacy)
    assert D.SCHEMA == D.SCHEMA_LEGACY
    with pytest.raises(RuntimeError, match="needs a v2 revision grid"):
        D.load_costs_v2()


def test_a_directory_that_is_neither_schema_is_refused(D, tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="neither a v2 grid"):
        D.set_rev_dir(empty)


# ── the one-area depot count, which used to be a typed-in literal ──────────
def test_one_cell_hubs_is_counted_not_typed(D, tmp_path, monkeypatch):
    """The count on the slides comes from the hub assignment, not a constant.

    It was a literal once, and the literal was wrong (9 against the
    compendium's and the figure's 8) on two built decks. This is the check
    that was missing.
    """
    legacy = tmp_path / "legacy_hubs"
    legacy.mkdir()
    rows = []
    for hub, cells in (("A", 1), ("B", 1), ("C", 3)):        # 2 of 3 one-cell
        for n in range(cells):
            rows.append(dict(penalty=0.0, share_willing=1.0, provider="DHL",
                             plz=f"3{hub}{n:03d}", hub=hub))
    rows.append(dict(penalty=0.0, share_willing=1.0, provider="GLS",
                     plz="30159", hub="only"))
    pd.DataFrame(rows).to_csv(
        legacy / "tab_per_plz_costs_theta1.csv", index=False)
    monkeypatch.setattr(D, "REV_LEGACY", legacy)
    D._CACHE.clear()
    assert D.one_cell_hubs("DHL") == (2, 3)
    assert D.one_cell_hubs("GLS") == (1, 1)
    with pytest.raises(AssertionError, match="no hubs for"):
        D.one_cell_hubs("Amazon")


# ── the discount scenario carries both lenses ──────────────────────────────
def test_discount_rows_carry_both_lenses(monkeypatch):
    """The flat-discount optimum is lens-specific, so both series must be there.

    Exercised on the real revision grid, because the compendium's ruling
    (§40.17) is about that grid: operator lens peaks at P = 0.25, routing lens
    at P = 0.5. A table showing only one of them cannot be read for the other.
    """
    monkeypatch.syspath_prepend(str(PRES))
    for mod in ("_data", "_revision"):
        sys.modules.pop(mod, None)
    import _data as RD
    if RD.SCHEMA != RD.SCHEMA_V2:
        pytest.skip(f"no v2 grid at {RD.REV}")
    import _revision as RV
    f = RV.Facts.load()
    rows = RV.discount_rows(f)
    assert all(len(r) == 6 for r in rows), "a lens column is missing"
    opt = RV.discount_optima(f)
    assert set(opt) == {"operator", "routing"}
    for lens, (P, net, runner, margin) in opt.items():
        assert P in (0.0, 0.25, 0.5, 0.75, 1.0), lens
        assert margin >= 0, f"{lens}: the runner-up beats the winner"
    # The line the slide prints must NAME a winner only where there is one.
    # On v6 the routing lens is a statistical tie (0.009 pp between P = 0.25
    # and P = 0.5), which is exactly the case the tie guard exists for.
    line = RV.discount_optimum_line(f)
    assert "lens-specific" in line
    for lens in ("operator", "routing"):
        P, net, runner, margin = opt[lens]
        if margin < RV.DISCOUNT_TIE_PP:
            assert f"level in the {lens} lens" in line, line
        else:
            assert f"P = {P:g} in the {lens} lens" in line, line
    # and the count the slides state is derived and correct
    assert (f.one_cell_hubs, f.dhl_hubs) == (8, 16)
    for mod in ("_data", "_revision"):
        sys.modules.pop(mod, None)


# ── the copy-only guard ────────────────────────────────────────────────────
@pytest.fixture
def guard(monkeypatch):
    monkeypatch.syspath_prepend(str(PRES))
    sys.modules.pop("_outguard", None)
    return importlib.import_module("_outguard")


def test_the_suffix_lands_on_the_stem_and_is_idempotent(guard):
    p = Path("x/EWGT_deck.pptx")
    once = guard.apply_suffix(p, "_rev2026-08")
    assert once.name == "EWGT_deck_rev2026-08.pptx"
    assert guard.apply_suffix(once, "_rev2026-08") == once


def test_writing_over_an_existing_deck_is_refused(guard, tmp_path):
    original = tmp_path / "EWGT_deck.pptx"
    original.write_bytes(b"the author's own file")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        guard.resolve(original)
    # and the file is untouched
    assert original.read_bytes() == b"the author's own file"


def test_the_suffix_is_what_makes_the_build_copy_only(guard, tmp_path):
    original = tmp_path / "EWGT_deck.pptx"
    original.write_bytes(b"the author's own file")
    target = guard.resolve(original, "_rev2026-08")
    assert target.name == "EWGT_deck_rev2026-08.pptx"
    assert not target.exists()
    assert original.read_bytes() == b"the author's own file"


def test_overwrite_is_the_only_way_past_the_guard(guard, tmp_path):
    original = tmp_path / "EWGT_deck.pptx"
    original.write_bytes(b"x")
    assert guard.resolve(original, overwrite=True) == original


# ══════════════════════════════════════════════════════════════════════════
# the schedule-choice dispatcher (Task 19-0)
#
# `load_chosen_stage3()` used to read `results/runs/path2_2026_05_29`
# unconditionally -- a pre-revision run whose stage 2 was frequency-
# PRESERVING. Every per-area frequency figure in the deck was therefore
# drawn from it while its caption named the revision grid. These tests pin
# the three things that must now hold: a v2 grid is served from its OWN
# table, a legacy directory still gets the legacy loader and its invariance
# check, and a v2 table missing a column fails loudly instead of falling
# back.
# ══════════════════════════════════════════════════════════════════════════
def _legacy_run(tmp_path):
    """A stand-in for the frozen 2026-05-29 run's two chosen tables."""
    run = tmp_path / "legacy_run"
    run.mkdir()
    rows2, rows3 = [], []
    for plz, idx, size, days in (("30159", 38, 6, "Mon,Tue,Wed,Thu,Fri,Sat"),
                                 ("30169", 0, 2, "Mon,Thu")):
        rows2.append(dict(penalty=0.0, share_willing=1.0, provider="DHL",
                          plz=plz, schedule_idx_init=idx,
                          schedule_size_init=size, weekdays_balanced=days,
                          weekly_parcels=1000.0))
        rows3.append(dict(penalty=0.0, share_willing=1.0, provider="DHL",
                          plz=plz, schedule_idx_system_smoothed=idx,
                          schedule_size_system_smoothed=size,
                          weekdays_system_smoothed=days))
    pd.DataFrame(rows2).to_csv(run / "tab_chosen_schedules.csv", index=False)
    pd.DataFrame(rows3).to_csv(
        run / "_tab_chosen_with_system_smoothing.csv", index=False)
    return run


def test_a_v2_grid_is_served_from_its_own_chosen_table(D, tmp_path,
                                                       monkeypatch):
    """The legacy run must not be touched at all on a v2 grid."""
    monkeypatch.setattr(D, "RUN", tmp_path / "does_not_exist")
    got = D.load_chosen_stage3()
    assert got.plan.unique().tolist() == [D.PLAN_BALANCED]
    # the operator plan puts the consolidated cell back on six days
    at = got[(got.penalty == 0.0) & (got.share_willing == 1.0)]
    assert set(at.schedule_size_system_smoothed) == {6}
    assert set(at.schedule_size_stage1) == {2, 6}
    assert set(at.weekdays_system_smoothed) == {"Mon,Tue,Wed,Thu,Fri,Sat"}
    # ... and the routing plan is a different picture from the same file
    r = D.load_chosen_stage3(D.PLAN_STAGE1)
    ar = r[(r.penalty == 0.0) & (r.share_willing == 1.0)]
    assert set(ar.schedule_size_system_smoothed) == {2, 6}
    assert r.plan.unique().tolist() == [D.PLAN_STAGE1]


def test_the_default_plan_is_the_operator_plan(D):
    """76_maps_v2.py draws the paper's maps on stage 2; so must the deck."""
    assert D.CHOSEN_PLAN_DEFAULT == D.PLAN_BALANCED
    assert D.grid_plan() == D.PLAN_BALANCED
    assert D.grid_plan(D.PLAN_OPERATOR) == D.PLAN_BALANCED
    assert D.grid_plan(D.PLAN_ROUTING) == D.PLAN_STAGE1
    assert D.grid_plan("stage1") == D.PLAN_STAGE1
    with pytest.raises(ValueError, match="unknown plan"):
        D.grid_plan("stage7")


def test_the_plan_reaches_the_provenance_stamp(D):
    """A figure has to be able to say which of the two plans it drew."""
    assert D.REV.name in D.plan_stamp()
    assert "stage 2" in D.plan_stamp(D.PLAN_BALANCED)
    assert "stage 1" in D.plan_stamp(D.PLAN_STAGE1)
    assert D.plan_stamp(D.PLAN_BALANCED) != D.plan_stamp(D.PLAN_STAGE1)


def test_a_legacy_grid_still_gets_the_legacy_loader(D, tmp_path, monkeypatch):
    legacy = tmp_path / "revision_legacy_chosen"
    legacy.mkdir()
    (legacy / "tab_costs_smoothed.csv").write_text(
        "penalty,share_willing,provider,dd_cost_stage3_eur,"
        "express_stage3_eur,total_stage3_eur\n0.0,1.0,DHL,1.0,0.0,1.0\n")
    monkeypatch.setattr(D, "RUN", _legacy_run(tmp_path))
    D.set_rev_dir(legacy)
    assert D.SCHEMA == D.SCHEMA_LEGACY
    got = D.load_chosen_stage3()
    assert "plan" not in got.columns          # legacy has only one
    assert set(got.schedule_size_system_smoothed) == {2, 6}
    # and it cannot be asked for a plan it does not have
    with pytest.raises(RuntimeError, match="carries ONE plan"):
        D.load_chosen_stage3(D.PLAN_STAGE1)


def test_the_legacy_frequency_invariance_check_still_bites(D, tmp_path,
                                                           monkeypatch):
    """The submission's stage 2 -> 3 preserved frequency; that is asserted.

    v6's stage 2 does not, which is why the assert had to move off the v2
    path rather than be deleted -- on a legacy grid it is still the thing
    that keeps a frequency map honest.
    """
    legacy = tmp_path / "revision_legacy_broken"
    legacy.mkdir()
    (legacy / "tab_costs_smoothed.csv").write_text("penalty\n0.0\n")
    run = _legacy_run(tmp_path)
    bad = pd.read_csv(run / "_tab_chosen_with_system_smoothing.csv",
                      dtype={"plz": str})
    bad["schedule_size_system_smoothed"] = 3      # no longer matches stage 2
    bad.to_csv(run / "_tab_chosen_with_system_smoothing.csv", index=False)
    monkeypatch.setattr(D, "RUN", run)
    D.set_rev_dir(legacy)
    with pytest.raises(AssertionError, match="frequency NOT preserved"):
        D.load_chosen_stage3()


def test_a_v2_chosen_table_missing_a_column_fails_loudly(D):
    """No silent fallback to the legacy run when the schema is short."""
    raw = pd.read_csv(D.REV / "_tab_chosen_v2.csv", dtype={"plz": str})
    raw.drop(columns=["schedule_idx_balanced"]).to_csv(
        D.REV / "_tab_chosen_v2.csv", index=False)
    D._CACHE.clear()
    with pytest.raises(KeyError, match="schedule_idx_balanced"):
        D.load_chosen_stage3()


def test_frequency_is_not_asserted_invariant_on_a_v2_grid(D, capsys):
    """v6's stage 2 is frequency-free; the loader reports it, never asserts."""
    got = D.load_chosen_stage3()
    out = capsys.readouterr().out
    assert "frequency-free stage 2" in out
    assert "40.14" in out
    # the synthetic grid moves one of two cells at (P=0, theta=1)
    moved = (got.schedule_size_stage1 != got.schedule_size_balanced)
    assert moved.any(), "the fixture no longer exercises a frequency change"


# ══════════════════════════════════════════════════════════════════════════
# the per-area table follows the grid
# ══════════════════════════════════════════════════════════════════════════
def _per_plz_v2(rev, plans=("stage1", "balanced")):
    rows = []
    for plan in plans:
        for plz, cost in (("30159", 80.0), ("30169", 90.0)):
            rows.append(dict(
                penalty=0.0, share_willing=1.0, provider="DHL", plz=plz,
                plan=plan, hub="depot", schedule_idx=38,
                cell_cost_eur=cost + (0.0 if plan == "stage1" else 5.0),
                cell_cost_baseline_eur=100.0, weekly_parcels=1000.0))
    out = rev.joinpath(*__import__("importlib").import_module(
        "_data").PER_PLZ_V2_REL)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def test_per_area_costs_come_from_the_grid_in_use(D):
    """Act 7 must read v6's per-area table, not the submission's."""
    with pytest.raises(FileNotFoundError,
                       match="00_recompute_per_plz_costs"):
        D.load_per_plz()
    _per_plz_v2(D.REV)
    D._CACHE.clear()
    a = D.load_per_plz(D.PLAN_BALANCED)
    b = D.load_per_plz(D.PLAN_STAGE1)
    assert a.plan.unique().tolist() == ["balanced"]
    assert b.plan.unique().tolist() == ["stage1"]
    # the legacy column names are aliased on, so Act 7 needs no edit
    assert a.dd_cost_stage3_eur.tolist() == [85.0, 95.0]
    assert b.dd_cost_stage3_eur.tolist() == [80.0, 90.0]
    assert set(a.dd_cost_baseline_eur) == {100.0}


def test_a_per_area_table_without_the_requested_plan_is_refused(D):
    _per_plz_v2(D.REV, plans=("stage1",))
    D._CACHE.clear()
    with pytest.raises(KeyError, match="balanced"):
        D.load_per_plz(D.PLAN_BALANCED)


# ══════════════════════════════════════════════════════════════════════════
# the validation must be finished before a figure is drawn from it
# ══════════════════════════════════════════════════════════════════════════
def _validation(tmp_path, items, planned=None, name="val"):
    """A v2 validation directory carrying exactly `items`."""
    val = tmp_path / name
    val.mkdir()
    rows = []
    for it in items:
        rows.append(dict(
            item=it, instance_id=f"i{it}", penalty=0.0,
            share_willing=0.0 if it == 0 else 1.0,
            plan="stage1" if it in (0, 2) else "balanced",
            provider="DHL", plz="30159", day=0, members="30159",
            instance_kind="delivery_single", predicted_cost_eur=100.0,
            vroom_cost_eur=90.0, vroom_n_routes=1, vroom_distance_km=10.0,
            vroom_n_parcels=100, vroom_status="OK", n_unassigned=0,
            jobs_removed=0))
    pd.DataFrame(rows).to_csv(val / "tab_vroom_v2.csv", index=False)
    if planned is not None:
        pd.DataFrame([dict(item=it, instance_id=f"q{it}_{k}")
                      for it, n in planned.items() for k in range(n)]
                     ).to_csv(val / "instance_queue.csv", index=False)
    return val


def test_the_validation_grid_name_is_v6(D):
    """The one line Part B moves; v6's validation finished 2026-08-28."""
    assert D.VAL_GRID_NAME == "revision_2026_08_v6"
    assert tuple(D.VAL_REQUIRED_ITEMS) == (0, 1, 2, 3)


def test_an_incomplete_validation_is_refused(D, tmp_path):
    """Items 0-3 or nothing: a half-written directory is not a validation."""
    D.set_val_dir(_validation(tmp_path, [1, 2, 3], name="half"))
    with pytest.raises(RuntimeError, match=r"INCOMPLETE.*missing \[0\]"):
        D.require_validation_items()
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        D.load_vroom_v2()


def test_a_complete_validation_passes_the_guard(D, tmp_path):
    D.set_val_dir(_validation(tmp_path, [0, 1, 2, 3], name="whole"))
    D.require_validation_items()
    assert D.validation_items() == {0, 1, 2, 3}
    assert len(D.load_vroom_v2()) == 4


def test_a_sampled_item_is_detected_from_the_queue(D, tmp_path):
    """Item 3 was cut to a sample by the G6 budget rule; no % may use it."""
    val = _validation(tmp_path, [0, 1, 2, 3],
                      planned={0: 1, 1: 1, 2: 1, 3: 4}, name="sampled")
    D.set_val_dir(val)
    assert D.validation_sampled_items() == {3}
    sv = D.load_savings_validation()
    assert set(sv[sv.sampled].item) == {3}
    assert not sv[sv.item == 0].saving_defined.any()
    assert sv[sv.item == 1].saving_defined.all()


def test_the_two_plans_of_a_validation_are_not_summed(D, tmp_path):
    """(P, theta) alone selects both plans; a per-cell figure needs one."""
    D.set_val_dir(_validation(tmp_path, [0, 1, 2, 3], name="plans"))
    both = D.load_vroom(theta=1.0)
    assert set(both.plan) == {"balanced", "stage1"}
    one = D.load_vroom(plan=D.PLAN_OPERATOR, theta=1.0)
    assert set(one.plan) == {"balanced"} and len(one) < len(both)
    with pytest.raises(AssertionError, match="no rows at theta"):
        D.load_vroom(theta=0.42)
