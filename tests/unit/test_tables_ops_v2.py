"""``scripts/revision/73_tables_ops_v2.py`` -- v2 ops/knee/value-of-stage-2
tables (the v5/v6-schema replacement for 40_/41_).

Covers, with tiny synthetic frames (no checkpoints, no Docker, no model):

* ``structural_facts`` -- the penalty/theta-independent PLZ features (hub
  distance, area, parcels per drop-site, region type), including the
  "cell has no raumtyp row" path;
* ``compute_plz_knee_with_features`` -- the join picks EACH LENS'S OWN P*
  row (routing lens/stage1 plan vs operator lens/balanced plan) and never
  cross-mixes them; missing theta=1 coverage in the per-cell table fails
  loud with ``PerCellKneeInputIncomplete`` instead of approximating;
* ``_operator_lens`` -- the variable/peak/OpCost reconstruction formula;
* ``compute_op_kpi_per_day`` / ``compute_op_kpi_weekly`` -- PARTIAL rows are
  counted and never dropped from the totals; predicted-vs-actual saving %
  is computed against the provider's own item-0 baseline when present, and
  is NaN with ``baseline_available=False`` when item 0 is absent;
* ``_gate_validation_rows`` -- the Sigma-reconciliation gate actually gates;
* ``compute_value_of_stage2`` -- plan2-minus-plan1 deltas in both lenses,
  and the theta=0 no-op invariant it asserts.
"""
from __future__ import annotations

import importlib.util
import pickle
import py_compile
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "scripts" / "revision"


def _load():
    sys.path.insert(0, str(REV))
    spec = importlib.util.spec_from_file_location(
        "_tables_ops_v2", REV / "73_tables_ops_v2.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()
H = M.H  # the exact _figs_tables_v2 module instance 73_ imported and uses


# ═════════════════════════════════════════════════════════════════════════
# structural_facts
# ═════════════════════════════════════════════════════════════════════════
def _optimization_data():
    return {
        "A": {"plz_data": {
            "00001": {"b2c": {0: 10.0, 1: 20.0}, "b2b": {0: 5.0},
                      "hub_dist_km": 2.0, "area_km2": 3.0,
                      "n_stops_per_day": 5.0}}},
        "B": {"plz_data": {
            "00002": {"b2c": {0: 40.0}, "b2b": {0: 0.0},
                      "hub_dist_km": 9.0, "area_km2": 1.5,
                      "n_stops_per_day": 8.0},
            "00003": {"b2c": {0: 1.0}, "b2b": {0: 0.0},
                      "hub_dist_km": 4.0, "area_km2": 2.0,
                      "n_stops_per_day": 2.0}}},
    }


def _raumtyp():
    # 00003 deliberately absent -- exercises the "no raumtyp row" path.
    return pd.DataFrame({"plz": ["00001", "00002"],
                         "raumtyp_3": ["urban", "rural"]})


def test_structural_facts_reads_expected_fields():
    df = M.structural_facts(_optimization_data(), _raumtyp())
    assert set(zip(df.provider, df.plz)) == {
        ("A", "00001"), ("B", "00002"), ("B", "00003")}
    a = df[(df.provider == "A") & (df.plz == "00001")].iloc[0]
    assert a.hub_dist_km == pytest.approx(2.0)
    assert a.area_km2 == pytest.approx(3.0)
    assert a.weekly_parcels == pytest.approx(10.0 + 20.0 + 5.0)
    assert a.n_stops_per_day == pytest.approx(5.0)
    assert a.parcels_per_stop == pytest.approx(35.0 / (5.0 * H.N_DAYS))
    assert a.raumtyp_3 == "urban"


def test_structural_facts_missing_raumtyp_is_nan_not_a_crash(capsys):
    df = M.structural_facts(_optimization_data(), _raumtyp())
    row = df[(df.provider == "B") & (df.plz == "00003")].iloc[0]
    assert pd.isna(row.raumtyp_3)
    assert "WARN" in capsys.readouterr().out


# ═════════════════════════════════════════════════════════════════════════
# compute_plz_knee_with_features
# ═════════════════════════════════════════════════════════════════════════
PER_CELL_BASE_COLS = dict(
    hub="H", schedule_idx=1, own_cost_eur=0.0, pool_share_eur=0.0,
    express_share_eur=0.0, cell_parcels_week=100.0, mean_days=6.0,
    wait_days=0.0, veh_days_share=1.0, peak_veh_share=1.0, hub_peak_day=0)


def _per_cell_row(provider, plz, penalty, share_willing, plan, cell_cost_eur):
    row = dict(provider=provider, plz=plz, penalty=penalty,
              share_willing=share_willing, plan=plan,
              cell_cost_eur=cell_cost_eur)
    row.update(PER_CELL_BASE_COLS)
    return row


def _per_cell_frame(theta1_rows: bool = True) -> pd.DataFrame:
    rows = [
        # baselines (theta=0, both plans, both cells) -- must be IDENTICAL
        # per cell across every P and plan (H.per_cell_savings asserts it).
        _per_cell_row("A", "00001", 0.0, 0.0, "stage1", 100.0),
        _per_cell_row("A", "00001", 0.0, 0.0, "balanced", 100.0),
        _per_cell_row("B", "00002", 0.0, 0.0, "stage1", 200.0),
        _per_cell_row("B", "00002", 0.0, 0.0, "balanced", 200.0),
    ]
    if theta1_rows:
        rows += [
            # provider A: P_star_routing=0.25, P_star_operator=0.5
            _per_cell_row("A", "00001", 0.25, 1.0, "stage1", 80.0),   # -> 20% (WANTED)
            _per_cell_row("A", "00001", 0.75, 1.0, "stage1", 50.0),   # distractor, must NOT be picked
            _per_cell_row("A", "00001", 0.5, 1.0, "balanced", 70.0),  # -> 30% (WANTED)
            _per_cell_row("A", "00001", 0.25, 1.0, "balanced", 90.0),  # distractor
            # provider B: P_star_routing=0.5, P_star_operator=0.25
            _per_cell_row("B", "00002", 0.5, 1.0, "stage1", 150.0),   # -> 25% (WANTED)
            _per_cell_row("B", "00002", 0.25, 1.0, "stage1", 190.0),  # distractor
            _per_cell_row("B", "00002", 0.25, 1.0, "balanced", 120.0),  # -> 40% (WANTED)
            _per_cell_row("B", "00002", 0.5, 1.0, "balanced", 180.0),  # distractor
        ]
    return pd.DataFrame(rows)


def _pstar_frame() -> pd.DataFrame:
    return pd.DataFrame([
        dict(provider="A", P_star_routing=0.25, carrier_class_routing="Service-bound",
            P_star_operator=0.5, carrier_class_operator="Hybrid"),
        dict(provider="B", P_star_routing=0.5, carrier_class_routing="Hybrid",
            P_star_operator=0.25, carrier_class_operator="Service-bound"),
    ])


def _structural_frame() -> pd.DataFrame:
    return M.structural_facts(
        {"A": {"plz_data": {"00001": {"b2c": {0: 500.0}, "b2b": {0: 0.0},
                                      "hub_dist_km": 2.0, "area_km2": 3.0,
                                      "n_stops_per_day": 5.0}}},
         "B": {"plz_data": {"00002": {"b2c": {0: 900.0}, "b2b": {0: 0.0},
                                      "hub_dist_km": 9.0, "area_km2": 1.5,
                                      "n_stops_per_day": 8.0}}}},
        pd.DataFrame({"plz": ["00001", "00002"],
                     "raumtyp_3": ["urban", "rural"]}))


def test_knee_join_picks_each_lenss_own_pstar_row_never_the_other_lens():
    structural = _structural_frame()
    per_cell = _per_cell_frame()
    pstar = _pstar_frame()
    out = M.compute_plz_knee_with_features(structural, per_cell, pstar)
    assert len(out) == 2

    a = out[out.provider == "A"].iloc[0]
    assert a.saving_pct_routing == pytest.approx(20.0)
    assert a.saving_pct_operator == pytest.approx(30.0)
    assert a.cell_cost_eur_routing == pytest.approx(80.0)
    assert a.cell_cost_eur_operator == pytest.approx(70.0)

    b = out[out.provider == "B"].iloc[0]
    assert b.saving_pct_routing == pytest.approx(25.0)
    assert b.saving_pct_operator == pytest.approx(40.0)

    # structural features rode along untouched
    assert a.hub_dist_km == pytest.approx(2.0)
    assert a.raumtyp_3 == "urban"


def test_knee_join_fails_loud_when_theta1_not_yet_covered():
    structural = _structural_frame()
    per_cell = _per_cell_frame(theta1_rows=False)  # only theta=0 rows exist
    pstar = _pstar_frame()
    with pytest.raises(M.PerCellKneeInputIncomplete, match="theta=1"):
        M.compute_plz_knee_with_features(structural, per_cell, pstar)


# ═════════════════════════════════════════════════════════════════════════
# _operator_lens
# ═════════════════════════════════════════════════════════════════════════
def test_operator_lens_formula():
    df = pd.DataFrame([
        dict(hub_name="H1", day=0, cost=1000.0, routes=3.0),
        dict(hub_name="H1", day=1, cost=1500.0, routes=5.0),
    ])
    out = M._operator_lens(df, "cost", "routes")
    expected_variable = (1000.0 - H.FIXED_COST_EUR * 3.0) + (1500.0 - H.FIXED_COST_EUR * 5.0)
    expected_peak = 5.0  # hub H1's own peak day (day 1)
    expected_opcost = expected_variable + H.WEEK_FIXED_COST_EUR * expected_peak
    assert out["variable_eur"] == pytest.approx(expected_variable)
    assert out["sum_hub_peak"] == pytest.approx(expected_peak)
    assert out["opcost_eur"] == pytest.approx(expected_opcost)
    assert out["routing_eur"] == pytest.approx(2500.0)
    assert out["vehicle_days"] == pytest.approx(8.0)


def test_operator_lens_empty_returns_zeros_not_a_crash():
    out = M._operator_lens(pd.DataFrame(columns=["hub_name", "day", "cost", "routes"]),
                           "cost", "routes")
    assert out == dict(variable_eur=0.0, sum_hub_peak=0.0, opcost_eur=0.0,
                       routing_eur=0.0, vehicle_days=0.0, n=0)


def test_operator_lens_missing_cost_still_counts_toward_peak():
    """An unpriced (NaN cost) row still needs its van for the peak."""
    df = pd.DataFrame([
        dict(hub_name="H1", day=0, cost=np.nan, routes=9.0),
        dict(hub_name="H1", day=1, cost=100.0, routes=1.0),
    ])
    out = M._operator_lens(df, "cost", "routes")
    assert out["sum_hub_peak"] == pytest.approx(9.0)
    assert out["routing_eur"] == pytest.approx(100.0)  # NaN excluded from cost


# ═════════════════════════════════════════════════════════════════════════
# compute_op_kpi_per_day / compute_op_kpi_weekly
# ═════════════════════════════════════════════════════════════════════════
def _vroom_row(item, penalty, share_willing, plan, provider, day, status,
              pred_cost, pred_routes, act_cost, act_routes, km, hours, parcels,
              hub_name="H1"):
    return dict(item=item, penalty=penalty, share_willing=share_willing,
               plan=plan, provider=provider, day=day, hub_name=hub_name,
               vroom_status=status,
               predicted_cost_eur=pred_cost, predicted_n_routes=pred_routes,
               vroom_cost_eur=act_cost, vroom_n_routes=act_routes,
               vroom_distance_km=km, vroom_duration_h=hours,
               vroom_n_parcels=parcels)


def _vroom_frame_with_baseline() -> pd.DataFrame:
    rows = [
        # item 0: theta=0 daily baseline, provider A, plan is always
        # "stage1" per 67_'s ITEMS dict (theta=0 is a stage-1/2 no-op).
        _vroom_row(0, 0.0, 0.0, "stage1", "A", 0, "OK", 500, 3, 520, 3, 100, 5, 1000),
        _vroom_row(0, 0.0, 0.0, "stage1", "A", 1, "CACHED", 600, 4, 590, 4, 120, 6, 1100),
        # item 1: (P=0.25, theta=1, balanced), provider A -- one PARTIAL row
        _vroom_row(1, 0.25, 1.0, "balanced", "A", 0, "OK", 300, 2, 310, 2, 60, 3.0, 600),
        _vroom_row(1, 0.25, 1.0, "balanced", "A", 1, "CACHED", 350, 3, 340, 3, 70, 3.5, 650),
        _vroom_row(1, 0.25, 1.0, "balanced", "A", 2, "PARTIAL", 100, 1, 90, 1, 20, 1.0, 150),
    ]
    return pd.DataFrame(rows)


def test_per_day_partial_rows_counted_and_included_in_totals():
    vroom = _vroom_frame_with_baseline()
    day = M.compute_op_kpi_per_day(vroom)
    d2 = day[(day.item == 1) & (day.day == 2)].iloc[0]
    assert d2.n_partial == 1
    assert d2.n_instances == 1
    # PARTIAL's own cost/routes/parcels are still in the totals, not dropped
    assert d2.routing_cost_actual_eur == pytest.approx(90.0)
    assert d2.parcels == pytest.approx(150.0)
    assert d2.n_routes == pytest.approx(1.0)

    week1a = M.compute_op_kpi_weekly(vroom)
    row = week1a[(week1a.item == 1) & (week1a.provider == "A")].iloc[0]
    assert row.n_instances == 3
    assert row.n_partial == 1
    assert row.n_clean == 2
    # "_all" NEVER drops the PARTIAL row's cost/routes
    assert row.routing_cost_actual_all_eur == pytest.approx(310 + 340 + 90)
    assert row.routing_cost_actual_clean_eur == pytest.approx(310 + 340)
    assert row.parcels_week == pytest.approx(600 + 650 + 150)


def test_weekly_savings_vs_baseline_when_item0_present():
    vroom = _vroom_frame_with_baseline()
    week = M.compute_op_kpi_weekly(vroom)
    row = week[(week.item == 1) & (week.provider == "A")].iloc[0]
    assert row.baseline_available == True  # noqa: E712

    base_pred_routing = 500 + 600
    base_act_routing = 520 + 590
    point_pred_routing = 300 + 350 + 100
    point_act_routing_all = 310 + 340 + 90
    point_act_routing_clean = 310 + 340

    expect_pred = (base_pred_routing - point_pred_routing) / base_pred_routing * 100
    expect_act_all = (base_act_routing - point_act_routing_all) / base_act_routing * 100
    expect_act_clean = (base_act_routing - point_act_routing_clean) / base_act_routing * 100

    assert row.pred_routing_save_pct == pytest.approx(expect_pred)
    assert row.act_routing_save_pct_all == pytest.approx(expect_act_all)
    assert row.act_routing_save_pct_clean == pytest.approx(expect_act_clean)


def test_weekly_savings_are_nan_when_item0_is_absent():
    vroom = _vroom_frame_with_baseline()
    no_baseline = vroom[vroom.item != 0].reset_index(drop=True)
    week = M.compute_op_kpi_weekly(no_baseline)
    row = week[(week.item == 1) & (week.provider == "A")].iloc[0]
    assert row.baseline_available == False  # noqa: E712
    for col in ("pred_routing_save_pct", "act_routing_save_pct_all",
               "act_routing_save_pct_clean", "pred_opcost_save_pct",
               "act_opcost_save_pct_all", "act_opcost_save_pct_clean"):
        assert pd.isna(row[col]), f"{col} should be NaN without an item-0 baseline"
    # the rest of the table must still be produced -- item 0 is optional,
    # never a hard requirement for the whole table.
    assert row.routing_cost_actual_all_eur == pytest.approx(310 + 340 + 90)


def test_validation_rows_gate_passes_on_correct_tables():
    vroom = _vroom_frame_with_baseline()
    day = M.compute_op_kpi_per_day(vroom)
    week = M.compute_op_kpi_weekly(vroom)
    M._gate_validation_rows(vroom, day, week)  # must not raise


def test_validation_rows_gate_fails_loud_on_a_dropped_row():
    vroom = _vroom_frame_with_baseline()
    day = M.compute_op_kpi_per_day(vroom)
    week = M.compute_op_kpi_weekly(vroom)
    corrupted_day = day.iloc[:-1].copy()  # silently drop one day's row
    with pytest.raises(AssertionError):
        M._gate_validation_rows(vroom, corrupted_day, week)


# ═════════════════════════════════════════════════════════════════════════
# compute_value_of_stage2
# ═════════════════════════════════════════════════════════════════════════
PROVS2 = ["A", "B"]
CELLS2 = [(0.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.5, 1.0)]


def _tiny_costs(theta0_noop_break: bool = False) -> pd.DataFrame:
    W = H.WEEK_FIXED_COST_EUR
    rows = []
    for P, th in CELLS2:
        for prov in PROVS2:
            if th == 0.0:
                r1 = 1000.0
                r2 = 999.0 if (theta0_noop_break and P == 0.5) else 1000.0
                v1 = v2 = 400.0
                p1 = p2 = 10.0
                vd1 = vd2 = 60.0
                pen1 = pen2 = 0.0
            else:
                r1, r2 = 800.0, 850.0
                v1, v2 = 380.0, 395.0
                p1, p2 = 14.0, 7.0
                vd1, vd2 = 50.0, 52.0
                pen1, pen2 = 5.0 * P, 9.0 * P
            rows.append(dict(
                penalty=P, share_willing=th, provider=prov,
                cost_stage1_eur=r1, cost_stage2_eur=r2,
                variable_before_eur=v1, variable_cost_eur=v2,
                sum_hub_peak_before=p1, sum_hub_peak=p2,
                vehicle_days_before=vd1, vehicle_days=vd2,
                penalty_before_eur=pen1, penalty_eur=pen2,
                operator_cost_before_eur=v1 + W * p1,
                operator_cost_eur=v2 + W * p2))
    return pd.DataFrame(rows)


def _tiny_wait() -> pd.DataFrame:
    rows = []
    for P, th in CELLS2:
        for prov in PROVS2:
            rows.append(dict(
                penalty=P, share_willing=th, provider=prov,
                total_parcels=1000.0, willing_parcels=1000.0 * th,
                wait_num_willing_stage1=300.0 * th,
                wait_num_all_stage1=300.0 * th,
                wait_num_willing=200.0 * th,
                wait_num_all=200.0 * th,
                mean_days_stage1=6.0 - 3.0 * th,
                mean_days=6.0 - 2.0 * th))
    return pd.DataFrame(rows)


def test_value_of_stage2_deltas():
    out = M.compute_value_of_stage2(_tiny_costs(), _tiny_wait())
    row = out[np.isclose(out.penalty, 0.5) & np.isclose(out.share_willing, 1.0)].iloc[0]
    n = len(PROVS2)  # routing/operator/peak are SUMMED across providers
    assert row.delta_routing_eur == pytest.approx(n * (850.0 - 800.0))
    assert row.delta_peak == pytest.approx(n * (7.0 - 14.0))
    # wait_d and mean_days are ratios/means, so identical providers leave
    # them unchanged regardless of provider count.
    assert row.delta_wait_d == pytest.approx(0.2 - 0.3)
    assert row.delta_mean_days == pytest.approx(4.0 - 3.0)
    W = H.WEEK_FIXED_COST_EUR
    expected_op1 = 380.0 + W * 14.0
    expected_op2 = 395.0 + W * 7.0
    assert row.delta_operator_eur == pytest.approx(n * (expected_op2 - expected_op1))

    theta0 = out[np.isclose(out.share_willing, 0.0)]
    assert np.allclose(theta0.delta_routing_eur, 0.0)
    assert np.allclose(theta0.delta_operator_eur, 0.0)
    assert np.allclose(theta0.delta_peak, 0.0)


def test_value_of_stage2_theta0_noop_violation_raises():
    with pytest.raises(AssertionError, match="no-op"):
        M.compute_value_of_stage2(_tiny_costs(theta0_noop_break=True), _tiny_wait())


# ═════════════════════════════════════════════════════════════════════════
# The six old-schema scripts: import-time DeprecationWarning + one-line
# docstring note. py_compile only -- these scripts are not safe to execute
# in a unit test (module-level checkpoint/Docker/filesystem side effects).
#
# 30_/31_/32_ are NOT in this list: since 74_v2_to_legacy_tables.py they are
# the FROZEN builders of the three ACCEPTED paper figures, not stale entry
# points -- see FROZEN_BUILDER_SCRIPTS below (Task 13B review, I3).
# ═════════════════════════════════════════════════════════════════════════
DEPRECATED_SCRIPTS = [
    "scripts/revision/20_validate_vroom_smoothed.py",
    "scripts/revision/21_pstar_knees_smoothed.py",
    "scripts/revision/40_tables_smoothed.py",
    "scripts/revision/41_op_kpi_tables_smoothed.py",
    "scripts/revision/50_recompute_fleet_wait_fixed.py",
    "scripts/pipeline/04_validate_vroom.py",
]


@pytest.mark.parametrize("relpath", DEPRECATED_SCRIPTS)
def test_deprecated_script_py_compiles(relpath, tmp_path):
    py_compile.compile(str(ROOT / relpath), cfile=str(tmp_path / "out.pyc"),
                       doraise=True)


@pytest.mark.parametrize("relpath", DEPRECATED_SCRIPTS)
def test_deprecated_script_warns_with_expected_wording(relpath):
    src = (ROOT / relpath).read_text(encoding="utf-8")
    assert "DeprecationWarning" in src
    assert "STALE entry point" in src
    assert "NOT comparable with the 2026-08" in src
    for successor in ("61_grid_run_v2.py", "67_validate_vroom_v2.py",
                      "70_figs_tables_v2.py", "73_tables_ops_v2.py"):
        assert successor in src, f"{relpath}: warning text lacks {successor}"
    docstring = src.split('"""', 2)[1]
    assert "DEPRECATED (2026-08 revision)" in docstring, (
        f"{relpath}: module docstring lacks the one-line deprecation note")


@pytest.mark.parametrize("relpath", DEPRECATED_SCRIPTS)
def test_deprecated_script_warning_fires_before_any_filterwarnings(relpath):
    """The warning must be emitted before the script's own
    warnings.filterwarnings('ignore') call (if it has one), or a
    DeprecationWarning would be silently swallowed on real use."""
    src = (ROOT / relpath).read_text(encoding="utf-8")
    warn_idx = src.index("_deprecation_warnings.warn(")
    fw_idx = src.find("warnings.filterwarnings(")
    if fw_idx != -1:
        assert warn_idx < fw_idx, (
            f"{relpath}: deprecation warning appears AFTER "
            "warnings.filterwarnings('ignore') -- it would be suppressed")


# ═════════════════════════════════════════════════════════════════════════
# 30_/31_/32_: the three FROZEN accepted-paper-figure builders. Since
# 74_v2_to_legacy_tables.py adapts a v5/v6 grid to the schema they read,
# they are no longer stale entry points -- they build the submitted Fig.
# 4/5/6 layouts on the new numbers, and their plotting code must not change.
# The banner must say so and must not claim they are deprecated/stale
# (Task 13B review, I3). py_compile only, same reason as above.
# ═════════════════════════════════════════════════════════════════════════
FROZEN_BUILDER_SCRIPTS = [
    "scripts/revision/30_fig5_heatmap_smoothed.py",
    "scripts/revision/31_fig6_structural_smoothed.py",
    "scripts/revision/32_fig4_mix.py",
]


@pytest.mark.parametrize("relpath", FROZEN_BUILDER_SCRIPTS)
def test_frozen_builder_script_py_compiles(relpath, tmp_path):
    py_compile.compile(str(ROOT / relpath), cfile=str(tmp_path / "out.pyc"),
                       doraise=True)


@pytest.mark.parametrize("relpath", FROZEN_BUILDER_SCRIPTS)
def test_frozen_builder_script_warns_with_expected_wording(relpath):
    src = (ROOT / relpath).read_text(encoding="utf-8")
    assert "DeprecationWarning" not in src, (
        f"{relpath}: still raises DeprecationWarning -- it is FROZEN, not "
        "deprecated, since it builds an accepted paper figure")
    assert "STALE entry point" not in src
    assert "FROZEN" in src
    assert "accepted paper Fig." in src
    assert "74_v2_to_legacy_tables.py" in src
    for token in ("REV_DIR", "REV_RUN_DIR", "REV_BASE_TOTAL",
                  "REV_BASELINE_CV"):
        assert token in src, f"{relpath}: banner text lacks {token}"
    docstring = src.split('"""', 2)[1]
    assert "FROZEN (2026-08 revision)" in docstring, (
        f"{relpath}: module docstring lacks the one-line FROZEN note")


@pytest.mark.parametrize("relpath", FROZEN_BUILDER_SCRIPTS)
def test_frozen_builder_warning_fires_before_any_filterwarnings(relpath):
    """Same hazard as the deprecated scripts: a notice raised after
    warnings.filterwarnings('ignore') is silently swallowed."""
    src = (ROOT / relpath).read_text(encoding="utf-8")
    warn_idx = src.index("_frozen_notice.warn(")
    fw_idx = src.find("warnings.filterwarnings(")
    if fw_idx != -1:
        assert warn_idx < fw_idx, (
            f"{relpath}: FROZEN notice appears AFTER "
            "warnings.filterwarnings('ignore') -- it would be suppressed")


# ═════════════════════════════════════════════════════════════════════════
# 30_/31_/32_: in-figure LABEL STRINGS follow the revised pipeline (2026-08
# revision, Task 13D). The paper is accepted, so geometry/colour/size/panel
# order of these frozen builders must not move -- that is proven separately
# by an image diff of the rendered figures, not by these tests. What these
# tests pin is the TEXT: the revised captions in
# paper/EWGT_2026_rev1/tbc_preprint_main.tex call the unit of analysis
# "cells" (not "postal-code areas"), name the two plans "routing-optimal
# plan" / "operator plan" (not "before/after fleet balancing" language left
# over from the submitted two-stage design), and state the per-LSP knee is
# the "routing-lens" P* (not "operator-optimal"). Each test asserts the new
# string is present AND the old one is gone, so a partial/reverted edit
# fails loud either way.
# ═════════════════════════════════════════════════════════════════════════

def test_fig4_y_axis_label_says_cells_not_postal_code_areas():
    src = (ROOT / "scripts/revision/32_fig4_mix.py").read_text(
        encoding="utf-8")
    assert "Share of cells [%]" in src
    assert "Share of postal-code areas [%]" not in src


def test_fig5_panel_subtitles_name_the_revised_plans():
    src = (ROOT / "scripts/revision/30_fig5_heatmap_smoothed.py").read_text(
        encoding="utf-8")
    assert "(routing-optimal plan)" in src
    assert "(operator plan)" in src
    assert "(before fleet balancing)" not in src
    assert "(after per-hub balancing and system smoothing)" not in src


def test_fig5_panel_e_title_uses_en_dash_not_double_hyphen():
    """Only the literal in-figure 'Mo--Sa' double hyphen is in scope (brief
    Task 13D); the console-log 'Mo-Sa' (single hyphen) prints elsewhere in
    this file and is deliberately left alone."""
    src = (ROOT / "scripts/revision/30_fig5_heatmap_smoothed.py").read_text(
        encoding="utf-8")
    assert "Mo–Sa coefficient of variation reduction" in src
    assert "Mo--Sa" not in src


def test_fig6_panel_titles_say_routing_lens_not_operator_optimal():
    src = (ROOT / "scripts/revision/31_fig6_structural_smoothed.py").read_text(
        encoding="utf-8")
    assert src.count(r"routing-lens $P^\star$") == 3
    assert r"operator-optimal $P^\star$" not in src
