"""Per-cell plan costs (``scripts/revision/72_per_cell_costs_v2.py``).

The grid runner records schedules and totals, never what one postal-code area
costs.  ``72_`` rebuilds that column by splitting the three disjoint routing
terms -- own tours (``dd_cost_mx``), the pooled small-delivery groups and the
express residual pool -- and its whole claim to be usable is the identity

    sum over cells of cell_cost_eur == the grid's routing total for that plan

which the script asserts per (P, theta, provider, plan).  These tests build a
real ``build_cost_matrices_ml`` result over three co-located cells (one above
``MIN_TOUR_PARCELS`` = its own tour, two below = pooled together) at
``theta < 1`` (so every non-delivery day also carries an express pool), and
check that:

* all three cost components are actually exercised by the fixture -- a test
  that silently priced only own tours would prove nothing;
* the per-cell decomposition sums back to the cost path's own totals, and the
  vehicle attribution rebuilds ``_daily_fleet_per_hub`` exactly;
* pooled prices and pooled vehicles are split **parcel-proportionally**, and a
  single-member tour keeps 100 % of its own price;
* the identity gate FAILS LOUD on a reference total that does not match.

No Docker, no model pickle: ``_stubs.StubPredictor`` is linear in
``n_parcels`` and ``hub_dist_km``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "revision"))

from _stubs import cell_matrices  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location(
        "_per_cell_costs_v2", ROOT / "scripts" / "revision"
        / "72_per_cell_costs_v2.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()

#: one big cell (own tour) + two small co-located ones (pooled together)
SPECS = [(400.0, 100.0), (30.0, 10.0), (25.0, 8.0)]
#: theta < 1 so a non-delivery day carries an express residual
FAST_SHARE = 0.6


@pytest.fixture(scope="module")
def fixture():
    plz_keys, m, sch, hub_plz_list, plz_hub_arr = cell_matrices(
        SPECS, fs=FAST_SHARE, spread=0.0)
    three_day = next(i for i, s in enumerate(sch) if len(s) == 3)
    chosen = np.array([three_day] * len(SPECS), dtype=np.int64)
    return dict(plz_keys=plz_keys, m=m, sch=sch, hub_plz_list=hub_plz_list,
                plz_hub_arr=plz_hub_arr, chosen=chosen)


@pytest.fixture(scope="module")
def dec(fixture):
    return M.decompose_plan(fixture["chosen"], fixture["plz_hub_arr"],
                            fixture["hub_plz_list"], fixture["sch"],
                            fixture["m"])


# ─────────────────────────────────────────────────────────────────────────
def test_the_fixture_exercises_all_three_cost_terms(dec):
    """Own tour, pooled delivery group and express pool are all non-zero."""
    assert dec["dd_total"] > 0, "no own-tour cost -- fixture is degenerate"
    assert dec["pool_total"] > 0, "no pooled small-delivery cost"
    assert dec["express_total"] > 0, "no express residual cost"
    # and they land on the cells they should: the big cell owns a tour,
    # the two small ones only ever ride in a pool
    assert dec["own_cost"][0] > 0 and dec["pool_share"][0] == 0.0
    assert (dec["own_cost"][1:] == 0).all()
    assert (dec["pool_share"][1:] > 0).all()


def test_cell_costs_sum_to_the_routing_total(dec):
    """The identity the whole script exists for, on the synthetic grid."""
    assert dec["cell_cost"].sum() == pytest.approx(dec["routing_total"],
                                                   rel=1e-12)
    assert dec["routing_total"] == pytest.approx(
        dec["dd_total"] + dec["pool_total"] + dec["express_total"], rel=1e-12)


def test_components_sum_per_cell(dec):
    assert np.allclose(dec["cell_cost"],
                       dec["own_cost"] + dec["pool_share"]
                       + dec["express_share"])


def test_vehicle_attribution_sums_to_the_fleet(dec):
    """Gate 2 and 3: vehicle-days and hub peaks are attributed exactly."""
    assert dec["veh_days_share"].sum() == pytest.approx(dec["vehicle_days"],
                                                        rel=1e-12)
    assert dec["peak_veh_share"].sum() == pytest.approx(dec["sum_hub_peak"],
                                                        rel=1e-12)
    assert dec["vehicle_days"] == pytest.approx(float(dec["fleet"].sum()))
    assert dec["sum_hub_peak"] == pytest.approx(
        float(dec["fleet"].max(axis=1).sum()))


def test_express_price_is_split_parcel_proportionally(fixture, dec):
    """Cells 1 and 2 share the express tour in the ratio of their residuals.

    ``raw_express[z, d]`` does not depend on the schedule and is the same on
    every day of this fixture, so the week's aggregate share ratio is the
    per-tour ratio exactly.
    """
    m, ch = fixture["m"], fixture["chosen"]
    d_off = next(d for d in range(6) if not m["sched_active"][ch[0], d])
    rx = m["raw_express"][:, d_off]
    want = rx[1] / (rx[1] + rx[2])
    got = dec["express_share"][1] / (dec["express_share"][1]
                                     + dec["express_share"][2])
    assert got == pytest.approx(want, rel=1e-9)


def test_pooled_delivery_price_is_split_parcel_proportionally(fixture):
    """Same rule on the delivery pool, whose weight is ``combined_demand``.

    Checked on the DAILY plan: with one source day per delivery day the
    pooled demand is identical on all six days, so the week's aggregate
    ratio is again the per-tour ratio and the assertion needs no per-day
    bookkeeping of its own.
    """
    m, sch = fixture["m"], fixture["sch"]
    daily = next(i for i, s in enumerate(sch) if len(s) == 6)
    ch = np.array([daily] * len(SPECS), dtype=np.int64)
    d = M.decompose_plan(ch, fixture["plz_hub_arr"], fixture["hub_plz_list"],
                         sch, m)
    assert d["pool_total"] > 0 and d["express_total"] == 0.0, (
        "the daily plan should pool the two small cells and have no "
        "express residual at all")
    cd = m["combined_demand"]
    per_day = np.array([cd[z, daily, 0] for z in (1, 2)])
    want = per_day[0] / per_day.sum()
    got = d["pool_share"][1] / (d["pool_share"][1] + d["pool_share"][2])
    assert got == pytest.approx(want, rel=1e-9)
    # and the vehicles of the shared tour follow the same weights
    got_v = d["pool_vd_share"][1] / (d["pool_vd_share"][1]
                                     + d["pool_vd_share"][2])
    assert got_v == pytest.approx(want, rel=1e-9)


def test_a_single_member_tour_keeps_its_whole_price(fixture, dec):
    """The big cell rides alone in express, so it pays that tour in full."""
    m, ch = fixture["m"], fixture["chosen"]
    sa = m["sched_active"]
    off = [d for d in range(6) if not sa[ch[0], d]]
    want = float(sum(m["express_cost"][0, d] for d in off))
    assert dec["express_share"][0] == pytest.approx(want, rel=1e-12)


def test_shares_are_proportional_and_sum_to_one():
    w = np.array([0.0, 30.0, 10.0, 0.0])
    s = M._shares((1, 2), w)
    assert s.sum() == pytest.approx(1.0)
    assert s[0] / s[1] == pytest.approx(3.0)


def test_a_zero_parcel_tour_refuses_to_be_split():
    with pytest.raises(AssertionError, match="parcel-proportional share"):
        M._shares((0, 1), np.zeros(3))


# ─────────────────────────────────────────────────────────────────────────
# the identity gate itself
# ─────────────────────────────────────────────────────────────────────────
def _ref_row(dec, plan="balanced", **override) -> pd.Series:
    ref = M.PLAN_REF[plan]
    row = {ref["cost"]: dec["routing_total"], ref["vd"]: dec["vehicle_days"],
           ref["peak"]: dec["sum_hub_peak"]}
    row.update(override)
    return pd.Series(row)


@pytest.mark.parametrize("plan", ["stage1", "balanced"])
def test_the_identity_gate_passes_on_the_true_totals(dec, plan):
    d = M.check_identity(dec, _ref_row(dec, plan), plan, "synthetic")
    assert all(abs(v) < 1e-9 for v in d.values())


def test_the_identity_gate_fails_on_a_wrong_cost(dec):
    bad = _ref_row(dec, "balanced",
                   cost_stage2_eur=dec["routing_total"] + 1.0)
    with pytest.raises(AssertionError, match="IDENTITY GATE FAILED"):
        M.check_identity(dec, bad, "balanced", "synthetic")


def test_the_identity_gate_fails_on_a_wrong_hub_peak(dec):
    bad = _ref_row(dec, "balanced", sum_hub_peak=dec["sum_hub_peak"] + 1.0)
    with pytest.raises(AssertionError, match="sum_hub_peak"):
        M.check_identity(dec, bad, "balanced", "synthetic")


def test_the_identity_gate_refuses_a_nan_reference(dec):
    bad = _ref_row(dec, "balanced", vehicle_days=np.nan)
    with pytest.raises(AssertionError, match="nothing to gate"):
        M.check_identity(dec, bad, "balanced", "synthetic")


def test_a_tiny_relative_drift_is_inside_the_window(dec):
    """1e-9 relative is float re-association, not a bookkeeping defect."""
    row = _ref_row(dec, "balanced",
                   cost_stage2_eur=dec["routing_total"] * (1 + 1e-12))
    M.check_identity(dec, row, "balanced", "synthetic")


# ─────────────────────────────────────────────────────────────────────────
# plan/reference wiring
# ─────────────────────────────────────────────────────────────────────────
def test_plan_columns_and_references_are_the_v5_v6_schema():
    assert M.PLAN_COL == {"stage1": "schedule_idx_stage1",
                          "balanced": "schedule_idx_balanced"}
    assert M.PLAN_REF["stage1"]["cost"] == "cost_stage1_eur"
    assert M.PLAN_REF["balanced"]["cost"] == "cost_stage2_eur"
    # the stage-1 anchor is the *_before family, never the plain names
    assert all(v.endswith("_before") or "_before_" in v
               for k, v in M.PLAN_REF["stage1"].items() if k != "cost")


def test_row_assembly_carries_every_documented_column(fixture, dec):
    rows = M.plan_rows(0.0, 0.6, "DHL", "balanced", dec,
                       fixture["plz_keys"], fixture["chosen"],
                       ["hub_0"], np.array([1.0, 2.0, 3.0]),
                       np.array([len(s) for s in fixture["sch"]]),
                       np.zeros(len(fixture["sch"])), "none")
    assert len(rows) == len(SPECS)
    for col in ("penalty", "share_willing", "provider", "plz", "plan",
                "head_id", "hub", "schedule_idx", "own_cost_eur",
                "pool_share_eur", "express_share_eur", "cell_cost_eur",
                "cell_parcels_week", "mean_days", "wait_days",
                "veh_days_share", "peak_veh_share", "hub_peak_day"):
        assert col in rows[0], f"row lacks {col!r}"
    assert sum(r["cell_cost_eur"] for r in rows) == pytest.approx(
        dec["routing_total"], rel=1e-12)


# ─────────────────────────────────────────────────────────────────────────
# resume bookkeeping
# ─────────────────────────────────────────────────────────────────────────
def _block(th, prov, n):
    return pd.DataFrame([dict(share_willing=th, provider=prov, plz=f"{i:05d}",
                              cell_cost_eur=1.0) for i in range(n)])


def test_load_done_keeps_complete_blocks_and_drops_short_ones(tmp_path):
    p = tmp_path / "per_cell.csv"
    pd.concat([_block(1.0, "DHL", 4), _block(0.5, "GLS", 2)]).to_csv(
        p, index=False)
    done = M.load_done(p, {(1.0, "DHL"): 4, (0.5, "GLS"): 4})
    assert done == {(1.0, "DHL")}
    left = pd.read_csv(p)
    assert set(left.provider) == {"DHL"}, "the short block was not dropped"


def test_load_done_on_a_missing_file_is_empty(tmp_path):
    assert M.load_done(tmp_path / "nope.csv", {}) == set()


def test_append_rows_refuses_a_schema_change(tmp_path):
    p = tmp_path / "per_cell.csv"
    M.append_rows(p, [dict(a=1, b=2)])
    with pytest.raises(SystemExit, match="SCHEMA MISMATCH"):
        M.append_rows(p, [dict(a=1, c=3)])


# ─────────────────────────────────────────────────────────────────────────
# head resolution
# ─────────────────────────────────────────────────────────────────────────
class _Args:
    def __init__(self, **kw):
        self.head = kw.get("head")
        self.head_path = kw.get("head_path")
        self.edges_path = kw.get("edges_path")


def test_a_pre_task11_grid_is_only_priceable_head_free(tmp_path):
    costs = pd.DataFrame([dict(penalty=0.0, share_willing=0.0)])
    hs = M.resolve_head_args(tmp_path, costs, _Args(head="none"))
    assert hs == dict(mode="none", path=None, edges=None, expect_id="none")
    with pytest.raises(SystemExit, match="pre-Task-11"):
        M.resolve_head_args(tmp_path, costs, _Args(head="installed"))


def test_head_defaults_come_from_the_grids_own_manifest(tmp_path):
    (tmp_path / "head_manifest.json").write_text(
        '{"mode": "installed", "head_id": "bundle_head@abc+def", '
        '"path": "p/bundle_head.pkl", "edges_path": "p/bundles_bins.json"}',
        encoding="utf-8")
    costs = pd.DataFrame([dict(head_id="bundle_head@abc+def")])
    hs = M.resolve_head_args(tmp_path, costs, _Args())
    assert hs["mode"] == "installed"
    assert hs["expect_id"] == "bundle_head@abc+def"
    assert hs["path"].name == "bundle_head.pkl"
    assert hs["edges"].name == "bundles_bins.json"


def test_an_explicit_head_flag_that_contradicts_the_manifest_refuses(tmp_path):
    (tmp_path / "head_manifest.json").write_text(
        '{"mode": "installed", "head_id": "h@1+2", "path": "p.pkl", '
        '"edges_path": "e.json"}', encoding="utf-8")
    costs = pd.DataFrame([dict(head_id="h@1+2")])
    with pytest.raises(SystemExit, match="contradicts"):
        M.resolve_head_args(tmp_path, costs, _Args(head="none"))


def test_a_grid_mixing_two_head_ids_is_refused(tmp_path):
    costs = pd.DataFrame([dict(head_id="a@1+2"), dict(head_id="b@3+4")])
    with pytest.raises(AssertionError, match="mixes 2 head_id"):
        M.resolve_head_args(tmp_path, costs, _Args())


def test_a_head_priced_grid_without_a_manifest_refuses(tmp_path):
    costs = pd.DataFrame([dict(head_id="bundle_head@abc+def")])
    with pytest.raises(SystemExit, match="no head_manifest.json"):
        M.resolve_head_args(tmp_path, costs, _Args())


# ---------------------------------------------------------------------
# cross-check against the runner's own per-cell columns (634433f, v7 on)
# ---------------------------------------------------------------------
def _long(vals=(10.0, 2.0, 1.0)):
    """Two cells x two plans in this file's LONG layout."""
    own, pool, exp = vals
    rows = []
    for plan in ("stage1", "balanced"):
        k = 1.0 if plan == "stage1" else 1.5
        for plz in ("30159", "30161"):
            rows.append(dict(penalty=0.0, share_willing=1.0, provider="DHL",
                             plz=plz, plan=plan,
                             own_cost_eur=k * own, pool_share_eur=k * pool,
                             express_share_eur=k * exp,
                             cell_cost_eur=k * (own + pool + exp)))
    return pd.DataFrame(rows)


def _wide(vals=(10.0, 2.0, 1.0)):
    """The same numbers in the runner's WIDE layout."""
    own, pool, exp = vals
    rows = []
    for plz in ("30159", "30161"):
        r = dict(penalty=0.0, share_willing=1.0, provider="DHL", plz=plz)
        for suf, k in (("_stage1", 1.0), ("_stage2", 1.5)):
            r[f"own_cost_eur{suf}"] = k * own
            r[f"pool_share_eur{suf}"] = k * pool
            r[f"express_share_eur{suf}"] = k * exp
            r[f"cell_cost_eur{suf}"] = k * (own + pool + exp)
        rows.append(r)
    return pd.DataFrame(rows)


def test_the_wide_cross_check_passes_on_agreeing_files():
    msg = M.crosscheck_wide(_long(), _wide())
    assert msg.startswith("PASSED"), msg
    assert "16" in msg, "2 cells x 2 plans x 4 columns = 16 comparisons"


def test_the_wide_cross_check_maps_balanced_to_the_runners_stage2():
    """The runner calls the operator plan ``stage2``; this file calls it
    ``balanced``.  A mapping that got that backwards would compare the two
    plans against each other and has to fail."""
    assert M.WIDE_SUFFIX == {"stage1": "_stage1", "balanced": "_stage2"}
    swapped = _wide()
    for b in M.WIDE_COLS:
        swapped[f"{b}_stage1"], swapped[f"{b}_stage2"] = (
            swapped[f"{b}_stage2"].copy(), swapped[f"{b}_stage1"].copy())
    with pytest.raises(AssertionError, match="WIDE CROSS-CHECK FAILED"):
        M.crosscheck_wide(_long(), swapped)


def test_the_wide_cross_check_fails_on_a_single_drifted_cell():
    w = _wide()
    w.loc[0, "cell_cost_eur_stage2"] += 1.0
    with pytest.raises(AssertionError,
                       match="WIDE CROSS-CHECK FAILED plan=balanced"):
        M.crosscheck_wide(_long(), w)


def test_the_wide_cross_check_tolerates_float_reassociation():
    w = _wide()
    for b in M.WIDE_COLS:
        w[f"{b}_stage1"] = w[f"{b}_stage1"] * (1 + 1e-15)
    assert M.crosscheck_wide(_long(), w).startswith("PASSED")


def test_the_wide_cross_check_skips_loudly_on_a_pre_634433f_grid():
    """v5 and v6 were written before the runner produced those columns:
    say so, never pass silently."""
    bare = _wide()[["penalty", "share_willing", "provider", "plz"]]
    msg = M.crosscheck_wide(_long(), bare)
    assert msg.startswith("SKIPPED")
    assert "own_cost_eur_stage1" in msg and "634433f" in msg


def test_the_wide_cross_check_refuses_a_different_cell_universe():
    w = _wide()
    w = w[w.plz == "30159"]
    with pytest.raises(AssertionError, match="cell universe"):
        M.crosscheck_wide(_long(), w)


def test_the_identity_window_is_the_runners_own():
    """1e-6 relative would leave a 1.5 EUR blind spot at a routing total of
    1.5e6; the runner uses a 1e-3 floor with a 1e-9 relative term."""
    assert M.ABS_TOL == 1e-3
    assert M.REL_TOL == 1e-9
    assert M._tol(1.5e6) == pytest.approx(1.5e-3)
    assert M._tol(1.0) == pytest.approx(1e-3)
