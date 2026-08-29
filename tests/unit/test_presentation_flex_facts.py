"""The flexibility deck's new fact loaders, on synthetic frames.

`scripts/presentation/_revision.py` gained three pure helpers when the
flexibility deck (`98_deck_v6_flexibility.py`) was written, and all three are
the kind of arithmetic that fails silently:

* `clean_vroom()` decides which solved instances a realised saving may be
  formed from. Include a partial solve and the numerator prices a different
  problem than the denominator, and nothing in the output says so.
* `lens_totals()` rebuilds the operator euro from a validation table --
  variable cost plus six days of fixed cost per vehicle a hub must keep. Sum
  the routes instead of taking each hub's maximum and the answer is still a
  plausible number, just the wrong one.
* `Flex.delayed()` turns "mean wait over all parcels" into "mean wait of a
  parcel that actually waited". Divide by the wrong quantity and the slide
  understates what a customer is asked for.

No file in `results/` is read: every frame here is built in the test, so the
real grid cannot satisfy these and a broken helper cannot hide behind it.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
PRES = ROOT / "scripts" / "presentation"

FIXED = 189.15
WEEK_FIXED = 6 * FIXED


@pytest.fixture(scope="module")
def RV():
    if str(PRES) not in sys.path:
        sys.path.insert(0, str(PRES))
    return importlib.import_module("_revision")


# ── clean_vroom ────────────────────────────────────────────────────────────
def _vroom_rows():
    """Four solves: two usable, two that must never enter a saving."""
    return pd.DataFrame([
        dict(hub_name="H1", day=0, vroom_status="OK", n_unassigned=0,
             jobs_removed=0, predicted_cost_eur=100.0, vroom_cost_eur=90.0,
             predicted_n_routes=1, vroom_n_routes=1),
        dict(hub_name="H1", day=1, vroom_status="CACHED", n_unassigned=0,
             jobs_removed=0, predicted_cost_eur=200.0, vroom_cost_eur=180.0,
             predicted_n_routes=2, vroom_n_routes=2),
        # a partial solve: one job never got assigned to a vehicle
        dict(hub_name="H2", day=0, vroom_status="OK", n_unassigned=1,
             jobs_removed=0, predicted_cost_eur=999.0, vroom_cost_eur=1.0,
             predicted_n_routes=9, vroom_n_routes=9),
        # jobs dropped before solving: a smaller problem than the one priced
        dict(hub_name="H2", day=1, vroom_status="OK", n_unassigned=0,
             jobs_removed=3, predicted_cost_eur=999.0, vroom_cost_eur=1.0,
             predicted_n_routes=9, vroom_n_routes=9),
    ])


def test_clean_vroom_drops_partial_and_trimmed_solves(RV):
    clean = RV.clean_vroom(_vroom_rows())
    assert len(clean) == 2
    assert set(clean.hub_name) == {"H1"}


def test_clean_vroom_keeps_cache_hits(RV):
    """A cached solve is a solve: excluding it would silently halve the basis."""
    clean = RV.clean_vroom(_vroom_rows())
    assert "CACHED" in set(clean.vroom_status)


def test_clean_vroom_rejects_an_unknown_status(RV):
    df = _vroom_rows()
    df.loc[0, "vroom_status"] = "PARTIAL"
    assert len(RV.clean_vroom(df)) == 1


# ── lens_totals ────────────────────────────────────────────────────────────
def _two_hub_week():
    """Two hubs over three days, with a deliberately uneven weekly profile.

    H1 runs 1, 5, 2 vehicles; H2 runs 4, 4, 4. The peak fleet is 5 + 4 = 9,
    NOT the 20 route-days, and not the 5 of the busiest system day.
    """
    rows = []
    for hub, routes in (("H1", [1, 5, 2]), ("H2", [4, 4, 4])):
        for day, n in enumerate(routes):
            rows.append(dict(hub_name=hub, day=day,
                             vroom_n_routes=n,
                             # 10 EUR of variable cost per route on top of the
                             # per-vehicle-day fixed cost
                             vroom_cost_eur=n * (FIXED + 10.0)))
    return pd.DataFrame(rows)


def test_lens_totals_peak_is_the_sum_of_per_hub_maxima(RV):
    t = RV.lens_totals(_two_hub_week(), "vroom_", fixed_eur=FIXED)
    assert t["peak"] == 9.0            # 5 + 4, not 20 and not 9 route-days
    assert t["routes"] == 20.0


def test_lens_totals_splits_variable_from_fixed(RV):
    t = RV.lens_totals(_two_hub_week(), "vroom_", fixed_eur=FIXED)
    assert t["variable"] == pytest.approx(200.0)          # 20 routes x 10 EUR
    assert t["routing"] == pytest.approx(20 * (FIXED + 10.0))


def test_lens_totals_operator_euro_bills_the_peak_for_a_full_week(RV):
    t = RV.lens_totals(_two_hub_week(), "vroom_", fixed_eur=FIXED)
    assert t["operator"] == pytest.approx(200.0 + 9 * WEEK_FIXED)


def test_lens_totals_operator_exceeds_routing_when_the_week_is_peaky(RV):
    """The whole point of the operator lens: an uneven week costs more."""
    t = RV.lens_totals(_two_hub_week(), "vroom_", fixed_eur=FIXED)
    assert t["operator"] > t["routing"]


def test_lens_totals_reads_the_prefix_it_is_given(RV):
    df = _two_hub_week().rename(
        columns={"vroom_n_routes": "predicted_n_routes",
                 "vroom_cost_eur": "predicted_cost_eur"})
    t = RV.lens_totals(df, "predicted_", fixed_eur=FIXED)
    assert t["peak"] == 9.0


# ── Flex.delayed ───────────────────────────────────────────────────────────
class _StubFacts:
    """Just the two dictionaries `Flex.delayed()` reaches into."""

    def __init__(self, delayed, wait, days_plan1, days_plan2):
        self.discount = {0.25: dict(delayed=delayed)}
        self.headline = {0.25: dict(wait2=wait, days2=days_plan2,
                                    days1=days_plan1)}


def test_delayed_share_is_against_all_parcels(RV):
    flex = RV.Flex(total_parcels=1000)
    d = flex.delayed(_StubFacts(250.0, 0.50, 3.0, 3.5), 0.25)
    assert d["share_pct"] == pytest.approx(25.0)


def test_wait_of_a_delayed_parcel_exceeds_the_mean_over_all_parcels(RV):
    """0.5 days averaged over everyone is 2 days for the quarter who waited."""
    flex = RV.Flex(total_parcels=1000)
    d = flex.delayed(_StubFacts(250.0, 0.50, 3.0, 3.5), 0.25)
    assert d["wait_all"] == pytest.approx(0.50)
    assert d["wait_delayed"] == pytest.approx(2.0)
    assert d["wait_delayed"] > d["wait_all"]


def test_delayed_survives_a_point_where_nobody_waits(RV):
    """At a punitive penalty no parcel is held; the ratio must not divide by 0."""
    flex = RV.Flex(total_parcels=1000)
    d = flex.delayed(_StubFacts(0.0, 0.0, 6.0, 6.0), 0.25)
    assert d["share_pct"] == 0.0
    assert d["wait_delayed"] == 0.0


def test_delayed_carries_both_plans_delivery_days(RV):
    flex = RV.Flex(total_parcels=1000)
    d = flex.delayed(_StubFacts(250.0, 0.50, 3.0, 3.5), 0.25)
    assert (d["days_plan1"], d["days"]) == (3.0, 3.5)


# ── the expectations the loaders are checked against ───────────────────────
def test_flex_expectations_name_the_compendium_values(RV):
    """The asserts exist to fail on a moved grid; the values must be the
    compendium's, not whatever the current grid happens to say."""
    e = RV.FLEX_EXPECT
    assert e["peaks_P0"] == (1239, 1666, 1030)
    assert e["peaks_P025"] == (1239, 1314, 1026)
    assert e["total_parcels"] == 1263130
    assert e["regular_eur_per_vd"] == 298.0


def test_vroom_expectations_name_the_realised_savings(RV):
    e = RV.VROOM_EXPECT_V6
    assert e["realised_routing_plan_P0"] == pytest.approx(20.58)
    assert e["realised_operator_plan_P0"] == pytest.approx(22.08)
    # the routing-optimal plan is WORSE than daily delivery in the operator
    # lens, and VROOM made it worse still: the sign is the finding
    assert e["realised_routing_plan_P0_oplens"] < 0
