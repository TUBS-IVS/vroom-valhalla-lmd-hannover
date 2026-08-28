"""Unit tests for ``scripts/revision/67_validate_vroom_v2.py`` (VROOM validation v2).

Covers the four things the script gets wrong silently if they are wrong:

* the operator-cost reconstruction from an ACTUAL-routes frame
  (``variable = cost - 189.15 * n_routes``; ``peak_h = max_d Sigma n_routes``;
  ``OpCost = Sigma variable + 1134.90 * Sigma_h peak_h``),
* the census / budget arithmetic (solve-time model, projected hours, the G6
  stratified fallback),
* the identity gates on a tiny synthetic matrices dict — the instance
  enumeration must reproduce the grid's own bundled routing total and its
  per-hub-day fleet,
* the append-only CSV header and the resume rule.

No Docker, no checkpoints, no model: every fixture is built in-process.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from batch_delivery.config.constants import (  # noqa: E402
    FIXED_COST_EUR, MIN_TOUR_PARCELS, N_DAYS,
)
from batch_delivery.optimization.balancing import (  # noqa: E402
    WEEK_FIXED_COST_EUR,
)


@pytest.fixture(scope="module")
def V():
    """The script under test, imported by path (its name starts with a digit)."""
    spec = importlib.util.spec_from_file_location(
        "validate_vroom_v2", ROOT / "scripts" / "revision"
        / "67_validate_vroom_v2.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Operator-cost reconstruction from a tiny actual-routes frame
# ─────────────────────────────────────────────────────────────────────────────

def _actual_frame() -> pd.DataFrame:
    """Two hubs, three days, seven instances — peaks on different days."""
    return pd.DataFrame([
        # hub A: 3 routes on d0, 5 on d1 (peak), 2 on d2
        dict(hub_name="A", day=0, vroom_cost_eur=1000.0, vroom_n_routes=3),
        dict(hub_name="A", day=1, vroom_cost_eur=1200.0, vroom_n_routes=4),
        dict(hub_name="A", day=1, vroom_cost_eur=300.0, vroom_n_routes=1),
        dict(hub_name="A", day=2, vroom_cost_eur=700.0, vroom_n_routes=2),
        # hub B: 2 routes on d0 (peak), 1 on d2
        dict(hub_name="B", day=0, vroom_cost_eur=600.0, vroom_n_routes=2),
        dict(hub_name="B", day=2, vroom_cost_eur=250.0, vroom_n_routes=1),
        dict(hub_name="B", day=2, vroom_cost_eur=0.0, vroom_n_routes=0),
    ])


def test_operator_cost_actual_matches_hand_computation(V):
    df = _actual_frame()
    got = V.operator_cost_actual(df)

    routing = 1000 + 1200 + 300 + 700 + 600 + 250 + 0
    vehicle_days = 3 + 4 + 1 + 2 + 2 + 1 + 0
    variable = routing - FIXED_COST_EUR * vehicle_days
    peak_a = max(3, 4 + 1, 2)            # 5
    peak_b = max(2, 1 + 0)               # 2
    opcost = variable + WEEK_FIXED_COST_EUR * (peak_a + peak_b)

    assert got["routing_eur"] == pytest.approx(routing)
    assert got["vehicle_days"] == pytest.approx(vehicle_days)
    assert got["variable_eur"] == pytest.approx(variable)
    assert got["sum_hub_peak"] == pytest.approx(peak_a + peak_b)
    assert got["opcost_eur"] == pytest.approx(opcost)
    assert got["n"] == 7 and got["n_missing_cost"] == 0


def test_operator_cost_actual_peak_is_per_hub_not_global(V):
    """The two hubs peak on DIFFERENT days — Sigma of maxima, not max of sums."""
    df = _actual_frame()
    per_day_total = df.groupby("day").vroom_n_routes.sum().max()   # 5 on d1
    got = V.operator_cost_actual(df)
    assert got["sum_hub_peak"] == 7          # 5 + 2
    assert got["sum_hub_peak"] != per_day_total


def test_operator_cost_actual_unpriced_row_counts_routes_not_cost(V):
    """A PARTIAL/failed row still needs its vans; it just has no price."""
    df = _actual_frame()
    df.loc[len(df)] = dict(hub_name="B", day=0, vroom_cost_eur=np.nan,
                           vroom_n_routes=4)
    got = V.operator_cost_actual(df)
    base = V.operator_cost_actual(_actual_frame())
    assert got["n_missing_cost"] == 1
    assert got["variable_eur"] == pytest.approx(base["variable_eur"])
    # hub B's peak moves from 2 to 6 (2 + 4 on day 0)
    assert got["sum_hub_peak"] == pytest.approx(base["sum_hub_peak"] + 4)


def test_operator_cost_actual_empty(V):
    got = V.operator_cost_actual(pd.DataFrame())
    assert got["opcost_eur"] == 0.0 and got["n"] == 0


def test_operator_cost_predicted_uses_the_same_algebra(V):
    df = _actual_frame().rename(columns={"vroom_cost_eur": "predicted_cost_eur",
                                         "vroom_n_routes": "predicted_n_routes"})
    p = V.operator_cost_predicted(df)
    a = V.operator_cost_actual(_actual_frame())
    assert p["opcost_eur"] == pytest.approx(a["opcost_eur"])
    assert p["sum_hub_peak"] == pytest.approx(a["sum_hub_peak"])


# ─────────────────────────────────────────────────────────────────────────────
# Census / budget arithmetic
# ─────────────────────────────────────────────────────────────────────────────

def test_time_model_falls_back_loudly_when_no_measurement(V, tmp_path):
    model = V.load_time_model(tmp_path / "nope.csv")
    assert model["bins"] is None
    assert "fallback" in model["source"]
    assert V.est_solve_seconds(120, 1, model) == V.FALLBACK_T_SINGLE_S
    assert V.est_solve_seconds(120, 3, model) == V.FALLBACK_T_BUNDLE_S


def test_time_model_from_measurements_is_monotone_in_n_jobs(V, tmp_path):
    rows = []
    for n_jobs, t in ((30, 3.0), (80, 10.0), (150, 37.0), (250, 110.0),
                      (350, 215.0), (450, 320.0)):
        rows += [dict(n_jobs=n_jobs, solve_time_s=t)] * 3
    p = tmp_path / "bundles_solved.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    model = V.load_time_model(p)
    assert model["bins"] is not None
    seq = [V.est_solve_seconds(n, 2, model) for n in (30, 80, 150, 250, 350, 450)]
    assert seq == sorted(seq)
    # above every measured bin: linear extrapolation from the top bin's edge
    assert V.est_solve_seconds(900, 2, model) > seq[-1]


def test_projected_hours_divides_by_parallelism(V):
    secs = [3600.0] * 6
    assert V.projected_hours(secs, 1) == pytest.approx(6.0)
    assert V.projected_hours(secs, 3) == pytest.approx(2.0)
    assert V.projected_hours([], 3) == 0.0


def test_g6_keeps_the_three_smallest_providers_whole(V):
    rows = []
    demand = {"S1": 100, "S2": 200, "S3": 300, "BIG1": 5000, "BIG2": 4000}
    for prov, total in demand.items():
        for i in range(10):
            rows.append(dict(instance_id=f"{prov}-{i}", provider=prov,
                             instance_kind=("delivery_single" if i % 2
                                            else "delivery_group"),
                             n_jobs=10 * (i + 1),
                             predicted_parcels=total / 10.0))
    keep = V.g6_select(rows)
    for prov in ("S1", "S2", "S3"):
        assert all(f"{prov}-{i}" in keep for i in range(10))
    for prov in ("BIG1", "BIG2"):
        sel = [r for r in rows if r["provider"] == prov
               and r["instance_id"] in keep]
        got = sum(r["predicted_parcels"] for r in sel)
        assert got >= 0.5 * demand[prov] - 1e-9
        assert len(sel) < 10                      # it really is a subset


def test_g6_is_stratified_over_kinds(V):
    rows = []
    for i in range(20):
        rows.append(dict(instance_id=f"BIG-{i}", provider="BIG",
                         instance_kind="delivery_group" if i < 4 else "delivery_single",
                         n_jobs=100 + i, predicted_parcels=1000 if i < 4 else 10))
    for prov in ("A", "B", "C"):
        rows.append(dict(instance_id=f"{prov}-0", provider=prov,
                         instance_kind="delivery_single", n_jobs=10,
                         predicted_parcels=1.0))
    keep = V.g6_select(rows)
    kinds = {r["instance_kind"] for r in rows
             if r["provider"] == "BIG" and r["instance_id"] in keep}
    # a pure largest-first rule would take only the four fat delivery_group
    # rows; round-robin over strata must also reach the singles
    assert kinds == {"delivery_group", "delivery_single"}


# ─────────────────────────────────────────────────────────────────────────────
# Identity gates on a tiny synthetic matrices dict
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_grid():
    """3 cells at one hub: one big delivery tour, two pooled small ones, express.

    Schedule 0 delivers on day 0 only, schedule 1 delivers on days 0 and 3.
    Cell 0 is above ``MIN_TOUR_PARCELS`` on its delivery day (its own tour);
    cells 1 and 2 are below it (pooled), and both carry express demand on
    day 1 while not delivering.
    """
    n_plz, n_sched = 3, 2
    schedules = [frozenset({0}), frozenset({0, 3})]
    sched_active = np.zeros((n_sched, N_DAYS), dtype=bool)
    sched_active[0, 0] = True
    sched_active[1, 0] = sched_active[1, 3] = True

    combined_demand = np.zeros((n_plz, n_sched, N_DAYS))
    combined_demand[0, 0, 0] = 900.0                     # big -> own tour
    combined_demand[1, 0, 0] = 100.0                     # small -> pooled
    combined_demand[2, 0, 0] = 80.0                      # small -> pooled
    combined_stops = np.ones((n_plz, n_sched, N_DAYS)) * 20.0

    small_delivery_mask = (
        (combined_demand > 0)
        & (combined_demand < MIN_TOUR_PARCELS)
        & sched_active[None, :, :])

    cost_3d = np.zeros((n_plz, n_sched, N_DAYS))
    cost_3d[0, 0, 0] = 4000.0
    veh_3d = np.zeros((n_plz, n_sched, N_DAYS))
    veh_3d[0, 0, 0] = float(np.maximum(1, np.ceil(900.0 / 230.0)))

    small_delivery_price = np.zeros((n_plz, n_sched, N_DAYS))
    small_delivery_price[1, 0, 0] = 700.0
    small_delivery_price[2, 0, 0] = 600.0

    raw_express = np.zeros((n_plz, N_DAYS))
    raw_express[1, 1] = 40.0
    raw_express[2, 1] = 30.0
    expr_stops = np.ones((n_plz, N_DAYS)) * 5.0
    express_cost = np.zeros((n_plz, N_DAYS))
    express_cost[1, 1] = 300.0
    express_cost[2, 1] = 250.0

    lon = {0: 9.70, 1: 9.71, 2: 9.72}
    lat = {0: 52.30, 1: 52.31, 2: 52.32}
    m = {
        "cost_3d": cost_3d, "veh_3d": veh_3d,
        "sched_active": sched_active,
        "small_delivery_mask": small_delivery_mask,
        "small_delivery_price": small_delivery_price,
        "combined_demand": combined_demand, "combined_stops": combined_stops,
        "raw_express": raw_express, "expr_stops": expr_stops,
        "express_cost": express_cost,
        "area_arr": np.array([3.0, 1.0, 1.0]),
        "hd_arr": np.array([5.0, 6.0, 7.0]),
        "_cent_lon": np.array([lon[z] for z in range(3)]),
        "_cent_lat": np.array([lat[z] for z in range(3)]),
        "plz_day_lon": [[np.array([lon[z]] * 4) for _ in range(N_DAYS)]
                        for z in range(3)],
        "plz_day_lat": [[np.array([lat[z]] * 4) for _ in range(N_DAYS)]
                        for z in range(3)],
    }
    od = {"plz_keys": ["30001", "30002", "30003"],
          "hub_plz_list": [np.array([0, 1, 2])]}
    chosen = np.zeros(3, dtype=np.int64)
    return dict(m=m, od=od, chosen=chosen, schedules=schedules)


def test_enumerate_instances_covers_every_kind(V, tiny_grid):
    rows = V.enumerate_instances(
        1, 0.25, 1.0, "balanced", "DHL", tiny_grid["chosen"], tiny_grid["od"],
        tiny_grid["m"], tiny_grid["schedules"], None)
    kinds = {r["instance_kind"] for r in rows}
    assert V.KIND_DELIVERY_SINGLE in kinds
    assert V.KIND_DELIVERY_GROUP in kinds or V.KIND_EXPRESS_SINGLE in kinds
    # cell 0 is its own tour, priced off cost_3d
    single = [r for r in rows if r["member_idx"] == [0]]
    assert len(single) == 1
    assert single[0]["predicted_cost_eur"] == pytest.approx(4000.0)
    assert single[0]["predicted_source"] == "cost_3d"
    # every instance carries a day, a hub and a positive predicted price
    assert all(r["predicted_cost_eur"] > 0 for r in rows)
    assert all(r["hub_idx"] == 0 for r in rows)


def test_identity_gate_routing_reproduces_the_grid_total(V, tiny_grid):
    rows = V.enumerate_instances(
        1, 0.25, 1.0, "balanced", "DHL", tiny_grid["chosen"], tiny_grid["od"],
        tiny_grid["m"], tiny_grid["schedules"], None)
    expected = 4000.0 + 700.0 + 600.0 + 300.0 + 250.0
    assert V.identity_gate_routing(rows, expected, "tiny") == pytest.approx(expected)


def test_identity_gate_routing_fails_loudly(V, tiny_grid):
    rows = V.enumerate_instances(
        1, 0.25, 1.0, "balanced", "DHL", tiny_grid["chosen"], tiny_grid["od"],
        tiny_grid["m"], tiny_grid["schedules"], None)
    with pytest.raises(AssertionError, match="G1 routing identity FAILED"):
        V.identity_gate_routing(rows, 5851.0, "tiny")


def test_identity_gate_fleet_matches_and_fails(V, tiny_grid):
    rows = V.enumerate_instances(
        1, 0.25, 1.0, "balanced", "DHL", tiny_grid["chosen"], tiny_grid["od"],
        tiny_grid["m"], tiny_grid["schedules"], None)
    per = {}
    for r in rows:
        per[(r["hub_idx"], r["day"])] = (per.get((r["hub_idx"], r["day"]), 0.0)
                                         + r["predicted_n_routes"])
    fleet = pd.DataFrame([{"hub_idx": h, "day": d, "fleet": v}
                          for (h, d), v in per.items()])
    V.identity_gate_fleet(rows, fleet, "tiny")          # must not raise
    fleet.loc[0, "fleet"] = fleet.loc[0, "fleet"] + 1
    with pytest.raises(AssertionError, match="G2 fleet identity FAILED"):
        V.identity_gate_fleet(rows, fleet, "tiny")


def test_pooled_group_vehicles_are_counted_once_per_tour(V, tiny_grid):
    """spec §4.3 v3: members' parcels sum BEFORE the ceil, never per member."""
    rows = V.enumerate_instances(
        1, 0.25, 1.0, "balanced", "DHL", tiny_grid["chosen"], tiny_grid["od"],
        tiny_grid["m"], tiny_grid["schedules"], None)
    pooled = [r for r in rows
              if r["vroom_kind"] == "delivery" and r["member_idx"] != [0]]
    assert pooled, "the two small cells must form a pooled instance"
    assert sum(r["predicted_n_routes"] for r in pooled) == 1.0   # 180 parcels


# ─────────────────────────────────────────────────────────────────────────────
# Header / resume behaviour
# ─────────────────────────────────────────────────────────────────────────────

def test_append_rows_writes_the_header_once_and_fixes_the_order(V, tmp_path):
    p = tmp_path / "tab.csv"
    cols = ["a", "b", "c"]
    V.append_rows(p, [{"b": 2, "a": 1, "c": 3}], cols)
    V.append_rows(p, [{"c": 6, "a": 4}], cols)          # different key order
    txt = p.read_text(encoding="utf-8").strip().splitlines()
    assert txt[0] == "a,b,c"
    assert len(txt) == 3
    df = pd.read_csv(p)
    assert list(df.columns) == cols
    assert df.iloc[1]["a"] == 4 and pd.isna(df.iloc[1]["b"])


def test_append_rows_ignores_empty(V, tmp_path):
    p = tmp_path / "tab.csv"
    V.append_rows(p, [], ["a"])
    assert not p.exists()


def test_load_done_skips_finished_and_retries_failures(V, tmp_path):
    p = tmp_path / "tab_vroom_v2.csv"
    rows = [
        dict(instance_id="ok", vroom_status="OK"),
        dict(instance_id="cached", vroom_status="CACHED"),
        dict(instance_id="partial", vroom_status="PARTIAL"),
        dict(instance_id="empty", vroom_status="EMPTY"),
        dict(instance_id="err", vroom_status="ERROR"),
        dict(instance_id="conn", vroom_status="CONN_ERROR"),
        dict(instance_id="timeout", vroom_status="TIMEOUT"),
    ]
    pd.DataFrame(rows).to_csv(p, index=False)
    done = V.load_done(p)
    assert done == {"ok", "cached", "partial", "empty"}
    assert V.load_done(tmp_path / "missing.csv") == set()


def test_load_done_refuses_an_incompatible_table(V, tmp_path):
    p = tmp_path / "tab_vroom_v2.csv"
    pd.DataFrame([{"foo": 1}]).to_csv(p, index=False)
    with pytest.raises(SystemExit):
        V.load_done(p)


def test_instance_id_is_stable_and_plan_aware(V):
    base = dict(item=1, penalty=0.25, share_willing=1.0, plan="balanced",
                provider="DHL", vroom_kind="delivery", day=2,
                members=["30159", "30161"])
    a = V.instance_id(base)
    assert a == V.instance_id(dict(base, members=json.dumps(base["members"])))
    assert a != V.instance_id(dict(base, plan="stage1"))
    assert a != V.instance_id(dict(base, day=3))


def test_default_cache_tag_inherits_the_right_namespaces(V):
    single = dict(n_members=1, vroom_kind="delivery", penalty=0.25,
                  share_willing=1.0, provider="DHL")
    assert V.default_cache_tag(single) == "smval_p0.25_s1.0_DHL"
    group = dict(single, n_members=3)
    assert V.default_cache_tag(group) == "bundle_DHL"
    express = dict(single, vroom_kind="express")
    assert V.default_cache_tag(express) == "bundle_DHL"


# ─────────────────────────────────────────────────────────────────────────────
# The cache probe's own gates (no cache content needed)
# ─────────────────────────────────────────────────────────────────────────────

def test_cache_probe_gates_hold_against_the_real_builders(V):
    """The offset reconstruction and the canonical-string assembly are exact."""
    from batch_delivery.routing.core import (
        build_vroom_jobs, build_vroom_vehicles,
    )
    pts = pd.DataFrame({"lon": np.linspace(9.70, 9.80, 25),
                        "lat": np.linspace(52.30, 52.40, 25),
                        "dhl_total": [9] * 25})
    jobs, total_demand = build_vroom_jobs(pts)
    for hub_typ in ("ZB", "ZSP"):                  # large and small hub windows
        hub = {"hub_lon": 9.75, "hub_lat": 52.35, "hub_typ": hub_typ}
        vehicles, _ = build_vroom_vehicles(
            hub=hub, total_demand=total_demand, day_idx=2,
            seed_key="unit-test", n_jobs=len(jobs))
        hit, tag, k = V.cache_probe(jobs, vehicles, hub, 2, "unit-test", {})
        assert (hit, tag, k) == (False, "", -1)     # empty index: no hit
        # a planted body is found, and the reported candidate reproduces it
        idx = {}
        vtw = V._vtw_for_hub(hub)
        prefix, parts = V._canonical_parts(jobs, vehicles, vtw)
        offs = V._offset_table(2, len(vehicles), max(0, V.VEH_START_LATEST - vtw[0]))[7]
        import hashlib
        body = V._assemble(prefix, parts, offs, vtw[0])
        idx[hashlib.sha256(body.encode()).hexdigest()[:16]] = "planted_tag"
        hit, tag, k = V.cache_probe(jobs, vehicles, hub, 2, "unit-test", idx)
        assert hit and tag == "planted_tag"
        assert list(V._offset_table(2, len(vehicles),
                                    max(0, V.VEH_START_LATEST - vtw[0]))[k]) \
            == list(offs)


# ─────────────────────────────────────────────────────────────────────────────
# Item 0 — the theta=0 baseline (Task 12b): enumeration is delivery-only,
# the P-invariance / stage1==balanced / no-express gates, queue-merge safety,
# and predicted-vs-actual saving % in both lenses.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_baseline_grid():
    """The theta=0 baseline: ONE schedule (the all-daily one), no express.

    Every cell is pinned to the same all-daily schedule (index 0) — the
    model's own theta=0 invariant (G-6f-1 in ``61_grid_run_v2.py``) — so
    ``sched_active`` is True on every day for every cell and
    ``_express_contributing`` can never find a non-delivery day, no matter
    how large ``raw_express`` is. Cell 0 is its own tour (day 0, above
    MIN_TOUR_PARCELS); cells 1/2 are small and pool into one delivery group.
    """
    n_plz, n_sched = 3, 1
    schedules = [frozenset(range(N_DAYS))]
    sched_active = np.ones((n_sched, N_DAYS), dtype=bool)

    combined_demand = np.zeros((n_plz, n_sched, N_DAYS))
    combined_demand[0, 0, 0] = 900.0                     # big -> own tour
    combined_demand[1, 0, 0] = 100.0                     # small -> pooled
    combined_demand[2, 0, 0] = 80.0                      # small -> pooled
    combined_stops = np.ones((n_plz, n_sched, N_DAYS)) * 20.0

    small_delivery_mask = (
        (combined_demand > 0)
        & (combined_demand < MIN_TOUR_PARCELS)
        & sched_active[None, :, :])

    cost_3d = np.zeros((n_plz, n_sched, N_DAYS))
    cost_3d[0, 0, 0] = 4000.0
    veh_3d = np.zeros((n_plz, n_sched, N_DAYS))
    veh_3d[0, 0, 0] = float(np.maximum(1, np.ceil(900.0 / 230.0)))

    small_delivery_price = np.zeros((n_plz, n_sched, N_DAYS))
    small_delivery_price[1, 0, 0] = 700.0
    small_delivery_price[2, 0, 0] = 600.0

    # Deliberately NON-zero: proves raw_express is IGNORED because every day
    # is a delivery day here, not merely absent from this fixture.
    raw_express = np.zeros((n_plz, N_DAYS))
    raw_express[1, 2] = 40.0
    expr_stops = np.ones((n_plz, N_DAYS)) * 5.0
    express_cost = np.zeros((n_plz, N_DAYS))
    express_cost[1, 2] = 300.0

    lon = {0: 9.70, 1: 9.71, 2: 9.72}
    lat = {0: 52.30, 1: 52.31, 2: 52.32}
    m = {
        "cost_3d": cost_3d, "veh_3d": veh_3d,
        "sched_active": sched_active,
        "small_delivery_mask": small_delivery_mask,
        "small_delivery_price": small_delivery_price,
        "combined_demand": combined_demand, "combined_stops": combined_stops,
        "raw_express": raw_express, "expr_stops": expr_stops,
        "express_cost": express_cost,
        "area_arr": np.array([3.0, 1.0, 1.0]),
        "hd_arr": np.array([5.0, 6.0, 7.0]),
        "_cent_lon": np.array([lon[z] for z in range(3)]),
        "_cent_lat": np.array([lat[z] for z in range(3)]),
        "plz_day_lon": [[np.array([lon[z]] * 4) for _ in range(N_DAYS)]
                        for z in range(3)],
        "plz_day_lat": [[np.array([lat[z]] * 4) for _ in range(N_DAYS)]
                        for z in range(3)],
    }
    od = {"plz_keys": ["30001", "30002", "30003"],
          "hub_plz_list": [np.array([0, 1, 2])]}
    chosen = np.zeros(3, dtype=np.int64)
    return dict(m=m, od=od, chosen=chosen, schedules=schedules)


def test_item0_enumeration_is_delivery_only_even_with_raw_express_present(
        V, tiny_baseline_grid):
    """The defining property of item 0: no express, no matter what raw_express
    holds, because an all-daily schedule leaves no non-delivery day."""
    rows = V.enumerate_instances(
        0, 0.0, 0.0, "stage1", "DHL", tiny_baseline_grid["chosen"],
        tiny_baseline_grid["od"], tiny_baseline_grid["m"],
        tiny_baseline_grid["schedules"], None)
    assert rows, "the baseline must still produce delivery instances"
    assert all(r["vroom_kind"] == "delivery" for r in rows)
    assert all(r["item"] == 0 for r in rows)
    kinds = {r["instance_kind"] for r in rows}
    assert kinds and kinds <= {V.KIND_DELIVERY_SINGLE, V.KIND_DELIVERY_GROUP}

    single = [r for r in rows if r["member_idx"] == [0]]
    assert len(single) == 1
    assert single[0]["predicted_cost_eur"] == pytest.approx(4000.0)
    assert single[0]["predicted_source"] == "cost_3d"

    pooled = [r for r in rows if r["member_idx"] != [0]]
    assert pooled, "cells 1/2 must pool into one small-delivery group"
    assert sum(r["predicted_cost_eur"] for r in pooled) == pytest.approx(1300.0)

    total = V.identity_gate_routing(rows, 4000.0 + 700.0 + 600.0,
                                    "tiny-baseline")
    assert total == pytest.approx(5300.0)


def test_item0_enumeration_gates_fleet_like_any_other_item(V, tiny_baseline_grid):
    """G2 is run for item 0 too (its plan is labelled "stage1", not
    "balanced", but at theta=0 the two coincide — see run_census)."""
    rows = V.enumerate_instances(
        0, 0.0, 0.0, "stage1", "DHL", tiny_baseline_grid["chosen"],
        tiny_baseline_grid["od"], tiny_baseline_grid["m"],
        tiny_baseline_grid["schedules"], None)
    per: dict = {}
    for r in rows:
        per[(r["hub_idx"], r["day"])] = (per.get((r["hub_idx"], r["day"]), 0.0)
                                         + r["predicted_n_routes"])
    fleet = pd.DataFrame([{"hub_idx": h, "day": d, "fleet": v}
                          for (h, d), v in per.items()])
    V.identity_gate_fleet(rows, fleet, "tiny-baseline")       # must not raise
    fleet.loc[0, "fleet"] = fleet.loc[0, "fleet"] + 1
    with pytest.raises(AssertionError, match="G2 fleet identity FAILED"):
        V.identity_gate_fleet(rows, fleet, "tiny-baseline")


# ── assert_baseline_invariant / grid_express_cost on a synthetic chosen/costs
# frame — the P-invariance, stage1==balanced and no-express checks item 0
# relies on to validate ONE representative P instead of every P in the grid.

def _baseline_chosen_df(bad_p: float | None = None,
                        bad_plan: str | None = None) -> pd.DataFrame:
    """theta=0 rows for DHL across three P values, plus a theta=1 distractor.

    ``bad_p``: corrupt that P's schedule so it differs from P=0's.
    ``bad_plan``: corrupt schedule_idx_balanced at P=0 (breaks stage1==balanced).
    """
    rows = []
    plzs = ["30001", "30002", "30003"]
    daily = [3, 3, 3]              # an arbitrary "daily" schedule index
    for P in (0.0, 0.25, 0.5):
        s1 = list(daily)
        if bad_p is not None and np.isclose(P, bad_p):
            s1 = [s1[0] + 1, s1[1], s1[2]]
        bal = list(s1)
        if bad_plan == "balanced" and np.isclose(P, 0.0):
            bal = [bal[0] + 1, bal[1], bal[2]]
        for plz, s, b in zip(plzs, s1, bal):
            rows.append(dict(penalty=P, share_willing=0.0, provider="DHL",
                             plz=plz, schedule_idx_stage1=s,
                             schedule_idx_balanced=b,
                             schedule_idx_system_smoothed=b))
    # theta=1 distractor rows: must be ignored by every theta=0 check
    for plz in plzs:
        rows.append(dict(penalty=0.0, share_willing=1.0, provider="DHL",
                         plz=plz, schedule_idx_stage1=0,
                         schedule_idx_balanced=1,
                         schedule_idx_system_smoothed=1))
    return pd.DataFrame(rows)


def _baseline_costs_df(express: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame([
        dict(penalty=P, share_willing=0.0, provider="DHL",
             cost_stage1_eur=5300.0, cost_stage2_eur=5300.0,
             express_cost_eur=(express if np.isclose(P, 0.0) else 0.0))
        for P in (0.0, 0.25, 0.5)
    ])


def test_assert_baseline_invariant_passes_on_a_consistent_grid(V):
    chosen_df = _baseline_chosen_df()
    costs_df = _baseline_costs_df()
    base, checked = V.assert_baseline_invariant(
        chosen_df, costs_df, "DHL", ["30001", "30002", "30003"])
    assert base.tolist() == [3, 3, 3]
    assert sorted(checked) == [0.25, 0.5]


def test_assert_baseline_invariant_catches_p_dependence(V):
    chosen_df = _baseline_chosen_df(bad_p=0.25)
    costs_df = _baseline_costs_df()
    with pytest.raises(AssertionError, match="P-invariant"):
        V.assert_baseline_invariant(chosen_df, costs_df, "DHL",
                                    ["30001", "30002", "30003"])


def test_assert_baseline_invariant_catches_stage1_balanced_mismatch(V):
    chosen_df = _baseline_chosen_df(bad_plan="balanced")
    costs_df = _baseline_costs_df()
    with pytest.raises(AssertionError, match="schedule_idx_balanced"):
        V.assert_baseline_invariant(chosen_df, costs_df, "DHL",
                                    ["30001", "30002", "30003"])


def test_assert_baseline_invariant_catches_nonzero_express(V):
    chosen_df = _baseline_chosen_df()
    costs_df = _baseline_costs_df(express=12.5)
    with pytest.raises(AssertionError, match="express_cost_eur"):
        V.assert_baseline_invariant(chosen_df, costs_df, "DHL",
                                    ["30001", "30002", "30003"])


def test_grid_express_cost_reads_the_column(V):
    costs_df = _baseline_costs_df(express=7.0)
    assert V.grid_express_cost(costs_df, 0.0, 0.0, "DHL") == pytest.approx(7.0)


def test_item_0_is_a_known_item_and_accepted_by_the_cli(V):
    assert 0 in V.ITEMS
    assert V.ITEMS[0]["theta"] == 0.0
    assert V.ITEMS[0]["plan"] == "stage1"
    assert V.ITEMS[0]["penalties"] == [0.0]
    args = V.parse_args(["--items", "0", "--census-only"])
    assert args.items == [0]
    args = V.parse_args(["--items", "0,1,2,3"])
    assert args.items == [0, 1, 2, 3]


# ── merge_queue_rows: the queue-merge that keeps a partial census (item 0
# alone) from clobbering other items' already-solved instance rows ──────────

def test_merge_queue_rows_preserves_other_items_and_replaces_requested_ones(V):
    old = pd.DataFrame([
        dict(instance_id="i1-a", item=1, cache_hit=True, g6_selected=True),
        dict(instance_id="i2-a", item=2, cache_hit=False, g6_selected=True),
        dict(instance_id="i3-a", item=3, cache_hit=False, g6_selected=False),
    ]).reindex(columns=V.QUEUE_COLS)
    fresh = pd.DataFrame([
        dict(instance_id="i0-a", item=0, cache_hit=True, g6_selected=True),
    ]).reindex(columns=V.QUEUE_COLS)

    merged = V.merge_queue_rows(old, fresh, [0])

    assert sorted(merged["instance_id"]) == ["i0-a", "i1-a", "i2-a", "i3-a"]
    # item 3's pre-existing G6 selection must survive a census that never
    # touched item 3
    i3 = merged[merged["item"] == 3].iloc[0]
    assert bool(i3["g6_selected"]) is False
    assert list(merged.columns) == V.QUEUE_COLS


def test_merge_queue_rows_replaces_only_the_requested_items(V):
    old = pd.DataFrame([
        dict(instance_id="i1-a", item=1, cache_hit=True),
        dict(instance_id="i1-b", item=1, cache_hit=False),
        dict(instance_id="i3-a", item=3, cache_hit=True),
    ]).reindex(columns=V.QUEUE_COLS)
    fresh = pd.DataFrame([
        dict(instance_id="i1-new", item=1, cache_hit=False),
    ]).reindex(columns=V.QUEUE_COLS)

    merged = V.merge_queue_rows(old, fresh, [1])

    assert sorted(merged["instance_id"]) == ["i1-new", "i3-a"]


def test_merge_queue_rows_first_census_has_no_old_queue(V):
    fresh = pd.DataFrame([dict(instance_id="i0-a", item=0)]).reindex(
        columns=V.QUEUE_COLS)
    merged = V.merge_queue_rows(None, fresh, [0])
    assert list(merged["instance_id"]) == ["i0-a"]
    assert list(merged.columns) == V.QUEUE_COLS


# ─────────────────────────────────────────────────────────────────────────────
# savings_vs_baseline: predicted vs ACTUAL saving %, both lenses, both totals
# (with and without PARTIAL rows) — the report item 0 exists to make possible
# ─────────────────────────────────────────────────────────────────────────────

def _savings_frame() -> pd.DataFrame:
    """A theta=0 baseline (2 clean rows) and one point with a PARTIAL row.

    The point's PARTIAL row (day 1) carries MORE routes (5) than its OK row
    (2) on purpose: it makes the operator lens's peak term — and therefore
    the "with vs without PARTIAL" saving % — visibly (here: sign-flippingly)
    different, which is exactly the distinction the report has to state both
    totals for.
    """
    return pd.DataFrame([
        dict(item=0, penalty=0.0, share_willing=0.0, plan="stage1",
             hub_name="A", day=0, vroom_cost_eur=1000.0, vroom_n_routes=3,
             predicted_cost_eur=1050.0, predicted_n_routes=3,
             vroom_status="OK", instance_kind="delivery_single"),
        dict(item=0, penalty=0.0, share_willing=0.0, plan="stage1",
             hub_name="A", day=1, vroom_cost_eur=800.0, vroom_n_routes=2,
             predicted_cost_eur=820.0, predicted_n_routes=2,
             vroom_status="OK", instance_kind="delivery_single"),
        dict(item=1, penalty=0.25, share_willing=1.0, plan="balanced",
             hub_name="A", day=0, vroom_cost_eur=600.0, vroom_n_routes=2,
             predicted_cost_eur=650.0, predicted_n_routes=2,
             vroom_status="OK", instance_kind="delivery_single"),
        dict(item=1, penalty=0.25, share_willing=1.0, plan="balanced",
             hub_name="A", day=1, vroom_cost_eur=np.nan, vroom_n_routes=5,
             predicted_cost_eur=150.0, predicted_n_routes=1,
             vroom_status="PARTIAL", instance_kind="delivery_single"),
    ])


def _queue_frame_full() -> pd.DataFrame:
    """The census matching ``_savings_frame`` exactly: every point FULL (no
    G6 restriction anywhere) — solved count == census count for both items.
    """
    rows = [dict(item=0, penalty=0.0, share_willing=0.0, plan="stage1",
                build_note="OK", g6_selected=True) for _ in range(2)]
    rows += [dict(item=1, penalty=0.25, share_willing=1.0, plan="balanced",
                 build_note="OK", g6_selected=True) for _ in range(2)]
    return pd.DataFrame(rows)


def test_savings_vs_baseline_both_lenses_with_and_without_partial(V):
    df = _savings_frame()
    rows = V.savings_vs_baseline(df, _queue_frame_full())
    assert len(rows) == 1
    r = rows[0]
    assert (r["item"], r["penalty"], r["share_willing"], r["plan"]) == (
        1, 0.25, 1.0, "balanced")
    assert r["n_all"] == 2 and r["n_clean"] == 1
    assert r["base_n_all"] == 2 and r["base_n_clean"] == 2
    # a FULL point (no G6 restriction): census == target == solved, and the
    # saving-% fields are real numbers, never the subset "n/a"
    assert r["is_g6_subset"] is False
    assert r["n_census"] == 2 and r["n_target"] == 2

    base_pred_routing = 1050.0 + 820.0
    point_pred_routing = 650.0 + 150.0
    assert r["pred_routing_save_pct"] == pytest.approx(
        (base_pred_routing - point_pred_routing) / base_pred_routing * 100)

    base_act_routing = 1000.0 + 800.0
    point_act_routing = 600.0                 # the PARTIAL row has no cost
    assert r["act_routing_save_pct_clean"] == pytest.approx(
        (base_act_routing - point_act_routing) / base_act_routing * 100)
    assert r["act_routing_save_pct_all"] == pytest.approx(
        (base_act_routing - point_act_routing) / base_act_routing * 100)

    base_var_pred = base_pred_routing - FIXED_COST_EUR * (3 + 2)
    base_opcost_pred = base_var_pred + WEEK_FIXED_COST_EUR * max(3, 2)
    point_var_pred = point_pred_routing - FIXED_COST_EUR * (2 + 1)
    point_opcost_pred = point_var_pred + WEEK_FIXED_COST_EUR * max(2, 1)
    assert r["pred_opcost_save_pct"] == pytest.approx(
        (base_opcost_pred - point_opcost_pred) / base_opcost_pred * 100)

    base_var_act = base_act_routing - FIXED_COST_EUR * (3 + 2)
    base_opcost_act = base_var_act + WEEK_FIXED_COST_EUR * max(3, 2)

    point_var_act_clean = point_act_routing - FIXED_COST_EUR * 2
    point_opcost_act_clean = point_var_act_clean + WEEK_FIXED_COST_EUR * 2
    assert r["act_opcost_save_pct_clean"] == pytest.approx(
        (base_opcost_act - point_opcost_act_clean) / base_opcost_act * 100)

    point_var_act_all = point_act_routing - FIXED_COST_EUR * 2   # PARTIAL unpriced
    point_opcost_act_all = point_var_act_all + WEEK_FIXED_COST_EUR * max(2, 5)
    assert r["act_opcost_save_pct_all"] == pytest.approx(
        (base_opcost_act - point_opcost_act_all) / base_opcost_act * 100)

    # the whole point of stating both totals: they must actually differ here
    # (in this fixture, dramatically enough to flip sign)
    assert r["act_opcost_save_pct_all"] < 0 < r["act_opcost_save_pct_clean"]


def test_savings_vs_baseline_empty_when_baseline_unsolved(V):
    df = _savings_frame()
    df = df[df["item"] != 0]
    assert V.savings_vs_baseline(df, _queue_frame_full()) == []


def test_write_report_includes_the_predicted_vs_actual_saving_section(V, tmp_path):
    df = _savings_frame()
    df.to_csv(tmp_path / V.SOLVED_CSV, index=False)
    _queue_frame_full().to_csv(tmp_path / V.QUEUE_CSV, index=False)
    args = SimpleNamespace(rev_dir=str(tmp_path))

    V.write_report(tmp_path, args)

    text = (tmp_path / V.REPORT_MD).read_text(encoding="utf-8")
    assert "Predicted vs actual saving" in text
    assert "0.25" in text and "balanced" in text

    # both the "clean" and "all" actual-saving columns must appear, and with
    # DIFFERENT values for this fixture (re-read the CSV exactly as
    # write_report did, so the formatted strings are guaranteed to line up).
    df_check = pd.read_csv(tmp_path / V.SOLVED_CSV)
    queue_check = pd.read_csv(tmp_path / V.QUEUE_CSV)
    r = V.savings_vs_baseline(df_check, queue_check)[0]
    clean_str = f"{r['act_opcost_save_pct_clean']:+.2f}"
    all_str = f"{r['act_opcost_save_pct_all']:+.2f}"
    assert clean_str in text
    assert all_str in text
    assert clean_str != all_str
    # the gap columns this report adds are present and populated
    assert "gap % routing" in text and "gap % OpCost" in text
    assert f"{r['gap_routing_pct']:+.2f}" in text


def test_write_report_notes_a_pending_baseline_when_item_0_absent(V, tmp_path):
    df = _savings_frame()
    df = df[df["item"] != 0]
    df.to_csv(tmp_path / V.SOLVED_CSV, index=False)
    _queue_frame_full().to_csv(tmp_path / V.QUEUE_CSV, index=False)
    args = SimpleNamespace(rev_dir=str(tmp_path))

    V.write_report(tmp_path, args)

    text = (tmp_path / V.REPORT_MD).read_text(encoding="utf-8")
    assert "has not been solved yet" in text


def test_write_report_requires_the_instance_queue(V, tmp_path):
    """Task 12c: a saving % needs the census to know which points are a G6
    subset — writing the report without it must raise, never guess."""
    df = _savings_frame()
    df.to_csv(tmp_path / V.SOLVED_CSV, index=False)
    args = SimpleNamespace(rev_dir=str(tmp_path))
    with pytest.raises(AssertionError, match="no instance queue"):
        V.write_report(tmp_path, args)


# ─────────────────────────────────────────────────────────────────────────────
# G6-subset awareness (Task 12c): a point solved on a stratified fallback
# subset (67_'s ``--g6-fallback``) must never state a saving % against the
# FULL baseline — the v6 bug this fixes had item 3 (1000 of 1594 census
# instances solved) print a meaningless +38.7 % this way. The predicted-
# vs-actual GAP on the point's own solved rows stays valid regardless.
# ─────────────────────────────────────────────────────────────────────────────

def _subset_frame() -> pd.DataFrame:
    """A full theta=0 baseline (item 0), a FULL item 1 point (2/2 census),
    and a G6 SUBSET item 3 point (2 of its 5 census instances solved)."""
    rows = []
    for d in range(2):
        rows.append(dict(item=0, penalty=0.0, share_willing=0.0,
                         plan="stage1", hub_name="A", day=d,
                         vroom_cost_eur=1000.0, vroom_n_routes=2,
                         predicted_cost_eur=1000.0, predicted_n_routes=2,
                         vroom_status="OK", instance_kind="delivery_single"))
    for d in range(2):
        rows.append(dict(item=1, penalty=0.25, share_willing=1.0,
                         plan="balanced", hub_name="A", day=d,
                         vroom_cost_eur=400.0, vroom_n_routes=1,
                         predicted_cost_eur=420.0, predicted_n_routes=1,
                         vroom_status="OK", instance_kind="delivery_single"))
    for d in range(2):                                 # only 2 of 5 census
        rows.append(dict(item=3, penalty=0.25, share_willing=0.5,
                         plan="balanced", hub_name="A", day=d,
                         vroom_cost_eur=300.0, vroom_n_routes=1,
                         predicted_cost_eur=310.0, predicted_n_routes=1,
                         vroom_status="OK", instance_kind="delivery_single"))
    return pd.DataFrame(rows)


def _subset_queue() -> pd.DataFrame:
    """item 0 and item 1 FULL (2/2 each); item 3 a G6 subset: 2 of a 5-row
    census selected — matching exactly what ``_subset_frame`` solved."""
    rows = [dict(item=0, penalty=0.0, share_willing=0.0, plan="stage1",
                build_note="OK", g6_selected=True) for _ in range(2)]
    rows += [dict(item=1, penalty=0.25, share_willing=1.0, plan="balanced",
                 build_note="OK", g6_selected=True) for _ in range(2)]
    rows += [dict(item=3, penalty=0.25, share_willing=0.5, plan="balanced",
                 build_note="OK", g6_selected=(i < 2)) for i in range(5)]
    return pd.DataFrame(rows)


def test_savings_vs_baseline_g6_subset_is_na_but_gap_is_valid(V):
    rows = V.savings_vs_baseline(_subset_frame(), _subset_queue())
    by_item = {r["item"]: r for r in rows}
    full, subset = by_item[1], by_item[3]

    # the FULL item is unaffected: real numbers, census == target == solved
    assert full["is_g6_subset"] is False
    assert full["n_census"] == full["n_target"] == full["n_all"] == 2
    assert not any(np.isnan(full[k]) for k in (
        "pred_routing_save_pct", "act_routing_save_pct_all",
        "pred_opcost_save_pct", "act_opcost_save_pct_all"))

    # the SUBSET item: every saving-% field is NaN, n_all == n_target == 2
    # (fully solved for ITS OWN target) but n_census == 5 (the full item)
    assert subset["is_g6_subset"] is True
    assert subset["n_all"] == 2 and subset["n_target"] == 2
    assert subset["n_census"] == 5
    for k in ("pred_routing_save_pct", "pred_opcost_save_pct",
             "act_routing_save_pct_clean", "act_routing_save_pct_all",
             "act_opcost_save_pct_clean", "act_opcost_save_pct_all"):
        assert np.isnan(subset[k]), f"{k} must be NaN for a G6 subset"

    # the GAP (predicted vs actual on the subset's OWN solved rows) stays a
    # real number: Sigma pred 620, Sigma actual 600 -> +3.33 %
    assert subset["point_routing_pred_eur"] == pytest.approx(620.0)
    assert subset["point_routing_act_all_eur"] == pytest.approx(600.0)
    assert subset["gap_routing_pct"] == pytest.approx(
        (620.0 - 600.0) / 600.0 * 100)
    assert not np.isnan(subset["gap_opcost_pct"])


def test_write_report_g6_subset_row_prints_na_and_keeps_the_gap(V, tmp_path):
    _subset_frame().to_csv(tmp_path / V.SOLVED_CSV, index=False)
    _subset_queue().to_csv(tmp_path / V.QUEUE_CSV, index=False)
    args = SimpleNamespace(rev_dir=str(tmp_path))

    V.write_report(tmp_path, args)
    text = (tmp_path / V.REPORT_MD).read_text(encoding="utf-8")

    na = "n/a (G6 subset, not comparable to the full baseline)"
    assert na in text
    # the subset row's own n column: solved/census, not solved/clean
    assert "| 3 | 0.25 | 0.5 | balanced | 2/5 |" in text

    # the FULL item 1 row must NOT be marked n/a -- scoped to the savings
    # table specifically, since the "Routing lens" table above it ALSO has
    # rows starting "| 1 |" (grouped by instance_kind) and never says n/a
    # regardless, which would make an unscoped search vacuously pass.
    section = text.split("## Predicted vs actual saving")[1]
    lines = [ln for ln in section.splitlines() if ln.startswith("| 1 |")]
    assert lines and na not in lines[0]
    # the gap columns are present and numeric for the subset row
    rows = V.savings_vs_baseline(_subset_frame(), _subset_queue())
    subset = [r for r in rows if r["item"] == 3][0]
    assert f"{subset['gap_routing_pct']:+.2f}" in text


def test_savings_vs_baseline_raises_when_baseline_itself_is_a_subset(V):
    df = _subset_frame()
    queue = _subset_queue()
    queue.loc[queue.item == 0, "g6_selected"] = [True, False]
    with pytest.raises(AssertionError, match="ITSELF a recorded G6 subset"):
        V.savings_vs_baseline(df, queue)


def test_savings_vs_baseline_raises_on_an_incompletely_solved_point(V):
    df = _subset_frame()
    df = df[~((df.item == 3) & (df.day == 1))]         # only 1 of 2 target
    queue = _subset_queue()                            # target is still 2
    with pytest.raises(AssertionError,
                       match="solved but its recorded target"):
        V.savings_vs_baseline(df, queue)


def test_savings_vs_baseline_raises_when_a_point_has_no_census_group(V):
    df = _subset_frame()
    queue = _subset_queue()
    queue = queue[queue.item != 3]                     # item 3 vanishes
    with pytest.raises(AssertionError, match="no census group"):
        V.savings_vs_baseline(df, queue)


def test_savings_vs_baseline_requires_a_queue(V):
    with pytest.raises(AssertionError, match="queue_df"):
        V.savings_vs_baseline(_subset_frame(), None)
