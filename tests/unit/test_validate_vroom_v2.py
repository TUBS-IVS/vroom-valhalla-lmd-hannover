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
