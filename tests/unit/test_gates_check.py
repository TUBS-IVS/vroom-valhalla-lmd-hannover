"""G1a's near-tie pair-polish tolerance rule (task: G1a tolerance).

Investigation:
``.superpowers/sdd/2026-08-25-realistic-tours-implementation/
g1a-v6-investigation.md`` root-caused G1a's 5/1656 stage-1 mismatches on the
head-enabled v6 grid to ``_pair_polish_round`` (``coordinate_descent.py``):
production runs it after the single-cell CD round converges, and it forces a
same-hub pair onto a JOINT one-day-toggle move, accepted whenever the
*combined* delta is negative even if the G1a-eligible cell's own share is a
small increase. It is never a head leak into the eligible cell's own price
-- ``dd_cost_mx`` is proven bit-identical head vs no-head on all 5 cases. All
5 instances are exact single-day-toggle neighbours of the canonical argmin,
moving the cell's own objective by 0.39-16.41 EUR (0.01-0.48 % of its own
weekly objective).

``scripts/revision/62_gates_check.py``'s ``_g1a_tolerance`` implements the
resulting rule (module docstring part (3)): a mismatch is TOLERATED
(reported, not hard-failed) iff BOTH:

  (i)  structural -- the two schedules differ by exactly one delivery day
       (``len(schedules[new_s1] ^ schedules[canon_s1]) == 1``), the only
       move shape ``_day_toggle_neighbors`` (used exclusively by
       ``_pair_polish_round``) can ever produce;
  (ii) numeric -- ``abs(obj[new_s1] - obj[canon_s1]) <= max(20 EUR, 0.5% of
       obj[canon_s1])``, on the same per-cell objective CD minimises at
       theta=1 (``dd_cost_mx + penalty``).

Anything else -- a bigger schedule change, or a one-day change priced above
tolerance -- still hard-fails, exactly as before this rule existed.

The gate script is imported by path (it is not an importable module name,
``62_gates_check``), the same way ``test_head_enabled_grid.py`` imports
``61_grid_run_v2.py``.
"""
import importlib.util
import logging
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "revision"
           / "62_gates_check.py")


@lru_cache(maxsize=1)
def gates_check():
    """Import ``scripts/revision/62_gates_check.py`` by path.

    Its module body disables INFO logging and installs a blanket warnings
    filter (both wanted for a multi-hour gate run against a live grid,
    neither wanted in a test session) -- undone here so importing it has no
    session-wide side effects, exactly as ``test_head_enabled_grid.runner``
    does for ``61_grid_run_v2.py``.
    """
    filters = warnings.filters[:]
    spec = importlib.util.spec_from_file_location("gates_check_v2", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    logging.disable(logging.NOTSET)
    warnings.filters[:] = filters
    return mod


# A tiny, hand-built schedule table -- NOT the real 39-pattern enumeration,
# just enough day-toggle structure to exercise the three shapes below.
# Index 0 plays "canonical" (the argmin) in every case.
SCHEDULES = [
    frozenset({0, 1, 3, 4}),       # 0: Mon,Tue,Thu,Fri            (canonical)
    frozenset({0, 1, 3, 4, 5}),    # 1: +Sat -- single-day toggle of 0
    frozenset({0, 2, 3, 4}),       # 2: Mon,Wed,Thu,Fri -- vs 0, symmetric
                                    #    difference is {Tue, Wed}, size 2
]


# ─────────────────────────────────────────────────────────────────────────────
# _g1a_tolerance -- the three shapes the controller asked for
# ─────────────────────────────────────────────────────────────────────────────

def test_single_day_toggle_within_tolerance_is_tolerated():
    """A 1-day toggle whose price move is within the EUR/pct bound passes.

    Uses a large canonical objective so the 0.5 % relative arm (50 EUR), not
    the 20 EUR flat floor, is what actually gates the 45 EUR move -- pinning
    the ``max(flat, relative)`` combination, not just one arm of it.
    """
    mod = gates_check()
    obj = np.array([10_000.0, 10_045.0, 1_000_000.0])
    r = mod._g1a_tolerance(new_s1=1, canon_s1=0, obj_row=obj, schedules=SCHEDULES)
    assert r["day_diff"] == 1
    assert r["delta_eur"] == pytest.approx(45.0)
    assert r["threshold_eur"] == pytest.approx(50.0)      # 0.5 % of 10_000
    assert r["tolerated"] is True


def test_single_day_toggle_above_half_percent_is_not_tolerated():
    """Same move shape, priced above both the flat and the relative bound."""
    mod = gates_check()
    obj = np.array([1_000.0, 1_200.0, 1_000_000.0])
    r = mod._g1a_tolerance(new_s1=1, canon_s1=0, obj_row=obj, schedules=SCHEDULES)
    assert r["day_diff"] == 1
    assert r["delta_eur"] == pytest.approx(200.0)          # 20 %, >> 0.5 %
    assert r["threshold_eur"] == pytest.approx(20.0)       # flat floor wins
    assert r["tolerated"] is False


def test_two_day_change_is_never_tolerated_even_when_cheap():
    """A >1-day change hard-fails regardless of price.

    The structural arm is checked independently of price: ``_pair_polish_
    round``'s only neighbourhood restriction (``_day_toggle_neighbors``) can
    never produce a >1-day move, so a mismatch of this shape cannot be an
    instance of the tolerated mechanism no matter how small its own price
    delta is.
    """
    mod = gates_check()
    obj = np.array([1_000.0, 1_000_000.0, 1_000.01])       # one-cent delta
    r = mod._g1a_tolerance(new_s1=2, canon_s1=0, obj_row=obj, schedules=SCHEDULES)
    assert r["day_diff"] == 2
    assert r["delta_eur"] == pytest.approx(0.01)
    assert r["tolerated"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Boundary + wiring
# ─────────────────────────────────────────────────────────────────────────────

def test_tolerance_boundary_is_inclusive():
    """The rule is ``<=``, not ``<`` (module docstring part (3) / investigation
    section 6): a delta exactly at the bound passes, one cent over does not.
    """
    mod = gates_check()
    obj_at = np.array([1_000.0, 1_020.0, 1_000_000.0])       # exactly 20 EUR
    r_at = mod._g1a_tolerance(new_s1=1, canon_s1=0, obj_row=obj_at,
                              schedules=SCHEDULES)
    assert r_at["threshold_eur"] == pytest.approx(20.0)
    assert r_at["tolerated"] is True

    obj_over = np.array([1_000.0, 1_020.01, 1_000_000.0])    # one cent over
    r_over = mod._g1a_tolerance(new_s1=1, canon_s1=0, obj_row=obj_over,
                                schedules=SCHEDULES)
    assert r_over["tolerated"] is False


def test_run_g1a_g1b_default_dict_has_the_tolerated_key():
    """``run_g1a_g1b``'s early-return path (no theta=1 rows yet) must still
    carry ``g1a_tolerated``, so ``render_report`` never ``KeyError``s on a
    partial grid -- exactly the kind of gap a field added to only the
    populated path would leave behind.
    """
    mod = gates_check()
    empty = pd.DataFrame({"share_willing": [], "provider": [], "penalty": []})
    out = mod.run_g1a_g1b(empty, empty, set(), {}, {}, None, SCHEDULES,
                          np.zeros(len(SCHEDULES)), {})
    assert out["available"] is False
    assert out["g1a_mismatches"] == []
    assert out["g1a_tolerated"] == []


def test_run_g1a_g1b_routes_a_tolerated_mismatch_to_pass_not_fail(monkeypatch):
    """End-to-end wiring: a G1a-eligible cell whose stored schedule is a
    tolerated near-tie must land in ``g1a_tolerated``, NOT
    ``g1a_mismatches`` -- and the resulting status must be PASS, matching
    ``main()``'s exit-code rule (``hard_fail = g1["g1a_status"] == "FAIL"
    or ...``). A regression that put every mismatch back into one list, or
    that computed status before the split, would fail this test even though
    the pure ``_g1a_tolerance`` classification above still passed.

    ``get_matrices`` and ``_penalty_mx`` are monkeypatched to hand back a
    small, fully controlled matrix instead of running the real ML surrogate
    -- this test is about ``run_g1a_g1b``'s bucketing/status logic, not
    matrix construction (that has its own runtime asserts in
    ``get_matrices``, and is exercised for real by the live-grid run in the
    task report).
    """
    mod = gates_check()
    prov = "DHL"
    plz = "30001"
    n_sched = len(SCHEDULES)

    dd_cost_mx = np.full((1, n_sched), 1_000_000.0)
    dd_cost_mx[0, 0] = 10_000.0   # canonical argmin (schedule 0)
    dd_cost_mx[0, 1] = 10_045.0   # stored value: 1-day toggle, +45 EUR (<= 50 EUR bound)
    daily_demand = np.full((1, 6), 500.0)  # comfortably >= MIN_TOUR_PARCELS

    fake_m = dict(daily_demand=daily_demand, dd_cost_mx=dd_cost_mx)
    monkeypatch.setattr(mod, "get_matrices", lambda *a, **k: fake_m)
    monkeypatch.setattr(mod, "_penalty_mx", lambda *a, **k: np.zeros((1, n_sched)))

    optim_data = {prov: {"plz_keys": [plz]}}
    ml_prep = {prov: {}}
    done_triples = {mod._key(0.5, 1.0, prov)}
    chosen_df = pd.DataFrame({
        "provider": [prov], "penalty": [0.5], "share_willing": [1.0],
        "plz": [plz], "schedule_idx_stage1": [1],
        "schedule_idx_system_smoothed": [1],
    })
    canonical_df = pd.DataFrame({
        "provider": [prov], "penalty": [0.5], "share_willing": [1.0],
        "plz": [plz], "schedule_idx_system_smoothed": [1],
    })

    out = mod.run_g1a_g1b(chosen_df, canonical_df, done_triples, optim_data,
                          ml_prep, None, SCHEDULES, np.zeros(n_sched), {})

    assert out["g1a_cells_checked"] == 1
    assert out["g1a_mismatches"] == []
    assert len(out["g1a_tolerated"]) == 1
    tol_row = out["g1a_tolerated"][0]
    assert tol_row["provider"] == prov and tol_row["plz"] == plz
    assert tol_row["new_stage1"] == 1 and tol_row["canonical_stage1"] == 0
    assert tol_row["tolerated"] is True
    assert out["g1a_status"] == "PASS"        # main()'s hard_fail rule reads this


def test_run_g1a_g1b_a_two_day_change_still_fails_the_gate(monkeypatch):
    """Same wiring, but the stored schedule is a >1-day change from the
    argmin -- must land in ``g1a_mismatches`` and flip the status to FAIL,
    even though nothing else about the setup differs from the PASS case
    above.
    """
    mod = gates_check()
    prov = "DHL"
    plz = "30001"
    n_sched = len(SCHEDULES)

    dd_cost_mx = np.full((1, n_sched), 1_000_000.0)
    dd_cost_mx[0, 0] = 1_000.0    # canonical argmin (schedule 0)
    dd_cost_mx[0, 2] = 1_000.01   # stored value: 2-day change, one-cent delta
    daily_demand = np.full((1, 6), 500.0)

    fake_m = dict(daily_demand=daily_demand, dd_cost_mx=dd_cost_mx)
    monkeypatch.setattr(mod, "get_matrices", lambda *a, **k: fake_m)
    monkeypatch.setattr(mod, "_penalty_mx", lambda *a, **k: np.zeros((1, n_sched)))

    optim_data = {prov: {"plz_keys": [plz]}}
    ml_prep = {prov: {}}
    done_triples = {mod._key(0.5, 1.0, prov)}
    chosen_df = pd.DataFrame({
        "provider": [prov], "penalty": [0.5], "share_willing": [1.0],
        "plz": [plz], "schedule_idx_stage1": [2],
        "schedule_idx_system_smoothed": [2],
    })
    canonical_df = pd.DataFrame({
        "provider": [prov], "penalty": [0.5], "share_willing": [1.0],
        "plz": [plz], "schedule_idx_system_smoothed": [2],
    })

    out = mod.run_g1a_g1b(chosen_df, canonical_df, done_triples, optim_data,
                          ml_prep, None, SCHEDULES, np.zeros(n_sched), {})

    assert out["g1a_tolerated"] == []
    assert len(out["g1a_mismatches"]) == 1
    assert out["g1a_mismatches"][0]["day_diff"] == 2
    assert out["g1a_status"] == "FAIL"
