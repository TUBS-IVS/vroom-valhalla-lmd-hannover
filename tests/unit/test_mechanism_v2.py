"""The pure builders in ``scripts/revision/77_mechanism_v2.py``.

77_ turns the v6 grid tables into the mechanism figure's two CSVs. Everything
it computes is a pure function of frames, so the tests run on a synthetic
2-provider grid (2 P x 3 theta) rather than on the real 616-triple run.

What the tests are actually protecting:

* ``mechanism_table`` re-derives the stage-1 saving from ``cost_stage1_eur``
  and gates it against the grid's own ``routing_saving_plan1_pct`` (G1). If
  that identity ever silently breaks -- a plan mix-up is the obvious way --
  the figure would draw one plan's saving under another plan's label.
* the express quantities are stage-2 quantities that must be exactly zero at
  theta in {0, 1} and positive in between (G2). Zero express vehicles with a
  non-zero express cost (or the reverse) is a division the module must refuse
  rather than paper over with a NaN.
* the histogram's cumulative share at a penalty threshold is the number the
  figure prints as "x % of cells save less than the penalty" (G4), so it is
  checked against the share computed straight from the values, and a bin
  width that does not put the threshold on an edge is a hard error.

The module's real-universe constants (``N_PROVIDERS`` = 7, ``N_CELLS`` = 312)
are monkeypatched down to the fixture's size instead of being removed from
the production code -- the completeness checks stay in force for the real run.
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
sys.path.insert(0, str(ROOT / "scripts" / "presentation"))

PROVS = ["P1", "P2"]
THETAS = [0.0, 0.5, 1.0]
PENALTIES = [0.0, 0.25]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_mechanism_v2", ROOT / "scripts" / "revision" / "77_mechanism_v2.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


@pytest.fixture(autouse=True)
def _two_providers(mod, monkeypatch):
    monkeypatch.setattr(mod, "N_PROVIDERS", len(PROVS))
    monkeypatch.setattr(mod, "N_CELLS", 4)


# ─────────────────────────────────────────────────────────────────────────
# fixture: 2 providers, 2 P, 3 theta -- consistent by construction
# ─────────────────────────────────────────────────────────────────────────
def _costs() -> pd.DataFrame:
    """Baseline 1 000 EUR/provider; every scenario cheaper; express only at
    0 < theta < 1, exactly as the real tables behave."""
    rows = []
    for P in PENALTIES:
        for th in THETAS:
            for i, prov in enumerate(PROVS):
                base = 1000.0 + 100.0 * i
                stage1 = base if th == 0 else base * (1 - 0.10 - 0.02 * th)
                stage2 = stage1 * 1.01
                express = 0.0 if th in (0.0, 1.0) else 40.0 * th * (1 + i)
                rows.append(dict(
                    penalty=P, share_willing=th, provider=prov,
                    express_cost_eur=express, routing_total_eur=stage2,
                    penalty_before_eur=10.0 * P * th,
                    penalty_eur=12.0 * P * th,
                    cost_stage1_eur=stage1, cost_stage2_eur=stage2))
    return pd.DataFrame(rows)


def _fleet(costs: pd.DataFrame) -> pd.DataFrame:
    """One hub-day row per (P, theta, provider): total fleet plus the express
    vehicles, which are zero exactly where the express cost is zero."""
    rows = []
    for _, r in costs.iterrows():
        rows.append(dict(penalty=r.penalty, share_willing=r.share_willing,
                         provider=r.provider,
                         express_veh=0.0 if r.express_cost_eur == 0 else 4.0,
                         fleet=20.0))
    return pd.DataFrame(rows)


def _grid(costs: pd.DataFrame) -> pd.DataFrame:
    """The grid table the module gates against, derived FROM the costs so the
    positive path is consistent and each negative test can corrupt one cell."""
    base = costs[(costs.penalty == 0) & (costs.share_willing == 0)]
    rt = float(base.cost_stage1_eur.sum())
    rows = []
    for P in PENALTIES:
        for th in THETAS:
            c = costs[np.isclose(costs.penalty, P)
                      & np.isclose(costs.share_willing, th)]
            rows.append(dict(
                penalty=P, share_willing=th, routing_cost_plan1_eur=rt,
                routing_saving_plan1_pct=100 * (rt - c.cost_stage1_eur.sum()) / rt,
                routing_saving_plan2_pct=100 * (rt - c.cost_stage2_eur.sum()) / rt,
                mean_days_plan1_provmean=6.0 - 3.0 * th,
                mean_days_plan2_provmean=6.0 - 2.5 * th,
                mean_days_plan1=6.0 - 3.1 * th, mean_days_plan2=6.0 - 2.6 * th,
                wait_d_plan1=0.5 * th, wait_d_plan2=0.4 * th))
    return pd.DataFrame(rows)


def _cells() -> pd.DataFrame:
    """4 cells x 2 endpoints on the stage-1 plan, plus a stage-2 block the
    builder must ignore."""
    rows = []
    for plan in ("stage1", "balanced"):
        for th, factor in ((0.0, 1.0), (1.0, 0.7)):
            for i, prov in enumerate(PROVS):
                for j, plz in enumerate(("30159", "30167")):
                    cost = (100.0 + 10.0 * i + 20.0 * j) * factor
                    rows.append(dict(penalty=0.0, share_willing=th,
                                     provider=prov, plz=plz, plan=plan,
                                     cell_cost_eur=cost,
                                     cell_parcels_week=50.0 + 10.0 * j))
    return pd.DataFrame(rows)


@pytest.fixture
def frames():
    c = _costs()
    return c, _fleet(c), _grid(c), _cells()


# ─────────────────────────────────────────────────────────────────────────
# mechanism_table
# ─────────────────────────────────────────────────────────────────────────
def test_mechanism_table_shape_and_saving_matches_the_grid(mod, frames):
    costs, fleet, grid, _ = frames
    tab = mod.mechanism_table(costs, fleet, grid)
    assert len(tab) == len(PENALTIES) * len(THETAS)
    m = tab.merge(grid, on=["penalty", "share_willing"])
    assert np.allclose(m.saving_pct, m.routing_saving_plan1_pct)
    assert np.allclose(m.saving_plan2_pct, m.routing_saving_plan2_pct)
    base = tab[(tab.penalty == 0) & (tab.share_willing == 0)].iloc[0]
    assert base.saving_pct == pytest.approx(0.0)
    assert base.mean_days_plan1 == pytest.approx(6.0)


def test_mechanism_table_express_is_nan_only_where_there_are_no_vehicles(mod, frames):
    costs, fleet, grid, _ = frames
    tab = mod.mechanism_table(costs, fleet, grid)
    ends = tab[np.isclose(tab.share_willing, 0.0)
               | np.isclose(tab.share_willing, 1.0)]
    mid = tab[np.isclose(tab.share_willing, 0.5)]
    assert ends.express_eur_per_vd.isna().all()
    assert (ends.express_veh_days == 0).all()
    assert (ends.express_share_pct == 0).all()
    assert mid.express_eur_per_vd.notna().all()
    # 40 * 0.5 * (1 + 0) + 40 * 0.5 * (1 + 1) = 60 EUR over 8 vehicle-days
    assert float(mid.express_eur_per_vd.iloc[0]) == pytest.approx(60.0 / 8.0)


def test_mechanism_table_penalty_columns_separate_the_two_plans(mod, frames):
    costs, fleet, grid, _ = frames
    tab = mod.mechanism_table(costs, fleet, grid)
    r = tab[np.isclose(tab.penalty, 0.25) & np.isclose(tab.share_willing, 1.0)].iloc[0]
    assert r.penalty_keur == pytest.approx(2 * 10.0 * 0.25 * 1.0 / 1000)
    assert r.penalty_plan2_keur == pytest.approx(2 * 12.0 * 0.25 * 1.0 / 1000)


def test_mechanism_table_refuses_when_the_saving_drifts_from_the_grid(mod, frames):
    costs, fleet, grid, _ = frames
    grid = grid.copy()
    grid.loc[grid.index[-1], "routing_saving_plan1_pct"] += 0.001
    with pytest.raises(AssertionError, match="G1"):
        mod.mechanism_table(costs, fleet, grid)


def test_mechanism_table_refuses_express_vehicles_at_full_adoption(mod, frames):
    costs, fleet, grid, _ = frames
    fleet = fleet.copy()
    sel = np.isclose(fleet.share_willing, 1.0)
    fleet.loc[sel, "express_veh"] = 3.0
    with pytest.raises(AssertionError, match="G2"):
        mod.mechanism_table(costs, fleet, grid)


def test_mechanism_table_refuses_a_partial_theta_without_express(mod, frames):
    costs, fleet, grid, _ = frames
    costs = costs.copy()
    costs.loc[np.isclose(costs.share_willing, 0.5), "express_cost_eur"] = 0.0
    with pytest.raises(AssertionError, match="G2"):
        mod.mechanism_table(costs, fleet, grid)


def test_mechanism_table_refuses_a_baseline_that_is_not_daily(mod, frames):
    costs, fleet, grid, _ = frames
    grid = grid.copy()
    sel = (grid.penalty == 0) & (grid.share_willing == 0)
    grid.loc[sel, "mean_days_plan1_provmean"] = 5.9
    with pytest.raises(AssertionError, match="G3"):
        mod.mechanism_table(costs, fleet, grid)


def test_mechanism_table_refuses_a_baseline_cost_mismatch(mod, frames):
    costs, fleet, grid, _ = frames
    grid = grid.copy()
    grid["routing_cost_plan1_eur"] = grid.routing_cost_plan1_eur + 1.0
    with pytest.raises(AssertionError, match="G3"):
        mod.mechanism_table(costs, fleet, grid)


def test_mechanism_table_refuses_a_missing_provider(mod, frames):
    costs, fleet, grid, _ = frames
    costs = costs[~((costs.provider == "P2") & np.isclose(costs.penalty, 0.25)
                    & np.isclose(costs.share_willing, 0.5))]
    with pytest.raises(AssertionError, match="cost rows"):
        mod.mechanism_table(costs, fleet, grid)


# ─────────────────────────────────────────────────────────────────────────
# regular_eur_per_vehicle_day
# ─────────────────────────────────────────────────────────────────────────
def test_regular_eur_per_vehicle_day(mod, frames):
    costs, fleet, _, _ = frames
    # baseline 1000 + 1100 EUR over 2 x 20 vehicle-days
    assert mod.regular_eur_per_vehicle_day(costs, fleet) == pytest.approx(2100 / 40)


def test_regular_eur_per_vehicle_day_refuses_an_empty_baseline_fleet(mod, frames):
    costs, fleet, _, _ = frames
    fleet = fleet.copy()
    fleet.loc[(fleet.penalty == 0) & (fleet.share_willing == 0), "fleet"] = 0.0
    with pytest.raises(AssertionError, match="G3"):
        mod.regular_eur_per_vehicle_day(costs, fleet)


# ─────────────────────────────────────────────────────────────────────────
# saving_per_parcel
# ─────────────────────────────────────────────────────────────────────────
def test_saving_per_parcel_uses_stage1_and_the_baseline_parcel_count(mod, frames):
    _, _, _, cells = frames
    sav = mod.saving_per_parcel(cells)
    assert len(sav) == 4
    r = sav[(sav.provider == "P1") & (sav.plz == "30159")].iloc[0]
    # cost 100 -> 70 over 50 baseline parcels
    assert r.saving_eur_per_parcel == pytest.approx(30.0 / 50.0)
    assert r.cell_parcels_week_base == pytest.approx(50.0)


def test_saving_per_parcel_refuses_an_incomplete_universe(mod, frames):
    _, _, _, cells = frames
    cells = cells[~((cells.provider == "P2") & (cells.plz == "30167")
                    & (cells.plan == "stage1"))]
    with pytest.raises(AssertionError, match="G5"):
        mod.saving_per_parcel(cells)


def test_saving_per_parcel_refuses_a_zero_parcel_baseline(mod, frames):
    _, _, _, cells = frames
    cells = cells.copy()
    sel = ((cells.plan == "stage1") & (cells.share_willing == 0)
           & (cells.provider == "P1") & (cells.plz == "30159"))
    cells.loc[sel, "cell_parcels_week"] = 0.0
    with pytest.raises(AssertionError, match="G5"):
        mod.saving_per_parcel(cells)


# ─────────────────────────────────────────────────────────────────────────
# hist_table
# ─────────────────────────────────────────────────────────────────────────
def _sav(values) -> pd.DataFrame:
    return pd.DataFrame(dict(provider=["P"] * len(values),
                             plz=["30159"] * len(values),
                             cell_parcels_week_base=[1.0] * len(values),
                             saving_eur_per_parcel=values))


def test_hist_table_counts_every_cell_and_marks_the_thresholds(mod):
    tab = mod.hist_table(_sav([0.02, 0.2, 0.3, 0.6, 1.4, -0.03]))
    assert tab.n_cells.sum() == 6
    assert tab.share_pct.sum() == pytest.approx(100.0)
    marked = tab[tab.penalty_threshold.notna()]
    assert sorted(np.round(marked.penalty_threshold, 2)) == [0.25, 0.5, 1.0]


def test_hist_table_cumulative_share_is_the_share_below_the_penalty(mod):
    v = [0.02, 0.2, 0.3, 0.6, 1.4, -0.03]
    tab = mod.hist_table(_sav(v))
    for P, expected in ((0.25, 3 / 6), (0.5, 4 / 6), (1.0, 5 / 6)):
        row = tab[np.isclose(tab.bin_right, P)].iloc[0]
        assert row.cum_share_pct == pytest.approx(100 * expected)


def test_hist_table_refuses_a_bin_width_that_hides_a_threshold(mod):
    with pytest.raises(AssertionError, match="not a bin edge"):
        mod.hist_table(_sav([0.1, 0.4, 0.9]), bin_width=0.07)


def test_hist_table_refuses_non_finite_values(mod):
    with pytest.raises(AssertionError, match="non-finite"):
        mod.hist_table(_sav([0.1, np.nan, 0.9]))
