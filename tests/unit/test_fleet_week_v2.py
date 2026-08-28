"""``fleet_week_table()`` in ``scripts/revision/78_fleet_week_v2.py``.

78_ is the per-LSP counterpart of 75_. It deliberately does NOT re-implement
the expensive profile rebuild -- it loads ``recompute_profiles`` out of 75_ --
so the only new logic is the roll-up into the long CSV plus two gates that
75_ does not have:

* **G6** ties the summed hub peaks of BOTH plans back to the grid's own
  ``sum_hub_peak_plan1`` / ``sum_hub_peak_plan2`` (75_ gates plan 2 only, and
  the routing plan's peak -- 1 666 vs a 1 239 baseline at P = 0 on v6 -- is
  exactly the number this figure exists to show), and the baseline back to
  the grid's (0, 0) row;
* **G7** ties the system stage-2 weekday profile to ``sys_Mon..sys_Sat``.

The fixture is a 2-provider / 2-P synthetic universe with theta = 1; the
module's real ``C.PROVIDERS`` is monkeypatched down to it rather than
parameterised away in the production code. One test also pins the reuse
contract itself (75_ still recomputes at theta = 1 and still exports
``recompute_profiles``), because a silent change there would move this
figure's numbers without touching this file.
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
P_VALUES = (0.0, 0.25)
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_fleet_week_v2", ROOT / "scripts" / "revision" / "78_fleet_week_v2.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


@pytest.fixture(autouse=True)
def _two_providers(mod, monkeypatch):
    monkeypatch.setattr(mod.C, "PROVIDERS", list(PROVS))


# ─────────────────────────────────────────────────────────────────────────
# fixture: 2 providers, 2 P, theta = 1
# ─────────────────────────────────────────────────────────────────────────
def _per_prov() -> dict:
    """Deliberately asymmetric weekdays so a transposed or mis-summed profile
    cannot pass by accident."""
    out = {}
    for i, name in enumerate(PROVS):
        base = np.array([10.0, 11.0, 12.0, 11.0, 10.0, 6.0]) + i
        out[name] = {
            "base": base, "base_peak": 12.0 + i, "n_hubs": 1 + i,
            "P": {
                0.0: {1: base * 1.2, 2: base * 0.9,
                      "peak1": 18.0 + i, "peak2": 9.0 + i},
                0.25: {1: base * 1.1, 2: base * 0.95,
                       "peak1": 15.0 + i, "peak2": 10.0 + i},
            },
        }
    return out


def _grid(per_prov: dict) -> pd.DataFrame:
    """The grid rows the gates read, derived from the fixture so the positive
    path is consistent and each negative test can corrupt one cell."""
    base = sum(per_prov[p]["base"] for p in PROVS)
    rows = [dict(penalty=0.0, share_willing=0.0,
                 sum_hub_peak_plan1=sum(per_prov[p]["base_peak"] for p in PROVS),
                 sum_hub_peak_plan2=sum(per_prov[p]["base_peak"] for p in PROVS),
                 **{f"sys_{d}": base[i] for i, d in enumerate(WEEKDAYS)})]
    for P in P_VALUES:
        p2 = sum(per_prov[p]["P"][P][2] for p in PROVS)
        rows.append(dict(
            penalty=P, share_willing=1.0,
            sum_hub_peak_plan1=sum(per_prov[p]["P"][P]["peak1"] for p in PROVS),
            sum_hub_peak_plan2=sum(per_prov[p]["P"][P]["peak2"] for p in PROVS),
            **{f"sys_{d}": p2[i] for i, d in enumerate(WEEKDAYS)}))
    return pd.DataFrame(rows)


@pytest.fixture
def data():
    pp = _per_prov()
    return pp, _grid(pp)


# ─────────────────────────────────────────────────────────────────────────
# positive path
# ─────────────────────────────────────────────────────────────────────────
def test_table_has_one_row_per_group_P_and_weekday(mod, data):
    pp, grid = data
    tab = mod.fleet_week_table(pp, grid, P_VALUES)
    assert len(tab) == (len(PROVS) + 1) * len(P_VALUES) * 6
    assert set(tab.provider) == set(PROVS) | {mod.SYSTEM_LABEL}
    assert list(tab.weekday.unique()) == WEEKDAYS
    assert (tab.share_willing == 1.0).all()


def test_system_rows_are_the_sum_over_providers(mod, data):
    pp, grid = data
    tab = mod.fleet_week_table(pp, grid, P_VALUES)
    sysr = tab[(tab.provider == mod.SYSTEM_LABEL)
               & np.isclose(tab.penalty, 0.0)].sort_values("day")
    provs = tab[(tab.provider != mod.SYSTEM_LABEL)
                & np.isclose(tab.penalty, 0.0)].groupby("day")
    for col in ("baseline", "plan1", "plan2"):
        assert np.allclose(sysr[col].to_numpy(), provs[col].sum().to_numpy())
    assert sysr.n_lsp.iloc[0] == len(PROVS)
    assert sysr.n_hubs.iloc[0] == sum(pp[p]["n_hubs"] for p in PROVS)


def test_kept_fleet_columns_are_the_summed_hub_peaks(mod, data):
    pp, grid = data
    tab = mod.fleet_week_table(pp, grid, P_VALUES)
    r = tab[(tab.provider == mod.SYSTEM_LABEL) & np.isclose(tab.penalty, 0.0)].iloc[0]
    assert r.peak_baseline == pytest.approx(12.0 + 13.0)
    assert r.peak_plan1 == pytest.approx(18.0 + 19.0)
    assert r.peak_plan2 == pytest.approx(9.0 + 10.0)
    # the kept fleet is constant within a (provider, P) block, not per day
    block = tab[(tab.provider == "P1") & np.isclose(tab.penalty, 0.25)]
    assert block.peak_plan1.nunique() == 1


# ─────────────────────────────────────────────────────────────────────────
# G6 / G7: every identity fails loud, one test per way it can drift
# ─────────────────────────────────────────────────────────────────────────
def test_refuses_when_the_routing_plan_peak_drifts_from_the_grid(mod, data):
    pp, grid = data
    grid = grid.copy()
    sel = np.isclose(grid.penalty, 0.25) & np.isclose(grid.share_willing, 1.0)
    grid.loc[sel, "sum_hub_peak_plan1"] += 1.0
    with pytest.raises(AssertionError, match="G6.*plan1"):
        mod.fleet_week_table(pp, grid, P_VALUES)


def test_refuses_when_the_operator_plan_peak_drifts_from_the_grid(mod, data):
    pp, grid = data
    grid = grid.copy()
    sel = np.isclose(grid.penalty, 0.0) & np.isclose(grid.share_willing, 1.0)
    grid.loc[sel, "sum_hub_peak_plan2"] -= 2.0
    with pytest.raises(AssertionError, match="G6.*plan2"):
        mod.fleet_week_table(pp, grid, P_VALUES)


def test_refuses_when_the_baseline_peak_drifts_from_the_grid(mod, data):
    pp, grid = data
    grid = grid.copy()
    sel = (grid.penalty == 0) & (grid.share_willing == 0)
    grid.loc[sel, "sum_hub_peak_plan1"] += 5.0
    with pytest.raises(AssertionError, match="G6: system baseline"):
        mod.fleet_week_table(pp, grid, P_VALUES)


def test_refuses_when_the_system_weekday_profile_drifts_from_the_grid(mod, data):
    pp, grid = data
    grid = grid.copy()
    sel = np.isclose(grid.penalty, 0.25) & np.isclose(grid.share_willing, 1.0)
    grid.loc[sel, "sys_Thu"] += 0.5
    with pytest.raises(AssertionError, match="G7"):
        mod.fleet_week_table(pp, grid, P_VALUES)


def test_refuses_a_missing_grid_row(mod, data):
    pp, grid = data
    grid = grid[~(np.isclose(grid.penalty, 0.25)
                  & np.isclose(grid.share_willing, 1.0))]
    with pytest.raises(AssertionError, match="G6: no unique grid row"):
        mod.fleet_week_table(pp, grid, P_VALUES)


def test_refuses_a_provider_missing_from_the_recomputation(mod, data):
    pp, grid = data
    pp = {k: v for k, v in pp.items() if k != "P2"}
    with pytest.raises(KeyError):
        mod.fleet_week_table(pp, grid, P_VALUES)


# ─────────────────────────────────────────────────────────────────────────
# naming + the reuse contract with 75_
# ─────────────────────────────────────────────────────────────────────────
def test_stem_for_matches_the_brief_naming(mod):
    assert mod.stem_for(0.0) == "supp_fig_fleet_week_v2_P0"
    assert mod.stem_for(0.25) == "supp_fig_fleet_week_v2_P025"


def test_75_is_the_single_source_of_the_profile_rebuild(mod):
    """78_ must not grow its own copy of the fleet counter: it loads 75_'s
    ``recompute_profiles`` and 75_ must still recompute at theta = 1."""
    src = (ROOT / "scripts" / "revision" / "78_fleet_week_v2.py").read_text(
        encoding="utf-8")
    assert "_daily_fleet_per_hub" not in src
    assert "build_cost_matrices_ml" not in src
    seventy_five = mod.load_75()
    assert seventy_five.THETA == 1.0
    assert callable(seventy_five.recompute_profiles)
