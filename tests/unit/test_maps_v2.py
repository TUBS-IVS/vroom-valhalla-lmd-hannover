"""Pure data-shaping functions of ``scripts/revision/76_maps_v2.py``.

76_ re-renders the v6 supplementary spatial figures (frequency, saving, wait
maps + the penalty-by-settlement-type curve) from ``tab_per_cell_costs_v2.csv``
and ``tab_grid_full_v2.csv``. Its ``at()`` helper hard-asserts the real
312-cell v6 universe, which a small synthetic grid cannot satisfy -- these
tests monkeypatch it to the fixture's own (2 providers x 3 cells = 6) count
instead, keeping the same *kind* of completeness gate at a testable scale.

Covers the four pure functions carved out of the figure builders (13C):

* ``wmedian``               -- the weighted-median primitive itself
* ``freq_table``            -- per-area parcel-weighted median frequency
* ``saving_draw_range``     -- the clip-to-0 rule and its -0.5 pp threshold
* ``saving_table``          -- per-area saving, gated against the grid total
* ``penalty_raumtyp_table`` -- euro-weighted saving by settlement type

No plotting, no file I/O, no geodata: these are exactly the functions that
build the DataFrames the figures then paint.
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


def _load():
    spec = importlib.util.spec_from_file_location(
        "_maps_v2", ROOT / "scripts" / "revision" / "76_maps_v2.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


@pytest.fixture(autouse=True)
def _small_universe(mod, monkeypatch):
    """Replace the 312-cell production gate with the fixture's own count.

    ``at()`` filters by (P, theta) and then hard-asserts the real v6 cell
    count; a 2-provider/3-cell synthetic grid has 6 rows per slice instead.
    The filtering behaviour (and the completeness *idea*) is unchanged --
    only the expected count is fixture-sized.
    """
    def _at(cells, P, th):
        sub = cells[np.isclose(cells.penalty, P) & np.isclose(cells.share_willing, th)]
        assert len(sub) == 6, f"fixture: {len(sub)} cells at P={P}, theta={th}"
        return sub
    monkeypatch.setattr(mod, "at", _at)


# ─────────────────────────────────────────────────────────────────────────
# wmedian -- the weighted-median primitive
# ─────────────────────────────────────────────────────────────────────────
def test_wmedian_equal_weights(mod):
    assert mod.wmedian(np.array([6.0, 2.0, 4.0]), np.array([1.0, 1.0, 1.0])) == 4.0


def test_wmedian_is_parcel_weighted_not_a_plain_median(mod):
    # the plain median of {2, 6} is 4; parcel weight 99:1 must instead pick
    # the value carried by almost all the parcels.
    assert mod.wmedian(np.array([2.0, 6.0]), np.array([1.0, 99.0])) == 6.0


def test_wmedian_never_interpolates_to_a_half_size(mod):
    # an exact 50/50 split must still resolve to one of the real schedule
    # sizes (2 or 4), never a fictitious "3".
    v = mod.wmedian(np.array([2.0, 4.0]), np.array([50.0, 50.0]))
    assert v in (2.0, 4.0)


# ─────────────────────────────────────────────────────────────────────────
# freq_table -- per-area parcel-weighted median delivery frequency
# ─────────────────────────────────────────────────────────────────────────
def _freq_cells() -> pd.DataFrame:
    """2 providers x 3 cells x 1 P x 2 theta; theta=0 forced all-daily."""
    rows = []
    for plz in ("10001", "10002", "10003"):
        for prov in ("A", "B"):
            rows.append(dict(provider=prov, plz=plz, penalty=0.0, share_willing=0.0,
                             mean_days=6.0, cell_parcels_week=10.0))
    # theta = 1: hand-picked so the parcel-weighted median differs from a
    # plain (unweighted) median in two of the three cells.
    theta1 = {
        "10001": [("A", 2.0, 100.0), ("B", 4.0, 10.0)],   # heavy A -> 2
        "10002": [("A", 3.0, 50.0), ("B", 3.0, 50.0)],    # tied -> 3
        "10003": [("A", 6.0, 1.0), ("B", 2.0, 99.0)],     # heavy B -> 2
    }
    for plz, entries in theta1.items():
        for prov, md, parcels in entries:
            rows.append(dict(provider=prov, plz=plz, penalty=0.0, share_willing=1.0,
                             mean_days=md, cell_parcels_week=parcels))
    return pd.DataFrame(rows)


def test_freq_table_theta0_is_all_daily(mod):
    tab = mod.freq_table(_freq_cells(), 0.0, (0.0, 1.0))
    t0 = tab[tab.share_willing == 0.0]
    assert len(t0) == 3
    assert (t0.freq == 6.0).all()


def test_freq_table_theta1_parcel_weighted_median_per_area(mod):
    tab = mod.freq_table(_freq_cells(), 0.0, (0.0, 1.0))
    t1 = tab[tab.share_willing == 1.0].set_index("unit")
    assert t1.loc["10001", "freq"] == 2.0
    assert t1.loc["10002", "freq"] == 3.0
    assert t1.loc["10003", "freq"] == 2.0
    assert (t1["n_lsp"] == 2).all()


def test_freq_table_theta0_violation_fails_loud(mod):
    cells = _freq_cells()
    cells.loc[(cells.share_willing == 0.0) & (cells.provider == "A")
              & (cells.plz == "10001"), "mean_days"] = 5.0
    with pytest.raises(AssertionError, match="not all-daily"):
        mod.freq_table(cells, 0.0, (0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────
# saving_draw_range -- the clip-to-0 rule and its -0.5 pp threshold
# ─────────────────────────────────────────────────────────────────────────
def test_saving_draw_range_all_positive_no_note(mod):
    tab = pd.DataFrame({"saving_pct": [5.0, 10.0, 20.0]})
    vmin, vmax, diverging, note, drawn = mod.saving_draw_range(tab)
    assert (vmin, vmax, diverging, note) == (0.0, 20.0, False, "")
    assert list(drawn) == [5.0, 10.0, 20.0]


def test_saving_draw_range_noise_negative_is_clipped_not_diverging(mod):
    # -0.2 pp is attribution noise (72_): clip to 0, keep the linear ramp,
    # and disclose it in the footnote note.
    tab = pd.DataFrame({"saving_pct": [-0.2, 5.0, 10.0]})
    vmin, vmax, diverging, note, drawn = mod.saving_draw_range(tab)
    assert diverging is False
    assert vmin == 0.0
    assert "1 area(s) at -0.2 %" in note
    assert list(drawn) == [0.0, 5.0, 10.0]


def test_saving_draw_range_material_negative_is_diverging(mod):
    # -3.2 pp is a real loss: switch to the diverging ramp, floor vmin to
    # the next 5-multiple, and do NOT clip the value away.
    tab = pd.DataFrame({"saving_pct": [-3.2, 5.0, 22.0]})
    vmin, vmax, diverging, note, drawn = mod.saving_draw_range(tab)
    assert diverging is True
    assert vmin == -5.0
    assert vmax == 25.0
    assert note == ""
    assert list(drawn) == [-3.2, 5.0, 22.0]


def test_saving_draw_range_boundary_is_not_diverging(mod):
    # exactly -0.5 pp: the rule is a strict "<", so the boundary itself
    # still counts as noise, not a material loss.
    tab = pd.DataFrame({"saving_pct": [-0.5, 5.0]})
    vmin, vmax, diverging, note, drawn = mod.saving_draw_range(tab)
    assert diverging is False
    assert vmin == 0.0
    assert "-0.5 %" in note


def test_saving_draw_range_threshold_is_a_real_parameter(mod):
    # the same -0.7 pp value flips sides depending on noise_pp: proves the
    # threshold is load-bearing, not a hardcoded -0.5 baked into the branch.
    tab = pd.DataFrame({"saving_pct": [-0.7, 5.0]})
    _, _, diverging_default, _, _ = mod.saving_draw_range(tab)
    _, _, diverging_wide, _, _ = mod.saving_draw_range(tab, noise_pp=1.0)
    assert diverging_default is True
    assert diverging_wide is False


# ─────────────────────────────────────────────────────────────────────────
# saving_table -- per-area saving, gated against the grid's own total
# ─────────────────────────────────────────────────────────────────────────
def _saving_fixture():
    """2 providers x 3 cells x 2 P at theta=1, base costs + a matching grid."""
    base = pd.DataFrame([
        dict(provider="A", plz="10001", base_cost_eur=100.0),
        dict(provider="A", plz="10002", base_cost_eur=200.0),
        dict(provider="A", plz="10003", base_cost_eur=50.0),
        dict(provider="B", plz="10001", base_cost_eur=80.0),
        dict(provider="B", plz="10002", base_cost_eur=120.0),
        dict(provider="B", plz="10003", base_cost_eur=40.0),
    ])
    # P = 0.0: two areas (A/10003, B/10003) come out slightly ABOVE base
    # cost -- saving_table must not suppress that; only saving_draw_range
    # (tested above) makes a drawing decision about negatives.
    plan_p0 = {("A", "10001"): 80.0, ("A", "10002"): 150.0, ("A", "10003"): 51.0,
               ("B", "10001"): 70.0, ("B", "10002"): 100.0, ("B", "10003"): 42.0}
    # P = 0.5: every area saves.
    plan_p5 = {("A", "10001"): 70.0, ("A", "10002"): 140.0, ("A", "10003"): 45.0,
               ("B", "10001"): 60.0, ("B", "10002"): 90.0, ("B", "10003"): 35.0}
    rows = []
    for P, plan in ((0.0, plan_p0), (0.5, plan_p5)):
        for (prov, plz), cost in plan.items():
            rows.append(dict(provider=prov, plz=plz, penalty=P, share_willing=1.0,
                             cell_cost_eur=cost))
    cells = pd.DataFrame(rows)

    base_total = base.base_cost_eur.sum()
    grid_rows = []
    for P, plan in ((0.0, plan_p0), (0.5, plan_p5)):
        plan_total = sum(plan.values())
        sysv = (1 - plan_total / base_total) * 100
        grid_rows.append(dict(penalty=P, share_willing=1.0, routing_saving_plan2_pct=sysv))
    grid = pd.DataFrame(grid_rows)
    return cells, base, grid


def test_saving_table_matches_independent_groupby(mod):
    cells, base, grid = _saving_fixture()
    tab = mod.saving_table(cells, base, (0.0, 0.5), grid)

    merged = cells.merge(base, on=["provider", "plz"])
    for P in (0.0, 0.5):
        sub = merged[merged.penalty == P]
        expect = sub.groupby("plz").apply(
            lambda g: (1 - g.cell_cost_eur.sum() / g.base_cost_eur.sum()) * 100,
            include_groups=False)
        got = tab[np.isclose(tab.penalty, P)].set_index("unit")["saving_pct"]
        for plz, v in expect.items():
            assert got[plz] == pytest.approx(v)


def test_saving_table_system_value_matches_grid(mod):
    cells, base, grid = _saving_fixture()
    tab = mod.saving_table(cells, base, (0.0, 0.5), grid)
    for P in (0.0, 0.5):
        expected = grid.loc[np.isclose(grid.penalty, P), "routing_saving_plan2_pct"].iloc[0]
        got = tab.loc[np.isclose(tab.penalty, P), "system_saving_pct"].unique()
        assert list(got) == pytest.approx([expected])


def test_saving_table_has_a_negative_area_at_p0(mod):
    # sanity: the fixture actually exercises the "area worse than baseline"
    # case saving_draw_range is built to handle downstream.
    cells, base, grid = _saving_fixture()
    tab = mod.saving_table(cells, base, (0.0, 0.5), grid)
    p0 = tab[np.isclose(tab.penalty, 0.0)]
    assert (p0.saving_pct < 0).any()


def test_saving_table_fails_loud_on_grid_mismatch(mod):
    cells, base, grid = _saving_fixture()
    grid.loc[np.isclose(grid.penalty, 0.0), "routing_saving_plan2_pct"] = 999.0
    with pytest.raises(AssertionError, match="cell sum"):
        mod.saving_table(cells, base, (0.0, 0.5), grid)


# ─────────────────────────────────────────────────────────────────────────
# penalty_raumtyp_table -- euro-weighted saving by settlement type
# ─────────────────────────────────────────────────────────────────────────
def _raumtyp():
    return pd.DataFrame({"plz": ["10001", "10002", "10003"],
                         "raumtyp_3": ["urban", "suburban", "rural"]})


def test_penalty_raumtyp_table_groups_and_system_row(mod):
    cells, base, grid = _saving_fixture()
    tab = mod.penalty_raumtyp_table(cells, base, grid, _raumtyp())

    p0 = tab[np.isclose(tab.penalty, 0.0)].set_index("raumtyp_3")
    assert set(p0.index) == {"urban", "suburban", "rural", "system"}
    # each settlement type here is exactly one (provider-summed) cell.
    assert p0.loc["urban", "base_eur"] == pytest.approx(180.0)     # 100 + 80
    assert p0.loc["urban", "plan_eur"] == pytest.approx(150.0)     # 80 + 70
    assert p0.loc["urban", "n_cells"] == 2
    expected_sys = grid.loc[np.isclose(grid.penalty, 0.0),
                            "routing_saving_plan2_pct"].iloc[0]
    assert p0.loc["system", "saving_pct"] == pytest.approx(expected_sys)


def test_penalty_raumtyp_table_missing_type_fails_loud(mod):
    cells, base, grid = _saving_fixture()
    rt = _raumtyp()
    rt = rt[rt.plz != "10003"]   # drop one cell's settlement type
    with pytest.raises(AssertionError, match="settlement type missing"):
        mod.penalty_raumtyp_table(cells, base, grid, rt)


def test_penalty_raumtyp_table_fails_loud_on_grid_mismatch(mod):
    cells, base, grid = _saving_fixture()
    grid.loc[np.isclose(grid.penalty, 0.5), "routing_saving_plan2_pct"] = -123.0
    with pytest.raises(AssertionError):
        mod.penalty_raumtyp_table(cells, base, grid, _raumtyp())
