"""The v2 -> legacy adapter (``scripts/revision/74_v2_to_legacy_tables.py``).

The paper is accepted, so its figures keep the SUBMITTED layout and only
their numbers change: ``30_``/``31_``/``32_`` render them, unchanged, from
the 2026-07 Stage-3 schema.  A v5/v6 grid does not have that schema, so
``74_`` writes it.  Everything here guards the one property that makes that
safe -- **the adapted tables are exactly the schema the frozen builders
read, and nothing in them is invented**:

* every declared schema matches the REAL frozen file, column for column and
  in order (the strongest available check that no legacy column is missing);
* ``assert_schema`` refuses a frame with a missing, extra or reordered
  column, so a half-adapted table can never reach a builder;
* every column declared as having no v5/v6 source is (a) actually in a
  schema and (b) genuinely unread by fig. 4, 5 and 6 -- checked by scanning
  the builders' source, not by assertion;
* both express-allocation modes reconstruct the plan's routing total, and
  the ``per-tour`` mode is the exact inverse of the frozen builder's own
  provider-proportional allocation;
* the environment that re-points the builders is refused once
  ``_stage3_common`` has frozen its constants.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "scripts" / "revision"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REV))


def _load():
    spec = importlib.util.spec_from_file_location(
        "_legacy_adapter_v2", REV / "74_v2_to_legacy_tables.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load()

#: where each frozen file really lives, for the schema cross-check
REAL = {
    "tab_balancing_summary.csv": ROOT / "results" / "runs" / "path2_2026_05_29",
    "tab_chosen_schedules.csv": ROOT / "results" / "runs" / "path2_2026_05_29",
    "_tab_chosen_with_system_smoothing.csv":
        ROOT / "results" / "runs" / "path2_2026_05_29",
    "tab_costs_smoothed.csv": ROOT / "results" / "revision_2026_07",
    "tab_wait_fixed.csv": ROOT / "results" / "revision_2026_07",
    "tab_wait_smoothed.csv": ROOT / "results" / "revision_2026_07",
    "tab_fleet_per_hub_fixed.csv": ROOT / "results" / "revision_2026_07",
    "tab_fleet_per_hub_smoothed.csv": ROOT / "results" / "revision_2026_07",
    "tab_express_smoothed.csv": ROOT / "results" / "revision_2026_07",
    "tab_pstar_knees_smoothed.csv": ROOT / "results" / "revision_2026_07",
    "tab_per_plz_costs_theta1.csv": ROOT / "results" / "revision_2026_07",
}


# ── the declared schema IS the frozen file's schema ──────────────────────
@pytest.mark.parametrize("name", sorted(A.SCHEMA))
def test_declared_schema_matches_the_real_frozen_file(name):
    path = REAL[name] / name
    if not path.exists():
        pytest.skip(f"{path} is not in this checkout")
    have = list(pd.read_csv(path, nrows=0).columns)
    assert A.SCHEMA[name] == have, (
        f"{name}: the adapter would write {A.SCHEMA[name]} but the frozen "
        f"file has {have}")


def test_every_file_is_assigned_to_exactly_one_root():
    assert A.IN_RUN_DIR <= set(A.SCHEMA)
    # the RUN_DIR half is the three Stage-1/Stage-2 tables the builders read
    # from the production run, never from the revision directory
    assert A.IN_RUN_DIR == {"tab_balancing_summary.csv",
                            "tab_chosen_schedules.csv",
                            "_tab_chosen_with_system_smoothing.csv"}


def test_assert_schema_refuses_a_missing_column():
    df = pd.DataFrame({"penalty": [0.0], "share_willing": [1.0]})
    with pytest.raises(AssertionError, match="column mismatch"):
        A.assert_schema("tab_wait_smoothed.csv", df)


def test_assert_schema_refuses_a_reordered_column():
    cols = A.SCHEMA["tab_wait_smoothed.csv"]
    df = pd.DataFrame({c: [0.0] for c in reversed(cols)})
    with pytest.raises(AssertionError, match="column mismatch"):
        A.assert_schema("tab_wait_smoothed.csv", df)


def test_assert_schema_accepts_the_exact_schema():
    cols = A.SCHEMA["tab_wait_smoothed.csv"]
    df = pd.DataFrame({c: [0.0] for c in cols})
    assert list(A.assert_schema("tab_wait_smoothed.csv", df).columns) == cols


# ── the NaN fills are declared, and genuinely unread ─────────────────────
def test_every_no_source_column_is_really_in_a_schema():
    for (fname, col) in A.NO_SOURCE:
        assert fname in A.SCHEMA, fname
        assert col in A.SCHEMA[fname], (fname, col)


@pytest.mark.parametrize("fname,col", sorted(A.NO_SOURCE))
def test_no_figure_reads_a_column_that_has_no_source(fname, col):
    """A guessed column would be bad; an unread one is merely honest.

    Scans the three frozen builders for the column name.  ``day`` is a
    special case: 31_ does read ``tab_express_smoothed.csv``, but only after
    grouping to (penalty, share_willing, provider), so the token is checked
    against the file's own read, not against the whole source.
    """
    if col == "day":
        src = (REV / "31_fig6_structural_smoothed.py").read_text(
            encoding="utf-8")
        assert "tab_express_smoothed" in src
        # the only aggregation of that frame is to provider level
        assert 'groupby(["penalty", "share_willing", "provider"]' in src
        return
    for builder in ("30_fig5_heatmap_smoothed.py",
                    "31_fig6_structural_smoothed.py", "32_fig4_mix.py"):
        src = (REV / builder).read_text(encoding="utf-8")
        assert col not in src, (
            f"{builder} reads {col!r}, which the adapter can only write as "
            "NaN -- that column needs a real source or the figure needs to "
            "stop reading it")


# ── the express allocation ───────────────────────────────────────────────
def _plan_frame() -> pd.DataFrame:
    """One (P, theta, provider) with two cells of unequal size."""
    rows = []
    for plz, w, own, pool, exp in (("30159", 900.0, 700.0, 50.0, 120.0),
                                   ("30161", 100.0, 60.0, 10.0, 8.0)):
        rows.append(dict(penalty=0.0, share_willing=0.5, provider="DHL",
                         plz=plz, plan="balanced", cell_parcels_week=w,
                         own_cost_eur=own, pool_share_eur=pool,
                         express_share_eur=exp,
                         cell_cost_eur=own + pool + exp))
    return pd.DataFrame(rows)


def test_provider_express_share_is_the_builders_own_expression():
    d = _plan_frame()
    s = A.provider_express_share(d)
    total_ex = d.express_share_eur.sum()
    assert s.sum() == pytest.approx(total_ex)
    assert s.iloc[0] == pytest.approx(total_ex * 900.0 / 1000.0)


def test_per_tour_mode_inverts_the_builders_allocation_exactly():
    """dd + (what 31_ adds back) == 72_'s per-cell cost, cell by cell."""
    d = _plan_frame()
    dd = A.dd_cost_column(d, "per-tour")
    rebuilt = dd + A.provider_express_share(d)
    assert np.allclose(rebuilt.values, d.cell_cost_eur.values)


def test_dd_only_mode_is_the_submitted_figures_own_rule():
    d = _plan_frame()
    dd = A.dd_cost_column(d, "dd-only")
    assert np.allclose(dd.values,
                       (d.own_cost_eur + d.pool_share_eur).values)


@pytest.mark.parametrize("mode", ["per-tour", "dd-only"])
def test_both_modes_sum_to_the_plans_routing_total(mode):
    d = _plan_frame()
    total = A.dd_cost_column(d, mode).sum() + d.express_share_eur.sum()
    assert total == pytest.approx(d.cell_cost_eur.sum())


def test_an_unknown_allocation_mode_refuses():
    with pytest.raises(SystemExit, match="unknown express allocation"):
        A.dd_cost_column(_plan_frame(), "whatever")


def test_a_negative_delivery_day_cost_refuses():
    """A cell whose parcel-proportional express share exceeds its own cost
    cannot be reconstructed this way -- say so instead of writing it."""
    d = _plan_frame()
    d.loc[d.plz == "30161", "cell_cost_eur"] = 1.0
    with pytest.raises(AssertionError, match="came out negative"):
        A.dd_cost_column(d, "per-tour")


# ── the grid's own baseline ──────────────────────────────────────────────
def _mini_grid(tmp_path, cost=1000.0, fleet_by_day=(10, 10, 10, 10, 10, 10)):
    rows = []
    for P in (0.0, 0.5):
        for th in (0.0, 1.0):
            rows.append(dict(penalty=P, share_willing=th, provider="DHL",
                             cost_stage2_eur=cost if th == 0 else 0.8 * cost))
    pd.DataFrame(rows).to_csv(tmp_path / "tab_costs_v2.csv", index=False)
    fl = []
    for P in (0.0, 0.5):
        for th in (0.0, 1.0):
            for d, v in enumerate(fleet_by_day):
                fl.append(dict(penalty=P, share_willing=th, provider="DHL",
                               hub="h", day=d, fleet=float(v)))
    pd.DataFrame(fl).to_csv(tmp_path / "tab_fleet_per_hub_v2.csv",
                            index=False)
    return tmp_path


def test_grid_baseline_is_the_grids_own_theta_zero_total(tmp_path):
    b = A.grid_baseline(_mini_grid(tmp_path))
    assert b["base_total_eur"] == pytest.approx(1000.0)
    assert b["baseline_cv_exact"] == pytest.approx(0.0)
    assert b["baseline_peak"] == pytest.approx(10.0)


def test_grid_baseline_cv_is_the_weekday_spread(tmp_path):
    b = A.grid_baseline(_mini_grid(tmp_path, fleet_by_day=(6, 10, 10, 10,
                                                           10, 14)))
    v = np.array([6, 10, 10, 10, 10, 14], dtype=float)
    assert b["baseline_cv_exact"] == pytest.approx(v.std() / v.mean())


def test_grid_baseline_refuses_a_grid_without_a_baseline(tmp_path):
    pd.DataFrame([dict(penalty=0.0, share_willing=1.0, provider="DHL",
                       cost_stage2_eur=1.0)]).to_csv(
        tmp_path / "tab_costs_v2.csv", index=False)
    pd.DataFrame([dict(penalty=0.0, share_willing=1.0, provider="DHL",
                       hub="h", day=0, fleet=1.0)]).to_csv(
        tmp_path / "tab_fleet_per_hub_v2.csv", index=False)
    with pytest.raises(AssertionError, match="no theta=0 rows"):
        A.grid_baseline(tmp_path)


# ── the environment guard ────────────────────────────────────────────────
def test_set_env_refuses_once_stage3_common_has_frozen_its_constants(
        tmp_path, monkeypatch):
    """_stage3_common freezes both roots and both baselines at IMPORT time,
    so setting the environment afterwards would silently render the wrong
    grid through the frozen builders."""
    monkeypatch.setitem(sys.modules, "_stage3_common", object())
    with pytest.raises(AssertionError, match="already imported"):
        A._set_env(tmp_path, dict(base_total_eur=1.0, baseline_cv=0.1))


def test_set_env_exports_every_knob_the_builders_read(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "_stage3_common", raising=False)
    for k in ("REV_DIR", "REV_RUN_DIR", "REV_BASE_TOTAL", "REV_BASELINE_CV",
              "REV_FREQ_INVARIANT"):
        monkeypatch.delenv(k, raising=False)
    A._set_env(tmp_path, dict(base_total_eur=1234.5, baseline_cv=0.139))
    import os
    assert Path(os.environ["REV_DIR"]).name == "rev"
    assert Path(os.environ["REV_RUN_DIR"]).name == "run"
    assert float(os.environ["REV_BASE_TOTAL"]) == pytest.approx(1234.5)
    assert float(os.environ["REV_BASELINE_CV"]) == pytest.approx(0.139)
    # v5/v6 stage 2 is frequency-FREE, so 32_'s invariance claim must NOT
    # be asserted -- and the fig-4 caption must not make it either
    assert os.environ["REV_FREQ_INVARIANT"] == "0"


def test_stage3_common_defaults_are_unchanged_without_the_environment(
        monkeypatch):
    """With no environment set the frozen builders must still reproduce the
    submitted revision figures from results/revision_2026_07."""
    for k in ("REV_DIR", "REV_RUN_DIR", "REV_BASE_TOTAL", "REV_BASELINE_CV",
              "REV_FREQ_INVARIANT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delitem(sys.modules, "_stage3_common", raising=False)
    spec = importlib.util.spec_from_file_location(
        "_stage3_common_probe", REV / "_stage3_common.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    assert C.OUT_DIR.name == "revision_2026_07"
    assert C.RUN_DIR.name == "path2_2026_05_29"
    assert C.BASE_TOTAL == pytest.approx(1909747.75)
    assert C.BASELINE_CV == pytest.approx(0.135)
    assert C.FREQ_INVARIANT is True


# ── the canonical figure stems the sync depends on ───────────────────────
def test_the_canonical_stems_are_the_paper_figure_slots():
    spec = importlib.util.spec_from_file_location(
        "_sync71_probe", REV / "71_sync_paper_figs.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    assert set(A.CANONICAL_FIGURES) == set(S.FIGURE_MAP), (
        "74_ renders one set of stems and 71_ syncs another")
    assert all(s.startswith("supp_") for s in S.COMPANION_MAP), (
        "the two-lens figures are supplementary and must be named so")
    assert not (set(S.FIGURE_MAP) & set(S.COMPANION_MAP))
