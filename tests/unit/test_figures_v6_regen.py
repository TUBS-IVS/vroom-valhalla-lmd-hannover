"""New pure functions added to individual ``scripts/figures/*.py`` scripts
while re-pointing them at v6 (Task 19 W1b). Synthetic frames only -- no
grid, no VROOM, no Docker, no matplotlib rendering of the full figures
(those are exercised by actually running each script against
``results/revision_2026_08_v6`` / the 74_-adapted legacy scratch dir, per
the task report).

Each module is loaded straight off disk (they are ordinary, digit-free
module names, so a plain ``import`` after a ``sys.path`` insert works --
unlike the numbered ``scripts/revision/NN_*.py`` files, which need
``importlib.util``). Importing must not have side effects: every module
under test guards its top-level work behind ``if __name__ == "__main__"``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "scripts" / "figures"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "revision"))
sys.path.insert(0, str(ROOT / "scripts" / "paper"))
sys.path.insert(0, str(FIGURES))

import fig_combined_heatmap as HEAT  # noqa: E402
import _fig_saving_fleet_heatmaps as SFH  # noqa: E402
import _fig_per_provider_sweetspots as SWEET  # noqa: E402


# ── fig_combined_heatmap._fleet_reduction_panels ──────────────────────────
def _sys_day(rows):
    return pd.DataFrame(rows, columns=["penalty", "share_willing", "day",
                                       "fleet"])


def test_fleet_reduction_panels_zero_change_gives_zero_reduction():
    # theta=0 baseline IS the only (P, theta) row -> peak/cv/total
    # reduction against itself must be exactly zero. A realistic (mildly
    # uneven, non-flat) weekday profile, not a degenerate all-equal one --
    # see the dedicated zero-variance-baseline test below for that case.
    profile = [10.0, 11.0, 9.0, 12.0, 10.0, 8.0]
    rows = [(0.0, 0.0, d, v) for d, v in enumerate(profile)]
    sys_day = _sys_day(rows)
    out = HEAT._fleet_reduction_panels(sys_day, "fleet")
    row = out.iloc[0]
    assert row.peak_red == pytest.approx(0.0)
    assert row.total_red == pytest.approx(0.0)
    assert row.cv_red == pytest.approx(0.0)


def test_fleet_reduction_panels_zero_variance_baseline_guarded_not_raised():
    # A perfectly flat baseline profile has base_cv == 0 -- "% CV
    # reduction" against a zero baseline is undefined, guarded to 0.0
    # (same convention already used for a zero-mean `cv` itself), not a
    # ZeroDivisionError.
    rows = [(0.0, 0.0, d, 10.0) for d in range(6)]
    sys_day = _sys_day(rows)
    out = HEAT._fleet_reduction_panels(sys_day, "fleet")
    assert out.iloc[0].cv_red == pytest.approx(0.0)


def test_fleet_reduction_panels_computes_real_peak_and_total_reduction():
    base = [(0.0, 0.0, d, 20.0) for d in range(6)]           # flat baseline
    cell = [(0.5, 1.0, d, v) for d, v in
            enumerate([10, 10, 10, 10, 10, 10])]               # halved
    sys_day = _sys_day(base + cell)
    out = HEAT._fleet_reduction_panels(sys_day, "fleet")
    row = out[np.isclose(out.penalty, 0.5)].iloc[0]
    assert row.peak_red == pytest.approx(50.0)     # 20 -> 10
    assert row.total_red == pytest.approx(50.0)    # 120 -> 60


def test_fleet_reduction_panels_baseline_must_be_theta0():
    # No share_willing==0.0 row at all -> base_day lookup fails loudly
    # rather than silently using some other theta as "baseline".
    sys_day = _sys_day([(0.5, 0.5, d, 10.0) for d in range(6)])
    with pytest.raises(KeyError):
        HEAT._fleet_reduction_panels(sys_day, "fleet")


# ── _fig_saving_fleet_heatmaps.load_agg (NO_SOURCE NaN detection) ────────
def _write_balancing_summary(path, with_fleet_before: bool):
    n = 4
    df = pd.DataFrame({
        "penalty": [0.0, 0.0, 0.5, 0.5],
        "share_willing": [0.0, 1.0, 0.0, 1.0],
        "init_cost_eur": [100.0, 80.0, 100.0, 70.0],
        "balanced_cost_eur": [100.0, 75.0, 100.0, 65.0],
        "max_fleet_before": ([10.0, 10.0, 10.0, 10.0] if with_fleet_before
                             else [np.nan] * n),
        "max_fleet_after": [10.0, 8.0, 10.0, 7.0],
    })
    df.to_csv(path, index=False)


def test_load_agg_keeps_panel_c_when_fleet_before_has_real_values(tmp_path):
    path = tmp_path / "tab_balancing_summary.csv"
    _write_balancing_summary(path, with_fleet_before=True)
    agg, baseline, have_fleet = SFH.load_agg(tmp_path)
    assert have_fleet is True
    assert "fleet_red_pct" in agg.columns
    assert baseline == pytest.approx(100.0)


def test_load_agg_drops_panel_c_when_fleet_before_is_all_nan(tmp_path, capsys):
    # This is exactly 74_'s NO_SOURCE case for
    # tab_balancing_summary.csv::max_fleet_before on v5/v6.
    path = tmp_path / "tab_balancing_summary.csv"
    _write_balancing_summary(path, with_fleet_before=False)
    agg, baseline, have_fleet = SFH.load_agg(tmp_path)
    assert have_fleet is False
    assert "fleet_red_pct" not in agg.columns
    assert "saving_init_pct" in agg.columns and "saving_bal_pct" in agg.columns
    assert "NO_SOURCE" in capsys.readouterr().out


def test_load_agg_fails_loud_on_missing_column(tmp_path):
    path = tmp_path / "tab_balancing_summary.csv"
    pd.DataFrame({"penalty": [0.0], "share_willing": [0.0]}).to_csv(
        path, index=False)
    with pytest.raises(AssertionError, match="missing required column"):
        SFH.load_agg(tmp_path)


# ── _fig_per_provider_sweetspots._load_official_knees ─────────────────────
def test_load_official_knees_returns_none_without_a_rev_sibling(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert SWEET._load_official_knees(run_dir) is None


def test_load_official_knees_reads_the_audited_table(tmp_path):
    run_dir = tmp_path / "out" / "run"
    rev_dir = tmp_path / "out" / "rev"
    run_dir.mkdir(parents=True)
    rev_dir.mkdir(parents=True)
    pd.DataFrame({
        "provider": ["Amazon", "DHL"], "P_star": [0.25, 0.25],
        "saving_pct": [16.6, 4.5], "wait_d": [0.51, 0.16],
        "chord_dist": [0.3, 0.1],
    }).to_csv(rev_dir / "tab_pstar_knees_smoothed.csv", index=False)
    knees = SWEET._load_official_knees(run_dir)
    assert knees is not None
    assert set(knees.provider) == {"Amazon", "DHL"}


def test_load_official_knees_fails_loud_on_missing_column(tmp_path):
    run_dir = tmp_path / "out" / "run"
    rev_dir = tmp_path / "out" / "rev"
    run_dir.mkdir(parents=True)
    rev_dir.mkdir(parents=True)
    pd.DataFrame({"provider": ["Amazon"]}).to_csv(
        rev_dir / "tab_pstar_knees_smoothed.csv", index=False)
    with pytest.raises(AssertionError, match="missing required column"):
        SWEET._load_official_knees(run_dir)
