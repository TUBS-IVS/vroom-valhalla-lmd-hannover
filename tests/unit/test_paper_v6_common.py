"""Schema guards for scripts/paper/_paper_v6_common.py (Task 19, wave W1a).

The shared helper every re-pointed scripts/paper/ legacy figure script now
uses to (a) refuse to silently zero-sum a legacy column v6 has no source
for, (b) read the one provider-level fleet before/after quantity v6-native
tables DO carry even though the legacy adapter leaves it NaN, and (c) stamp
a reproducible provenance footer + pinned metadata on every regenerated
figure.  All on synthetic frames -- no grid, no VROOM, no real results/ data.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "scripts" / "paper"


def _load():
    sys.path.insert(0, str(PAPER))
    spec = importlib.util.spec_from_file_location(
        "_paper_v6_common", PAPER / "_paper_v6_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load()


# ─────────────────────────────────────────────────────────────────────────
# assert_has_data
# ─────────────────────────────────────────────────────────────────────────
def test_assert_has_data_passes_when_some_values_present():
    df = pd.DataFrame({"max_fleet_after": [1.0, 2.0, np.nan]})
    C.assert_has_data(df, "max_fleet_after", context="t")  # no raise


def test_assert_has_data_raises_on_all_nan_column():
    """The exact failure mode this guard exists for: tab_balancing_summary's
    max_fleet_before is NaN for every v6 row (74_'s NO_SOURCE), and
    pandas' default skipna=True would otherwise turn .sum() into a silent
    0.0 instead of surfacing the missing source."""
    df = pd.DataFrame({"max_fleet_before": [np.nan, np.nan, np.nan]})
    with pytest.raises(C.NoV6Source, match="max_fleet_before"):
        C.assert_has_data(df, "max_fleet_before", context="FB6 heatmap")
    # and prove the failure mode is real: naive code would not have caught it
    assert df["max_fleet_before"].sum() == 0.0


def test_assert_has_data_raises_on_missing_column():
    df = pd.DataFrame({"other": [1, 2, 3]})
    with pytest.raises(C.NoV6Source, match="not in this frame"):
        C.assert_has_data(df, "max_fleet_before", context="t")


def test_assert_has_data_empty_frame_is_not_silently_ok():
    """An empty selection (e.g. a (P, theta) cell with no rows) must not be
    mistaken for 'has data' -- .isna().all() on an empty Series is True."""
    df = pd.DataFrame({"max_fleet_before": pd.Series([], dtype=float)})
    with pytest.raises(C.NoV6Source):
        C.assert_has_data(df, "max_fleet_before", context="t")


# ─────────────────────────────────────────────────────────────────────────
# load_fleet_before_after
# ─────────────────────────────────────────────────────────────────────────
def _synthetic_costs_v2(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "penalty": [0.0, 0.0, 0.5, 0.5],
        "share_willing": [0.0, 1.0, 0.0, 1.0],
        "provider": ["DHL", "DHL", "DHL", "DHL"],
        "sum_hub_peak_before": [100.0, 100.0, 100.0, 100.0],
        "sum_hub_peak_after": [100.0, 70.0, 100.0, 60.0],
        "vehicle_days_before": [500.0, 500.0, 500.0, 500.0],
        "vehicle_days": [500.0, 480.0, 500.0, 470.0],
        "sum_hub_peak": [100.0, 70.0, 100.0, 60.0],  # decoy column, unrelated
    })
    df.to_csv(tmp_path / "tab_costs_v2.csv", index=False)
    return tmp_path


def test_load_fleet_before_after_reads_provider_level_totals(tmp_path):
    rev = _synthetic_costs_v2(tmp_path)
    out = C.load_fleet_before_after(rev)
    assert list(out.columns) == [
        "penalty", "share_willing", "provider",
        "sum_hub_peak_before", "sum_hub_peak_after",
        "vehicle_days_before", "vehicle_days_after",
    ]
    row = out[(out.penalty == 0.5) & (out.share_willing == 1.0)].iloc[0]
    assert row.sum_hub_peak_before == 100.0
    assert row.sum_hub_peak_after == 60.0
    assert row.vehicle_days_after == 470.0


def test_load_fleet_before_after_fails_loud_on_missing_columns(tmp_path):
    pd.DataFrame({"penalty": [0.0], "share_willing": [0.0],
                  "provider": ["DHL"]}).to_csv(
        tmp_path / "tab_costs_v2.csv", index=False)
    with pytest.raises(AssertionError, match="not a v6-schema grid"):
        C.load_fleet_before_after(tmp_path)


# ─────────────────────────────────────────────────────────────────────────
# provenance footer + pinned metadata
# ─────────────────────────────────────────────────────────────────────────
def test_provenance_text_format():
    txt = C.provenance_text(plan="operator-polished (balanced)",
                            script="paper_sweet_spot_math.py",
                            source="tab_costs_smoothed.csv")
    assert txt == ("v6 · operator-polished (balanced) · "
                   "paper_sweet_spot_math.py · tab_costs_smoothed.csv")


def test_add_provenance_footer_writes_expected_text_to_figure():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    C.add_provenance_footer(fig, plan="n/a", script="x.py", source="y.csv")
    texts = [t.get_text() for t in fig.texts]
    assert "v6 · n/a · x.py · y.csv" in texts
    plt.close(fig)


def test_pdf_png_meta_have_no_variable_fields():
    """CreationDate/Software are the two fields matplotlib otherwise stamps
    with the render wall-clock/library version; pinning them to None is
    what makes two regenerations on the same data byte-comparable."""
    assert C.PDF_META == {"CreationDate": None}
    assert C.PNG_META == {"Software": None}


def test_savefig_pair_creates_parent_dirs_and_both_files(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    png = tmp_path / "nested" / "dir" / "fig.png"
    pdf = tmp_path / "nested" / "dir" / "fig.pdf"
    C.savefig_pair(fig, png, pdf)
    assert png.exists() and pdf.exists()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# build_penalty_series
# ─────────────────────────────────────────────────────────────────────────
def _synthetic_legacy_rev(tmp_path: Path) -> Path:
    """Two penalties x two thetas (incl. theta=0 baseline), one provider."""
    costs = pd.DataFrame({
        "penalty": [0.0, 0.0, 0.5, 0.5],
        "share_willing": [0.0, 1.0, 0.0, 1.0],
        "provider": ["DHL"] * 4,
        "dd_cost_stage3_eur": [800.0, 700.0, 800.0, 750.0],
        "express_stage3_eur": [200.0, 150.0, 200.0, 180.0],
        "total_stage3_eur": [1000.0, 850.0, 1000.0, 930.0],
    })
    wait = pd.DataFrame({
        "penalty": [0.0, 0.5],
        "share_willing": [1.0, 1.0],
        "avg_wait_d_stage3": [0.5, 0.2],
    })
    costs.to_csv(tmp_path / "tab_costs_smoothed.csv", index=False)
    wait.to_csv(tmp_path / "tab_wait_smoothed.csv", index=False)
    return tmp_path


def test_build_penalty_series_saving_and_wait(tmp_path):
    rev = _synthetic_legacy_rev(tmp_path)
    out = C.build_penalty_series(rev, share=1.0)
    assert list(out.columns) == ["penalty", "saving_pct", "avg_wait"]
    assert list(out.penalty) == [0.0, 0.5]
    # baseline (theta=0) system total is 1000; theta=1 totals are 850/930
    assert out.saving_pct.iloc[0] == pytest.approx(15.0)
    assert out.saving_pct.iloc[1] == pytest.approx(7.0)
    assert out.avg_wait.iloc[0] == pytest.approx(0.5)
    assert out.avg_wait.iloc[1] == pytest.approx(0.2)


def test_build_penalty_series_rejects_nonconstant_baseline(tmp_path):
    """theta=0 must be daily (same cost) at every P; a grid where it drifts
    is not one this derivation can trust -- fail loud, not average it away."""
    costs = pd.DataFrame({
        "penalty": [0.0, 0.5],
        "share_willing": [0.0, 0.0],
        "provider": ["DHL", "DHL"],
        "dd_cost_stage3_eur": [800.0, 900.0],
        "express_stage3_eur": [200.0, 200.0],
        "total_stage3_eur": [1000.0, 1100.0],
    })
    costs.to_csv(tmp_path / "tab_costs_smoothed.csv", index=False)
    pd.DataFrame({"penalty": [0.0, 0.5], "share_willing": [1.0, 1.0],
                  "avg_wait_d_stage3": [0.5, 0.2]}).to_csv(
        tmp_path / "tab_wait_smoothed.csv", index=False)
    with pytest.raises(AssertionError, match="not constant across P"):
        C.build_penalty_series(tmp_path, share=1.0)


def test_build_penalty_series_fails_loud_on_missing_share(tmp_path):
    """share=0.3 exists in neither synthetic table -- must raise, not
    silently return an empty frame (an empty selection's .notna().all() is
    vacuously True, which is exactly the bug this guard exists to catch)."""
    rev = _synthetic_legacy_rev(tmp_path)
    with pytest.raises(AssertionError, match="no share_willing=0.3 rows"):
        C.build_penalty_series(rev, share=0.3)


# ─────────────────────────────────────────────────────────────────────────
# run_legacy_adapter idempotency (no subprocess actually invoked here)
# ─────────────────────────────────────────────────────────────────────────
def test_run_legacy_adapter_is_a_noop_when_output_already_present(
        tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "tab_chosen_schedules.csv").write_text("penalty\n0.0\n")

    def _boom(*a, **k):
        raise AssertionError("subprocess.run should not be called "
                             "when the legacy tables already exist")

    monkeypatch.setattr(C.subprocess, "run", _boom)
    run, rev = C.run_legacy_adapter(tmp_path / "rev_dir_unused", tmp_path)
    assert run == run_dir
    assert rev == tmp_path / "rev"
