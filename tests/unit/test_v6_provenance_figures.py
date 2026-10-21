"""``scripts/figures/_v6_provenance.py`` (Task 19 W1b).

Shared plumbing every regenerated ``scripts/figures/*.py`` script (status
A/B/D) uses to point at v6 data, stamp a provenance footer, pin figure
metadata, and fail loud on a missing or NO_SOURCE (all-NaN) legacy column.
These are pure functions on synthetic frames -- no grid, no VROOM, no
Docker.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

import _v6_provenance as V  # noqa: E402


# ── chord_knee ────────────────────────────────────────────────────────────
def test_chord_knee_picks_the_geometric_elbow():
    # A textbook diminishing-returns curve: saving climbs fast at first,
    # then flattens, while wait grows ~linearly -- the knee should land
    # where the gap (saving_n - wait_n) is largest, i.e. index 1.
    wait = np.array([0.0, 0.3, 0.6, 1.0])
    saving = np.array([0.0, 18.0, 22.0, 24.0])
    assert V.chord_knee(wait, saving) == 1


def test_chord_knee_matches_lsp_knees_formula_directly():
    # Reproduce _figs_tables_v2.lsp_knees' own normalisation + argmax
    # inline, so a change to chord_knee's formula cannot silently drift
    # from the audited per-LSP knee method.
    wait = np.array([0.9, 0.4, 0.2, 0.1, 0.05])
    saving = np.array([22.6, 18.7, 13.5, 10.1, 7.6])
    w_n = (wait - wait.min()) / (wait.max() - wait.min())
    s_n = (saving - saving.min()) / (saving.max() - saving.min())
    expect = int(np.argmax((s_n - w_n) / np.sqrt(2)))  # scale is irrelevant
    assert V.chord_knee(wait, saving) == expect


def test_chord_knee_refuses_fewer_than_three_points():
    with pytest.raises(AssertionError, match="need >=3"):
        V.chord_knee([0.0, 1.0], [0.0, 10.0])


def test_chord_knee_refuses_mismatched_lengths():
    with pytest.raises(AssertionError, match="need >=3"):
        V.chord_knee([0.0, 1.0, 2.0], [0.0, 10.0])


# ── require_columns / require_nonnull (fail-loud schema guards) ──────────
def test_require_columns_passes_when_all_present():
    df = pd.DataFrame({"penalty": [0.0], "share_willing": [0.0]})
    V.require_columns(df, ["penalty", "share_willing"], source="unit-test")


def test_require_columns_fails_loud_on_missing_column():
    df = pd.DataFrame({"penalty": [0.0]})
    with pytest.raises(AssertionError, match="missing required column"):
        V.require_columns(df, ["penalty", "share_willing"], source="tab_x")


def test_require_columns_reports_every_missing_name():
    df = pd.DataFrame({"penalty": [0.0]})
    with pytest.raises(AssertionError) as exc:
        V.require_columns(df, ["penalty", "a", "b"], source="tab_x")
    assert "'a'" in str(exc.value) and "'b'" in str(exc.value)


def test_require_nonnull_passes_with_some_real_values():
    df = pd.DataFrame({"max_fleet_after": [1.0, np.nan, 3.0]})
    V.require_nonnull(df, "max_fleet_after", source="tab_balancing_summary")


def test_require_nonnull_fails_loud_on_all_nan_no_source_column():
    # tab_balancing_summary.csv::max_fleet_before is exactly this case on
    # v5/v6: the column exists (assert_schema passes) but every value is
    # NaN because the grid never wrote a stage-1 per-hub-day fleet.
    df = pd.DataFrame({"max_fleet_before": [np.nan, np.nan, np.nan]})
    with pytest.raises(AssertionError, match="entirely NaN"):
        V.require_nonnull(df, "max_fleet_before",
                          source="tab_balancing_summary")


def test_require_nonnull_fails_loud_if_column_absent():
    df = pd.DataFrame({"other": [1.0]})
    with pytest.raises(AssertionError, match="not present at all"):
        V.require_nonnull(df, "max_fleet_before", source="tab_balancing_summary")


# ── add_v6_args ────────────────────────────────────────────────────────────
def test_add_v6_args_defaults_preserve_original_paths():
    ap = argparse.ArgumentParser()
    V.add_v6_args(ap, default_rev="results/overnight_2026_05_29_path2",
                  default_out="results/EWGT_Results", rev_help="x")
    args = ap.parse_args([])
    assert args.rev_dir == "results/overnight_2026_05_29_path2"
    assert args.out_dir == "results/EWGT_Results"


def test_add_v6_args_accepts_overrides():
    ap = argparse.ArgumentParser()
    V.add_v6_args(ap, default_rev="old", default_out="old_out", rev_help="x")
    args = ap.parse_args(["--rev-dir", "results/revision_2026_08_v6",
                          "--out-dir", "results/new"])
    assert args.rev_dir == "results/revision_2026_08_v6"
    assert args.out_dir == "results/new"


# ── legacy_baseline ────────────────────────────────────────────────────────
def test_legacy_baseline_reads_the_grids_own_manifest(tmp_path):
    manifest = {"base_total_eur": 1898090.80, "baseline_cv": 0.139,
                "head_id": "bundle_head@test"}
    (tmp_path / "legacy_manifest.json").write_text(json.dumps(manifest),
                                                    encoding="utf-8")
    out = V.legacy_baseline(tmp_path)
    assert out["base_total_eur"] == pytest.approx(1898090.80)
    assert out["head_id"] == "bundle_head@test"


def test_legacy_baseline_fails_loud_when_manifest_missing(tmp_path):
    with pytest.raises(AssertionError, match="missing -- run"):
        V.legacy_baseline(tmp_path / "does_not_exist")


# ── base_total_with_path2_fallback ─────────────────────────────────────────
def test_base_total_prefers_the_legacy_manifest_when_present(tmp_path):
    # rev is <out>/run; the manifest sits at <out>/legacy_manifest.json.
    out = tmp_path / "legacy_out"
    run_dir = out / "run"
    run_dir.mkdir(parents=True)
    (out / "legacy_manifest.json").write_text(
        json.dumps({"base_total_eur": 1898090.80}), encoding="utf-8")
    # A tab_balancing_summary.csv with a DIFFERENT total is also present, to
    # prove the manifest wins over recomputing from the table.
    pd.DataFrame({"penalty": [0.0], "share_willing": [0.0],
                 "balanced_cost_eur": [999999.0]}).to_csv(
        run_dir / "tab_balancing_summary.csv", index=False)
    assert V.base_total_with_path2_fallback(run_dir) == pytest.approx(1898090.80)


def test_base_total_falls_back_to_theta0_row_without_a_manifest(tmp_path):
    run_dir = tmp_path / "path2_run"  # no legacy_manifest.json in .parent
    run_dir.mkdir()
    pd.DataFrame({
        "penalty": [0.0, 0.0, 0.25, 0.25],
        "share_willing": [0.0, 1.0, 0.0, 1.0],
        "balanced_cost_eur": [1909747.75, 1500000.0, 1909747.75, 1400000.0],
    }).to_csv(run_dir / "tab_balancing_summary.csv", index=False)
    assert V.base_total_with_path2_fallback(run_dir) == pytest.approx(1909747.75)


def test_base_total_fails_loud_without_theta0_rows(tmp_path):
    run_dir = tmp_path / "path2_run"
    run_dir.mkdir()
    pd.DataFrame({"penalty": [0.0], "share_willing": [1.0],
                 "balanced_cost_eur": [1500000.0]}).to_csv(
        run_dir / "tab_balancing_summary.csv", index=False)
    with pytest.raises(AssertionError, match="no theta=0 rows"):
        V.base_total_with_path2_fallback(run_dir)


# ── footer / savefig_pinned (matplotlib smoke tests) ──────────────────────
def test_footer_and_savefig_pinned_smoke(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    V.footer(fig, plan=V.PLAN2, script="test_script.py",
             source="tab_costs_v2.csv")
    written = V.savefig_pinned(fig, tmp_path, "smoke_fig")
    plt.close(fig)
    assert len(written) == 2
    for p in written:
        assert p.exists() and p.stat().st_size > 0


def test_savefig_pinned_is_byte_identical_across_renders(tmp_path):
    """The whole point of PDF_META/PNG_META: unchanged inputs -> identical
    bytes, which is what the provenance manifest's md5 check relies on."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def render(out_dir):
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 4])
        written = V.savefig_pinned(fig, out_dir, "repeat_fig")
        plt.close(fig)
        return written

    first = render(tmp_path / "a")
    second = render(tmp_path / "b")
    for p1, p2 in zip(first, second):
        assert p1.read_bytes() == p2.read_bytes(), (
            f"{p1.name}: re-render of identical content produced different "
            "bytes -- the metadata pin is not doing its job")
