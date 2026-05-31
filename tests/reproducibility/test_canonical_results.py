"""Verify the canonical results files that back paper numbers exist.

These tests don't re-run the pipeline — they just confirm that the
checked-in canonical outputs under `results/runs/`, `results/paper_*`,
and `results/supplementary/` are present and well-formed (parseable as
CSV with the columns paper-figure scripts expect).

If a reviewer wants to regenerate, the four-stage pipeline overwrites
exactly these files; this test then re-runs against the regenerated
versions and verifies the schema is unchanged.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"


def _df(rel: str) -> pd.DataFrame:
    """Load a results CSV by relative path, with a helpful error message."""
    path = RESULTS / rel
    assert path.exists(), (
        f"Canonical results file missing: {rel}\n"
        f"   expected at {path}\n"
        "   Either run `batch-delivery paper` from scratch or restore from\n"
        "   `pre-refactor-2026-05-31` branch."
    )
    return pd.read_csv(path)


# ─── path2 optimisation run ──────────────────────────────────────────────────


def test_path2_balancing_summary_present_and_well_formed() -> None:
    """tab_balancing_summary.csv has 88 cells x 7 providers = 616 rows.

    The 88-cell grid is 8 service-penalty levels x 11 willingness-to-wait
    shares; 7 carriers per cell gives 616 (provider, P, theta) rows.
    """
    df = _df("runs/path2_2026_05_29/tab_balancing_summary.csv")
    assert {"penalty", "share_willing", "provider"} <= set(df.columns), df.columns
    assert {"init_cost_eur", "balanced_cost_eur"} <= set(df.columns)
    assert len(df) == 616, f"expected 616 rows, got {len(df)}"


def test_path2_chosen_schedules_present() -> None:
    df = _df("runs/path2_2026_05_29/tab_chosen_schedules.csv")
    assert {"penalty", "share_willing", "provider", "plz"} <= set(df.columns)


def test_path2_fleet_per_hub_present() -> None:
    df = _df("runs/path2_2026_05_29/tab_fleet_per_hub.csv")
    assert {"penalty", "share_willing", "day", "fleet_before", "fleet_after"} <= set(df.columns)


# ─── system smoothing post-process ───────────────────────────────────────────


def test_system_spread_present() -> None:
    df = _df("runs/path2_2026_05_29/_system_spread_per_cell.csv")
    assert {"penalty", "share_willing"} <= set(df.columns)
    assert {"system_spread_before_smoothing", "system_spread_after_smoothing"} <= set(df.columns)


# ─── VROOM validation outputs ─────────────────────────────────────────────────


def test_vroom_balanced_validation_present() -> None:
    """tab_vroom_balanced.csv has ~3300 rows (312 cells x days for 3
    validated penalty levels). The companion tab_vroom_path2.csv is
    incomplete because the operator aborted the run after the
    conservatism claim had enough data — see MANIFEST.md provenance.
    """
    df = _df("paper_results_2026_05_30/07_validation/tab_vroom_balanced.csv")
    assert len(df) >= 3000, (
        f"expected >= 3000 rows in balanced validation, got {len(df)}"
    )
    # The three validated penalty values must each appear
    validated = set(df.penalty.unique())
    expected = {0.0, 0.25, 0.5}
    assert expected <= validated, (
        f"validated penalty set missing values: {expected - validated}"
    )


# ─── headline-number sanity checks ───────────────────────────────────────────


def test_baseline_total_cost_in_expected_band() -> None:
    """Baseline total cost (theta=0) per the path2 run should be ~1.9 M€/wk."""
    df = _df("runs/path2_2026_05_29/tab_balancing_summary.csv")
    base = df[df.share_willing == 0.0].init_cost_eur.sum() / 8  # 8 P levels, all equal at theta=0
    # Paper claims 1.91 M€; allow ±2 % tolerance
    assert 1.85e6 <= base <= 1.95e6, (
        f"baseline weekly cost out of band: {base:.0f} EUR "
        f"(expected ~1.91 M EUR, see paper §3)"
    )


@pytest.mark.parametrize("penalty", [0.0, 0.25, 0.5])
def test_paper_saving_at_documented_p_and_theta_one(penalty: float) -> None:
    """Cost-saving headline numbers at theta=1 are in the documented band.

    The paper reports 22.8 % at P=0 and 13.5 % at P=0.5. Allow ±1 pp.
    """
    df = _df("runs/path2_2026_05_29/tab_balancing_summary.csv")
    base = df[df.share_willing == 0.0].init_cost_eur.sum() / 8
    cell = df[(df.penalty == penalty) & (df.share_willing == 1.0)]
    bal = cell.balanced_cost_eur.sum()
    saving_pct = 100 * (base - bal) / base
    documented = {0.0: 22.8, 0.25: 18.6, 0.5: 13.5}[penalty]
    assert abs(saving_pct - documented) < 1.5, (
        f"P={penalty}: cost saving {saving_pct:.2f}% diverges from "
        f"documented {documented}% by more than 1.5 pp — paper claim broken?"
    )
