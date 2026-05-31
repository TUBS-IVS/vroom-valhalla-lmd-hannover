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
    validated penalty levels P in {0, 0.25, 0.5}). The companion
    tab_vroom_path2.csv is intentionally not used by the paper
    pipeline — see MANIFEST.md provenance.
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


def test_baseline_total_cost_matches_paper() -> None:
    """Baseline weekly cost (theta=0, every P level) must be 1.91 M EUR
    when rounded to the paper's 2-decimal precision (1.9097 M actual).
    """
    df = _df("runs/path2_2026_05_29/tab_balancing_summary.csv")
    base = df[df.share_willing == 0.0].init_cost_eur.sum() / 8  # 8 P levels, all equal at theta=0
    base_m = round(base / 1e6, 2)
    assert base_m == 1.91, (
        f"baseline weekly cost {base:.2f} EUR rounds to {base_m} M, "
        f"expected 1.91 M EUR (paper §3)"
    )


@pytest.mark.parametrize(
    "penalty,documented_pct",
    [
        (0.0,  22.8),  # paper Table 1 / abstract: 22.8 %
        (0.25, 18.6),  # paper §3.2:               18.6 %
        (0.5,  13.5),  # paper abstract:           13.5 %
    ],
)
def test_paper_saving_at_documented_p_and_theta_one(
    penalty: float, documented_pct: float,
) -> None:
    """Cost-saving headline numbers at theta=1 must equal the paper claim
    to 1-decimal precision (the precision the paper actually quotes).

    The exact unrounded values in the canonical CSV are:
      P=0.00 -> 22.8093 %  (rounds to 22.8)
      P=0.25 -> 18.5538 %  (rounds to 18.6)
      P=0.50 -> 13.5170 %  (rounds to 13.5)
    """
    df = _df("runs/path2_2026_05_29/tab_balancing_summary.csv")
    base = df[df.share_willing == 0.0].init_cost_eur.sum() / 8
    cell = df[(df.penalty == penalty) & (df.share_willing == 1.0)]
    bal = cell.balanced_cost_eur.sum()
    saving_pct = 100 * (base - bal) / base
    rounded = round(saving_pct, 1)
    assert rounded == documented_pct, (
        f"P={penalty}: cost saving {saving_pct:.4f}% rounds to "
        f"{rounded}%, expected {documented_pct}% — paper claim broken?"
    )
