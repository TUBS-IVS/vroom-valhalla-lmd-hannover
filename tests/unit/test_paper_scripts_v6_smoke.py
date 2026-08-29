"""End-to-end schema-guard smoke tests for three of the Task 19 W1a
re-pointed scripts/paper/ scripts, on tiny SYNTHETIC legacy-schema frames
(no real results/ data, no grid, no VROOM).

These exercise the actual subprocess CLI path (``--legacy-dir``/``--rev-dir``/
``--out-dir``) against a fixture built to 74_v2_to_legacy_tables.py's own
SCHEMA dict, so a future change to that schema -- or a regression in one of
these scripts' v6 wiring -- breaks a fast unit test instead of only
surfacing on the next real v6 run. Picked to cover the two hardest-earned
patterns in this wave:

* ``paper_sweet_spot_math.py`` -- the simplest straight B repoint (baseline
  sanity: the CLI plumbing itself works end to end).
* ``paper_final_init_vs_balanced.py`` -- the legacy-column-NaN -> v6-native
  fallback (``_paper_v6_common.load_fleet_before_after``) that several
  scripts in this wave share.
* ``paper_fleet_balancing_detail.py`` -- the same fallback PLUS a genuine E
  skip (FB7 has no v6 source at all) that must not crash the run.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "scripts" / "paper"

PROVIDERS = ["DHL", "GLS"]
PENALTIES = [0.0, 0.5]
SHARES = [0.0, 1.0]
PLZ = ["30159", "30167"]


def _rows(*, per_provider_plz=True):
    """Every (penalty, share, provider[, plz]) combination, in schema order."""
    for p in PENALTIES:
        for s in SHARES:
            for prov in PROVIDERS:
                if per_provider_plz:
                    for plz in PLZ:
                        yield p, s, prov, plz
                else:
                    yield p, s, prov


def _write_legacy_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    chosen_cols = ["penalty", "share_willing", "provider", "plz", "weekly_parcels",
                  "schedule_idx_init", "schedule_idx_balanced", "schedule_size_init",
                  "schedule_size_balanced", "weekdays_init", "weekdays_balanced",
                  "avg_wait_d_init", "avg_wait_d_balanced", "dd_cost_init",
                  "dd_cost_balanced", "veh_init", "veh_balanced"]
    rows = []
    for p, s, prov, plz in _rows():
        rows.append(dict(zip(chosen_cols, [
            p, s, prov, plz, 500.0, 0, 1 if s > 0 else 0, 6, 3 if s > 0 else 6,
            "Mon,Tue,Wed,Thu,Fri,Sat", "Mon,Wed,Fri",
            0.0, 0.4 if s > 0 else 0.0, 1000.0, 900.0 if s > 0 else 1000.0,
            5.0, 4.0 if s > 0 else 5.0,
        ])))
    pd.DataFrame(rows, columns=chosen_cols).to_csv(
        run_dir / "tab_chosen_schedules.csv", index=False)

    bal_cols = ["penalty", "share_willing", "provider", "n_plz", "init_cost_eur",
               "balanced_cost_eur", "cost_delta_eur", "cost_delta_pct",
               "imbalance_before", "imbalance_after", "imbalance_reduction_pct",
               "max_fleet_before", "max_fleet_after", "total_routes_before",
               "total_routes_after", "swaps_made"]
    rows = []
    for p, s, prov in _rows(per_provider_plz=False):
        rows.append(dict(zip(bal_cols, [
            p, s, prov, len(PLZ), 2000.0, 1900.0, -100.0, -5.0,
            10.0, 4.0, 60.0,
            float("nan"), 12.0, 10.0, 9.0, 3,
        ])))
    pd.DataFrame(rows, columns=bal_cols).to_csv(
        run_dir / "tab_balancing_summary.csv", index=False)


def _write_legacy_rev(rev_dir: Path) -> None:
    rev_dir.mkdir(parents=True, exist_ok=True)
    cs_cols = ["penalty", "share_willing", "provider", "dd_cost_stage3_eur",
              "express_stage3_eur", "total_stage3_eur"]
    rows = []
    for p, s, prov in _rows(per_provider_plz=False):
        total = 1000.0 if s == 0.0 else 900.0
        rows.append(dict(zip(cs_cols, [p, s, prov, total * 0.8, total * 0.2, total])))
    pd.DataFrame(rows, columns=cs_cols).to_csv(
        rev_dir / "tab_costs_smoothed.csv", index=False)

    wait_cols = ["penalty", "share_willing", "avg_wait_d_stage3"]
    rows = []
    for p in PENALTIES:
        for s in SHARES:
            rows.append(dict(zip(wait_cols, [p, s, 0.0 if s == 0.0 else 0.3])))
    pd.DataFrame(rows, columns=wait_cols).to_csv(
        rev_dir / "tab_wait_smoothed.csv", index=False)

    fh_cols = ["penalty", "share_willing", "provider", "hub", "day", "dd_veh",
              "expr_veh_old", "expr_veh_fixed", "fleet_old", "fleet_fixed"]
    rows = []
    for p, s, prov in _rows(per_provider_plz=False):
        for day in range(6):
            rows.append(dict(zip(fh_cols, [
                p, s, prov, f"{prov}_hub0", day, 2.0,
                float("nan"), 1.0, float("nan"), 3.0,
            ])))
    pd.DataFrame(rows, columns=fh_cols).to_csv(
        rev_dir / "tab_fleet_per_hub_fixed.csv", index=False)


def _write_v6_native_costs(rev_root: Path) -> None:
    """The one v6-native table load_fleet_before_after() reads."""
    cols = ["penalty", "share_willing", "provider", "sum_hub_peak_before",
           "sum_hub_peak_after", "vehicle_days_before", "vehicle_days"]
    rows = []
    for p, s, prov in _rows(per_provider_plz=False):
        rows.append(dict(zip(cols, [p, s, prov, 20.0, 20.0 if s == 0 else 14.0,
                                    100.0, 100.0 if s == 0 else 92.0])))
    pd.DataFrame(rows, columns=cols).to_csv(rev_root / "tab_costs_v2.csv", index=False)


@pytest.fixture()
def legacy_fixture(tmp_path):
    """<tmp>/legacy/{run,rev} (74_-shaped) + <tmp>/rev_dir/tab_costs_v2.csv
    (v6-native, for the fleet-before-after fallback)."""
    legacy = tmp_path / "legacy"
    _write_legacy_run(legacy / "run")
    _write_legacy_rev(legacy / "rev")
    rev_dir = tmp_path / "rev_dir"
    rev_dir.mkdir()
    _write_v6_native_costs(rev_dir)
    return dict(legacy_run=legacy / "run", rev_dir=rev_dir)


def _run(script: str, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(PAPER / script), *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def test_paper_sweet_spot_math_runs_on_synthetic_legacy_frame(legacy_fixture, tmp_path):
    out = tmp_path / "out"
    r = _run("paper_sweet_spot_math.py",
            "--legacy-dir", str(legacy_fixture["legacy_run"]), "--out-dir", str(out))
    assert r.returncode == 0, r.stderr
    assert (out / "fig_sweet_spot_math.png").exists()
    assert (out / "fig_sweet_spot_math.pdf").exists()
    assert (out / "tab_sweet_spot_data.csv").exists()


def test_paper_final_init_vs_balanced_uses_v6_native_fleet_fallback(
        legacy_fixture, tmp_path):
    """max_fleet_before is NaN in the synthetic tab_balancing_summary.csv
    (matching 74_'s real NO_SOURCE) -- the script must fall back to the
    v6-native tab_costs_v2.csv aggregate and still render BOTH panels of
    fig_FB5, not silently sum the NaN column to zero and not crash."""
    out = tmp_path / "out"
    r = _run("paper_final_init_vs_balanced.py",
            "--legacy-dir", str(legacy_fixture["legacy_run"]),
            "--rev-dir", str(legacy_fixture["rev_dir"]),
            "--out-dir", str(out))
    assert r.returncode == 0, r.stderr
    assert "FB5: freq_shift" in r.stdout
    assert "fleet before/after E" not in r.stdout  # the fallback found data
    fb5 = out / "06_balancing" / "tab_init_vs_balanced.csv"
    assert fb5.exists()
    df = pd.read_csv(fb5)
    assert "fleet_before" in df.columns and df.fleet_before.notna().all()
    assert (out / "11_spatial_maps" / "fig_MAP4_init_vs_balanced_P0.png").exists()


def test_paper_final_init_vs_balanced_marks_fleet_panel_e_without_rev_dir(
        legacy_fixture, tmp_path):
    """Without --rev-dir there is no v6-native fallback available; the
    script must degrade to the frequency-only panel and say so, not crash
    or fabricate a fleet-before number."""
    out = tmp_path / "out"
    r = _run("paper_final_init_vs_balanced.py",
            "--legacy-dir", str(legacy_fixture["legacy_run"]), "--out-dir", str(out))
    assert r.returncode == 0, r.stderr
    assert "[E]" in r.stdout
    df = pd.read_csv(out / "06_balancing" / "tab_init_vs_balanced.csv")
    assert "fleet_before" not in df.columns


def test_paper_fleet_balancing_detail_reconstructs_fb6_and_skips_fb7(
        legacy_fixture, tmp_path):
    """FB6 (system-level) must be reconstructed from the v6-native fallback;
    FB7 (per-hub) has no v6 source at any grain and must be skipped with an
    [E] message, not crash and not invent a per-hub 'before'."""
    out = tmp_path / "out"
    r = _run("paper_fleet_balancing_detail.py",
            "--legacy-dir", str(legacy_fixture["legacy_run"]),
            "--rev-dir", str(legacy_fixture["rev_dir"]),
            "--out-dir", str(out))
    assert r.returncode == 0, r.stderr
    assert (out / "fig_FB6_peak_day_heatmap.png").exists()
    assert "[E] FB7" in r.stdout
    assert not (out / "fig_FB7_avg_peak_per_hub.png").exists()
    assert not (out / "tab_per_hub_summary.csv").exists()
    per_prov = pd.read_csv(out / "tab_per_provider_fleet_summary.csv")
    assert "peak_reduction_pct" in per_prov.columns
    assert per_prov.peak_reduction_pct.notna().all()
