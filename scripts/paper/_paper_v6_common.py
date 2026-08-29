"""Shared v6-analysis helpers for the legacy scripts/paper/ figure scripts
(Task 19, wave W1a: "every historical results/ analysis re-generated on v6").

Every script this module supports was written against a pre-revision run
directory (``overnight_2026_05_27*``, ``overnight_2026_05_29_path2``, ...)
that no longer exists on disk except as an unindexed archive copy.  Task 19
re-points each script at the v6 grid -- either the v6-native tables under
``results/revision_2026_08_v6/`` directly, or the legacy-schema tables
``scripts/revision/74_v2_to_legacy_tables.py`` derives from it -- WITHOUT
changing what the script does when it is run with no flags: every new CLI
argument this task adds defaults to the script's original hardcoded path,
so historical behaviour (when the historical inputs happen to still be
present on disk) is untouched.

This module supplies the pieces every touched script needs in common:

* :func:`add_provenance_footer` + :data:`PNG_META`/:data:`PDF_META` -- the
  brief's "every figure gets the provenance footer and pinned metadata"
  rule, in the ``fig.text`` style ``scripts/revision/76_maps_v2.py`` uses
  for its own caption footers.
* :func:`assert_has_data` -- a fail-loud guard for a legacy-adapter column
  that is NaN-only for a documented reason (74_'s ``NO_SOURCE`` list): it
  stops a script from silently plotting zeros out of an all-NaN column via
  pandas' default ``skipna=True`` sum/mean, which would fabricate a number
  no v6 source backs.
* :func:`load_fleet_before_after` -- the ONE aggregate quantity the legacy
  adapter deliberately leaves NaN (``tab_balancing_summary.csv``'s
  ``max_fleet_before``) but which DOES exist, at (penalty, share_willing,
  provider) grain, directly on the v6-native grid: ``sum_hub_peak_before``
  / ``sum_hub_peak_after`` and ``vehicle_days_before`` / ``vehicle_days``
  on ``tab_costs_v2.csv``.  A PER-HUB-DAY before/after fleet split (the old
  ``tab_fleet_per_hub.csv``'s ``fleet_before``) has NO v6 source at all --
  v6 only ever computes a per-hub-day fleet at the final plan -- so that
  split stays E (not approximated from this aggregate).
* :func:`run_legacy_adapter` -- runs ``74_v2_to_legacy_tables.py`` once
  (table-only, no ``--render``, so the frozen paper figures are never
  touched and the read-only v6 directory is never written to) into a
  scratch directory under the task's own output tree, and returns its
  ``run/`` and ``rev/`` roots.

Legacy invariants that are FALSE on v6 (Task 19 brief; corrected wherever a
touched script asserted them in a title/caption -- see each script's own
comment for which of these applied):

* "balancing/system-smoothing preserves delivery frequency" -- true only at
  theta=0 (pinned to daily); at theta>0 stage 2 is frequency-free (Task 6f,
  compendium 40.14), so a v6 balanced/init schedule-size comparison is a
  real reallocation, not a same-frequency reshuffle.
* "every region type is driven to zero saving by P=5" -- v6 keeps a small
  residual pooling saving past P=5 (compendium 40.23b); the correct v6
  phrasing is "from P=5 on, every type keeps <= X% saving" for whatever X
  the current grid shows, never a bare "-> 0".
* "nine one-cell depots" -- v6 has eight single-cell DHL depots (compendium
  40.23b), not nine.
* a specific numeric "bulge" magnitude at (P, theta)=(10, 0.1) quoted from
  the submission (41.7% non-daily cells) -- v6's own value is unrelated
  (0.40% partial adoption, compendium 40.21) and must not be restated as
  the submission's number.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

#: matplotlib savefig metadata pinned to remove run-to-run noise (the
#: ``_PDF_META``/``_PNG_META`` pattern ``scripts/revision/70_figs_tables_v2.py``
#: uses): CreationDate/Software otherwise embed the render wall-clock time,
#: which would make two regenerations of the same figure on the same data
#: differ byte-for-byte.
PDF_META = {"CreationDate": None}
PNG_META = {"Software": None}

#: Same small, muted-gray caption style as 76_maps_v2.py's own FOOT_KW.
FOOTER_KW = dict(ha="center", va="bottom", fontsize=6.5, color="0.45")


def provenance_text(*, plan: str, script: str, source: str) -> str:
    """The exact stamp string: ``v6 · <plan> · <script> · <source table>``."""
    return f"v6 · {plan} · {script} · {source}"


def add_provenance_footer(fig, *, plan: str, script: str, source: str,
                          y: float = -0.01) -> None:
    """Stamp ``v6 · <plan> · <script> · <source table>`` at the bottom.

    ``plan`` is one of the two v5/v6 plans in prose (e.g. "stage-1
    routing-optimal" or "operator-polished (balanced)") or "n/a" for a
    plan-independent figure.  ``script`` is this producing script's own
    filename, so a reader can find the code from the figure alone.
    ``source`` names the v6/legacy table(s) the numbers came from.
    """
    fig.text(0.5, y, provenance_text(plan=plan, script=script, source=source),
             transform=fig.transFigure, **FOOTER_KW)


def savefig_pair(fig, png_path: Path, pdf_path: Path) -> None:
    """Save PNG+PDF with pinned metadata; creates the parent dir first."""
    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight", metadata=PNG_META)
    fig.savefig(pdf_path, bbox_inches="tight", metadata=PDF_META)


class NoV6Source(AssertionError):
    """A column a legacy script wants to plot/aggregate has no v6 source."""


def assert_has_data(df: pd.DataFrame, col: str, *, context: str) -> None:
    """Fail loud (never silently zero-sum) on an all-NaN legacy column.

    ``74_v2_to_legacy_tables.py`` writes a handful of pre-revision-only
    legacy columns as NaN with a documented reason (its ``NO_SOURCE`` dict)
    rather than guessing a number.  pandas' default ``skipna=True`` turns
    ``all_nan_column.sum()`` into a silent ``0.0`` -- exactly the kind of
    fabricated "fleet before" or "vehicles saved" a fail-loud pipeline must
    never produce.  Call this BEFORE aggregating such a column; catch
    :class:`NoV6Source` to skip just that one panel (mark it E in
    ``_STATUS.md`` with the reason), never to substitute a filled-in value.
    """
    if col not in df.columns:
        raise NoV6Source(f"{context}: column {col!r} is not in this frame "
                          f"at all (have {list(df.columns)})")
    if df[col].isna().all():
        raise NoV6Source(
            f"{context}: {col!r} is NaN for all {len(df)} row(s) -- v6 has "
            "no source for this legacy quantity (see 74_'s NO_SOURCE); "
            "skip this panel, do not sum/plot/fillna it")


#: Columns tab_costs_v2.csv carries at (penalty, share_willing, provider)
#: grain that the legacy adapter otherwise leaves NaN in
#: tab_balancing_summary.csv's max_fleet_before -- see module docstring.
_FLEET_BA_COLS = ("sum_hub_peak_before", "sum_hub_peak_after",
                  "vehicle_days_before", "vehicle_days")


def load_fleet_before_after(rev_dir: Path) -> pd.DataFrame:
    """Provider-level (NOT per-hub-day) peak-fleet & vehicle-days, before/after.

    v6-native ``tab_costs_v2.csv`` carries these totals directly at
    (penalty, share_willing, provider) grain -- the one aggregate the
    legacy adapter otherwise leaves as ``max_fleet_before = NaN`` in
    ``tab_balancing_summary.csv`` (see ``74_``'s ``NO_SOURCE``).  There is
    still NO per-hub-day before/after split in v6 (only the final plan's
    per-hub-day fleet is ever computed), so this frame can back a
    provider- or system-level before/after comparison but never a
    per-hub-day one -- that stays E.
    """
    rev_dir = Path(rev_dir)
    path = rev_dir / "tab_costs_v2.csv"
    costs = pd.read_csv(path)
    missing = [c for c in _FLEET_BA_COLS if c not in costs.columns]
    assert not missing, (
        f"{path}: lacks {missing} -- not a v6-schema grid (tab_costs_v2.csv "
        "should carry sum_hub_peak_before/after and vehicle_days_before/"
        "vehicle_days)")
    out = costs[["penalty", "share_willing", "provider", *_FLEET_BA_COLS]].copy()
    return out.rename(columns={"vehicle_days": "vehicle_days_after"})


def run_legacy_adapter(rev_dir: Path, out_dir: Path, *,
                       express_allocation: str = "per-tour") -> tuple[Path, Path]:
    """Run 74_'s table build (NOT ``--render``) into *out_dir*; return (run, rev).

    Never touches ``rev_dir`` itself (74_'s ``--out`` writes only under
    *out_dir*), and never renders the three frozen paper figures -- this is
    the table-only half of 74_, reused across every B-classified script in
    this wave.  Idempotent: if ``<out_dir>/run/tab_chosen_schedules.csv``
    already exists this is a no-op, so many scripts can each call it with
    the same *out_dir* and only the first pays the (cheap, ~5s) cost.
    """
    out_dir = Path(out_dir)
    run_dir, rev_sub = out_dir / "run", out_dir / "rev"
    if not (run_dir / "tab_chosen_schedules.csv").exists():
        adapter = REPO_ROOT / "scripts" / "revision" / "74_v2_to_legacy_tables.py"
        cmd = [sys.executable, str(adapter), "--rev-dir", str(rev_dir),
              "--out", str(out_dir), "--express-allocation", express_allocation]
        print(f"[legacy-adapter] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    assert (run_dir / "tab_chosen_schedules.csv").exists(), (
        f"legacy adapter did not produce {run_dir / 'tab_chosen_schedules.csv'}")
    return run_dir, rev_sub


def build_penalty_series(legacy_rev_dir: Path, *, share: float = 1.0) -> pd.DataFrame:
    """System-level (penalty, saving_pct, avg_wait) at one theta, from
    74_-legacy's tab_costs_smoothed.csv + tab_wait_smoothed.csv.

    Several pre-revision scripts (``paper_final_sweetspot.py``,
    ``paper_final_sweetspot_plots.py``) read a dedicated fine-P-grid table
    (``tab_penalty_finegrid_production.csv``) that only a separate,
    out-of-scope-for-this-wave C-port script produces. v6's own standard
    8-point P grid gives the same THREE columns directly, at the resolution
    the (cheap, already-computed) legacy adapter provides: ``total_stage3_eur``
    summed over providers is the system routing-lens cost of the
    operator-polished plan (74_'s own name for "stage3" is that plan, not a
    third stage -- stage 3 is OFF in v5/v6); the theta=0 row of the SAME
    table is the system's own daily baseline (share=0 is daily at every P,
    asserted below); ``avg_wait_d_stage3`` is already the parcel-weighted
    SYSTEM average wait 74_ computes from tab_wait_fixed's willing/total
    parcel counts. This is coarser than the old fine grid, not narrower in
    scope -- it never invents a P value the v6 grid does not have.
    """
    legacy_rev_dir = Path(legacy_rev_dir)
    costs = pd.read_csv(legacy_rev_dir / "tab_costs_smoothed.csv")
    wait = pd.read_csv(legacy_rev_dir / "tab_wait_smoothed.csv")
    sysc = costs.groupby(["penalty", "share_willing"], as_index=False) \
                .total_stage3_eur.sum()
    base_rows = sysc[np.isclose(sysc.share_willing, 0.0)]
    assert len(base_rows), f"{legacy_rev_dir}: no share_willing=0 baseline rows"
    baseline = base_rows.total_stage3_eur
    assert np.allclose(baseline, baseline.iloc[0]), (
        "theta=0 system cost is not constant across P -- share=0 should be "
        "all-daily regardless of P, so this is not a v6 grid this function "
        "can trust")
    baseline = float(baseline.iloc[0])

    sub = sysc[np.isclose(sysc.share_willing, share)]
    assert len(sub), (
        f"{legacy_rev_dir}: tab_costs_smoothed.csv has no share_willing="
        f"{share} rows at all -- not a value this v6 grid was run at")
    sub = sub.merge(
        wait[np.isclose(wait.share_willing, share)][["penalty", "avg_wait_d_stage3"]],
        on="penalty", how="left")
    assert sub.avg_wait_d_stage3.notna().all(), (
        f"{legacy_rev_dir}: tab_wait_smoothed.csv is missing share={share} "
        "for some penalty -- cannot pair a saving with a wait")
    out = pd.DataFrame({
        "penalty": sub.penalty,
        "saving_pct": 100 * (baseline - sub.total_stage3_eur) / baseline,
        "avg_wait": sub.avg_wait_d_stage3,
    }).sort_values("penalty").reset_index(drop=True)
    return out


def add_v6_cli_args(ap, *, needs_legacy: bool = False) -> None:
    """Add the common ``--rev-dir``/``--legacy-dir``/``--out-dir`` flags.

    Every default is ``None``: when nothing is passed, the script keeps
    reading/writing its own hardcoded historical paths, so its default
    invocation is unchanged by this task (per the brief: "keep defaults
    unchanged so historical behaviour is untouched").
    """
    ap.add_argument("--rev-dir", default=None,
                    help="v6 grid directory, e.g. results/revision_2026_08_v6 "
                         "(default: use this script's historical input path)")
    if needs_legacy:
        ap.add_argument("--legacy-dir", default=None,
                        help="74_'s output root (containing run/ and rev/); "
                             "built automatically under <out-dir>/_legacy "
                             "when --rev-dir is given and this is omitted")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: this script's "
                         "historical output path)")
