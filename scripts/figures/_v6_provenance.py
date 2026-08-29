"""Shared v6-provenance helpers for ``scripts/figures/`` (Task 19 W1b).

Every regenerated ``scripts/figures/*.py`` script (status A/B/D per
``.superpowers/sdd/2026-08-25-realistic-tours-implementation/task-19-inventory-A.md``)
uses this module for the pieces that must be identical across the whole set
or the regenerated figures stop being comparable to each other:

1. :func:`add_v6_args` -- one ``--rev-dir``/``--out-dir`` CLI pair per
   script.  Defaults are the script's ORIGINAL hardcoded historical path, so
   running a script with no flags still reproduces its documented
   pre-revision behaviour; the actual v6 regeneration run always passes
   both flags explicitly ("defaults unchanged", Task 19 brief).
2. :func:`footer` -- the tiny provenance line every regenerated figure
   carries: ``v6 · <plan> · <script> · <source table>``.
3. :data:`PDF_META` / :data:`PNG_META` + :func:`savefig_pinned` -- the
   byte-identical-re-render metadata pattern from
   ``scripts/revision/70_figs_tables_v2.py`` (matplotlib stamps a
   ``/CreationDate``/timestamp into every render, which would make two
   renders of identical content differ by md5 otherwise).
4. :func:`require_columns` / :func:`require_nonnull` -- fail loud (assert,
   no ``fillna``, no ``.get(..., 0)``) the moment a table is missing a
   column a script needs, or carries one that is a 74_ ``NO_SOURCE``
   all-NaN column (e.g. ``tab_balancing_summary.csv::max_fleet_before``).
5. :func:`chord_knee` -- the chord-distance Pareto-knee finder used by more
   than one script here; ``scripts/revision/_figs_tables_v2.py``'s
   ``lsp_knees`` uses the identical formula for the audited per-LSP P*
   tables, so a knee computed here from a script's OWN curve cannot drift
   from that method.
6. :func:`legacy_baseline` -- reads the grid's OWN baseline from 74_'s
   ``legacy_manifest.json`` instead of a hardcoded constant (74_'s own
   docstring: a v6 saving must never be taken against the 2026-07/path2
   denominator).

None of this changes cost-model semantics; it is plumbing only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Matplotlib stamps a /CreationDate into every PDF and a timestamp into
# every PNG, so two renders of identical content get different md5s.
# Suppressed here so a re-render of unchanged v6 inputs is byte-identical
# (same rationale, same values, as 70_figs_tables_v2.py's _PDF_META/_PNG_META).
PDF_META = {"CreationDate": None}
PNG_META = {"Software": None}

#: plan-declaration text, consistent wording across every regenerated figure
PLAN1 = "routing-optimal plan (stage 1)"
PLAN2 = "operator-polished plan (stage 2)"
PLAN_BOTH = "routing-optimal (stage 1) & operator-polished (stage 2) plans"


def add_v6_args(parser: argparse.ArgumentParser, *, default_rev,
                default_out, rev_help: str) -> argparse.ArgumentParser:
    """Add ``--rev-dir``/``--out-dir``; defaults are the script's ORIGINAL
    hardcoded paths so an unflagged run keeps its documented pre-revision
    behaviour (Task 19 W1b: "defaults unchanged")."""
    parser.add_argument("--rev-dir", default=str(default_rev), help=rev_help)
    parser.add_argument(
        "--out-dir", default=str(default_out),
        help="output directory (default: the script's original historical "
             "output folder)")
    return parser


def savefig_pinned(fig, out_dir, stem: str) -> list[Path]:
    """Save png+pdf with pinned metadata so re-renders are byte-identical."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext, meta in (("png", PNG_META), ("pdf", PDF_META)):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", metadata=meta)
        written.append(path)
    return written


def footer(fig, *, plan: str, script: str, source: str, rev_label: str = "v6",
          y: float = -0.02, fontsize: float = 6.5) -> None:
    """The tiny provenance line every regenerated figure carries.

    ``v6 · <plan> · <script> · <source table>`` (Task 19 brief, "Rules for
    every regenerated figure").
    """
    fig.text(0.5, y,
              f"{rev_label} · {plan} · {script} · {source}",
              ha="center", va="top", fontsize=fontsize, color="0.35")


def require_columns(df: pd.DataFrame, columns, *, source: str) -> None:
    """Fail loud the moment a table is missing a column this script reads.

    No ``fillna``, no ``.get(..., 0)``.  A legacy column with NO v5/v6
    source (74_'s ``NO_SOURCE``) is written as an all-NaN column, never
    dropped -- so a *missing* column here always means a real schema break,
    not the known NO_SOURCE case (that is :func:`require_nonnull`).
    """
    missing = [c for c in columns if c not in df.columns]
    assert not missing, (
        f"{source}: missing required column(s) {missing} -- refusing to "
        "silently proceed (no fillna, no default); re-check the adapter "
        "schema or --rev-dir")


def require_nonnull(df: pd.DataFrame, column: str, *, source: str) -> None:
    """Fail loud if a required column is present but entirely NaN.

    This is the case 74_'s ``NO_SOURCE`` documents explicitly (e.g.
    ``tab_balancing_summary.csv::max_fleet_before``): the column exists (so
    :func:`require_columns` passes) but carries no v5/v6 value anywhere.
    Reading it without this check would silently plot all-NaN as zero.
    """
    assert column in df.columns, f"{source}: {column!r} not present at all"
    assert df[column].notna().any(), (
        f"{source}: {column!r} is entirely NaN on this grid (a known 74_ "
        "NO_SOURCE column with no v5/v6 source) -- the quantity that "
        "depends on it cannot be computed; drop that panel, do not impute")


def chord_knee(wait, saving) -> int:
    """Index of the chord-distance Pareto knee on a (wait, saving) front.

    Same rule as ``scripts/revision/_figs_tables_v2.py::lsp_knees``: min-max
    normalise both axes over the curve's own range and take the point that
    maximises the signed chord distance ``saving_n - wait_n``.  Ties resolve
    to the first (lowest-index) occurrence via ``argmax``.
    """
    wait = np.asarray(wait, dtype=float)
    saving = np.asarray(saving, dtype=float)
    assert len(wait) == len(saving) and len(wait) >= 3, (
        f"chord_knee: need >=3 paired points, got "
        f"{len(wait)}/{len(saving)}")
    w_n = (wait - wait.min()) / (wait.max() - wait.min() + 1e-12)
    s_n = (saving - saving.min()) / (saving.max() - saving.min() + 1e-12)
    return int(np.argmax(s_n - w_n))


def legacy_baseline(legacy_out) -> dict:
    """This grid's OWN baseline, from 74_'s ``legacy_manifest.json``.

    Never hardcode a ``BASE_TOTAL`` constant: 74_'s own docstring rules that
    a v6 saving must not be taken against the 2026-07 (or path2) EUR
    denominator -- the bundle head re-prices the theta=0 baseline too.
    """
    path = Path(legacy_out) / "legacy_manifest.json"
    assert path.exists(), (
        f"{path} missing -- run scripts/revision/74_v2_to_legacy_tables.py "
        f"--rev-dir <v6-dir> --out {legacy_out} first")
    return json.loads(path.read_text(encoding="utf-8"))


def base_total_with_path2_fallback(rev) -> float:
    """This grid's OWN daily-delivery baseline EUR total, for a script that
    reads ``<rev>/tab_balancing_summary.csv``.

    Prefers 74_'s ``legacy_manifest.json`` (v6: ``rev`` is the adapter's
    ``<out>/run`` directory, the manifest sits at its parent). Falls back to
    the legacy table's own theta=0 row when there is no manifest -- the
    script's ORIGINAL (path2) default directory has no adapter run at all,
    so this keeps that default path working, still never against a
    hardcoded constant.
    """
    rev = Path(rev)
    manifest = rev.parent / "legacy_manifest.json"
    if manifest.exists():
        return float(legacy_baseline(rev.parent)["base_total_eur"])
    summ_path = rev / "tab_balancing_summary.csv"
    summ = pd.read_csv(summ_path)
    require_columns(summ, ["penalty", "share_willing", "balanced_cost_eur"],
                    source=str(summ_path))
    b0 = summ[np.isclose(summ.share_willing, 0.0)]
    assert len(b0), f"{summ_path}: no theta=0 rows to compute a baseline from"
    P0 = sorted(b0.penalty.unique())[0]
    return float(b0[np.isclose(b0.penalty, P0)].balanced_cost_eur.sum())
