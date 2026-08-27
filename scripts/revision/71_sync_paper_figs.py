"""Gate G7: sync the revision figures into the paper folders, md5-verified.

The trap this exists to close (Kompendium §38.8): ``paper/EWGT_2026_rev1/``
has three times been found holding **md5-identical copies of the submitted
figures** while everybody assumed it held the revision.  A stale Fig. 5
survived a full round that way.  So this script never assumes -- it copies,
then re-reads both files and compares md5 digests, and additionally reports
whether each destination is still byte-identical to the frozen submission in
``paper/EWGT_2026/figures/``.

Two destinations, two naming schemes
------------------------------------
``paper/EWGT_2026_rev1/figures/``          preprint build (tracked)
    fig4_delivery_frequency_mix.pdf
    fig5_cost_wait_fleet_heatmaps.pdf
    fig6_pareto_structural_breakdown.pdf

``paper/EWGT_2026_rev1/elsevier_source/``  Elsevier camera-ready (GITIGNORED)
    fig_SM_mix_pct_8P.pdf
    fig_grid_heatmap_6.pdf
    fig_structural_grid_6.pdf

``elsevier_source/`` is the copy that lies with the venue and is ignored by
git on purpose (§38.8) -- this script writes into it but NEVER stages it, and
refuses to run any git command at all.

Exit code
---------
0 only if every copy verifies PASS.  Any FAIL, MISSING or SRC_MISSING row
exits 1, so the gate can be wired into a checklist.

Usage
-----
    python scripts/revision/71_sync_paper_figs.py --dry-run       # report only
    python scripts/revision/71_sync_paper_figs.py                 # copy + verify
    python scripts/revision/71_sync_paper_figs.py --rev-dir results/revision_2026_08_head
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _figs_tables_v2 as H  # noqa: E402

DEFAULT_REV = ROOT / "results" / "revision_2026_08_v5"
PAPER_REV = ROOT / "paper" / "EWGT_2026_rev1"
PAPER_SUB = ROOT / "paper" / "EWGT_2026"

# source stem in <rev>/figures  ->  (preprint name, elsevier name)
FIGURE_MAP = {
    "fig4_freq_mix_two_plans": ("fig4_delivery_frequency_mix.pdf",
                                "fig_SM_mix_pct_8P.pdf"),
    "fig5_grid_heatmap_v2": ("fig5_cost_wait_fleet_heatmaps.pdf",
                             "fig_grid_heatmap_6.pdf"),
    "fig6_structural_v2": ("fig6_pareto_structural_breakdown.pdf",
                           "fig_structural_grid_6.pdf"),
}

# Companion figures the manuscript does not \includegraphics yet.  Off by
# default: syncing a file the .tex never references only invites the next
# stale-figure confusion.
COMPANION_MAP = {
    "fig5b_offdiagonal_v2": ("fig5b_offdiagonal_lens_plan.pdf",
                             "fig_grid_heatmap_offdiag.pdf"),
    "fig4b_mean_days": ("fig4b_mean_delivery_days.pdf",
                        "fig_SM_mean_days.pdf"),
}


def plan(rev: Path, include_companions: bool) -> list[tuple[Path, Path, str]]:
    src_dir = rev / "figures"
    mapping = dict(FIGURE_MAP)
    if include_companions:
        mapping.update(COMPANION_MAP)
    jobs = []
    for stem, (preprint, elsevier) in mapping.items():
        src = src_dir / f"{stem}.pdf"
        jobs.append((src, PAPER_REV / "figures" / preprint, "preprint"))
        jobs.append((src, PAPER_REV / "elsevier_source" / elsevier,
                     "elsevier (gitignored)"))
    return jobs


def submission_twin(dst: Path) -> Path | None:
    """The frozen submission file with the same name, if there is one."""
    if dst.parent.name != "figures":
        return None
    cand = PAPER_SUB / "figures" / dst.name
    return cand if cand.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rev-dir", default=str(DEFAULT_REV))
    ap.add_argument("--dry-run", action="store_true",
                    help="verify only; copy nothing")
    ap.add_argument("--include-companions", action="store_true",
                    help="also sync fig5b/fig4b (not referenced by the .tex)")
    args = ap.parse_args(argv)

    rev = Path(args.rev_dir)
    if not rev.is_absolute():
        rev = (ROOT / rev).resolve()

    jobs = plan(rev, args.include_companions)
    mode = "DRY RUN (no file written)" if args.dry_run else "COPY + VERIFY"
    print("=" * 100)
    print(f"G7 paper-figure sync -- {mode}")
    print(f"source: {rev / 'figures'}")
    print("=" * 100)

    rows = []
    for src, dst, kind in jobs:
        if not args.dry_run:
            if not src.exists():
                rows.append(dict(kind=kind, **H.verify_copy(src, dst)))
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        r = H.verify_copy(src, dst)
        r["kind"] = kind
        twin = submission_twin(dst)
        r["stale"] = ("yes" if (twin is not None and dst.exists()
                                and H.md5_of(twin) == H.md5_of(dst))
                      else "no" if twin is not None else "-")
        rows.append(r)

    w = max(len(Path(r["dst"]).name) for r in rows) + 2
    print(f"\n{'status':>11}  {'destination':<{w}} {'kind':<22} "
          f"{'src md5':<34} {'dst md5':<34} {'==submission?':>13}")
    print("-" * (11 + 2 + w + 1 + 22 + 1 + 34 + 1 + 34 + 1 + 13))
    n_bad = 0
    for r in rows:
        ok = r["status"] == "PASS"
        n_bad += 0 if ok else 1
        print(f"{r['status']:>11}  {Path(r['dst']).name:<{w}} "
              f"{r['kind']:<22} {str(r['src_md5']):<34} "
              f"{str(r['dst_md5']):<34} {r.get('stale', '-'):>13}")

    stale = [r for r in rows if r.get("stale") == "yes"]
    print()
    if stale:
        print(f"WARNING: {len(stale)} destination(s) are still md5-identical "
              "to the frozen submission figure -- that is exactly the §38.8 "
              "trap. Re-run without --dry-run.")
    if n_bad:
        print(f"FAIL: {n_bad} of {len(rows)} copies did not verify.")
        if args.dry_run:
            print("      (expected in --dry-run before the first real sync)")
        return 1
    print(f"PASS: all {len(rows)} copies verify byte-identical.")
    print("\nReminder: paper/EWGT_2026_rev1/elsevier_source/ is gitignored "
          "on purpose (§38.8) -- never `git add` it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
