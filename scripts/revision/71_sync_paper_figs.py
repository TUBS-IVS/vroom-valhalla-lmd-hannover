"""Gate G7: sync the revision figures into the paper folders, md5-verified.

The trap this exists to close (Kompendium §38.8): ``paper/EWGT_2026_rev1/``
has three times been found holding **md5-identical copies of the submitted
figures** while everybody assumed it held the revision.  A stale Fig. 5
survived a full round that way.  So this script never assumes.

Provenance first, copying second
--------------------------------
The source is **not** a directory this script guesses.  ``70_`` writes a
provenance manifest (``<rev>/figures/manifest.json``: rev dir, git HEAD,
timestamp, md5 of every produced figure and of the four grid CSVs) and this
script takes its source from that file.  A bare call therefore cannot sync a
grid nobody rendered:

* ``--manifest PATH``  -- use exactly that render.
* ``--rev-dir DIR``    -- use ``DIR/figures/manifest.json`` and refuse if the
  manifest was written for a different grid.
* neither flag        -- discover ``results/*/figures/manifest.json``.  Zero
  matches refuses ("run 70_ first"); **two or more matches also refuses** and
  lists them, because "the newest one" is exactly the guess that puts the
  wrong grid in the paper.

Before a single byte is copied the render is checked for staleness
(``H.check_manifest_fresh``) and the run aborts on any of:

* a grid CSV changed after the render -- the figures no longer show what the
  tables say;
* a figure's md5 differs from the manifest -- it was edited or replaced;
* a figure is older than the newest grid CSV -- it was carried over from an
  earlier render.

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
runs no git command at all.

Exit code
---------
0 only if the provenance check passes AND every copy verifies PASS AND no
destination is still identical to the frozen submission.  That last one is a
FAILURE, not a warning: a destination byte-identical to the submitted figure
is the §38.8 trap itself, and a gate that prints it and then exits 0 is not a
gate.  ``--allow-identical-to-submission`` is the deliberate override for the
case where a revision figure genuinely did not change.

Usage
-----
    python scripts/revision/71_sync_paper_figs.py --dry-run       # report only
    python scripts/revision/71_sync_paper_figs.py                 # copy + verify
    python scripts/revision/71_sync_paper_figs.py --rev-dir results/revision_2026_08_head
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _figs_tables_v2 as H  # noqa: E402

RESULTS = ROOT / "results"
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


class ProvenanceError(RuntimeError):
    """The source render cannot be identified or cannot be trusted."""


def resolve_manifest(manifest: str | None, rev_dir: str | None) -> Path:
    """Find the one render this sync is allowed to copy from."""
    if manifest:
        path = Path(manifest)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        if not path.exists():
            raise ProvenanceError(f"--manifest {path} does not exist")
        return path
    if rev_dir:
        rev = Path(rev_dir)
        if not rev.is_absolute():
            rev = ROOT / rev
        rev = rev.resolve()
        path = rev / "figures" / H.MANIFEST_NAME
        if not path.exists():
            raise ProvenanceError(
                f"{path} does not exist -- run\n"
                f"    python scripts/revision/70_figs_tables_v2.py "
                f"--rev-dir {rev_dir}\n"
                "before syncing that grid into the paper")
        doc = H.read_manifest(path)
        recorded = os.path.normcase(str(Path(doc["rev_dir"]).resolve()))
        typed = os.path.normcase(str(rev))
        if recorded != typed:
            raise ProvenanceError(
                f"{path} was written for {doc['rev_dir']}, not for {rev} -- "
                "re-run 70_ on this grid rather than syncing someone "
                "else's render")
        return path
    found = H.find_manifests(RESULTS)
    if not found:
        raise ProvenanceError(
            f"no render manifest under {RESULTS}/*/figures/ -- run\n"
            "    python scripts/revision/70_figs_tables_v2.py "
            "--rev-dir <grid>\n"
            "first; this script refuses to guess which grid to sync")
    if len(found) > 1:
        listing = "\n".join(
            f"    --manifest {f.relative_to(ROOT)}   "
            f"(rev_dir={Path(H.read_manifest(f)['rev_dir']).name}, "
            f"{H.read_manifest(f)['rendered_utc']})" for f in found)
        raise ProvenanceError(
            f"{len(found)} render manifests exist and this script will not "
            "pick one for you -- naming the newest is exactly how the wrong "
            "grid reaches the paper. Pass one explicitly:\n" + listing)
    return found[0]


def plan(fig_dir: Path, include_companions: bool
         ) -> list[tuple[Path, Path, str]]:
    mapping = dict(FIGURE_MAP)
    if include_companions:
        mapping.update(COMPANION_MAP)
    jobs = []
    for stem, (preprint, elsevier) in mapping.items():
        src = fig_dir / f"{stem}.pdf"
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
    ap.add_argument("--manifest", default=None,
                    help="render manifest to sync from "
                         "(<rev>/figures/manifest.json)")
    ap.add_argument("--rev-dir", default=None,
                    help="grid directory; its figures/manifest.json is used")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify only; copy nothing")
    ap.add_argument("--include-companions", action="store_true",
                    help="also sync fig5b/fig4b (not referenced by the .tex)")
    ap.add_argument("--allow-identical-to-submission", action="store_true",
                    help="permit a destination byte-identical to the frozen "
                         "submission figure (default: that FAILS the gate)")
    args = ap.parse_args(argv)

    mode = "DRY RUN (no file written)" if args.dry_run else "COPY + VERIFY"
    print("=" * 100)
    print(f"G7 paper-figure sync -- {mode}")

    # ---- provenance --------------------------------------------------
    try:
        mpath = resolve_manifest(args.manifest, args.rev_dir)
        doc = H.read_manifest(mpath)
    except (ProvenanceError, AssertionError) as exc:
        print("=" * 100)
        print(f"REFUSED: {exc}")
        return 1

    fig_dir = mpath.parent
    print(f"manifest: {mpath}")
    print(f"  rev_dir   {doc['rev_dir']}")
    print(f"  git HEAD  {doc['git_head']}")
    print(f"  rendered  {doc['rendered_utc']}")
    print(f"  figures   {len(doc['figures'])} files, "
          f"{len(doc['grid_csvs'])} grid tables")
    print("=" * 100)

    stale = H.check_manifest_fresh(doc, fig_dir)
    if stale:
        print("\nREFUSED -- the render is stale, nothing copied:")
        for r in stale:
            print(f"  * {r}")
        rev_for_msg = Path(doc["rev_dir"])
        try:
            rev_for_msg = rev_for_msg.relative_to(ROOT)
        except ValueError:
            pass          # rev_dir is not under ROOT -- print it absolute
        print("\nRe-run 70_figs_tables_v2.py --rev-dir "
              f"{rev_for_msg} and try again.")
        return 1

    jobs = plan(fig_dir, args.include_companions)
    missing_src = [s for s, _, _ in jobs if not s.exists()]
    if missing_src:
        print("\nREFUSED -- source figure(s) absent, nothing copied:")
        for m in sorted(set(missing_src)):
            print(f"  * {m}")
        return 1

    # ---- copy + verify -----------------------------------------------
    rows = []
    for src, dst, kind in jobs:
        if not args.dry_run:
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
        bad = r["status"] != "PASS"
        n_bad += 1 if bad else 0
        print(f"{r['status']:>11}  {Path(r['dst']).name:<{w}} "
              f"{r['kind']:<22} {str(r['src_md5']):<34} "
              f"{str(r['dst_md5']):<34} {r.get('stale', '-'):>13}")

    identical = [r for r in rows if r.get("stale") == "yes"]
    print()
    if identical:
        # §38.8: this is the trap, not a note about it. It fails the gate
        # unless the operator says out loud that the figure is unchanged.
        verdict = ("WARNING (overridden by --allow-identical-to-submission)"
                   if args.allow_identical_to_submission else "FAIL")
        print(f"{verdict}: {len(identical)} destination(s) are byte-identical "
              "to the frozen submission figure:")
        for r in identical:
            print(f"  * {Path(r['dst']).name}")
        print("  Either 70_ rendered a figure that did not change, or the "
              "source directory is stale. Verify before shipping; pass "
              "--allow-identical-to-submission only when the revision "
              "genuinely reproduces the submitted figure.")
        if not args.allow_identical_to_submission:
            n_bad += len(identical)

    if n_bad:
        print(f"\nFAIL: {n_bad} of {len(rows)} check(s) did not pass.")
        if args.dry_run:
            print("      Re-run WITHOUT --dry-run to perform the copy, then "
                  "re-check.")
        else:
            print("      The copy ran; the failures above are real and the "
                  "paper folders must not be trusted until they clear.")
        return 1
    print(f"PASS: all {len(rows)} copies verify byte-identical, and no "
          "destination matches the frozen submission.")
    print("\nReminder: paper/EWGT_2026_rev1/elsevier_source/ is gitignored "
          "on purpose (§38.8) -- never `git add` it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
