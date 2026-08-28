"""Structural guard for a manuscript .tex + its built PDF.

Born out of the Part-C incident (task 14C): the manuscript uses a ``% src:``
convention to record where each number comes from, and twice a paragraph was
swallowed by one of those comment blocks -- the text simply disappeared from
the PDF and nothing said so. Everything here is a tripwire for that class of
silent loss, and every check is a hard failure, never a warning:

1. **comment-swallow** -- no comment line may hide a LaTeX control sequence
   (``\\section``, ``\\caption``, ``\\label``, ``\\ref``, ``\\cite``,
   ``\\footnote`` and relatives). A commented-out ``\\section`` is almost
   always a paragraph that fell into a ``%`` block, not a deliberate edit;
   the few deliberate ones are named in :data:`DEFAULT_DRAFT_BLOCKS`, which
   is applied automatically, and extra ones with ``--ignore-prefix``. The
   guard **reports how many comment lines each exemption suppressed**, so a
   broad exemption can never hide a growing block in silence.
2. **page count** -- the built PDF's pages, read with ``pdftotext`` (pages
   are separated by form feeds), against ``--expect-pages``. This is the
   page-budget tripwire: a submission limit is not something to discover
   after the fact.
3. **bibitem count** -- ``\\bibitem`` entries in the .bbl against
   ``--expect-bibitems``. A reference that silently disappears with a
   swallowed paragraph shows up here first.
4. **cite resolution** -- every key cited in the .tex must resolve to a
   ``\\bibitem`` in the .bbl (or, when only a .bib is available, to a .bib
   entry). An unresolved key renders as a bold ``[?]`` that is easy to miss
   in a 17-page PDF.

The .bbl is produced by the build (``tectonic --keep-intermediates``), not
committed, so ``--bbl`` normally points into the build directory; without it
the guard looks for ``<stem>.bbl`` next to the .tex and next to the PDF, then
falls back to the ``\\bibliography{...}`` .bib next to the .tex (and says
which source it used -- it never silently checks nothing).

Usage -- this is the whole invocation; the draft-block exemptions are the
script's own and no longer have to be passed in::

    python scripts/paper/guard_tex.py --tex paper/EWGT_2026_rev1/tbc_preprint_main.tex \\
        --pdf build/tbc_preprint_main.pdf --bbl build/tbc_preprint_main.bbl \\
        --expect-pages 17 --expect-bibitems 23

    python scripts/paper/guard_tex.py --tex paper/EWGT_2026_rev1/supplementary.tex \\
        --pdf build/supplementary.pdf --expect-pages 7

Pass ``--no-default-ignores`` to see the manuscript's deliberate draft blocks
reported alongside real defects.

Exit code 0 if every requested check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

#: Control sequences that must never appear on a comment line. ``item``,
#: ``begin`` and ``end`` are included because a swallowed list or environment
#: is the same defect wearing different clothes.
CONTROL_SEQUENCES = (
    "section", "subsection", "subsubsection", "paragraph", "caption",
    "label", "ref", "autoref", "eqref", "cite", "citep", "citet",
    "footnote", "url", "includegraphics", "item", "begin", "end",
)
_HIDDEN_RE = re.compile(r"\\(" + "|".join(CONTROL_SEQUENCES) + r")\b")

#: Comment prefixes that mark a **deliberate** commented-out draft block in
#: ``paper/EWGT_2026_rev1/tbc_preprint_main.tex``. Applied by default so the
#: guard passes on a clean tree when invoked the way ``--help`` shows; turn
#: them off with ``--no-default-ignores``.
#:
#: These lived only in ``tests/unit/test_guard_tex.py`` until the final review
#: (M6): the script's own documented invocation then failed on a clean tree,
#: which teaches a reader to distrust it. The test now imports this list.
#:
#: What each one covers, as of the rev1 manuscript:
#:
#: ``"% \\subsection"`` / ``"% \\label"``
#:     Two subsections that moved to the supplementary and are kept as a
#:     record of the submitted method -- "Willingness to wait and residual
#:     handling" and "Fleet balancing" (the latter explicitly marked
#:     ``SUPERSEDED`` in the .tex). Narrow: they exempt exactly the
#:     ``\\subsection`` / ``\\label`` line, not the prose under it.
#: ``"% We propose a machine learning"``
#:     An alternative introduction paragraph, commented out whole. This one
#:     is **broad** -- it exempts a whole paragraph by its opening words, and
#:     the paragraph contains ``\\citep`` calls -- so if that text were ever
#:     re-swallowed the guard would stay silent about it. That is why the
#:     suppression counts are printed: this prefix must suppress exactly ONE
#:     line, and a rising count is the signal to look.
DEFAULT_DRAFT_BLOCKS: tuple[str, ...] = (
    "% \\subsection",
    "% \\label",
    "% We propose a machine learning",
)
_CITE_RE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}")
_BIBITEM_RE = re.compile(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}")
_BIB_ENTRY_RE = re.compile(r"^\s*@\w+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
_BIBLIOGRAPHY_RE = re.compile(r"\\bibliography\{([^}]+)\}")


class ToolMissing(RuntimeError):
    """A required external binary is not on PATH."""


# ─────────────────────────────────────────────────────────────────────────
# pure checks
# ─────────────────────────────────────────────────────────────────────────
def hidden_control_sequences(tex: str,
                             ignore_prefixes: tuple[str, ...] = ()
                             ) -> list[tuple[int, str]]:
    """``(line number, line)`` for every comment line hiding a control sequence.

    A line counts as a comment when its first non-blank character is ``%``
    (a mid-line ``%`` only comments out the rest of a line that is otherwise
    real text, which is not this defect). ``ignore_prefixes`` are matched
    against the stripped line, so a deliberately commented-out block can be
    named explicitly rather than switching the check off.
    """
    out = []
    for i, line in enumerate(tex.replace("\r\n", "\n").split("\n"), 1):
        s = line.lstrip()
        if not s.startswith("%"):
            continue
        if any(s.startswith(p) for p in ignore_prefixes):
            continue
        if _HIDDEN_RE.search(s):
            out.append((i, s[:120]))
    return out


def suppressed_by_prefix(tex: str, ignore_prefixes: tuple[str, ...]
                         ) -> dict[str, int]:
    """How many hiding comment lines each prefix in *ignore_prefixes* removes.

    An exemption that suppresses nothing is stale; one that suppresses more
    lines than it used to has quietly grown to cover text nobody vetted.
    Neither is visible unless the counts are printed, which is what this is
    for -- an exemption is a hole in a tripwire and must be as legible as the
    tripwire itself. Each line is attributed to the FIRST matching prefix, so
    the counts sum to the number of lines actually suppressed.
    """
    counts = dict.fromkeys(ignore_prefixes, 0)
    for line in tex.replace("\r\n", "\n").split("\n"):
        s = line.lstrip()
        if not s.startswith("%") or not _HIDDEN_RE.search(s):
            continue
        for p in ignore_prefixes:
            if s.startswith(p):
                counts[p] += 1
                break
    return counts


def bibitem_keys(bbl: str) -> list[str]:
    """Keys of every ``\\bibitem`` in a .bbl, in document order (duplicates
    kept, because a duplicated key is itself a defect worth seeing)."""
    return _BIBITEM_RE.findall(bbl)


def bib_entry_keys(bib: str) -> set[str]:
    """Keys of every ``@type{key,`` entry in a .bib."""
    return set(_BIB_ENTRY_RE.findall(bib))


def cited_keys(tex: str) -> set[str]:
    """Every key cited anywhere in the .tex, across all ``\\cite*`` variants."""
    keys: set[str] = set()
    for m in _CITE_RE.finditer(tex):
        keys.update(k.strip() for k in m.group(1).split(",") if k.strip())
    return keys


def pdftotext_page_count(pdf: Path, exe: str = "pdftotext") -> int:
    """Pages of ``pdf``, as ``pdftotext`` sees them.

    ``pdftotext`` separates pages with a form feed and emits a trailing one,
    so the page count is the number of form feeds. Raises ``ToolMissing`` if
    the binary is not on PATH -- the caller decides whether that is a skip or
    a failure; it is never silently treated as "no pages".
    """
    if shutil.which(exe) is None:
        raise ToolMissing(f"{exe} is not on PATH")
    res = subprocess.run([exe, str(pdf), "-"], capture_output=True, timeout=120)
    if res.returncode != 0:
        raise RuntimeError(
            f"{exe} failed on {pdf}: {res.stderr.decode(errors='replace')[:400]}")
    return res.stdout.count(b"\f")


# ─────────────────────────────────────────────────────────────────────────
# orchestration
# ─────────────────────────────────────────────────────────────────────────
def resolve_bibliography(tex_path: Path, pdf: Path | None,
                         bbl: Path | None, bib: Path | None
                         ) -> tuple[Path | None, Path | None]:
    """Find the .bbl (built) and/or .bib (source) backing the citations."""
    if bbl is None:
        for cand in ([tex_path.with_suffix(".bbl")]
                     + ([pdf.with_suffix(".bbl")] if pdf else [])):
            if cand.exists():
                bbl = cand
                break
    if bib is None:
        m = _BIBLIOGRAPHY_RE.search(tex_path.read_text(encoding="utf-8",
                                                       errors="replace"))
        if m:
            for name in m.group(1).split(","):
                cand = tex_path.parent / (name.strip() + ".bib")
                if cand.exists():
                    bib = cand
                    break
    return bbl, bib


def run_checks(tex_path: Path, pdf: Path | None = None,
               bbl: Path | None = None, bib: Path | None = None,
               expect_pages: int | None = None,
               expect_bibitems: int | None = None,
               ignore_prefixes: tuple[str, ...] = (),
               echo=print) -> list[str]:
    """Run every applicable check; return the list of failure messages."""
    failures: list[str] = []
    tex = tex_path.read_text(encoding="utf-8", errors="replace")

    hidden = hidden_control_sequences(tex, ignore_prefixes)
    echo(f"1. comment lines hiding a control sequence: {len(hidden)}"
         f"  {'OK' if not hidden else '*** FAIL ***'}")
    for i, s in hidden:
        echo(f"      line {i}: {s}")
    if ignore_prefixes:
        counts = suppressed_by_prefix(tex, ignore_prefixes)
        echo(f"   exempted draft blocks ({sum(counts.values())} line(s) "
             f"suppressed):")
        for p, n in counts.items():
            # A 0 is not automatically a defect: the defaults describe the
            # MAIN manuscript, and the supplementary legitimately matches
            # none of them. On a document that used to match, a 0 means the
            # block moved or was deleted and the exemption can go.
            note = "  <-- no match in this document" if n == 0 else ""
            echo(f"      {n:>3}x  {p!r}{note}")
    if hidden:
        failures.append(f"{len(hidden)} comment line(s) hide a control sequence "
                        f"in {tex_path.name}")

    if pdf is not None:
        pages = pdftotext_page_count(pdf)
        if expect_pages is None:
            echo(f"2. pages: {pages}  (no expectation given)")
        else:
            ok = pages == expect_pages
            echo(f"2. pages: {pages} (want {expect_pages})"
                 f"  {'OK' if ok else '*** FAIL ***'}")
            if not ok:
                failures.append(f"{pdf.name} has {pages} pages, expected "
                                f"{expect_pages}")
    else:
        echo("2. pages: skipped (no --pdf)")

    bbl_path, bib_path = resolve_bibliography(tex_path, pdf, bbl, bib)
    cited = cited_keys(tex)
    keys: set[str] = set()
    if bbl_path is not None:
        items = bibitem_keys(bbl_path.read_text(encoding="utf-8",
                                                errors="replace"))
        keys = set(items)
        if len(items) != len(keys):
            dupes = sorted({k for k in items if items.count(k) > 1})
            failures.append(f"duplicate \\bibitem key(s) in {bbl_path.name}: "
                            f"{dupes}")
            echo(f"3. bibitems: {len(items)} entries, {len(keys)} distinct "
                 f"*** FAIL *** duplicates {dupes}")
        elif expect_bibitems is None:
            echo(f"3. bibitems: {len(items)}  (no expectation given, "
                 f"from {bbl_path.name})")
        else:
            ok = len(items) == expect_bibitems
            echo(f"3. bibitems: {len(items)} (want {expect_bibitems}, "
                 f"from {bbl_path.name})  {'OK' if ok else '*** FAIL ***'}")
            if not ok:
                failures.append(f"{bbl_path.name} has {len(items)} \\bibitem "
                                f"entries, expected {expect_bibitems}")
    elif bib_path is not None:
        keys = bib_entry_keys(bib_path.read_text(encoding="utf-8",
                                                 errors="replace"))
        echo(f"3. bibitems: no .bbl built -- resolving citations against "
             f"{bib_path.name} ({len(keys)} entries) instead")
        if expect_bibitems is not None:
            failures.append("--expect-bibitems was given but no .bbl was found; "
                            "build the document first (a .bib has every entry, "
                            "not only the cited ones)")
    elif cited:
        failures.append(f"no .bbl and no .bib found for {tex_path.name} -- "
                        f"its {len(cited)} citation(s) cannot be checked")
        echo("3. bibitems: *** FAIL *** no bibliography source found")
    else:
        # A document that cites nothing (the supplementary, for instance)
        # legitimately has no bibliography -- that is not a defect.
        echo("3. bibitems: no bibliography and no citations -- nothing to check")

    if keys:
        missing = sorted(cited - keys)
        echo(f"4. cited keys: {len(cited)}, unresolved: {len(missing)}"
             f"  {'OK' if not missing else '*** FAIL ***'}")
        if missing:
            echo(f"      {missing}")
            failures.append(f"{len(missing)} cited key(s) do not resolve: "
                            f"{missing}")
    else:
        echo(f"4. cited keys: {len(cited)}, not checked (no bibliography source)")
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tex", required=True, type=Path)
    ap.add_argument("--pdf", type=Path,
                    help="built PDF; without it the page check is skipped")
    ap.add_argument("--bbl", type=Path, help="built .bbl (default: auto)")
    ap.add_argument("--bib", type=Path, help="source .bib (default: auto)")
    ap.add_argument("--expect-pages", type=int)
    ap.add_argument("--expect-bibitems", type=int)
    ap.add_argument("--ignore-prefix", action="append", default=[],
                    help="an ADDITIONAL comment prefix that is a deliberate "
                         "draft block; repeatable. The manuscript's own "
                         "blocks are already covered by DEFAULT_DRAFT_BLOCKS")
    ap.add_argument("--no-default-ignores", action="store_true",
                    help="do not apply DEFAULT_DRAFT_BLOCKS, so the "
                         "manuscript's deliberate draft blocks are reported "
                         "as defects too")
    a = ap.parse_args(argv)

    prefixes = tuple(a.ignore_prefix)
    if not a.no_default_ignores:
        prefixes = DEFAULT_DRAFT_BLOCKS + prefixes

    print(f"guard_tex: {a.tex}")
    failures = run_checks(a.tex, a.pdf, a.bbl, a.bib, a.expect_pages,
                          a.expect_bibitems, prefixes)
    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  * {f}")
        return 1
    print("\nPASS: every requested check is green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
