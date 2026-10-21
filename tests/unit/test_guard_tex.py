"""``scripts/paper/guard_tex.py`` -- the manuscript structure guard.

The guard exists because of a defect that already happened twice (task 14C):
the manuscript records provenance in ``% src:`` comment blocks, and a
paragraph slid into one of them -- the text vanished from the PDF and nothing
failed. These tests cover each tripwire on synthetic files, then run the real
guard on ``paper/EWGT_2026_rev1/tbc_preprint_main.tex``.

Rule 1a (a comment line hiding a LaTeX control sequence) missed the defect
twice more (task 14C/14D) because the swallowed sentence was ordinary prose
with inline math and no control sequence. Rule 1b closes that gap: a
``% src:`` provenance line, or one of its continuations, may never carry a
body-text marker (``$``, ``---``, ``~``), because provenance never needs any
of the three. Four real instances motivate its regression cases below,
quoted verbatim from
``.superpowers/sdd/2026-08-25-realistic-tours-implementation/
task-14c-fix-report.md`` (I1, sites :290/:364/:497 of the manuscript as
committed at ``df723d9``) and ``task-14d-report.md`` (the fourth, from
commit ``a2342ca``'s message). None of the four survives in the current
tree -- each was caught and fixed before or within the commit that
introduced it.

The end-to-end test needs ``tectonic`` (to build the PDF and the .bbl, which
is not committed) and ``pdftotext`` (to count pages). If either binary is
missing the test **skips with a message naming it** rather than passing
quietly. If tectonic is present but the build fails, that is a real failure
of the paper folder and the test says so instead of skipping.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper" / "EWGT_2026_rev1"
MAIN_TEX = PAPER / "tbc_preprint_main.tex"

#: Tripwire values for the committed manuscript (task 14C fix round 3,
#: commit f848de8). They are meant to be updated **deliberately, in the same
#: commit** as a real change to the manuscript -- a page or a reference that
#: moves on its own is exactly the silent loss this guard exists to catch.
EXPECTED_PAGES = 17
EXPECTED_BIBITEMS = 23

#: The three deliberately commented-out draft blocks in the manuscript: an
#: alternative introduction paragraph and two subsections that moved to the
#: supplementary (the second is explicitly labelled "SUPERSEDED" in the .tex).
#: Everything else on a comment line is treated as a swallowed paragraph.
#:
#: This list is the SCRIPT's (``guard_tex.DEFAULT_DRAFT_BLOCKS``), applied by
#: the guard automatically; the value here only PINS it. Final-review finding
#: M6: while the exemptions lived only in this test file, the guard failed on
#: a clean tree when run the way its own ``--help`` documents.
EXPECTED_DRAFT_BLOCKS = ("% \\subsection", "% \\label",
                         "% We propose a machine learning")


def _load():
    spec = importlib.util.spec_from_file_location(
        "_guard_tex", ROOT / "scripts" / "paper" / "guard_tex.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


# ─────────────────────────────────────────────────────────────────────────
# 1. comment-swallow
# ─────────────────────────────────────────────────────────────────────────
def test_a_commented_section_is_reported(mod):
    tex = "Real text.\n% \\section{Results}\nMore text.\n"
    assert mod.hidden_control_sequences(tex) == [(2, "% \\section{Results}")]


@pytest.mark.parametrize("hidden", [
    "% \\caption{x}", "%\\label{sec:a}", "  % \\ref{fig:1}",
    "% \\citep{Key2020}", "% \\footnote{note}", "% \\begin{itemize}",
])
def test_every_guarded_control_sequence_is_caught(mod, hidden):
    assert len(mod.hidden_control_sequences(f"a\n{hidden}\nb\n")) == 1


@pytest.mark.parametrize("benign", [
    "% src: results/revision_2026_08_v6/tables/tab_grid_full_v2.csv",
    "% ---------------------------------------------",
    "Real text with a trailing comment % \\section is not commented out",
    "% 100 % of the parcels",
])
def test_benign_comment_lines_are_not_reported(mod, benign):
    assert mod.hidden_control_sequences(f"a\n{benign}\nb\n") == []


def test_named_draft_blocks_are_exempt_but_nothing_else_is(mod):
    tex = ("% \\subsection{Moved to the supplementary}\n"
           "% \\caption{this one was swallowed}\n")
    out = mod.hidden_control_sequences(tex, ignore_prefixes=("% \\subsection",))
    assert [i for i, _ in out] == [2]


def test_the_draft_block_allow_list_lives_in_the_script(mod):
    """M6: the exemptions are the script's, not this test's private list.

    A guard whose documented invocation fails on a clean tree teaches its
    reader to ignore it. Pinning the list here means a new exemption has to
    be added deliberately, in the script, and shows up in review.
    """
    assert mod.DEFAULT_DRAFT_BLOCKS == EXPECTED_DRAFT_BLOCKS


def test_suppression_counts_make_every_exemption_visible(mod):
    """An exemption is a hole in a tripwire; the counts keep it legible.

    Guards the second-order risk in ``"% We propose a machine learning"``,
    which exempts a whole paragraph by its opening words: a rising count is
    the only signal that it has started covering text nobody vetted.
    """
    tex = ("% \\subsection{a}\n"
           "% \\subsection{b}\n"
           "% \\label{c}\n"
           "% src: a benign comment with no control sequence\n")
    counts = mod.suppressed_by_prefix(tex, ("% \\subsection", "% \\label",
                                            "% never matches anything"))
    assert counts == {"% \\subsection": 2, "% \\label": 1,
                      "% never matches anything": 0}


def test_default_ignores_are_applied_by_main_and_can_be_switched_off(mod, capsys):
    """``main`` applies the allow-list itself; ``--no-default-ignores`` does not."""
    rc = mod.main(["--tex", str(MAIN_TEX)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "comment lines hiding a control sequence: 0" in out
    assert "exempted draft blocks (5 line(s) suppressed)" in out
    assert "% src: line(s) carrying a body-text marker ($, ---, ~): 0" in out

    rc = mod.main(["--tex", str(MAIN_TEX), "--no-default-ignores"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "comment lines hiding a control sequence: 5" in out
    assert "% src: line(s) carrying a body-text marker ($, ---, ~): 0" in out


# ─────────────────────────────────────────────────────────────────────────
# 1b. % src: provenance lines carrying a body-text marker
# ─────────────────────────────────────────────────────────────────────────
def test_a_swallowed_sentence_after_src_is_reported(mod):
    """The motivating defect: a % src: comment sits at the start of a line
    that still carries body text, and LaTeX's % swallows the rest of the
    line. The swallowed sentence has no LaTeX control sequence, so rule 1a
    is blind to it -- this is exactly why rule 1b exists."""
    tex = ("Some preceding sentence.\n"
           "% src: results/revision_2026_08_v6/foo.csv, column bar\n"
           "%      The Monday--Saturday coefficient of variation is "
           "$0.139$.\n"
           "More text follows.\n")
    assert mod.hidden_control_sequences(tex) == []
    hits = mod.src_provenance_body_markers(tex)
    assert [i for i, _ in hits] == [3]


@pytest.mark.parametrize("benign", [
    "% src: results/revision_2026_08_v6/tables/tab_x.csv, column y",
    "% src: results/revision_2026_08_v6/tables/tab_grid_full_v2.csv, "
    "row penalty=0",
])
def test_legitimate_src_lines_are_not_reported(mod, benign):
    assert mod.src_provenance_body_markers(f"a\n{benign}\nb\n") == []


def test_src_continuations_are_covered_not_just_the_first_line(mod):
    """The swallow can land on a wrapped continuation line, several lines
    after the initial % src: line -- the whole block must stay in scope,
    not only its first line."""
    tex = ("% src: a.csv, col\n"
           "%      benign continuation\n"
           "%      Swallowed body text with $math$ in it.\n")
    hits = mod.src_provenance_body_markers(tex)
    assert [i for i, _ in hits] == [3]


def test_src_block_ends_at_a_non_continuation_line(mod):
    """A % comment that follows a src block without the block's own
    indentation is a new, unrelated comment, not a continuation -- it must
    not be swept into the block (or out of it, for a later real src line)."""
    tex = ("% src: a.csv, col\n"
           "% an unrelated single-space comment with $math$ in it\n"
           "% src: b.csv, col2\n"
           "%      real continuation with $math$\n")
    hits = mod.src_provenance_body_markers(tex)
    assert [i for i, _ in hits] == [4]


def test_src_ignore_prefix_exempts_a_named_line_but_nothing_else(mod):
    """Rule 1b shares rule 1a's allow-list mechanism: a line starting with
    an ignore prefix is skipped, everything else is still checked."""
    tex = ("% src: exempt line with $math$ that would otherwise be flagged\n"
           "% src: another line with $math$ that is NOT exempt\n")
    out = mod.src_provenance_body_markers(
        tex, ignore_prefixes=("% src: exempt line",))
    assert [i for i, _ in out] == [2]


def test_the_rejected_sentence_boundary_heuristic_would_have_misfired(mod):
    """A prior candidate rule -- flag a bare sentence boundary, literal ". "
    followed by a capital letter -- was tried first and rejected (task 14D)
    because it false-positives on legitimate provenance exactly like this,
    quoted verbatim from the manuscript's own committed text (around line
    525 of tbc_preprint_main.tex)."""
    tex = ("% src: over-pricing ranges below -- clean-basis total-cost gaps\n"
           "%      (sum(pred) - sum(actual)) / sum(actual) over the rows of\n"
           "%      validation/tab_vroom_v2.csv with n_unassigned == 0 and\n"
           "%      jobs_removed == 0 and vroom_status in {OK, CACHED}. BOTH\n"
           "%      ranges span the SAME six consolidated points.\n")
    assert mod.src_provenance_body_markers(tex) == []


# ─────────────────────────────────────────────────────────────────────────
# 1b regression cases: the four real instances behind task 14C/14D
# ─────────────────────────────────────────────────────────────────────────
# Quoted verbatim from the manuscript's own git history -- see the module
# docstring above for sourcing. Each was committed briefly and fixed in the
# same or a later commit; none survives in the current tree.
def test_regression_i1_site_290_framework_paragraph(mod):
    """task-14c-fix-report.md I1, site :290 (commit df723d9). Swallowed the
    only in-text refs to Fig. 1/2, the code-availability footnote, and the
    postal-code-decomposition sentence. This one is caught by rule 1a
    (\\autoref, \\footnote and \\url survive on the comment line); it
    happens to carry none of rule 1b's markers, so 1b does not need to
    fire here too -- included for completeness of the four known sites."""
    line = (
        r"% src: tab_grid_full_v2.csv, row penalty=0 share_willing=0, "
        r"routing_cost_plan1_eur = 1,898,091. As \autoref{fig:input_data} "
        r"shows, demand varies strongly between dense urban and sparse "
        r"rural postal-code areas, peaks midweek, and is lowest on "
        r"Saturday. To make repeated routing evaluations tractable, we "
        r"decomposed the study area into postal-code-level routing "
        r"subproblems while preserving the demand, spatial, and "
        r"depot-access characteristics relevant for delivery cost. The "
        r"pipeline structure is summarized in \autoref{fig:framework}."
        r"\footnote{The implementation of the proposed framework is "
        r"available at: \url{https://github.com/TUBS-IVS/"
        r"vroom-valhalla-lmd-hannover}}"
    )
    tex = f"Preceding text.\n{line}\nMore text.\n"
    assert len(mod.hidden_control_sequences(tex)) == 1
    assert mod.src_provenance_body_markers(tex) == []


def test_regression_i1_site_364_shadow_price_paragraph(mod):
    """task-14c-fix-report.md I1, site :364 (commit df723d9). Swallowed the
    cross-provider independence statement, the shadow-price definition of
    P, and the (P, theta) grid definition. Caught by both rules here: a
    \\citep/\\eqref survive on the comment line (1a) and the same text
    carries $, ---, and ~ (1b)."""
    start = ("% src: results/revision_2026_08_v6/gates_report.md (G1a "
             "PASS: 1,656 checked, 0 hard,\n")
    swallowed = (
        r"%      5 tolerated); tolerances G1A_TOLERANCE_FLAT_EUR = 20.0, "
        r"G1A_TOLERANCE_REL = 0.005 No parcels, routes, or vehicles are "
        r"shared across providers, and system-wide figures are obtained "
        r"solely by aggregating the independently optimized provider "
        r"solutions. The service penalty $P$ is a shadow price "
        r"\citep{BoydVandenberghe2004} of waiting --- the marginal cost "
        r"the operator trades for one parcel-day of delay --- and enters "
        r"\eqref{eq:schedule_objective} as a steering term only, which "
        r"keeps the cost-service trade-off continuous. We swept "
        r"$(P, \theta)$ over eight $P$ levels from $0$ to $10$~\euro/p/d "
        r"and eleven $\theta$ levels from $0$ to $1$ in steps of $0.1$."
    )
    tex = f"Preceding text.\n{start}{swallowed}\nMore text.\n"
    assert len(mod.hidden_control_sequences(tex)) == 1
    hits = mod.src_provenance_body_markers(tex)
    assert [i for i, _ in hits] == [3]


def test_regression_i1_site_497_equity_paragraph(mod):
    """task-14c-fix-report.md I1, site :497 (commit df723d9). Swallowed the
    equity sentence (a protected item) and the per-depot mechanism sentence
    with its Fig. S6 pointer. Caught by both rules here: two \\citep calls
    survive on the comment line (1a) and the Fig.~S6 tie survives (1b)."""
    start = ("% src: tab_per_cell_structural_v2.csv, level=cell "
             "lens='routing lens' plan=balanced\n")
    swallowed = (
        r"%      value=saving_pct, medians over share_willing <= 0.9 This "
        r"efficiency pattern creates an equity concern: service quality "
        r"between urban and rural areas should be maintained "
        r"\citep{Pereira04032017}, especially since rural and suburban "
        r"deliveries account for a substantial share of LMD activity "
        r"\citep{BIENZEISLER2026104682}. The depot-structure mechanism "
        r"runs alongside it and is visible per depot rather than per "
        r"area: depots serving a single cell stay at six delivery days "
        r"across the entire adoption range while those serving thirteen "
        r"or more fall to two, although only DHL operates a multi-depot "
        r"network here, so that half of the mechanism is evidenced "
        r"within one carrier (Supplementary Fig.~S6)."
    )
    tex = f"Preceding text.\n{start}{swallowed}\nMore text.\n"
    assert len(mod.hidden_control_sequences(tex)) == 1
    hits = mod.src_provenance_body_markers(tex)
    assert [i for i, _ in hits] == [3]


def test_regression_monday_saturday_coefficient_of_variation(mod):
    """task-14d-report.md / commit a2342ca's message: adding the Fig. S12
    pointer's % src: block re-swallowed this sentence -- the same defect
    class, a third time in this file's real history. Rule 1a passes it (no
    control sequence anywhere in the sentence); this is the case that
    motivated rule 1b."""
    start = ("% src: results/revision_2026_08_v6/tables/"
             "tab_fleet_week_by_provider_v2.csv,\n")
    swallowed = (
        r"%      peak_baseline / peak_plan1 / peak_plan2 summed over the "
        r"seven providers The Monday--Saturday coefficient of variation "
        r"of the system fleet falls from $0.139$ in the baseline to "
        r"$0.023$ at $(0.25, 1)$ and $0.044$ at $(0.5, 1)$, that is by "
        r"$84$ and $68\%$, with a grid maximum of $87\%$ at $(0, 0.7)$."
    )
    tex = f"Preceding text.\n{start}{swallowed}\nMore text.\n"
    assert mod.hidden_control_sequences(tex) == []
    hits = mod.src_provenance_body_markers(tex)
    assert [i for i, _ in hits] == [3]


# ─────────────────────────────────────────────────────────────────────────
# 2. pages
# ─────────────────────────────────────────────────────────────────────────
def test_page_count_refuses_when_pdftotext_is_missing(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod.shutil, "which", lambda _: None)
    with pytest.raises(mod.ToolMissing, match="pdftotext"):
        mod.pdftotext_page_count(tmp_path / "x.pdf")


def test_page_count_is_the_number_of_form_feeds(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod.shutil, "which", lambda _: "pdftotext")

    class _Res:
        returncode = 0
        stdout = b"page one\fpage two\fpage three\f"
        stderr = b""

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _Res())
    assert mod.pdftotext_page_count(tmp_path / "x.pdf") == 3


# ─────────────────────────────────────────────────────────────────────────
# 3/4. bibliography
# ─────────────────────────────────────────────────────────────────────────
def test_bibitem_keys_reads_both_forms_and_keeps_duplicates(mod):
    bbl = ("\\bibitem[Author(2020)]{A2020}\n"
           "\\bibitem{B2021}\n"
           "\\bibitem[C(2022)]{A2020}\n")
    assert mod.bibitem_keys(bbl) == ["A2020", "B2021", "A2020"]


def test_bib_entry_keys(mod):
    bib = "@article{Smith2020,\n title={x}\n}\n@book{ Jones1999 ,\n}\n"
    assert mod.bib_entry_keys(bib) == {"Smith2020", "Jones1999"}


def test_cited_keys_covers_every_cite_variant(mod):
    tex = (r"\citep{A} \citet{B} \cite[p.~5]{C} \citep[e.g.][]{D,E} "
           r"\citeauthor*{F}")
    assert mod.cited_keys(tex) == {"A", "B", "C", "D", "E", "F"}


# ─────────────────────────────────────────────────────────────────────────
# run_checks / main on synthetic files: every failure path is non-zero
# ─────────────────────────────────────────────────────────────────────────
def _mini(tmp_path: Path, tex_body: str, bbl_body: str) -> tuple[Path, Path]:
    tex = tmp_path / "doc.tex"
    tex.write_text(tex_body, encoding="utf-8")
    bbl = tmp_path / "doc.bbl"
    bbl.write_text(bbl_body, encoding="utf-8")
    return tex, bbl


def test_run_checks_is_green_on_a_consistent_document(mod, tmp_path):
    tex, bbl = _mini(tmp_path, "Text \\citep{A2020}.\n",
                     "\\bibitem[A(2020)]{A2020}\n")
    assert mod.run_checks(tex, bbl=bbl, expect_bibitems=1, echo=lambda *a: None) == []


def test_run_checks_reports_an_unresolved_citation(mod, tmp_path):
    tex, bbl = _mini(tmp_path, "Text \\citep{Missing2020}.\n",
                     "\\bibitem[A(2020)]{A2020}\n")
    fails = mod.run_checks(tex, bbl=bbl, echo=lambda *a: None)
    assert any("do not resolve" in f and "Missing2020" in f for f in fails)


def test_run_checks_reports_a_bibitem_count_change(mod, tmp_path):
    tex, bbl = _mini(tmp_path, "Text \\citep{A2020}.\n",
                     "\\bibitem[A(2020)]{A2020}\n")
    fails = mod.run_checks(tex, bbl=bbl, expect_bibitems=23,
                           echo=lambda *a: None)
    assert any("expected 23" in f for f in fails)


def test_run_checks_reports_duplicate_bibitem_keys(mod, tmp_path):
    tex, bbl = _mini(tmp_path, "Text \\citep{A2020}.\n",
                     "\\bibitem{A2020}\n\\bibitem{A2020}\n")
    fails = mod.run_checks(tex, bbl=bbl, echo=lambda *a: None)
    assert any("duplicate" in f for f in fails)


def test_run_checks_refuses_a_document_that_cites_with_no_bibliography_source(
        mod, tmp_path):
    tex = tmp_path / "doc.tex"
    tex.write_text("Text \\citep{A2020}.\n", encoding="utf-8")
    fails = mod.run_checks(tex, echo=lambda *a: None)
    assert any("cannot be checked" in f for f in fails)


def test_a_document_without_citations_needs_no_bibliography(mod, tmp_path):
    """The supplementary cites nothing; that is not a defect, so the guard
    must not invent one (it would make the check useless by crying wolf)."""
    tex = tmp_path / "doc.tex"
    tex.write_text("Text with no citations.\n", encoding="utf-8")
    assert mod.run_checks(tex, echo=lambda *a: None) == []


def test_run_checks_refuses_expect_bibitems_without_a_built_bbl(mod, tmp_path):
    tex = tmp_path / "doc.tex"
    tex.write_text("Text \\citep{A2020}.\n\\bibliography{refs}\n",
                   encoding="utf-8")
    (tmp_path / "refs.bib").write_text("@article{A2020,\n}\n", encoding="utf-8")
    fails = mod.run_checks(tex, expect_bibitems=23, echo=lambda *a: None)
    assert any("no .bbl was found" in f for f in fails)


def test_main_exits_non_zero_on_a_swallowed_paragraph(mod, tmp_path, capsys):
    tex, bbl = _mini(tmp_path, "% \\section{Results}\n",
                     "\\bibitem{A2020}\n")
    assert mod.main(["--tex", str(tex), "--bbl", str(bbl)]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_main_exits_zero_when_everything_is_green(mod, tmp_path, capsys):
    tex, bbl = _mini(tmp_path, "Text \\citep{A2020}.\n", "\\bibitem{A2020}\n")
    assert mod.main(["--tex", str(tex), "--bbl", str(bbl),
                     "--expect-bibitems", "1"]) == 0
    assert "PASS" in capsys.readouterr().out


# ─────────────────────────────────────────────────────────────────────────
# end-to-end on the real manuscript
# ─────────────────────────────────────────────────────────────────────────
def test_the_committed_manuscript_passes_the_guard(mod, tmp_path, capsys):
    """Build ``tbc_preprint_main.tex`` and run every check against it.

    Skips loudly when the toolchain is absent; a tectonic that IS present but
    cannot build the paper folder is a real defect, so that fails.
    """
    for exe in ("tectonic", "pdftotext"):
        if shutil.which(exe) is None:
            pytest.skip(f"{exe} is not on PATH -- the manuscript guard cannot "
                        f"run end to end (install it to exercise this test)")
    assert MAIN_TEX.exists(), MAIN_TEX

    build = subprocess.run(
        ["tectonic", "--keep-intermediates", "--outdir", str(tmp_path),
         str(MAIN_TEX)], capture_output=True, timeout=900)
    pdf, bbl = tmp_path / "tbc_preprint_main.pdf", tmp_path / "tbc_preprint_main.bbl"
    if build.returncode != 0 or not pdf.exists():
        pytest.fail("tectonic failed to build the committed manuscript:\n"
                    + build.stderr.decode(errors="replace")[-2000:])
    assert bbl.exists(), "tectonic produced no .bbl (--keep-intermediates)"

    # No --ignore-prefix: the guard applies DEFAULT_DRAFT_BLOCKS itself, so
    # this is exactly the invocation its --help documents (finding M6).
    argv = ["--tex", str(MAIN_TEX), "--pdf", str(pdf), "--bbl", str(bbl),
            "--expect-pages", str(EXPECTED_PAGES),
            "--expect-bibitems", str(EXPECTED_BIBITEMS)]
    rc = mod.main(argv)
    out = capsys.readouterr().out
    assert rc == 0, (
        "the committed manuscript no longer passes its own structure guard "
        "(if the change was deliberate, update EXPECTED_PAGES / "
        f"EXPECTED_BIBITEMS in this file in the same commit):\n{out}")
    assert f"pages: {EXPECTED_PAGES}" in out
    assert f"bibitems: {EXPECTED_BIBITEMS}" in out
    assert "% src: line(s) carrying a body-text marker ($, ---, ~): 0" in out
