"""``scripts/paper/guard_tex.py`` -- the manuscript structure guard.

The guard exists because of a defect that already happened twice (task 14C):
the manuscript records provenance in ``% src:`` comment blocks, and a
paragraph slid into one of them -- the text vanished from the PDF and nothing
failed. These tests cover each tripwire on synthetic files, then run the real
guard on ``paper/EWGT_2026_rev1/tbc_preprint_main.tex``.

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
DRAFT_BLOCKS = ("% \\subsection", "% \\label",
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

    argv = ["--tex", str(MAIN_TEX), "--pdf", str(pdf), "--bbl", str(bbl),
            "--expect-pages", str(EXPECTED_PAGES),
            "--expect-bibitems", str(EXPECTED_BIBITEMS)]
    for prefix in DRAFT_BLOCKS:
        argv += ["--ignore-prefix", prefix]
    rc = mod.main(argv)
    out = capsys.readouterr().out
    assert rc == 0, (
        "the committed manuscript no longer passes its own structure guard "
        "(if the change was deliberate, update EXPECTED_PAGES / "
        f"EXPECTED_BIBITEMS in this file in the same commit):\n{out}")
    assert f"pages: {EXPECTED_PAGES}" in out
    assert f"bibitems: {EXPECTED_BIBITEMS}" in out
