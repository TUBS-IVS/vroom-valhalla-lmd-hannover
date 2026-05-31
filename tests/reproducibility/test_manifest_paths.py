"""Verify every artefact referenced by paper/EWGT_2026/MANIFEST.md exists.

If a paper claim is moved or deleted in a future refactor, this test
breaks loudly so a reviewer's "click through every figure to verify"
walk-through never hits a 404.

Strategy: parse MANIFEST.md, extract every relative path that ends in
.pdf / .png / .csv / .tex / .bib / .md, and assert each one exists on
disk.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "paper" / "EWGT_2026"
MANIFEST = PAPER_DIR / "MANIFEST.md"

# Path-like tokens we count as references. We look at any backticked
# filename ending in one of these extensions, treating it as relative to
# either paper/EWGT_2026/ or the repo root.
_EXT_RE = re.compile(
    r"`([\w./\\-]+\.(?:pdf|png|csv|tex|bib|md))`",
    re.IGNORECASE,
)


def _resolved_path(rel: str) -> Path | None:
    """Resolve a manifest reference to a real path or None if not found."""
    # Strip any leading './' or directory separator.
    rel = rel.lstrip("./").replace("\\", "/")
    candidates = [
        PAPER_DIR / rel,
        REPO_ROOT / rel,
        # Sometimes MANIFEST references files relative to the paper dir
        # without the "paper/EWGT_2026/" prefix.
        PAPER_DIR / "figures" / Path(rel).name,
        PAPER_DIR / "tables" / Path(rel).name,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


@pytest.fixture(scope="module")
def manifest_text() -> str:
    assert MANIFEST.exists(), f"MANIFEST.md not found at {MANIFEST}"
    return MANIFEST.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def manifest_paths(manifest_text: str) -> list[str]:
    return [m.group(1) for m in _EXT_RE.finditer(manifest_text)]


def test_manifest_has_paper_paths(manifest_paths: list[str]) -> None:
    """At least the six numbered paper figures must be referenced."""
    fig_names = {p for p in manifest_paths if p.startswith("fig")}
    expected = {
        "fig1_input_data_lsp_volumes.pdf",
        "fig2_framework.tex",
        "fig3_methodology_daganzo_progression.pdf",
        "fig4_delivery_frequency_mix.pdf",
        "fig5_cost_wait_fleet_heatmaps.pdf",
        "fig6_pareto_structural_breakdown.pdf",
    }
    missing = expected - fig_names
    assert not missing, (
        f"MANIFEST.md is supposed to reference every paper figure; "
        f"missing: {sorted(missing)}"
    )


def test_every_paper_artefact_exists(manifest_paths: list[str]) -> None:
    """Every figure/table/csv/bib mentioned in MANIFEST.md is on disk."""
    paper_refs = [
        p for p in manifest_paths
        if (
            p.startswith("fig")
            or p.startswith("tab")
            or p.endswith(".bib")
            or p.endswith(".tex") and "framework" in p.lower()
        )
    ]
    assert paper_refs, "No paper artefacts found in MANIFEST.md — parser broken?"

    missing = []
    for rel in sorted(set(paper_refs)):
        if _resolved_path(rel) is None:
            missing.append(rel)
    assert not missing, (
        "MANIFEST.md references files that don't exist:\n  "
        + "\n  ".join(missing)
    )


def test_main_tex_compiles_resolvable_paths() -> None:
    """Every \\includegraphics in the main TeX must resolve under figures/."""
    tex = PAPER_DIR / "TRPRO_EWGT2026.tex"
    assert tex.exists(), f"main TeX file missing: {tex}"
    text = tex.read_text(encoding="utf-8")
    fig_refs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    # Sanity: paper has at least 5 PDF figures
    assert len(fig_refs) >= 5, (
        f"Expected >= 5 includegraphics calls in main TeX, found {len(fig_refs)}"
    )
    missing = []
    for ref in fig_refs:
        candidate = PAPER_DIR / "figures" / ref
        if not candidate.exists():
            missing.append(ref)
    assert not missing, (
        "Main TeX references figures that don't exist under figures/:\n  "
        + "\n  ".join(missing)
    )
