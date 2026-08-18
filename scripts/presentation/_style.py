"""Two render styles for the presentation figure set.

Every figure script calls `for style in styles(): apply(style) ...` and saves
through `save(fig, name, style)`, so a single script produces both the
publication rendering and the slide rendering from one code path. That keeps the
two variants from drifting apart, which is the usual failure mode when slide
figures are hand-tweaked copies.

  paper  -- serif, 11 pt, column-width, PDF + PNG at 300 dpi. Matches the
            rcParams already used by scripts/revision/3*.py so figures dropped
            into the paper are visually consistent with fig4/5/6.
  slides -- sans-serif, 16-20 pt, 16:9 canvas, PNG at 200 dpi, thicker lines
            and higher-contrast text for projection.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import rcParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "results" / "presentation_2026_08"

STYLES = ("paper", "slides")

# Tier A is the core narrative of the talk; tier B is backup / deep-dive
# material. Assigned per figure in _manifest.py and used to pick the slide
# subdirectory, so the deck's spine is obvious on disk.
TIER_A = "tierA"
TIER_B = "tierB"

_PAPER = {
    "font.family": "serif",
    "font.size": 11,
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "patch.linewidth": 0.6,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}

_SLIDES = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 17,
    "mathtext.fontset": "dejavusans",
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "axes.linewidth": 1.4,
    "lines.linewidth": 2.6,
    "patch.linewidth": 1.0,
    "grid.linewidth": 1.0,
    "axes.labelcolor": "#111111",
    "text.color": "#111111",
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


def styles() -> tuple[str, ...]:
    return STYLES


def apply(style: str) -> None:
    """Reset to defaults, then apply the requested style."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}, expected one of {STYLES}")
    rcParams.update(matplotlib.rcParamsDefault)
    rcParams.update(_PAPER if style == "paper" else _SLIDES)


def figsize(style: str, paper: tuple[float, float],
            slides: tuple[float, float] | None = None) -> tuple[float, float]:
    """Per-style figure size. Slide default is 16:9 at the paper height."""
    if style == "paper":
        return paper
    if slides is not None:
        return slides
    h = paper[1]
    return (h * 16.0 / 9.0, h)


def scale(style: str, paper_value: float, factor: float = 1.6) -> float:
    """Scale a hand-tuned size (marker size, annotation pt) for slides."""
    return paper_value if style == "paper" else paper_value * factor


def outdir(style: str, tier: str) -> Path:
    """Target directory for a rendered figure.

    paper  -> presentation_2026_08/paper/
    slides -> presentation_2026_08/slides/tierA|tierB/
    """
    d = OUT_ROOT / "paper" if style == "paper" else OUT_ROOT / "slides" / tier
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(fig, name: str, style: str, tier: str = TIER_A) -> list[Path]:
    """Save one figure in the formats appropriate to its style."""
    d = outdir(style, tier)
    exts = ("pdf", "png") if style == "paper" else ("png",)
    written = []
    for ext in exts:
        p = d / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        written.append(p)
    plt.close(fig)
    for p in written:
        print(f"  saved {p.relative_to(ROOT)}")
    return written
