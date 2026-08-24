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

# ─────────────────────────────────────────────────────────────────────────────
# Colour system — the paper is the reference
# ─────────────────────────────────────────────────────────────────────────────
# The deck and the manuscript must be indistinguishable, so every colour here is
# taken from the figure scripts that produce the paper:
#
#   scripts/revision/30_fig5_heatmap_smoothed.py   fig 5 heatmaps
#   scripts/revision/31_fig6_structural_smoothed.py fig 6 pareto + structure
#   scripts/revision/32_fig4_mix.py                 fig 4 frequency mix
#   scripts/paper/paper_final_v2.py                 PROV_COLOR, used by fig 1/3
#
# Do not "improve" these values in isolation. If a colour changes, it changes in
# the paper first and is copied here, otherwise the two drift apart again.

BRAND = "#BE1E3C"          # TU red: deck chrome only, never inside a figure
BRAND_DARK = "#8f142b"
INK = "#15181d"
INK_SOFT = "#5c616b"
GRID = "#d9d9de"
MISSING = "#e6e6e6"        # areas outside the model, as in the paper's maps

# Colormaps, per quantity, exactly as the paper uses them.
CMAP_SAVING = "viridis"    # fig 5 (a), (b): cost saving
CMAP_WAIT = "YlOrRd"       # fig 5 (c): additional customer wait
CMAP_FLEET = "magma"       # fig 5 (d), (e): peak-fleet and CV reduction
CMAP_CHANGE = "RdBu_r"     # fig 5 (f): signed total fleet change
CMAP_THETA = "viridis"     # fig 6: adoption levels
CMAP_PENALTY = "plasma"    # fig 6: service-penalty levels
CMAP_QUARTILE = "YlOrBr"   # fig 6: structural quartiles
CMAP_DEMAND = CMAP_SAVING  # maps have no paper counterpart, so they
                           # follow the paper's magnitude convention

# Settlement type — paper fig 6 RAUMTYP_PAL. (Note: the paper's fig 1 uses a
# different assignment for the same three classes; fig 6 is the results figure
# the deck's structural slides correspond to, so fig 6 wins here.)
RAUMTYP = {"urban": "#1d3557", "suburban": "#2a9d8f", "rural": "#e76f51"}

# Weekly delivery frequency — paper fig 4 FREQ_COLOR.
FREQ = {2: "#1d3557", 3: "#2a9d8f", 4: "#e9c46a", 5: "#f4a261", 6: "#e76f51"}

# Providers — paper PROV_COLOR (scripts/paper/paper_final_v2.py), the palette
# behind fig 1 and the LSP colouring of fig 3.
PROVIDER = {"Amazon": "#003049", "DHL": "#d62828", "DPD": "#f77f00",
            "FedEx": "#5a189a", "GLS": "#2a9d8f", "Hermes": "#9d4edd",
            "UPS": "#7d5a50"}

# Pipeline stages have no counterpart in the paper; kept in its colour family.
STAGE = {"baseline": "#8d99ae", "stage2": "#457b9d", "stage3": "#1d3557"}


def seq_cmap(reverse: bool = False):
    """Magnitude colormap — the paper's saving ramp."""
    return CMAP_SAVING + ("_r" if reverse else "")


def seq_warm_cmap(reverse: bool = False):
    """Waiting-time colormap — the paper's wait ramp."""
    return CMAP_WAIT + ("_r" if reverse else "")


def fleet_cmap(reverse: bool = False):
    """Fleet-reduction colormap — the paper's fleet ramp."""
    return CMAP_FLEET + ("_r" if reverse else "")


def div_cmap():
    """Diverging colormap for signed quantities — the paper's change ramp."""
    return CMAP_CHANGE


def freq_colors(sizes) -> list[str]:
    """Ordered colours for the given delivery-frequency classes."""
    return [FREQ[int(s)] for s in sizes]


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
