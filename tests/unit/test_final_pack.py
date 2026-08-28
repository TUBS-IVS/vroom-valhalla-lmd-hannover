"""``scripts/revision/79_build_final_pack.py`` -- the provenance pack builder.

79_ copies the v6 grid into ``results/revision_2026_08_final/`` and writes a
README that names, for every file, the script that produced it. The parts
worth testing are the ones that decide what the pack *claims*:

* ``producer_of`` must refuse an unattributable file rather than shipping an
  "unknown" row -- a provenance pack with a file nobody can trace is worse
  than one that is missing it;
* ``compendium_excerpt`` must cut the exact section range or fail, never
  return a silently truncated half;
* ``GALLERY`` must cover exactly the supplementary stems 71_ knows about.
  That cross-file invariant is the same one that failed in task 13C/13D: a
  figure was rendered, nobody registered it anywhere, and it silently never
  reached the paper or the gallery.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "revision"))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / "revision" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load("_final_pack", "79_build_final_pack.py")


@pytest.fixture(scope="module")
def sync_mod():
    return _load("_sync_paper_figs", "71_sync_paper_figs.py")


# ─────────────────────────────────────────────────────────────────────────
# attribution
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("rel,expect", [
    ("figures/supp_fig_mechanism_v2.pdf", "77_mechanism_v2.py"),
    ("figures/supp_fig_fleet_week_v2_P0.png", "78_fleet_week_v2.py"),
    ("figures/supp_map_saving_P_v2.pdf", "76_maps_v2.py"),
    ("figures/supp_fig7_fleet_week_classes.pdf", "75_fig_fleet_week_classes.py"),
    ("figures/supp_fig5_grid_heatmap_v2.pdf", "70_figs_tables_v2.py"),
    ("figures/fig5_grid_heatmap_6_smoothed.pdf", "30_fig5_heatmap_smoothed.py"),
    ("tables/tab_per_cell_costs_v2.csv", "72_per_cell_costs_v2.py"),
    ("tables/tab_value_of_stage2_v2.csv", "73_tables_ops_v2.py"),
    ("tables/tab_co2_km_v2.tex", "70_figs_tables_v2.py"),
    ("validation/gates_report.md", "62_gates_check.py"),
    ("validation/tab_vroom_v2.csv", "67_validate_vroom_v2.py"),
])
def test_every_known_artifact_names_its_producing_script(mod, rel, expect):
    script, _ = mod.producer_of(rel)
    assert expect in script


def test_the_delta_table_is_labelled_as_a_cross_grid_comparison(mod):
    _, grid = mod.producer_of("tables/tab_grid_delta_v2.csv")
    assert grid == "v6 vs v5"


@pytest.mark.parametrize("rel", [
    "scratch/notes.txt",
    # deliberately NOT a catch-all: a new figure from a new script must fail
    # attribution rather than be shipped as if 70_ had drawn it
    "figures/supp_fig9_from_a_future_script.pdf",
    "figures/fig6_bucket_composition.csv",
])
def test_an_unattributable_file_is_refused(mod, rel):
    with pytest.raises(AssertionError, match="matches no PRODUCERS rule"):
        mod.producer_of(rel)


def test_the_superseded_bucket_csv_is_named_and_not_shipped(mod):
    """`fig6_bucket_composition.csv` sits in the live grid's figures/ but is
    the pre-rename filename of `supp_fig6_bucket_composition.csv`: same
    content, absent from the render manifest, produced by nothing. It must be
    skipped by name, not silently carried along."""
    assert "fig6_bucket_composition.csv" in mod.FIGURE_SUPERSEDED
    assert "supp_fig6_bucket_composition.csv" in mod.FIGURE_EXTRA
    script, _ = mod.producer_of("figures/supp_fig6_bucket_composition.csv")
    assert "70_figs_tables_v2.py" in script


def test_the_accepted_figures_are_attributed_to_the_frozen_builders(mod):
    """30_/31_/32_ render the three accepted paper figures; 70_ only drives
    them. Attributing those to 70_ would hide which code owns the layout the
    paper was accepted with."""
    for rel, builder in (("figures/fig4_SM_mix_pct_8P.pdf", "32_"),
                         ("figures/fig5_grid_heatmap_6_smoothed.pdf", "30_"),
                         ("figures/fig6_structural_grid_6_smoothed.pdf", "31_")):
        script, _ = mod.producer_of(rel)
        assert builder in script and "via 70_" in script


# ─────────────────────────────────────────────────────────────────────────
# compendium excerpt
# ─────────────────────────────────────────────────────────────────────────
_DOC = ("## 40.13 before\nbefore text\n\n"
        "## 40.14 start\nfirst\n\n"
        "## 40.20 middle\nmid\n\n"
        "## 40.28 end\nlast line\n\n"
        "## 41 after\nafter text\n")


def test_excerpt_starts_and_ends_on_the_named_sections(mod):
    out = mod.compendium_excerpt(_DOC, "## 40.14", "## 40.28")
    assert out.startswith("## 40.14 start")
    assert "mid" in out and out.rstrip().endswith("last line")
    assert "before text" not in out and "after text" not in out


def test_excerpt_runs_to_the_end_when_nothing_follows(mod):
    out = mod.compendium_excerpt(_DOC.split("## 41 after")[0],
                                 "## 40.14", "## 40.28")
    assert out.rstrip().endswith("last line")


def test_excerpt_refuses_a_missing_heading(mod):
    with pytest.raises(AssertionError, match="not found"):
        mod.compendium_excerpt(_DOC, "## 40.14", "## 40.99")


def test_excerpt_refuses_a_reversed_range(mod):
    with pytest.raises(AssertionError, match="comes after"):
        mod.compendium_excerpt(_DOC, "## 40.28", "## 40.14")


def test_the_real_compendium_range_is_extractable(mod):
    out = mod.compendium_excerpt(
        mod.COMPENDIUM.read_text(encoding="utf-8"))
    assert out.startswith(mod.EXCERPT_FROM)
    assert mod.EXCERPT_TO_END_OF in out
    # the sections in between must all be there, not just the two endpoints
    for sec in ("## 40.21", "## 40.22b", "## 40.23b", "## 40.25", "## 40.27"):
        assert sec in out


# ─────────────────────────────────────────────────────────────────────────
# gallery manifest
# ─────────────────────────────────────────────────────────────────────────
def test_the_gallery_covers_exactly_the_supplementary_stems_71_knows(
        mod, sync_mod):
    """71_'s COMPANION_MAP is the list of supplementary stems that reach the
    paper. The gallery must cover the same set: a stem in one and not the
    other is a figure that exists in the paper but not in the gallery, or
    the reverse -- exactly the 13C/13D gap."""
    assert set(mod.GALLERY) == set(sync_mod.COMPANION_MAP)


def test_every_gallery_entry_is_german_and_names_its_script(mod):
    for stem, (title, caption, script) in mod.GALLERY.items():
        assert title and not title.endswith("."), stem
        assert caption.endswith("."), stem
        assert script.startswith("scripts/revision/"), stem
        assert (ROOT / script).exists(), f"{stem}: {script} does not exist"
