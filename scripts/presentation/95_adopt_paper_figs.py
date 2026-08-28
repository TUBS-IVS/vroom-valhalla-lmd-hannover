"""Adopt the Stage-3 renders that already exist, rather than re-plotting them.

Two revision figures are dense multi-panel paper figures whose plotting code
carries its own verification asserts (fig5's baseline-CV check, fig6's knee
recomputation). Re-implementing them for this set would risk silent divergence
from the figures the paper itself carries, for no gain: they are already on the
Stage-3 basis. They are therefore copied verbatim into paper/ and recorded in
the manifest with their producing script.

Neither gets a slide variant. Both are too dense to project; the slide deck
covers the same ground through fig31/32/33 (fig5's panels, split) and
fig34/fig36 (fig6's Pareto and knee content).

The 2026-08 revision adds a second group, `ADOPT_REV`: the regenerated
Fig. 4/4b/5/5b/6 that Task 13 renders from the revision grid into
`<rev>/figures/`. These DO get a copy under `slides/tierB/`, because the decks
show them as backup — "here is the whole grid" — rather than as a tier-A
statement. They are six- and eight-panel paper figures at 8 pt: readable when
somebody walks up to the screen after the talk, not from the back of the room.
A tier-A slide-styled render of the revision grid does not exist yet; the
revision story slides carry that content as native shapes instead.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _data as D
import _style as S

ADOPT = [
    dict(
        name="fig81_structural_grid",
        src=D.REV_LEGACY / "figures" / "fig6_structural_grid_6_smoothed",
        script="scripts/revision/31_fig6_structural_smoothed.py",
        title="Structural breakdown and per-provider knee points",
        act="8 - Adopted paper figures",
        basis="submission grid (per-hub balancing + system smoothing)",
        claim="Six-panel structural breakdown of the Stage-3 result, including "
              "the per-provider knee points P* on the cost/wait trade-off.",
        caveats="Adopted verbatim from the revision render; paper-styled, no "
                "slide variant. Its own script asserts the knee recomputation.",
    ),
    dict(
        name="fig82_cluster_evidence",
        src=D.REV_LEGACY / "figures" / "fig_S1_cluster_evidence",
        script="scripts/revision/12_fig3_cluster_evidence.py",
        title="Why the training pool clusters linearly",
        act="8 - Adopted paper figures",
        basis="sweep_v3_mergefix training pool",
        claim="The apparent linear bands in the cost/demand scatter are tour-count "
              "bands (ceil(p/Q)) plus same-area augmentation families, not a "
              "modelling artefact.",
        caveats="Adopted verbatim; answers a reviewer question about Figure 3. "
                "Paper-styled, no slide variant.",
    ),
]


# The revision's own figures, adopted from whichever grid D.REV points at.
# `slide_tier` sends a copy to slides/<tier>/ as well, under the same name, so
# the deck builders can pick it up by name and the manifest self-check finds
# both files.
ADOPT_REV = [
    dict(
        name="fig90_grid_two_lenses",
        src=["supp_fig5_grid_heatmap_v2", "fig5_grid_heatmap_v2",
             "fig5_grid_heatmap_6_smoothed"],
        slide_tier=S.TIER_B,
        script="scripts/revision/70_figs_tables_v2.py",
        title="The (P, theta) grid in both lenses",
        act="8 - Adopted paper figures",
        basis="revision grid, both plans",
        claim="Six panels over the full (P, theta) grid: routing-cost saving "
              "of the routing-optimal plan, operator-cost saving of the "
              "operator-polished plan, mean added wait, sum of hub peaks "
              "against the baseline, Mon-Sat CV of the system fleet, and "
              "weekly vehicle-days.",
        caveats="Panels (a) and (b) are different lenses AND different plans "
                "and must not be read as one series -- the figure's own "
                "caption says so. Adopted verbatim; backup tier.",
    ),
    dict(
        name="fig91_offdiagonal",
        src=["supp_fig5b_offdiagonal_v2", "fig5b_offdiagonal_v2"],
        slide_tier=S.TIER_B,
        script="scripts/revision/70_figs_tables_v2.py",
        title="Each plan priced in the other lens",
        act="8 - Adopted paper figures",
        basis="revision grid, both plans",
        claim="The two off-diagonal combinations: the routing-optimal plan "
              "seen through the operator lens and the operator-polished plan "
              "seen through the routing lens.",
        caveats="Adopted verbatim; backup tier.",
    ),
    dict(
        name="fig92_freq_mix_two_plans",
        src=["supp_fig4_freq_mix_two_plans", "fig4_freq_mix_two_plans"],
        slide_tier=S.TIER_B,
        script="scripts/revision/70_figs_tables_v2.py",
        title="Delivery-frequency mix, both plans",
        act="8 - Adopted paper figures",
        basis="revision grid, both plans",
        claim="How the mix of weekly delivery frequencies moves over the "
              "(P, theta) grid, for the routing-optimal and the "
              "operator-polished plan.",
        caveats="Adopted verbatim; backup tier.",
    ),
    dict(
        name="fig93_mean_days",
        src=["supp_fig4b_mean_days", "fig4b_mean_days"],
        slide_tier=S.TIER_B,
        script="scripts/revision/70_figs_tables_v2.py",
        title="Mean delivery days per area",
        act="8 - Adopted paper figures",
        basis="revision grid, both plans",
        claim="Mean delivery days per area across the grid, for both plans.",
        caveats="Adopted verbatim; backup tier.",
    ),
    dict(
        name="fig94_structural_two_lenses",
        src=["supp_fig6_structural_v2", "fig6_structural_v2",
             "fig6_structural_grid_6_smoothed"],
        slide_tier=S.TIER_B,
        script="scripts/revision/70_figs_tables_v2.py",
        title="Fronts, knees and structure, in both lenses",
        act="8 - Adopted paper figures",
        basis="revision grid, both plans",
        claim="System Pareto front in each lens, the per-LSP knee P* in both "
              "lenses, and delivery frequency by hub distance, parcels per "
              "drop-site visit and hub size. Panel (f) carries the one-area "
              "depot finding: single-cell hubs stay at six delivery days for "
              "every adoption level.",
        caveats="Panel (f) is a within-DHL statement -- the 1 / 2-4 / 5-12 "
                "hub-size buckets hold DHL cells only, because DHL is the "
                "only multi-depot network in the case study. Adopted "
                "verbatim; backup tier.",
    ),
    dict(
        name="fig95_operator_lens",
        src=["supp_fig6b_operator_lens_v2"],
        slide_tier=S.TIER_B,
        script="scripts/revision/70_figs_tables_v2.py",
        title="The operator lens on its own",
        act="8 - Adopted paper figures",
        basis="revision grid, operator-polished plan",
        claim="The operator-cost view of the grid as a standalone figure, so "
              "the operator lens can be shown without the routing lens beside "
              "it inviting a like-for-like reading.",
        caveats="Adopted verbatim; backup tier. Only rendered from v6 onwards.",
    ),
]


def _adopt_rev():
    """Adopt Task 13's revision figures from the grid currently selected."""
    src_dir = D.REV / "figures"
    if not src_dir.exists():
        print(f"  no revision figures at {src_dir} -- skipped")
        return
    paper = S.outdir("paper", S.TIER_A)
    for spec in ADOPT_REV:
        wrote = []
        stems = ([spec["src"]] if isinstance(spec["src"], str)
                 else list(spec["src"]))
        stem = next((st for st in stems
                     if (src_dir / f"{st}.png").exists()), None)
        if stem is None:
            print(f"  {spec['name']}: none of {stems} in {src_dir.name} "
                  f"-- skipped")
            continue
        for ext in ("pdf", "png"):
            src = src_dir / f"{stem}.{ext}"
            if not src.exists():
                print(f"  MISSING {src} -- skipped")
                continue
            shutil.copy2(src, paper / f"{spec['name']}.{ext}")
            if ext == "png" and spec.get("slide_tier"):
                shutil.copy2(src, S.outdir("slides", spec["slide_tier"])
                             / f"{spec['name']}.png")
            D.prov.record(src)
            wrote.append(src)
            print(f"  adopted {src.relative_to(D.ROOT)} -> {spec['name']}.{ext}")
        if not wrote:
            print(f"  {spec['name']}: nothing adopted, no manifest entry")
            continue
        D.prov.sources.setdefault(spec["script"], {"bytes": 0, "mtime": "n/a"})
        D.prov.write(spec["name"], title=spec["title"], tier=S.TIER_B,
                     act=spec["act"],
                     basis=f"{spec['basis']} ({D.REV.name}, {stem})",
                     claim=spec["claim"], caveats=spec["caveats"])
        D.prov.sources.clear()


def main():
    _adopt_rev()
    out = S.outdir("paper", S.TIER_A)
    for spec in ADOPT:
        wrote = []
        for ext in ("pdf", "png"):
            src = spec["src"].with_suffix(f".{ext}")
            if not src.exists():
                print(f"  MISSING {src} -- skipped")
                continue
            dst = out / f"{spec['name']}.{ext}"
            shutil.copy2(src, dst)
            D.prov.record(src)
            wrote.append(dst)
            print(f"  adopted {src.relative_to(D.ROOT)} -> "
                  f"{dst.relative_to(D.ROOT)}")
        if not wrote:
            print(f"  {spec['name']}: nothing adopted, no manifest entry written")
            continue
        D.prov.sources.setdefault(spec["script"], {"bytes": 0, "mtime": "n/a"})
        D.prov.write(spec["name"], title=spec["title"], tier=S.TIER_A,
                     act=spec["act"], basis=spec["basis"], claim=spec["claim"],
                     caveats=spec["caveats"])
        D.prov.sources.clear()


if __name__ == "__main__":
    main()
