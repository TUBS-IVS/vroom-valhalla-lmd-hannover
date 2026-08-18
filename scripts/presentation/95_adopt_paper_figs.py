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
        src=D.REV / "figures" / "fig6_structural_grid_6_smoothed",
        script="scripts/revision/31_fig6_structural_smoothed.py",
        title="Structural breakdown and per-provider knee points",
        act="8 - Adopted paper figures",
        basis="Stage 3 (per-hub balancing + system smoothing)",
        claim="Six-panel structural breakdown of the Stage-3 result, including "
              "the per-provider knee points P* on the cost/wait trade-off.",
        caveats="Adopted verbatim from the revision render; paper-styled, no "
                "slide variant. Its own script asserts the knee recomputation.",
    ),
    dict(
        name="fig82_cluster_evidence",
        src=D.REV / "figures" / "fig_S1_cluster_evidence",
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


def main():
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
