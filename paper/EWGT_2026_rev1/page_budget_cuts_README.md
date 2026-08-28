# page_budget_cuts.patch

Two `git format-patch` commits implementing blocks 1-18 of `task-14b-review.md`
section 6.3, in order. Not applied by default; regenerated on top of the final
fix round (branch head `820819b`), where it was verified to apply with
`git apply --check` and no offsets.

* **Blocks 1-13** (first commit): ten passages move into `supplementary.tex`,
  two duplications are deleted. **17 -> 16 pages**; supplementary 7 -> 9.
* **Blocks 14-18** (second commit): duplication and density in the abstract,
  the introduction, section 2, limitation (vi) and the conclusion. 2,011
  further characters; page count stays at **16**.

Apply both: `git am paper/EWGT_2026_rev1/page_budget_cuts.patch`
(blocks 1-13 only: `git am` both, then `git reset --hard HEAD~1`).
Without commits: `git apply paper/EWGT_2026_rev1/page_budget_cuts.patch`.

Five defects the reviews flagged are repaired in this version: the G1a tolerance
sentence stays in section 2.4; block 8 is dropped (its premise was false); the
cross-provider independence statement is kept verbatim in Supplementary S4,
which cites "Section 2.4" as literal text because a `\ref` across the two
separately built documents would print "??"; and limitation (vi) keeps its
labour clause and the 0.139/0.135 numbers.

Both patched builds compile clean with **23 bibitems**, no undefined references
and no comment line hiding a control sequence. Going below 16 pages means
cutting substance -- the abstract, section 3.1 or section 3.3 -- which is the
author's call.

**Interaction with the final fix round.** Block 1 shortens the label-cost
paragraph of section 2.2, which the I3 fix had rewritten. The patched text
keeps the I3 wording, so the surviving sentence still names its population
("Over the surrogate's training pool ... 72.2, 6.0, and 21.7 %; over the
re-routed validation labels ... 70.7, 6.1, and 23.2 %"), and keeps the
paragraph's `% src:` block, which is a comment and costs no printed line. The
I1 range sentence (section 3.4) and the M1 `P * theta = 0` paragraph (section
2.4) are in passages the patch does not touch and survive it verbatim.
Verified after patching: main 16 pp, supplementary 9 pp, 23 bibitems,
0 unresolved citations, `guard_tex.py` green on both documents.
