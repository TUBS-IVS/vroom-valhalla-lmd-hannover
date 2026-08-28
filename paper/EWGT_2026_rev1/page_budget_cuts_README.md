# page_budget_cuts.patch

Two `git format-patch` commits implementing blocks 1-18 of `task-14b-review.md`
section 6.3, in order. Not applied by default; regenerated on top of fix round 2.

* **Blocks 1-13** (first commit): ten passages move into `supplementary.tex`,
  two duplications are deleted. **17 -> 15 pages**; supplementary 7 -> 9.
* **Blocks 14-18** (second commit): duplication and density in the abstract,
  the introduction, section 2, limitation (vi) and the conclusion. 2,011
  further characters; page count stays at **15**.

Apply both: `git am paper/EWGT_2026_rev1/page_budget_cuts.patch`
(blocks 1-13 only: `git am` both, then `git reset --hard HEAD~1`).
Without commits: `git apply paper/EWGT_2026_rev1/page_budget_cuts.patch`.

Four casualties the review flagged are repaired in this version: the G1a
tolerance sentence stays in section 2.4, block 8 is dropped (its premise was
false), the cross-provider independence statement is kept verbatim in
Supplementary S4, and limitation (vi) keeps its labour clause and the
0.139/0.135 numbers.

Both patched builds compile clean with **23 bibitems** and no undefined
references. Going below 15 pages means cutting substance -- the abstract,
section 3.1 or section 3.3 -- which is the author's call.
