# page_budget_cuts.patch

Two `git format-patch` commits implementing blocks 1-18 of `task-14b-review.md`
section 6.3, in order. Not applied by default; regenerated on top of task 14D.

* **Blocks 1-13** (first commit): ten passages move into `supplementary.tex`,
  two duplications are deleted. **17 -> 16 pages**; supplementary 8 -> 10.
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
labour clause and the 0.139/0.135 numbers. The task-14D pointer to Supplementary
Fig. S12 is carried into block 10's rewritten fleet sentence, so it survives the
cut.

Both patched builds compile clean at **16 + 10 pages** with **23 bibitems**, no
undefined references, and no comment line hiding a control sequence or prose.
Going below 16 pages means cutting substance -- the abstract, section 3.1 or
section 3.3 -- which is the author's call.
