# page_budget_cuts.patch

Two `git format-patch` commits on top of `df723d9` (revision part C), implementing
blocks 1-18 of `task-14b-review.md` section 6.3 in order. Not applied by default.

* **Blocks 1-13** (first commit): ten passages move into `supplementary.tex`,
  three duplications are deleted. **16 -> 15 pages**; supplementary 7 -> 9.
* **Blocks 14-18** (second commit): duplication and density in the abstract, the
  introduction, section 2, limitation (vi) and the conclusion. 2,135 further
  characters; page count stays at **15**.

Apply both: `git am paper/EWGT_2026_rev1/page_budget_cuts.patch`
(blocks 1-13 only: `git am` both, then `git reset --hard HEAD~1`).
Without commits: `git apply paper/EWGT_2026_rev1/page_budget_cuts.patch`.

Measured, not estimated: the review budgeted ~13,400 characters for 13 pages;
these blocks yield 8,694 without losing a claim, and figure scaling still buys
nothing at 15 pages. Going below 15 means cutting substance -- the abstract,
section 3.1 or section 3.3 -- which is the author's call.
