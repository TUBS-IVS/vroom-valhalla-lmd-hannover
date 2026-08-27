# Response to Reviewers — skeleton (revision part A, 2026-08-27)

**Status: skeleton.** Prose is drafted; every number marked `[P]` is provisional
(v5 grid) and must be replaced from the final head-priced grid in part B, in
lockstep with the `\provisional{}` markers in `tbc_preprint_main.tex`.

## On the source of the reviewer comments

**No verbatim reviewer file exists in this repository.** A search over `paper/`
and `docs/` for files matching *review*, *reviewer*, or *comment* returned
nothing but the change logs themselves. What we do have is the reviewer-point
table at the top of `paper/EWGT_2026_rev1/CHANGES_rev1.md`: eleven numbered
points, recorded in German, together with what was changed for each. That table
is used below as the authoritative list of reviewer comments; the original
letter must be pasted in before this file becomes a real response letter, and
the wording of each point below should then be replaced by the reviewer's own.

The response therefore has two parts:

- **Part 1** maps the eleven recorded reviewer points to their resolution
  (mostly settled in the 2026-08-18 round; three points are touched again by
  this revision).
- **Part 2** groups the changes this revision makes on our own initiative, by
  theme. These are not reviewer-requested. They are self-identified corrections
  and extensions, and two of them retract statements from the submitted version.
  Announcing them plainly is the point.

---

## Part 1 — Recorded reviewer points

| # | Point (as recorded) | Resolution | Touched again in this revision? |
|---|---|---|---|
| 1 | "Classical problem" / novelty unclear | Introduction now states explicit contributions | Yes — three contributions become four (tour rule, two-lens accounting) |
| 2 | Off-the-shelf tools only | "our own HAGRID demand model" | No |
| 3 | Acronyms undefined | MAPE and pp defined in the abstract; "Monday–Saturday"; "Daganzo-LGB-Hybrid" at first use | **Open:** HAGRID full form still missing |
| 4 | Practical relevance | Managerial implications woven into the conclusion | Yes — the roster/schedule message is new and is the strongest practical result |
| 5 | Simulation vs. optimization | Framed as surrogate-assisted schedule optimization | No |
| 6 | Per-LSP vs. system-wide | Optimization runs within each provider's depot network; no sharing across providers | No |
| 7 | Not reproducible | Repository footnote extended (parameters, hyperparameters, seeds, 39 patterns) | No |
| 8 | A single global alpha | Data given: per-LSP 3.2 % vs. global 2.9 % OOF MAPE | No |
| 9 | 39 patterns: total or per cell? | "39 distinct admissible patterns; the same candidate set for every cell" | No |
| 10 | Eq. (3) looks like a Nash equilibrium | Coordinate descent on a common depot objective; restart spread < 1e-12 relative; no global-optimality claim | Yes — the equation's coupling term is removed and replaced by an explicit statement of the residual coupling inside pooling groups |
| 11 | Clusters in Fig. 3 | Explained as tour bands from `ceil(p/Q)` plus same-postcode augmentation families; folds are group-safe | No |

**Draft text for point 3 (still open).** *We thank the reviewer. All remaining
acronyms are now defined at first use. [HAGRID full form to be supplied by the
authors.]*

**Draft text for point 10.** *We thank the reviewer for pressing on this. In
revising the model we found a stronger answer than a wording change: the shared
depot tour that created the coupling has been removed entirely. Every tour is
now formed by one universal rule that applies to the daily baseline and to every
consolidation scenario alike, so the depot objective is separable wherever a
cell is large enough to be served on its own. A residual coupling remains only
inside pooling groups of sub-threshold cells, and Eq. (3) retains the depot
objective precisely in order to keep that residual coupling priced. We do not
claim separability.*

---

## Part 2 — Changes made on our own initiative

Grouped by theme. Every item points at its entry in `docs/CHANGES_rev1.md`,
which carries the old sentence, the new sentence, and the evidence.

### Theme A — A correction we must announce: the partial-adoption results

*Draft text.* *While preparing this revision we found that the submitted model
gave the consolidation scenarios a structural advantage the daily baseline did
not have: only the scenarios were allowed to pool the standard-delivery parcels
of areas without an own tour onto a single shared depot tour. At partial
adoption roughly a quarter of all parcels rode a tour the baseline could not
form. This inflated the savings reported for willingness-to-wait shares below
one, and the mechanism we offered for the persistent saving at high penalty and
low adoption ("hub bundling covers the penalty") was an explanation of an
artefact. The submitted claim that 42 % of areas remain non-daily at
(P, theta) = (10, 0.1) with a 3.6 % system saving does not survive: under the
corrected model the figures are 2.9 %[P] of areas and 0.03 %[P] of routing cost.
The full-adoption headline is not affected — at theta = 1 no shared tour exists
at all, which we verified directly — and a symmetric re-computation of the
submitted grid bounds the whole effect at that point to 0.57 percentage points
(22.79 % to 22.22 %).*
Entries: B2, B3, B6, A3.

### Theme B — A validity domain for the surrogate

*Draft text.* *We now constrain every instance the surrogate is asked to price
to the family it was trained on: a minimum of 230 parcels per tour, a maximum of
556 stops and 159 km2 of service area, and a convex hull no larger than 1.22
times the summed member area. Each bound is derived from a measurement, not
chosen: the out-of-fold error triples below 230 parcels, the stop and area
bounds are the 99th and 95th percentiles of the training pool, and 1.22 is the
largest hull ratio the pool contains. Where the rule pools several cells into
one tour, the pooled instance is priced by a separate head that is installed
only in the bins where it is certified against out-of-fold error and bias, and
that refuses to price outside them. We regard this as a strengthening of the
paper rather than a concession: the validity domain is now measured and stated.*
Entries: A2, A5, B10, B11(v).

### Theme C — The cost model as it really is

*Draft text.* *We correct the description of the cost model. The routing solver
applied its default per-hour rate to every label, so the effective model is
189.15 EUR per vehicle-day, 0.3864 EUR per kilometre, and 36 EUR per route-hour.
The submitted text claimed that driver labour is inside the fixed cost so that
no time term is needed; the first half is true, the second is not. No number
changes: baseline, scenarios, prediction and out-of-sample validation all carry
the same term, and because route time is dominated by per-parcel service time it
is near-constant across schedules and damps the reported relative savings rather
than inflating them.*
Entry: A1.

### Theme D — Two cost lenses, and why the recommendation changes

*Draft text.* *A schedule that minimizes per-day routing cost is not
automatically good for the operator who pays for the week. We therefore report
every result under two lenses: a routing lens that charges each vehicle-day in
full, identical to the submitted accounting, and an operator lens that charges
only variable cost below each depot's weekly fleet peak, since drivers are
employed and vans are owned for the week. The result is the central finding of
this revision. Under the routing lens the cost-optimal plan saves 23.1 %[P];
under the operator lens the very same plan is 7.8 %[P] worse than daily
delivery, because its two-day patterns raise the summed depot peak by
34.5 %[P]. Re-optimizing the second stage in the operator's currency turns this
into 24.7 %[P] of operator saving at a 16.9 %[P] lower peak fleet. Both lenses
point to P = 0.25 EUR per parcel-day as the operating point, and the carrier
taxonomy the reviewer asked us to justify turns out to be lens-dependent: three
of the seven carriers move up one class under the operator lens.*
Entries: A6, A7, B1, B4, B5, B7, B8.

### Theme E — The second stage, honestly described

*Draft text.* *The submitted post-processing was described as a frequency-
preserving greedy search under a 5 % cost budget. We rebuilt it, because the
budget never bound and the frequency lock blocked exactly the moves that carry
the operator-side value: at depots serving a single cell, a two-day pattern has
no rotation available and the peak falls only with more delivery days. The new
second stage accepts a move only when variable cost, peak cost and waiting
penalty jointly fall, may change a cell's delivery frequency when anyone is
willing to wait, and is pinned to daily delivery when nobody is. We state
plainly that it is a local search and report its remaining gap to a
flat-profile lower bound. We also state plainly that the resulting plan is not
service-neutral relative to the routing-optimal plan: its waiting time moves in
both directions, so we report both plans side by side throughout.*
Entries: A7, A8, B5, B11(iii).

### Theme F — Paying for the delay

*Draft text.* *Because a reader may reasonably ask what the service penalty
means in cash, we added an analysis in which it is actually paid out. Under a
flat 0.50 EUR discount per delayed parcel the cost-optimal extreme is no longer
best — it delays too many parcels — and the optimum moves to the same
P = 0.25–0.5 range the lenses already indicate. The break-even discount is
0.77–2.24 EUR[P] per delayed parcel in the operator lens.*
Entry: B9.

### Theme G — What is not yet validated

*Draft text.* *We are explicit about the limits. The solver-validated operating
points available so far all lie at full adoption, where no pooled tour exists,
so the partial-adoption regime is not yet validated against the solver; that
validation is reported in Section 3.4. The operator-lens magnitude is shaped by
one carrier's depot structure, pooling is within-depot only, and the pooled-tour
head is certified on 44 %[P] of the instances that occur, with a conservative
fallback elsewhere. A dedicated limitations subsection now states all of this.*
Entries: B10, B11.

---

## Checklist before this becomes a real response letter

- [ ] Paste the reviewers' verbatim comments and re-key Part 1 against them.
- [ ] Replace every `[P]` value from the final head-priced grid.
- [ ] Insert the completed validation results (both lenses, one partial-adoption
      point) into Theme G.
- [ ] Supply the HAGRID full form (reviewer point 3).
- [ ] Confirm the page budget after the mandatory trim pass.
- [ ] Add per-change line/section references into the revised manuscript once
      the final layout is fixed.
