# Response to Reviewers — skeleton (revision part C, 2026-08-28)

**Status: skeleton, numbers final.** Prose is drafted; every number below now
comes from grid v6 (`results/revision_2026_08_v6/`) or from that grid's VROOM
validation (items 0, 1, 2), matching `tbc_preprint_main.tex` and sections F and
G of `docs/CHANGES_rev1.md`. The one block still open is validation item 3, the
partial-adoption point, which the manuscript carries as a single `% PART C3`
marker.

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
| 10 | Eq. (3) looks like a Nash equilibrium (Eq. (3) as submitted is Eq. (5) of the revision) | Coordinate descent on a common depot objective; restart spread < 1e-12 relative; no global-optimality claim | Yes — the equation's coupling term is removed and replaced by an explicit statement of the residual coupling inside pooling groups |
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
inside pooling groups of sub-threshold cells, and Eq. (5) of the revision
(Eq. (3) as submitted) retains the depot objective precisely in order to keep that residual coupling priced. We do not
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
corrected model the figures are 9.6 % of cells and 0.40 % of routing cost --
reduced by almost an order of magnitude, but not to zero.
The full-adoption headline is not affected — a symmetric re-computation of the
submitted grid bounds the whole effect at that point to 0.57 percentage points
(22.79 % to 22.22 %). We have also withdrawn the per-area saving figures of the
submitted version rather than reprinting them: they rest on the same grid, and
the spatial breakdown is re-derived from the revised per-cell costs.*
Entries: B2, B3, B6, A3, B12.

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
this revision. Under the routing lens the cost-optimal plan saves 22.6 %;
under the operator lens the very same plan is 8.4 % worse than daily
delivery, because its two-day patterns raise the summed depot peak by
34.5 %. Re-optimizing the second stage in the operator's currency turns this
into 24.3 % of operator saving at a 16.9 % lower summed depot peak. On cost
alone each lens, read on its own plan, still prefers the cost-optimal extreme,
by 3.9 pp and 1.7 pp; we recommend P = 0.25 EUR per parcel-day because that is
where the service side is priced in, and because a flat 0.50 EUR discount per
delayed parcel makes it the best point on the grid in the operator lens, ahead
of P = 0.5 by 0.8 pp, with the routing lens rating the two within 0.01 pp. The
carrier taxonomy the reviewer asked us to justify turns out to be
lens-dependent: it holds as stated in the routing lens, while under the
operator lens Amazon and Hermes move up one class and GLS leaves the
0.25-0.75 band altogether.*
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
0.75-2.25 EUR per delayed parcel in the operator lens and 0.56-1.21 EUR in the
routing lens. The full scenario is in the supplementary material.*
Entry: B9.

### Theme G — What is not yet validated

*Draft text.* *We are explicit about the limits. We have now re-routed 7,610
instances with the solver — the daily baseline and six full-adoption operating
points across both plans — and report predicted and realised savings side by
side. The partial-adoption regime, where the express residual makes pooled
tours far more frequent, is still being re-routed; its figures remain
predictions. That validation is reported in Section 3.3. The operator-lens magnitude is shaped by
one carrier's depot structure, pooling is within-depot only, and the pooled-tour
head prices 53.9 % of the pooled tours that occur and 27.2 % of the pooled
cost -- three different coverage rates that we now report separately rather
than under one number -- with a conservative
fallback elsewhere. A dedicated limitations subsection now states all of this.*
Entries: B10, B11.

---

## Checklist before this becomes a real response letter

- [ ] Paste the reviewers' verbatim comments and re-key Part 1 against them.
- [x] Replace every `[P]` value from the final head-priced grid (v6, part B).
- [ ] Replace the VROOM-validation figures once the v6 re-validation lands
      (the `% PART C` markers in `tbc_preprint_main.tex`).
- [ ] Insert the completed validation results (both lenses, one partial-adoption
      point) into Theme G.
- [ ] Supply the HAGRID full form (reviewer point 3).
- [ ] Confirm the page budget after the mandatory trim pass.
- [ ] Add per-change line/section references into the revised manuscript once
      the final layout is fixed.

---

## Part B addendum (2026-08-28) --- what changed since the part A skeleton

Grid v6 (v5 plus the certified bundle head) is the production grid; every number
above is now taken from it. Four statements changed in substance, not only in
value, and the response letter must carry them:

1. **The flat-discount optimum is a tie in the routing lens.** Part A claimed
   P = 0.5 led P = 0.25 by 0.2 pp there. On v6 the two are 0.01 pp apart, which
   is below the grid's own resolution. The operator lens prefers P = 0.25 by
   0.8 pp.
2. **The operator-lens carrier shifts are not the ones part A named.** Amazon
   and Hermes move up one class; FedEx does not move; GLS moves to P* = 1.0 and
   therefore out of the class band, so it is unclassified in that lens. The
   three-class taxonomy is a routing-lens statement.
3. **The (10, 0.1) corner is small, not flat.** 0.40 % of routing cost and
   9.6 % non-daily cells against the submitted 3.6 % and 42 %. The mechanism
   retraction stands; the wording "reduced to" replaces "gone".
4. **Bundle-head coverage is three numbers, not one** (44.0 % pre-run
   certification, 53.9 % of realised pooled tours, 27.2 % of pooled cost).

Also corrected against part A: the claim that no pooled tour exists at theta = 1
is false and has been removed from the validation subsection; the spatial
breakdown that part A left pending is now derived on v6 and reproduces the
submitted ordering, so the equity argument is restored unconditionally.

---

## Part C addendum (2026-08-28) --- the validation reverses one submitted claim

The v6 VROOM re-validation (7,610 instances: the daily baseline plus, at full
adoption, the operator-polished plan at P = 0/0.25/0.5/0.75 and the
routing-optimal plan at P = 0/0.25; 1 PARTIAL) changes the direction of the
paper's accuracy statement, and the response letter must say so before a
reviewer notices:

1. **The surrogate is not conservative; it over-prices.** Mean per-tour bias is
   positive in every group we report: +5.4 % on the daily baseline (+4.9 %
   single-cell, +14.5 % pooled) against +2.4 % at the consolidated points.
   Overall 3.51 % MAPE at +3.05 % bias over the 7,609 clean instances, the
   basis used throughout (the one PARTIAL instance is dropped from the error
   statistics and from the totals alike).
2. **Predicted savings are therefore an upper bound** --- by 1.3 to 2.5 pp
   for the operator-polished plan and 2.1 to 3.7 pp for the routing-optimal
   one. The
   over-priced baseline is the denominator. Realised against predicted:
   operator plan 17.5 / 14.5 / 10.6 / 7.7 % routing saving (predicted 20.0 /
   16.7 / 12.3 / 9.0) and 22.1 / 20.7 / 16.3 / 12.8 % operator saving
   (predicted 24.3 / 22.6 / 17.8 / 14.1); routing-optimal plan 20.6 and
   16.4 % routing saving (predicted 22.6 and 18.7).
3. **The submitted claim of conservatism is withdrawn.** The submitted version
   reported 24.3 % realised against 22.8 % predicted at P = 0. That comparison predates
   the universal tour rule, and this is the first validation in which the daily
   baseline is itself solver-routed — the baseline's own over-pricing is what
   reverses the sign.
4. **No conclusion changes sign or order, and the central one strengthens.**
   The operator-lens penalty of the routing-optimal plan is worse in reality:
   −12.1 % against −8.4 % predicted at (0, 1). The fleet counting rule is
   near-exact: summed depot peaks never differ by more than five vehicles in a
   thousand at the consolidated points, and by ten in the baseline's 1,239.

Still open: validation item 3, the partial-adoption point (P, theta) =
(0.25, 0.5) on the operator plan.
